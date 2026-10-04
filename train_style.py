#===================================================================================================
# Monster Genie train_style.py Python module
# Training with style cross-attention conditioning + dual conditioning (buttons + harmony)
#
# Pickle format: flat [dtime, dur, pitch, vel, chan] with:
#   - Harmony movements as [0, 0, movement_type, 0, 3] (channel 3)
#   - Piece boundaries as [126, 126, 0, 0, 0]
#
# For each sample we draw TWO non-overlapping windows from the same piece:
#   - target window → pitch + harmony regime/strength (decoder input/target)
#   - style window  → pitch-only sequence (style encoder cross-attention context)
# 
# Copyright 2025 Alex Barrachina
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.'''
#===================================================================================================


import os
from training_trace import TrainingTrace, enable_fault_handler
enable_fault_handler()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import tqdm
import glob
import re
import hashlib
import random as python_random
from pathlib import Path

os.environ['USE_FLASH_ATTENTION'] = '1'

from random import Random, randint, random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from midiUtils import Any_Pickle_File_Reader
from model_loader import load_model
from models import get_model_hparams
from params import *
from x_transformer import *
from timing import TIME_UNIT, HISTORY_SIZE, timing_features, augment_timing

NSTEPS_INIT = 4508
RESUME = os.environ.get('RESUME', '1') != '0'

JOKER_GUIDED_FRACTION = 0.25
JOKER_MIXED_FRACTION = 0.50
JOKER_MIN_SPAN = 32
JOKER_MAX_SPAN = 128

#==========================================================================

def find_latest_checkpoint(checkpoint_dir: str = './save_models',
                           model_name: str = None) -> tuple[str, int, int]:
    """
    Find the latest checkpoint in the save_models directory.
    
    Returns:
        tuple: (checkpoint_path, epoch, steps) or (None, 0, 0) if no checkpoint found
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return None, 0, 0
    
    # Pattern to match checkpoint files: MODEL_NAME_epoch_eps_steps_steps_loss_loss_acc_acc.pth
    file_pattern = f"{model_name}_*.pth" if model_name else "*.pth"
    pattern = os.path.join(checkpoint_dir, file_pattern)
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return None, 0, 0
    
    # Extract epoch and steps from filename
    latest_checkpoint = None
    max_steps = -1
    max_epoch = -1
    
    for checkpoint_file in checkpoint_files:
        filename = os.path.basename(checkpoint_file)
        # Parse filename: MODEL_NAME_epoch_eps_steps_steps_loss_loss_acc_acc.pth
        match = re.search(r'(\d+)_eps_(\d+)_steps', filename)
        if match:
            epoch = int(match.group(1))
            steps = int(match.group(2))
            
            # Choose checkpoint with highest steps (most recent)
            if steps > max_steps or (steps == max_steps and epoch > max_epoch):
                max_steps = steps
                max_epoch = epoch
                latest_checkpoint = checkpoint_file
    
    if latest_checkpoint:
        print(f"Found latest checkpoint: {latest_checkpoint}")
        print(f"Resuming from epoch {max_epoch}, step {max_steps}")
        return latest_checkpoint, max_epoch, max_steps
    else:
        print("No valid checkpoint files found")
        return None, 0, 0

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   checkpoint_path: str, device: torch.device, scaler=None) -> tuple[int, int]:
    """
    Load model and optimizer state from checkpoint.
    
    Returns:
        tuple: (start_epoch, start_steps)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load the state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if model.cfg.get('timing_enabled', False):
        for key in ('timing_enabled', 'timing_warmup_steps', 'timing_total_steps',
                    'timing_lr', 'timing_decoder_lr'):
            if checkpoint['cfg'].get(key) != model.cfg.get(key):
                raise ValueError(f'Timing resume configuration differs for {key}')
        expected_stage = ('warmup' if checkpoint['steps'] < model.cfg['timing_warmup_steps']
                          else 'adaptation')
        if checkpoint['stage'] != expected_stage:
            raise ValueError('Timing checkpoint stage does not match its step count')
    
    # If checkpoint is just the model state dict (as saved in original code)
    if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
        # This is just the model state dict
        model.load_state_dict(checkpoint)
        print("Loaded model state dict from checkpoint")
        
        # Extract epoch and steps from filename
        filename = os.path.basename(checkpoint_path)
        match = re.search(r'(\d+)_eps_(\d+)_steps', filename)
        if match:
            start_epoch = int(match.group(1))
            start_steps = int(match.group(2))
        else:
            start_epoch = 0
            start_steps = 0
            
    else:
        # This is a full checkpoint with model, optimizer, etc.
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        start_steps = checkpoint.get('steps', 0)
        print("Loaded full checkpoint with model and optimizer state")
        if model.cfg.get('timing_enabled', False):
            model.timing_base_checkpoint = checkpoint['base_checkpoint']
            python_random.setstate(checkpoint['python_rng'])
            torch.set_rng_state(checkpoint['torch_rng'].cpu())
            if torch.cuda.is_available() and checkpoint['cuda_rng'] is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng']])
            if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            set_timing_stage(model, optimizer, start_steps)
    
    return start_epoch, start_steps


def warm_start_timing(model, checkpoint_path):
    """Allow only the new timing weights to be absent in the base checkpoint."""
    state = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected = {name for name in model.state_dict() if name.startswith('decoder.timing_mlp.')}
    if set(missing) != expected or unexpected:
        raise ValueError(f'Unsafe timing warm start: missing={missing}, unexpected={unexpected}')
    projection = model.decoder.timing_mlp[-1]
    if projection.weight.count_nonzero() or projection.bias.count_nonzero():
        raise ValueError('Timing projection must be zero at warm start')
    digest = hashlib.sha256()
    with open(checkpoint_path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    model.timing_base_checkpoint = {
        'path': str(Path(checkpoint_path).resolve()), 'sha256': digest.hexdigest(),
    }


def make_timing_optimizer(model):
    """Select existing modules explicitly; never unfreeze style cross-attention."""
    cfg = model.cfg
    model.requires_grad_(False)
    layers = model.decoder.attn_layers
    block_starts = [i for i, kind in enumerate(layers.layer_types) if kind == 'a']
    start = block_starts[max(0, len(block_starts) - 2)]
    modules = [layer for kind, layer in zip(layers.layer_types[start:], layers.layers[start:])
               if kind in ('a', 'f')]
    modules.extend([layers.final_norm, model.decoder.to_logits])
    optimizer = torch.optim.Adam([
        {'params': model.decoder.timing_mlp.parameters(), 'lr': cfg['timing_lr']},
        {'params': [p for module in modules for p in module.parameters()],
         'lr': cfg['timing_decoder_lr']},
    ])
    set_timing_stage(model, optimizer, 0)
    return optimizer


def set_timing_stage(model, optimizer, step):
    adapting = step >= model.cfg['timing_warmup_steps']
    if adapting and getattr(model, '_timing_stage', None) == 'warmup':
        projection = model.decoder.timing_mlp[-1]
        if not projection.weight.count_nonzero() or not torch.isfinite(projection.weight).all():
            raise RuntimeError('Timing warm-up produced an inactive or nonfinite projection')
    for parameter in optimizer.param_groups[0]['params']:
        parameter.requires_grad_(True)
    for parameter in optimizer.param_groups[1]['params']:
        parameter.requires_grad_(adapting)
    model._timing_stage = 'adaptation' if adapting else 'warmup'
    # Frozen encoders provide stable button and style conditioning, without dropout.
    model.encoder.eval()
    model.style_encoder.eval()


def save_timing_checkpoint(model, optimizer, epoch, step, scaler=None):
    """Keep one resumable checkpoint; publish it only after the write completes."""
    path = Path(model.cfg['ckpt_file_name'])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch, 'steps': step,
        'stage': 'warmup' if step < model.cfg['timing_warmup_steps'] else 'adaptation',
        'cfg': dict(model.cfg), 'base_checkpoint': model.timing_base_checkpoint,
        'python_rng': python_random.getstate(), 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, temporary)
    os.replace(temporary, path)


#==========================================================================

def make_joker_training_mask(batch_size, seq_len, device):
    """Mix guided, one long joker passage, and entirely joker sequences.

    The proportions are per sequence, in expectation across training batches.
    The mask aligns with target pitches (pitch[:, 1:]), not the primer pitch.
    """
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    if batch_size == 0 or seq_len == 0:
        return mask

    modes = torch.rand(batch_size, device=device)
    mixed = (modes >= JOKER_GUIDED_FRACTION) & (
        modes < JOKER_GUIDED_FRACTION + JOKER_MIXED_FRACTION
    )
    mask[modes >= JOKER_GUIDED_FRACTION + JOKER_MIXED_FRACTION] = True

    min_span = min(seq_len, JOKER_MIN_SPAN)
    max_span = min(seq_len, JOKER_MAX_SPAN)
    lengths = torch.randint(min_span, max_span + 1, (batch_size,), device=device)
    starts = (torch.rand(batch_size, device=device) * (seq_len - lengths + 1)).long()
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    mask |= mixed.unsqueeze(1) & (positions >= starts.unsqueeze(1)) & (
        positions < (starts + lengths).unsqueeze(1)
    )
    return mask


@torch.no_grad()
def encoder_outside_percentage(model, pitch):
    """Percentage of non-padding target encoder values outside [-1, 1]."""
    raw_values = model.encoder({'pitch': pitch[:, 1:]})
    valid = pitch[:, 1:] != model.ignore_index
    raw_values = raw_values[valid]
    if raw_values.numel() == 0:
        return None
    return (raw_values.abs() > 1).float().mean().mul(100).item()


#==========================================================================

class StyleMusicSamplerDataset(Dataset):
    """
    Dataset for style-conditioned model.

    For each sample, selects a single piece (bounded by [126,126,0,0,0] events),
    then draws two non-overlapping windows:
      - target window: pitch sequence for encoder/decoder (standard autoregressive)
      - style window:  pitch-only sequence for StyleEncoder cross-attention

    Same transposition is applied to both windows to preserve key consistency.
    """
    def __init__(self, data, seq_len, style_seq_len, is_eval=False, cfg=None):
        super().__init__()

        self.data = data
        self.seq_len = seq_len
        self.style_seq_len = style_seq_len
        self.tokens_per_note = 5  # dtime, dur, pitch, vel, chan
        self.cfg = cfg if cfg is not None else {}
        self.is_eval = is_eval
        self.joker_param_model = self.cfg.get('model_type') == 'AE_style_jokerParam'
        self.timing_enabled = self.cfg.get('timing_enabled', False)
        self.deterministic_eval = self.joker_param_model and self.is_eval

        # Pre-compute piece boundaries from [126,126,0,0,0] events
        self.piece_ranges = self._find_piece_boundaries()

        # Oversample factor: ~2x to account for harmony events mixed in with notes
        self.target_oversample = self.seq_len * 2
        self.style_oversample = self.style_seq_len * 2
        min_events = self.target_oversample + self.style_oversample

        # Keep only pieces with enough events for both windows
        self.valid_pieces = [
            (start, end) for start, end in self.piece_ranges
            if (end - start) >= min_events
        ]

        if len(self.valid_pieces) == 0:
            raise ValueError(
                f"No pieces have enough events ({min_events}) for "
                f"seq_len={seq_len} + style_seq_len={style_seq_len}. "
                f"Total pieces found: {len(self.piece_ranges)}"
            )

        print(f"  Pieces total: {len(self.piece_ranges)}, "
              f"valid (long enough): {len(self.valid_pieces)}")

        # Pre-compute events view (no copy, just reshape)
        num_events = len(self.data) // self.tokens_per_note
        self.all_events = self.data[:num_events * self.tokens_per_note].view(
            num_events, self.tokens_per_note
        ).long()

        # Estimate dataset size: total note-events in valid pieces / seq_len
        self._total_samples = sum(
            (end - start) // (self.seq_len + 1)
            for start, end in self.valid_pieces
        )

    def _find_piece_boundaries(self):
        """
        Detect [126,126,0,0,0] boundary events and return per-piece (start, end)
        ranges in event-index space (each event = 5 tokens).
        """
        num_events = len(self.data) // self.tokens_per_note
        events = self.data[:num_events * self.tokens_per_note].view(
            num_events, self.tokens_per_note
        ).long()

        is_boundary = (
            (events[:, 0] == 126) &
            (events[:, 1] == 126) &
            (events[:, 2] == 0) &
            (events[:, 3] == 0) &
            (events[:, 4] == 0)
        )
        boundary_indices = torch.where(is_boundary)[0].tolist()

        if len(boundary_indices) == 0:
            return [(0, num_events)]

        ranges = []
        for i in range(len(boundary_indices)):
            start = boundary_indices[i] + 1  # skip the boundary event itself
            end = boundary_indices[i + 1] if i + 1 < len(boundary_indices) else num_events
            if end > start:
                ranges.append((start, end))

        return ranges

    def __len__(self):
        return max(1, self._total_samples)

    def __getitem__(self, index):
        # Joker validation must return the same excerpt for a given index.
        rng = Random(index) if self.deterministic_eval else None
        draw_int = rng.randint if rng is not None else randint
        draw_float = rng.random if rng is not None else random

        piece_idx = draw_int(0, len(self.valid_pieces) - 1)
        start_event, end_event = self.valid_pieces[piece_idx]
        piece_events = self.all_events[start_event:end_event]  # [L, 5]
        piece_len = piece_events.shape[0]

        total_needed = self.target_oversample + self.style_oversample
        max_start = piece_len - total_needed
        window_start = draw_int(0, max(0, max_start))

        # Randomly assign order so model doesn't learn positional bias
        if draw_float() < 0.5:
            target_start = window_start
            target_events = piece_events[window_start : window_start + self.target_oversample]
            style_events = piece_events[window_start + self.target_oversample : window_start + total_needed]
        else:
            target_start = window_start + self.style_oversample
            style_events = piece_events[window_start : window_start + self.style_oversample]
            target_events = piece_events[window_start + self.style_oversample : window_start + total_needed]

        # Same transposition for both windows (preserves key relationship)
        transposition = 0 if self.deterministic_eval else draw_int(
            -self.cfg.get('data_augment_transpose_max', 6),
            self.cfg.get('data_augment_transpose_max', 6)
        )

        target_data = self._process_target_window(target_events, transposition)
        if self.timing_enabled:
            target_data.update(self._target_timing(piece_events, target_start))
        style_data = self._process_style_window(style_events, transposition)

        # Merge into a single dict
        target_data.update(style_data)
        return target_data

    def _target_timing(self, piece_events, start):
        """Include 17 preceding notes so all 16 preceding intervals are known."""
        end = start + self.target_oversample
        positions = torch.where(piece_events[:end, 4] != HARMONY_CHANNEL)[0]
        offset = int((positions < start).sum())
        begin = max(0, offset - HISTORY_SIZE - 1)
        selected = positions[begin:offset + self.seq_len + 1]
        prefix_len = offset - begin
        features = torch.zeros(self.seq_len + 1, 4)
        valid = torch.zeros(self.seq_len + 1, dtype=torch.bool)
        if len(selected):
            # Sum across harmony markers before selecting note onsets.
            origin = int(selected[0])
            onsets = piece_events[origin:int(selected[-1]) + 1, 0].double().cumsum(0)
            onsets = onsets[selected - origin] * TIME_UNIT
            intervals = [None] + torch.diff(onsets).tolist()
            if not self.is_eval:
                intervals = augment_timing(intervals, prefix_len)
            encoded, known = timing_features(intervals)
            count = len(selected) - prefix_len
            features[:count] = encoded[prefix_len:]
            valid[:count] = known[prefix_len:]
        if not self.is_eval and random() < 0.25:
            valid.zero_()
        return {'timing_features': features, 'timing_mask': valid}

    # ------------------------------------------------------------------ #
    #  Target window: pitch + harmony regime/strength                      #
    #  (replicates train_harm.py MusicSamplerDataset logic)                #
    # ------------------------------------------------------------------ #

    def _process_target_window(self, events, transposition):
        """
        Extract pitch + harmony regime/strength from an event window.
        Returns dict with 'pitch' [T+1], 'harm_regime' [T+1], 'harm_strength' [T+1].
        """
        target_len = self.seq_len + 1

        # --- Vectorized harmony regime + strength (same as train_harm.py) ---
        is_harmony = (events[:, 4] == HARMONY_CHANNEL)
        is_note = ~is_harmony
        note_indices = torch.where(is_note)[0]

        if len(note_indices) == 0:
            return self._create_dummy_target()

        harm_cumsum = torch.cumsum(is_harmony.long(), dim=0)
        harm_event_indices = torch.where(is_harmony)[0]
        num_harm = harm_event_indices.shape[0]

        move_lookup = torch.zeros(num_harm + 1, dtype=torch.long)
        if num_harm > 0:
            move_lookup[1:] = torch.clamp(events[harm_event_indices, 2] - 60, min=0, max=7)

        harm_group_notes = harm_cumsum[note_indices]
        harm_regime_notes = move_lookup[harm_group_notes]

        group_changes = torch.cat([
            torch.tensor([True]),
            harm_group_notes[1:] != harm_group_notes[:-1]
        ])
        group_start_indices = torch.where(group_changes)[0]
        note_arange = torch.arange(len(harm_group_notes))
        group_of_note = torch.searchsorted(group_start_indices, note_arange, side='right') - 1
        position_in_group = note_arange - group_start_indices[group_of_note]

        span_lengths = torch.diff(
            group_start_indices, append=torch.tensor([len(harm_group_notes)])
        )
        span_length_per_note = span_lengths[group_of_note]

        harm_strength_notes = 1.0 - position_in_group.float() / span_length_per_note.float()
        unguided = (harm_group_notes == 0)
        harm_strength_notes = harm_strength_notes.masked_fill(unguided, 0.0)

        # --- Filter to note events and truncate/pad to target_len ---
        note_events = events[note_indices]
        note_count = min(len(note_events), target_len)
        if len(note_events) < target_len:
            if self.timing_enabled:
                pad = target_len - len(note_events)
                note_events = torch.cat([note_events, note_events.new_zeros(pad, 5)])
                harm_regime_notes = F.pad(harm_regime_notes, (0, pad))
                harm_strength_notes = F.pad(harm_strength_notes, (0, pad))
            else:
                num_repeats = (target_len // len(note_events)) + 1
                note_events = note_events.repeat(num_repeats, 1)[:target_len]
                harm_regime_notes = harm_regime_notes.repeat(num_repeats)[:target_len]
                harm_strength_notes = harm_strength_notes.repeat(num_repeats)[:target_len]
        else:
            note_events = note_events[:target_len]
            harm_regime_notes = harm_regime_notes[:target_len]
            harm_strength_notes = harm_strength_notes[:target_len]

        pitches = note_events[:, 2]
        dtimes = note_events[:, 0]
        durs = note_events[:, 1]
        harm_regime = harm_regime_notes
        harm_strength = harm_strength_notes

        # Validation has no random augmentation for the joker model.
        if not self.deterministic_eval and not self.timing_enabled:
            stretch_factor = random() * self.cfg.get('data_augment_time_stretch_max', 0.05) * 2
            stretch_factor += 1 - self.cfg.get('data_augment_time_stretch_max', 0.05)
            dtimes = (dtimes.float() * stretch_factor).long()
            dtimes = torch.clamp(dtimes, min=0, max=RANGE_DTIME_SHIFT)

            stretch_factor = random() * self.cfg.get('data_augment_time_stretch_max', 0.05) * 2
            stretch_factor += 1 - self.cfg.get('data_augment_time_stretch_max', 0.05)
            durs = (durs.float() * stretch_factor).long()
            durs = torch.clamp(durs, min=0, max=RANGE_DUR_SHIFT)

        # The joker decoder sees pitches only. Keep source chord-note order;
        # preserve the original micro-alterations for other style models.
        if not self.joker_param_model:
            abs_times = torch.cumsum(dtimes, dim=0)
            chord_groups = []
            current_chord = [0]
            for i in range(1, len(abs_times)):
                if abs_times[i] - abs_times[i-1] <= self.cfg.get('data_augment_chord_threshold', 2):
                    current_chord.append(i)
                else:
                    if len(current_chord) > 1:
                        chord_groups.append(current_chord)
                    current_chord = [i]
            if len(current_chord) > 1:
                chord_groups.append(current_chord)
            for chord in chord_groups:
                shifts = torch.randint(-1, 2, (len(chord),))
                abs_times[chord] = abs_times[chord] + shifts

            sorted_indices = torch.argsort(abs_times)
            abs_times = abs_times[sorted_indices]
            pitches = pitches[sorted_indices]
            harm_regime = harm_regime[sorted_indices]
            harm_strength = harm_strength[sorted_indices]

            dtimes = torch.cat([abs_times[0:1], abs_times[1:] - abs_times[:-1]])
            dtimes = torch.clamp(dtimes, min=0, max=RANGE_DTIME_SHIFT)

        # Transposition (shared with style window)
        pitches = torch.clamp(pitches + transposition, min=0, max=VOCAB_SIZE_PITCH - 1)
        if self.timing_enabled:
            pitches[note_count:] = PAD_IDX

        return {
            'pitch': pitches,
            'harm_regime': harm_regime,
            'harm_strength': harm_strength,
        }

    # ------------------------------------------------------------------ #
    #  Style window: pitch-only (for StyleEncoder)                         #
    # ------------------------------------------------------------------ #

    def _process_style_window(self, events, transposition):
        """
        Extract pitch-only sequence from a style reference window.
        Returns dict with 'style_pitch' [S] and 'style_mask' [S].
        """
        # Filter to note events only (drop harmony events)
        is_note = (events[:, 4] != HARMONY_CHANNEL)
        note_events = events[is_note]

        pitches = note_events[:, 2].clone()

        # Same transposition as target
        pitches = torch.clamp(pitches + transposition, min=0, max=VOCAB_SIZE_PITCH - 1)

        # Truncate or pad to style_seq_len
        if len(pitches) >= self.style_seq_len:
            pitches = pitches[:self.style_seq_len]
            style_mask = torch.ones(self.style_seq_len, dtype=torch.bool)
        else:
            pad_len = self.style_seq_len - len(pitches)
            style_mask = torch.cat([
                torch.ones(len(pitches), dtype=torch.bool),
                torch.zeros(pad_len, dtype=torch.bool)
            ])
            pitches = torch.cat([pitches, torch.zeros(pad_len, dtype=torch.long)])

        return {
            'style_pitch': pitches,
            'style_mask': style_mask,
        }

    # ------------------------------------------------------------------ #
    #  Dummy samples for degenerate cases                                  #
    # ------------------------------------------------------------------ #

    def _create_dummy_target(self):
        target_len = self.seq_len + 1
        print("********** Creating dummy target sample **********")
        return {
            'pitch': torch.full((target_len,), PAD_IDX if self.timing_enabled else 60, dtype=torch.long),
            'harm_regime': torch.zeros(target_len, dtype=torch.long),
            'harm_strength': torch.zeros(target_len, dtype=torch.float),
        }


#==========================================================================

def main():
    # Set up CUDA settings
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)

    #==========================================================================

    ''' DEVICE '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    trace = TrainingTrace(torch, device)
    trace.mark('startup', resources=True)
    if trace.enabled:
        print(f'Trace environment: torch={torch.__version__} CUDA={torch.version.cuda} '
              f'cuDNN={torch.backends.cudnn.version()} '
              f'GPU={torch.cuda.get_device_name() if device_type == "cuda" else "CPU"}',
              flush=True)

    #==========================================================================

    ''' MODEL & HYPERPARAMETERS '''
    project_name = 'monsterGenie_style'
    # Keep this trainer reusable while making the new joker model the default.
    # Override with MODEL_NAME=... when training another style configuration.
    model_name = os.environ.get('MODEL_NAME', 'AE_style_jokerParam_tester_v2')
    cfg = get_model_hparams(model_name)
    is_timing_model = cfg.get('timing_enabled', False)
    cfg['batch_size'] = int(os.environ.get('BATCH_SIZE', cfg['batch_size']))
    cfg['num_workers'] = int(os.environ.get('NUM_WORKERS', cfg['num_workers']))
    pin_memory = device_type == 'cuda' and os.environ.get('PIN_MEMORY', '1') != '0'
    persistent_workers = cfg['num_workers'] > 0 and os.environ.get('PERSISTENT_WORKERS', '1') != '0'
    model = load_model(model_name=model_name, cfg=cfg, set_only=True)
    model.to(device)

    #==========================================================================

    ''' WANDB '''
    if cfg['use_logs']:
        import wandb
        wandb.login()
        wandb.init(project=project_name, name=model_name, config=cfg)

    #==========================================================================

    ''' DATA '''

    train_data = Any_Pickle_File_Reader(cfg['dataset_train_path'])
    data_train = torch.Tensor(train_data)

    style_seq_len = cfg.get('style_seq_len', 256)

    print("Building training dataset...")
    train_dataset = StyleMusicSamplerDataset(
        data_train, cfg['seq_len'], style_seq_len, cfg=cfg
    )
    print(f"BATCH_SIZE: {cfg['batch_size']}")
    print(f"NUM_WORKERS: {cfg['num_workers']}")
    print(f"Dataset size: {len(train_dataset)}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        shuffle=True,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    print(f"Number of batches: {len(train_loader)}")

    if not is_timing_model:
        eval_data = Any_Pickle_File_Reader(cfg['dataset_val_path'])
        data_eval = torch.Tensor(eval_data)
        print("Building validation dataset...")
        val_dataset = StyleMusicSamplerDataset(
            data_eval, cfg['seq_len'], style_seq_len, is_eval=True, cfg=cfg
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg['batch_size'],
            num_workers=0,
            shuffle=False,
            pin_memory=pin_memory,
        )
        val_iter = iter(val_loader)

    #==========================================================================

    ''' PRECISION/OPTIMIZER/SCALER '''

    dtype = torch.bfloat16

    is_joker_param_model = cfg['model_type'] == 'AE_style_jokerParam'
    if is_timing_model:
        optim = make_timing_optimizer(model)
    elif is_joker_param_model:
        joker_params = []
        inherited_params = []
        for param_name, parameter in model.named_parameters():
            if param_name == 'decoder.joker_mode':
                joker_params.append(parameter)
            else:
                inherited_params.append(parameter)
        if len(joker_params) != 1:
            raise RuntimeError(
                "AE_style_jokerParam must expose exactly decoder.joker_mode; "
                f"found {len(joker_params)} matching parameters"
            )
        optim = torch.optim.Adam([
            {
                'params': inherited_params,
                'lr': cfg['learning_rate'] * cfg.get('base_lr_multiplier', 0.1),
            },
            {
                'params': joker_params,
                'lr': cfg['learning_rate'] * cfg.get('joker_lr_multiplier', 1.0),
            },
        ])
    else:
        optim = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    scaler = torch.amp.GradScaler(device_type)

    ''' LOAD CHECKPOINT '''
    nsteps = 0 
    resumed_from_checkpoint = False

    def warm_start_from_base():
        init_ckpt = cfg.get('init_from_ckpt', '')
        if not init_ckpt:
            return
        if is_timing_model:
            warm_start_timing(model, init_ckpt)
            print(f'Warm-started timing model from {init_ckpt}')
            return
        state = torch.load(init_ckpt, map_location=device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        missing, unexpected = model.load_state_dict(state, strict=False)

        if is_joker_param_model:
            expected_missing = {'decoder.joker_mode'}
            if set(missing) != expected_missing or unexpected:
                raise RuntimeError(
                    "Unsafe AE_style_v2 warm start: expected only "
                    f"{sorted(expected_missing)} to be missing and no unexpected keys; "
                    f"got missing={missing}, unexpected={unexpected}"
                )
            if model.decoder.joker_mode.detach().count_nonzero().item() != 0:
                raise RuntimeError("decoder.joker_mode must remain exactly zero after warm start")
        print(f"Warm-started from {init_ckpt}: missing={missing}, unexpected={unexpected}")

    if RESUME:
        if is_timing_model:
            checkpoint_path = cfg['ckpt_file_name'] if os.path.isfile(cfg['ckpt_file_name']) else None
        else:
            checkpoint_path, start_epoch, start_steps = find_latest_checkpoint(
                model_name=model_name
            )
    
        if checkpoint_path:
            start_epoch, start_steps = load_checkpoint(model, optim, checkpoint_path, device, scaler)
            print(f"Resuming training from epoch {start_epoch}, step {start_steps}")
            nsteps = start_steps
            resumed_from_checkpoint = True
        else:
            start_epoch = 0
            start_steps = 0
            print("No model-specific checkpoint found; applying configured warm start")
            warm_start_from_base()
    else:
        start_epoch = 0
        start_steps = 0
        warm_start_from_base()


    ''' TRAINING '''

    val_loss_temp = 0.0
    if is_timing_model and nsteps >= cfg['timing_total_steps']:
        print('Timing training already reached its configured step limit')
        return
    for ep in range(start_epoch if is_timing_model else 0, cfg['epochs']):
        print('Epoch #', ep)
        trace.mark('epoch_start_fetch_next', ep, resources=True)

        joker_only_warmup = (
            is_joker_param_model
            and not resumed_from_checkpoint
            and ep < int(cfg.get('joker_param_warmup_epochs', 0))
        )
        if is_joker_param_model and not is_timing_model:
            for param_name, parameter in model.named_parameters():
                parameter.requires_grad = (
                    param_name == 'decoder.joker_mode' or not joker_only_warmup
                )
            if joker_only_warmup:
                print("Joker adaptation warmup: training decoder.joker_mode only")
            elif ep == int(cfg.get('joker_param_warmup_epochs', 0)):
                print("Joint fine-tuning: inherited AE_style_v2 parameters unfrozen")

        model.train()
        with tqdm.tqdm(total=len(train_dataset), unit='samp') as bar_train:
            for i, batch in enumerate(train_loader):
                trace.mark('batch_received_transfer', ep, i)
                if is_timing_model:
                    set_timing_stage(model, optim, nsteps)
                optim.zero_grad()

                x = {
                    'pitch': batch['pitch'].to(device, non_blocking=True),
                    'harm_regime': batch['harm_regime'].to(device, non_blocking=True),
                    'harm_strength': batch['harm_strength'].to(device, non_blocking=True),
                    'style_pitch': batch['style_pitch'].to(device, non_blocking=True),
                    'style_mask': batch['style_mask'].to(device, non_blocking=True),
                }
                if is_timing_model:
                    for key in ('timing_features', 'timing_mask'):
                        x[key] = batch[key].to(device, non_blocking=True)
                if is_joker_param_model:
                    x['joker_mask'] = make_joker_training_mask(
                        x['pitch'].shape[0], x['pitch'].shape[1] - 1, device
                    )

                trace.mark('forward', ep, i)
                with torch.amp.autocast(device_type=device_type, dtype=dtype):
                    loss, acc = model(x)
                objective = loss['loss_recons'] if is_timing_model else loss['loss_total']
                trace.mark('backward', ep, i)
                scaler.scale(objective).backward()
                trace.mark('logging', ep, i)

                if (i % cfg['print_stats_every'] == 0) or TESTING:
                    if cfg['use_logs']:
                        wandb.log({"loss_total": loss['loss_total'].item()}, step=nsteps)
                        wandb.log({"train_acc": acc.item()}, step=nsteps)
                        if cfg.get('loss_recons', 0) > 0 and 'loss_recons' in loss:
                            wandb.log({"loss_recons": cfg['loss_recons'] * loss['loss_recons'].item()}, step=nsteps)
                        if cfg.get('loss_margin', 0) > 0 and 'loss_margin' in loss:
                            wandb.log({"loss_margin": cfg['loss_margin'] * loss['loss_margin'].item()}, step=nsteps)
                        if cfg.get('loss_deviate', 0) > 0 and 'loss_deviate' in loss:
                            wandb.log({"loss_deviate": cfg['loss_deviate'] * loss['loss_deviate'].item()}, step=nsteps)
                        if cfg.get('loss_contour', 0) > 0 and 'loss_contour' in loss:
                            wandb.log({"loss_contour_all": cfg['loss_contour'] * loss['loss_contour'].item()}, step=nsteps)
                        if cfg.get('loss_contour_perc', 0) > 0 and 'loss_contour_perc' in loss:
                            wandb.log({"loss_contour_perc": cfg['loss_contour'] * cfg['loss_contour_perc'] * loss['loss_contour_perc'].item()}, step=nsteps)
                        if cfg.get('loss_multi_step_perc', 0) > 0 and 'loss_multi_step_perc' in loss:
                            wandb.log({"loss_multi_step": cfg['loss_contour'] * cfg['loss_multi_step_perc'] * loss['loss_multi_step_perc'].item()}, step=nsteps)
                        if cfg.get('loss_interval_perc', 0) > 0 and 'loss_interval_perc' in loss:
                            wandb.log({"loss_interval": cfg['loss_contour'] * cfg['loss_interval_perc'] * loss['loss_interval_perc'].item()}, step=nsteps)
                        if cfg.get('loss_shape_perc', 0) > 0 and 'loss_shape_perc' in loss:
                            wandb.log({"loss_shape": cfg['loss_contour'] * cfg['loss_shape_perc'] * loss['loss_shape_perc'].item()}, step=nsteps)
                        if cfg.get('loss_button_held', 0) > 0 and 'loss_button_held' in loss:
                            wandb.log({"loss_button_held": cfg['loss_button_held'] * loss['loss_button_held'].item()}, step=nsteps)
                        for metric_name in (
                            'loss_recons_guided', 'loss_recons_joker',
                            'acc_guided', 'acc_joker', 'joker_fraction', 'timing_contribution_ratio'
                        ):
                            if metric_name in loss:
                                wandb.log({metric_name: loss[metric_name].item()}, step=nsteps)
                        if is_joker_param_model and not is_timing_model:
                            with torch.amp.autocast(device_type=device_type, dtype=dtype):
                                train_outside_pct = encoder_outside_percentage(
                                    model, x['pitch']
                                )
                            if train_outside_pct is not None:
                                wandb.log({
                                    'train_encoder_outside_pct': train_outside_pct
                                }, step=nsteps)

                        if not is_timing_model:
                            nsteps += 1
                    if is_timing_model:
                        print(f"Timing {model._timing_stage}: step={nsteps} "
                              f"loss={objective.item():.4f} "
                              f"contribution={loss['timing_contribution_ratio'].item():.4f}")

                trace.mark('optimizer', ep, i)
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'],
                                               error_if_nonfinite=is_timing_model)
                scaler.step(optim)
                scaler.update()
                if is_timing_model:
                    nsteps += 1
                    finished = nsteps >= cfg['timing_total_steps']
                    if finished or nsteps % cfg['timing_save_every'] == 0:
                        save_timing_checkpoint(model, optim, ep, nsteps, scaler)
                    if finished:
                        print(f'Timing training saved after {nsteps} steps')
                        return

                if i % 25 == 0 or i == len(train_loader) - 1:
                    bar_train.set_description(f'Epoch: {ep} Loss: {float(loss["loss_total"]):.4}')
                bar_train.update(batch['pitch'].shape[0])

                if not is_timing_model and ((i % cfg['validate_every'] == 0) or TESTING):
                    trace.mark('validation_fetch', ep, i)
                    try:
                        val_batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_batch = next(val_iter)
                    model.eval()
                    trace.mark('validation_forward', ep, i)
                    with torch.no_grad():
                        with torch.amp.autocast(device_type=device_type, dtype=dtype):
                            vx = {
                                'pitch': val_batch['pitch'].to(device, non_blocking=True),
                                'harm_regime': val_batch['harm_regime'].to(device, non_blocking=True),
                                'harm_strength': val_batch['harm_strength'].to(device, non_blocking=True),
                                'style_pitch': val_batch['style_pitch'].to(device, non_blocking=True),
                                'style_mask': val_batch['style_mask'].to(device, non_blocking=True),
                            }
                            val_loss, val_acc = model(vx)
                            val_loss_temp = val_loss['loss_total'].item()

                            val_joker_loss = None
                            val_joker_acc = None
                            if is_joker_param_model:
                                vx_joker = dict(vx)
                                vx_joker['joker_mask'] = torch.ones_like(
                                    vx['pitch'][:, 1:], dtype=torch.bool
                                )
                                val_joker_loss, val_joker_acc = model(vx_joker)
                                val_outside_pct = (
                                    encoder_outside_percentage(model, vx['pitch'])
                                    if cfg['use_logs'] else None
                                )

                        if cfg['use_logs']:
                            wandb.log({"val_loss": val_loss['loss_total'].item()}, step=nsteps)
                            wandb.log({"val_acc": val_acc.item()}, step=nsteps)
                            if val_joker_loss is not None:
                                wandb.log({
                                    "val_joker_loss": val_joker_loss['loss_recons_joker'].item(),
                                    "val_joker_acc": val_joker_loss['acc_joker'].item(),
                                    "val_guided_loss": val_loss['loss_recons'].item(),
                                    "val_guided_acc": val_acc.item(),
                                }, step=nsteps)
                                if val_outside_pct is not None:
                                    wandb.log({
                                        'val_encoder_outside_pct': val_outside_pct
                                    }, step=nsteps)
                    model.train()
                    del val_batch, vx
                    torch.cuda.empty_cache()
                trace.mark('batch_end', ep, i)
                trace.mark('fetch_next', ep, i)

        trace.mark('epoch_end', ep, resources=True)
        if not is_timing_model and ep % cfg['save_every'] == 0:
            trace.mark('checkpoint_save', ep)
            fname = (
                './save_models/' + cfg['model_name'] + '_' +
                str(ep) + '_eps_' +
                str(nsteps) + '_steps_' +
                str(round(float(loss['loss_total'].item()), 4)) + '_loss_' + str(round(float(val_loss_temp), 4)) + '_val_loss_' +
                str(round(float(acc.item()), 4)) + '_acc.pth'
            )
            torch.save(model.state_dict(), fname)


if __name__ == '__main__':
    mp.freeze_support()
    main()
