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
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch.multiprocessing as mp

import tqdm
import glob
import re

os.environ['USE_FLASH_ATTENTION'] = '1'

from random import randint, random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from midiUtils import Any_Pickle_File_Reader
from model_loader import load_model
from models import get_model_hparams
from params import *
from x_transformer import *

NSTEPS_INIT = 2268
RESUME = True
#==========================================================================

def find_latest_checkpoint(checkpoint_dir: str = './save_models') -> tuple[str, int, int]:
    """
    Find the latest checkpoint in the save_models directory.
    
    Returns:
        tuple: (checkpoint_path, epoch, steps) or (None, 0, 0) if no checkpoint found
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return None, 0, 0
    
    # Pattern to match checkpoint files: MODEL_NAME_epoch_eps_steps_steps_loss_loss_acc_acc.pth
    pattern = os.path.join(checkpoint_dir, "*.pth")
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
                   checkpoint_path: str, device: torch.device) -> tuple[int, int]:
    """
    Load model and optimizer state from checkpoint.
    
    Returns:
        tuple: (start_epoch, start_steps)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load the state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    return start_epoch, start_steps

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
        # Pick a random valid piece
        piece_idx = randint(0, len(self.valid_pieces) - 1)
        start_event, end_event = self.valid_pieces[piece_idx]
        piece_events = self.all_events[start_event:end_event]  # [L, 5]
        piece_len = piece_events.shape[0]

        total_needed = self.target_oversample + self.style_oversample
        max_start = piece_len - total_needed
        window_start = randint(0, max(0, max_start))

        # Randomly assign order so model doesn't learn positional bias
        if random() < 0.5:
            target_events = piece_events[window_start : window_start + self.target_oversample]
            style_events = piece_events[window_start + self.target_oversample : window_start + total_needed]
        else:
            style_events = piece_events[window_start : window_start + self.style_oversample]
            target_events = piece_events[window_start + self.style_oversample : window_start + total_needed]

        # Same transposition for both windows (preserves key relationship)
        transposition = randint(
            -self.cfg.get('data_augment_transpose_max', 6),
            self.cfg.get('data_augment_transpose_max', 6)
        )

        target_data = self._process_target_window(target_events, transposition)
        style_data = self._process_style_window(style_events, transposition)

        # Merge into a single dict
        target_data.update(style_data)
        return target_data

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
        if len(note_events) < target_len:
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

        # --- Data augmentation (target only) ---
        # Time stretching
        stretch_factor = random() * self.cfg.get('data_augment_time_stretch_max', 0.05) * 2
        stretch_factor += 1 - self.cfg.get('data_augment_time_stretch_max', 0.05)
        dtimes = (dtimes.float() * stretch_factor).long()
        dtimes = torch.clamp(dtimes, min=0, max=RANGE_DTIME_SHIFT)

        stretch_factor = random() * self.cfg.get('data_augment_time_stretch_max', 0.05) * 2
        stretch_factor += 1 - self.cfg.get('data_augment_time_stretch_max', 0.05)
        durs = (durs.float() * stretch_factor).long()
        durs = torch.clamp(durs, min=0, max=RANGE_DUR_SHIFT)

        # Chord micro-alterations
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
            'pitch': torch.full((target_len,), 60, dtype=torch.long),
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

    #==========================================================================

    ''' MODEL & HYPERPARAMETERS '''
    project_name = 'monsterGenie_antic_style'
    model_name = 'AE_antic_style_v1'
    cfg = get_model_hparams(model_name)
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
    eval_data = Any_Pickle_File_Reader(cfg['dataset_val_path'])
    data_eval = torch.Tensor(eval_data)

    style_seq_len = cfg.get('style_seq_len', 256)

    print("Building training dataset...")
    train_dataset = StyleMusicSamplerDataset(
        data_train, cfg['seq_len'], style_seq_len, cfg=cfg
    )
    print(f"BATCH_SIZE: {cfg['batch_size']}")
    print(f"Dataset size: {len(train_dataset)}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        shuffle=True
    )
    print(f"Number of batches: {len(train_loader)}")

    print("Building validation dataset...")
    val_dataset = StyleMusicSamplerDataset(
        data_eval, cfg['seq_len'], style_seq_len, is_eval=True, cfg=cfg
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        shuffle=False
    )
    val_iter = iter(val_loader)

    #==========================================================================

    ''' PRECISION/OPTIMIZER/SCALER '''

    dtype = torch.bfloat16

    optim = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    scaler = torch.amp.GradScaler(device_type)

    ''' LOAD CHECKPOINT '''
    nsteps = 0

    if RESUME:
        checkpoint_path, start_epoch, start_steps = find_latest_checkpoint()
    
        if checkpoint_path:
            start_epoch, start_steps = load_checkpoint(model, optim, checkpoint_path, device)
            print(f"Resuming training from epoch {start_epoch}, step {start_steps}")
            nsteps = NSTEPS_INIT
        else:
            start_epoch = 0
            start_steps = 0
            print("Starting training from scratch (no checkpoint found)")

    ''' TRAINING '''


    for ep in range(cfg['epochs']):
        print('Epoch #', ep)

        model.train()
        with tqdm.tqdm(total=len(train_loader)) as bar_train:
            for i, batch in enumerate(train_loader):
                optim.zero_grad()

                x = {
                    'pitch': batch['pitch'].to(device),
                    'harm_regime': batch['harm_regime'].to(device),
                    'harm_strength': batch['harm_strength'].to(device),
                    'style_pitch': batch['style_pitch'].to(device),
                    'style_mask': batch['style_mask'].to(device),
                }

                with torch.amp.autocast(device_type=device_type, dtype=dtype):
                    loss, acc = model(x)
                scaler.scale(loss['loss_total']).backward()

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

                        nsteps += 1

                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                scaler.step(optim)
                scaler.update()

                bar_train.set_description(f'Epoch: {ep} Loss: {float(loss["loss_total"]):.4}')
                bar_train.update(1)

                if (i % cfg['validate_every'] == 0) or TESTING:
                    try:
                        val_batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_batch = next(val_iter)
                    model.eval()
                    with torch.no_grad():
                        with torch.amp.autocast(device_type=device_type, dtype=dtype):
                            vx = {
                                'pitch': val_batch['pitch'].to(device),
                                'harm_regime': val_batch['harm_regime'].to(device),
                                'harm_strength': val_batch['harm_strength'].to(device),
                                'style_pitch': val_batch['style_pitch'].to(device),
                                'style_mask': val_batch['style_mask'].to(device),
                            }
                            val_loss, val_acc = model(vx)
                            val_loss_temp = val_loss['loss_total'].item()

                        if cfg['use_logs']:
                            wandb.log({"val_loss": val_loss['loss_total'].item()}, step=nsteps)
                            wandb.log({"val_acc": val_acc.item()}, step=nsteps)
                    model.train()
                    del val_batch, vx
                    torch.cuda.empty_cache()

        if ep % cfg['save_every'] == 0:
            fname = (
                './save_models/' + cfg['model_name'] + '_' +
                str(ep) + '_eps_' +
                str(nsteps) + '_steps_' +
                str(round(float(loss['loss_total'].item()), 4)) + '_loss_' +
                str(round(float(val_loss_temp), 4)) + '_val_loss_' +
                str(round(float(acc.item()), 4)) + '_acc.pth'
            )
            torch.save(model.state_dict(), fname)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    mp.freeze_support()
    main()
