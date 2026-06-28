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
# Reduce CUDA allocator fragmentation / peak VRAM (must be set before torch
# initialises the CUDA context, i.e. before importing torch below).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

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

NSTEPS_INIT = 56
RESUME = True

#==========================================================================

#def find_latest_checkpoint(checkpoint_dir: str = './save_models') -> tuple[str, int, int]:
def find_latest_checkpoint(checkpoint_dir: str = './save_models', model_name: str = '') -> tuple[str, int, int]:
    """Find the latest checkpoint in the save_models directory.
    
    Returns:
        tuple: (checkpoint_path, epoch, steps) or (None, 0, 0) if no checkpoint found
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return None, 0, 0
    
    # Pattern to match checkpoint files: MODEL_NAME_epoch_eps_steps_steps_loss_loss_acc_acc.pth
    prefix = f"{model_name}_" if model_name else ""
    pattern = os.path.join(checkpoint_dir, f"{prefix}*.pth")
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
            # Skip corrupted (truncated) checkpoint files
            try:
                import zipfile
                with zipfile.ZipFile(checkpoint_file, 'r'):
                    pass
            except Exception:
                print(f"Skipping corrupted checkpoint: {checkpoint_file}")
                continue

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
        Extract pitch + forward-filled harmony conditioning from an event window.

        Returns per-note tensors (length seq_len+1):
          'pitch'            [T+1] long
          'harm_movement'    [T+1] long  (0..8; 8 = joker; forward-filled from ch3)
          'transition_phase' [T+1] float (1.0 at movement onset -> ~0 before next; 0 = unguided)
          'chroma'           [T+1, 12] float (active chord pitch classes)
          'bass_pc'          [T+1] long  (0..11, 12 = none)
          'root_pc'          [T+1] long  (0..11, 12 = unknown)
          'quality_id'       [T+1] long
          'function_id'      [T+1] long
          'key_pc'           [T+1] long  (0..11, 12 = unknown)
          'mode'             [T+1] long  (0=maj, 1=min, 2=unknown)
          'intensity'        [T+1] float (conditioning strength; 1.0 in data)

        This is CPU-side dataloader code: a single linear pass over the window
        (~1k events) is clearest/cheapest; transposition uses tensor ops.
        NOTE: the old time-stretch / chord-reorder augmentations are dropped here
        because they would scramble the per-note harmony alignment; transposition
        (shared with the style window) is the augmentation that matters for harmony.
        """
        target_len = self.seq_len + 1
        ev = events.tolist()

        # rolling conditioning state
        cur_move = MOVE_STABILIZE
        cur_root = PC_UNKNOWN; cur_qual = QUALITY_UNKNOWN; cur_func = FUNC_UNKNOWN
        cur_key = PC_UNKNOWN; cur_mode = MODE_UNKNOWN
        active_pcs = set(); active_bass_pitch = -1   # lowest sounding chord-tone pitch
        move_group = 0                               # 0 = before any movement (unguided)

        p_pitch = []; p_move = []; p_root = []; p_qual = []; p_func = []
        p_key = []; p_mode = []; p_chroma = []; p_bass = []; p_group = []

        for t0, t1, t2, t3, t4 in ev:
            if t4 == HARMONY_CHANNEL:
                cur_move = min(max(t2 - MOVE_PITCH_BASE, 0), NUM_MOVEMENTS - 1)
                move_group += 1
            elif t4 == CHORD_LABEL_CHANNEL:
                cur_root = t1; cur_qual = t2; cur_func = t3
                active_pcs = set(); active_bass_pitch = -1   # a new chord begins
            elif t4 == KEY_CHANNEL:
                cur_key = t1; cur_mode = t2
            elif t4 == CHORDS_CHANNEL:
                if t2 == 0 and t3 == 0:                      # chord-off marker
                    active_pcs = set(); active_bass_pitch = -1
                else:                                        # chord tone
                    active_pcs.add(t2 % 12)
                    if active_bass_pitch < 0 or t2 < active_bass_pitch:
                        active_bass_pitch = t2
            else:
                # playable note: snapshot the current conditioning state
                p_pitch.append(t2)
                p_move.append(cur_move)
                p_root.append(cur_root); p_qual.append(cur_qual); p_func.append(cur_func)
                p_key.append(cur_key); p_mode.append(cur_mode)
                row = [0.0] * 12
                for pc in active_pcs:
                    row[pc] = 1.0
                p_chroma.append(row)
                p_bass.append(active_bass_pitch % 12 if active_bass_pitch >= 0 else PC_UNKNOWN)
                p_group.append(move_group)

        if len(p_pitch) == 0:
            return self._create_dummy_target()

        # transition_phase: linear decay over each movement span (group 0 = unguided = 0.0)
        n = len(p_group)
        p_phase = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and p_group[j] == p_group[i]:
                j += 1
            if p_group[i] > 0:
                span = j - i
                for k in range(span):
                    p_phase[i + k] = 1.0 - k / span
            i = j

        pitch = torch.tensor(p_pitch, dtype=torch.long)
        move = torch.tensor(p_move, dtype=torch.long)
        phase = torch.tensor(p_phase, dtype=torch.float)
        root = torch.tensor(p_root, dtype=torch.long)
        qual = torch.tensor(p_qual, dtype=torch.long)
        func = torch.tensor(p_func, dtype=torch.long)
        key = torch.tensor(p_key, dtype=torch.long)
        mode = torch.tensor(p_mode, dtype=torch.long)
        bass = torch.tensor(p_bass, dtype=torch.long)
        chroma = torch.tensor(p_chroma, dtype=torch.float)   # [n, 12]

        def fit(x):
            if x.shape[0] < target_len:
                reps = (target_len // x.shape[0]) + 1
                x = x.repeat(reps) if x.dim() == 1 else x.repeat(reps, 1)
            return x[:target_len]

        pitch = fit(pitch); move = fit(move); phase = fit(phase)
        root = fit(root); qual = fit(qual); func = fit(func)
        key = fit(key); mode = fit(mode); bass = fit(bass); chroma = fit(chroma)

        # --- Transposition (shared with style window) ---
        pitch = torch.clamp(pitch + transposition, min=0, max=VOCAB_SIZE_PITCH - 1)
        t = transposition % 12  # non-negative equivalent for pitch-class rotation
        chroma = torch.roll(chroma, shifts=t, dims=1)
        bass = torch.where(bass != PC_UNKNOWN, (bass + t) % 12, bass)
        root = torch.where(root != PC_UNKNOWN, (root + t) % 12, root)
        key = torch.where(key != PC_UNKNOWN, (key + t) % 12, key)

        return {
            'pitch': pitch,
            'harm_movement': move,
            'transition_phase': phase,
            'chroma': chroma,
            'bass_pc': bass,
            'root_pc': root,
            'quality_id': qual,
            'function_id': func,
            'key_pc': key,
            'mode': mode,
            'intensity': torch.ones(target_len, dtype=torch.float),
        }

    # ------------------------------------------------------------------ #
    #  Style window: pitch-only (for StyleEncoder)                         #
    # ------------------------------------------------------------------ #

    def _process_style_window(self, events, transposition):
        """
        Extract pitch-only sequence from a style reference window.
        Returns dict with 'style_pitch' [S] and 'style_mask' [S].
        """
        # Filter to playable note events only (drop all harmony pseudo-events:
        # movements, chord tones, chord labels and key markers)
        chan = events[:, 4]
        is_note = (
            (chan != HARMONY_CHANNEL) &
            (chan != CHORDS_CHANNEL) &
            (chan != CHORD_LABEL_CHANNEL) &
            (chan != KEY_CHANNEL)
        )
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
            'harm_movement': torch.zeros(target_len, dtype=torch.long),
            'transition_phase': torch.zeros(target_len, dtype=torch.float),
            'chroma': torch.zeros(target_len, 12, dtype=torch.float),
            'bass_pc': torch.full((target_len,), PC_UNKNOWN, dtype=torch.long),
            'root_pc': torch.full((target_len,), PC_UNKNOWN, dtype=torch.long),
            'quality_id': torch.zeros(target_len, dtype=torch.long),
            'function_id': torch.zeros(target_len, dtype=torch.long),
            'key_pc': torch.full((target_len,), PC_UNKNOWN, dtype=torch.long),
            'mode': torch.full((target_len,), MODE_UNKNOWN, dtype=torch.long),
            'intensity': torch.zeros(target_len, dtype=torch.float),
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
    project_name = 'monsterGenie_style_harm'
    model_name = 'AE_style_harm_tester_v1'
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

    # Load as a compact dtype and free the raw Python list immediately: token
    # values are small ints (0..128) so int16 halves the float32 host footprint
    # and drops the duplicate list. The dataset's .view(...).long() still works.
    train_data = Any_Pickle_File_Reader(cfg['dataset_train_path'])
    data_train = torch.tensor(train_data, dtype=torch.int16)
    del train_data
    eval_data = Any_Pickle_File_Reader(cfg['dataset_val_path'])
    data_eval = torch.tensor(eval_data, dtype=torch.int16)
    del eval_data

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
    val_loss_temp = 0.0

    if RESUME:
        #checkpoint_path, start_epoch, start_steps = find_latest_checkpoint()
        checkpoint_path, start_epoch, start_steps = find_latest_checkpoint(model_name=model_name)
    
        if checkpoint_path:
            start_epoch, start_steps = load_checkpoint(model, optim, checkpoint_path, device)
            print(f"Resuming training from epoch {start_epoch}, step {start_steps}")
            nsteps = NSTEPS_INIT
        else:
            start_epoch = 0
            start_steps = 0
            print("Starting training from scratch (no checkpoint found)")

    else:
        # Optional warm-start from a base AE_style_v2 checkpoint (strict=False): the
        # zero-initialised harmony FiLM / planner / aux heads start as identity, so the
        # warm-started model reproduces the base style+button model exactly.
        init_ckpt = cfg.get('init_from_ckpt', '')
        if init_ckpt:
            sd = torch.load(init_ckpt, map_location=device)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"Warm-started from {init_ckpt}: "
                f"{len(missing)} new params (harmony/planner/aux), "
                f"{len(unexpected)} unexpected keys")
            


    ''' TRAINING '''

    for ep in range(cfg['epochs']):
        print('Epoch #', ep)

        model.train()
        with tqdm.tqdm(total=len(train_loader)) as bar_train:
            for i, batch in enumerate(train_loader):
                optim.zero_grad()

                # AE_style_harm consumes pitch + style + the forward-filled harmony
                # fields (movement, chroma, root/quality/function/key/mode, phase,
                # intensity) for AdaLN-Zero FiLM + chord planner + aux supervision.
                x = {k: v.to(device) for k, v in batch.items()}

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
                        if 'loss_chord_plan' in loss:
                            wandb.log({"loss_chord_plan": cfg.get('loss_chord_plan', 0.0) * loss['loss_chord_plan'].item()}, step=nsteps)
                        if 'loss_aux_chord' in loss:
                            wandb.log({"loss_aux_chord": cfg.get('loss_aux_chord', 0.0) * loss['loss_aux_chord'].item()}, step=nsteps)
                        if 'loss_move_recover' in loss:
                            wandb.log({"loss_move_recover": cfg.get('loss_move_recover', 0.0) * loss['loss_move_recover'].item()}, step=nsteps)
                        if 'loss_film_reg' in loss:
                            wandb.log({"loss_film_reg": cfg.get('loss_film_reg', 0.0) * loss['loss_film_reg'].item()}, step=nsteps)

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
                            vx = {k: v.to(device) for k, v in val_batch.items()}
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
            fname_tmp = fname + '.tmp'
            torch.save(model.state_dict(), fname_tmp)
            os.replace(fname_tmp, fname)



if __name__ == '__main__':
    mp.freeze_support()
    main()
