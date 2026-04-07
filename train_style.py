#===================================================================================================
# Monster Genie train_style.py Python module
# Training with style cross-attention conditioning (no harmony)
# Based on train.py with piece-boundary-aware sampling for style reference windows
#
# Pickle format: flat [dtime, dur, pitch, vel, chan] (5 tokens per event)
# Piece boundaries: [126, 126, 0, 0, 0]
#
# For each sample we draw TWO non-overlapping windows from the same piece:
#   - target window → pitch (for encoder/decoder, standard autoregressive)
#   - style window  → pitch-only sequence (for StyleEncoder cross-attention)
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
mp.set_start_method('spawn', force=True)

import time
import tqdm

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

#==========================================================================

class StyleMusicSamplerDataset(Dataset):
    """
    Dataset for style-conditioned model (no harmony).

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

        # Events needed per window (no harmony filtering, so 1:1 events-to-notes)
        self.target_events = self.seq_len + 1
        self.style_events = self.style_seq_len
        total_needed = self.target_events + self.style_events

        # Keep only pieces with enough events for both windows
        self.valid_pieces = [
            (start, end) for start, end in self.piece_ranges
            if (end - start) >= total_needed
        ]

        if len(self.valid_pieces) == 0:
            raise ValueError(
                f"No pieces have enough events ({total_needed}) for "
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

        # Estimate dataset size
        self._total_samples = sum(
            (end - start) // (self.seq_len + 1)
            for start, end in self.valid_pieces
        )

    def _find_piece_boundaries(self):
        """
        Detect [126,126,0,0,0] boundary events and return per-piece (start, end)
        ranges in event-index space.
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

        total_needed = self.target_events + self.style_events
        max_start = piece_len - total_needed
        window_start = randint(0, max(0, max_start))

        # Randomly assign order so model doesn't learn positional bias
        if random() < 0.5:
            target_slice = piece_events[window_start : window_start + self.target_events]
            style_slice = piece_events[window_start + self.target_events : window_start + total_needed]
        else:
            style_slice = piece_events[window_start : window_start + self.style_events]
            target_slice = piece_events[window_start + self.style_events : window_start + total_needed]

        # Same transposition for both windows
        transposition = randint(
            -self.cfg.get('data_augment_transpose_max', 6),
            self.cfg.get('data_augment_transpose_max', 6)
        )

        target_data = self._process_target_window(target_slice, transposition)
        style_data = self._process_style_window(style_slice, transposition)

        target_data.update(style_data)
        return target_data

    # ------------------------------------------------------------------ #
    #  Target window: pitch with augmentation (same as train.py)           #
    # ------------------------------------------------------------------ #

    def _process_target_window(self, events, transposition):
        """
        Extract pitch from an event window with data augmentation.
        Returns dict with 'pitch' [T+1].
        """
        pitches = events[:, 2].clone()
        dtimes = events[:, 0].clone()
        durs = events[:, 1].clone()

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
        pitches = pitches[sorted_indices]

        # Transposition (shared with style window)
        pitches = torch.clamp(pitches + transposition, min=0, max=VOCAB_SIZE_PITCH - 1)

        return {
            'pitch': pitches,
        }

    # ------------------------------------------------------------------ #
    #  Style window: pitch-only (for StyleEncoder)                         #
    # ------------------------------------------------------------------ #

    def _process_style_window(self, events, transposition):
        """
        Extract pitch-only sequence from a style reference window.
        Returns dict with 'style_pitch' [S] and 'style_mask' [S].
        """
        pitches = events[:, 2].clone()

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
    project_name = 'monsterGenie_style'
    model_name = 'AE_style_tester'
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

    ''' TRAINING '''

    nsteps = 0

    for ep in range(cfg['epochs']):
        print('Epoch #', ep)

        model.train()
        with tqdm.tqdm(total=len(train_loader)) as bar_train:
            for i, batch in enumerate(train_loader):
                optim.zero_grad()

                x = {
                    'pitch': batch['pitch'].to(device),
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
                        if cfg.get('loss_norm_pos', 0) > 0 and 'loss_norm_pos' in loss:
                            wandb.log({"loss_norm_pos": cfg['loss_norm_pos'] * loss['loss_norm_pos'].item()}, step=nsteps)
                        if cfg.get('loss_deviate', 0) > 0 and 'loss_deviate' in loss:
                            wandb.log({"loss_deviate": cfg['loss_deviate'] * loss['loss_deviate'].item()}, step=nsteps)
                        if cfg.get('loss_margin', 0) > 0 and 'loss_margin' in loss:
                            wandb.log({"loss_margin": cfg['loss_margin'] * loss['loss_margin'].item()}, step=nsteps)
                        if cfg.get('loss_pitch_button', 0) > 0 and 'loss_pitch_button' in loss:
                            wandb.log({"loss_pitch_button": cfg['loss_pitch_button'] * loss['loss_pitch_button'].item()}, step=nsteps)
                        if cfg.get('loss_button_concentration', 0) > 0 and 'loss_button_concentration' in loss:
                            wandb.log({"loss_button_concentration": cfg['loss_button_concentration'] * loss['loss_button_concentration'].item()}, step=nsteps)
                        if cfg.get('loss_window_corr', 0) > 0 and 'loss_window_corr' in loss:
                            wandb.log({"loss_window_corr": cfg['loss_window_corr'] * loss['loss_window_corr'].item()}, step=nsteps)
                        if cfg.get('loss_latent_velocity', 0) > 0 and 'loss_latent_velocity' in loss:
                            wandb.log({"loss_latent_velocity": cfg['loss_latent_velocity'] * loss['loss_latent_velocity'].item()}, step=nsteps)
                        if cfg.get('loss_drift', 0) > 0 and 'loss_drift' in loss:
                            wandb.log({"loss_drift": cfg['loss_drift'] * loss['loss_drift'].item()}, step=nsteps)
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
                                'style_pitch': val_batch['style_pitch'].to(device),
                                'style_mask': val_batch['style_mask'].to(device),
                            }
                            val_loss, val_acc = model(vx)

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
                str(round(float(acc.item()), 4)) + '_acc.pth'
            )
            torch.save(model.state_dict(), fname)


if __name__ == '__main__':
    mp.freeze_support()
    main()
