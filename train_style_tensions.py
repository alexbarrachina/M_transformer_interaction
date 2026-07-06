#===================================================================================================
# Monster Genie train_style_tensions.py Python module
# Training with style cross-attention conditioning + dual conditioning
# (buttons + CONTINUOUS tonal tension).
#
# Unlike train_style_harm.py (which reads pre-labelled chord/movement pseudo-events
# from the pickle), the tonal-tension conditioning is SELF-SUPERVISED: the per-note
# TIV/TIS tension feature vector is computed on the fly from the target window's
# pitch sequence by tension_extractor.extract_tension_features. The pickle therefore
# only needs plain note events (giantmidi_full_*), and transposition augmentation is
# applied to the pitches before the features are (re)computed (plan Module 12).
#
# For each sample we draw TWO non-overlapping windows from the same piece:
#   - target window -> pitch + per-note tension feature (decoder input/target)
#   - style window  -> pitch-only sequence (style encoder cross-attention context)
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
from tension_extractor import (
    extract_tension_features, future_aggregate_features,
    estimate_key_prior_from_pitches, TENSION_FEATURE_DIM,
)

NSTEPS_INIT = 0
RESUME = False

#==========================================================================

def find_latest_checkpoint(checkpoint_dir: str = './save_models', model_name: str = '') -> tuple:
    """Find the latest checkpoint in the save_models directory.

    Returns:
        tuple: (checkpoint_path, epoch, steps) or (None, 0, 0) if no checkpoint found
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return None, 0, 0

    prefix = f"{model_name}_" if model_name else ""
    pattern = os.path.join(checkpoint_dir, f"{prefix}*.pth")
    checkpoint_files = glob.glob(pattern)

    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return None, 0, 0

    latest_checkpoint = None
    max_steps = -1
    max_epoch = -1

    for checkpoint_file in checkpoint_files:
        filename = os.path.basename(checkpoint_file)
        match = re.search(r'(\d+)_eps_(\d+)_steps', filename)
        if match:
            try:
                import zipfile
                with zipfile.ZipFile(checkpoint_file, 'r'):
                    pass
            except Exception:
                print(f"Skipping corrupted checkpoint: {checkpoint_file}")
                continue

            epoch = int(match.group(1))
            steps = int(match.group(2))

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
                    checkpoint_path: str, device: torch.device) -> tuple:
    """Load model and optimizer state from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict from checkpoint")

        filename = os.path.basename(checkpoint_path)
        match = re.search(r'(\d+)_eps_(\d+)_steps', filename)
        if match:
            start_epoch = int(match.group(1))
            start_steps = int(match.group(2))
        else:
            start_epoch = 0
            start_steps = 0
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        start_steps = checkpoint.get('steps', 0)
        print("Loaded full checkpoint with model and optimizer state")

    return start_epoch, start_steps


#==========================================================================

# Channels that are NOT playable notes (harmony pseudo-events); filtered out so
# the tension extractor only sees the actual pitch stream.
_NON_NOTE_CHANNELS = (HARMONY_CHANNEL, CHORDS_CHANNEL, CHORD_LABEL_CHANNEL, KEY_CHANNEL)


class TensionMusicSamplerDataset(Dataset):
    """Dataset for the tonal-tension-conditioned style model.

    Draws two non-overlapping windows from the same piece:
      - target window: pitch sequence + self-supervised per-note tension feature
      - style window:  pitch-only sequence for the StyleEncoder

    The same transposition is applied to both windows; the tension feature is
    recomputed from the transposed pitches so augmentation is consistent.
    """
    def __init__(self, data, seq_len, style_seq_len, is_eval=False, cfg=None):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.style_seq_len = style_seq_len
        self.tokens_per_note = 5  # dtime, dur, pitch, vel, chan
        self.cfg = cfg if cfg is not None else {}
        self.is_eval = is_eval

        # Tension-extractor hyperparameters (from cfg).
        self.t_short = int(self.cfg.get('tension_short_window', 8))
        self.t_alpha = float(self.cfg.get('tension_key_alpha', 2.0))
        self.t_trans_alpha = float(self.cfg.get('tension_trans_alpha', 100.0))
        self.t_tau_low = float(self.cfg.get('tension_tau_low', 8.0))
        self.t_tau_high = float(self.cfg.get('tension_tau_high', 3.0))
        self.t_profiles = str(self.cfg.get('tension_profiles', 'genie'))
        self.t_bass_weight = float(self.cfg.get('tension_bass_weight', 2.0))
        self.t_cadence_gap = float(self.cfg.get('tension_cadence_gap', 1.0))
        self.t_cadence_boost = float(self.cfg.get('tension_cadence_boost', 2.0))
        self.t_horizon = int(self.cfg.get('tension_cond_horizon', 12))

        self.piece_ranges = self._find_piece_boundaries()

        self.target_oversample = self.seq_len * 2
        self.style_oversample = self.style_seq_len * 2
        min_events = self.target_oversample + self.style_oversample

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

        num_events = len(self.data) // self.tokens_per_note
        self.all_events = self.data[:num_events * self.tokens_per_note].view(
            num_events, self.tokens_per_note
        ).long()

        self._total_samples = sum(
            (end - start) // (self.seq_len + 1)
            for start, end in self.valid_pieces
        )

    def _find_piece_boundaries(self):
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
            start = boundary_indices[i] + 1
            end = boundary_indices[i + 1] if i + 1 < len(boundary_indices) else num_events
            if end > start:
                ranges.append((start, end))
        return ranges

    def __len__(self):
        return max(1, self._total_samples)

    def __getitem__(self, index):
        piece_idx = randint(0, len(self.valid_pieces) - 1)
        start_event, end_event = self.valid_pieces[piece_idx]
        piece_events = self.all_events[start_event:end_event]  # [L, 5]
        piece_len = piece_events.shape[0]

        total_needed = self.target_oversample + self.style_oversample
        max_start = piece_len - total_needed
        window_start = randint(0, max(0, max_start))

        if random() < 0.5:
            target_start = window_start
            target_events = piece_events[target_start: target_start + self.target_oversample]
            style_events = piece_events[window_start + self.target_oversample: window_start + total_needed]
        else:
            style_events = piece_events[window_start: window_start + self.style_oversample]
            target_start = window_start + self.style_oversample
            target_events = piece_events[target_start: window_start + total_needed]

        # Notes preceding the target window: used to pre-analyze the key so the
        # key filter starts primed, mirroring inference (prompt priming).
        prefix_events = piece_events[max(0, target_start - 192): target_start]

        transposition = randint(
            -self.cfg.get('data_augment_transpose_max', 6),
            self.cfg.get('data_augment_transpose_max', 6)
        )

        target_data = self._process_target_window(target_events, transposition, prefix_events)
        style_data = self._process_style_window(style_events, transposition)

        target_data.update(style_data)
        return target_data

    # ------------------------------------------------------------------ #
    #  Target window: pitch + self-supervised tension feature              #
    # ------------------------------------------------------------------ #
    def _process_target_window(self, events, transposition, prefix_events=None):
        target_len = self.seq_len + 1

        # Keep playable notes only (drop any harmony pseudo-events).
        chan = events[:, 4]
        is_note = torch.ones_like(chan, dtype=torch.bool)
        for c in _NON_NOTE_CHANNELS:
            is_note = is_note & (chan != c)
        note_events = events[is_note]

        pitch = note_events[:, 2].clone().long()
        if pitch.shape[0] == 0:
            return self._create_dummy_target()

        def fit(x):
            if x.shape[0] < target_len:
                reps = (target_len // x.shape[0]) + 1
                x = x.repeat(reps) if x.dim() == 1 else x.repeat(reps, 1)
            return x[:target_len]

        pitch = fit(pitch)
        dtime = fit(note_events[:, 0].clone().float())
        vel = fit(note_events[:, 3].clone().float())
        # dtime tokens are in 32 ms units -> onset times in seconds for the
        # leaky-integrator key evidence (velocity-weighted, time-decayed).
        onsets = torch.cumsum(dtime, dim=0) * 0.032

        # Transposition (shared with the style window) BEFORE feature extraction so
        # the tension features are consistent with the augmented pitches.
        pitch = torch.clamp(pitch + transposition, min=0, max=VOCAB_SIZE_PITCH - 1)

        # Pre-analyze the key from the notes preceding the target window (same
        # transposition), so the key filter starts primed as at inference.
        init_posterior = None
        if prefix_events is not None and prefix_events.shape[0] > 0:
            pchan = prefix_events[:, 4]
            p_is_note = torch.ones_like(pchan, dtype=torch.bool)
            for c in _NON_NOTE_CHANNELS:
                p_is_note = p_is_note & (pchan != c)
            prefix_notes = prefix_events[p_is_note]
            if prefix_notes.shape[0] > 0:
                prefix_pitch = torch.clamp(
                    prefix_notes[:, 2].long() + transposition,
                    min=0, max=VOCAB_SIZE_PITCH - 1)
                init_posterior = estimate_key_prior_from_pitches(
                    prefix_pitch, 5.0,
                    velocities=prefix_notes[:, 3].float(), profiles=self.t_profiles,
                    bass_weight=self.t_bass_weight)

        realized_feat = extract_tension_features(
            pitch, self.t_short,
            key_alpha=self.t_alpha, trans_alpha=self.t_trans_alpha,
            tau_low=self.t_tau_low, tau_high=self.t_tau_high,
            profiles=self.t_profiles, onsets=onsets, velocities=vel,
            bass_weight=self.t_bass_weight, cadence_gap=self.t_cadence_gap,
            cadence_boost=self.t_cadence_boost,
            init_posterior=init_posterior)                                 # [T+1, F]

        # LEAK-FREE conditioning: aggregate the realized features over the
        # NEXT t_horizon notes (see future_aggregate_features). Conditioning on
        # the next note's own realized feature hands the model the predicted
        # pitch class (chroma/TIV contain it) -> ~1.0 train accuracy and a
        # channel that steers destructively at inference.
        cond_feat, cond_chroma = future_aggregate_features(
            realized_feat, pitch, self.t_horizon)                          # [T+1,F], [T+1,12]

        return {
            'pitch': pitch,
            'tension_feat': cond_feat,
            'chroma': cond_chroma,
            'intensity': self._sample_intensity(target_len),
        }

    def _sample_intensity(self, n):
        """Per-position conditioning intensity matching the inference-time
        usage pattern: mostly unconditioned, with a few command-like bursts
        that jump to a random level and release linearly over ~8-30 notes
        (mirroring interaction's TENS_DECAY_STEP). A fraction of sequences is
        fully unconditional (CFG / release branch) or fully conditioned.

        The final t_horizon positions are ALWAYS forced to 0: there the
        future-aggregate conditioning window is truncated (down to a single
        note at the last position), which would reintroduce the per-note
        leakage — and those positions are positionally identifiable in a
        fixed-length window, so the model could learn to exploit it."""
        r = random()
        if r < 0.25:
            inten = torch.zeros(n, dtype=torch.float)         # unconditional
        elif r < 0.35:
            inten = torch.ones(n, dtype=torch.float)          # fully conditioned
        else:
            inten = torch.zeros(n, dtype=torch.float)
            for _ in range(randint(2, 6)):
                start = randint(0, max(0, n - 2))
                span = randint(8, 30)
                level = 0.6 + 0.4 * random()
                ramp = torch.linspace(level, 0.0, steps=min(span, n - start))
                inten[start:start + ramp.shape[0]] = torch.maximum(
                    inten[start:start + ramp.shape[0]], ramp)
        inten[n - min(self.t_horizon, n):] = 0.0              # leak-prone tail off
        return inten

    # ------------------------------------------------------------------ #
    #  Style window: pitch-only (for StyleEncoder)                         #
    # ------------------------------------------------------------------ #
    def _process_style_window(self, events, transposition):
        chan = events[:, 4]
        is_note = torch.ones_like(chan, dtype=torch.bool)
        for c in _NON_NOTE_CHANNELS:
            is_note = is_note & (chan != c)
        note_events = events[is_note]

        pitches = note_events[:, 2].clone()
        pitches = torch.clamp(pitches + transposition, min=0, max=VOCAB_SIZE_PITCH - 1)

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
    def _create_dummy_target(self):
        target_len = self.seq_len + 1
        print("********** Creating dummy target sample **********")
        return {
            'pitch': torch.full((target_len,), 60, dtype=torch.long),
            'tension_feat': torch.zeros(target_len, TENSION_FEATURE_DIM, dtype=torch.float),
            'chroma': torch.zeros(target_len, 12, dtype=torch.float),
            'intensity': torch.zeros(target_len, dtype=torch.float),
        }


#==========================================================================

def main():
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    ''' MODEL & HYPERPARAMETERS '''
    project_name = 'monsterGenie_style_tensions'
    model_name = 'AE_style_tensions_tester_v1'
    cfg = get_model_hparams(model_name)
    model = load_model(model_name=model_name, cfg=cfg, set_only=True)
    model.to(device)

    ''' WANDB '''
    if cfg['use_logs']:
        import wandb
        wandb.login()
        wandb.init(project=project_name, name=model_name, config=cfg)

    ''' DATA '''
    train_data = Any_Pickle_File_Reader(cfg['dataset_train_path'])
    data_train = torch.tensor(train_data, dtype=torch.int16)
    del train_data
    eval_data = Any_Pickle_File_Reader(cfg['dataset_val_path'])
    data_eval = torch.tensor(eval_data, dtype=torch.int16)
    del eval_data

    style_seq_len = cfg.get('style_seq_len', 256)

    print("Building training dataset...")
    train_dataset = TensionMusicSamplerDataset(
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
    val_dataset = TensionMusicSamplerDataset(
        data_eval, cfg['seq_len'], style_seq_len, is_eval=True, cfg=cfg
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        shuffle=False
    )
    val_iter = iter(val_loader)

    ''' PRECISION/OPTIMIZER/SCALER '''
    dtype = torch.bfloat16
    optim = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    scaler = torch.amp.GradScaler(device_type)

    ''' LOAD CHECKPOINT '''
    nsteps = 0
    val_loss_temp = 0.0

    if RESUME:
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
        # Optional warm-start from a base AE_style checkpoint (strict=False): the
        # zero-initialised tension FiLM / aux head start as identity, so the
        # warm-started model reproduces the base style+button model exactly.
        init_ckpt = cfg.get('init_from_ckpt', '')
        if init_ckpt:
            sd = torch.load(init_ckpt, map_location=device)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"Warm-started from {init_ckpt}: "
                  f"{len(missing)} new params (tension FiLM/aux), "
                  f"{len(unexpected)} unexpected keys")

    ''' TRAINING '''
    for ep in range(cfg['epochs']):
        print('Epoch #', ep)

        model.train()
        with tqdm.tqdm(total=len(train_loader)) as bar_train:
            for i, batch in enumerate(train_loader):
                optim.zero_grad()

                # AE_style_tensions consumes pitch + style + the per-note tension
                # feature (+ chroma, intensity) for AdaLN-Zero FiLM + aux regression.
                x = {k: v.to(device) for k, v in batch.items()}

                with torch.amp.autocast(device_type=device_type, dtype=dtype):
                    loss, acc = model(x)
                scaler.scale(loss['loss_total']).backward()

                if (i % cfg['print_stats_every'] == 0) or TESTING:
                    if cfg['use_logs']:
                        wandb.log({"loss_total": loss['loss_total'].item()}, step=nsteps)
                        wandb.log({"train_acc": acc.item()}, step=nsteps)
                        if cfg.get('loss_margin', 0) > 0 and 'loss_margin' in loss:
                            wandb.log({"loss_margin": cfg['loss_margin'] * loss['loss_margin'].item()}, step=nsteps)
                        if cfg.get('loss_deviate', 0) > 0 and 'loss_deviate' in loss:
                            wandb.log({"loss_deviate": cfg['loss_deviate'] * loss['loss_deviate'].item()}, step=nsteps)
                        if cfg.get('loss_contour', 0) > 0 and 'loss_contour' in loss:
                            wandb.log({"loss_contour_all": cfg['loss_contour'] * loss['loss_contour'].item()}, step=nsteps)
                        if 'loss_aux_tension' in loss:
                            wandb.log({"loss_aux_tension": cfg.get('loss_aux_tension', 0.0) * loss['loss_aux_tension'].item()}, step=nsteps)
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
