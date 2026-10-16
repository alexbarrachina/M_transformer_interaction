#===================================================================================================
# Monster Genie train_chords.py Python module
# Training the frozen Joker pitch decoder with momentary channel-5 chord FiLM
#
# Pickle format: flat [dtime, dur, pitch, vel, chan]. Channel 4 contains the
# chord tones, channel 120 starts a chord, and [0, 0, 0, 0, 4] ends it.
# Piece boundaries are [126, 126, 0, 0, 0]. Only performed pitches are targets.
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
# limitations under the License.
#===================================================================================================


import os
from training_trace import TrainingTrace, enable_fault_handler
enable_fault_handler()

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import tqdm
import glob
import json
import logging
import re
from itertools import count
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ['USE_FLASH_ATTENTION'] = '1'

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from midiUtils import Any_Pickle_File_Reader
from model_loader import load_model, warm_start_chord
from models import get_model_hparams
from params import CHORD_LABEL_CHANNEL, CHORDS_CHANNEL, HARMONY_CHANNEL, TESTING
from x_transformer import D_Chord_Joker

NSTEPS_INIT = 0
RESUME = os.environ.get('RESUME', '1') != '0'

#==========================================================================

class ChordPitchSamplerDataset(Dataset):
    """Full pitch history with one simulated press interval per training window.

    Start markers delimit chords even when their pitch classes are identical.
    All tones of a marked chord are assembled before assigning it to notes;
    tones may be interleaved with played notes in the serialized event stream.
    """

    def __init__(self, data: np.ndarray, seq_len: int, is_eval: bool = False,
                 transpose_max: int = 6, whole_span_prob: float = 0.5,
                 validation_seed: int = 1729, max_samples: int = 0) -> None:
        super().__init__()
        if seq_len < 1 or data.size % 5:
            raise ValueError('Expected seq_len >= 1 and flat five-token pickle events')
        if transpose_max < 0 or not 0.0 <= whole_span_prob <= 1.0 or max_samples < 0:
            raise ValueError('Invalid chord sampling settings')
        self.seq_len = seq_len
        self.is_eval = is_eval
        self.transpose_max = transpose_max
        self.whole_span_prob = whole_span_prob
        self.validation_seed = validation_seed
        events = data.reshape(-1, 5)
        boundary = (events == np.array([126, 126, 0, 0, 0])).all(axis=1)
        channel = events[:, 4]
        notes = (~boundary & (channel >= 0) & (channel < 16) & (channel != 9)
                 & (channel != HARMONY_CHANNEL) & (channel != CHORDS_CHANNEL)
                 & (events[:, 3] > 0))
        self.pitches = events[notes, 2].astype(np.int16)
        if ((self.pitches < 0) | (self.pitches > 127)).any():
            raise ValueError('Performed pitches must be in [0, 127]')
        tones = (channel == CHORDS_CHANNEL) & (events[:, 3] > 0)
        if not tones.any():
            raise ValueError('No usable channel-5 chords (internal channel 4); rebuild the updated pickles first')
        if ((events[tones, 2] < 0) | (events[tones, 2] > 127)).any():
            raise ValueError('Chord pitches must be in [0, 127]')
        starts = channel == CHORD_LABEL_CHANNEL
        off = (channel == CHORDS_CHANNEL) & (events[:, 3] == 0)
        reset = boundary | starts | off
        groups = np.cumsum(reset, dtype=np.int32)
        count = int(groups[-1]) + 1
        is_chord = np.zeros(count, dtype=np.bool_)
        is_chord[groups[starts]] = True
        if not is_chord[groups[tones]].all():
            raise ValueError('Channel-5 tones require a preceding chord-start marker on channel 120')
        self.chroma = np.zeros((count, 12), dtype=np.bool_)
        self.chroma[groups[tones], events[tones, 2] % 12] = True
        lowest = np.full(count, 128, dtype=np.int16)
        np.minimum.at(lowest, groups[tones], events[tones, 2])
        self.anchors = (lowest % 12).astype(np.int64)
        self.anchors[lowest == 128] = 0
        self.groups = groups[notes]
        self.groups[lowest[self.groups] == 128] = 0

        note_counts = np.cumsum(notes, dtype=np.int64)
        self.piece_edges = np.concatenate(([0], note_counts[boundary], [len(self.pitches)]))
        self.note_piece = np.cumsum(boundary, dtype=np.int32)[notes]
        lengths = np.diff(self.piece_edges)
        positions = np.arange(len(self.pitches))
        eligible = ((self.groups > 0) & (lengths[self.note_piece] > seq_len)
                    & (positions > self.piece_edges[self.note_piece]))
        self.targets = np.flatnonzero(eligible)
        if not len(self.targets):
            raise ValueError('No marked channel-5 chord targets with enough within-piece pitch history')
        self.sample_count = max(1, len(self.targets) // seq_len)
        if max_samples:
            self.sample_count = min(self.sample_count, max_samples)

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        generator = torch.Generator().manual_seed(self.validation_seed + index) if self.is_eval else None
        slot = (index * len(self.targets) // len(self) if self.is_eval
                else int(torch.randint(len(self.targets), (), generator=generator)))
        target = int(self.targets[slot])
        piece = int(self.note_piece[target])
        lower, upper = int(self.piece_edges[piece]), int(self.piece_edges[piece + 1])
        low = max(lower, target - self.seq_len)
        high = min(target - 1, upper - self.seq_len - 1)
        start = int(torch.randint(low, high + 1, (), generator=generator))
        stop = start + self.seq_len + 1
        pitch = torch.from_numpy(self.pitches[start:stop].astype(np.int64))
        shift = 0
        if not self.is_eval and self.transpose_max:
            low = max(-self.transpose_max, -int(pitch.min()))
            high = min(self.transpose_max, 127 - int(pitch.max()))
            shift = int(torch.randint(low, high + 1, (), generator=generator))
            pitch += shift

        group = int(self.groups[target])
        positions = np.flatnonzero(self.groups[start:stop] == group)
        left, right = max(1, int(positions[0])), int(positions[-1]) + 1
        if float(torch.rand((), generator=generator)) >= self.whole_span_prob:
            # Press/release inside this known chord; no future chord is exposed.
            left = int(torch.randint(left, target - start + 1, (), generator=generator))
            right = int(torch.randint(target - start + 1, right + 1, (), generator=generator))
        active = torch.zeros(self.seq_len + 1, dtype=torch.bool)
        active[left:right] = True
        chord = torch.from_numpy(self.chroma[group].copy()).roll(shift)
        chroma = active.unsqueeze(-1) & chord.unsqueeze(0)
        anchor = torch.zeros(self.seq_len + 1, dtype=torch.long)
        anchor[active] = (int(self.anchors[group]) + shift) % 12
        return {'pitch': pitch, 'chord_chroma': chroma, 'chord_anchor_pc': anchor,
                'chord_active': active}


def find_latest_checkpoint(checkpoint_dir: str = './save_models',
                           model_name: Optional[str] = None,
                           latest_path: Optional[str] = None) -> Tuple[Optional[str], int, int]:
    """Find the most recent numbered or rolling checkpoint for this model."""
    file_pattern = f'{model_name}_*.pth' if model_name else '*.pth'
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, file_pattern))
    latest_checkpoint = None
    max_steps, max_epoch = -1, -1
    for checkpoint_file in checkpoint_files:
        match = re.search(r'(\d+)_eps_(\d+)_steps', os.path.basename(checkpoint_file))
        if match:
            epoch, steps = int(match.group(1)), int(match.group(2))
            if (steps, epoch) > (max_steps, max_epoch):
                latest_checkpoint, max_steps, max_epoch = checkpoint_file, steps, epoch
    if latest_path and os.path.isfile(latest_path):
        saved = torch.load(latest_path, map_location='cpu', weights_only=False)
        if saved.get('cfg', {}).get('model_name') != model_name:
            raise ValueError(f'Checkpoint does not belong to {model_name}: {latest_path}')
        epoch, steps = int(saved['epoch']), int(saved['steps'])
        if (steps, epoch) >= (max_steps, max_epoch):
            latest_checkpoint, max_steps, max_epoch = latest_path, steps, epoch
    if latest_checkpoint:
        print(f'Found latest checkpoint: {latest_checkpoint}')
        print(f'Resuming from epoch {max_epoch}, step {max_steps}')
        return latest_checkpoint, max_epoch, max_steps
    print('No model-specific checkpoint found')
    return None, 0, 0


def load_checkpoint(model: D_Chord_Joker, optimizer: torch.optim.Optimizer,
                    checkpoint_path: str, device: torch.device,
                    scaler: Optional[torch.amp.GradScaler] = None) -> Tuple[int, int, float]:
    """Restore the adapter and optimizer without changing the frozen pitch prior."""
    print(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get('data_signature') != model.cfg['data_signature']:
        raise ValueError('Dataset files changed since this checkpoint; use RESUME=0 for the updated pickles')
    for key in ('model_name', 'seq_len', 'chord_cond_dim', 'chord_film_start_frac',
                'chord_film_scale_limit', 'chord_film_shift_limit', 'chord_whole_span_prob',
                'chord_validation_seed'):
        if checkpoint['cfg'].get(key) != model.cfg.get(key):
            raise ValueError(f'Chord resume configuration differs for {key}')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    model.cfg['init_from_ckpt'] = checkpoint['cfg']['init_from_ckpt']
    # A capped/interrupted epoch resumes with newly sampled windows in that epoch.
    start_epoch = checkpoint['epoch'] + int(checkpoint.get('epoch_complete', True))
    print('Loaded full checkpoint with model and optimizer state')
    return start_epoch, checkpoint['steps'], checkpoint.get('best_val_nll', float('inf'))


def save_checkpoint(path: str, checkpoint: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    torch.save(checkpoint, temporary)
    os.replace(temporary, destination)


@torch.inference_mode()
def validate_chords(model: D_Chord_Joker, loader: DataLoader,
                    device: torch.device) -> Dict[str, float]:
    """Same target notes and full history for true, free and shifted-chord inputs."""
    model.eval()
    sums = torch.zeros(5, dtype=torch.float64, device=device)
    for batch in loader:
        tokens = {key: value.to(device) for key, value in batch.items()}
        labels = tokens['pitch'][:, 1:]
        weights = tokens['chord_active'][:, 1:].float()
        shifted = dict(tokens)
        shifted['chord_chroma'] = tokens['chord_chroma'].roll(1, dims=-1)
        shifted['chord_anchor_pc'] = (tokens['chord_anchor_pc'] + 1) % 12
        for index, variant in enumerate((tokens, {'pitch': tokens['pitch']}, shifted)):
            logits = model.pitch_logits(variant)
            ce = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
            sums[index] += (ce * weights).sum()
            if index == 0:
                sums[3] += ((logits.argmax(-1) == labels) * weights).sum()
        sums[4] += weights.sum()
    if not bool(sums[4] > 0):
        raise ValueError('Validation contains no active chord targets')
    true, free, shifted, accuracy = (sums[:4] / sums[4]).cpu().tolist()
    return {'nll_true': true, 'nll_free': free, 'nll_shifted': shifted,
            'nll_gain_vs_free': free - true, 'nll_gap_shifted': shifted - true,
            'accuracy': accuracy}


#==========================================================================

def main() -> None:
    # Only the parent process opens this log; spawned workers do not call main().
    log_path = Path(os.environ.get('TRAIN_LOG_FILE', 'train_chords.log')).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.Logger('train_chords', level=logging.INFO)
    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s [%(process)d] %(levelname)s %(message)s'))
    logger.addHandler(handler)
    print(f'Local training log: {log_path.resolve()}', flush=True)
    logger.info('Training run started')
    try:
        _train(logger)
    except Exception:
        logger.exception('Training failed')
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()


def _train(logger: logging.Logger) -> None:
    # Set up CUDA settings
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)

    #==========================================================================

    ''' DEVICE '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = device.type
    trace = TrainingTrace(torch, device)
    trace.mark('startup', resources=True)

    #==========================================================================

    ''' MODEL & HYPERPARAMETERS '''
    project_name = 'monsterGenie_chords'
    model_name = os.environ.get('MODEL_NAME', 'D_Chord_Joker_little_v1')
    cfg = get_model_hparams(model_name)
    if cfg['model_type'] != 'D_Chord_Joker':
        raise ValueError('train_chords.py requires a D_Chord_Joker preset')
    cfg['batch_size'] = int(os.environ.get('BATCH_SIZE', cfg['batch_size']))
    cfg['num_workers'] = int(os.environ.get('NUM_WORKERS', cfg['num_workers']))
    max_steps = int(os.environ.get('MAX_STEPS', '0'))  # Additional updates; 0 is unlimited.
    if max_steps < 0 or cfg['batch_size'] < 1 or cfg['num_workers'] < 0:
        raise ValueError('Invalid batch size, worker count or MAX_STEPS')
    logger.info('Environment: device=%s torch=%s CUDA=%s', device, torch.__version__, torch.version.cuda)
    logger.info('Configuration: %s', json.dumps(cfg, sort_keys=True))
    pin_memory = device_type == 'cuda' and os.environ.get('PIN_MEMORY', '1') != '0'
    persistent_workers = cfg['num_workers'] > 0 and os.environ.get('PERSISTENT_WORKERS', '1') != '0'
    model = load_model(model_name=model_name, cfg=cfg, set_only=True)
    model.to(device)

    #==========================================================================

    ''' WANDB '''
    wandb_run = None
    if cfg['use_logs']:
        import wandb
        wandb.login()
        wandb_run = wandb.init(project=project_name, name=model_name, config=cfg)

    #==========================================================================

    ''' DATA '''

    train_data = Any_Pickle_File_Reader(cfg['dataset_train_path'])
    # Keep compact CPU arrays; the sampler returns tensors for the selected window.
    data_train = np.asarray(train_data, dtype=np.int16)
    del train_data
    print('Building training dataset...')
    train_dataset = ChordPitchSamplerDataset(
        data_train, cfg['seq_len'], transpose_max=cfg['data_augment_transpose_max'],
        whole_span_prob=cfg['chord_whole_span_prob'], validation_seed=cfg['chord_validation_seed'],
        max_samples=cfg.get('chord_max_samples', 0))
    del data_train
    print(f'BATCH_SIZE: {cfg["batch_size"]}')
    print(f'NUM_WORKERS: {cfg["num_workers"]}')
    print(f'Dataset size: {len(train_dataset)}')
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
        shuffle=True, pin_memory=pin_memory, persistent_workers=persistent_workers)
    print(f'Number of batches: {len(train_loader)}')
    logger.info('Training dataset: samples=%d batches=%d batch_size=%d num_workers=%d',
                len(train_dataset), len(train_loader), cfg['batch_size'], cfg['num_workers'])

    eval_data = Any_Pickle_File_Reader(cfg['dataset_val_path'])
    data_eval = np.asarray(eval_data, dtype=np.int16)
    del eval_data
    print('Building validation dataset...')
    val_dataset = ChordPitchSamplerDataset(
        data_eval, cfg['seq_len'], is_eval=True,
        whole_span_prob=cfg['chord_whole_span_prob'], validation_seed=cfg['chord_validation_seed'],
        max_samples=cfg.get('chord_max_val_samples', 0))
    del data_eval
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], num_workers=0,
                            shuffle=False, pin_memory=pin_memory)
    val_iter = iter(val_loader)
    logger.info('Validation dataset: samples=%d batches=%d validate_every=%d',
                len(val_dataset), len(val_loader), cfg['validate_every'])
    cfg['data_signature'] = {}
    for key in ('dataset_train_path', 'dataset_val_path'):
        path = Path(cfg[key] if cfg[key].endswith('.pickle') else cfg[key] + '.pickle').resolve()
        cfg['data_signature'][key] = (str(path), path.stat().st_size, path.stat().st_mtime_ns)

    #==========================================================================

    ''' PRECISION/OPTIMIZER/SCALER '''
    dtype = torch.bfloat16
    # The base decoder stays frozen throughout this experiment.
    optim = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg['learning_rate'])
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == torch.float16))

    ''' LOAD CHECKPOINT '''
    nsteps = NSTEPS_INIT
    start_epoch = 0
    best_val_nll = float('inf')
    checkpoint_path = None
    if RESUME:
        checkpoint_path, _, _ = find_latest_checkpoint(
            checkpoint_dir=str(Path(cfg['ckpt_file_name']).parent), model_name=model_name,
            latest_path=cfg['ckpt_file_name'])
    if checkpoint_path:
        start_epoch, nsteps, best_val_nll = load_checkpoint(model, optim, checkpoint_path, device, scaler)
        print(f'Resuming training from epoch {start_epoch}, step {nsteps}')
        logger.info('Resuming from %s: epoch=%d step=%d', checkpoint_path, start_epoch, nsteps)
    else:
        warm_start_chord(model, cfg['init_from_ckpt'])
        print(f'Warm-started chord adapter from {cfg["init_from_ckpt"]}')
        logger.info('Warm-started chord adapter from %s', cfg['init_from_ckpt'])

    ''' TRAINING '''
    val_loss_temp = 0.0
    validation: Dict[str, float] = {}
    stop_step = nsteps + max_steps if max_steps else 0
    ep = start_epoch
    epoch_complete = False

    def checkpoint() -> dict:
        return {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict(),
                'scaler_state_dict': scaler.state_dict(), 'cfg': cfg, 'epoch': ep,
                'epoch_complete': epoch_complete, 'steps': nsteps, 'best_val_nll': best_val_nll,
                'validation': validation, 'data_signature': cfg['data_signature']}

    epochs = count(start_epoch) if cfg['epochs'] is None else range(start_epoch, cfg['epochs'])
    try:
        for ep in epochs:
            print('Epoch #', ep)
            logger.info('Epoch started: epoch=%d step=%d', ep, nsteps)
            trace.mark('epoch_start_fetch_next', ep, resources=True)
            epoch_complete = False
            model.train()
            with tqdm.tqdm(total=len(train_dataset), unit='samp') as bar_train:
                for i, batch in enumerate(train_loader):
                    trace.mark('batch_received_transfer', ep, i)
                    optim.zero_grad()
                    x = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                    trace.mark('forward', ep, i)
                    with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=device_type == 'cuda'):
                        loss, acc = model(x)
                    trace.mark('backward', ep, i)
                    scaler.scale(loss['loss_total']).backward()
                    trace.mark('logging', ep, i)
                    if (i % cfg['print_stats_every'] == 0) or TESTING:
                        metrics = {name: value.item() for name, value in loss.items()}
                        metrics.update(epoch=ep, batch=i, step=nsteps, train_acc=acc.item(),
                                       grad_scale=scaler.get_scale())
                        logger.info('Training metrics: %s', json.dumps(metrics, sort_keys=True))
                        if cfg['use_logs']:
                            wandb.log({**{name: value.item() for name, value in loss.items()},
                                       'train_acc': acc.item()}, step=nsteps)
                    trace.mark('optimizer', ep, i)
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                    scaler.step(optim)
                    scaler.update()
                    nsteps += 1  # Checkpoints count optimizer updates, independently of logging.
                    if i % 25 == 0 or i == len(train_loader) - 1:
                        bar_train.set_description(f'Epoch: {ep} Loss: {float(loss["loss_total"]):.4}')
                    bar_train.update(batch['pitch'].shape[0])

                    if (i % cfg['validate_every'] == 0) or TESTING:
                        trace.mark('validation_fetch', ep, i)
                        try:
                            val_batch = next(val_iter)
                        except StopIteration:
                            val_iter = iter(val_loader)
                            val_batch = next(val_iter)
                        model.eval()
                        trace.mark('validation_forward', ep, i)
                        with torch.no_grad():
                            with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=device_type == 'cuda'):
                                vx = {key: value.to(device, non_blocking=True) for key, value in val_batch.items()}
                                val_loss, val_acc = model(vx)
                                val_loss_temp = val_loss['loss_total'].item()
                            logger.info('Validation metrics: epoch=%d batch=%d step=%d val_loss=%.6f val_acc=%.6f',
                                        ep, i, nsteps, val_loss_temp, val_acc.item())
                            if cfg['use_logs']:
                                wandb.log({'val_loss': val_loss_temp, 'val_acc': val_acc.item()}, step=nsteps)
                        model.train()
                        del val_batch, vx
                        torch.cuda.empty_cache()
                    trace.mark('batch_end', ep, i)
                    trace.mark('fetch_next', ep, i)
                    epoch_complete = i + 1 == len(train_loader)
                    if stop_step and nsteps >= stop_step:
                        break

            trace.mark('epoch_end', ep, resources=True)
            logger.info('Epoch ended: epoch=%d step=%d', ep, nsteps)
            # Reconstruction diagnostics reveal whether chord inputs affect predictions;
            # they are not a musical-quality or performer-controllability score.
            validation = validate_chords(model, val_loader, device)
            val_loss_temp = validation['nll_true']
            improved = val_loss_temp < best_val_nll
            best_val_nll = min(best_val_nll, val_loss_temp)
            logger.info('Chord validation: %s', json.dumps(validation, sort_keys=True))
            print('Chord validation:', validation)
            if cfg['use_logs']:
                wandb.log({f'validation/{key}': value for key, value in validation.items()}, step=nsteps)

            trace.mark('checkpoint_save', ep)
            state = checkpoint()
            save_checkpoint(cfg['ckpt_file_name'], state)
            destination = Path(cfg['ckpt_file_name'])
            if improved:
                save_checkpoint(str(destination.with_name(destination.stem + '_best.pth')), state)
            if epoch_complete and ep % cfg['save_every'] == 0:
                fname = (
                    str(destination.parent) + '/' + cfg['model_name'] + '_' +
                    str(ep) + '_eps_' + str(nsteps) + '_steps_' +
                    str(round(float(loss['loss_total'].item()), 4)) + '_loss_' +
                    str(round(float(val_loss_temp), 4)) + '_val_loss_' +
                    str(round(float(acc.item()), 4)) + '_acc.pth')
                save_checkpoint(fname, state)
                logger.info('Checkpoint saved: path=%s epoch=%d step=%d', fname, ep, nsteps)
            logger.info('Rolling checkpoint saved: path=%s step=%d', cfg['ckpt_file_name'], nsteps)
            if stop_step and nsteps >= stop_step:
                break
    except KeyboardInterrupt:
        save_checkpoint(cfg['ckpt_file_name'], checkpoint())
        print(f'Interrupted; saved step {nsteps} to {cfg["ckpt_file_name"]}')
        logger.info('Interrupted: saved step=%d', nsteps)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    logger.info('Training finished: step=%d', nsteps)


if __name__ == '__main__':
    mp.freeze_support()
    main()
