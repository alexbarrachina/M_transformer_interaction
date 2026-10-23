#===================================================================================================
# Monster Genie train_style_chords.py Python module
# Training the chord and change FiLM adapters of D_Chord_Joker over the frozen style joker
# (AE_style_jokerParam_chords), one stage after the other:
#   1. AE_style_jokerParam_chords: chord adapter over AE_style_jokerParam_tester_v2.
#   2. AE_style_jokerParam_change: change adapter over AE_style_jokerParam_chords_best.pth.
# MODEL_NAME=<preset> runs a single stage. Both use the warmup + cosine LR schedule of train_chords.py.
#
# Pickle format: as in train_chords.py (giantmidi_full_w_chords_unseen_*). Each sample is a
# train_chords.py press window plus a style window from the same piece, as in train_style.py.
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

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import tqdm
import json
import logging
from itertools import count
from pathlib import Path
from typing import Dict

os.environ['USE_FLASH_ATTENTION'] = '1'

import numpy as np
import torch
from torch.utils.data import DataLoader

from midiUtils import Any_Pickle_File_Reader
from model_loader import load_model, warm_start_chord
from models import get_model_hparams
from params import TESTING
from train_chords import (ChordPitchSamplerDataset, collate_chord_windows, find_latest_checkpoint,
                          load_checkpoint, save_checkpoint, scheduled_lr, validate_chords)

NSTEPS_INIT = 0
RESUME = os.environ.get('RESUME', '1') != '0'

STAGES = ('AE_style_jokerParam_chords', 'AE_style_jokerParam_change')

JOKER_GUIDED_FRACTION = 0.25
JOKER_MIXED_FRACTION = 0.50
JOKER_MIN_SPAN = 32
JOKER_MAX_SPAN = 128

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


#==========================================================================

class StyleChordSamplerDataset(ChordPitchSamplerDataset):
    """ChordPitchSamplerDataset press windows plus a style window for the StyleEncoder.

    The style window holds the style_seq_len performed notes next to the target window, on its
    longer side, with the same transposition (train_style.py also takes both windows from one
    stretch of a piece). Only pieces with room for both windows are used, so a style window is
    at least half full; a shorter one is padded and masked.
    """

    def __init__(self, data: np.ndarray, seq_len: int, style_seq_len: int, **kwargs) -> None:
        super().__init__(data, seq_len, include_source=True, **kwargs)
        self.style_seq_len = style_seq_len
        lengths = np.diff(self.piece_edges)
        roomy = lengths[self.note_piece] >= seq_len + 1 + style_seq_len
        self.targets = self.targets[roomy[self.targets]]
        self.onsets = self.onsets[roomy[self.onsets]]
        if not len(self.targets) or ((self.onset_prob or self.change_control) and not len(self.onsets)):
            raise ValueError('No chord targets in pieces long enough for a style window')
        self.sample_count = min(self.sample_count, max(1, len(self.targets) // seq_len))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        start, stop, shift = (int(sample.pop(key)) for key in ('source_start', 'source_stop', 'transpose'))
        piece = int(self.note_piece[start])
        lower, upper = int(self.piece_edges[piece]), int(self.piece_edges[piece + 1])
        if upper - stop >= start - lower:
            begin, end = stop, min(upper, stop + self.style_seq_len)
        else:
            begin, end = max(lower, start - self.style_seq_len), start
        style = torch.from_numpy(self.pitches[begin:end].astype(np.int64)) + shift
        sample['style_pitch'] = torch.zeros(self.style_seq_len, dtype=torch.long)
        sample['style_pitch'][:len(style)] = style.clamp(0, 127)
        sample['style_mask'] = torch.arange(self.style_seq_len) < len(style)
        return sample


@torch.inference_mode()
def validate_style_chords(model, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """validate_chords with guided buttons everywhere, then with joker buttons everywhere."""
    validation: Dict[str, float] = {}
    for mode, joker in (('guided', False), ('joker', True)):
        batches = ({**batch, 'joker_mask': torch.full_like(batch['pitch'][:, 1:], joker, dtype=torch.bool)}
                   for batch in loader)
        validation.update({f'{mode}_{key}': value
                           for key, value in validate_chords(model, batches, device).items()})
    return validation


#==========================================================================

def main():
    # Only the parent process opens this log; spawned dataset workers import
    # the module without calling main(). FileHandler flushes after each record.
    log_path = Path(os.environ.get('TRAIN_LOG_FILE', 'train_style_chords.log')).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.Logger('train_style_chords', level=logging.INFO)
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


def _train(logger):
    # Set up CUDA settings
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)

    #==========================================================================

    ''' DEVICE '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trace = TrainingTrace(torch, device)
    trace.mark('startup', resources=True)
    logger.info('Environment: device=%s torch=%s CUDA=%s', device, torch.__version__, torch.version.cuda)

    # The change stage starts from the chord stage's best checkpoint, so it runs second.
    stages = [os.environ['MODEL_NAME']] if 'MODEL_NAME' in os.environ else STAGES
    for model_name in stages:
        if not _train_stage(model_name, device, trace, logger):
            break  # Interrupted: never start the change stage from an unfinished chord stage.


def _train_stage(model_name, device, trace, logger) -> bool:
    """Train one adapter until its schedule ends. Returns False when interrupted."""
    device_type = device.type

    #==========================================================================

    ''' MODEL & HYPERPARAMETERS '''
    project_name = 'monsterGenie_chords'
    cfg = get_model_hparams(model_name)
    if cfg['model_type'] != 'AE_style_jokerParam_chords':
        raise ValueError('train_style_chords.py requires an AE_style_jokerParam_chords preset')
    cfg['batch_size'] = int(os.environ.get('BATCH_SIZE', cfg['batch_size']))
    cfg['num_workers'] = int(os.environ.get('NUM_WORKERS', cfg['num_workers']))
    max_steps = int(os.environ.get('MAX_STEPS', '0'))  # Additional updates per stage; 0 runs to chord_total_steps.
    if max_steps < 0 or cfg['batch_size'] < 1 or cfg['num_workers'] < 0:
        raise ValueError('Invalid batch size, worker count or MAX_STEPS')
    cfg['chord_training_seed'] = int(os.environ.get('SEED', cfg.get('chord_training_seed', 42)))
    torch.manual_seed(cfg['chord_training_seed'])
    logger.info('Stage %s configuration: %s', model_name, json.dumps(cfg, sort_keys=True))
    pin_memory = device_type == 'cuda' and os.environ.get('PIN_MEMORY', '1') != '0'
    persistent_workers = cfg['num_workers'] > 0 and os.environ.get('PERSISTENT_WORKERS', '1') != '0'
    model = load_model(model_name=model_name, cfg=cfg, set_only=True)
    model.to(device)

    #==========================================================================

    ''' DATA '''

    style_seq_len = cfg.get('style_seq_len', 256)

    train_data = Any_Pickle_File_Reader(cfg['dataset_train_path'])
    # Keep compact CPU arrays; the sampler returns tensors for the selected windows.
    data_train = np.asarray(train_data, dtype=np.int16)
    del train_data
    print("Building training dataset...")
    train_dataset = StyleChordSamplerDataset(
        data_train, cfg['seq_len'], style_seq_len, transpose_max=cfg['data_augment_transpose_max'],
        whole_span_prob=cfg['chord_whole_span_prob'], validation_seed=cfg['chord_validation_seed'],
        max_samples=cfg.get('chord_max_samples', 0),
        onset_prob=cfg.get('chord_onset_prob', 0.0), onset_weight=cfg.get('chord_onset_weight', 0.0),
        onset_tau=cfg.get('chord_onset_tau', 3.0), change_control=cfg.get('chord_change_enabled', False),
        change_min_notes=cfg.get('chord_change_min_notes', 4),
        change_min_new_tones=cfg.get('chord_change_min_new_tones', 2))
    del data_train
    print(f"BATCH_SIZE: {cfg['batch_size']}")
    print(f"NUM_WORKERS: {cfg['num_workers']}")
    print(f"Dataset size: {len(train_dataset)}")
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
        shuffle=True, pin_memory=pin_memory, persistent_workers=persistent_workers,
        collate_fn=collate_chord_windows)
    print(f"Number of batches: {len(train_loader)}")
    logger.info('Training dataset: samples=%d batches=%d batch_size=%d num_workers=%d',
                len(train_dataset), len(train_loader), cfg['batch_size'], cfg['num_workers'])

    eval_data = Any_Pickle_File_Reader(cfg['dataset_val_path'])
    data_eval = np.asarray(eval_data, dtype=np.int16)
    del eval_data
    print("Building validation dataset...")
    val_dataset = StyleChordSamplerDataset(
        data_eval, cfg['seq_len'], style_seq_len, is_eval=True,
        whole_span_prob=cfg['chord_whole_span_prob'], validation_seed=cfg['chord_validation_seed'],
        max_samples=cfg.get('chord_max_val_samples', 0), change_control=cfg.get('chord_change_enabled', False),
        change_min_notes=cfg.get('chord_change_min_notes', 4),
        change_min_new_tones=cfg.get('chord_change_min_new_tones', 2))
    del data_eval
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], num_workers=0,
                            shuffle=False, pin_memory=pin_memory, collate_fn=collate_chord_windows)
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
    # Only this stage's adapter is trainable; the style joker model stays frozen.
    optim = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg['learning_rate'])
    # BF16 has FP32's exponent range and does not need gradient scaling.
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
        print(f"Resuming training from epoch {start_epoch}, step {nsteps}")
        logger.info('Resuming from %s: epoch=%d step=%d', checkpoint_path, start_epoch, nsteps)
    else:
        warm_start_chord(model, cfg['init_from_ckpt'])
        print(f"Warm-started {model_name} from {cfg['init_from_ckpt']}")
        logger.info('Warm-started %s from %s', model_name, cfg['init_from_ckpt'])

    # Training ends where the cosine schedule ends; MAX_STEPS can stop it earlier.
    stop_step = min(cfg['chord_total_steps'], nsteps + max_steps) if max_steps else cfg['chord_total_steps']
    if nsteps >= stop_step:
        # A rerun after this stage finished goes straight to the next stage.
        print(f'{model_name} already finished at step {nsteps}')
        logger.info('Stage already finished: %s step=%d', model_name, nsteps)
        return True

    #==========================================================================

    ''' WANDB '''
    wandb_run = None
    if cfg['use_logs']:
        import wandb
        wandb.login()
        wandb_run = wandb.init(project=project_name, name=model_name, config=cfg)
        wandb_run.define_metric('epoch')
        wandb_run.define_metric('validation/*', step_metric='epoch')
        for mode in ('guided', 'joker'):
            for metric, summary in (
                    ('nll_true', 'min'),
                    ('nll_gain_vs_free', 'max'),
                    ('nll_gap_shifted', 'max')):
                wandb_run.define_metric(f'validation/{mode}_{metric}', step_metric='epoch', summary=summary)

    ''' TRAINING '''
    val_loss_temp = 0.0
    validation: Dict[str, float] = {}
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
                    batch.pop('history_kind')  # Always full history in these presets.
                    trace.mark('batch_received_transfer', ep, i)
                    optim.zero_grad()
                    # The rate follows the step count, so a resumed run continues the same schedule.
                    for group in optim.param_groups:
                        group['lr'] = scheduled_lr(cfg, nsteps)
                    x = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                    # Chords are trained under guided, mixed and all-joker buttons, as in train_style.py.
                    x['joker_mask'] = make_joker_training_mask(
                        x['pitch'].shape[0], x['pitch'].shape[1] - 1, device
                    )
                    trace.mark('forward', ep, i)
                    with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=device_type == 'cuda'):
                        loss, acc = model(x)
                    trace.mark('backward', ep, i)
                    scaler.scale(loss['loss_total']).backward()
                    trace.mark('logging', ep, i)
                    if (i % cfg['print_stats_every'] == 0) or TESTING:
                        metrics = {name: value.item() for name, value in loss.items()}
                        metrics.update(epoch=ep, batch=i, step=nsteps, train_acc=acc.item(),
                                       grad_scale=scaler.get_scale(), lr=optim.param_groups[0]['lr'])
                        logger.info('Training metrics: %s', json.dumps(metrics, sort_keys=True))
                        if cfg['use_logs']:
                            wandb.log({**{name: value.item() for name, value in loss.items()},
                                       'train_acc': acc.item(), 'lr': metrics['lr']}, step=nsteps)
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
                                # No joker_mask in eval mode: guided buttons everywhere.
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
                    if nsteps >= stop_step:
                        break

            trace.mark('epoch_end', ep, resources=True)
            logger.info('Epoch ended: epoch=%d step=%d', ep, nsteps)
            # Full validation (three passes per window and button mode) runs every few epochs
            # and at the end of training. The best checkpoint has the lowest mean nll_true.
            validated = ep % cfg['chord_validate_every_epochs'] == 0 or nsteps >= stop_step
            improved = False
            if validated:
                validation = validate_style_chords(model, val_loader, device)
                val_loss_temp = (validation['guided_nll_true'] + validation['joker_nll_true']) / 2
                improved = val_loss_temp < best_val_nll
                best_val_nll = min(best_val_nll, val_loss_temp)
                logger.info('Chord validation: %s', json.dumps(validation, sort_keys=True))
                print('Chord validation:', validation)
                if cfg['use_logs']:
                    wandb.log({'epoch': ep, **{f'validation/{key}': value for key, value in validation.items()}},
                              step=nsteps)

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
            if nsteps >= stop_step:
                break
    except KeyboardInterrupt:
        save_checkpoint(cfg['ckpt_file_name'], checkpoint())
        print(f"Interrupted; saved step {nsteps} to {cfg['ckpt_file_name']}")
        logger.info('Interrupted: saved step=%d', nsteps)
        return False
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    logger.info('Stage finished: %s step=%d', model_name, nsteps)
    return True


if __name__ == '__main__':
    mp.freeze_support()
    main()
