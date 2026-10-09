#===================================================================================================
# Monster Genie train_selection.py Python module
# Training with GIANTsel dataset
# 
# Copyright 2025 Alex Barrachina
#
# Based on Project Los Angeles / Tegridy Code 2025
# https://github.com/asigalov61/monsterpianotransformer
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

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import tqdm
import glob
import re
import copy
import hashlib
from typing import Dict, Optional, Tuple

os.environ['USE_FLASH_ATTENTION'] = '1'

from random import randint, random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from midiUtils import Any_Pickle_File_Reader
from model_loader import load_model, warm_start_tension
from models import get_model_hparams
from params import *
from x_transformer import *

NSTEPS_INIT = 256
RESUME = os.environ.get('RESUME', '1') != '0'

#==========================================================================

def find_latest_checkpoint(checkpoint_dir: str = './save_models',
                           model_name: Optional[str] = None) -> tuple[Optional[str], int, int]:
    """
    Find the latest checkpoint in the save_models directory.
    
    Returns:
        tuple: (checkpoint_path, epoch, steps) or (None, 0, 0) if no checkpoint found
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return None, 0, 0
    
    # Pattern to match checkpoint files: MODEL_NAME_epoch_eps_steps_steps_loss_loss_acc_acc.pth
    pattern = os.path.join(checkpoint_dir, f"{model_name}_*.pth" if model_name else "*.pth")
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


#==========================================================================

class PitchSamplerDataset(Dataset):
    """Pitch-only windows within pieces, retaining the stored note order."""

    def __init__(self, data: torch.Tensor, seq_len: int, is_eval: bool = False,
                 transpose_max: int = 6) -> None:
        super().__init__()
        if seq_len < 1 or data.numel() % 5 != 0:
            raise ValueError('Expected seq_len >= 1 and flat five-token events')
        self.seq_len = seq_len
        self.is_eval = is_eval
        self.transpose_max = transpose_max
        events = data.reshape(-1, 5).long()
        boundary = (events == events.new_tensor([126, 126, 0, 0, 0])).all(dim=1)
        # Harmony annotations and boundary markers are not played pitches.
        notes = (~boundary & (events[:, 4] >= 0) & (events[:, 4] < 16)
                 & (events[:, 4] != HARMONY_CHANNEL)
                 & (events[:, 4] != CHORDS_CHANNEL)
                 & (events[:, 3] > 0))
        self.pitches = events[notes, 2].clone()
        if ((self.pitches < 0) | (self.pitches >= VOCAB_SIZE_PITCH)).any():
            raise ValueError('Pitch tokens must be MIDI pitches in [0, 127]')
        note_counts = notes.long().cumsum(0)
        edges = torch.cat((note_counts.new_zeros(1), note_counts[boundary],
                           note_counts.new_full((1,), self.pitches.numel())))
        self.piece_edges = edges
        self._set_windows(edges[:-1], edges[1:], (edges[1:] - edges[:-1]) > seq_len)

    def _set_windows(self, starts: torch.Tensor, ends: torch.Tensor, keep: torch.Tensor) -> None:
        lengths = ends - starts
        valid = keep & (lengths > self.seq_len)
        self.piece_ids = torch.nonzero(valid).flatten()
        self.starts = starts[valid]
        self.window_counts = (lengths[valid] - self.seq_len).cumsum(0)
        self.sample_count = int((lengths[valid] // (self.seq_len + 1)).sum())
        if self.sample_count == 0:
            raise ValueError('No piece has enough notes for a pitch training window')
        self.total_windows = int(self.window_counts[-1])

    def __len__(self) -> int:
        return self.sample_count

    def _sample_window(self, index: int) -> Tuple[int, int, torch.Tensor, int]:
        # Validation covers fixed windows and performs no augmentation.
        window = (index * self.total_windows // self.sample_count if self.is_eval
                  else int(torch.randint(self.total_windows, (1,))))
        piece = int(torch.searchsorted(self.window_counts, window, right=True))
        preceding = int(self.window_counts[piece - 1]) if piece > 0 else 0
        start = int(self.starts[piece]) + window - preceding
        pitch = self.pitches[start:start + self.seq_len + 1].clone()
        shift = 0
        if not self.is_eval and self.transpose_max > 0:
            # Bound the shift instead of clipping notes and changing intervals.
            low = max(-self.transpose_max, -int(pitch.min()))
            high = min(self.transpose_max, VOCAB_SIZE_PITCH - 1 - int(pitch.max()))
            shift = int(torch.randint(low, high + 1, (1,)))
            pitch += shift
        return int(self.piece_ids[piece]), start, pitch, shift

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        _, _, pitch, _ = self._sample_window(index)
        return {'pitch': pitch}


class TensionPitchSamplerDataset(PitchSamplerDataset):
    """Aligned corpus controls; validation shares immutable arrays with training."""

    def __init__(self, data: torch.Tensor, seq_len: int, is_eval: bool = False,
                 transpose_max: int = 6, cfg: Optional[dict] = None,
                 split: str = 'train') -> None:
        super().__init__(data, seq_len, is_eval, transpose_max)
        self.tens_cfg = dict(cfg or {})
        self.validation_seed = int(self.tens_cfg.get('tens_validation_seed', 1729))
        self.p_all_cond = float(self.tens_cfg.get('tens_p_all_cond', 0.2))
        self.home_drop = float(self.tens_cfg.get('home_key_drop_prob', 0.1))
        fraction = float(self.tens_cfg.get('tens_validation_fraction', 0.05))
        if not 0 < fraction < 1 or not 0 <= self.p_all_cond <= 1 or not 0 <= self.home_drop <= 1:
            raise ValueError('Invalid tension split or conditioning probability')
        events = data.reshape(-1, 5).long()
        boundary = (events == events.new_tensor([126, 126, 0, 0, 0])).all(dim=1)
        piece = boundary.long().cumsum(0)
        notes = (~boundary & (events[:, 4] >= 0) & (events[:, 4] < 16)
                 & (events[:, 4] != HARMONY_CHANNEL) & (events[:, 4] != CHORDS_CHANNEL)
                 & (events[:, 3] > 0))
        tension = events[:, 4] == TENSION_CHANNEL
        home = events[:, 4] == HOME_KEY_CHANNEL
        if ((events[tension, 1] < 0) | (events[tension, 1] >= NUM_TENSION_LEVELS)).any():
            raise ValueError('Tension labels must be in [0, 4]')
        if ((events[home, 1] < 0) | (events[home, 1] > 11)
                | (events[home, 2] < 0) | (events[home, 2] > 1)).any():
            raise ValueError('Home keys require pc 0..11 and mode 0..1')
        count = self.piece_edges.numel() - 1
        if (torch.bincount(piece[home], minlength=count) > 1).any():
            raise ValueError('Expected at most one home-key annotation per piece')
        self.home_keys = torch.full((count,), HOME_KEY_UNKNOWN, dtype=torch.long)
        self.home_keys[piece[home]] = events[home, 1] + 12 * events[home, 2]
        indices = torch.arange(events.size(0))
        last = torch.where(tension, indices, -1).cummax(0).values[notes]
        note_piece = piece[notes]
        known = (last >= 0) & (piece[last.clamp_min(0)] == note_piece)
        self.levels = torch.where(known, events[last.clamp_min(0), 1] + 1, TENSION_NULL)

        # Exclude windows ending before the first known target, without crossing pieces.
        first_known = torch.full((count,), self.pitches.numel(), dtype=torch.long)
        first_known.scatter_reduce_(0, note_piece[known], torch.arange(self.pitches.numel())[known],
                                    reduce='amin', include_self=True)
        self.tens_starts = torch.maximum(self.piece_edges[:-1], first_known - seq_len)
        self.validation_pieces = torch.zeros(count, dtype=torch.bool)
        seed = str(int(self.tens_cfg.get('tens_split_seed', 42))).encode()
        # Piece hashes keep exact duplicate performances together and survive reordered builds.
        for index in range(count):
            start, end = int(self.piece_edges[index]), int(self.piece_edges[index + 1])
            if end <= start:
                continue
            content = self.pitches[start:end].numpy().astype('<i2').tobytes()
            digest = hashlib.sha256(seed + content).digest()
            self.validation_pieces[index] = int.from_bytes(digest[:8], 'big') / 2**64 < fraction
        self.select_split(split)

    def select_split(self, split: str) -> None:
        if split not in ('train', 'validation', 'all'):
            raise ValueError('split must be train, validation or all')
        keep = (self.validation_pieces if split == 'validation' else ~self.validation_pieces)
        if split == 'all':
            keep = torch.ones_like(keep)
        self._set_windows(self.tens_starts, self.piece_edges[1:], keep)
        limit = int(self.tens_cfg.get('tens_max_val_samples' if self.is_eval else 'tens_max_samples', 0))
        if limit > 0:
            self.sample_count = min(self.sample_count, limit)

    def validation_dataset(self) -> 'TensionPitchSamplerDataset':
        dataset = copy.copy(self)
        dataset.is_eval = True
        dataset.select_split('validation')
        return dataset

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        piece, start, pitch, shift = self._sample_window(index)
        generator = torch.Generator().manual_seed(self.validation_seed + index) if self.is_eval else None
        target = self.levels[start:start + self.seq_len + 1].clone()
        cut = 1
        if self.seq_len > 1 and float(torch.rand((), generator=generator)) >= self.p_all_cond:
            cut = int(torch.randint(2, self.seq_len + 1, (), generator=generator))
        target[:cut] = TENSION_NULL
        home = int(self.home_keys[piece])
        if home != HOME_KEY_UNKNOWN:
            home = (home % 12 + shift) % 12 + 12 * (home // 12)
        if not self.is_eval and float(torch.rand(())) < self.home_drop:
            home = HOME_KEY_UNKNOWN
        return {'pitch': pitch, 'tension_target': target, 'home_key': torch.tensor(home)}


def file_fingerprint(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def validate_tension(model: D_Tension_Joker, loader: DataLoader,
                     device: torch.device) -> Dict[str, float]:
    """Fixed windows, identical masks for true, NULL, wrong and unknown-home controls."""
    model.eval()
    sums = torch.zeros(5, device=device, dtype=torch.float64)
    for batch in loader:
        tokens = {key: value.to(device) for key, value in batch.items()}
        target = tokens['tension_target']
        mask = (target[:, 1:] != TENSION_NULL) & (tokens['pitch'][:, 1:] != model.ignore_index)
        wrong = torch.where(target > 0, target % NUM_TENSION_LEVELS + 1, TENSION_NULL)
        variants = (tokens, {'pitch': tokens['pitch']},
                    {**tokens, 'tension_target': wrong},
                    {**tokens, 'home_key': torch.full_like(tokens['home_key'], HOME_KEY_UNKNOWN)})
        for index, variant in enumerate(variants):
            logits = model.pitch_logits(variant)
            ce = F.cross_entropy(logits.transpose(1, 2), tokens['pitch'][:, 1:],
                                 ignore_index=model.ignore_index, reduction='none')
            sums[index] += (ce * mask).sum().double()
        sums[4] += mask.sum()
    totals = sums.cpu().tolist()
    if totals[4] == 0:
        raise ValueError('Validation contains no labelled tension targets')
    return {name: totals[index] / totals[4] for index, name in enumerate(
        ('nll_true', 'nll_null', 'nll_wrong', 'nll_unknown_home'))}


class MusicSamplerDataset(Dataset):
    def __init__(self, data, seq_len, is_eval=False, cfg=None):
        super().__init__()

        self.data = data
        self.seq_len = seq_len
        self.tokens_per_note = 5  # dtime, dur, chan, pitch, vel
        self.seq_tot_tokens = self.seq_len * self.tokens_per_note + self.tokens_per_note  # 5 tokens per note + 5 for the current note
        self.cfg = cfg if cfg is not None else {}


    def __len__(self):
        return int(self.data.size(0) / self.seq_tot_tokens)  #  self.seq_len if you want exact training time per epoch

    def __getitem__(self, index): # TODO concatenates all data, end of files with begining of files
        seq_tot_tokens = self.seq_tot_tokens
        # We only pick starting positions that are multiples of seq_tot_tokens, // seq_tot_tokens: Integer division to get how many complete sequences of size seq_tot_tokens we can fit
        # We don't exceed the data boundaries
        #rand = secrets.randbelow((self.data.size(0)-seq_tot_tokens) // seq_tot_tokens) * seq_tot_tokens
        rand = randint(0, (self.data.size(0)-seq_tot_tokens) // seq_tot_tokens) * seq_tot_tokens

        # Extract sequences for each feature, +1 to include the current token
        x = self.data[rand: rand + seq_tot_tokens] # we take an extra token

        # Convert to tensors
        # Pickle format: [dtime, dur, pitch, vel, chan] (5 tokens per note, no offsets)
        dtimes = x[0::5].long()  # Every 5th token starting at index 0
        durs = x[1::5].long()  # Every 5th token starting at index 1
        pitches = x[2::5].long()  # Every 5th token starting at index 2
        vels = x[3::5].long()  # Every 5th token starting at index 3
        channels = x[4::5].long()  # Every 5th token starting at index 4

        # Data augmentation
        # Time stretching
        stretch_factor = random() * self.cfg['data_augment_time_stretch_max'] * 2
        stretch_factor += 1 - self.cfg['data_augment_time_stretch_max']
        dtimes = (dtimes.float() * stretch_factor).long()
        dtimes = torch.clamp(dtimes, min=0, max=RANGE_DTIME_SHIFT)
  
        stretch_factor = random() * self.cfg['data_augment_time_stretch_max'] * 2
        stretch_factor += 1 - self.cfg['data_augment_time_stretch_max']
        durs = (durs.float() * stretch_factor).long()
        durs = torch.clamp(durs, min=0, max=RANGE_DUR_SHIFT)

        # Chord micro-alterations
        # Convert to absolute times for easier chord detection
        abs_times = torch.cumsum(dtimes, dim=0)
              
        # Find chord groups
        chord_groups = []
        current_chord = [0]  # Start with first note
        
        for i in range(1, len(abs_times)):
            if abs_times[i] - abs_times[i-1] <= self.cfg['data_augment_chord_threshold']:
                current_chord.append(i)
            else:
                if len(current_chord) > 1:  # Only process if it's actually a chord
                    chord_groups.append(current_chord)
                current_chord = [i]
        
        if len(current_chord) > 1:
            chord_groups.append(current_chord)
        
        # Apply micro-alterations to chord notes
        for chord in chord_groups:
            # Generate small random shifts for each note in the chord
            shifts = torch.randint(-1, 2, (len(chord),))  # Random shifts of -1, 0, or 1
            abs_times[chord] = abs_times[chord] + shifts
        
        # Re-sort the sequence based on new absolute times
        sorted_indices = torch.argsort(abs_times)
        abs_times = abs_times[sorted_indices]
        durs = durs[sorted_indices]
        pitches = pitches[sorted_indices]
        channels = channels[sorted_indices]
        
        # Convert back to delta times
        dtimes = torch.cat([abs_times[0:1], abs_times[1:] - abs_times[:-1]])
        dtimes = torch.clamp(dtimes, min=0, max=RANGE_DTIME_SHIFT)

        # Transposition
        transposition_factor = randint(
            -self.cfg['data_augment_transpose_max'], self.cfg['data_augment_transpose_max']
        )
        # Apply transposition and ensure pitches stay within valid range (0-127)
        # TODO: Clamp isn't a good idea as we alter the interval relationships. But we hope transposing +-6 we don't clamp
        pitches = torch.clamp(pitches + transposition_factor, min=0, max=VOCAB_SIZE_PITCH-1)

        feature_data = {
                'dtime': dtimes,
                'dur': durs,
                'channel': channels,
                'pitch': pitches
                #'vel': vels
            }
        return feature_data

def main():
    # Set up CUDA settings
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
        torch.backends.cuda.enable_cudnn_sdp(False)

    #==========================================================================

    ''' DEVICE '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type='cuda' if torch.cuda.is_available() else 'cpu'

    #==========================================================================

    ''' MODEL & HYPERPARAMETERS '''
    project_name = 'harmony'
    model_name = os.environ.get('MODEL_NAME', 'D_Tension_Joker_little_v1')
    cfg = get_model_hparams(model_name)
    is_base_joker = cfg['model_type'] == 'D_Base_Joker'
    is_tension_joker = cfg['model_type'] == 'D_Tension_Joker'
    if is_tension_joker:
        cfg['init_from_ckpt'] = os.environ.get('INIT_FROM_CKPT', cfg['init_from_ckpt'])
    model = load_model(model_name=model_name, cfg=cfg, set_only=True)  
    model.to(device)
    checkpoint_path, start_epoch, start_steps = (
        find_latest_checkpoint(model_name=model_name) if RESUME else (None, 0, 0))
    if is_tension_joker and checkpoint_path is None:
        warm_start_tension(model, cfg['init_from_ckpt'])
    #print(model)
    
    #==========================================================================

    ''' WANDB '''
    if(cfg['use_logs']):
        import wandb
        wandb.login()
        wandb.init(project=project_name, name=model_name, config=cfg)


    #==========================================================================

    ''' DATA '''

    """ LOAD TRAINING DATA """

    # Loading dataset from a pickle in ./Training-Data
    train_data = Any_Pickle_File_Reader(cfg['dataset_train_path'])   
    data_train = torch.tensor(train_data, dtype=torch.int16) if is_tension_joker else torch.Tensor(train_data)
    del train_data
    if not is_tension_joker:
        eval_data = Any_Pickle_File_Reader(cfg['dataset_val_path'])
        data_eval = torch.Tensor(eval_data)
        del eval_data

    # Dataloader
    if is_tension_joker:
        train_dataset = TensionPitchSamplerDataset(
            data_train, cfg['seq_len'], transpose_max=cfg['data_augment_transpose_max'], cfg=cfg)
        val_dataset = train_dataset.validation_dataset()
        data_path = cfg['dataset_train_path']
        if not data_path.endswith('.pickle'):
            data_path += '.pickle'
        cfg['dataset_sha256'] = file_fingerprint(data_path)
        cfg['train_piece_ids'] = train_dataset.piece_ids.tolist()
        cfg['validation_piece_ids'] = val_dataset.piece_ids.tolist()
        if checkpoint_path is None:
            cfg['base_checkpoint_sha256'] = file_fingerprint(cfg['init_from_ckpt'])
        del data_train
    elif is_base_joker:
        train_dataset = PitchSamplerDataset(
            data_train, cfg['seq_len'], transpose_max=cfg['data_augment_transpose_max'])
    else:
        train_dataset = MusicSamplerDataset(data_train, cfg['seq_len'], cfg=cfg) # train in chunks of SEQ_LEN
    print(f"BATCH_SIZE: {cfg['batch_size']}")
    print(f"Dataset size: {len(train_dataset)}")
    train_loader  = DataLoader(train_dataset, batch_size = cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True)
    print(f"Number of batches: {len(train_loader)}")
    if is_tension_joker:
        pass  # Shares the immutable pitch/annotation arrays above; never reads the test pickle.
    elif is_base_joker:
        val_dataset = PitchSamplerDataset(data_eval, cfg['seq_len'], is_eval=True)
    else:
        val_dataset = MusicSamplerDataset(data_eval, cfg['seq_len'], is_eval=True, cfg=cfg) # train in chunks of SEQ_LEN
    val_loader  = DataLoader(val_dataset, batch_size = cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=False)

    # Right after val_loader is created and before model definition, add a reusable iterator for streaming validation
    val_iter = iter(val_loader)  # will be cycled through inside training loop

    #==========================================================================

    ''' PRECISION/OPTIMIZER/SCALER '''

    dtype = torch.bfloat16

    #ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)

    optim = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg['learning_rate'])

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16 and device_type == 'cuda'))

    ''' LOAD CHECKPOINT '''
    nsteps = 0
    best_val = float('inf')
    if checkpoint_path:
        if is_tension_joker:
            saved = torch.load(checkpoint_path, map_location='cpu')
            old_cfg = saved.get('cfg', {})
            for key in ('dataset_sha256', 'tens_split_seed', 'tens_validation_fraction',
                        'tens_validation_seed', 'tens_max_val_samples', 'tens_p_all_cond',
                        'seq_len', 'train_piece_ids', 'validation_piece_ids'):
                if old_cfg.get(key) != cfg.get(key):
                    raise ValueError(f'Tension resume would change {key}; start a new run explicitly')
            cfg['base_checkpoint_sha256'] = old_cfg.get('base_checkpoint_sha256', '')
            cfg['init_from_ckpt'] = old_cfg.get('init_from_ckpt', cfg['init_from_ckpt'])
            best_val = float(saved.get('best_val_nll', float('inf')))
            del saved
        start_epoch, start_steps = load_checkpoint(model, optim, checkpoint_path, device)
        print(f"Resuming training from epoch {start_epoch}, step {start_steps}")
        nsteps = start_steps
        start_epoch += 1
    else:
        start_epoch = 0
        print('Starting adapter training from the base checkpoint' if is_tension_joker
              else 'Starting training from scratch (no checkpoint found)')


    ''' TRAINING '''
    for ep in range(start_epoch, cfg['epochs']):
        print('Epoch #', ep)
        
        model.train()
        with tqdm.tqdm(total=len(train_loader)) as bar_train:
            for i, batch in enumerate(train_loader):            
                optim.zero_grad()

                # move to device
                x = {key: value.to(device) for key, value in batch.items()}

                with torch.amp.autocast(device_type=device_type, dtype=dtype):
                    loss, acc = model(x)  # Update your model to accept target separately
                scaler.scale(loss['loss_total']).backward()
                
                if (i % cfg['print_stats_every'] == 0) or TESTING:
                    if( cfg['use_logs']):                
                        wandb.log({"loss_total": loss['loss_total'].item()}, step=nsteps)
                        wandb.log({"train_acc": acc.item()}, step=nsteps)
                        if is_tension_joker:
                            wandb.log({'loss_film_reg': loss['loss_film_reg'].item()}, step=nsteps)
                        if cfg['loss_norm_pos']>0 and 'loss_norm_pos' in loss:
                            wandb.log({"loss_norm_pos": cfg['loss_norm_pos']*loss['loss_norm_pos'].item()}, step=nsteps)
                        if cfg['loss_deviate']>0 and 'loss_deviate' in loss:
                            wandb.log({"loss_deviate": cfg['loss_deviate']*loss['loss_deviate'].item()}, step=nsteps)
                        if cfg['loss_margin']>0 and 'loss_margin' in loss:
                            wandb.log({"loss_margin": cfg['loss_margin']*loss['loss_margin'].item()}, step=nsteps)
                        if cfg['loss_pitch_button']>0 and 'loss_pitch_button' in loss:
                            wandb.log({"loss_pitch_button": cfg['loss_pitch_button']*loss['loss_pitch_button'].item()}, step=nsteps)
                        if cfg['loss_button_concentration']>0 and 'loss_button_concentration' in loss:
                            wandb.log({"loss_button_concentration": cfg['loss_button_concentration']*loss['loss_button_concentration'].item()}, step=nsteps)
                        if cfg['loss_window_corr']>0 and 'loss_window_corr' in loss:
                            wandb.log({"loss_window_corr": cfg['loss_window_corr']*loss['loss_window_corr'].item()}, step=nsteps)
                        if cfg.get('loss_latent_velocity', 0)>0 and 'loss_latent_velocity' in loss:
                            wandb.log({"loss_latent_velocity": cfg['loss_latent_velocity']*loss['loss_latent_velocity'].item()}, step=nsteps)
                        if cfg.get('loss_drift', 0)>0 and 'loss_drift' in loss:
                            wandb.log({"loss_drift": cfg['loss_drift']*loss['loss_drift'].item()}, step=nsteps)

                        if cfg['loss_contour']>0 and 'loss_contour' in loss:
                            wandb.log({"loss_contour_all": cfg['loss_contour']*loss['loss_contour'].item()}, step=nsteps)
                        
                            if cfg['loss_contour_perc']>0 and 'loss_contour_perc' in loss:
                                wandb.log({"loss_contour_perc": cfg['loss_contour']*cfg['loss_contour_perc']*loss['loss_contour_perc'].item()}, step=nsteps)
                            if cfg['loss_multi_step_perc']>0 and 'loss_multi_step_perc' in loss:
                                wandb.log({"loss_multi_step": cfg['loss_contour']*cfg['loss_multi_step_perc']*loss['loss_multi_step_perc'].item()}, step=nsteps)
                            if cfg['loss_interval_perc']>0 and 'loss_interval' in loss:
                                wandb.log({"loss_interval": cfg['loss_contour']*cfg['loss_interval_perc']*loss['loss_interval_perc'].item()}, step=nsteps)
                            if cfg['loss_shape_perc']>0 and 'loss_shape_perc' in loss:
                                wandb.log({"loss_shape": cfg['loss_contour']*cfg['loss_shape_perc']*loss['loss_shape_perc'].item()}, step=nsteps)
                        
                        if cfg['loss_button_held']>0 and 'loss_button_held' in loss: 
                            wandb.log({"loss_button_held": cfg['loss_button_held']*loss['loss_button_held'].item()}, step=nsteps)
                        if cfg['loss_recons']>0 and 'loss_recons' in loss: 
                            wandb.log({"loss_recons": cfg['loss_recons']*loss['loss_recons'].item()}, step=nsteps)
                        if cfg.get('loss_arrow_consistency', 0)>0 and 'loss_arrow_consistency' in loss: 
                            wandb.log({"loss_arrow_consistency": cfg['loss_arrow_consistency']*loss['loss_arrow_consistency'].item()}, step=nsteps)
                        if cfg.get('loss_coarse_direction', 0)>0 and 'loss_coarse_direction' in loss: 
                            wandb.log({"loss_coarse_direction": cfg['loss_coarse_direction']*loss['loss_coarse_direction'].item()}, step=nsteps)
                        


                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                scaler.step(optim)
                scaler.update()
                nsteps += 1


                bar_train.set_description(f'Epoch: {ep} Loss: {float(loss["loss_total"]):.4}')# LR: {float(lr):.8}')
                bar_train.update(1)

                if (i % cfg['validate_every'] == 0) or TESTING:
                    try:
                        val_batch = next(val_iter) # extract batches from test dataloader
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_batch = next(val_iter) # extract batches from test dataloader           
                    model.eval()
                    with torch.no_grad():
                        with torch.amp.autocast(device_type=device_type, dtype=dtype):
                            # move to device
                            vx = {key: value.to(device) for key, value in val_batch.items()}
                            # run the model
                            val_loss, val_acc = model(vx)  # Update your model to accept target separately
                            val_loss_temp = val_loss['loss_total'].item()

                        if(cfg['use_logs']):                
                            wandb.log({"val_loss": val_loss['loss_total'].item()}, step=nsteps)
                            wandb.log({"val_acc": val_acc.item()}, step=nsteps)
                    model.train()
                    del val_batch, vx
                    torch.cuda.empty_cache()

        
        improved = False
        if is_tension_joker:
            diagnostics = validate_tension(model, val_loader, device)
            val_loss_temp = diagnostics['nll_true']
            improved = val_loss_temp < best_val
            best_val = min(best_val, val_loss_temp)
            print('Tension validation:', diagnostics)
            if cfg['use_logs']:
                wandb.log({f'validation/{key}': value for key, value in diagnostics.items()}, step=nsteps)

        if ep % cfg['save_every'] == 0:
            fname = './save_models/' + cfg['model_name'] + '_' + str(ep) + '_eps_' + str(nsteps) + '_steps_' + str(round(float(loss['loss_total'].item()), 4)) + '_loss_' + str(round(float(val_loss_temp), 4)) + '_val_loss_' + str(round(float(acc.item()), 4)) + '_acc.pth'
            if is_base_joker or is_tension_joker:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'epoch': ep,
                    'steps': nsteps,
                    'cfg': cfg,
                }
                if is_tension_joker:
                    checkpoint['best_val_nll'] = best_val
                    checkpoint['validation'] = diagnostics
                torch.save(checkpoint, fname)
                torch.save(checkpoint, cfg['ckpt_file_name'])
                if is_tension_joker and improved:
                    root, extension = os.path.splitext(cfg['ckpt_file_name'])
                    torch.save(checkpoint, root + '_best' + extension)
            else:
                torch.save(model.state_dict(), fname)


if __name__ == '__main__':
    mp.freeze_support()
    main()
