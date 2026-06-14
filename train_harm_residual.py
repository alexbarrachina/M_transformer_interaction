#===================================================================================================
# Monster Genie train_harm_residual.py Python module
# Training the chord-conditioned residual adapter on top of a frozen base autoencoder.
# The base model (AutoregressiveAutoencoder_no_dtime) is loaded from a checkpoint and frozen.
# Only the chord encoder + adapter MLP parameters are trained.
# Pickle format: flat [dtime, dur, pitch, vel, chan] with chord events as [0, dur, pitch, vel, 4]
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

NSTEPS_INIT = 256
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


#==========================================================================

class MusicSamplerDataset(Dataset):
    """
    Dataset for chord-conditioned residual adapter training.
    
    Reads flat pickle format: [dtime, dur, pitch, vel, chan] (5 tokens per event).
    Chord events: [0, dur, pitch, vel, 4] (chan=CHORDS_CHANNEL).
    Chord-off markers: [0, 0, 0, 0, 4] (vel=0 distinguishes from real chords).
    Note events have chan in {0, 10, ...}.
    
    For each note, builds:
    - chord_pcs: 12-dim pitch-class multi-hot of the currently active chord
    - bass_pc:   bass pitch class (lowest note of the active chord, 0-11)
    
    A chord is active from its onset until its notes expire (chord-off marker).
    After chord-off, notes get zero multi-hot until the next chord appears.
    """
    def __init__(self, data, seq_len, is_eval=False, cfg=None):
        super().__init__()

        self.data = data
        self.seq_len = seq_len
        self.tokens_per_note = 5  # dtime, dur, pitch, vel, chan
        self.cfg = cfg if cfg is not None else {}

        # Oversample by 2x to account for chord/harmony events mixed in
        self.seq_tot_tokens = self.seq_len * self.tokens_per_note * 2

    def __len__(self):
        return int(self.data.size(0) / (self.seq_len * self.tokens_per_note + self.tokens_per_note))

    def __getitem__(self, index):
        seq_tot_tokens = self.seq_tot_tokens
        # Pick a random starting position aligned to 5-token boundaries
        max_start = (self.data.size(0) - seq_tot_tokens) // self.tokens_per_note
        if max_start <= 0:
            max_start = 1
        rand = randint(0, max_start) * self.tokens_per_note

        x = self.data[rand: rand + seq_tot_tokens]

        # Reshape to (num_events, 5): [dtime, dur, pitch, vel, chan]
        num_events = len(x) // self.tokens_per_note
        if num_events == 0:
            # print("********** No events found, retrying **********")
            return self.__getitem__(randint(0, len(self) - 1))
        events = x[:num_events * self.tokens_per_note].view(num_events, self.tokens_per_note).long()

        # --- Identify event types ---
        is_chord = (events[:, 4] == CHORDS_CHANNEL)
        is_note = (events[:, 4] != CHORDS_CHANNEL) & (events[:, 4] != HARMONY_CHANNEL)
        note_indices = torch.where(is_note)[0]

        if len(note_indices) == 0:
            print("********** note indices is empty, retrying **********")
            return self.__getitem__(randint(0, len(self) - 1))

        target_len = self.seq_len + 1

        # --- Build chord multi-hot table per chord group ---
        # A new chord group starts when is_chord transitions from False→True
        prev_is_chord = torch.cat([torch.tensor([False]), is_chord[:-1]])
        chord_group_starts = is_chord & ~prev_is_chord
        # Cumulative group IDs: 0 = before any chord, 1 = first chord group, etc.
        chord_group_ids = torch.cumsum(chord_group_starts.long(), dim=0)

        # Only real chord notes (vel > 0) contribute to multi-hot.
        # Chord-off markers [0,0,0,0,4] have vel=0 and form their own group
        # whose multi-hot stays all-zero (= no active chord).
        real_chord_mask = is_chord & (events[:, 3] > 0)
        num_real_chord = real_chord_mask.sum().item()
        num_groups = int(chord_group_ids.max().item()) + 1

        # Build pitch-class multi-hot per group (group 0 = no chord = zeros)
        chord_pcs_table = torch.zeros(num_groups, 12)
        # Build bass (minimum pitch) per group
        bass_min_pitch = torch.full((num_groups,), 127, dtype=torch.long)

        if num_real_chord > 0:
            chord_pitches = events[real_chord_mask, 2]                # raw pitches
            chord_pcs_idx = (chord_pitches % 12).long()               # pitch classes
            chord_grps = chord_group_ids[real_chord_mask].long()

            # Scatter-add one-hot pitch classes into per-group multi-hot
            pcs_one_hot = torch.zeros(num_real_chord, 12)
            pcs_one_hot.scatter_(1, chord_pcs_idx.unsqueeze(1), 1.0)
            chord_pcs_table.scatter_add_(
                0, chord_grps.unsqueeze(1).expand_as(pcs_one_hot), pcs_one_hot
            )
            chord_pcs_table = (chord_pcs_table > 0).float()          # clamp to binary

            # Bass: minimum pitch per group via scatter_reduce
            bass_min_pitch.scatter_reduce_(
                0, chord_grps, chord_pitches, reduce='amin', include_self=True
            )

        # Groups with no real chord notes (group 0 or chord-off groups): zero multi-hot
        no_chord_groups = (chord_pcs_table.sum(dim=1) == 0)
        bass_pc_table = (bass_min_pitch % 12).long()
        bass_pc_table[no_chord_groups] = 0

        # --- Forward-fill chord group to note events ---
        chord_group_per_event = chord_group_ids
        chord_group_per_note = chord_group_per_event[note_indices].long()
        chord_pcs_notes = chord_pcs_table[chord_group_per_note]       # [N_notes, 12]
        bass_pc_notes = bass_pc_table[chord_group_per_note]           # [N_notes]

        # --- Filter to note events and truncate to target_len ---
        note_events = events[note_indices]
        if len(note_events) < target_len:
            num_repeats = (target_len // len(note_events)) + 1
            note_events = note_events.repeat(num_repeats, 1)[:target_len]
            chord_pcs_notes = chord_pcs_notes.repeat(num_repeats, 1)[:target_len]
            bass_pc_notes = bass_pc_notes.repeat(num_repeats)[:target_len]
        else:
            note_events = note_events[:target_len]
            chord_pcs_notes = chord_pcs_notes[:target_len]
            bass_pc_notes = bass_pc_notes[:target_len]

        pitches = note_events[:, 2]
        dtimes = note_events[:, 0]
        durs = note_events[:, 1]

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
              
        # Find chord groups (simultaneous note groups)
        chord_groups = []
        current_chord = [0]  # Start with first note
        
        for i in range(1, len(abs_times)):
            if abs_times[i] - abs_times[i-1] <= self.cfg['data_augment_chord_threshold']:
                current_chord.append(i)
            else:
                if len(current_chord) > 1: # Only process if it's actually a chord
                    chord_groups.append(current_chord)
                current_chord = [i]
        
        if len(current_chord) > 1:
            chord_groups.append(current_chord)
        
        # Apply micro-alterations to chord notes
        for chord in chord_groups:
            shifts = torch.randint(-1, 2, (len(chord),))  # Random shifts of -1, 0, or 1
            abs_times[chord] = abs_times[chord] + shifts
        
        # Re-sort the sequence based on new absolute times
        sorted_indices = torch.argsort(abs_times)
        abs_times = abs_times[sorted_indices]
        durs = durs[sorted_indices]
        pitches = pitches[sorted_indices]
        chord_pcs_notes = chord_pcs_notes[sorted_indices]
        bass_pc_notes = bass_pc_notes[sorted_indices]
        
        # Convert back to delta times
        dtimes = torch.cat([abs_times[0:1], abs_times[1:] - abs_times[:-1]])
        dtimes = torch.clamp(dtimes, min=0, max=RANGE_DTIME_SHIFT)

        # Transposition: shift note pitches AND rotate chord pitch classes
        transposition_factor = randint(
            -self.cfg['data_augment_transpose_max'], self.cfg['data_augment_transpose_max']
        )
        # TODO: Clamp isn't a good idea as we alter the interval relationships. But we hope transposing +-6 we don't clamp
        pitches = torch.clamp(pitches + transposition_factor, min=0, max=VOCAB_SIZE_PITCH-1)

        # Rotate multi-hot by the transposition amount to keep chord aligned
        if transposition_factor != 0:
            shift = transposition_factor % 12
            chord_pcs_notes = torch.roll(chord_pcs_notes, shifts=shift, dims=1)
            bass_pc_notes = (bass_pc_notes + transposition_factor) % 12

        feature_data = {
            'pitch': pitches,
            'chord_pcs': chord_pcs_notes,   # [T+1, 12] float
            'bass_pc': bass_pc_notes,        # [T+1] long
        }
        return feature_data


def main():
    # Set up CUDA settings
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)

    #==========================================================================

    ''' DEVICE '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type='cuda' if torch.cuda.is_available() else 'cpu'

    #==========================================================================

    ''' MODEL & HYPERPARAMETERS '''
    project_name = 'monsterGenie_chord_residual'
    model_name = 'AE_residual_v1'
    cfg = get_model_hparams(model_name)

    # ---- Load frozen base model from its pretrained checkpoint ----
    base_model_name = cfg['base_model_name']
    base_cfg = get_model_hparams(base_model_name)
    base_model = load_model(model_name=base_model_name, cfg=base_cfg,
                            set_only=False, compile_mode='none')

    # ---- Wrap in residual chord adapter ----
    model = AE_buttons_p_residual(base_model=base_model, cfg=cfg)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.2f}%)")

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
    data_train = torch.Tensor(train_data)
    eval_data = Any_Pickle_File_Reader(cfg['dataset_val_path'])   
    data_eval = torch.Tensor(eval_data)

    # Dataloader
    train_dataset = MusicSamplerDataset(data_train, cfg['seq_len'], cfg=cfg)
    print(f"BATCH_SIZE: {cfg['batch_size']}")
    print(f"Dataset size: {len(train_dataset)}")
    train_loader  = DataLoader(train_dataset, batch_size = cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True)
    print(f"Number of batches: {len(train_loader)}")
    val_dataset = MusicSamplerDataset(data_eval, cfg['seq_len'], is_eval=True, cfg=cfg)
    val_loader  = DataLoader(val_dataset, batch_size = cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=False)

    val_iter = iter(val_loader)

    #==========================================================================

    ''' PRECISION/OPTIMIZER/SCALER '''

    dtype = torch.bfloat16

    # Only optimize the chord adapter parameters (base model is frozen)
    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['learning_rate'],
    )

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

                # move to device
                x = {
                    'pitch': batch['pitch'].to(device),
                    'chord_pcs': batch['chord_pcs'].to(device),
                    'bass_pc': batch['bass_pc'].to(device),
                }

                with torch.amp.autocast(device_type=device_type, dtype=dtype):
                    loss, acc = model(x)
                scaler.scale(loss['loss_total']).backward()
                
                if (i % cfg['print_stats_every'] == 0) or TESTING:
                    if( cfg['use_logs']):                
                        wandb.log({
                            "loss_total": loss['loss_total'].item(),
                            "loss_recons": loss['loss_recons'].item(),
                            "train_acc": acc.item(),
                        }, step=nsteps)
                        
                        nsteps += 1


                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    cfg['grad_clip'],
                )
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
                                'chord_pcs': val_batch['chord_pcs'].to(device),
                                'bass_pc': val_batch['bass_pc'].to(device),
                            }
                            val_loss, val_acc = model(vx)

                        if(cfg['use_logs']):                
                            wandb.log({
                                "val_loss": val_loss['loss_total'].item(),
                                "val_acc": val_acc.item(),
                            }, step=nsteps)
                    model.train()
                    del val_batch, vx
                    torch.cuda.empty_cache()

        
        if ep % cfg['save_every'] == 0:
            fname = ('./save_models/' + cfg['model_name'] + '_' + str(ep) + '_eps_'
                     + str(nsteps) + '_steps_'
                     + str(round(float(loss['loss_total'].item()), 4)) + '_loss_'
                     + str(round(float(acc.item()), 4)) + '_acc.pth')
            torch.save(model.state_dict(), fname)


if __name__ == '__main__':
    mp.freeze_support()
    main()
