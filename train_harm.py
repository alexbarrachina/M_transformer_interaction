#===================================================================================================
# Monster Genie train_harm.py Python module
# Training with dual conditioning: melodic shape buttons + harmony movement (exponential decay)
# Pickle format: flat [dtime, dur, pitch, vel, chan] with harmony movements as [0, move_type, 0, 0, 3]
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
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

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
    Dataset for dual-conditioned model (melodic shape buttons + harmony movements).
    
    Reads flat pickle format: [dtime, dur, pitch, vel, chan] (5 tokens per event).
    Harmony movement events: [0, 0, movement_type, 0, 3] (chan=3).
    Note events have chan in {0, 10, ...}.
    
    For each note, builds:
    - harm_regime: forward-filled movement type from the most recent harmony event
    - harm_strength: exponential decay from that event's position (1.0 at onset -> ~0.05 at span end)
    """
    def __init__(self, data, seq_len, is_eval=False, cfg=None):
        super().__init__()

        self.data = data
        self.seq_len = seq_len
        self.tokens_per_note = 5  # dtime, dur, pitch, vel, chan
        self.cfg = cfg if cfg is not None else {}

        # To sample, we need enough tokens to extract seq_len notes after filtering
        # Oversample by 2x to account for harmony events mixed in
        self.seq_tot_tokens = self.seq_len * self.tokens_per_note * 2

    def __len__(self):
        # return int(self.data.size(0) / self.seq_tot_tokens)  #  self.seq_len if you want exact training time per epoch
        return int(self.data.size(0) / (self.seq_len * self.tokens_per_note + self.tokens_per_note))

    def __getitem__(self, index): # TODO concatenates all data, end of files with begining of files
        seq_tot_tokens = self.seq_tot_tokens
        # Pick a random starting position aligned to 5-token boundaries
        max_start = (self.data.size(0) - seq_tot_tokens) // self.tokens_per_note
        if max_start <= 0:
            max_start = 1
        rand = randint(0, max_start) * self.tokens_per_note

        x = self.data[rand: rand + seq_tot_tokens]

        # Reshape to (num_events, 5): [dtime, dur, pitch, vel, chan]
        # Harmony events follow the same layout: [0, 0, movement_type, 0, 3]
        num_events = len(x) // self.tokens_per_note
        if num_events == 0:
            return self._create_dummy_sample()
        events = x[:num_events * self.tokens_per_note].view(num_events, self.tokens_per_note).long()

        # --- Vectorized harmony regime + strength computation (no for loops) ---
        is_harmony = (events[:, 4] == HARMONY_CHANNEL)
        is_note = ~is_harmony
        note_indices = torch.where(is_note)[0]

        if len(note_indices) == 0:
            return self._create_dummy_sample()

        target_len = self.seq_len + 1

        # Forward-fill movement type via cumsum grouping over all events
        harm_cumsum = torch.cumsum(is_harmony.long(), dim=0)  # group id per event (0 = before any harmonyunguided, 1 = after first harmony event, etc.)
        harm_event_indices = torch.where(is_harmony)[0]
        num_harm = harm_event_indices.shape[0]

        # Lookup table: group_id -> movement_type (group 0 = before any harmony = unguided)
        move_lookup = torch.zeros(num_harm + 1, dtype=torch.long)
        if num_harm > 0:
            move_lookup[1:] = torch.clamp(events[harm_event_indices, 2] - 60, min=0, max=7) # MIDI note numbers are 0-127, but we want to map to 0-7 for the movement types

        # Project to note-space
        harm_group_notes = harm_cumsum[note_indices]
        harm_regime_notes = move_lookup[harm_group_notes]

        # Position within each group (in note-space) via diff on group boundaries
        group_changes = torch.cat([
            torch.tensor([True]),
            harm_group_notes[1:] != harm_group_notes[:-1]
        ])
        group_start_indices = torch.where(group_changes)[0]
        note_arange = torch.arange(len(harm_group_notes))
        group_of_note = torch.searchsorted(group_start_indices, note_arange, side='right') - 1
        position_in_group = note_arange - group_start_indices[group_of_note]

        # Span lengths per group, broadcast to each note
        span_lengths = torch.diff(group_start_indices, append=torch.tensor([len(harm_group_notes)]))
        span_length_per_note = span_lengths[group_of_note]

        # Linear decay from 1.0 at onset to 0.0 at next onset
        harm_strength_notes = 1.0 - position_in_group.float() / span_length_per_note.float()

        # Zero out unguided notes (group 0 = before any harmony event)
        unguided = (harm_group_notes == 0)
        harm_strength_notes = harm_strength_notes.masked_fill(unguided, 0.0)

        # --- Filter to note events and truncate to target_len ---
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
        harm_regime = harm_regime[sorted_indices]
        harm_strength = harm_strength[sorted_indices]
        
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
            #'dtime': dtime,
            #'dur': dur,
            #'channel': channel,
            'pitch': pitches,
            'harm_regime': harm_regime,
            'harm_strength': harm_strength,
        }
        return feature_data
    
    def _create_dummy_sample(self) -> dict:
        target_len = self.seq_len + 1
        print("********** Creating dummy sample **********")
        return {
            #'dtime': torch.full((target_len,), 1, dtype=torch.long),
            #'dur': torch.full((target_len,), 1, dtype=torch.long),
            # 'channel': torch.zeros(target_len, dtype=torch.long),
            'pitch': torch.full((target_len,), 60, dtype=torch.long),
            'harm_regime': torch.zeros(target_len, dtype=torch.long),
            'harm_strength': torch.zeros(target_len, dtype=torch.float),
        }


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
    project_name = 'monsterGenie_harmony_movement'
    model_name = 'AE_dual_tester_v4'
    cfg = get_model_hparams(model_name)
    model = load_model(model_name=model_name, cfg=cfg, set_only=True)  
    model.to(device)
    
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
    train_dataset = MusicSamplerDataset(data_train, cfg['seq_len'], cfg=cfg)  # train in chunks of SEQ_LEN
    print(f"BATCH_SIZE: {cfg['batch_size']}")
    print(f"Dataset size: {len(train_dataset)}")
    train_loader  = DataLoader(train_dataset, batch_size = cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True)
    print(f"Number of batches: {len(train_loader)}")
    val_dataset = MusicSamplerDataset(data_eval, cfg['seq_len'], is_eval=True, cfg=cfg)
    val_loader  = DataLoader(val_dataset, batch_size = cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=False)

    # Right after val_loader is created and before model definition, add a reusable iterator for streaming validation
    val_iter = iter(val_loader)  # will be cycled through inside training loop

    #==========================================================================

    ''' PRECISION/OPTIMIZER/SCALER '''

    dtype = torch.bfloat16

    #ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)

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

                # move to device
                x = {
                    #'dtime': batch['dtime'].to(device),
                    #'dur': batch['dur'].to(device),
                    # 'channel': batch['channel'].to(device),
                    'pitch': batch['pitch'].to(device),
                    'harm_regime': batch['harm_regime'].to(device),
                    'harm_strength': batch['harm_strength'].to(device),
                }

                with torch.amp.autocast(device_type=device_type, dtype=dtype):
                    loss, acc = model(x)  # Update your model to accept target separately
                scaler.scale(loss['loss_total']).backward()
                
                if (i % cfg['print_stats_every'] == 0) or TESTING:
                    if( cfg['use_logs']):                
                        wandb.log({"loss_total": loss['loss_total'].item()}, step=nsteps)
                        wandb.log({"train_acc": acc.item()}, step=nsteps)
                        if cfg.get('loss_norm_pos', 0)>0 and 'loss_norm_pos' in loss:
                            wandb.log({"loss_norm_pos": cfg['loss_norm_pos']*loss['loss_norm_pos'].item()}, step=nsteps)
                        if cfg.get('loss_deviate', 0)>0 and 'loss_deviate' in loss:
                            wandb.log({"loss_deviate": cfg['loss_deviate']*loss['loss_deviate'].item()}, step=nsteps)
                        if cfg.get('loss_margin', 0)>0 and 'loss_margin' in loss:
                            wandb.log({"loss_margin": cfg['loss_margin']*loss['loss_margin'].item()}, step=nsteps)
                        if cfg.get('loss_pitch_button', 0)>0 and 'loss_pitch_button' in loss:
                            wandb.log({"loss_pitch_button": cfg['loss_pitch_button']*loss['loss_pitch_button'].item()}, step=nsteps)
                        if cfg.get('loss_button_concentration', 0)>0 and 'loss_button_concentration' in loss:
                            wandb.log({"loss_button_concentration": cfg['loss_button_concentration']*loss['loss_button_concentration'].item()}, step=nsteps)
                        if cfg.get('loss_window_corr', 0)>0 and 'loss_window_corr' in loss:
                            wandb.log({"loss_window_corr": cfg['loss_window_corr']*loss['loss_window_corr'].item()}, step=nsteps)
                        if cfg.get('loss_latent_velocity', 0)>0 and 'loss_latent_velocity' in loss:
                            wandb.log({"loss_latent_velocity": cfg['loss_latent_velocity']*loss['loss_latent_velocity'].item()}, step=nsteps)
                        if cfg.get('loss_drift', 0)>0 and 'loss_drift' in loss:
                            wandb.log({"loss_drift": cfg['loss_drift']*loss['loss_drift'].item()}, step=nsteps)

                        if cfg['loss_contour']>0 and 'loss_contour' in loss:
                            wandb.log({"loss_contour_all": cfg['loss_contour']*loss['loss_contour'].item()}, step=nsteps)
                        
                        if cfg.get('loss_contour_perc', 0)>0 and 'loss_contour_perc' in loss:
                            wandb.log({"loss_contour_perc": cfg['loss_contour']*cfg['loss_contour_perc']*loss['loss_contour_perc'].item()}, step=nsteps)
                        if cfg.get('loss_multi_step_perc', 0)>0 and 'loss_multi_step_perc' in loss:
                            wandb.log({"loss_multi_step": cfg['loss_contour']*cfg['loss_multi_step_perc']*loss['loss_multi_step_perc'].item()}, step=nsteps)
                        if cfg.get('loss_interval_perc', 0)>0 and 'loss_interval' in loss:
                            wandb.log({"loss_interval": cfg['loss_contour']*cfg['loss_interval_perc']*loss['loss_interval_perc'].item()}, step=nsteps)
                        if cfg.get('loss_shape_perc', 0)>0 and 'loss_shape_perc' in loss:
                            wandb.log({"loss_shape": cfg['loss_contour']*cfg['loss_shape_perc']*loss['loss_shape_perc'].item()}, step=nsteps)
                        
                        if cfg.get('loss_button_held', 0)>0 and 'loss_button_held' in loss: 
                            wandb.log({"loss_button_held": cfg['loss_button_held']*loss['loss_button_held'].item()}, step=nsteps)
                        if cfg.get('loss_recons', 0)>0 and 'loss_recons' in loss: 
                            wandb.log({"loss_recons": cfg['loss_recons']*loss['loss_recons'].item()}, step=nsteps)
                        if cfg.get('loss_arrow_consistency', 0)>0 and 'loss_arrow_consistency' in loss: 
                            wandb.log({"loss_arrow_consistency": cfg['loss_arrow_consistency']*loss['loss_arrow_consistency'].item()}, step=nsteps)
                        if cfg.get('loss_coarse_direction', 0)>0 and 'loss_coarse_direction' in loss: 
                            wandb.log({"loss_coarse_direction": cfg['loss_coarse_direction']*loss['loss_coarse_direction'].item()}, step=nsteps)
                        
                        nsteps += 1


                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                scaler.step(optim)
                scaler.update()


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
                            vx = {
                                #'dtime': val_batch['dtime'].to(device),
                                #'dur': val_batch['dur'].to(device),
                                #'channel': val_batch['channel'].to(device),
                                'pitch': val_batch['pitch'].to(device),
                                'harm_regime': val_batch['harm_regime'].to(device),
                                'harm_strength': val_batch['harm_strength'].to(device),
                            }
                            # run the model
                            val_loss, val_acc = model(vx)  # Update your model to accept target separately

                        if(cfg['use_logs']):                
                            wandb.log({"val_loss": val_loss['loss_total'].item()}, step=nsteps)
                            wandb.log({"val_acc": val_acc.item()}, step=nsteps)
                    model.train()
                    del val_batch, vx
                    torch.cuda.empty_cache()

        
        if ep % cfg['save_every'] == 0:
            fname = './save_models/' + cfg['model_name'] + '_' + str(ep) + '_eps_' + str(nsteps) + '_steps_' + str(round(float(loss['loss_total'].item()), 4)) + '_loss_' + str(round(float(acc.item()), 4)) + '_acc.pth'
            torch.save(model.state_dict(), fname)


if __name__ == '__main__':
    mp.freeze_support()
    main()
