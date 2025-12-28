#===================================================================================================
# Monster Genie train_harmony.py Python module
# Training with GIANTsel dataset with harmony conditioning
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

import time
import tqdm
#from params import *

os.environ['USE_FLASH_ATTENTION'] = '1'

from random import randint, random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset, load_from_disk

from midiUtils import Any_Pickle_File_Reader
from model_loader import load_model
from models import get_model_hparams
from params import *
from x_transformer import *

#==========================================================================
# Channel markers for event types (must match midis2pickles_harmony.py)
CHANNEL_HARMONY = 17  # Marks a harmony event
CHANNEL_NEW_SONG = 18  # Marks a new song event
HARMONY_UNKNOWN = 128  # Value for unknown harmony (out of 0-127 range)
KEY_UNKNOWN = 24  # Value for unknown key (0-11 major, 12-23 minor, 24 unknown)

class MusicSamplerDataset(Dataset):
    """
    Dataset for harmony-augmented pickles (FLAT TOKEN FORMAT).
    
    Pickle contains a single flat list of integers (5 tokens per event):
    - Note event:     [channel, dtime, dur, pitch, vel]  where channel = 0-15
    - Harmony event:  [17, harm_x, harm_y, harm_r, key_root]
    - New song event: [18, key, 0, 0, 0]
    
    All values are in range 0-127 (except channel markers 17, 18).
    
    Harmony events are inserted at regular intervals in the data.
    For each note, we assign harmony from the most recent harmony event.
    Notes without harmony info get default values (128 for harmony, 24 for key).
    """

    def __init__(self, data: list, seq_len: int, is_eval: bool = False, cfg: dict = None):
        super().__init__()
        
        self.seq_len = seq_len
        self.is_eval = is_eval
        self.cfg = cfg if cfg is not None else {}
        
        # Convert flat list to tensor of shape (num_events, 5)
        # data is a flat list: [t0, t1, t2, t3, t4, t0, t1, ...]
        num_tokens = len(data)
        num_events = num_tokens // 5
        
        # Reshape to (num_events, 5): each row is [channel, t1, t2, t3, t4]
        self.data = torch.tensor(data, dtype=torch.long).view(num_events, 5)
        
        # Find positions of "new_song" markers (channel == 18)
        # These mark the start of each song in the flat sequence
        channels = self.data[:, 0]
        song_mask = (channels == CHANNEL_NEW_SONG)
        self.song_pos = torch.where(song_mask)[0]  # Tensor of song start indices
        
        # If no songs found (shouldn't happen), treat position 0 as start
        if len(self.song_pos) == 0:
            self.song_pos = torch.tensor([0], dtype=torch.long)
        
        # Calculate total samples (for epoch length)
        # Count total number of note events (channel < 17)
        note_mask = (channels < CHANNEL_HARMONY)
        self._total_samples = int(note_mask.sum().item())  # Total number of note events
        
        print(f"Dataset: {num_events} events, {len(self.song_pos)} songs, {self._total_samples} samples/epoch")

    def __len__(self) -> int:
        return self._total_samples // self.seq_len 

    def __getitem__(self, index: int) -> dict:
        """
        Sample a sequence of notes with associated harmony information.
        
        1. Pick a song start position (deterministic for validation, random for training)
        2. Extract seq_len * 2 events (to ensure enough notes after filtering)
        3. Filter to keep only note events (channel < 17)
        4. Take first seq_len notes
        5. Build feature tensors with harmony assigned to following notes
        """
        # 1. Pick song and position (deterministic for validation, random for training)
        if self.is_eval:
            # Deterministic sampling for validation
            # Use index to select song and position within song
            song_idx = index % len(self.song_pos)
            song_start = int(self.song_pos[song_idx])
            
            # Get song end (next song start or end of data)
            if song_idx + 1 < len(self.song_pos):
                song_end = int(self.song_pos[song_idx + 1])
            else:
                song_end = len(self.data)
            
            # Deterministic offset within song based on index
            sample_within_song = index // len(self.song_pos)
            offset = sample_within_song * self.seq_len  # Stride by seq_len for coverage
            start_event = min(song_start + offset, song_end - 1)
        else:
            # Random sampling for training (full diversity)
            song_idx = randint(0, len(self.song_pos) - 1)
            song_start = int(self.song_pos[song_idx])
            
            # Get song end (next song start or end of data)
            if song_idx + 1 < len(self.song_pos):
                song_end = int(self.song_pos[song_idx + 1])
            else:
                song_end = len(self.data)
            
            # Random position within the song
            song_length = song_end - song_start
            if song_length > self.seq_len * 2:
                # Pick random offset, leaving room for seq_len*2 events
                max_offset = song_length - self.seq_len * 2
                offset = randint(0, max_offset)
                start_event = song_start + offset
            else:
                start_event = song_start
        
        # 2. Calculate end position (extract more events than needed to ensure enough notes)
        # We need seq_len notes, but some events are harmony/song markers
        # Extract seq_len * 2 events to be safe
        end_event = min(start_event + self.seq_len * 2, len(self.data))
        
        # Get the slice of events
        events_slice = self.data[start_event:end_event]  # (num_events, 5)
        
        if len(events_slice) == 0:
            # Edge case: empty slice, create dummy data
            return self._create_dummy_sample()
        
        # 3. Separate event types
        channels = events_slice[:, 0]
        
        # Find note events (channel < 17) and harmony events (channel == 17)
        note_mask = (channels < CHANNEL_HARMONY)
        harmony_mask = (channels == CHANNEL_HARMONY)
        
        # Get note events: [channel, dtime, dur, pitch, vel]
        note_events = events_slice[note_mask]  # (num_notes, 5)
        
        # Get harmony events: [17, harm_x, harm_y, harm_r, key_root]
        harmony_events = events_slice[harmony_mask]  # (num_harmony, 5)
        
        # 4. Take first seq_len notes (or pad if not enough)
        num_notes = len(note_events)
        
        if num_notes == 0:
            return self._create_dummy_sample()
        
        if num_notes < self.seq_len:
            # Pad by repeating the sequence
            num_repeats = (self.seq_len // num_notes) + 1
            note_events = note_events.repeat(num_repeats, 1)[:self.seq_len]
        else:
            note_events = note_events[:self.seq_len]
        
        # 5. Extract note features: [channel, dtime, dur, pitch, vel]
        channels_out = note_events[:, 0]
        dtimes = note_events[:, 1]
        durs = note_events[:, 2]
        pitches = note_events[:, 3]
        vels = note_events[:, 4]
        
        # 6. Build harmony tensors (VECTORIZED - no loops)
        # Harmony shape: (seq_len, 4) = [harm_x, harm_y, harm_r, harm_active]
        # harm_active = 1.0 when harmony is present, 0.0 when unknown
        # This allows model to distinguish "C Major" (0,0,0,1) from "No Guidance" (0,0,0,0)
        harmonies = torch.zeros((self.seq_len, 4), dtype=torch.float)  # Default: all zeros (no harmony)
        keys = torch.full((self.seq_len,), KEY_UNKNOWN, dtype=torch.long)
        
        if len(harmony_events) > 0:
            # Get positions in original slice
            note_positions = torch.where(note_mask)[0][:self.seq_len]
            harmony_positions = torch.where(harmony_mask)[0]
            
            if len(harmony_positions) > 0 and len(note_positions) > 0:
                # VECTORIZED: For each harmony, find the NEXT note event (the one that follows it)
                # searchsorted with side='left' gives the first note position >= harmony position
                next_note_indices = torch.searchsorted(note_positions, harmony_positions, side='left')
                
                # Filter out harmony events that don't have a following note in our slice
                valid_mask = (next_note_indices < len(note_positions))
                valid_harm_indices = torch.where(valid_mask)[0]
                
                if len(valid_harm_indices) > 0:
                    # Get the note indices (in output tensor) for each valid harmony
                    target_note_indices = next_note_indices[valid_harm_indices]
                    
                    # Get harmony positions in events_slice for valid harmonies
                    harm_pos_in_slice = harmony_positions[valid_harm_indices]
                    
                    # Gather harmony values from events_slice
                    # harm_event format: [17, harm_x, harm_y, harm_r, key_root]
                    harm_values = events_slice[harm_pos_in_slice]  # (num_valid_harm, 5)
                    
                    # Assign harmony to the corresponding note positions
                    # Scale from [0,127] to [-1, 1] directly
                    harmonies[target_note_indices, 0] = (harm_values[:, 1].float() / 127.0) * 2.0 - 1.0  # harm_x
                    harmonies[target_note_indices, 1] = (harm_values[:, 2].float() / 127.0) * 2.0 - 1.0  # harm_y
                    harmonies[target_note_indices, 2] = (harm_values[:, 3].float() / 127.0) * 2.0 - 1.0  # harm_r
                    harmonies[target_note_indices, 3] = 1.0  # harm_active = 1.0 (harmony is present)
                    keys[target_note_indices] = harm_values[:, 4]  # key_root
        
        # Data augmentation
        # Time stretching
        if not self.is_eval:
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
            vels = vels[sorted_indices]
            channels = channels[sorted_indices]
            harmonies = harmonies[sorted_indices]
            keys = keys[sorted_indices]
            # Convert back to delta times
            dtimes = torch.cat([abs_times[0:1], abs_times[1:] - abs_times[:-1]])
            dtimes = torch.clamp(dtimes, min=0, max=RANGE_DTIME_SHIFT)
            
            # Transposition
            transposition_factor = randint(
                -self.cfg['data_augment_transpose_max'], self.cfg['data_augment_transpose_max'])

            # Check if all pitches will remain in valid range after transposition
            pitches_after_transpose = pitches + transposition_factor
            if torch.any(pitches_after_transpose < 0) or torch.any(pitches_after_transpose > 127):
                # Skip transposition if any pitch would go out of range
                # All pitches are in valid range, proceed with transposition
                pass #do nothing
            else:
                pitches = pitches_after_transpose
 
                # Transpose key_root (VECTORIZED - no loops!)
                # key_root encoding: 0-11 = major keys, 12-23 = minor keys, 24 = unknown
                # Keep as long tensor for embedding indices
                is_unknown = (keys == 24)
                is_major = (keys < 12)
                is_minor = (keys >= 12) & (keys < 24)
        
                key_root_transposed = keys.clone()
        
                # Apply transposition to major and minor keys separately using modulo
                key_root_transposed[is_major] = (keys[is_major] + transposition_factor) % 12
                key_root_transposed[is_minor] = ((keys[is_minor] - 12 + transposition_factor) % 12) + 12
                keys = key_root_transposed
            # is_unknown keys remain 24 (no change needed)
        
        # Build feature dictionary
        # harmonies: shape (seq_len, 4) = [harm_x, harm_y, harm_r, harm_active]
        #   - harm_x, harm_y, harm_r: scaled to [-1, 1]
        #   - harm_active: 1.0 when harmony present, 0.0 when unknown (presence bit)
        # keys: KEY_UNKNOWN (24) for notes without harmony, actual key for notes with harmony
        feature_data = {
            'pitch': pitches,
            'dtime': dtimes,
            'dur': durs,
            'vel': vels,
            'channel': channels,
            'harmony': harmonies,
            'key': keys,
        }
        return feature_data
    
    def _create_dummy_sample(self) -> dict:
        """Create a dummy sample when no valid data is available."""
        print("Sampling ERROR. Creating dummy sample")
        return {
            'pitch': torch.full((self.seq_len,), 60, dtype=torch.long),  # Middle C
            'dtime': torch.zeros(self.seq_len, dtype=torch.long),
            'dur': torch.ones(self.seq_len, dtype=torch.long),
            'vel': torch.full((self.seq_len,), 64, dtype=torch.long),
            'channel': torch.zeros(self.seq_len, dtype=torch.long),
            'harmony': torch.zeros(self.seq_len, 4, dtype=torch.float),  # Shape (seq_len, 4)
            'key': torch.full((self.seq_len,), KEY_UNKNOWN, dtype=torch.long),
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
    project_name = 'monsterGenie_harmony'
    model_name = 'autoenc_just_harmony_v1b'
    cfg = get_model_hparams(model_name)
    model = load_model(model_name=model_name, cfg=cfg, set_only=True)  
    model.to(device)
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

    # Loading dataset from a pickle in ./Training-Data (FLAT TOKEN FORMAT)
    # Format: Single flat list of integers (5 tokens per event)
    # Event types: Note [chan, dtime, dur, pitch, vel], Harmony [17, x, y, r, key], NewSong [18, key, 0, 0, 0]
    print("Loading training data...")
    train_data = Any_Pickle_File_Reader(cfg['dataset_train_path'])
    # Validate: should be a flat list of integers
    if len(train_data) > 0 and isinstance(train_data[0], dict):
        raise TypeError(f"Expected pickle file to contain a flat list of integers, "
                      f"but found dict format. Please regenerate the pickle file using midis2pickles_harmony.py")
    print(f"Training data: {len(train_data)} tokens ({len(train_data)//5} events)")
    
    print("Loading validation data...")
    eval_data = Any_Pickle_File_Reader(cfg['dataset_val_path'])
    if len(eval_data) > 0 and isinstance(eval_data[0], dict):
        raise TypeError(f"Expected eval pickle file to contain a flat list of integers, "
                      f"but found dict format. Please regenerate the pickle file using midis2pickles_harmony.py")
    print(f"Validation data: {len(eval_data)} tokens ({len(eval_data)//5} events)")

    # Dataloader
    train_dataset = MusicSamplerDataset(train_data, cfg['seq_len'], cfg=cfg)
    print(f"BATCH_SIZE: {cfg['batch_size']}")
    print(f"Number of songs: {len(train_dataset.song_pos)}")
    print(f"Dataset size (samples per epoch): {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True, pin_memory=True)
    print(f"Number of batches per epoch: {len(train_loader)}")
    
    val_dataset = MusicSamplerDataset(eval_data, cfg['seq_len'], is_eval=True, cfg=cfg)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=False, pin_memory=True)

    # Right after val_loader is created and before model definition, add a reusable iterator for streaming validation
    val_iter = iter(val_loader)  # will be cycled through inside training loop

    #==========================================================================

    ''' PRECISION/OPTIMIZER/SCALER '''

    dtype = torch.bfloat16

    #ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)

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

                # move to device
                x = {
                    'dtime': batch['dtime'].to(device),
                    'dur': batch['dur'].to(device),
                    'channel': batch['channel'].to(device),
                    'pitch': batch['pitch'].to(device),
                    'harmony': batch['harmony'].to(device),
                    'key': batch['key'].to(device)
                }

                # Dropout for Sparsity
                #if model.training:
                    # SIMULATE SPARSE HARMONY (Simulate intermittent user guidance)
                    # Most steps (90%) will have no active harmonic goal
                    # This forces the model to rely on buttons/contour by default
                    #step_mask = torch.rand(x['key'].shape[0], x['key'].shape[1], device=device) < 0.9
                    #x['key'][step_mask] = 24  # Use index 24 for "Unknown" TODO: 24 as a global parameter
                    #x['harmony'][step_mask] = 0.0  # Set all 3 harmony components (harm_x, harm_y, harm_r) to 0

                with torch.amp.autocast(device_type=device_type, dtype=dtype):
                    loss, acc = model(x)  # Update your model to accept target separately
                scaler.scale(loss['loss_total']).backward()
                
                if (i % cfg['print_stats_every'] == 0) or TESTING:
                    if( cfg['use_logs']):                
                        wandb.log({"loss_total": loss['loss_total'].item()}, step=nsteps)
                        wandb.log({"train_acc": acc.item()}, step=nsteps)
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
                                'dtime': val_batch['dtime'].to(device),
                                'dur': val_batch['dur'].to(device),
                                'channel': val_batch['channel'].to(device),
                                'pitch': val_batch['pitch'].to(device),
                                'harmony': val_batch['harmony'].to(device),
                                'key': val_batch['key'].to(device)
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

