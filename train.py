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
    torch.backends.cuda.enable_cudnn_sdp(False)

    #==========================================================================

    ''' DEVICE '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type='cuda' if torch.cuda.is_available() else 'cpu'

    #==========================================================================

    ''' MODEL & HYPERPARAMETERS '''
    project_name = 'monsterGenie_melody'
    model_name = 'melody_arrow_v1'
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

    # Loading dataset from a pickle in ./Training-Data
    train_data = Any_Pickle_File_Reader(cfg['dataset_train_path'])   
    data_train = torch.Tensor(train_data)
    eval_data = Any_Pickle_File_Reader(cfg['dataset_val_path'])   
    data_eval = torch.Tensor(eval_data)

    # Dataloader
    train_dataset = MusicSamplerDataset(data_train, cfg['seq_len'], cfg=cfg) # train in chunks of SEQ_LEN
    print(f"BATCH_SIZE: {cfg['batch_size']}")
    print(f"Dataset size: {len(train_dataset)}")
    train_loader  = DataLoader(train_dataset, batch_size = cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True)
    print(f"Number of batches: {len(train_loader)}")
    val_dataset = MusicSamplerDataset(data_eval, cfg['seq_len'], is_eval=True, cfg=cfg) # train in chunks of SEQ_LEN
    val_loader  = DataLoader(val_dataset, batch_size = cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=False)

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
                    'pitch': batch['pitch'].to(device)
                }

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
                                'pitch': val_batch['pitch'].to(device)
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

