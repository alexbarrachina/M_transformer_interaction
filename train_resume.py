#===================================================================================================
# Monster Genie train_resume.py Python module
# Resume training from checkpointt
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
import glob
import re
#from torch.utils.tensorboard import SummaryWriter

#!set USE_FLASH_ATTENTION=1
os.environ['USE_FLASH_ATTENTION'] = '1'

from random import randint, random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset, load_from_disk

from params import *
from midiUtils import tokens_to_dict, Any_Pickle_File_Reader
from model_loader import load_model
from x_transformer import *

#==========================================================================

class MusicSamplerDataset(Dataset):
    def __init__(self, data, seq_len, is_eval=False):
        super().__init__()

        self.data = data
        self.seq_len = seq_len
        self.seq_tot_tokens = self.seq_len * 4 + 4 # 4 tokens per note + 4 for the current note


    def __len__(self):
        return int(self.data.size(0) / (self.seq_len * 4 + 4))  #  self.seq_len if you want exact training time per epoch

    def __getitem__(self, index): # TODO concatenates all data, end of files with begining of files
        seq_tot_tokens = self.seq_tot_tokens
        # We only pick starting positions that are multiples of seq_tot_tokens, // seq_tot_tokens: Integer division to get how many complete sequences of size seq_tot_tokens we can fit
        # We don't exceed the data boundaries
        #rand = secrets.randbelow((self.data.size(0)-seq_tot_tokens) // seq_tot_tokens) * seq_tot_tokens
        rand = randint(0, (self.data.size(0)-seq_tot_tokens) // seq_tot_tokens) * seq_tot_tokens

        # Extract sequences for each feature, +1 to include the current token
        x = self.data[rand: rand + seq_tot_tokens] # we take an extra token

        # convert to tensors, move to device
        dtimes = x[0::4].long()  # Every 4th token starting at index 0. observed min = 0, max = 70
        durs = (x[1::4] - OFFSET_DUR).long()  # Every 4th token starting at index 2. observed min = 1, max = 74
        pitches = (x[2::4] - OFFSET_PITCH).long()  # Every 4th token starting at index 3. observed min = 30, max = 88
        #vels = (x[3::4] - OFFSET_VEL).long()  # Every 4th token starting at index 4. observed min = 30, max = 88

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
                'dur':  durs,
                'pitch': pitches
                #'vel': vels
            }
        return feature_data

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
    model_name = 'encoder_button_held'
    cfg = get_model_hparams(model_name)
    model = load_model(model_name=model_name, cfg=cfg, set_only=True)  
    model.to(device)
    #print(model)

    #==========================================================================

    ''' WANDB '''
    if(cfg['use_logs']):
        #tensorboard_summary = SummaryWriter()
        import wandb
        wandb.login()
        wandb.init(project="monsterGenie", config=cfg)


    #==========================================================================

    ''' DATA '''

    """ LOAD TRAINING DATA """

    # Loading dataset from a pickle in ./Training-Data
    train_data = Any_Pickle_File_Reader(DATASET_TRAIN_PATH)   
    data_train = torch.Tensor(train_data)
    eval_data = Any_Pickle_File_Reader(DATASET_VAL_PATH)   
    data_eval = torch.Tensor(eval_data)

    # Dataloader
    train_dataset = MusicSamplerDataset(data_train, cfg['seq_len'], cfg=cfg) # train in chunks of SEQ_LEN
    print(f"BATCH_SIZE: {cfg['batch_size']}")
    print(f"Dataset size: {len(train_dataset)}")
    train_loader  = DataLoader(train_dataset, batch_size = cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True)
    print(f"Number of batches: {len(train_loader)}")
    val_dataset = MusicSamplerDataset(data_eval, cfg['seq_len'], is_eval=True) # train in chunks of SEQ_LEN
    val_loader  = DataLoader(val_dataset, batch_size = cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=False)

    #==========================================================================
 
    ''' PRECISION/OPTIMIZER/SCALER '''

    dtype = torch.bfloat16

    ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.amp.GradScaler(device_type)

    #==========================================================================

    ''' LOAD CHECKPOINT '''
    
    checkpoint_path, start_epoch, start_steps = find_latest_checkpoint()
    
    if checkpoint_path:
        start_epoch, start_steps = load_checkpoint(model, optim, checkpoint_path, device)
        print(f"Resuming training from epoch {start_epoch}, step {start_steps}")
    else:
        start_epoch = 0
        start_steps = 0
        print("Starting training from scratch (no checkpoint found)")

    #==========================================================================

    ''' WANDB '''
    if(USE_LOGS):
        # Resume wandb run if we have a checkpoint
        if checkpoint_path:
            wandb.init(project="monsterGenie", config=config, resume="allow")
        else:
            wandb.init(project="monsterGenie", config=config)

    #==========================================================================

    ''' TRAINING '''

    nsteps = start_steps

    for ep in range(start_epoch, NUM_EPOCHS):
        print(f'Epoch #{ep} (resuming from step {nsteps})')
        
        model.train()
        with tqdm.tqdm(total=len(train_loader)) as bar_train:
            for i, batch in enumerate(train_loader):            
                optim.zero_grad()

                # move to device
                x = {
                    'dtime': batch['dtime'].to(device),
                    'dur': batch['dur'].to(device),
                    'pitch': batch['pitch'].to(device)
                }

                with ctx:
                    loss, acc = model(x)  # Update your model to accept target separately
                scaler.scale(loss['loss_total']).backward()
                
                if (i % cfg['validate_every'] == 0) or TESTING:
                    if(cfg['use_logs']):                
                        wandb.log({"train_loss": loss['loss_total'].item()}, step=nsteps)
                        wandb.log({"train_acc": acc.item()}, step=nsteps)
                        nsteps += 1

                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                scaler.step(optim)
                scaler.update()

                bar_train.set_description(f'Epoch: {ep} Loss: {float(loss["loss_total"]):.4}')
                bar_train.update(1)

                if (i % cfg['print_stats_every'] == 0) or TESTING:
                    try:
                        x = next(iter(val_loader)) # extract batches from test dataloader
                    except StopIteration:
                        val_loader_iter = iter(val_loader)
                        batch = next(iter(val_loader_iter)) # extract batches from test dataloader           
                    model.eval()
                    with torch.no_grad():
                        with ctx:
                            # move to device
                            x = {
                                'dtime': batch['dtime'].to(device),
                                'dur': batch['dur'].to(device),
                                'pitch': batch['pitch'].to(device)
                            }
                            # run the model
                            val_loss, val_acc = model(x)  # Update your model to accept target separately

                        if(cfg['use_logs']):                
                            wandb.log({"val_loss": val_loss['loss_total'].item()}, step=nsteps)
                            wandb.log({"val_acc": val_acc.item()}, step=nsteps)

                    model.train()

        
        if ep % cfg['save_every'] == 0:
            fname = './save_models/' + cfg['model_name'] + '_' + str(ep) + '_eps_' + str(nsteps) + '_steps_' + str(round(float(loss['loss_total'].item()), 4)) + '_loss_' + str(round(float(acc.item()), 4)) + '_acc.pth'
            torch.save(model.state_dict(), fname)


if __name__ == '__main__':
    mp.freeze_support()
    main() 