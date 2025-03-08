import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import tqdm
from torch.utils.tensorboard import SummaryWriter

USE_TENSORBOARD = True
if(USE_TENSORBOARD):
    tensorboard_summary = SummaryWriter()

#!set USE_FLASH_ATTENTION=1
os.environ['USE_FLASH_ATTENTION'] = '1'

import torch
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from datasets import load_dataset, load_from_disk
import TMIDIX

from x_transformer_1_23_2 import *

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(False)

import random

#==========================================================================

''' SETUP MODEL '''

# constants
SEQ_LEN = 1024 # 2048
SEQ_OVERLAP = 512 # 1024
PAD_IDX = 128 # 384 # Model pad index

VALIDATE_EVERY  = 500
SAVE_EVERY = 2500
GENERATE_EVERY  = 1000
GENERATE_LENGTH = 512
PRINT_STATS_EVERY = 50

NUM_EPOCHS = 10

#BATCH_SIZE = 116 # original
#BATCH_SIZE = 20 
# TESTING ALEX
BATCH_SIZE = 1 

LEARNING_RATE = 1e-4
GRAD_CLIP = 1.5

# Path to your locally saved dataset
local_dataset_path = "../../../Datasets/MIDI/asigalov61___monster-piano"
#local_dataset_path = "/Volumes/DADES/Datasets/MIDI/asigalov61___monster-piano"
#==========================================================================

''' DEVICE '''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_type='cuda' if torch.cuda.is_available() else 'cpu'

#==========================================================================

''' DATA '''
class MusicSamplerDataset(Dataset):
    def __init__(self, data, seq_len, is_eval=False):
        super().__init__()

        self.feature_data = []  # Changed from set to list for indexing
        self.seq_len = seq_len
        self.num_notes = 0
        self.is_eval = is_eval
        self.indices = []

        filtered_score = []
 
        for entry in tqdm.tqdm(data):
            score = entry['midi_score']

            i = 0
            while i < len(score):
                if score[i] < 384:  # Process valid tokens (dtime, pitch, dur)
                    # If we're missing notes in a triplet (chord case), add dtime=0
                    if score[i] > 127 and len(filtered_score) % 3 == 0: # dur or pitch in dtime postion
                        filtered_score.append(0)  # Insert dtime=0 for chord notes
                    
                    # range checker. We don't need it for now.
                    '''if len(filtered_score) % 3 == 0:
                        assert(score[i] < 128), "not a valid dtime"
                    if len(filtered_score) % 3 == 1:
                        assert(127 < score[i] < 256), "not a valid dur"
                    if len(filtered_score) % 3 == 2:
                        assert (255 < score[i] < 384), "not a valid pitch"'''

                    # revert offsets
                    offset = 0
                    if len(filtered_score) % 3 == 1:
                        offset = OFFSET_DUR
                    elif len(filtered_score) % 3 == 2:
                        offset = OFFSET_PITCH

                    filtered_score.append(score[i] - offset)
                i += 1

        # Ensure we have complete triplets
        assert len(filtered_score) % 3 == 0, "Data length must be divisible by 3 (dtime, pitch, dur)"

        self.num_notes = len(filtered_score) // 3
        self.feature_data = {
            'dtime': filtered_score[0::3],  # Every 3rd token starting at index 0. observed min = 0, max = 70
            'dur': filtered_score[1::3],  # Every 3rd token starting at index 2. observed min = 1, max = 74
            'pitch': filtered_score[2::3]     # Every 3rd token starting at index 3. observed min = 30, max = 88
        }
        
        if self.is_eval:
            max_indices = self.num_notes - self.seq_len
            self.indices = list(range(0, max_indices, self.seq_len))

    def __len__(self):
        return self.num_notes // self.seq_len  # Return number of possible sequences

    def __getitem__(self, index):
        # Calculate start position for this index
        if self.is_eval:
            # For evaluation, use sequential samples
            rand = self.indices[index % len(self.indices)]
        else:
            # For training, use random sampling
            max_start_idx = self.num_notes - self.seq_len
            rand = torch.randint(0, max_start_idx, (1,)).item()

        # Extract sequences for each feature
        x = {
            'dtime': torch.tensor(self.feature_data['dtime'][rand:rand + self.seq_len], dtype=torch.long),
            'dur': torch.tensor(self.feature_data['dur'][rand:rand + self.seq_len], dtype=torch.long),
            'pitch': torch.tensor(self.feature_data['pitch'][rand:rand + self.seq_len], dtype=torch.long)
        }

        return x

#monster_piano = load_dataset('asigalov61/Monster-Piano') # original
#monster_piano = load_dataset('asigalov61/Monster-Piano', split='train[:70%]') 
#monster_piano_val = load_dataset('asigalov61/Monster-Piano', split='train[95%:100%]') 
# TESTING ALEX

# Load the dataset from disk
monster_piano = load_from_disk(local_dataset_path)

#monster_piano = load_dataset('asigalov61/Monster-Piano', split='train[:1%]') 
#monster_piano_val = load_dataset('asigalov61/Monster-Piano', split='train[99%:100%]') 
# If you need specific splits, you can select them after loading
train_dataset = monster_piano['train']
monster_piano_train = train_dataset.select(range(int(len(train_dataset) * 0.01)))  # 1% for training
monster_piano_val = train_dataset.select(range(int(len(train_dataset) * 0.99), len(train_dataset)))  # Last 1% for validation


# Dataloader
train_data = MusicSamplerDataset(monster_piano_train, SEQ_LEN)
val_data = MusicSamplerDataset(monster_piano_val, SEQ_LEN, is_eval=True) 

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

#==========================================================================

''' MODEL '''

model = AutoregressiveAutoencoder(
    ignore_index = PAD_IDX, 
    pad_value=PAD_IDX,
    decoder = Decoder(
        num_tokens = PAD_IDX+1,
        max_seq_len = SEQ_LEN,
        dim = 2048,
        depth = 4,
        heads = 32,
        rotary_pos_emb = True,
        attn_flash = True
        ),
    encoder = Encoder(
        num_tokens = PAD_IDX+1,
        max_seq_len = SEQ_LEN,
        dim = 2048,
        depth = 4,
        heads = 32,
        rotary_pos_emb = True,
        attn_flash = True
        )
    )

model.to(device)

#print(model)

''' PRECISION/OPTIMIZER/SCALER '''

dtype = torch.bfloat16

ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

scaler = torch.amp.GradScaler(device_type)

''' TRAINING '''

# Train the model

train_losses = []
val_losses = []

train_accs = []
val_accs = []

nsteps = 0

for ep in range(NUM_EPOCHS):
    print('Epoch #', ep)
    model.train()
    
    for i, x in enumerate(tqdm.tqdm(train_loader, desc='Training')):
        # Move data to device
        x = {k: v.to(device) for k, v in x.items()}

        # x['dtime'].shape = (BATCH_SIZE, SEQ_LEN)
        # x['pitch'].shape = (BATCH_SIZE, SEQ_LEN)
        # x['dur'].shape = (BATCH_SIZE, SEQ_LEN)
        
        optim.zero_grad()
        with ctx:
            loss, acc = model(x)  # Update your model to accept target separately
        scaler.scale(loss).backward()
        
        if i % PRINT_STATS_EVERY == 0:
            if(USE_TENSORBOARD):                
                tensorboard_summary.add_scalar("train_loss", loss.item(), nsteps)

            train_losses.append(loss.item())
            train_accs.append(acc.item())

        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optim)
        scaler.update()

        nsteps += 1

        # testing ALEX
        if True:
        #if i % VALIDATE_EVERY == 0:
            try:
                x = next(iter(val_loader)) # extract batches from test dataloader
            except StopIteration:
                val_loader_iter = iter(val_loader)
                x = next(iter(val_loader_iter)) # extract batches from test dataloader           
            model.eval()
            with torch.no_grad():
                # Move data to device
                x = {k: v.to(device) for k, v in x.items()}
                    
                with ctx:
                    # run the model
                    val_loss, val_acc = model(x)

                if(USE_TENSORBOARD):                
                    tensorboard_summary.add_scalar("val_loss", val_loss.item(), nsteps)
                    tensorboard_summary.add_scalar("val_acc", val_acc.item(), nsteps)

            model.train()

        '''if i % GENERATE_EVERY == 0:
            model.eval()

            inp = random.choice(get_batch(train_data, i, BATCH_SIZE))[:GENERATE_LENGTH]

            #print(inp)

            with ctx:
                sample = model.generate(inp[None, ...], GENERATE_LENGTH)

            #print(sample)

            data = sample.tolist()[0]

            #print('Sample INTs', data[:15])

            if len(data) != 0:

                song = data
                song_f = []

                time = 0
                dur = 1
                vel = 90
                pitch = 60
                channel = 0
                patch = 0

                patches = [0] * 16

                for m in song:

                    if 0 <= m < 128:
                        time += m * 32
                
                    elif 128 < m < 256:
                        dur = (m-128) * 32
                
                    elif 256 < m < 384:
                        pitch = (m-256)
                
                        song_f.append(['note', time, dur, 0, pitch, vel, 0])


                detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                                          output_signature = 'Monster Piano Transformer',
                                                                          output_file_name = './out/train_sample',
                                                                          track_name='Project Los Angeles',
                                                                          list_of_MIDI_patches=patches
                                                                          )

            #print('Done!')

            model.train()
        '''
        if i % SAVE_EVERY == 0:

            print('Saving model progress. Please wait...')
            print('model_checkpoint_' + str(nsteps) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth')

            fname = './save_models/model_checkpoint_' + str(ep) + '_eps_' + str(nsteps) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth'

            torch.save(model.state_dict(), fname)

            data = [train_losses, train_accs, val_losses, val_accs]

            TMIDIX.Tegridy_Any_Pickle_File_Writer(data, './save_models/losses_accs')

            #print('Done!')
