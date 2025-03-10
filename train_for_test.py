import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import pickle
import random
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

#from torchsummary import summary
#from sklearn import metrics

from datasets import load_dataset
import TMIDIX

from x_transformer_1_23_2 import *

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(False)

import random

#monster_piano = load_dataset('asigalov61/Monster-Piano') # original
monster_piano = load_dataset('asigalov61/Monster-Piano', split='train[:1%]') # 
#monster_piano_val = load_dataset('asigalov61/Monster-Piano', split='train[99%:100%]') # 
print('data loaded')

SEQ_LEN = 1024
SEQ_OFFSET = 512
PAD_IDX = 384 # Model pad index

#==========================================================================

print('=' * 70)
print('Loading data files...')
print('Please wait...')
print('=' * 70)

train_data = set()
val_data = set()

chunks_counter = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_type='cuda' if torch.cuda.is_available() else 'cpu'

#for entry in tqdm.tqdm(monster_piano['train']):
for entry in tqdm.tqdm(monster_piano): # with a split parameter, you get direct access to that split's data

    score = entry['midi_score']
    score = [t for t in score if t < 384] # why filter values < 384? maybe no velocity? PAD_IDX is 384

    if 0 <= max(score) < PAD_IDX: # final data integrity check

        for i in range(0, len(score), SEQ_LEN-SEQ_OFFSET):
            
            chunk = score[i:i+SEQ_LEN+1]

            chunks_counter += 1

            if len(chunk) < SEQ_LEN+1: 
                # pad the chunk with PAD_IDX if it's less than SEQ_LEN+1
                chunk += [PAD_IDX] * (SEQ_LEN+1 - len(chunk))

            train_data.add(tuple(chunk))

    else:
        print('Bad data!!!')


#==========================================================================

train_data = list(train_data)

#==========================================================================

print('Done!')
print('=' * 70)
print('Total number of main chunks:', chunks_counter)
print('All data is good:', len(max(train_data, key=len)) == len(min(train_data, key=len)))
print('=' * 70)
print('Randomizing train data...')
random.shuffle(train_data)
print('Done!')
print('=' * 70)
print('Total length of train data:', len(train_data))
print('=' * 70)

''' SETUP MODEL '''

# constants

VALIDATE_EVERY  = 500
SAVE_EVERY = 2500
GENERATE_EVERY  = 1#000
GENERATE_LENGTH = 512
PRINT_STATS_EVERY = 50

NUM_EPOCHS = 10

#BATCH_SIZE = 116 # original
BATCH_SIZE = 1 

LEARNING_RATE = 1e-4
GRAD_CLIP = 1.5

# instantiate the model

model = TransformerAutoencoder(
    num_tokens = PAD_IDX+1,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(dim = 2048,
                          depth = 4,
                          heads = 32,
                          rotary_pos_emb = True,
                          attn_flash = True
                         )
    )

model = AutoregressiveWrapper(model, ignore_index = PAD_IDX, pad_value=PAD_IDX)

model.to(device)

print('Done!')

print(model)
#summary(model)


# Dataloader

def get_train_data_batch(tdata, index, seq_len, batch_size, pad_idx):

    batch = tdata[(index*batch_size):(index*batch_size)+batch_size]

    return torch.LongTensor(batch).to(device)

# precision/optimizer/scaler

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

        print('=' * 70)
        print('Randomizing train data...')
        random.shuffle(train_data)
        print('=' * 70)

        print('=' * 70)
        print('Epoch #', ep)
        print('=' * 70)

        NUM_BATCHES = len(train_data) // BATCH_SIZE

        model.train()

        for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='Training'):

            optim.zero_grad()

            with ctx:
                x = get_train_data_batch(train_data, i, SEQ_LEN, BATCH_SIZE, PAD_IDX)
                loss, acc = model(x)
            scaler.scale(loss).backward()

            if i % PRINT_STATS_EVERY == 0:
                if(USE_TENSORBOARD):                
                    tensorboard_summary.add_scalar("train_loss", loss.item(), nsteps)

                #print(f'Training loss: {loss.item()}')
                #print(f'Training acc: {acc.item()}')

            train_losses.append(loss.item())
            train_accs.append(acc.item())

            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optim)
            scaler.update()

            nsteps += 1

            if i % GENERATE_EVERY == 0:
                model.eval()

                inp = random.choice(get_train_data_batch(train_data, i, SEQ_LEN, BATCH_SIZE, PAD_IDX))[:GENERATE_LENGTH]

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

            if i % SAVE_EVERY == 0:

                print('Saving model progress. Please wait...')
                print('model_checkpoint_' + str(nsteps) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth')

                fname = './save_models/model_checkpoint_' + str(ep) + '_eps_' + str(nsteps) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth'

                torch.save(model.state_dict(), fname)

                data = [train_losses, train_accs, val_losses, val_accs]

                TMIDIX.Tegridy_Any_Pickle_File_Writer(data, './save_models/losses_accs')

                #print('Done!')
