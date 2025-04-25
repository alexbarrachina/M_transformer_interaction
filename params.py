from typing import Final
import torch

########################################################

''' TESTING '''
TESTING = False
LIGHT_MODEL = True
LIGHT_DATASET = True
USE_TOPK = False

MODEL_NAME = 'autoenc_apr25_light_hi_losses_light_dataset'

''' MODEL '''
# constants
if LIGHT_MODEL:
    SEQ_LEN = 256 # orig 2048
    NUM_LAYERS = 4
    EMB_DIM = 512 # 2048
    NUM_HEADS = 32
    SAVE_EVERY = 10000
    BATCH_SIZE = 20 # 20 in esmuc
else:
    SEQ_LEN = 512 # orig 2048
    NUM_LAYERS = 4
    EMB_DIM = 2048 # 2048
    NUM_HEADS = 32
    SAVE_EVERY = 5000
    BATCH_SIZE = 28 # 10 upf decoder-only,  20 esmuc orignal decoder_only
 
if LIGHT_DATASET:
    DATA_SIZE = 1 # 1%
else:
    DATA_SIZE = 20 # 20%

''' TRAINING '''
# Taken from the paper
if torch.cuda.is_available(): 
    WORKERS = 4
    print("using CUDA")
else: # on macbook pro  
    WORKERS = 1
    print("using CPU/MPS")

if TESTING:
    BATCH_SIZE = 1
    SEQ_LEN = 16 

VALIDATE_EVERY  = 500
GENERATE_EVERY  = 500
GENERATE_LENGTH = 512
PRINT_STATS_EVERY = 50

NUM_EPOCHS = 12

LEARNING_RATE = 1e-4
GRAD_CLIP = 1.5

LOSS_MARGIN_MULTIPLIER:Final[float] = 0.1 #0.01 # 
LOSS_CONTOUR_MULTIPLIER:Final[float] = 0.1 # 0.1, but 0.01 original
LOSS_DEVIATE_MULTIPLIER:Final[float] = 0.1 # 0.01

USE_TENSORBOARD = True


''' VOCABULARY '''

#PIANO_NUM_KEYS:Final[int] = 88 
VOCAB_SIZE_PITCH:Final[int] = 128
#PIANO_LOWEST_KEY_MIDI_PITCH:Final[int] = 21
SOS:Final[int] = 127
PAD_IDX = 128 

NUM_BUTTONS:Final[int] = 12
SOS_BUTTONS:Final[int] = NUM_BUTTONS
VOCAB_SIZE_BUTTONS:Final[int] = NUM_BUTTONS + 1

RANGE_DTIME_SHIFT:Final[int] = 127
VOCAB_SIZE_DTIME:Final[int] = RANGE_DTIME_SHIFT + 1

RANGE_DUR_SHIFT:Final[int] = 127
VOCAB_SIZE_DUR:Final[int] = RANGE_DUR_SHIFT + 1

RANGE_VEL:Final[int] = 127 
VOCAB_SIZE_VEL:Final[int] = RANGE_VEL + 1

OFFSET_DTIME:Final[int] = 0
OFFSET_DUR:Final[int] = 128
OFFSET_PITCH:Final[int] = 256
OFFSET_VEL:Final[int] = 384

''' DATASET '''
# if using pickle files
dataset_train_path = './Training-Data/asigalov_train'
dataset_val_path = './Training-Data/asigalov_val'

if torch.cuda.is_available(): 
    # Path to your locally saved dataset
    local_dataset_path = "../Datasets/asigalov61___monster-piano"
else:
    local_dataset_path = "../../../Datasets/MIDI/asigalov61___monster-piano"
