from typing import Final
import torch

########################################################

''' TESTING '''
MODEL_TYPE = 'super' # 'light', 'big' or 'super'
LIGHT_DATASET = False
TRAIN_SELECTION = True

TESTING = False
UPF = False

if torch.cuda.is_available(): 
    USE_LOGS = True
else:
    USE_LOGS = False

USE_TOPK = False

MODEL_NAME = 'mai24_no_dtime_ultra_resume'
DESCRIPTION = 'resume, 6layers, full model, giantsel dataset, original losses'

''' MODEL '''
# constants
if MODEL_TYPE == 'light':
    SEQ_LEN = 256 # orig 2048
    NUM_LAYERS = 4
    EMB_DIM = 512 # 2048
    NUM_HEADS = 32
    if TRAIN_SELECTION:
        SAVE_EVERY = 20 # in epochs 
    else:
        SAVE_EVERY = 25000 # 25000 orig in steps
    if UPF:
        BATCH_SIZE = 10 # 10 upf decoder-only, 
    else:
        BATCH_SIZE = 420 # 20 esmuc orignal decoder_only
    
elif MODEL_TYPE == 'big':
    SEQ_LEN = 1024 # orig 2048
    NUM_LAYERS = 4 # orig 4
    EMB_DIM = 2048 # 2048
    NUM_HEADS = 32
    if TRAIN_SELECTION:
        SAVE_EVERY = 20 # in epochs 
    else:
        SAVE_EVERY = 25000 #  25000 orig in steps
    if UPF:
        BATCH_SIZE = 10
    else:
        BATCH_SIZE = 20 # 10 upf decoder-only,  20 esmuc orignal 

elif MODEL_TYPE == 'super':
    SEQ_LEN = 1024 # orig 2048
    NUM_LAYERS = 6 # orig 4
    EMB_DIM = 2048 # 2048
    NUM_HEADS = 32
    if TRAIN_SELECTION:
        SAVE_EVERY = 40 # in epochs 
    else:
        SAVE_EVERY = 25000 # in epochs # 25000 orig in steps
    if UPF:
        BATCH_SIZE = 10
    else:
        BATCH_SIZE = 8 # 10 upf decoder-only,  20 esmuc orignal 

 
if LIGHT_DATASET:
    DATA_SIZE = 1 # 1%
else:
    DATA_SIZE = 20 # 20%

''' TRAINING '''
# Taken from the paper
if torch.cuda.is_available(): 
    NUM_WORKERS = 10
    #print("using CUDA")
else: # on macbook pro  
    NUM_WORKERS = 1
    print("using CPU/MPS")

if TESTING:
    BATCH_SIZE = 1
    SEQ_LEN = 16 
    USE_LOGS = True

VALIDATE_EVERY  = 500
GENERATE_EVERY  = 10000
GENERATE_LENGTH = 512
PRINT_STATS_EVERY = VALIDATE_EVERY

NUM_EPOCHS = 6000

LEARNING_RATE = 1e-4
GRAD_CLIP = 1.5

LOSS_MARGIN_MULTIPLIER:Final[float] = 0.01 #0.01 # 
LOSS_DEVIATE_MULTIPLIER:Final[float] = 0.01 # 0.01
LOSS_CONTOUR_MULTIPLIER:Final[float] = 0.01 # 0.1, but 0.01 original
LOSS_BUTTON_HELD_MULTIPLIER:Final[float] = 0.0 #0.01 #
LOSS_NORM_POS_MULTIPLIER:Final[float] = 0.0 #0.01 #
LOSS_PITCH_BUTTON_MULTIPLIER:Final[float] = 0.0 #0.01 # Multiplier for pitch-button correlation loss
LOSS_BUTTON_CONCENTRATION_MULTIPLIER:Final[float] = 0.0 #0.01 # Multiplier for button concentration loss

# % of every component in loss contour
LOSS_CONTOUR_PERC:Final[float] = 0. # 0.4, original
LOSS_MULTI_STEP_PERC:Final[float] = 1. # 0.3, original
LOSS_INTERVAL_PERC:Final[float] = 0. # 0.2, original
LOSS_SHAPE_PERC:Final[float] = 0. # 0.1, original

''' VOCABULARY '''

#PIANO_NUM_KEYS:Final[int] = 88 
VOCAB_SIZE_PITCH:Final[int] = 128
#PIANO_LOWEST_KEY_MIDI_PITCH:Final[int] = 21
SOS:Final[int] = 127
PAD_IDX = 128 

NUM_BUTTONS:Final[int] = 12 # 19 buttons (0-18) with central button at 9 # 12 original
BUTTON_CONCENTRATION_WINDOW_SIZE:Final[int] = 12
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

# Max time stretch for data augmentation (+- 5%)
DATA_AUGMENT_TIME_STRETCH_MAX:Final[float] = 0.05
# Max transpose for data augmentation (+- 6 semitones, tritone)
DATA_AUGMENT_TRANSPOSE_MAX:Final[int] = 6
# Define chord threshold (e.g., notes within 2 time units are considered part of same chord)
AUGMENT_CHORD_THRESHOLD:Final[int] = 2
  
''' DATASET '''
# if using pickle files
dataset_train_path = './Training-Data/asigalov_train'
dataset_val_path = './Training-Data/asigalov_val'

if torch.cuda.is_available(): 
    # Path to your locally saved dataset
    local_dataset_path = "../Datasets/asigalov61___monster-piano"
else:
    local_dataset_path = "../../../Datasets/MIDI/asigalov61___monster-piano"
