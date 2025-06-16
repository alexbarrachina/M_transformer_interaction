from typing import Final
import torch

from models import *
########################################################

''' TESTING '''
#MODEL_TYPE = 'light' # 'light', 'big' or 'super'
#LIGHT_DATASET = False
TRAIN_SELECTION = True
TESTING = False
UPF = False

if torch.cuda.is_available(): 
    USE_LOGS = True
else:
    USE_LOGS = False

USE_TOPK = False

''' DEFAULT MODEL PARAMETERS '''

MODEL_NAME = 'no_name'
DESCRIPTION = 'no description'
EMB_DIM = 2048
NUM_LAYERS = 4
NUM_HEADS = 32
SEQ_LEN = 2048
BATCH_SIZE = 20
SAVE_EVERY = 25000
DATA_SIZE = 20


# constants
'''if MODEL_TYPE == 'light':
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
        BATCH_SIZE = 240 # 20 esmuc orignal decoder_only
    
elif MODEL_TYPE == 'big':
    SEQ_LEN = 1024 # orig 2048
    NUM_LAYERS = 4 # orig 4
    EMB_DIM = 2048 # 2048
    NUM_HEADS = 32
    if TRAIN_SELECTION:
        SAVE_EVERY = 1 # in epochs 
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
'''


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
    DATA_SIZE = 1

VALIDATE_EVERY  = 500
GENERATE_EVERY  = 10000
GENERATE_LENGTH = 512
PRINT_STATS_EVERY = VALIDATE_EVERY

NUM_EPOCHS = 6000

LEARNING_RATE = 1e-4
GRAD_CLIP = 1.5

NUM_VAL_BATCHES_PER_STEP = 1 # 8 validation batches per step

LOSS_MARGIN_MULTIPLIER:Final[float] = 0.01 #0.01 # 
LOSS_DEVIATE_MULTIPLIER:Final[float] = 0.0 # 0.01
LOSS_CONTOUR_MULTIPLIER:Final[float] = 0.01 # 0.1, but 0.01 original
LOSS_BUTTON_HELD_MULTIPLIER:Final[float] = 0.0 #0.01 #
LOSS_NORM_POS_MULTIPLIER:Final[float] = 0.0 #0.01 #
LOSS_PITCH_BUTTON_MULTIPLIER:Final[float] = 0.0 #0.01 # Multiplier for pitch-button correlation loss
LOSS_BUTTON_CONCENTRATION_MULTIPLIER:Final[float] = 0.0 #0.01 # Multiplier for button concentration loss

# % of every component in loss contour
LOSS_CONTOUR_PERC:Final[float] = 1. # 0.4, original
LOSS_MULTI_STEP_PERC:Final[float] = 0. # 0.3, original
LOSS_INTERVAL_PERC:Final[float] = 0. # 0.2, original
LOSS_SHAPE_PERC:Final[float] = 0. # 0.1, original

''' VOCABULARY '''

#PIANO_NUM_KEYS:Final[int] = 88 
VOCAB_SIZE_PITCH:Final[int] = 128
#PIANO_LOWEST_KEY_MIDI_PITCH:Final[int] = 21
SOS:Final[int] = 127
PAD_IDX = 128 

NUM_BUTTONS:Final[int] = 8 # 19 buttons (0-18) with central button at 9 # 12 original
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
if TRAIN_SELECTION:
    DATASET_TRAIN_PATH = './Training-Data/giantMIDI_sel' 
    DATASET_VAL_PATH = './Training-Data/giantMIDI_sel_test'
    #DATASET_TRAIN_PATH = './Training-Data/asigalov_train'
    #DATASET_VAL_PATH = './Training-Data/asigalov_val'

# ASIGALOV DATASET path
if torch.cuda.is_available(): 
    # Path to your locally saved dataset
    local_dataset_path = "../Datasets/asigalov61___monster-piano"
else:
    local_dataset_path = "../../../Datasets/MIDI/asigalov61___monster-piano"


def load_hyperparameters(model_name='default',
               ):
    """
    Set hyperparameters for specific models in models.py.

    Parameters:
    model_name (str): The name of the model to load from MODELS_INFO dictionary. 
    Only modifies hyperparameters if model_name is in MODELS_PARAMETERS.
    """
    
    if model_name not in MODELS_PARAMETERS:
        print('=' * 70)
        print('Available models:')
        
        for n, d in MODELS_INFO.items():
            print('=' * 70)
            print('MODEL NAME:', n)
            print('-' * 70)
            print('MODEL INFO:', d)

        print('=' * 70)
        return []

    if model_name not in MODELS_PARAMETERS:
        MODEL_DESCRIPTION = MODELS_INFO[model_name]

    MODEL_NAME = model_name

    if 'num_buttons' in MODELS_PARAMETERS[model_name]:
        NUM_BUTTONS = MODELS_PARAMETERS[model_name]['num_buttons']

    if 'batch_size' in MODELS_PARAMETERS[model_name]:
        BATCH_SIZE = MODELS_PARAMETERS[model_name]['batch_size']

    if 'learning_rate' in MODELS_PARAMETERS[model_name]:
        LEARNING_RATE = MODELS_PARAMETERS[model_name]['learning_rate']

    if 'grad_clip' in MODELS_PARAMETERS[model_name]:
        GRAD_CLIP = MODELS_PARAMETERS[model_name]['grad_clip']
        
    if 'num_workers' in MODELS_PARAMETERS[model_name]:
        NUM_WORKERS = MODELS_PARAMETERS[model_name]['num_workers']

    if 'num_val_batches_per_step' in MODELS_PARAMETERS[model_name]:
        NUM_VAL_BATCHES_PER_STEP = MODELS_PARAMETERS[model_name]['num_val_batches_per_step']

    if 'loss_margin_multiplier' in MODELS_PARAMETERS[model_name]:
        LOSS_MARGIN_MULTIPLIER = MODELS_PARAMETERS[model_name]['loss_margin_multiplier']

    if 'loss_deviate_multiplier' in MODELS_PARAMETERS[model_name]:
        LOSS_DEVIATE_MULTIPLIER = MODELS_PARAMETERS[model_name]['loss_deviate_multiplier']

    if 'loss_contour_multiplier' in MODELS_PARAMETERS[model_name]:
        LOSS_CONTOUR_MULTIPLIER = MODELS_PARAMETERS[model_name]['loss_contour_multiplier']

    if 'loss_button_held_multiplier' in MODELS_PARAMETERS[model_name]:
        LOSS_BUTTON_HELD_MULTIPLIER = MODELS_PARAMETERS[model_name]['loss_button_held_multiplier']

    if 'loss_norm_pos_multiplier' in MODELS_PARAMETERS[model_name]:
        LOSS_NORM_POS_MULTIPLIER = MODELS_PARAMETERS[model_name]['loss_norm_pos_multiplier']

    if 'loss_pitch_button_multiplier' in MODELS_PARAMETERS[model_name]:
        LOSS_PITCH_BUTTON_MULTIPLIER = MODELS_PARAMETERS[model_name]['loss_pitch_button_multiplier']

    if 'loss_button_concentration_multiplier' in MODELS_PARAMETERS[model_name]:
        LOSS_BUTTON_CONCENTRATION_MULTIPLIER = MODELS_PARAMETERS[model_name]['loss_button_concentration_multiplier']
        
    if 'loss_contour_perc' in MODELS_PARAMETERS[model_name]:
        LOSS_CONTOUR_PERC = MODELS_PARAMETERS[model_name]['loss_contour_perc']

    if 'loss_multi_step_perc' in MODELS_PARAMETERS[model_name]:
        LOSS_MULTI_STEP_PERC = MODELS_PARAMETERS[model_name]['loss_multi_step_perc']

    if 'loss_interval_perc' in MODELS_PARAMETERS[model_name]:
        LOSS_INTERVAL_PERC = MODELS_PARAMETERS[model_name]['loss_interval_perc']
        
    if 'loss_shape_perc' in MODELS_PARAMETERS[model_name]:
        LOSS_SHAPE_PERC = MODELS_PARAMETERS[model_name]['loss_shape_perc']

    if 'loss_contour_perc' in MODELS_PARAMETERS[model_name]:
        LOSS_CONTOUR_PERC = MODELS_PARAMETERS[model_name]['loss_contour_perc']

    if 'loss_multi_step_perc' in MODELS_PARAMETERS[model_name]:
        LOSS_MULTI_STEP_PERC = MODELS_PARAMETERS[model_name]['loss_multi_step_perc']
    
    if 'save_every' in MODELS_PARAMETERS[model_name]:
        SAVE_EVERY = MODELS_PARAMETERS[model_name]['save_every']

    if 'data_size' in MODELS_PARAMETERS[model_name]: # % of dataset to use
        DATA_SIZE = MODELS_PARAMETERS[model_name]['data_size']
        
    if 'chord_threshold' in MODELS_PARAMETERS[model_name]:
        AUGMENT_CHORD_THRESHOLD = MODELS_PARAMETERS[model_name]['chord_threshold']

    if 'data_augment_time_stretch_max' in MODELS_PARAMETERS[model_name]:
        DATA_AUGMENT_TIME_STRETCH_MAX = MODELS_PARAMETERS[model_name]['data_augment_time_stretch_max']

    if 'data_augment_transpose_max' in MODELS_PARAMETERS[model_name]:
        DATA_AUGMENT_TRANSPOSE_MAX = MODELS_PARAMETERS[model_name]['data_augment_transpose_max']
        
    if 'num_epochs' in MODELS_PARAMETERS[model_name]:
        NUM_EPOCHS = MODELS_PARAMETERS[model_name]['num_epochs']

    if 'emb_dim' in MODELS_PARAMETERS[model_name]:
        EMB_DIM = MODELS_PARAMETERS[model_name]['emb_dim']

    if 'num_layers' in MODELS_PARAMETERS[model_name]:
        NUM_LAYERS = MODELS_PARAMETERS[model_name]['num_layers']

    if 'num_heads' in MODELS_PARAMETERS[model_name]:
        NUM_HEADS = MODELS_PARAMETERS[model_name]['num_heads']
        
        