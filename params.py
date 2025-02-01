from typing import Final
import torch

########################################################

''' HYPERPARAMETERS '''
# Taken from the paper
ADAM_BETA_1:Final[float]             = 0.9
ADAM_BETA_2:Final[float]             = 0.98
ADAM_EPSILON :Final[float]           = 10e-9
LR_DEFAULT_START :Final[float]       = 1.0
SCHEDULER_WARMUP_STEPS:Final[int]  = 4000

LOSS_MARGIN_MULTIPLIER:Final[float] = 0.01 # TODO decide values
LOSS_CONTOUR_MULTIPLIER:Final[float] = 0.01
LOSS_DEVIATE_MULTIPLIER:Final[float] = 0.01

SEQ_LEN = 1024 # 2048 # block_size
#The block_size parameter for position embeddings should match the sequence length of your input tokens (T). 
# Since you're using summed embeddings, the sequence length remains the same as the original note sequence length 
# (not multiplied by the number of features).
BLOCK_SIZE = SEQ_LEN
#DIC_SIZE = 524 # vocab_size + 12 (12 buttons)
DIM_FEEDFORWARD = 2048 # Size of the feedforward linear layer after attention
N_LAYERS = 24 # Number of layers
N_HEADS = 8 # Number of attention heads
N_EMBED = 1024 # Number of embeddings
EPOCHS = 5 # Number of epochs
if torch.cuda.is_available(): # on pepinón
    NUM_WORKERS = 20 # Number of workers  
    BATCH_SIZE = 3 #2 # Change this to your specs (4 batches per 48GB)
else: # on macbook pro
    NUM_WORKERS = 0 # Number of workers,  
    BATCH_SIZE = 1  

''' CONSTANTS '''

PIANO_NUM_KEYS:Final[int] = 88 
#SOS_PITCH:Final[int] = PIANO_NUM_KEYS
VOCAB_SIZE_PITCH:Final[int] = PIANO_NUM_KEYS + 1
PIANO_LOWEST_KEY_MIDI_PITCH:Final[int] = 21

NUM_BUTTONS:Final[int] = 12
#SOS_BUTTONS:Final[int] = NUM_BUTTONS
VOCAB_SIZE_BUTTONS:Final[int] = NUM_BUTTONS + 1

RANGE_DTIME_SHIFT:Final[int] = 127
SOS_DTIME:Final[int] = RANGE_DTIME_SHIFT
VOCAB_SIZE_DTIME:Final[int] = RANGE_DTIME_SHIFT + 1

RANGE_DUR_SHIFT:Final[int] = 127
SOS_DUR:Final[int] = RANGE_DUR_SHIFT
VOCAB_SIZE_DUR:Final[int] = RANGE_DUR_SHIFT + 1

RANGE_VEL:Final[int] = 32 
SOS_VEL:Final[int] = RANGE_VEL
VOCAB_SIZE_VEL:Final[int] = RANGE_VEL + 1

# TODO solve for separate vocab sizes. Provisionally the maxium vocab_size
TOKEN_END:Final[int]  = VOCAB_SIZE_DTIME
TOKEN_PAD:Final[int]  = SOS_DTIME

#SEQUENCE_START:Final[int] = 0
#RANGE_NOTE_ON:Final[int] = 128
#RANGE_NOTE_OFF:Final[int] = 128
#RANGE_VEL:Final[int] = 32
#RANGE_TIME_SHIFT:Final[int] = 100

#TOKEN_PAD:Final[int]               = TOKEN_END + 1

#PREPEND_ZEROS_WIDTH:Final[int]     = 4