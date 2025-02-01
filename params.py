from typing import Final
import torch

########################################################

''' VOCABULARY '''

# Offsets create non-overlapping ranges for each token type
# DTIME 0-127
DUR_OFF = 128
# DUR 128-255
PITCH_OFF = 256
# PITCH 256-383
VEL_OFF = 384
# VEL 384-511

TOKEN_END = 511 # not using explicit end tokens in the dataset. It works in fixed length sequences
TOKEN_PAD = TOKEN_END # fill sequences to a fixed length. Ignored in loss computation. The model should not learn to predict it
VOCAB_SIZE = TOKEN_END+1

''' HYPERPARAMETERS '''

SEQ_LEN = 1024 # block_size 2048
DIC_SIZE = VOCAB_SIZE # vocab_size 513
BATCH_SIZE = 8 # Change this to your specs (4 batches per 48GB)
DIM_FEEDFORWARD = 2048 # Size of the feedforward linear layer after attention
N_LAYERS = 24 # 24 Number of layers
N_HEADS = 8 # Number of attention heads
N_EMBED = 1024 # Number of embeddings
EPOCHS = 5 # Number of epochs
if torch.cuda.is_available():
    NUM_WORKERS = 27 # Number of workers    
else:
    NUM_WORKERS = 0 # Number of workers

# Taken from the paper
ADAM_BETA_1:Final[float]             = 0.9
ADAM_BETA_2:Final[float]             = 0.98
ADAM_EPSILON :Final[float]           = 10e-9
LR_DEFAULT_START :Final[float]       = 0.0001 # 1.0 
SCHEDULER_WARMUP_STEPS:Final[int]  = 4000
ADAM_WEIGHT_DECAY:Final[float] = 0.01

DROPOUT:Final[float] = 0.1

LOG_FREQ = 800 # originally 200
SAVE_FREQ = 40000 # originally 4000

''' CONSTANTS '''

SEQUENCE_START:Final[int] = 0
RANGE_NOTE_ON:Final[int] = 128
RANGE_NOTE_OFF:Final[int] = 128
RANGE_VEL:Final[int] = 32
RANGE_TIME_SHIFT:Final[int] = 100

# TOKEN_END:Final[int]               = 256+512 # RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT
#TOKEN_PAD:Final[int]               = TOKEN_END + 1

#VOCAB_SIZE :Final[int]             = TOKEN_PAD + 1

#PREPEND_ZEROS_WIDTH:Final[int]     = 4


