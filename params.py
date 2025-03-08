from typing import Final
import torch

########################################################

''' HYPERPARAMETERS '''
# Taken from the paper
if torch.cuda.is_available(): # on pepinón
    BATCH_SIZE = 3 #2 # Change this to your specs (4 batches per 48GB)
    WORKERS = 4
else: # on macbook pro
    BATCH_SIZE = 1  
    WORKERS = 1
    
''' CONSTANTS '''

#PIANO_NUM_KEYS:Final[int] = 88 
VOCAB_SIZE_PITCH:Final[int] = 128
#PIANO_LOWEST_KEY_MIDI_PITCH:Final[int] = 21
SOS:Final[int] = 127

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

LOSS_MARGIN_MULTIPLIER:Final[float] = 0.01 # TODO decide values
LOSS_CONTOUR_MULTIPLIER:Final[float] = 0.01
LOSS_DEVIATE_MULTIPLIER:Final[float] = 0.01

# TODO solve for separate vocab sizes. Provisionally the maxium vocab_size
#TOKEN_END:Final[int]  = VOCAB_SIZE_DTIME
#TOKEN_PAD:Final[int]  = SOS_DTIME

#SEQUENCE_START:Final[int] = 0
#RANGE_NOTE_ON:Final[int] = 128
#RANGE_NOTE_OFF:Final[int] = 128
#RANGE_VEL:Final[int] = 32
#RANGE_TIME_SHIFT:Final[int] = 100

#TOKEN_PAD:Final[int]               = TOKEN_END + 1

#PREPEND_ZEROS_WIDTH:Final[int]     = 4