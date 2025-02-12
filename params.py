from typing import Final

########################################################

''' HYPERPARAMETERS '''
# Taken from the paper
ADAM_BETA_1:Final[float]             = 0.9
ADAM_BETA_2:Final[float]             = 0.98
ADAM_EPSILON :Final[float]           = 10e-9
LR_DEFAULT_START :Final[float]       = 1.0
SCHEDULER_WARMUP_STEPS:Final[int]  = 4000

''' CONSTANTS '''

SEQUENCE_START:Final[int] = 0
RANGE_NOTE_ON:Final[int] = 128
RANGE_NOTE_OFF:Final[int] = 128
RANGE_VEL:Final[int] = 32
RANGE_TIME_SHIFT:Final[int] = 100

TOKEN_END:Final[int]               = 256+512 # RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT
TOKEN_PAD:Final[int]               = TOKEN_END + 1

VOCAB_SIZE :Final[int]             = TOKEN_PAD + 1

PREPEND_ZEROS_WIDTH:Final[int]     = 4
