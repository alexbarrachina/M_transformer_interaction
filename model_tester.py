
import torch
from model import *
from params import *

########################################################

''' ************* TESTER ************* 
************************************** '''

# Initialize configuration

batch_size = 2
seq_len = 4  # number of past notes

config = GPTConfig(
    vocab_size=512,
    block_size=seq_len,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dim_feedforward=3072,
    enable_rpr=True,
    er_len=1024,
)
'''config = GPTConfig(512,
                  2048,
                  dim_feedforward=2048,
                  n_layer=24,
                  n_head=8,
                  n_embd=1024,
                  enable_rpr=True,
                  er_len=2048)'''

''' TESTER FUNCTIONS '''

def decoder_test(): 
    # Initialize model
    model = GPT(config)

    # Prepare example inputs
    current_seq = torch.full((batch_size, seq_len), 1, dtype=torch.long)

    # Forward pass
    pitch_logits = model(current_seq)
    print(pitch_logits)


''' RUN TESTS '''
decoder_test()