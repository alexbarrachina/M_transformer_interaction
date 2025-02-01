import torch
from model import *
from params import *

########################################################

''' ************* TESTER ************* 
************************************** '''

# Initialize configuration

batch_size = 2
seq_len = 512  # number of past notes

config = GPTConfig(
    block_size=seq_len, 
    n_layer=12,
    n_head=12,
    n_embd=768,
    dim_feedforward=3072,
    enable_rpr=True,
    er_len=1024,
)

''' TESTER FUNCTIONS '''

def decoder_test(): 
    # Initialize model
    model = Decoder(config)

    # Prepare example inputs
    past_tokens = {
        'dtime': torch.full((batch_size, seq_len), 1, dtype=torch.long),
        'vel': torch.full((batch_size, seq_len), 2, dtype=torch.long),
        'pitch': torch.full((batch_size, seq_len), 3, dtype=torch.long),
        'dur': torch.full((batch_size, seq_len), 4, dtype=torch.long),
        'but': torch.full((batch_size, seq_len), 5, dtype=torch.long)
    }

    current_dtime = torch.full((batch_size, 1), 1, dtype=torch.long)
    current_vel = torch.full((batch_size, 1), 2, dtype=torch.long)
    current_button = torch.full((batch_size, 1), 5, dtype=torch.long)

    # Forward pass
    pitch_logits = model(past_tokens, current_dtime, current_vel, current_button)
    print(pitch_logits)
    
def encoder_test(): 
    # Initialize model
    model = Encoder(config)

    # Prepare example inputs
    past_tokens = {
        'dtime': torch.full((batch_size, seq_len), 1, dtype=torch.long),
        'vel': torch.full((batch_size, seq_len), 2, dtype=torch.long),
        'pitch': torch.full((batch_size, seq_len), 3, dtype=torch.long),
        'dur': torch.full((batch_size, seq_len), 4, dtype=torch.long)
    }

    # Forward pass
    encoded_buttons = model(past_tokens)
    print(encoded_buttons)

def autoencoder_test(): 
    # Initialize model
    model = TransformerAutoencoder(config)

    # Prepare example inputs
    past_tokens = {
        'dtime': torch.full((batch_size, seq_len), 1, dtype=torch.long),
        'vel': torch.full((batch_size, seq_len), 2, dtype=torch.long),
        'pitch': torch.full((batch_size, seq_len), 3, dtype=torch.long),
        'dur': torch.full((batch_size, seq_len), 4, dtype=torch.long)
    }

    # Current tokens should not have sequence dimension
    current_dtime = torch.full((batch_size, 1), 1, dtype=torch.long)    
    current_vel = torch.full((batch_size, 1), 2, dtype=torch.long)
    current_button = torch.full((batch_size, 1), 5, dtype=torch.long)

    # Forward pass
    output, encoded_buttons = model(past_tokens, current_dtime, current_vel, current_button)
    print('output: ', output)
    print('encoded_buttons: ', encoded_buttons)

''' RUN TESTS '''
#decoder_test()
#encoder_test()
autoencoder_test()