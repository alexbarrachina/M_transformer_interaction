
from model import *
from midiUtils import *
import numpy as np

"""# SETTINGS """
# Play with the settings to get different results
full_path_to_model_checkpoint = "./SaveModel/giantMIDI_Sel_latency_412000_steps_0.0564_loss.pth" 
midi_file = './Samples/clair_train_full.midi'
nameOut = './Out/generate_full_buttons'

num_primer = 128 # min:32, max:256 num of notes in primer
num_of_tokens_to_generate = 1024 # num of notes to generate    
temperature = 1.0 # min:0.1, max:1
show_stats = True 


def get_buttons_from_encoder(inputs, target_seq_length=1024, num_batches=1):
    

    # re-structure input data
    feature_data = {
    'dtime': inputs[0::4],  # Every 4th token starting at index 0
    'vel': inputs[1::4],    # Every 4th token starting at index 1
    'pitch': inputs[2::4],  # Every 4th token starting at index 2
    'dur': inputs[3::4]     # Every 4th token starting at index 3
    }
    
    num_primer = min(len(inputs), target_seq_length) # primer shape (512)
    x = {
            'dtime': feature_data['dtime'][:num_primer].long(),
            'vel': feature_data['vel'][:num_primer].long(),
            'pitch': feature_data['pitch'][:num_primer].long(),
            'dur': feature_data['dur'][:num_primer].long()
        }

    buttons_seq = torch.full((num_batches,target_seq_length), TOKEN_PAD, dtype=torch.long, device=device) # shape(2,1024)

    buttons_seq[..., :num_primer] = torch.tensor(inputs, dtype=torch.long, device=device) # shape(2, 512)

    b = model.generate_buttons(buttons_seq[..., :num_primer])
    return b

def substitute_pitch_with_buttons(inputs, buttons, target_seq_length=1024, num_batches=1):
    """
    Substitutes pitch values in MIDI messages with button information.
    
    Args:
        inputs (list): List of MIDI messages where every 4 elements represent [dtime, velocity, pitch, duration]
        buttons (torch.Tensor): Tensor of button information with shape (num_batches, target_seq_length)
        target_seq_length (int, optional): Length of the target sequence. Defaults to 1024.
        num_batches (int, optional): Number of batches. Defaults to 1.
    
    Returns:
        list: Modified MIDI messages with pitch values replaced by button information
    """
    # Convert inputs to numpy for easier manipulation
    inputs = np.array(inputs)
    
    # Ensure inputs length is divisible by 4
    assert len(inputs) % 4 == 0, "Input length must be divisible by 4 (dtime, velocity, pitch, duration)"
    
    # Calculate number of notes
    num_notes = len(inputs) // 4
    
    # Create output array
    output = inputs.copy()
    
    # Extract pitch indices (every 3rd element starting from index 2)
    pitch_indices = np.arange(2, len(inputs), 4)
    
    # Get button values from the first batch (assuming single batch for inference)
    button_values = buttons[0, :num_notes].cpu().numpy()
    
    # Replace pitch values with button values
    output[pitch_indices] = button_values[:len(pitch_indices)]
    
    return output.tolist()



"""# SET MODEL PRECISION""" 
# Model precision option
if torch.backends.mps.is_available(): 
  model_precision = "float32" # @param ["bfloat16", "float16", "float32"]
  device = torch.device("mps")
else:
  model_precision = "bfloat16"
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# bfloat16 == Third precision/triple speed (if supported, otherwise the model will default to float16)
# float16 == Half precision/double speed
# float32 == Full precision/normal speed

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


"""# (LOAD MODEL)"""

print('=' * 70)
print('Loading GIGA-Piano XL model...')
config = GPTConfig(
                   block_size=BLOCK_SIZE, # block_size
                   dim_feedforward=DIM_FEEDFORWARD, # 2048 Size of the feedforward linear layer after attention
                   n_layer=N_LAYERS, 
                   n_head=N_HEADS, 
                   n_embd=N_EMBED, # 1024 Number of embeddings
                   enable_rpr=True,
                   er_len=SEQ_LEN)


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerAutoencoder(config)

model.load_state_dict(torch.load(full_path_to_model_checkpoint, map_location=device))
model.to(device)
model.eval()

"""# GET PRIMER FROM MIDI FILE """

inputs = get_midifile_data(midi_file)

"""# GET BUTTONS FROM ENCODER """
# Example usage
inputs = [
    10, 64, 60, 120,  # First note: dtime=10, velocity=64, pitch=60, duration=120
    20, 72, 64, 100,  # Second note: dtime=20, velocity=72, pitch=64, duration=100
    15, 68, 62, 110   # Third note: dtime=15, velocity=68, pitch=62, duration=110
]


buttons = get_buttons_from_encoder(feature_data)


"""# INFERENCE  """

# TODO: current button, vel, dtime
# Assuming buttons is a tensor of shape (1, 1024) containing button information
modified_messages = substitute_pitch_with_buttons(inputs, buttons)

generate_midifile_from_list(modified_messages, nameOut) 

