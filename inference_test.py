# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt
from model_loader import load_model
from sample_midis import get_sample_midi_files
from midi_processors import midi_to_tokens, tokens_to_midi
from monsterpianotransformer import generate
import torch
# Model precision option
if torch.backends.mps.is_available(): 
  device = torch.device("cpu")
else:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

full_path_to_model_checkpoint = './save_models/150_epochs_overfit_model_checkpoint_233604_steps_0.0083_loss_0.9987_acc.pth'
# Load desired Monster Piano Transformer model
# There are several to choose from...
model = load_model(device='cpu')

# Load and adjust state dict
#state_dict = torch.load(full_path_to_model_checkpoint, map_location=device)
#new_state_dict = {}
#for k, v in state_dict.items():
#    new_state_dict[f"_orig_mod.{k}"] = v

# Load adjusted state dict
#model.load_state_dict(new_state_dict)
model.to(device)

# Get sample seed MIDI path
#sample_midi_path = './seed_midis/Monster-Piano-Transformer-Piano-Seed-3.mid'
sample_midi_path = './samples/clairTester.midi'
output_midi_name = './out/continuator_clairTesterc'

# Load seed MIDI
input_tokens = midi_to_tokens(sample_midi_path)

for i in range(29):

    # Generate seed MIDI continuation
    if isinstance(model, list):
        model = model[0]
    output_tokens = generate(model, input_tokens, num_gen_tokens=600, return_prime=True)

    # Save output batch # 0 to MIDI
    tokens_to_midi(output_tokens[0], output_midi_name=output_midi_name+str(i+1))
    print('Continuation', i+1, 'saved')