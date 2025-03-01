# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt
from model_loader import load_model
from midi_processors import midi_to_tokens, tokens_to_midi
from monsterpianotransformer import inpaint_pitches

# Load desired Monster Piano Transformer model
# There are several to choose from...
model = load_model(device='cpu')

# Get sample seed MIDI path
#sample_midi_path = './seed_midis/Monster-Piano-Transformer-Piano-Seed-3.mid'
sample_midi_path = './samples/clairTester.midi'
output_midi_name = './out/continuator_CLAIR_pitches'

# Load seed MIDI
input_tokens = midi_to_tokens(sample_midi_path)

for i in range(9):

    # Generate seed MIDI continuation
    output_tokens = inpaint_pitches(model, input_tokens)

    # Save output batch # 0 to MIDI
    tokens_to_midi(output_tokens, output_midi_name=output_midi_name+str(i+1))
    print('Continuation', i+1, 'saved')