# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt
from model_loader import load_model
from midi_processors import midi_to_chords, tokens_to_midi
from monsterpianotransformer import texture_chords

# Load desired Monster Piano Transformer model
# There are several to choose from...
model = load_model('chords texturing - 3 epochs', device='cpu')

# Get sample seed MIDI path
#sample_midi_path = './seed_midis/Monster-Piano-Transformer-Piano-Seed-3.mid'
sample_midi_path = './samples/scott_chords.midi'
output_midi_name = './out/continuator_scott_textured'


# Convert MIDI to chords list
chords_list = midi_to_chords(sample_midi_path)

for i in range(9):

    # Generate seed MIDI continuation
    output_tokens = texture_chords(model, chords_list)

    # Save output batch # 0 to MIDI
    tokens_to_midi(output_tokens, output_midi_name=output_midi_name+str(i+1))
    print('Continuation', i+1, 'saved')