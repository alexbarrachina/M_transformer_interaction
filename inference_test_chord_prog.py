# Import Monster Piano Transformer as mpt
import monsterpianotransformer as mpt
from model_loader import load_model
from midi_processors import midi_to_tokens, tokens_to_midi
from monsterpianotransformer import chords_to_chords_tokens, generate

# Load desired Monster Piano Transformer model
# There are several to choose from...
model = load_model('chords progressions - 3 epochs', device='cpu')

# Prime chord(s) as a list of lists of semitones and/or pitches
prime_chords = [
                [0],
                [0, 2],
                [0, 2, 4],
                [60],
                [60, 62]
               ]

# Convert chords to chords tokens
chords_tokens = chords_to_chords_tokens(prime_chords)

output_midi_name = './out/chord_progressions'


for i in range(9):

    # Generate chord progression continuation
    output_tokens = generate(model, chords_tokens)

    # Save output batch # 0 to MIDI
    tokens_to_midi(output_tokens[0], output_midi_name=output_midi_name+str(i+1))
    print('Continuation', i+1, 'saved')