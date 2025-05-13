# Import Monster Piano Transformer as mpt
from model_loader import load_encoder
from midi_processors import midi_to_tokens, tokens_to_midi
#from monsterpianotransformer import generate
import torch
import TMIDIX
import matplotlib.pyplot as plt

''' DEVICE '''
# Model precision option
if torch.backends.mps.is_available(): 
  device = torch.device("cpu")
else:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' MODEL '''

model = load_encoder(model_name='encoder_only', device='cpu')
model.to(device)


''' PARAMS '''
# Get sample seed MIDI path
#sample_midi_path = './seed_midis/Monster-Piano-Transformer-Piano-Seed-3.mid'
sample_midi_path = './samples/Bach_BWV_891.mid'
output_midi_name = './out/continuator_clairTester_to_end'
output_e_midi_name = './out/interactive_performance_e'
output_butt_midi_name = './out/buttons_bach'
CTX_LEN = 256 # num notes in context. tokens = CTX_LENGTH * 3
TOTAL_GEN_LEN = 4096 # num notes to generate


''' BUILD CTX '''
# Load seed MIDI
input_tokens = midi_to_tokens(sample_midi_path) # tokens, without vel

dict_input_tokens, num_notes = TMIDIX.midi_tokens_to_dict(input_tokens) # vel already filtered out

print("num_notes",num_notes)

# Build context tokens
#ctx_tokens = torch.tensor(dict_input_tokens['dtime'][:CTX_LEN], dtype=torch.long).to(device)
context = {
  'dtime': torch.tensor(dict_input_tokens['dtime'][:CTX_LEN], dtype=torch.long).unsqueeze(0),
  'pitch': torch.tensor(dict_input_tokens['pitch'][:CTX_LEN], dtype=torch.long).unsqueeze(0),
  'dur': torch.tensor(dict_input_tokens['dur'][:CTX_LEN], dtype=torch.long).unsqueeze(0)
        }
e = model.gen_buttons(context).squeeze(0)
b = model.real_to_discrete(e)

context = {
  'dtime': dict_input_tokens['dtime'][:CTX_LEN],
  'pitch': dict_input_tokens['pitch'][:CTX_LEN],
  'dur': dict_input_tokens['dur'][:CTX_LEN]
        }
# Convert buttons to values similar to pitch, just to represent the melodic contour
b = torch.add(b, 60)

context['pitch'] = b[:TOTAL_GEN_LEN].tolist()
song_d = TMIDIX.dict_to_song(context)

detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_d,
                                                                          output_signature = 'T GENIE',
                                                                          output_file_name = output_butt_midi_name,
                                                                          track_name='tgenie test',
                                                                          list_of_MIDI_patches=[0] * 16,
                                                                          timings_multiplier=2
                                                                          )

re_int = e[:TOTAL_GEN_LEN]
re_int = torch.add(re_int, 1) # shift to [0, 2]
re_int = torch.mul(re_int, 0.5) #  to [0, 1]
re_int = torch.mul(re_int, 12) #  to [0, 12]
re_int = torch.add(re_int, 60)
context['pitch'] = re_int.int().tolist()
# generate a midi file from buttons
song_d = TMIDIX.dict_to_song(context)
'''detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_d,
                                                                          output_signature = 'T GENIE',
                                                                          output_file_name = output_e_midi_name,
                                                                          track_name='tgenie test',
                                                                          list_of_MIDI_patches=[0] * 16,
                                                                          timings_multiplier=2
                                                                          )
res = b - re_int
'''