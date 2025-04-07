# Import Monster Piano Transformer as mpt
from model_loader import load_model
from midi_processors import midi_to_tokens, tokens_to_midi
#from monsterpianotransformer import generate
import torch
import TMIDIX

''' DEVICE '''
# Model precision option
if torch.backends.mps.is_available(): 
  device = torch.device("cpu")
else:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' MODEL '''

model = load_model(model_name='light__apr4_autoencoder', device='cpu')
model.to(device)
#print(model)

''' PARAMS '''
# Get sample seed MIDI path
#sample_midi_path = './seed_midis/Monster-Piano-Transformer-Piano-Seed-3.mid'
sample_midi_path = './samples/clairTester_to_end.midi'
output_midi_name = './out/continuator_clairTester_to_end'
output_butt_midi_name = './out/continuator_clairTester_buttons'
output_e_midi_name = './out/continuator_clairTester_e'
CTX_LEN = 120 # num notes in context. tokens = CTX_LENGTH * 3
TOTAL_GEN_LEN = 500 # num notes to generate


''' BUILD CTX '''
# Load seed MIDI
input_tokens = midi_to_tokens(sample_midi_path) # tokens, without vel

output_tokens = input_tokens.copy()

dict_input_tokens, num_notes = TMIDIX.midi_tokens_to_dict(input_tokens) # vel already filtered out
dict_output_tokens, num_notes = TMIDIX.midi_tokens_to_dict(output_tokens) # vel already filtered out

print("num_notes",num_notes)

''' GENERATE 10 files'''

for j in range(0, 10):  # generate 10 continuation files
  # Build context tokens
  context = {
    'dtime': torch.tensor(dict_input_tokens['dtime'], dtype=torch.long).unsqueeze(0),
    'pitch': torch.tensor(dict_input_tokens['pitch'], dtype=torch.long).unsqueeze(0),
    'dur': torch.tensor(dict_input_tokens['dur'], dtype=torch.long).unsqueeze(0)
          }
  #b = model.gen_buttons(context).squeeze(0) # generate buttons, continuous values
  #e = model.encoder(context).squeeze(0)
  e = model.encoder(context) # encoder output (batch, seq_len)
  b = model.real_to_discrete(e).squeeze(0) # generate buttons (batch, seq_len)
  e = e.squeeze(0)

  # generate pitches
  for i in range(0, TOTAL_GEN_LEN-1-CTX_LEN):

    context = {
      'dtime': torch.tensor(dict_input_tokens['dtime'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'pitch': torch.tensor(dict_input_tokens['pitch'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'dur': torch.tensor(dict_input_tokens['dur'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'button': torch.tensor(b[i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0)
    }

    new_pitch_token = model.gen_pitch_token(context)
    dict_output_tokens['pitch'][i+CTX_LEN] = new_pitch_token
    print(new_pitch_token)


  context = {
      'dtime': dict_output_tokens['dtime'][:TOTAL_GEN_LEN],
      'pitch': dict_output_tokens['pitch'][:TOTAL_GEN_LEN],
      'dur': dict_output_tokens['dur'][:TOTAL_GEN_LEN],
    }

  # generate a midi file from generated pitches
  song_d = TMIDIX.dict_to_song(context)
  detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_d, output_file_name = output_midi_name+str(j),
                                                            timings_multiplier=1
                                                            )

  # Convert buttons to values similar to pitch, just to represent the melodic contour
  b = torch.add(b, 60)

  context['pitch'] = b[:TOTAL_GEN_LEN].tolist()
  song_d = TMIDIX.dict_to_song(context)

  detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_d, output_file_name = output_butt_midi_name+str(j),
                                                            timings_multiplier=1
                                                            )
  re_int = e[:TOTAL_GEN_LEN]
  re_int = torch.add(re_int, 1) # shift to [0, 2]
  re_int = torch.mul(re_int, 0.5) #  to [0, 1]
  re_int = torch.mul(re_int, 12) #  to [0, 12]
  re_int = torch.add(re_int, 60)
  context['pitch'] = re_int.int().tolist()
  # generate a midi file from buttons
  song_d = TMIDIX.dict_to_song(context)
  detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_d,output_file_name = output_e_midi_name+str(j),  
                                                              timings_multiplier=2
                                                            )
