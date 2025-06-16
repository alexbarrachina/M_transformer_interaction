''' Automatic inferences, from 9 MIDI files as context, 
guided with buttons extracted from the same MIDI file 
By default, the context len is fixed to 120 notes. Once reached 120 notes, the first ones are discarded.
'''

import time

# Import Monster Piano Transformer as mpt
from model_loader import load_model
from midi_processors import midi_to_tokens, tokens_to_midi
#from monsterpianotransformer import generate
import torch
import TMIDIX
from params import load_hyperparameters

''' DEVICE '''
#device = torch.device('cpu')
device = torch.device('mps') 


''' MODEL '''
model = load_model(model_name='no_dtime_good_reference')
model.to(device)
model.eval()
load_hyperparameters(model_name='no_dtime_good_reference')

#print(model)

''' PARAMS '''
# Get sample seed MIDI path
#sample_midi_path = './seed_midis/Monster-Piano-Transformer-Piano-Seed-3.mid'
sample_midi_path = './samples/test'
output_midi_name = './out/continuator_test'
output_butt_midi_name = './out/continuator_test_b'
output_e_midi_name = './out/continuator_test_e'
CTX_LEN = 512 # num notes in context. 

for j in range(1, 8):  # generate 10 continuation files

  ''' BUILD CTX '''
  # Load seed MIDI
  input_tokens = midi_to_tokens(sample_midi_path+str(j)+'.midi') # tokens, without vel

  output_tokens = input_tokens.copy()

  dict_input_tokens, num_notes = TMIDIX.midi_tokens_to_dict(input_tokens) # vel already filtered out
  dict_output_tokens, num_notes = TMIDIX.midi_tokens_to_dict(output_tokens) # vel already filtered out
  TOTAL_GEN_LEN = num_notes

  print("num_notes",num_notes)
  
  # Build context tokens
  context = {
    'dtime': torch.tensor(dict_input_tokens['dtime'], dtype=torch.long).unsqueeze(0),
    'pitch': torch.tensor(dict_input_tokens['pitch'], dtype=torch.long).unsqueeze(0),
    'dur': torch.tensor(dict_input_tokens['dur'], dtype=torch.long).unsqueeze(0)
          }
  context = TMIDIX.to_device(context, device)
  
  with torch.inference_mode():
    e = model.encoder(context) # encoder output (batch, seq_len)
    b = model.real_to_discrete(e).squeeze(0) # generate buttons (batch, seq_len)
    e = e.squeeze(0)

  #timeStart = time.perf_counter()
  # generate pitches
  for i in range(0, TOTAL_GEN_LEN-1-CTX_LEN):
    
    context = {
      'dtime': torch.tensor(dict_input_tokens['dtime'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'pitch': torch.tensor(dict_input_tokens['pitch'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'dur': torch.tensor(dict_input_tokens['dur'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'button': torch.tensor(b[i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0)
    }

    context = TMIDIX.to_device(context, device)
  
    with torch.inference_mode():
        new_pitch_token = model.gen_pitch_token(context)
    dict_output_tokens['pitch'][i+CTX_LEN] = new_pitch_token
    print(new_pitch_token)

  #timeEnd = time.perf_counter()
  #print("t=", (timeEnd-timeStart) * 1000 / (TOTAL_GEN_LEN-CTX_LEN), "ms") # in miliseconds, mean time per note

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
                                                              timings_multiplier=1
                                                            )
