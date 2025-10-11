#===================================================================================================
# Monster Genie inference_continuator_all.py Python module
# Automatic inferences, from 9 MIDI files as context,
# Useful for comparing compressed button structures
# 
# Copyright 2025 Alex Barrachina
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.'''
#===================================================================================================

import time
import torch

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_tokens, midi_tokens_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter

''' DEVICE '''
#device = torch.device('cpu')
device = torch.device('mps') 


''' MODEL '''
model_name = 'loss_norm_pos'
cfg = get_model_hparams(model_name)
model = load_model(model_name=model_name, cfg=cfg )
model.to(device)
model.eval()
#load_hyperparameters(model_name='no_dtime_good_reference')

#print(model)

''' PARAMS '''
# Get sample seed MIDI path
#sample_midi_path = './seed_midis/Monster-Piano-Transformer-Piano-Seed-3.mid'
sample_midi_path = './samples/test_mono'
output_midi_name = './out/test_mono'
output_butt_midi_name = './out/test_b'
output_e_midi_name = './out/test_e'
CTX_LEN = 512 # num notes in context. 

for j in range(1, 9):  # generate 10 continuation files

  ''' BUILD CTX '''
  # Load seed MIDI
  input_tokens = midi_to_tokens(sample_midi_path+str(j)+'.midi') # tokens, without vel

  output_tokens = input_tokens.copy()

  dict_input_tokens, num_notes = midi_tokens_to_dict(input_tokens) # vel already filtered out
  dict_output_tokens, num_notes = midi_tokens_to_dict(output_tokens) # vel already filtered out

  print("num_notes",num_notes)
  
  # Build context tokens
  context = {
    'dtime': torch.tensor(dict_input_tokens['dtime'], dtype=torch.long).unsqueeze(0),
    'pitch': torch.tensor(dict_input_tokens['pitch'], dtype=torch.long).unsqueeze(0),
    'dur': torch.tensor(dict_input_tokens['dur'], dtype=torch.long).unsqueeze(0)
          }
  context = to_device(context, device)
  
  with torch.inference_mode():
    e = model.encoder(context) # encoder output (batch, seq_len)
    b = model.real_to_discrete(e).squeeze(0) # generate buttons (batch, seq_len)
    e = e.squeeze(0)

  #timeStart = time.perf_counter()
  # generate pitches
  for i in range(0, num_notes-1-CTX_LEN):
    
    context = {
      'dtime': torch.tensor(dict_input_tokens['dtime'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'pitch': torch.tensor(dict_input_tokens['pitch'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'dur': torch.tensor(dict_input_tokens['dur'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'button': torch.tensor(b[i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0)
    }

    context = to_device(context, device)
  
    with torch.inference_mode():
        new_pitch_token = model.gen_pitch_token(context)
    dict_output_tokens['pitch'][i+CTX_LEN] = new_pitch_token
    print(new_pitch_token)

  #timeEnd = time.perf_counter()
  #print("t=", (timeEnd-timeStart) * 1000 / (TOTAL_GEN_LEN-CTX_LEN), "ms") # in miliseconds, mean time per note

  context = {
      'dtime': dict_output_tokens['dtime'][:num_notes],
      'pitch': dict_output_tokens['pitch'][:num_notes],
      'dur': dict_output_tokens['dur'][:num_notes],
    }

  # generate a midi file from generated pitches
  song_d = dict_to_song(context)
  detailed_stats = ms_SONG_to_MIDI_Converter(song_d, output_file_name = output_midi_name+str(j),
                                                            timings_multiplier=2
                                                            )

  # Convert buttons to values similar to pitch, just to represent the melodic contour
  b = torch.add(b, 60)

  context['pitch'] = b[:num_notes].tolist()
  song_d = dict_to_song(context)

  detailed_stats = ms_SONG_to_MIDI_Converter(song_d, output_file_name = output_butt_midi_name+str(j),
                                                            timings_multiplier=2
                                                            )
  re_int = e[:num_notes]
  re_int = torch.add(re_int, 1) # shift to [0, 2]
  re_int = torch.mul(re_int, 0.5) #  to [0, 1]
  re_int = torch.mul(re_int, 12) #  to [0, 12]
  re_int = torch.add(re_int, 60)
  context['pitch'] = re_int.int().tolist()
  # generate a midi file from buttons
  song_d = dict_to_song(context)
  detailed_stats = ms_SONG_to_MIDI_Converter(song_d,output_file_name = output_e_midi_name+str(j),  
                                                              timings_multiplier=2
                                                            )
