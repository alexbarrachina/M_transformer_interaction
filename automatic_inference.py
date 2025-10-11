#==================================================================================================
# Monster Genie inference_continuator.py Python module
# Automatic inference, from a MIDI file as context,
# guided with buttons extracted from the same MIDI file
# By default, the context len is fixed to 120 notes. Once reached 120 notes, the first ones are discarded.
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

temperature = 1.0

''' DEVICE '''
#device = torch.device('cpu')
device = torch.device('mps') 


''' MODEL '''
model_name = 'no_dtime_good_reference'
cfg = get_model_hparams(model_name)
model = load_model(model_name=model_name, cfg=cfg )
model.to(device)
model.eval()

#print(model)

''' PARAMS '''
# Get sample seed MIDI path
sample_midi_path = './samples/clairTester_to_end_monophonic.midi'
output_midi_name = './out/continuator_clairTester_to_end'
output_butt_midi_name = './out/continuator_clairTester_to_end_buttons'
output_e_midi_name = './out/continuator_clairTester_to_end_e'
CTX_LEN = 218 # num notes in context. 
#TOTAL_GEN_LEN = 500 # num notes to generate


''' BUILD CTX '''
# Load seed MIDI
input_tokens = midi_to_tokens(sample_midi_path) # tokens, without vel

output_tokens = input_tokens.copy()

dict_input_tokens, num_notes = midi_tokens_to_dict(input_tokens) # vel already filtered out
dict_output_tokens, num_notes = midi_tokens_to_dict(output_tokens) # vel already filtered out

print("num_notes",num_notes)

''' GENERATE 10 files'''

for j in range(0, 10):  # generate 10 continuation files
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

  timeStart = time.perf_counter()
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
        new_pitch_token = model.gen_pitch_token(context, temperature=temperature)
    # update output tokens generated pitch, original dtime, original dur
    dict_output_tokens['pitch'][i] = new_pitch_token
    dict_output_tokens['dtime'][i] = dict_input_tokens['dtime'][i+CTX_LEN+1]
    dict_output_tokens['dur'][i] = dict_input_tokens['dur'][i+CTX_LEN+1]
    print(new_pitch_token)

  #timeEnd = time.perf_counter()
  #print("t=", (timeEnd-timeStart) * 1000 / i, "ms") # in miliseconds, promig

  context = {
      'dtime': dict_output_tokens['dtime'][:num_notes-CTX_LEN],
      'pitch': dict_output_tokens['pitch'][:num_notes-CTX_LEN],
      'dur': dict_output_tokens['dur'][:num_notes-CTX_LEN],
    }

  # generate a midi file from generated pitches
  song_d = dict_to_song(context)
  detailed_stats = ms_SONG_to_MIDI_Converter(song_d, output_file_name = output_midi_name+str(j),
                                                            timings_multiplier=2
                                                            )

  
  # Convert buttons to values similar to pitch, just to represent the melodic contour
  b = torch.add(b, 60)

  context['pitch'] = b[:num_notes-CTX_LEN].tolist()
  song_d = dict_to_song(context)

  detailed_stats = ms_SONG_to_MIDI_Converter(song_d, output_file_name = output_butt_midi_name+str(j),
                                                            timings_multiplier=2
                                                            )
  '''

  re_int = e[:num_notes-CTX_LEN]
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
  '''