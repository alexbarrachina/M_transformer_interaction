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

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import torch

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter

''' DEVICE '''
#device = torch.device('cpu')
device = torch.device('mps') 


''' MODEL '''
#model_name = 'AE_no_dtime_saturation_v1' # button saturation at extremes, more free pitch generation
#model_name = 'no_dtime_good_reference' # original Genie
model_name = 'good_ref_88buttons'
cfg = get_model_hparams(model_name)
model = load_model(model_name=model_name, cfg=cfg )
model.to(device)
model.eval()
#load_hyperparameters(model_name='no_dtime_good_reference')

#print(model)

''' PARAMS '''
# Get sample seed MIDI path
#sample_midi_path = './seed_midis/Monster-Piano-Transformer-Piano-Seed-3.mid'
sample_midi_path = '../samples/test1'
output_midi_name = './out/test1'
output_butt_midi_name = './out/test1_b'

CTX_LEN = 512 # num notes in context. 

if True:
  ''' BUILD CTX '''
  # Load seed MIDI
  dict_input_tokens, num_notes = midi_to_dict(sample_midi_path+'.midi') # tokens

  dict_output_tokens = dict_input_tokens.copy()

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

 
  # Convert buttons to values similar to pitch, just to represent the melodic contour
  b = torch.add(b, 60)

  context['pitch'] = b[:num_notes].tolist()
  # Around line 81, ensure ALL fields are converted from tensors to Python types
  context['dtime'] = context['dtime'].squeeze(0).tolist()
  context['dur'] = context['dur'].squeeze(0).tolist()  
  song_d = dict_to_song(context)

  detailed_stats = ms_SONG_to_MIDI_Converter(song_d, output_file_name = output_butt_midi_name,
                                                            timings_multiplier=2
                                                            )
 