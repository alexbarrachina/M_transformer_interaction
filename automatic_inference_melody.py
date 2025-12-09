#==================================================================================================
# Monster Genie automatic_inference_melody.py Python module
# Automatic inference, from a MIDI file as context,
# guided with arrows extracted from the same MIDI file
# By default, the context len is fixed to 218 notes. Once reached 218 notes, the first ones are discarded.
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
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter

temperature = 1.0
  
''' DEVICE '''
#device = torch.device('cpu')
device = torch.device('mps') 


''' MODEL '''
model_name = 'melody_arrow_v4'
#model_name = 'no_dtime_good_reference'
cfg = get_model_hparams(model_name)
model = load_model(model_name=model_name, cfg=cfg )
model.to(device)
model.eval()

#print(model)

''' PARAMS '''
# Get sample seed MIDI path
sample_midi_path = './samples/clairTester_to_end_monophonic.midi'
output_midi_name = './out/continuator_clairTester_to_end'
input_arrows_midi_name = './out/continuator_clairTester_input_arrows'
output_arrows_midi_name = './out/continuator_clairTester_output_arrows'
CTX_LEN = 218 # num notes in context. 
#TOTAL_GEN_LEN = 500 # num notes to generate


''' BUILD CTX '''
# Load seed MIDI
dict_input_tokens, num_notes = midi_to_dict(sample_midi_path)

print("num_notes", num_notes)

# Pre-compute ORIGINAL arrows from input MIDI (user guidance)
# These arrows will guide generation - they come from the TARGET melody
original_pitch_tensor = torch.tensor(dict_input_tokens['pitch'], dtype=torch.long).unsqueeze(0)
original_arrows = model.pitch_to_arrow(original_pitch_tensor).squeeze(0).tolist()  # [num_notes-1] arrows
print(f"Original arrows computed: {len(original_arrows)} arrows from {num_notes} pitches")

''' GENERATE 10 files'''

for j in range(0, 10):  # generate 10 continuation files
  # Maintain a rolling buffer for autoregressive generation
  pitch_buffer = dict_input_tokens['pitch'].copy()  # Start with original sequence
  
  # Build output sequence (shorter than input)
  dict_output_tokens = {
      'dtime': [],
      'pitch': [],
      'dur': []
  }
  
  print("Generating continuation file", j)
  timeStart = time.perf_counter()
  # generate pitches autoregressively
  for i in range(0, num_notes-1-CTX_LEN):
    
    # Context pitch: use GENERATED pitches from rolling buffer (autoregressive)
    # Context arrows: use ORIGINAL arrows from target melody (user guidance)
    # 
    # At position i, we're generating pitch at i+CTX_LEN+1
    # arrow[i+CTX_LEN] tells us the direction FROM pitch[i+CTX_LEN] TO pitch[i+CTX_LEN+1]
    context = {
      'pitch': torch.tensor(pitch_buffer[i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'arrow': torch.tensor(original_arrows[i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),  # From ORIGINAL melody!
    }

    context = to_device(context, device)
  
    with torch.inference_mode():
        new_pitch_token = model.gen_pitch_token(context, temperature=temperature)
    
    # Update the rolling buffer (so next iteration uses this generated pitch in context)
    pitch_buffer[i+CTX_LEN+1] = new_pitch_token
    
    # Build output sequence
    dict_output_tokens['pitch'].append(new_pitch_token)
    dict_output_tokens['dtime'].append(dict_input_tokens['dtime'][i+CTX_LEN+1])
    dict_output_tokens['dur'].append(dict_input_tokens['dur'][i+CTX_LEN+1])
    #print(new_pitch_token)

  #timeEnd = time.perf_counter()
  #print("t=", (timeEnd-timeStart) * 1000 / i, "ms") # in miliseconds, promig

  # Output context (already built to correct length in loop above)
  context = {
      'dtime': dict_output_tokens['dtime'],
      'pitch': dict_output_tokens['pitch'],
      'dur': dict_output_tokens['dur'],
    }

  # generate a midi file from generated pitches
  song_d = dict_to_song(context)
  detailed_stats = ms_SONG_to_MIDI_Converter(song_d, output_file_name = output_midi_name+str(j),
                                                            timings_multiplier=2
                                                            )
  # generate a midi file from generated arrows
  pitch_tensor = torch.tensor(context['pitch'], dtype=torch.long).unsqueeze(0)  # [1, T]
  arrows = model.pitch_to_arrow(pitch_tensor).squeeze(0)
  arrows = torch.add(arrows, -3)
  arrows = torch.add(arrows, 60) # shift to C3-C4 range

  arrow_context = {
      'dtime': context['dtime'][1:],  # skip first, keep T-1 elements
      'dur': context['dur'][1:],      # skip first, keep T-1 elements
      'pitch': arrows       # T-1 elements
  }
  song_d = dict_to_song(arrow_context)

  detailed_stats = ms_SONG_to_MIDI_Converter(song_d, output_file_name = output_arrows_midi_name+str(j),
                                                            timings_multiplier=2
                                                            ) 


  # generate a midi file from input arrows
  pitch_tensor = torch.tensor(dict_input_tokens['pitch'][CTX_LEN:num_notes], dtype=torch.long).unsqueeze(0)  # [1, T]
  arrows2 = model.pitch_to_arrow(pitch_tensor).squeeze(0)
  arrows2 = torch.add(arrows2, -3) # shift to C3-C4 range
  arrows2 = torch.add(arrows2, 60) # shift to C3-C4 range

  # generate a midi file from input arrows
  arrow_context = {
      'dtime': dict_input_tokens['dtime'][1+CTX_LEN:num_notes], # skip first, keep T-1 elements
      'dur': dict_input_tokens['dur'][1+CTX_LEN:num_notes],      # skip first, keep T-1 elements
      'pitch': arrows2,  # T-1 elements
    }
  # generate a midi file from generated pitches
  song_d = dict_to_song(arrow_context)
  detailed_stats = ms_SONG_to_MIDI_Converter(song_d, output_file_name = input_arrows_midi_name+str(j),
                                                            timings_multiplier=2
                                                            )

  # Calculate percentage of different arrows
  # arrows: generated arrows (from output pitch sequence)
  # arrows2: original arrows (from input pitch sequence)
  
  # Ensure both are on same device and shape

  diff = (arrows != arrows2[1:]).float().mean().item()
      
  print(f"Iteration {j}: Arrow match percentage = {(1-diff) * 100:.2f}%")