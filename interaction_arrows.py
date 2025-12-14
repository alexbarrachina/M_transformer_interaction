#===================================================================================================
# Monster Genie interaction_arrows.py Python module
# Interaction, generating arrows from QWERTY keyboard,
# starting with a context extracted from a MIDI file
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
import sys
import fluidsynth
import os 
# pip install pyfluidsynth
from typing import Optional, List
from rtmidi.midiconstants import NOTE_ON, NOTE_OFF
from rtmidi.midiutil import open_midiinput
import rtmidi
# pip install python-rtmidi
from threading import Lock

import torch
import pygame
from pygame.locals import *

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter
from visualizer import Visualizer

TRACES = False
AUTOMATIC_ARROWS = False # if True, use original midi file arrows for guidance

''' DEVICE '''
#device = torch.device('cpu') 
device = torch.device('mps') 

''' MODEL '''
model_name = 'melody_arrow_v7'
cfg = get_model_hparams(model_name)
model = load_model(model_name=model_name, cfg=cfg )
model.to(device)
model.eval()

''' PARAMS '''
# Get sample seed MIDI path
#sample_midi_path = './seed_midis/Monster-Piano-Transformer-Piano-Seed-3.mid'
sample_midi_path = './samples/clairTester_to_end_monophonic.midi'
sample_midi_path2 = './samples/clara.mid'
output_midi_name = './out/interactive_performance'

CTX_LEN = 256 # num notes in context. tokens = CTX_LENGTH * 3
TOTAL_GEN_LEN = 2048 # num notes to generate
temperature = 0.0001 # sampling temperature

'''VISUALIZER'''
visualizer = Visualizer()

'''KEY MAPPING'''
# Fine arrows (0-6): specific interval ranges
# Coarse arrows (7-8): direction only (any down / any up)
# Arrow 3 (stay) is shared between fine and coarse modes
KEY_MAPPING = {
    # Fine arrows (specific intervals)
    # Row 1: v=large_down, c=medium_down, x=small_down, w=stay, e=small_up, r=medium_up, t=large_up
    K_v: 0, K_c: 1, K_x: 2, K_w: 3, K_e: 4, K_r: 5, K_t: 6,
    # Row 2: alternative keys (f, d, s, 2, 3, 4, 5)
    K_f: 0, K_d: 1, K_s: 2, K_2: 3, K_3: 4, K_4: 5, K_5: 6,
    # Coarse arrows (direction only, no specific interval)
    K_DOWN: 7,   # Coarse down: any negative pitch change
    K_UP: 8,     # Coarse up: any positive pitch change
    # Note: K_w / K_2 already map to 3 (stay) - shared between fine and coarse
}

"""# FLUIDSYNTH INIT """
fs = fluidsynth.Synth()
fs.start()
sfid = fs.sfload("./piano.sf2")
fs.program_select(0, sfid, 0, 0)

def playNote(note, velocity=100):
    if TRACES:
        print("fluidNote", note, velocity)
    if velocity > 0:
        fs.noteon(0, note, velocity)
    else:
        fs.noteoff(0, note) 

def save_performance():
  global pitch_buffer
  global dict_input_tokens
  global i

  # Use generated pitches with original dtime/dur for saving
  context = {
      'dtime': dict_input_tokens['dtime'][:i+CTX_LEN+1],
      'pitch': pitch_buffer[:i+CTX_LEN+1],
      'dur': dict_input_tokens['dur'][:i+CTX_LEN+1],
    }

  if TRACES:
    print("Saving performance with", i+CTX_LEN+1, "notes")

  # generate a midi file from generated pitches
  song_d = dict_to_song(context)

  detailed_stats = ms_SONG_to_MIDI_Converter(song_d, output_file_name = output_midi_name,
                                                            timings_multiplier=2
                                                            )
  if TRACES:
    print("saved performance")

def reset_context(dict_input):
    global i
    global pitch_buffer
    global arrows
    global first_note

    # 1. Capture the last N notes of the current performance
    PRESERVE_LEN = 16
    # Ensure we have enough notes generated to capture
    current_end_idx = i + CTX_LEN
    preserved_pitch = []
    
    if i > 0:
        # Get the last 16 notes from the current buffer
        start_slice = max(0, current_end_idx - PRESERVE_LEN)
        preserved_pitch = pitch_buffer[start_slice:current_end_idx]
        if TRACES:
            print(f"Preserving {len(preserved_pitch)} notes")
    
    # 2. Reset global variables
    i = 0
    first_note = True
    # 3. Reload the original seed content
    # Start with a fresh copy of the original inputs
    pitch_buffer = dict_input['pitch'].copy()
    # 4. Splice the preserved notes into the end of the context window
    # The context window is pitch_buffer[0 : CTX_LEN]
    if len(preserved_pitch) > 0:
        splice_start = CTX_LEN - len(preserved_pitch)
        # Overwrite the end of the seed context with our preserved notes
        pitch_buffer[splice_start : CTX_LEN] = preserved_pitch

    # 5. Regenerate arrows for this new hybrid sequence
    # We need to ensure the arrows match the new pitch sequence
    # Create a tensor for the whole buffer (or at least enough for generation)
    # We'll re-calculate arrows for the whole buffer to be safe and consistent
    
    # IMPORTANT: The model needs arrows for the *entire* potential generation length
    # We must ensure pitch_buffer is long enough if it was short
    if len(pitch_buffer) < TOTAL_GEN_LEN:
         # Pad if necessary (though dict_input_tokens should be long enough usually)
         pitch_buffer += [0] * (TOTAL_GEN_LEN - len(pitch_buffer))

    current_pitch_tensor = torch.tensor(pitch_buffer[:TOTAL_GEN_LEN], dtype=torch.long).unsqueeze(0)
    
    # Recalculate arrows based on this new hybrid melody
    # This is crucial because the spliced notes create new intervals
    new_arrows = model.pitch_to_arrow(current_pitch_tensor).squeeze(0).tolist()
    arrows = new_arrows # Update the global arrows list

''' VARIABLES '''

context = None
timeLast = 0
i = 0 # num current tokens in context after CTX_LEN
noteOn_dict = {}
first_note = True

''' BUILD CTX '''
# Load seed MIDI
dict_input_tokens, num_notes = midi_to_dict(sample_midi_path) # tokens, without vel
dict_input_tokens2, _ = midi_to_dict(sample_midi_path2) # tokens, without vel

original_pitch_tensor = torch.tensor(dict_input_tokens['pitch'], dtype=torch.long).unsqueeze(0)
original_arrows = model.pitch_to_arrow(original_pitch_tensor).squeeze(0).tolist() 

arrows = original_arrows.copy()
pitch_buffer = dict_input_tokens['pitch'].copy()  # Start with original sequence
# Build context tokens

context = {
      'pitch': torch.tensor(pitch_buffer[0:CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'arrow': torch.tensor(original_arrows[0:CTX_LEN+1], dtype=torch.long).unsqueeze(0),  # From ORIGINAL melody!
    }
context = to_device(context, device)
  
# Visualizer needs lists and dtimes for time axis
visualizer.primer(
    pitch_buffer[:CTX_LEN+1], 
    dict_input_tokens['dtime'][:CTX_LEN+1], 
    original_arrows[:CTX_LEN+1]
)

def manage_button_input(key, velocity): 
  global context  # Access the global context
  global timeLast # time of last note, global variable
  global original_arrows # arrows array
  global arrows # arrows array
  global i # num current tokens in context after CTX_LEN
  global pitch_buffer # output tokens
  global noteOn_dict # button: (pitch, timeIn)
  global first_note
  global visualizer
  global KEY_MAPPING
  
  if TRACES:
    print("key", key)

  but = KEY_MAPPING[key]
  timeNew = time.perf_counter()*1000 /32 # in miliseconds /32 as in midi_to_dict()

  if velocity > 0: # noteOn
    # Update position token
    dtime = max(0, min(127, int(timeNew) - int(timeLast))) # time difference from previous events, but trunk to maximum 127
    if first_note:
        dtime = 0
        first_note = False

    timeLast = timeNew
    
    # Update dtime for saving later (optional but good for recording)
    if i+CTX_LEN < len(dict_input_tokens['dtime']):
        dict_input_tokens['dtime'][i+CTX_LEN] = dtime
    else:
        dict_input_tokens['dtime'].append(dtime)

    # If button is already pressed, don't generate a new note
    '''if but in noteOn_dict:
      if TRACES:
        print("in_Noteon_dict")
      # get pitch and time in dictionary of accumulated notesOns without noteOff
      pitch, noteOn_time = noteOn_dict[but]
      playNote(pitch, 0)
      visualizer.get_note(pitch, 0)
      visualizer.get_button(but, 0)'''

    # Update button with user input (providing arrows)
    # The arrow at index [N-1] guides the transition to N.
    # We are about to generate note at [i+CTX_LEN], so we set arrow at [i+CTX_LEN-1]
    arrow_idx = i + CTX_LEN - 1
    # Update button with user input (providing arrows)
    try:
        if arrow_idx < len(arrows):
            if AUTOMATIC_ARROWS:
                arrows[arrow_idx] = original_arrows[arrow_idx]
            else:   
                arrows[arrow_idx] = but
        else:
            if AUTOMATIC_ARROWS:
                arrows.append(original_arrows[arrow_idx])
            else:
                arrows.append(but)
    except Exception as e:
        print("ERROR updating arrows:", e)

    context = {
      'pitch': torch.tensor(pitch_buffer[i:i+CTX_LEN], dtype=torch.long).unsqueeze(0),
      'arrow': torch.tensor(arrows[i:i+CTX_LEN], dtype=torch.long).unsqueeze(0)
    }
    context = to_device(context, device)
                 
    with torch.inference_mode():
        new_pitch_token = model.gen_pitch_token(context, temperature=temperature)
    if TRACES:
        print("new_pitch_token", new_pitch_token)

    # Store the generated pitch
    if i+CTX_LEN < len(pitch_buffer):
        pitch_buffer[i+CTX_LEN] = new_pitch_token
    else:
        pitch_buffer.append(new_pitch_token)

    playNote(new_pitch_token, velocity) 
    visualizer.get_note(new_pitch_token, velocity)
    visualizer.get_button(but, velocity)

    # add (user_note, pitch, time) to dictionary
    noteOn_dict[key] = (new_pitch_token, timeNew)
    i += 1

  else: # noteOff
    if key in noteOn_dict:
      if TRACES:
        print("in_Noteon_dict")
      # get pitch and time in dictionary of accumulated notesOns without noteOff
      pitch, noteOn_time = noteOn_dict[key]
      # Clean up noteOn_dict when note is released
      del noteOn_dict[key]
      playNote(pitch, 0)
      visualizer.get_note(pitch, 0)
      visualizer.get_button(but, 0)
      #visualizer.update(noteOn_time)


"""# INPUT LOOP (QWERTY) """

try:
    print("Starting interaction loop. Use QWERTY keys Z,X,C,A,S,D,F for arrows.")
    print("P to Save, 0 to Reset.")

    while True:
        time.sleep(0.0001)
        
        # Handle Pygame events for QWERTY input
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
                
            elif event.type == KEYDOWN:
                if TRACES:
                    print("event", event.key)
                    print("noteOn_dict", noteOn_dict)
                if event.key in KEY_MAPPING:
                    if TRACES:
                        print("key", event.key, "arrow", KEY_MAPPING[event.key])
                    manage_button_input(event.key, 100) # 100 is default velocity
                elif event.key == K_p: # Save
                    print("saving performance")
                    save_performance()
                    os._exit(1)                              
                elif event.key == K_SPACE: # Reset
                    if TRACES:
                        print("resetting context")
                    reset_context(dict_input_tokens)
                elif event.key == 1073742051: # Reset
                    if TRACES:
                        print("resetting context")
                    reset_context(dict_input_tokens2)
                elif event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()

            elif event.type == KEYUP:
                if event.key in KEY_MAPPING:
                    manage_button_input(event.key, 0)

        # Draw visualizer (without handling events internally)
        visualizer.draw(handle_events=False)
        
except (EOFError, KeyboardInterrupt):
    print("Bye.")
