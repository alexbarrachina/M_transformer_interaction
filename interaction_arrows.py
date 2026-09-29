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
import datetime
import sys
from tkinter import TRUE
import fluidsynth
import os 
import atexit
# pip install pyfluidsynth
from typing import Optional, List
from rtmidi.midiconstants import NOTE_ON, NOTE_OFF
from rtmidi.midiutil import open_midiinput
import rtmidi
# pip install python-rtmidi
from threading import Lock

from sympy.sets.sets import false, true
import torch
import pygame
from pygame.locals import *

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter
from visualizer import Visualizer

TRACES = false
AUTOMATIC_ARROWS = False # if True, use original midi file arrows for guidance
USE_CACHE = False
CACHE_IDLE_TIMEOUT = 2.0  # seconds - clear KV cache after this idle gap


''' DEVICE '''
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda')


''' MODEL '''
model_name = 'melody_arrow_v10'
cfg = get_model_hparams(model_name)
model = load_model(model_name=model_name, cfg=cfg )
model.to(device)
model.eval()

''' PARAMS '''
# Get sample seed MIDI path
#sample_midi_path = './seed_midis/test_mono3.midi'
#sample_midi_path = './samples/clairTester_to_end_monophonic.midi'
#sample_midi_path = './samples/Bach_Prelude_and_Fugue_in_C_major.mid'
#sample_midi_path = './samples/clairTester_to_end.midi'
sample_midi_path = './samples/Chopin_Nocturnes_Op9No1_In_B_Flat_Minor.mid'
sample_midi_path2 = './samples/clara.mid'
# Base path; save_performance appends _YYYYMMDD_HHMMSS before .mid
output_midi_name = './out/interactive_performance'

CTX_LEN = 128 # num notes in context. tokens = CTX_LENGTH * 3
# Audible primer preview: only the tail of the stored primer in visualizer.
PRIMER_PLAYBACK_LAST_N = 40
PRIMER_PLAYBACK_SPEED = 1
TOTAL_GEN_LEN = 1024 # num notes to generate
temperature = 0.0001 # sampling temperature

'''VISUALIZER'''
visualizer = Visualizer(button_slots=cfg.get('num_buttons', 12))

'''KEY MAPPING'''
# Fine arrows (0-6): specific interval ranges
# Coarse arrows (7-8): direction only (any down / any up)
# Arrow 3 (stay) is shared between fine and coarse modes
KEY_MAPPING = {
    # Fine arrows (specific intervals)
    # Row 1: v=large_down, c=medium_down, x=small_down, w=stay, e=small_up, r=medium_up, t=large_up
    K_SPACE: 3,
    # up keys
    K_e: 6, K_r: 5, K_t: 4,
    # Row 2: alternative keys for up
    K_3: 6, K_4: 5, K_5: 4, 
    # down keys
    K_d: 0, K_f: 1, K_g: 2, 
    # Row 2: alternative keys for down
    K_c: 0, K_v: 1, K_b: 2, 
    # Coarse arrows (direction only, no specific interval)
    K_x: 7,  K_s: 7, # Coarse down: any negative pitch change
    K_w: 8,  K_2: 8, # Coarse up: any positive pitch change
    # Note: K_w / K_2 already map to 3 (stay) - shared between fine and coarse
    # No influence (model decides freely)
    #K_g: 9, K_b: 9, # No influence: model generates freely from pitch history

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

  if i <= 0:
      print("nothing recorded to save")
      return

  end = CTX_LEN + i
  # Only generated continuation (indices CTX_LEN .. CTX_LEN+i-1), not the seed primer.
  context = {
      'dtime': dict_input_tokens['dtime'][CTX_LEN:end],
      'pitch': pitch_buffer[CTX_LEN:end],
      'dur': dict_input_tokens['dur'][CTX_LEN:end],
    }

  if TRACES:
    print("Saving performance with", i, "generated notes (no primer)")

  song_d = dict_to_song(context)

  stamped_path = output_midi_name + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  detailed_stats = ms_SONG_to_MIDI_Converter(song_d, output_file_name=stamped_path,
                                                            timings_multiplier=1
                                                            )
  print("saved performance", stamped_path + '.mid')

def reset_context(dict_input):
    global i
    global pitch_buffer
    global arrows
    global first_note
    global kv_cache

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
    kv_cache = None
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
    print("RESET_CONTEXT")
    visualizer.primer(
        pitch_buffer[:CTX_LEN + 1],
        dict_input['dtime'][:CTX_LEN + 1],
        arrows[:CTX_LEN + 1],
        dict_input['dur'][:CTX_LEN + 1],
    )
    visualizer.play_primer(playNote, last_n=PRIMER_PLAYBACK_LAST_N,
                           playback_speed=PRIMER_PLAYBACK_SPEED)

''' VARIABLES '''

context = None
timeLast = 0
i = 0 # num current tokens in context after CTX_LEN
noteOn_dict = {}
first_note = True
kv_cache = None
last_gen_time: float = 0.0

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
    pitch_buffer[:CTX_LEN + 1],
    dict_input_tokens['dtime'][:CTX_LEN + 1],
    original_arrows[:CTX_LEN + 1],
    dict_input_tokens['dur'][:CTX_LEN + 1],
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
  global kv_cache
  global last_gen_time
  global visualizer
  global KEY_MAPPING
  
  if TRACES:
    print("key", key)

  but = KEY_MAPPING[key]
  timeNew = time.perf_counter()*1000 /32 # in miliseconds /32 as in midi_to_dict()

  if velocity > 0: # noteOn
    now = time.perf_counter()
    if USE_CACHE and kv_cache is not None and (now - last_gen_time) > CACHE_IDLE_TIMEOUT:
        kv_cache = None
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
                 
    if device.type == 'mps':
        torch.mps.synchronize()  # ensure previous GPU work is done
    t0 = time.perf_counter()
    
    with torch.inference_mode():
        if USE_CACHE:
            new_pitch_token, kv_cache = model.gen_pitch_token(context, temperature=temperature, cache=kv_cache, use_cache=True)
        else:
            new_pitch_token = model.gen_pitch_token(context, temperature=temperature)
    
    if device.type == 'mps':
        torch.mps.synchronize()  # wait for GPU to finish
    t1 = time.perf_counter()
    print(f"gen_pitch_token: {(t1-t0)*1000:.2f} ms")
    last_gen_time = t1
    
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

    startup_primer_done = False
    while True:
        time.sleep(0.0001)

        if not startup_primer_done:
            visualizer.play_primer(playNote, last_n=PRIMER_PLAYBACK_LAST_N,
                                   playback_speed=PRIMER_PLAYBACK_SPEED)
            startup_primer_done = True

        # Handle Pygame events for QWERTY input
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit(0)
                
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
                    sys.exit(0)                              
                #elif event.key == K_SPACE: # Reset
                #    if TRACES:
                #        print("resetting context")
                #    reset_context(dict_input_tokens)
                elif event.key == 1073742051 or event.key == K_a: # Reset
                    if TRACES:
                        print("resetting context")
                    reset_context(dict_input_tokens2)
                elif event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)

            elif event.type == KEYUP:
                if event.key in KEY_MAPPING:
                    manage_button_input(event.key, 0)

        # Draw visualizer (without handling events internally)
        visualizer.draw(handle_events=False)
        
except (EOFError, KeyboardInterrupt, SystemExit):
    print("Bye.")
