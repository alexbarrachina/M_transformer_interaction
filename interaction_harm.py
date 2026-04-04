#===================================================================================================
# Monster Genie interaction_harm.py Python module
# Interaction with dual-conditioned model (melodic shape buttons + harmony movements)
# Buttons guide melodic shape (each press generates a note).
# Harmony movement keys set the harmonic regime (affects following notes, decays over time).
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
import atexit
# pip install pyfluidsynth
from typing import Optional, List
from rtmidi.midiconstants import NOTE_ON, NOTE_OFF
from rtmidi.midiutil import open_midiinput
import rtmidi
# pip install python-rtmidi
from threading import Lock

from sympy.sets.sets import false
import torch
import pygame
from pygame.locals import *

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter
from visualizer import Visualizer

TRACES = True

''' DEVICE '''
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda')


''' MODEL '''
model_name = 'AE_dual_tester_v1'
cfg = get_model_hparams(model_name)
model = load_model(model_name=model_name, cfg=cfg )
model.to(device)
model.eval()

''' PARAMS '''
# Get sample seed MIDI path
sample_midi_path = './seed_midis/test_mono3.midi'
sample_midi_path = './samples/clairTester_to_end_monophonic.midi'
sample_midi_path2 = './samples/clara.mid'
output_midi_name = './out/interactive_performance'

CTX_LEN = 512 # num notes in context
TOTAL_GEN_LEN = 2048 # num notes to generate
temperature = 0.0001 # sampling temperature

# Linear harmony decay per note step during inference.
# Assumes ~30 notes between harmony presses: 1.0/30 ≈ 0.033 per step, reaches 0 in ~30 notes.
HARM_DECAY_STEP = 1.0 / 30.0

'''VISUALIZER'''
visualizer = Visualizer() 

'''KEY MAPPING'''
# --- Button keys: melodic shape (each press generates a note) ---
# 12 buttons mapped to keys on two QWERTY rows
# Adjust key codes for your keyboard layout (values below assume Spanish QWERTY)
BUTTON_KEY_MAPPING = {
    K_y: 0, K_u: 1, K_i: 2, K_o: 3, K_p: 4,
    K_LEFTBRACKET: 5,  # ` key (right of p, adjust for your keyboard)
    K_h: 6, K_j: 7, K_k: 8, K_l: 9,
    K_SEMICOLON: 10,   # ñ key (right of l on Spanish keyboard)
    K_QUOTE: 11,       # ´ key (right of ñ on Spanish keyboard)
}

# --- Harmony movement keys: set harmonic regime (no note generated) ---
# 8 harmony movements (0-7)
HARMONY_KEY_MAPPING = {
    K_6: MOVE_STABILIZE, 
    K_7: MOVE_RECOLOR, 
    K_8: MOVE_PREPARE, 
    K_9: MOVE_TENSION, 
    K_0: MOVE_RESOLVE,
    K_n: MOVE_EVADE, 
    K_m: MOVE_CHROMATIC,
    K_COMMA: MOVE_MODULATE,  # < key (adjust for your keyboard)
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
    global button_buffer
    global harm_regime_buffer
    global harm_strength_buffer
    global current_harm_regime
    global current_harm_strength
    global first_note

    # 1. Capture the last N notes of the current performance
    PRESERVE_LEN = 16
    current_end_idx = i + CTX_LEN
    preserved_pitch = []
    
    if i > 0:
        start_slice = max(0, current_end_idx - PRESERVE_LEN)
        preserved_pitch = pitch_buffer[start_slice:current_end_idx]
        if TRACES:
            print(f"Preserving {len(preserved_pitch)} notes")
    
    # 2. Reset global variables
    i = 0
    first_note = True
    current_harm_regime = 0
    current_harm_strength = 0.0

    # 3. Reload the original seed content
    pitch_buffer = dict_input['pitch'].copy()

    # 4. Splice the preserved notes into the end of the context window
    if len(preserved_pitch) > 0:
        splice_start = CTX_LEN - len(preserved_pitch)
        pitch_buffer[splice_start : CTX_LEN] = preserved_pitch

    if len(pitch_buffer) < TOTAL_GEN_LEN:
         pitch_buffer += [0] * (TOTAL_GEN_LEN - len(pitch_buffer))

    # 5. Recalculate buttons from the encoder for the new pitch sequence
    current_pitch_tensor = torch.tensor(pitch_buffer[:TOTAL_GEN_LEN], dtype=torch.long).unsqueeze(0).to(device)
    new_buttons = model.gen_buttons({'pitch': current_pitch_tensor}).squeeze(0).cpu().tolist()
    button_buffer = new_buttons

    # 6. Reset harmony buffers (unguided)
    harm_regime_buffer = [0] * TOTAL_GEN_LEN
    harm_strength_buffer = [0.0] * TOTAL_GEN_LEN

    print("RESET_CONTEXT")

''' VARIABLES '''

context = None
timeLast = 0
i = 0 # num current tokens in context after CTX_LEN
noteOn_dict = {}
first_note = True
current_harm_regime = 0       # currently active harmony movement type (0-7)
current_harm_strength = 0.0   # current harmony strength (1.0 at onset, decays)

''' BUILD CTX '''
# Load seed MIDI
dict_input_tokens, num_notes = midi_to_dict(sample_midi_path) # tokens, without vel
dict_input_tokens2, _ = midi_to_dict(sample_midi_path2) # tokens, without vel

# Derive initial buttons from the encoder
original_pitch_tensor = torch.tensor(dict_input_tokens['pitch'], dtype=torch.long).unsqueeze(0).to(device)
original_buttons = model.gen_buttons({'pitch': original_pitch_tensor}).squeeze(0).cpu().tolist()

button_buffer = original_buttons.copy()
pitch_buffer = dict_input_tokens['pitch'].copy()

# Initialize harmony buffers as unguided (all zeros)
harm_regime_buffer = [0] * len(pitch_buffer)
harm_strength_buffer = [0.0] * len(pitch_buffer)

# Ensure buffers are long enough for generation
for buf_name in ['pitch_buffer', 'button_buffer', 'harm_regime_buffer', 'harm_strength_buffer']:
    buf = locals()[buf_name] if buf_name in locals() else globals()[buf_name]
    if len(buf) < TOTAL_GEN_LEN:
        if isinstance(buf[0], float):
            globals()[buf_name] = buf + [0.0] * (TOTAL_GEN_LEN - len(buf))
        else:
            globals()[buf_name] = buf + [0] * (TOTAL_GEN_LEN - len(buf))

# Build initial context (CTX_LEN+1 tokens: gen_pitch_token slices internally)
context = {
      'pitch': torch.tensor(pitch_buffer[0:CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'button': torch.tensor(button_buffer[0:CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'harm_regime': torch.tensor(harm_regime_buffer[0:CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'harm_strength': torch.tensor(harm_strength_buffer[0:CTX_LEN+1], dtype=torch.float).unsqueeze(0),
    }
context = to_device(context, device)
  
# Visualizer needs lists and dtimes for time axis
visualizer.primer(
    pitch_buffer[:CTX_LEN+1], 
    dict_input_tokens['dtime'][:CTX_LEN+1], 
    button_buffer[:CTX_LEN+1]
)

def manage_button_input(key, velocity): 
  global context
  global timeLast
  global i
  global pitch_buffer
  global button_buffer
  global harm_regime_buffer
  global harm_strength_buffer
  global current_harm_regime
  global current_harm_strength
  global noteOn_dict
  global first_note
  global visualizer
  
  if TRACES:
    print("key", key)

  but = BUTTON_KEY_MAPPING[key]
  timeNew = time.perf_counter()*1000 /32 # in miliseconds /32 as in midi_to_dict()

  if velocity > 0: # noteOn
    # Update position token
    dtime = max(0, min(127, int(timeNew) - int(timeLast))) # time difference from previous events, but trunk to maximum 127
    if first_note:
        dtime = 0
        first_note = False

    timeLast = timeNew
    
    # Update dtime for saving later
    if i+CTX_LEN < len(dict_input_tokens['dtime']):
        dict_input_tokens['dtime'][i+CTX_LEN] = dtime
    else:
        dict_input_tokens['dtime'].append(dtime)

    # Set current button and harmony state at the generation position
    gen_pos = i + CTX_LEN
    if gen_pos < len(button_buffer):
        button_buffer[gen_pos] = but
    else:
        button_buffer.append(but)

    if gen_pos < len(harm_regime_buffer):
        harm_regime_buffer[gen_pos] = current_harm_regime
    else:
        harm_regime_buffer.append(current_harm_regime)

    if gen_pos < len(harm_strength_buffer):
        harm_strength_buffer[gen_pos] = current_harm_strength
    else:
        harm_strength_buffer.append(current_harm_strength)

    # Build context (CTX_LEN+1 tokens, gen_pitch_token slices internally)
    context = {
      'pitch': torch.tensor(pitch_buffer[i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'button': torch.tensor(button_buffer[i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'harm_regime': torch.tensor(harm_regime_buffer[i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'harm_strength': torch.tensor(harm_strength_buffer[i:i+CTX_LEN+1], dtype=torch.float).unsqueeze(0),
    }
    if TRACES:
        print("****", context['harm_regime'][:,-1], context['harm_strength'][:,-1])
    context = to_device(context, device)
                 
    with torch.inference_mode():
        new_pitch_token = model.gen_pitch_token(context, temperature=temperature)
    if TRACES:
        print("new_pitch_token", new_pitch_token)

    # Store the generated pitch
    if gen_pos < len(pitch_buffer):
        pitch_buffer[gen_pos] = new_pitch_token
    else:
        pitch_buffer.append(new_pitch_token)

    playNote(new_pitch_token, velocity) 
    visualizer.get_note(new_pitch_token, velocity)
    visualizer.get_button(but, velocity)

    # add (user_note, pitch, time) to dictionary
    noteOn_dict[key] = (new_pitch_token, timeNew)
    i += 1

    # Linear decay: subtract fixed step, clamp to 0
    current_harm_strength = max(0.0, current_harm_strength - HARM_DECAY_STEP)

  else: # noteOff
    if key in noteOn_dict:
      if TRACES:
        print("in_Noteon_dict")
      pitch, noteOn_time = noteOn_dict[key]
      del noteOn_dict[key]
      playNote(pitch, 0)
      visualizer.get_note(pitch, 0)
      visualizer.get_button(but, 0)


def manage_harmony_input(key):
  """Set the active harmony movement. No note is generated."""
  global current_harm_regime
  global current_harm_strength

  current_harm_regime = HARMONY_KEY_MAPPING[key]
  current_harm_strength = 1.0
  if TRACES:
    print(f"Harmony movement set: {current_harm_regime}, strength reset to 1.0")


"""# INPUT LOOP (QWERTY) """

try:
    print("Starting interaction loop.")
    print("  Button keys (generate notes): y,u,i,o,p,[  h,j,k,l,;,'")
    print("  Harmony keys (set movement):  6,7,8,9,0  n,m,<")
    print("  P = Save, ESC = Quit, A = Reset context")

    while True:
        time.sleep(0.0001)
        
        # Handle Pygame events for QWERTY input
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit(0)
                
            elif event.type == KEYDOWN:
                if TRACES:
                    print("event", event.key)
                    print("noteOn_dict", noteOn_dict)
                if event.key in BUTTON_KEY_MAPPING:
                    manage_button_input(event.key, 100)
                elif event.key in HARMONY_KEY_MAPPING:
                    manage_harmony_input(event.key)
                elif event.key == K_p: # Save
                    print("saving performance")
                    save_performance()
                    sys.exit(0)                              
                elif event.key == 1073742051 or event.key == K_a: # Reset
                    if TRACES:
                        print("resetting context")
                    reset_context(dict_input_tokens2)
                elif event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)

            elif event.type == KEYUP:
                if event.key in BUTTON_KEY_MAPPING:
                    manage_button_input(event.key, 0)

        # Draw visualizer (without handling events internally)
        visualizer.draw(handle_events=False)
        
except (EOFError, KeyboardInterrupt, SystemExit):
    print("Bye.")
