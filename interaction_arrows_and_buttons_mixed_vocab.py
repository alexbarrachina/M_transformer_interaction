#===================================================================================================
# Monster Genie interaction_arrows_and_buttons.py Python module
# Interaction, generating pitches using arrows (melody) and buttons (accompaniment) 
# from QWERTY keyboard, starting with a context extracted from a MIDI file
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
import pygame.midi
from pygame.locals import *

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter
from visualizer import Visualizer

TRACES = False
AUTOMATIC_ARROWS = False # if True, use original midi file arrows for guidance

K_ENYE = 241
K_ACCENT = 180
K_C_TRENCADA = 231

# Model constants (matching AE_mixed_vocab)
ROLE_MELODY: int = 0
ROLE_ACCOMP: int = 1
KEY_NA: int = 0  # Not applicable
# Mixed vocabulary: arrows 1-7, buttons 8-19
ARROW_OFFSET: int = 1  # arrows are 1-7 (user arrow 0-6 + offset)
BUTTON_OFFSET: int = 8  # buttons are 8-19 (user button 0-11 + offset)
NUM_ARROWS: int = 7
NUM_BUTTONS: int = 12

''' DEVICE '''
#device = torch.device('cpu') 
device = torch.device('mps') 

''' MODEL '''
model_name = 'AE_mixed_vocab_tester_v1'
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

''' MIDI INITIALIZATION '''
pygame.midi.init()

# Setup MIDI input device
midi_input = None  # type: Optional[pygame.midi.Input]
midi_device_id = pygame.midi.get_default_input_id()

if midi_device_id >= 0:
    try:
        midi_input = pygame.midi.Input(midi_device_id)
        device_info = pygame.midi.get_device_info(midi_device_id)
        print(f"MIDI input device opened: {device_info[1].decode()}")
    except Exception as e:
        print(f"Error opening MIDI device: {e}")
        midi_input = None
else:
    print("No MIDI input device found. MIDI controller disabled.")

# Dictionary to track MIDI note-on events (for note-off handling)
key_noteOn_dict = {}  # type: dict


'''KEY MAPPING'''
# Fine arrows (0-6): specific interval ranges
# Coarse arrows (7-8): direction only (any down / any up)
# Arrow 3 (stay) is shared between fine and coarse modes
KEY_MAPPING_ARROWS = {
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
}

KEY_MAPPING_BUTTONS = {
    
    K_6: 0, K_7: 1, K_8: 2, K_9: 3, K_0: 4,
    K_y: 5, K_u: 6, K_i: 7, K_o: 8, K_p: 9, K_BACKQUOTE: 10, K_PLUS: 11,
    K_h: 12, K_j: 13, K_k: 14, K_l: 15, K_ENYE: 16, K_ACCENT: 17, K_C_TRENCADA: 18,
    K_n: 19, K_m: 20, K_COMMA: 21, K_PERIOD: 22, K_SLASH: 23, 
}

def midi_note_to_button(midi_note):
    toWhite = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 10, 10, 11, 11, 12, 12, 13, 14, 14, 15, 15, 16, 17, 17, 18, 18, 19, 19, 20, 21, 21, 22,22,23,23]
    button = toWhite[midi_note - 48] # convert to white key index
    #print("k_2_b", button)
    if TRACES:
        print("button", button)
    return button

"""# FLUIDSYNTH INIT """
fs = fluidsynth.Synth()
fs.start()
sfid = fs.sfload("./piano.sf2")
fs.program_select(0, sfid, 0, 0)

def playNote(note, velocity=100):
    #if TRACES:
    #    print("fluidNote", note, velocity)
    if velocity > 0:
        fs.noteon(0, note, velocity)
    else:
        fs.noteoff(0, note)

def build_context_from_midi(dict_input_tokens):
    # type: (dict) -> tuple
    """
    Build initial context sequences from MIDI data.
    - Keys: mixed vocabulary with arrows (1-7) for melody and buttons (8-19) for accomp
    - Role: 0 for melody, 1 for accompaniment
    
    Returns:
        keys: list of key indices (1-7 for arrows/melody, 8-19 for buttons/accomp)
        role: list of role indices (0=melody, 1=accomp)
    """
    pitch_seq = dict_input_tokens['pitch']
    chan_seq = dict_input_tokens['chan']
    
    num_notes = len(pitch_seq)
    keys = []  # type: List[int]
    role = []  # type: List[int]
    
    # Track the last melody pitch for arrow calculation
    last_melody_pitch = None  # type: Optional[int]
    
    for i in range(num_notes):
        current_pitch = pitch_seq[i]
        current_chan = chan_seq[i]
        
        if current_chan == 0:  # Melody note - use arrows (1-7)
            if last_melody_pitch is None:
                # First melody note - use "stay" arrow (4 in 1-7 range)
                keys.append(4)  # stay = arrow 3 + ARROW_OFFSET(1) = 4
            else:
                # Calculate arrow from pitch difference
                diff = current_pitch - last_melody_pitch
                if diff <= -8:
                    arrow = 1  # large descending
                elif diff <= -3:
                    arrow = 2  # medium descending
                elif diff <= -1:
                    arrow = 3  # small descending
                elif diff == 0:
                    arrow = 4  # stay
                elif diff <= 2:
                    arrow = 5  # small ascending
                elif diff <= 7:
                    arrow = 6  # medium ascending
                else:
                    arrow = 7  # large ascending
                keys.append(arrow)
            
            role.append(0)  # Melody
            last_melody_pitch = current_pitch
            
        else:  # Accompaniment note (channel > 0) - use buttons (8-19)
            # Assign button based on pitch (quantize to NUM_BUTTONS)
            # Map pitch range (typically 21-108) to 0-(NUM_BUTTONS-1)
            button_value = max(0, min(NUM_BUTTONS - 1, (current_pitch - 21) * NUM_BUTTONS // 88))
            keys.append(button_value + BUTTON_OFFSET)  # Shift to 8-19 range
            role.append(1)  # Accompaniment
    
    return keys, role 

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
    # type: (dict) -> None
    global i
    global pitch_buffer
    global keys
    global role
    global first_note

    # 1. Capture the last N notes of the current performance
    PRESERVE_LEN = 16
    # Ensure we have enough notes generated to capture
    current_end_idx = i + CTX_LEN
    preserved_pitch = []
    preserved_keys = []
    preserved_role = []
    
    if i > 0:
        # Get the last 16 notes from the current buffer
        start_slice = max(0, current_end_idx - PRESERVE_LEN)
        preserved_pitch = pitch_buffer[start_slice:current_end_idx]
        preserved_keys = keys[start_slice:current_end_idx] if start_slice < len(keys) else []
        preserved_role = role[start_slice:current_end_idx] if start_slice < len(role) else []
        if TRACES:
            print(f"Preserving {len(preserved_pitch)} notes")
    
    # 2. Reset global variables
    i = 0
    first_note = True
    
    # 3. Rebuild context from the new MIDI input
    new_keys, new_role = build_context_from_midi(dict_input)
    
    pitch_buffer = dict_input['pitch'].copy()
    keys = new_keys
    role = new_role
    
    # 4. Splice the preserved notes into the end of the context window
    if len(preserved_pitch) > 0:
        splice_start = CTX_LEN - len(preserved_pitch)
        pitch_buffer[splice_start : CTX_LEN] = preserved_pitch
        if len(preserved_keys) > 0:
            keys[splice_start : CTX_LEN] = preserved_keys
        if len(preserved_role) > 0:
            role[splice_start : CTX_LEN] = preserved_role

    # 5. Ensure buffers are long enough for generation
    if len(pitch_buffer) < TOTAL_GEN_LEN:
         pitch_buffer += [0] * (TOTAL_GEN_LEN - len(pitch_buffer))
    if len(keys) < TOTAL_GEN_LEN:
         keys += [4] * (TOTAL_GEN_LEN - len(keys))  # Default stay (arrow 4)
    if len(role) < TOTAL_GEN_LEN:
         role += [0] * (TOTAL_GEN_LEN - len(role))  # Default melody

''' VARIABLES '''

context = None
timeLast = 0
i = 0 # num current tokens in context after CTX_LEN
first_note = True
role = []  # Track role (0=melody, 1=accomp) for each note

''' BUILD CTX '''
# Load seed MIDI
dict_input_tokens, num_notes = midi_to_dict(sample_midi_path) # tokens, without vel
dict_input_tokens2, _ = midi_to_dict(sample_midi_path2) # tokens, without vel

# Build context sequences from MIDI (keys = arrows 1-7 for melody, buttons 8-19 for accomp)
original_keys, original_role = build_context_from_midi(dict_input_tokens)

keys = original_keys.copy()
role = original_role.copy()
pitch_buffer = dict_input_tokens['pitch'].copy()  # Start with original sequence

# Build context tokens for model (mixed vocabulary)
context = {
      'pitch': torch.tensor(pitch_buffer[0:CTX_LEN], dtype=torch.long).unsqueeze(0),
      'key': torch.tensor(keys[0:CTX_LEN], dtype=torch.long).unsqueeze(0),
      'role': torch.tensor(role[0:CTX_LEN], dtype=torch.long).unsqueeze(0),
    }
context = to_device(context, device)
  
# Visualizer needs lists and dtimes for time axis
visualizer.primer(
    pitch_buffer[:CTX_LEN+1], 
    dict_input_tokens['dtime'][:CTX_LEN+1], 
    original_keys[:CTX_LEN+1]
)



def manage_midi_button_input(user_value, velocity):
    # type: (int, int) -> None
    """
    Handle MIDI note input as button press for accompaniment.
    Maps MIDI notes to button values (0-11) -> keys (8-19).
    
    Args:
        user_value: button value (0-11)
        velocity: MIDI velocity (0 for note-off)
    """
    global context
    global timeLast
    global keys
    global i, pitch_buffer
    global key_noteOn_dict
    global first_note
    global visualizer
    
    if TRACES:
        print(f"MIDI button {user_value} -> key {user_value + BUTTON_OFFSET}, velocity {velocity}")
    
    timeNew = time.perf_counter() * 1000 / 32  # in milliseconds /32 as in midi_to_dict()
    
    if velocity > 0:  # Note-on
        dtime = max(0, min(127, int(timeNew) - int(timeLast)))
        if first_note:
            dtime = 0
            first_note = False
        
        timeLast = timeNew
        
        # Update dtime for saving
        if i + CTX_LEN < len(dict_input_tokens['dtime']):
            dict_input_tokens['dtime'][i + CTX_LEN] = dtime
        else:
            dict_input_tokens['dtime'].append(dtime)
        
        # Update keys: button value + BUTTON_OFFSET -> keys 8-19
        current_idx = i + CTX_LEN - 1
        key_value = user_value + BUTTON_OFFSET  # Convert to key range 8-19
        
        # Check if this key_value is already sounding - if so, ignore this press
        if key_value in key_noteOn_dict:
            if TRACES:
                print(f"MIDI key value {key_value} already active, ignoring duplicate press")
            return
        
        try:
            if current_idx < len(keys):
                keys[current_idx] = key_value
                role[current_idx] = 1  # Accompaniment for MIDI buttons
            else:
                keys.append(key_value)
                role.append(1)  # Accompaniment for MIDI buttons
                
        except Exception as e:
            print("ERROR updating keys (MIDI):", e)
        
        # Build context for model (mixed vocabulary)
        context = {
            'pitch': torch.tensor(pitch_buffer[i:i + CTX_LEN], dtype=torch.long).unsqueeze(0),
            'key': torch.tensor(keys[i:i + CTX_LEN], dtype=torch.long).unsqueeze(0),
            'role': torch.tensor(role[i:i + CTX_LEN], dtype=torch.long).unsqueeze(0),
        }
        context = to_device(context, device)
        
        with torch.inference_mode():
            new_pitch_token = model.gen_pitch_token(context, temperature=temperature)
        
        if TRACES:
            print("new_pitch_token (MIDI)", new_pitch_token)
        
        # Store generated pitch
        if i + CTX_LEN < len(pitch_buffer):
            pitch_buffer[i + CTX_LEN] = new_pitch_token
        else:
            pitch_buffer.append(new_pitch_token)
        
        playNote(new_pitch_token, velocity)
        visualizer.get_note(new_pitch_token, velocity)
        visualizer.get_button(user_value, velocity)
        
        # Use key_value (8-19 for buttons) as dictionary key
        key_noteOn_dict[key_value] = (new_pitch_token, timeNew)
        i += 1
        
    else:  # Note-off
        key_value = user_value + BUTTON_OFFSET  # 0-11 -> 8-19
        if key_value in key_noteOn_dict:
            pitch, noteOn_time = key_noteOn_dict[key_value]
            del key_noteOn_dict[key_value]
            playNote(pitch, 0)
            visualizer.get_note(pitch, 0)
            visualizer.get_button(user_value, 0)

def manage_key_input(user_value, velocity, is_arrow_key):
  # type: (int, int, bool) -> None
  """
  Handle key input for both arrow and button keys.
  Arrows (0-6) -> keys (1-7), Buttons (0-11) -> keys (8-19)
  
  Args:
      user_value: button/arrow value (0-6 for arrows, 0-11 for buttons)
      velocity: note velocity (0 for noteOff)
      is_arrow_key: True if key is an arrow key, False if button key
  """
  global context
  global timeLast
  global original_keys
  global keys
  global i
  global pitch_buffer
  global key_noteOn_dict
  global first_note
  global visualizer
  global KEY_MAPPING_ARROWS
  global KEY_MAPPING_BUTTONS
    
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

    # Convert user_value to key value
    current_idx = i + CTX_LEN - 1
    
    try:
        if is_arrow_key:
            # Arrow key: user_value 0-6 -> key 1-7
            if AUTOMATIC_ARROWS and current_idx < len(original_keys):
                key_value = original_keys[current_idx]
            else:
                key_value = user_value + ARROW_OFFSET  # 0-6 -> 1-7
        else:
            # Button key: user_value 0-11 -> key 8-19
            key_value = user_value + BUTTON_OFFSET  # 0-11 -> 8-19
        
        # Check if this key_value is already sounding - if so, ignore this press
        if key_value in key_noteOn_dict:
            if TRACES:
                print(f"Key value {key_value} already active, ignoring duplicate press")
            return
        
        # Determine role from key type (0=melody/arrows, 1=accomp/buttons)
        current_role = 0 if is_arrow_key else 1
        
        # Update or append to keys and role
        if current_idx < len(keys):
            keys[current_idx] = key_value
            role[current_idx] = current_role
        else:
            keys.append(key_value)
            role.append(current_role)
            
    except Exception as e:
        print("ERROR updating keys:", e)

    # Build context for model (mixed vocabulary)
    context = {
      'pitch': torch.tensor(pitch_buffer[i:i+CTX_LEN], dtype=torch.long).unsqueeze(0),
      'key': torch.tensor(keys[i:i+CTX_LEN], dtype=torch.long).unsqueeze(0),
      'role': torch.tensor(role[i:i+CTX_LEN], dtype=torch.long).unsqueeze(0),
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
    visualizer.get_button(user_value, velocity)

    # Use key_value (1-7 for arrows, 8-19 for buttons) as dictionary key
    key_noteOn_dict[key_value] = (new_pitch_token, timeNew)
    i += 1

  else: # noteOff
      # Compute key_value the same way as noteOn
      if is_arrow_key:
          key_value = user_value + ARROW_OFFSET  # 0-6 -> 1-7
      else:
          key_value = user_value + BUTTON_OFFSET  # 0-11 -> 8-19
      
      pitch = 0
      if key_value in key_noteOn_dict:
          pitch, noteOn_time = key_noteOn_dict[key_value]
          del key_noteOn_dict[key_value]

      playNote(pitch, 0)
      visualizer.get_note(pitch, 0)
      visualizer.get_button(user_value, 0)


"""# INPUT LOOP (QWERTY) """

try:
    print("Starting interaction loop.")
    print("Arrow keys (melody): E,R,T/3,4,5 for up; D,F,G/C,V,B for down; SPACE for stay")
    print("              Coarse: X,S for down; W,2 for up")
    print("Button keys (accomp): 6-0, Y-P, H-;, N-/")
    print("P to Save, ESC to Exit.")

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
                    
                if event.key in KEY_MAPPING_ARROWS:
                    user_value = KEY_MAPPING_ARROWS[event.key]
                    manage_key_input(user_value, 100, is_arrow_key=True)
                    if TRACES:
                        print("key", event.key, "arrow", user_value)
                elif event.key in KEY_MAPPING_BUTTONS:
                    user_value = KEY_MAPPING_BUTTONS[event.key]
                    manage_key_input(user_value, 100, is_arrow_key=False)
                    if TRACES:
                        print("key", event.key, "button", user_value)
                elif event.key == K_p: # Save
                    print("saving performance")
                    save_performance()
                    os._exit(1)                              
                #elif event.key == K_SPACE: # Reset
                #    if TRACES:
                #        print("resetting context")
                #    reset_context(dict_input_tokens)
                elif event.key == 1073742051: # Reset
                    print("resetting context")
                    reset_context(dict_input_tokens2)
                elif event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()

            elif event.type == KEYUP:
                if event.key in KEY_MAPPING_ARROWS:
                    user_value = KEY_MAPPING_ARROWS[event.key]
                    manage_key_input(user_value, 0, is_arrow_key=True)
                elif event.key in KEY_MAPPING_BUTTONS:
                    user_value = KEY_MAPPING_BUTTONS[event.key]
                    manage_key_input(user_value, 0, is_arrow_key=False)

        # Handle pygame.midi input for buttons (accompaniment)
        if midi_input is not None and midi_input.poll():
            midi_events = midi_input.read(10)  # Read up to 10 MIDI events
            for midi_event in midi_events:
                midi_data = midi_event[0]  # [status, note, velocity, 0]
                status = midi_data[0]
                midi_note = midi_data[1]
                velocity = midi_data[2]
                
                user_value = midi_note_to_button(midi_note)
                # Handle Note-On (0x90-0x9F) and Note-Off (0x80-0x8F)
                if 0x90 <= status <= 0x9F:  # Note-On
                    if velocity > 0:
                        manage_midi_button_input(user_value, velocity)
                    else:
                        # Note-On with velocity 0 is treated as Note-Off
                        manage_midi_button_input(user_value, 0)
                elif 0x80 <= status <= 0x8F:  # Note-Off
                    manage_midi_button_input(user_value, 0)

        # Draw visualizer (without handling events internally)
        visualizer.draw(handle_events=False)
        
except (EOFError, KeyboardInterrupt):
    print("Bye.")
finally:
    # Clean up pygame.midi
    if midi_input is not None:
        midi_input.close()
        print("MIDI input closed.")
    pygame.midi.quit()