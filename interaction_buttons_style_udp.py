#===================================================================================================
# Monster Genie interaction_buttons_style_udp.py Python module
# Interaction, generating buttons from MPR121 capacitive sensors via UDP,
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
import atexit
# pip install pyfluidsynth
from typing import Optional, List, Tuple
import socket
# UDP imports for MPR121
from threading import Lock
try:
    from pynput import keyboard as pkeyboard
except ImportError:
    pkeyboard = None

import torch

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter
from visualizer import Visualizer

TRACES = False
USE_CACHE = False
CACHE_IDLE_TIMEOUT = 2.0  # seconds - clear KV cache after this idle gap
TEMPERATURE = 1

''' UDP CONFIGURATION '''
UDP_IP = ""  # Listen on all interfaces
UDP_PORT = 3000

''' DEVICE SPECIFIC PARAMETERS '''
if torch.backends.mps.is_available():
    # CASA
    device = torch.device('mps')
    CTX_LEN = 128 # num notes in context.
    TOTAL_GEN_LEN = 800 # num notes to generate
else:
    # ESMUC
    device = torch.device('cuda')
    CTX_LEN = 512 # num notes in context.
    TOTAL_GEN_LEN = 1024 # num notes to generate


''' MODEL '''
model_name = 'AE_style_jokerParam_tester_v1'

cfg = get_model_hparams(model_name)
model = load_model(model_name=model_name, cfg=cfg )
model.to(device)
model.eval()


''' PARAMS '''
# Get sample seed MIDI path
sample_midi_path1 = './samples/Bach_Prelude_and_Fugue_in_C_major.mid'
sample_midi_path2 = './samples/clairTester_to_end.midi'
sample_midi_path3 = './samples/Chopin_Nocturnes_Op9No1_In_B_Flat_Minor.mid'
sample_midi_path4 = './samples/Scott_Cyril_Lotus_Land.mid'
sample_midi_path5 = './samples/Satie_Gymnopedie_No1.mid'

sample_midi_path_init = sample_midi_path2
STYLE_IDX_INIT = 2

# Style prompts for keys 1--5. Each prompt is encoded once at startup so a
# live style switch does not interrupt the performance.
style_prompt_midi_paths: List[str] = [
    sample_midi_path1,
    sample_midi_path2,
    sample_midi_path3,
    sample_midi_path4,
    sample_midi_path5,
]
output_midi_name = './out/interactive_performance'

NUM_BUTTONS = cfg['num_buttons']
HIGHLIGHT_MOTIF_LEN = 30
MOTIF_STYLE_KEY = '6'

'''THREADING'''
# Add these at the global scope after your imports
buffer_lock = Lock()
save_lock = Lock()

''' UDP PARSING FUNCTIONS '''
def parse_message(message):
    """Parse incoming UDP message and return structured data"""
    try:
        parts = message.decode('utf-8').strip().split()
        if len(parts) < 2:
            return None
        
        if parts[0] == 'sw' and len(parts) == 3:
            # Switch message: "sw 1 1" or "sw 1 0"
            return {
                'type': 'switch',
                'number': int(parts[1]),
                'state': int(parts[2]),
                'raw': message.decode('utf-8').strip()
            }
        elif parts[0] in ['x1', 'x2'] and len(parts) == 3:
            # Capacitive sensor message: "x1 0 1" or "x2 5 0"
            pin = -1
            if parts[0]=='x2' and parts[1]== '5':
                pin = 0
            if parts[0]=='x2' and parts[1]== '6':
                pin = 1
            if parts[0]=='x2' and parts[1]== '7':
                pin = 2
            if parts[0]=='x2' and parts[1]== '8':
                pin = 3
            if parts[0]=='x2' and parts[1]== '9':
                pin = 4
            if parts[0]=='x2' and parts[1]== 'a':
                pin = 5
            if parts[0]=='x2' and parts[1]== 'b':
                pin = 6
            if parts[0]=='x1' and parts[1]== '1':
                pin = 7
            if parts[0]=='x1' and parts[1]== '0':
                pin = 8
            if parts[0]=='x1' and parts[1]== '3':
                pin = 9
            if parts[0]=='x1' and parts[1]== '4':
                pin = 10
            if parts[0]=='x1' and parts[1]== '5':
                pin = 11
            if parts[0]=='x1' and parts[1]== 'a':
                reset_context()
                pin = 0

            return {
                'type': 'capacitive',
                'sensor': parts[0],
                'pin': pin,  # Convert hex to int
                'state': int(parts[2]),
                'raw': message.decode('utf-8').strip()
            }
        else:
            return {
                'type': 'unknown',
                'raw': message.decode('utf-8').strip()
            }
    except Exception as e:
        return {
            'type': 'error',
            'raw': message.decode('utf-8', errors='ignore'),
            'error': str(e)
        }

'''VISUALIZER'''
visualizer = Visualizer(button_slots=NUM_BUTTONS)

''' MPR121 TO BUTTON MAPPING '''
def mpr121_to_button(sensor, pin):
    """Map MPR121 capacitive sensor data to button number (0-11)"""
    # Map sensor and pin combination to button number
    # Assuming x1 has pins 0-11 and x2 has pins 0-11 for a total of 24 buttons
    # But we only need 12 buttons (0-11) for the model
    if sensor == 'x1':
        return pin % 12  # Map x1 pins 0-11 to buttons 0-11
    elif sensor == 'x2':
        return pin % 12  # Map x2 pins 0-11 to buttons 0-11
    else:
        return 0  # Default to button 0

''' UDP MESSAGE HANDLER '''
def handle_udp_message(data):
    """Handle incoming UDP message from MPR121"""
    parsed = parse_message(data)
    if not parsed:
        return
    
    if parsed['type'] == 'capacitive':
        # Map sensor data to button
        #button = mpr121_to_button(parsed['sensor'], parsed['pin'])
        button = parsed['pin']

        if not 0 <= button < NUM_BUTTONS:
            print(f"Ignoring unmapped MPR121 input: {parsed['sensor']} pin {button}")
            return

        velocity = 100 if parsed['state'] else 0  # Convert state to velocity
        
        if TRACES:
            print(f"MPR121: {parsed['sensor']} pin {parsed['pin']} -> button {button}, state {parsed['state']}")
        
        with buffer_lock:  # lock to avoid race condition
            manageNote(button, velocity)  # Use button as "note" for manageNote function
    
    elif parsed['type'] == 'switch':
        # Handle switch messages for special functions
        if parsed['number'] == 1 and parsed['state']:  # Switch 1 for save
            with save_lock:
                print("saving performance")
                save_performance()
                sys.exit(0)
        elif parsed['number'] == 2 and parsed['state']:  # Switch 2 for reset
            with save_lock:
                print("resetting context")
                reset_context()
    
    elif parsed['type'] == 'error':
        print(f"UDP parsing error: {parsed['error']}")
    
    elif parsed['type'] == 'unknown':
        if TRACES:
            print(f"Unknown UDP message: {parsed['raw']}")


def key_to_button(key):
    key = key - 48 # keyboard starts at C = 48
    button = key % 20 # 12 white keys, 8 black keys
    toWhite = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 10, 10, 11, 11]
    button = toWhite[button] # convert to white key index
    #print("k_2_b", button)
    return button

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
  global dict_output_tokens
  global i

  context = {
      'dtime': dict_output_tokens['dtime'][:i+CTX_LEN+1],
      'pitch': dict_output_tokens['pitch'][:i+CTX_LEN+1],
      'dur': dict_output_tokens['dur'][:i+CTX_LEN+1],
    }

  if TRACES:
    print("dtime_save", dict_output_tokens['dtime'][i:i+CTX_LEN+1])

  # generate a midi file from generated pitches
  song_d = dict_to_song(context)

  detailed_stats = ms_SONG_to_MIDI_Converter(song_d, output_file_name = output_midi_name,
                                                            timings_multiplier=2
                                                            )
  print("saved performance")

def reset_context():
    global i
    global dict_output_tokens, dict_input_tokens
    global kv_cache
    global b
    global first_note, timeLast, noteOn_dict

    with buffer_lock:
        i = 0
        kv_cache = None
        first_note = True
        timeLast = 0
        noteOn_dict = {}

        # Reset both the musical history and the corresponding inferred buttons.
        required_len = TOTAL_GEN_LEN + CTX_LEN
        for key in dict_input_tokens.keys():
            extended_list = dict_input_tokens[key].copy()
            if len(extended_list) < required_len:
                extended_list.extend([0] * (required_len - len(extended_list)))
            dict_output_tokens[key] = extended_list

        seed_context = {
            'dtime': torch.tensor(dict_input_tokens['dtime'], dtype=torch.long).unsqueeze(0).to(device),
            'pitch': torch.tensor(dict_input_tokens['pitch'], dtype=torch.long).unsqueeze(0).to(device),
            'dur': torch.tensor(dict_input_tokens['dur'], dtype=torch.long).unsqueeze(0).to(device),
        }
        with torch.inference_mode():
            encoded = model.encoder(seed_context)
            b = model.real_to_discrete(encoded).squeeze(0).clone().detach().tolist()
        if len(b) < required_len:
            b.extend([0] * (required_len - len(b)))

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
dict_input_tokens, num_notes = midi_to_dict(sample_midi_path_init) # tokens

# Extend dict_output_tokens to accommodate TOTAL_GEN_LEN + CTX_LEN tokens
required_len = TOTAL_GEN_LEN + CTX_LEN
dict_output_tokens = {}
for key in dict_input_tokens.keys():
    extended_list = dict_input_tokens[key].copy()
    if len(extended_list) < required_len:
        extended_list.extend([0] * (required_len - len(extended_list)))
    dict_output_tokens[key] = extended_list

if TRACES:  
    print("num_notes", num_notes)
# Build context tokens
context = {
    'dtime': torch.tensor(dict_input_tokens['dtime'], dtype=torch.long).unsqueeze(0),
    'pitch': torch.tensor(dict_input_tokens['pitch'], dtype=torch.long).unsqueeze(0),
    'dur': torch.tensor(dict_input_tokens['dur'], dtype=torch.long).unsqueeze(0)
    }
context = to_device(context, device)

style_seq_len = int(cfg.get('style_seq_len', CTX_LEN))
MOTIF_STYLE_IDX = len(style_prompt_midi_paths)

def _encode_style_pitch_list(
    pitch_list: List[int], repeat_to_style_len: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a MIDI pitch sequence in the form expected by the style model."""
    valid_len = min(len(pitch_list), style_seq_len)
    if valid_len <= 0:
        style_pitches = torch.full(
            (1, style_seq_len), PAD_IDX, dtype=torch.long, device=device
        )
        style_mask = torch.zeros((1, style_seq_len), dtype=torch.bool, device=device)
    elif repeat_to_style_len:
        motif = torch.tensor(pitch_list[:valid_len], dtype=torch.long, device=device)
        repeat_count = (style_seq_len + valid_len - 1) // valid_len
        style_pitches = motif.repeat(repeat_count)[:style_seq_len].unsqueeze(0)
        style_mask = torch.ones((1, style_seq_len), dtype=torch.bool, device=device)
    else:
        padded_pitches = pitch_list[:style_seq_len]
        if valid_len < style_seq_len:
            padded_pitches = padded_pitches + [PAD_IDX] * (style_seq_len - valid_len)
        mask = [True] * valid_len + [False] * (style_seq_len - valid_len)
        style_pitches = torch.tensor(padded_pitches, dtype=torch.long, device=device).unsqueeze(0)
        style_mask = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

    with torch.inference_mode():
        style_context = model.encode_style(style_pitches, style_mask)
    return style_context, style_mask

# Pre-encode all MIDI style prompts to keep style switches free of inference latency.
style_contexts: List[torch.Tensor] = []
style_context_masks_list: List[torch.Tensor] = []
for style_path in style_prompt_midi_paths:
    style_tokens, _ = midi_to_dict(style_path)
    encoded_style, encoded_mask = _encode_style_pitch_list(
        style_tokens['pitch'][:style_seq_len]
    )
    style_contexts.append(encoded_style)
    style_context_masks_list.append(encoded_mask)
    print(f"Style {len(style_contexts)} encoded: {style_path}")

active_style_idx: int = STYLE_IDX_INIT - 1
highlight_motif_pitch_tokens: List[int] = []
motif_style_ready = False
# Reserve a stable slot for a performer-captured motif style.
style_contexts.append(style_contexts[active_style_idx])
style_context_masks_list.append(style_context_masks_list[active_style_idx])
style_context = style_contexts[active_style_idx]
style_context_mask = style_context_masks_list[active_style_idx]

def capture_highlight_motif() -> None:
    """Turn the most recent generated phrase into a reusable style prompt."""
    global highlight_motif_pitch_tokens, motif_style_ready, kv_cache

    with buffer_lock:
        end_idx = CTX_LEN + i
        start_idx = max(CTX_LEN, end_idx - HIGHLIGHT_MOTIF_LEN)
        pitch_list = list(dict_output_tokens['pitch'][start_idx:end_idx])

    if not pitch_list:
        print("No generated pitch tokens yet for highlight motif")
        return

    encoded_style, encoded_mask = _encode_style_pitch_list(
        pitch_list, repeat_to_style_len=True
    )
    with buffer_lock:
        highlight_motif_pitch_tokens = pitch_list
        style_contexts[MOTIF_STYLE_IDX] = encoded_style
        style_context_masks_list[MOTIF_STYLE_IDX] = encoded_mask
        motif_style_ready = True
        kv_cache = None
    print(
        f"Highlight motif saved: {len(highlight_motif_pitch_tokens)} pitch tokens. "
        f"Press {MOTIF_STYLE_KEY} to activate."
    )
  
with torch.inference_mode():
    e = model.encoder(context) # encoder output (batch, seq_len)
    b = model.real_to_discrete(e).squeeze(0) # generate buttons (batch, seq_len)
    b = b.clone().detach().tolist()
    # Extend b to accommodate TOTAL_GEN_LEN + CTX_LEN tokens
    if len(b) < required_len:
        b.extend([0] * (required_len - len(b)))

visualizer.primer(
    dict_input_tokens['pitch'][:CTX_LEN],
    dict_input_tokens['dtime'][:CTX_LEN],
    b[:CTX_LEN],
    dict_input_tokens['dur'][:CTX_LEN],
)

def manageNote(button, velocity): 
  global context  # Access the global context
  global timeLast # time of last note, global variable
  global b # button array
  global i # num current tokens in context after CTX_LEN
  global dict_output_tokens # output tokens
  global noteOn_dict # button: (pitch, timeIn)
  global first_note
  global visualizer
  global kv_cache
  global last_gen_time
  global active_style_idx
  
  if TRACES:
    print("button", button)

  timeNew = time.perf_counter()*1000 /32 # in miliseconds /32 as in midi_to_dict()

  if velocity > 0: # button pressed
    now = time.perf_counter()
    if USE_CACHE and kv_cache is not None and (now - last_gen_time) > CACHE_IDLE_TIMEOUT:
      kv_cache = None
    # Update position token
    dtime = max(0, min(127, int(timeNew) - int(timeLast))) # time difference from previous events, but trunk to maximum 127
    if first_note:
        dtime = 0
        first_note = False

    timeLast = timeNew
    dict_output_tokens['dtime'][i+CTX_LEN] = dtime
    # The UDP/MPR121 mapping already yields the model's button index.
    but = button
    b[i+CTX_LEN] = but
    context = {
      'dtime': torch.tensor(dict_output_tokens['dtime'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'pitch': torch.tensor(dict_output_tokens['pitch'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'dur': torch.tensor(dict_output_tokens['dur'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'button': torch.tensor(b[i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0)
    }
    context = to_device(context, device)
    if TRACES:
        print("dtime", dict_output_tokens['dtime'][i:i+CTX_LEN+1])
                 
    with torch.inference_mode():
        if USE_CACHE:
            new_pitch_token, kv_cache = model.gen_pitch_token(
                context,
                style_context=style_contexts[active_style_idx],
                style_context_mask=style_context_masks_list[active_style_idx],
                cache=kv_cache,
            )
        else:
            new_pitch_token, _ = model.gen_pitch_token(
                context,
                style_context=style_contexts[active_style_idx],
                style_context_mask=style_context_masks_list[active_style_idx],
                temperature=TEMPERATURE,
            )
        last_gen_time = time.perf_counter()
    dict_output_tokens['pitch'][i+CTX_LEN] = new_pitch_token

    playNote(new_pitch_token, velocity) 
    visualizer.get_note(new_pitch_token, velocity)
    visualizer.get_button(but, velocity)

    # add (user_note, pitch, time) to dictionary
    noteOn_dict[but] = (new_pitch_token, timeNew)
    i += 1

  else: # button released
    if TRACES:
        print("buttonOff", button)
    but = button  # button is already the correct value
    #print("but", but)
    if but in noteOn_dict:
      if TRACES:
        print("in_Noteon_dict")
      # get pitch and time in dictionary of accumulated notesOns without noteOff
      pitch, noteOn_time = noteOn_dict[but]
      playNote(pitch, 0)
      visualizer.get_note(pitch, 0)
      visualizer.get_button(but, 0)
      #visualizer.update(noteOn_time)


def _activate_style(style_idx: int) -> None:
    """Select a pre-encoded prompt and invalidate generation cache safely."""
    global active_style_idx, style_context, style_context_mask, kv_cache

    with buffer_lock:
        active_style_idx = style_idx
        style_context = style_contexts[active_style_idx]
        style_context_mask = style_context_masks_list[active_style_idx]
        kv_cache = None

    if style_idx == MOTIF_STYLE_IDX:
        print("Highlight motif active")
    else:
        print(f"Style {style_idx + 1} active: {style_prompt_midi_paths[style_idx]}")


"""# KEYBOARD LISTENER — keys 1--5 switch styles, space captures a motif, 6 activates it """
def _on_key_press(key) -> None:
    if pkeyboard is not None and key == pkeyboard.Key.space:
        capture_highlight_motif()
        return

    try:
        char = key.char  # type: ignore[union-attr]
    except AttributeError:
        return

    if char in ('1', '2', '3', '4', '5'):
        _activate_style(int(char) - 1)
    elif char == MOTIF_STYLE_KEY:
        if motif_style_ready:
            _activate_style(MOTIF_STYLE_IDX)
        else:
            print("No highlight motif saved yet. Press space first.")
    elif char == 'r':
        print("resetting context")
        reset_context()
    elif char == 's':
        with save_lock:
            save_performance()


if pkeyboard is not None:
    _key_listener = pkeyboard.Listener(on_press=_on_key_press)
    _key_listener.start()
else:
    print("pynput is unavailable; live keyboard style switching and motif controls are disabled.")


"""# UDP RECEIVER """

try:
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Bind to all interfaces on the specified port
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(0.001)  # 1ms timeout for responsive visualizer
    
    print(f"MPR121 UDP Receiver starting...")
    print(f"Listening on port {UDP_PORT}")
    print(f"Waiting for data from MPR121...")
    print("-" * 50)
    print("✓ Receiver ready and listening!")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    message_count = 0
    
    while True:
        try:
            # Receive data
            data, addr = sock.recvfrom(1024)
            message_count += 1
            
            # Handle the message
            handle_udp_message(data)
            
            # Show sender info periodically
            if message_count % 100 == 0:
                print(f"Received {message_count} messages from {addr[0]}")
                
        except socket.timeout:
            # Timeout occurred, continue with visualizer update
            pass
        
        # Update visualizer
        visualizer.draw()
        
except (EOFError, KeyboardInterrupt, SystemExit):
    print("Bye.")
    
finally:
    sock.close()
    print("Receiver stopped.")
