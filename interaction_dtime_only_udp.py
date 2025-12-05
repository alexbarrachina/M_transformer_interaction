#===================================================================================================
# Monster Genie interaction_dtime_only_udp.py Python module
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
# pip install pyfluidsynth
from typing import Optional, List
import socket
# UDP imports for MPR121
from threading import Lock

import torch

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter
from visualizer import Visualizer

TRACES = False

''' UDP CONFIGURATION '''
UDP_IP = ""  # Listen on all interfaces
UDP_PORT = 3000

''' DEVICE '''
#device = torch.device('cpu')
device = torch.device('mps') 

''' MODEL '''
model_name = 'no_dtime_good_reference'
cfg = get_model_hparams(model_name)
model = load_model(model_name=model_name, cfg=cfg )
model.to(device)
model.eval()


''' PARAMS '''
# Get sample seed MIDI path
#sample_midi_path = './seed_midis/Monster-Piano-Transformer-Piano-Seed-3.mid'
sample_midi_path = './samples/clairTester_to_end.midi'
output_midi_name = './out/interactive_performance'

CTX_LEN = 256 # num notes in context. tokens = CTX_LENGTH * 3
TOTAL_GEN_LEN = 1024 # num notes to generate

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
visualizer = Visualizer()

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
                os._exit(1)
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

    i = 0
    dict_output_tokens['dtime'] = dict_input_tokens['dtime'] 
    dict_output_tokens['pitch'] = dict_input_tokens['pitch'] 
    dict_output_tokens['dur'] = dict_input_tokens['dur'] 
    dict_output_tokens['button'] = dict_input_tokens['button'] 

''' VARIABLES '''
context = None
timeLast = 0
i = 0 # num current tokens in context after CTX_LEN
noteOn_dict = {}
first_note = True

''' BUILD CTX '''
# Load seed MIDI
dict_input_tokens, num_notes = midi_to_dict(sample_midi_path) # tokens

dict_output_tokens = dict_input_tokens.copy()

if TRACES:  
    print("num_notes", num_notes)
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
    b = b.clone().detach().tolist()

visualizer.primer(dict_input_tokens['pitch'][:CTX_LEN], dict_input_tokens['dtime'][:CTX_LEN ], b[:CTX_LEN])

def manageNote(button, velocity): 
  global context  # Access the global context
  global timeLast # time of last note, global variable
  global b # button array
  global i # num current tokens in context after CTX_LEN
  global dict_output_tokens # output tokens
  global noteOn_dict # button: (pitch, timeIn)
  global first_note
  global visualizer
  
  if TRACES:
    print("button", button)

  timeNew = time.perf_counter()*1000 /32 # in miliseconds /32 as in midi_to_dict()

  if velocity > 0: # button pressed
    # Update position token
    dtime = max(0, min(127, int(timeNew) - int(timeLast))) # time difference from previous events, but trunk to maximum 127
    if first_note:
        dtime = 0
        first_note = False

    timeLast = timeNew
    dict_output_tokens['dtime'][i+CTX_LEN] = dtime
    # Use button directly (no conversion needed)
    try:
        but = button  # button is already the correct value
        b[i+CTX_LEN] = but
    except:
        print("ERROR", b[i+CTX_LEN])
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
        new_pitch_token = model.gen_pitch_token(context)
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
        
except KeyboardInterrupt:
    print(f"\nStopping receiver...")
    print(f"Total messages received: {message_count}")
    
except Exception as e:
    print(f"Error: {e}")
    
finally:
    sock.close()
    print("Receiver stopped.")

