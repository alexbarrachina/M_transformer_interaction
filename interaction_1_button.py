#===================================================================================================
# Monster Genie interaction_dtime_only.py Python module
# Interaction, generating buttons from MIDI keyboard,
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
from typing import Optional, List
from rtmidi.midiconstants import NOTE_ON, NOTE_OFF
from rtmidi.midiutil import open_midiinput
import rtmidi
# pip install python-rtmidi
from threading import Lock

from sympy import false
import torch

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter
from visualizer import Visualizer

TRACES = False

''' DEVICE '''
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda')


''' MODEL '''
model_name = 'Dec_no_conditioning_v1'
cfg = get_model_hparams(model_name)
model = load_model(model_name=model_name, cfg=cfg )
model.to(device)
model.eval()

''' PARAMS '''
# Get sample seed MIDI path
#sample_midi_path = './samples/Bach_Prelude_and_Fugue_in_C_major.mid'
sample_midi_path = './samples/Chopin_Nocturnes_Op9No1_In_B_Flat_Minor.mid'
output_midi_name = './out/interactive_performance'

CTX_LEN = 512 # num notes in context. tokens = CTX_LENGTH * 3
TOTAL_GEN_LEN = 2048 # num notes to generate

'''THREADING'''
# Add these at the global scope after your imports
buffer_lock = Lock()
save_lock = Lock()

'''VISUALIZER'''
visualizer = Visualizer()

'''MIDI IN CALLBACK'''
def midiin_callback(event, data=None):
    message, deltatime = event

    if message[0] & 0xF0 == NOTE_ON:
        status, note, velocity = message
        #channel = (status & 0xF) + 1
        with buffer_lock: # lock to avoid race condition
            manageNote(note, velocity)

    if message[0] & 0xF0 == NOTE_OFF: 
        status, note, velocity = message
        #with buffer_lock: # lock to avoid race condition, temporary disabled for debugging
        manageNote(note, 0)
    
    if message[0] & 0xF0 == 176:  # 176 is the status for control change

        if message[1] == 18 and message[2] > 0: # Using REC button as a trigger to save performance
          with save_lock:
            print("saving performance")
            save_performance()
            sys.exit(0)

        if message[1] == 17 and message[2] > 0: # Using PLAY button as a trigger to reset the context
          with save_lock:
            print("resetting context")
            reset_context()

def key_to_button(key):
    key = key - 48 # keyboard starts at C = 48
    button = key #% 20 # 12 white keys, 8 black keys
    toWhite = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 10, 10, 11, 11, 12, 12, 13, 14, 14, 15, 15, 16, 17, 17, 18, 18, 19, 19, 20, 21, 21, 22,22,23,23,24]
    button = toWhite[button] # convert to white key index
    #print("k_2_b", button)
    if TRACES:
        print("button", button)
    return button

"""# FLUIDSYNTH INIT """    
fs = fluidsynth.Synth()
fs.start()
sfid = fs.sfload("./piano.sf2")
fs.program_select(0, sfid, 0, 0)


def cleanup():
    """Properly release audio and MIDI resources before exit"""
    print("Cleaning up...")
    try:
        fs.delete()
    except:
        pass
    try:
        midiin.close_port()
    except:
        pass

atexit.register(cleanup)

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
dict_input_tokens, num_notes = midi_to_dict(sample_midi_path) # tokens, without vel

dict_output_tokens = dict_input_tokens.copy()

b = torch.zeros(len(dict_input_tokens['pitch']), dtype=torch.long)

if TRACES:  
    print("num_notes", num_notes)
# Build context tokens
context = {
    'pitch': torch.tensor(dict_input_tokens['pitch'], dtype=torch.long).unsqueeze(0),
    }
context = to_device(context, device)

visualizer.primer(dict_input_tokens['pitch'][:CTX_LEN], dict_input_tokens['dtime'][:CTX_LEN ], b[:CTX_LEN])

def manageNote(note, velocity): 
  global context  # Access the global context
  global timeLast # time of last note, global variable
  global b # button array
  global i # num current tokens in context after CTX_LEN
  global dict_output_tokens # output tokens
  global noteOn_dict # button: (pitch, timeIn)
  global first_note
  global visualizer
  
  if TRACES:
    print("key", note)

  timeNew = time.perf_counter()*1000 /32 # in miliseconds /32 as in midi_to_dict()

  if velocity > 0: # noteOn
    # Update position token
    dtime = max(0, min(127, int(timeNew) - int(timeLast))) # time difference from previous events, but trunk to maximum 127
    if first_note:
        dtime = 0
        first_note = False

    timeLast = timeNew
    dict_output_tokens['dtime'][i+CTX_LEN] = dtime
    # MIDI note to button
    try:
        but = key_to_button(note)
        b[i+CTX_LEN] = but
    except:
        print("ERROR", b[i+CTX_LEN])
    context = {
      'pitch': torch.tensor(dict_output_tokens['pitch'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
    }
    context = to_device(context, device)
                 
    with torch.inference_mode():
        new_pitch_token = model.gen_pitch_token(context)
    dict_output_tokens['pitch'][i+CTX_LEN] = new_pitch_token

    playNote(new_pitch_token, velocity) 
    visualizer.get_note(new_pitch_token, velocity)
    visualizer.get_button(0, velocity) # button is 0. No button influence, no button visualization

    # add (user_note, pitch, time) to dictionary
    noteOn_dict[but] = (new_pitch_token, timeNew)
    i += 1

  else: # noteOff
    but = key_to_button(note)
    #print("but", but)
    if but in noteOn_dict:

      # get pitch and time in dictionary of accumulated notesOns without noteOff
      pitch, noteOn_time = noteOn_dict[but]
      playNote(pitch, 0)
      visualizer.get_note(pitch, 0)
      visualizer.get_button(but, 0)
      #visualizer.update(noteOn_time)


"""# MIDI IN """

try:
   # Create MIDI input object
    if torch.backends.mps.is_available(): 
        midiin = rtmidi.MidiIn(rtmidi.API_MACOSX_CORE)  #  for Mac
        MIDI_PORT = 0 #  Axiom 0 for Mac
    else:
        midiin = rtmidi.MidiIn(rtmidi.API_LINUX_ALSA)  #  for Linux
        MIDI_PORT = 1 # minicontrol32 in linux

    
    # List available ports
    available_ports = midiin.get_ports()
    
    if available_ports:
        print("Available MIDI input ports:")
        for i, port in enumerate(available_ports):
            print(f"[{i}] {port}")
        # Open first available port
        midiin.open_port(MIDI_PORT) 
      
        print(f"Using MIDI input port: {available_ports[MIDI_PORT]}")

    midiin.set_callback(midiin_callback)

    while True:
      time.sleep(0.0001)
      #visualizer.get_note(60, 100)
      #visualizer.get_button(0, 100)
      visualizer.draw()
except (EOFError, KeyboardInterrupt, SystemExit):
    print("Bye.")

