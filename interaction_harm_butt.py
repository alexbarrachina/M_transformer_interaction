#!/usr/bin/env python3
"""
Real-Time  interaction with buttons and harmony features

"""

import sys
import time
import fluidsynth
import argparse
from typing import List, Tuple
from harmony_extractor import harmony_visualizer, estimate_key_from_pitches

from threading import Lock

import torch
import pygame
from pygame.locals import *


from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter

'''PARAMETERS'''
global_key = None  # Will be auto-detected from MIDI file
visualize = True
chord_threshold = 0.3
device_id = 0
#sample_midi_path = './samples/clairTester_to_end.midi'
sample_midi_path = './samples/Bach_Prelude_and_Fugue_in_C_major.mid'
output_midi_name = './out/interactive_performance'
temperature = 0.0001

CTX_LEN = 128 # num notes in context. tokens = CTX_LENGTH * 3
TOTAL_GEN_LEN = 512 # num notes to generate
TRACES = False


'''KEY MAPPING'''
KEY_MAPPING = {
    K_a: 0, K_s: 1, K_d: 2, K_f: 3, K_g: 4, K_h: 5, K_j: 6, K_k: 7, K_l: 8, K_z: 9, K_x: 10, K_c: 11,
}

'''THREADING'''
# Add these at the global scope after your imports
buffer_lock = Lock()
save_lock = Lock()

"""# PYGAME INIT """
pygame.init()

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

def reset_context():
    global i
    global dict_output_tokens, dict_input_tokens

    i = 0
    dict_output_tokens['dtime'] = dict_input_tokens['dtime'] 
    dict_output_tokens['pitch'] = dict_input_tokens['pitch'] 
    dict_output_tokens['dur'] = dict_input_tokens['dur'] 
    dict_output_tokens['button'] = dict_input_tokens['button'] 

''' DEVICE '''
#device = torch.device('cpu')
device = torch.device('mps') 

''' MODEL '''
model_name = 'no_dtime_good_reference'
cfg = get_model_hparams(model_name)
model = load_model(model_name=model_name, cfg=cfg )
model.to(device)
model.eval()

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
pitch_buffer = dict_input_tokens['pitch'].copy()  # Start with original sequence

# Estimate global key from pitches
global_key = estimate_key_from_pitches(dict_input_tokens['pitch'])
from harmony_extractor import PITCH_NAMES
key_root = global_key % 12
key_mode = 'major' if global_key < 12 else 'minor'
print(f"Detected key: {PITCH_NAMES[key_root]} {key_mode} (encoded: {global_key})")

# Build context tokens
context = {
    'dtime': torch.tensor(dict_input_tokens['dtime'], dtype=torch.long).unsqueeze(0),
    'pitch': torch.tensor(dict_input_tokens['pitch'], dtype=torch.long).unsqueeze(0),
    'dur': torch.tensor(dict_input_tokens['dur'], dtype=torch.long).unsqueeze(0)
    }
context = to_device(context, device)

context = to_device(context, device)
  
with torch.inference_mode():
    e = model.encoder(context) # encoder output (batch, seq_len)
    buttons = model.real_to_discrete(e).squeeze(0) # generate buttons (batch, seq_len)
    buttons = buttons.clone().detach().tolist()

    print(f"Initializing MIDI input device {device_id}...")

''' HARMONY VISUALIZER '''
visualizer = harmony_visualizer(
    global_key=global_key,
    buffer_size=32,
    chord_threshold=chord_threshold,
    visualize=visualize
)
    
def manage_button_input(key, velocity): 
  global context  # Access the global context
  global timeLast # time of last note, global variable
  global original_buttons # arrows array
  global buttons # arrows array
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

    # The arrow at index [N-1] guides the transition to N.
    # We are about to generate note at [i+CTX_LEN], so we set arrow at [i+CTX_LEN-1]
    button_idx = i + CTX_LEN - 1
    # Update button with user input
    if button_idx < len(buttons):  
        buttons[button_idx] = but
    else:
        buttons.append(but)

    context = {
      'pitch': torch.tensor(pitch_buffer[i:i+CTX_LEN], dtype=torch.long).unsqueeze(0),
      'button': torch.tensor(buttons[i:i+CTX_LEN], dtype=torch.long).unsqueeze(0)
    }
    context = to_device(context, device)
                 
    with torch.inference_mode():
        new_pitch_token = model.gen_pitch_token(context, temperature=temperature) # TODO: temperature 0.00001
    if TRACES:
        print("new_pitch_token", new_pitch_token)

    # Store the generated pitch
    if i+CTX_LEN < len(pitch_buffer):
        pitch_buffer[i+CTX_LEN] = new_pitch_token
    else:
        pitch_buffer.append(new_pitch_token)

    playNote(new_pitch_token, velocity) 
    visualizer.get_note(new_pitch_token, velocity)

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


"""# INPUT LOOP (QWERTY) """

try:
    print("Starting interaction loop. Use QWERTY keys A,S,D,F,G,H,J,K,L,Z,X,C for buttons.")
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
                    #save_performance()
                    #os._exit(1)                              
                elif event.key == K_SPACE: # Reset
                    if TRACES:
                        print("resetting context")
                    reset_context(dict_input_tokens)
                elif event.key == 1073742051: # Reset
                    if TRACES:
                        print("resetting context")
                    #reset_context(dict_input_tokens2)
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
    pygame.quit()
    sys.exit()



