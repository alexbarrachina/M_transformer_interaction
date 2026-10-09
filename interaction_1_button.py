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
import datetime
import sys
import fluidsynth
import os 
import atexit
import json
import pygame
# pip install pyfluidsynth
from typing import Optional, List
from rtmidi.midiconstants import NOTE_ON, NOTE_OFF
from rtmidi.midiutil import open_midiinput
import rtmidi
# pip install python-rtmidi
from threading import Lock, Event

from sympy import false
import torch

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter
from visualizer import Visualizer
from tension_joker import TensionControls, read_tension_performance

TRACES = False
USE_CACHE = False
CACHE_IDLE_TIMEOUT = 2.0  # seconds - clear KV cache after this idle gap

''' DEVICE '''
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


''' MODEL '''
model_name = os.environ.get('MODEL_NAME', 'Dec_no_conditioning_v1')
cfg = get_model_hparams(model_name)
is_tension_joker = cfg['model_type'] == 'D_Tension_Joker'
is_base_joker = cfg['model_type'] in ('D_Base_Joker', 'D_Tension_Joker')
model = load_model(model_name=model_name, cfg=cfg,
                   compile_mode='none' if is_base_joker else 'max-autotune')
model.to(device)
model.eval()
if is_base_joker:
    USE_CACHE = False

''' PARAMS '''
# Get sample seed MIDI path
sample_midi_path = os.environ.get('PRIMER_MIDI', './samples/Bach_Prelude_and_Fugue_in_C_major.mid')
#sample_midi_path = './samples/Chopin_Nocturnes_Op9No1_In_B_Flat_Minor.mid'
# Base path; save_performance appends _YYYYMMDD_HHMMSS before .mid
output_midi_name = './out/interactive_performance'
# Raw MIDI note numbers: control only (no model / no button mapping)
MIDI_NOTE_RESET_CONTEXT = 30
MIDI_NOTE_SAVE_PERFORMANCE = 31
JOKER_MIDI_NOTE = int(os.environ.get('JOKER_MIDI_NOTE', '48'))

CTX_LEN = min(300, cfg['seq_len']) # num notes in context. tokens = CTX_LENGTH * 3
# Audible primer preview: only the tail of the context (model + visualizer still use full CTX_LEN).
PRIMER_PLAYBACK_LAST_N = 40
# >1.0 shortens wall-clock waits during play_primer (musical spacing unchanged in tokens).
PRIMER_PLAYBACK_SPEED = 1
TOTAL_GEN_LEN = 1024 # num notes to generate

'''THREADING'''
# Add these at the global scope after your imports
buffer_lock = Lock()
save_lock = Lock()
# pygame/SDL must run on the main thread (macOS); MIDI callback is on another thread
reset_requested = Event()

'''VISUALIZER'''
visualizer = Visualizer(button_slots=cfg.get('num_buttons', 12),
    controls_legend=[('0–4', 'tension'), ('space', 'free'), ('[ / ]', 'guidance')]
    if is_tension_joker else None,
    status_text='Loading tension controls…' if is_tension_joker else '')

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
        with buffer_lock:
            manageNote(note, 0)
    
    if message[0] & 0xF0 == 176:  # 176 is the status for control change

        if message[1] == 18 and message[2] > 0: # Using REC button as a trigger to save performance
          with save_lock:
            print("saving performance")
            save_performance()
            sys.exit(0)

        if message[1] == 17 and message[2] > 0: # Using PLAY button as a trigger to reset the context
          print("resetting context")
          reset_requested.set()

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

  if i <= 0:
      print("nothing recorded to save")
      return

  # Only the generated continuation (indices CTX_LEN .. CTX_LEN+i-1), not the seed primer.
  context = {
      'dtime': dict_output_tokens['dtime'][CTX_LEN:CTX_LEN + i],
      'pitch': dict_output_tokens['pitch'][CTX_LEN:CTX_LEN + i],
      'dur': dict_output_tokens['dur'][CTX_LEN:CTX_LEN + i],
    }
  if is_tension_joker:
      context['vel'] = dict_output_tokens['vel'][CTX_LEN:CTX_LEN + i]
      context['chan'] = dict_output_tokens['chan'][CTX_LEN:CTX_LEN + i]

  if TRACES:
    print("dtime_save", dict_output_tokens['dtime'][CTX_LEN:CTX_LEN + i])

  song_d = dict_to_song(context, force_vel=not is_tension_joker)

  stamped_path = output_midi_name + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  detailed_stats = ms_SONG_to_MIDI_Converter(song_d, output_file_name=stamped_path,
                                                            timings_multiplier=1
                                                            )
  print("saved performance", stamped_path + '.mid')
  if is_tension_joker:
      with open(stamped_path + '.json', 'w') as output:
          json.dump({'model': model_name, 'home_key': home_key, 'controls': control_events,
                     'notes': note_events}, output, indent=2)

def reset_context() -> None:
    global i
    global dict_output_tokens, dict_input_tokens
    global kv_cache
    global b
    global first_note, timeLast, last_gen_time

    with buffer_lock:
        i = 0
        kv_cache = None
        first_note = True
        timeLast = 0
        last_gen_time = 0.0
        for pitch, _, _ in noteOn_dict.values():
            playNote(pitch, 0)
        noteOn_dict.clear()
        for key in dict_input_tokens.keys():
            extended_list = dict_input_tokens[key].copy()
            extended_list.extend([0] * (TOTAL_GEN_LEN + CTX_LEN - len(extended_list)))
            dict_output_tokens[key] = extended_list
        b = torch.zeros(len(dict_output_tokens['pitch']), dtype=torch.long)
        if is_tension_joker:
            tension_controls.reset(CTX_LEN)
            control_events.clear()
            note_events.clear()
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
home_key = HOME_KEY_UNKNOWN
if is_tension_joker:
    performance = read_tension_performance(sample_midi_path, os.environ.get('HOME_KEY'))
    dict_input_tokens = performance.tokens()
    num_notes = len(performance.pitches)
    home_key = performance.home_key
else:
    dict_input_tokens, num_notes = midi_to_dict(sample_midi_path) # tokens, without vel
CTX_LEN = min(CTX_LEN, len(dict_input_tokens['pitch']))
if CTX_LEN == 0:
    raise ValueError('The primer MIDI must contain at least one pitch')
tension_controls = TensionControls(CTX_LEN, CTX_LEN)
control_events = []
note_events = []

dict_output_tokens = {}
for key in dict_input_tokens.keys():
    extended_list = dict_input_tokens[key].copy()
    extended_list.extend([0] * (TOTAL_GEN_LEN + CTX_LEN - len(extended_list)))
    dict_output_tokens[key] = extended_list

b = torch.zeros(len(dict_output_tokens['pitch']), dtype=torch.long)

if TRACES:  
    print("num_notes", num_notes)
# Build context tokens
context = {
    'pitch': torch.tensor(dict_input_tokens['pitch'], dtype=torch.long).unsqueeze(0),
    }
context = to_device(context, device)

# Full CTX_LEN in the primer strip; audible tail runs after MIDI opens (main loop).
visualizer.primer(dict_input_tokens['pitch'][:CTX_LEN], dict_input_tokens['dtime'][:CTX_LEN],
                  b[:CTX_LEN], dict_input_tokens['dur'][:CTX_LEN])

def manageNote(note: int, velocity: int) -> None:
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
  
  if TRACES:
    print("key", note)

  timeNew = time.perf_counter()*1000 /32 # in miliseconds /32 as in midi_to_dict()

  if velocity > 0: # noteOn
    now = time.perf_counter()
    if USE_CACHE and kv_cache is not None and (now - last_gen_time) > CACHE_IDLE_TIMEOUT:
      kv_cache = None

    if note == MIDI_NOTE_RESET_CONTEXT:
        print("resetting context (midi note %d)" % (MIDI_NOTE_RESET_CONTEXT,))
        reset_requested.set()
        return
    if note == MIDI_NOTE_SAVE_PERFORMANCE:
        with save_lock:
            print("saving performance (midi note %d)" % (MIDI_NOTE_SAVE_PERFORMANCE,))
            save_performance()
        sys.exit(0)

    if is_base_joker and note != JOKER_MIDI_NOTE:
        return
    if i + CTX_LEN >= len(dict_output_tokens['pitch']):
        for values in dict_output_tokens.values():
            values.extend([0] * TOTAL_GEN_LEN)
        b = torch.cat((b, b.new_zeros(TOTAL_GEN_LEN)))

    # Update position token
    dtime = max(0, min(127, int(timeNew) - int(timeLast))) # time difference from previous events, but trunk to maximum 127
    if first_note:
        dtime = 0
        first_note = False

    timeLast = timeNew
    dict_output_tokens['dtime'][i+CTX_LEN] = dtime
    # MIDI note to button
    but = 0
    if not is_base_joker:
        try:
            but = key_to_button(note)
        except:
            print("ERROR key_to_button", note)
    b[i+CTX_LEN] = but
    context = {
      'pitch': torch.tensor(dict_output_tokens['pitch'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
    }
    if is_tension_joker:
        context['tension_target'] = torch.tensor([tension_controls.next_targets()], dtype=torch.long)
        context['home_key'] = torch.tensor([home_key], dtype=torch.long)
    context = to_device(context, device)
                 
    with torch.inference_mode():
        if is_tension_joker:
            generated_at = time.perf_counter()
            new_pitch_token = model.gen_pitch_token(context, cfg_weight=tension_controls.cfg_weight)
            note_events.append({'note_index': i, 'pitch': new_pitch_token,
                                'target': tension_controls.target, 'velocity': velocity,
                                'guidance': tension_controls.cfg_weight,
                                'generation_ms': (time.perf_counter() - generated_at) * 1000})
            tension_controls.commit_note()
            dict_output_tokens['vel'][i+CTX_LEN] = velocity
            dict_output_tokens['chan'][i+CTX_LEN] = 0
        elif USE_CACHE:
            new_pitch_token, kv_cache = model.gen_pitch_token(context, cache=kv_cache, use_cache=True)
        else:
            new_pitch_token = model.gen_pitch_token(context)
    last_gen_time = time.perf_counter()
    dict_output_tokens['pitch'][i+CTX_LEN] = new_pitch_token

    playNote(new_pitch_token, velocity) 
    visualizer.get_note(new_pitch_token, velocity, is_joker=is_base_joker)
    if is_base_joker:
        visualizer.get_joker(velocity)
    else:
        visualizer.get_button(0, velocity) # button is 0. No button influence, no button visualization

    # add (user_note, pitch, time) to dictionary
    noteOn_dict[note] = (new_pitch_token, timeNew, i + CTX_LEN)
    i += 1

  else: # noteOff
    if note == MIDI_NOTE_RESET_CONTEXT or note == MIDI_NOTE_SAVE_PERFORMANCE:
        return
    if note in noteOn_dict:

      # get pitch and time in dictionary of accumulated notesOns without noteOff
      pitch, noteOn_time, position = noteOn_dict.pop(note)
      dict_output_tokens['dur'][position] = max(1, int(timeNew - noteOn_time))
      playNote(pitch, 0)
      visualizer.get_note(pitch, 0)
      if is_base_joker:
          visualizer.get_joker(0)
      else:
          visualizer.get_button(0, 0)
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

    startup_primer_done = False
    while True:
      time.sleep(0.0001)
      if not startup_primer_done:
          with buffer_lock:
              visualizer.play_primer(playNote, last_n=PRIMER_PLAYBACK_LAST_N,
                                     playback_speed=PRIMER_PLAYBACK_SPEED)
          startup_primer_done = True
      if reset_requested.is_set():
          reset_requested.clear()
          reset_context()
      #visualizer.get_note(60, 100)
      #visualizer.get_button(0, 100)
      if is_tension_joker:
          # Handle keys on the existing SDL/main thread; no global keyboard listener.
          for event in pygame.event.get():
              if event.type == pygame.QUIT:
                  visualizer.stop()
                  raise SystemExit
              if event.type == pygame.KEYDOWN:
                  with buffer_lock:
                      if tension_controls.handle_key(event.unicode):
                          control_events.append({'note_index': i, 'key': event.unicode,
                                                 'target': tension_controls.target,
                                                 'guidance': tension_controls.cfg_weight})
          visualizer.status_text = tension_controls.status(home_key)
      visualizer.draw(handle_events=not is_tension_joker)
except (EOFError, KeyboardInterrupt, SystemExit):
    print("Bye.")
