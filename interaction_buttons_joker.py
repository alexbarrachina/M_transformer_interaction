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

''' DEVICE SPECIFIC PARAMETERS '''
if torch.backends.mps.is_available():
    # CASA
    device = torch.device('mps')
    CTX_LEN = 128 # num notes in context. tokens = CTX_LENGTH * 3
    TOTAL_GEN_LEN = 512 # num notes to generate
    KEY_OFFSET = 48 # esmuc 34, casa 48 
else:
    # ESMUC
    device = torch.device('cuda')
    CTX_LEN = 512 # num notes in context. tokens = CTX_LENGTH * 3
    TOTAL_GEN_LEN = 1024 # num notes to generate
    KEY_OFFSET = 34 # esmuc 34, casa 48 

''' PARAMS '''

TRACES = False
USE_CACHE = False
CACHE_IDLE_TIMEOUT = 2.0  # seconds - clear KV cache after this idle gap

# Joker detection: fast alternation of exactly 2 keys triggers joker mode
JOKER_WINDOW_SIZE = 4       # minimum note-on events to detect the pattern
JOKER_MAX_INTERVAL_MS = 200.0  # max ms between consecutive notes to count as "fast"

# Get sample seed MIDI path
#sample_midi_path = './samples/clairTester_to_end.midi'
#sample_midi_path = './samples/Bach_Prelude_and_Fugue_in_C_major.mid'
sample_midi_path = './samples/Chopin_Nocturnes_Op9No1_In_B_Flat_Minor.mid'
# Base path; save_performance appends _YYYYMMDD_HHMMSS before .mid
output_midi_name = './out/interactive_performance'

# MODELS
#model_name = 'no_dtime_good_reference' # original Genie
#model_name = 'no_dtime_button_concentration_tester_v3' # 12 buttons the button extremes pushes the pitch up/down
#model_name = 'AE_non_linear_compression_12but_tester_v1' # 12 buttons non-linear compression: more control in middle, less at extremes
#model_name = 'AE_non_linear_compression_tester_v1' # 18 buttons
#model_name = 'no_dtime_joker_v1' # 8 buttons + joker
model_name = 'no_dtime_joker_18buttons_v1' # 18 buttons + joker

# Raw MIDI note numbers: control only (no model / no button mapping)
MIDI_NOTE_RESET_CONTEXT = 30
MIDI_NOTE_SAVE_PERFORMANCE = 31

# Audible primer preview: only the tail of the context (model + visualizer still use full CTX_LEN).
PRIMER_PLAYBACK_LAST_N = 40
# >1.0 shortens wall-clock waits during play_primer (musical spacing unchanged in tokens).
PRIMER_PLAYBACK_SPEED = 1


class JokerDetector:
    """Detects fast repetitive alternation of exactly 2 keys and tracks
    which specific keys are the current joker pair.

    Only the 2 keys involved in the fast alternation become joker keys;
    all other keys remain normal buttons. The joker pair stays active as
    long as the performer keeps pressing those keys fast. Once the gap
    exceeds max_interval_ms the pair is cleared. A new fast 2-key pattern
    (possibly with different keys) establishes a new joker pair."""

    def __init__(self, window_size: int = JOKER_WINDOW_SIZE,
                 max_interval_ms: float = JOKER_MAX_INTERVAL_MS):
        self.window_size: int = window_size
        self.max_interval_ms: float = max_interval_ms
        self.history: List[tuple] = []  # (key, timestamp_ms) sliding window
        self.joker_keys: set = set()    # the 2 MIDI keys currently acting as joker (empty = no joker)
        self.last_joker_time_ms: float = 0.0  # timestamp of last joker-key press

    def update(self, key: int, timestamp_ms: float) -> bool:
        """Register a note-on event. Returns True if THIS key is a joker key."""

        # Expire joker pair if the performer stopped pressing them fast
        if self.joker_keys and (timestamp_ms - self.last_joker_time_ms) > self.max_interval_ms:
            self.joker_keys = set()

        # If this key is already one of the active joker keys, keep it joker
        if key in self.joker_keys:
            self.last_joker_time_ms = timestamp_ms
            self._push_history(key, timestamp_ms)
            return True

        # Not a current joker key — add to history for new-pattern detection
        self._push_history(key, timestamp_ms)

        # Check if the history now shows a new fast 2-key alternation
        if self._detect_pattern():
            self.joker_keys = set(h[0] for h in self.history)
            self.last_joker_time_ms = timestamp_ms
            return True  # this key is part of the newly detected pair

        return False

    def _push_history(self, key: int, timestamp_ms: float) -> None:
        self.history.append((key, timestamp_ms))
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def _detect_pattern(self) -> bool:
        if len(self.history) < self.window_size:
            return False
        # All consecutive intervals must be fast
        for idx in range(1, len(self.history)):
            if self.history[idx][1] - self.history[idx - 1][1] > self.max_interval_ms:
                return False
        # Exactly 2 distinct keys
        keys = set(h[0] for h in self.history)
        return len(keys) == 2

    def reset(self) -> None:
        self.history.clear()
        self.joker_keys = set()
        self.last_joker_time_ms = 0.0

''' MODEL '''

cfg = get_model_hparams(model_name)
model = load_model(model_name=model_name, cfg=cfg )
model.to(device)
model.eval()

NUM_BUTTONS = cfg['num_buttons']


'''THREADING'''
# Add these at the global scope after your imports
buffer_lock = Lock()
save_lock = Lock()
# pygame/SDL must run on the main thread (macOS); MIDI callback is on another thread
reset_requested = Event()

'''VISUALIZER'''
visualizer = Visualizer(button_slots=cfg['num_buttons'])

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
          print("resetting context")
          reset_requested.set()

def key_to_button(key):
    key = key - KEY_OFFSET # keyboard starts at C = 48
    button = key #% 20 # 12 white keys, 8 black keys
    toWhite = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 10, 10, 11, 11, 12, 12, 13, 14, 14, 15, 15, 16, 17, 17, 18, 18, 19, 19, 20, 21, 21, 22,22,23,23,24]
    if key < 0 or key >= len(toWhite):
        button = 0
    else:
        button = toWhite[button] # convert to white key index
    button = max(0, min(cfg['num_buttons'] - 1, button))
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

  if TRACES:
    print("dtime_save", dict_output_tokens['dtime'][CTX_LEN:CTX_LEN + i])

  # generate a midi file from generated pitches
  song_d = dict_to_song(context)

  stamped_path = output_midi_name + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  detailed_stats = ms_SONG_to_MIDI_Converter(song_d, output_file_name=stamped_path,
                                                            timings_multiplier=1
                                                            )
  print("saved performance", stamped_path + '.mid')

def reset_context():
    global i
    global dict_output_tokens, dict_input_tokens
    global kv_cache

    with buffer_lock:
        i = 0
        kv_cache = None
        joker_detector.reset()
        # Reset and extend dict_output_tokens to accommodate TOTAL_GEN_LEN + CTX_LEN tokens
        for key in dict_input_tokens.keys():
            extended_list = dict_input_tokens[key].copy()
            extended_list.extend([0] * (TOTAL_GEN_LEN + CTX_LEN - len(extended_list)))
            dict_output_tokens[key] = extended_list
        #visualizer.play_primer(playNote, last_n=PRIMER_PLAYBACK_LAST_N,
        #                              playback_speed=PRIMER_PLAYBACK_SPEED)

''' VARIABLES '''
context = None
timeLast = 0
i = 0 # num current tokens in context after CTX_LEN
noteOn_dict = {} # note: (pitch, timeIn, button)
first_note = True
kv_cache = None
last_gen_time: float = 0.0
joker_detector = JokerDetector()

''' BUILD CTX '''
# Load seed MIDI
dict_input_tokens, num_notes = midi_to_dict(sample_midi_path) # tokens, without vel

# Extend dict_output_tokens to accommodate TOTAL_GEN_LEN + CTX_LEN tokens
dict_output_tokens = {}
for key in dict_input_tokens.keys():
    # Create a list with enough space for all generated tokens
    extended_list = dict_input_tokens[key].copy()
    # Extend with zeros to ensure we have enough space
    extended_list.extend([0] * (TOTAL_GEN_LEN + CTX_LEN - len(extended_list)))
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
  
with torch.inference_mode():
    e = model.encoder(context) # encoder output (batch, seq_len)
    b = model.real_to_discrete(e).squeeze(0) # generate buttons (batch, seq_len)
    b = b.clone().detach().tolist()

# Full CTX_LEN in the primer strip and in the model; audible tail runs after MIDI opens (main loop).
visualizer.primer(dict_input_tokens['pitch'][:CTX_LEN], dict_input_tokens['dtime'][:CTX_LEN ],
                  b[:CTX_LEN], dict_input_tokens['dur'][:CTX_LEN])

def manageNote(note, velocity): 
  global context  # Access the global context
  global timeLast # time of last note, global variable
  global b # button array
  global i # num current tokens in context after CTX_LEN
  global dict_output_tokens # output tokens
  global noteOn_dict # note: (pitch, timeIn, button)
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

    # Update position token

    dtime = max(0, min(127, int(timeNew) - int(timeLast))) # time difference from previous events, but trunk to maximum 127
    if first_note:
        dtime = 0
        first_note = False

    timeLast = timeNew
    dict_output_tokens['dtime'][i+CTX_LEN] = dtime
    # MIDI note to button — joker detector overrides when fast 2-key alternation
    timestamp_ms = time.perf_counter() * 1000
    is_joker = joker_detector.update(note, timestamp_ms)

    if is_joker:
        but = model.joker_button_idx
        if TRACES:
            print("JOKER detected")
    else:
        try:
            but = key_to_button(note)
        except:
            but = 0
            print("ERROR key_to_button", note)

    b[i+CTX_LEN] = but
    context = {
      'dtime': torch.tensor(dict_output_tokens['dtime'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'pitch': torch.tensor(dict_output_tokens['pitch'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'dur': torch.tensor(dict_output_tokens['dur'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'button': torch.tensor(b[i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0)
    }
    context = to_device(context, device)
    if TRACES:
        print("dtime")
    with torch.inference_mode():
        if USE_CACHE:
            new_pitch_token, kv_cache = model.gen_pitch_token(context, cache=kv_cache, use_cache=True)
        else:
            new_pitch_token = model.gen_pitch_token(context)
        last_gen_time = time.perf_counter()
    dict_output_tokens['pitch'][i+CTX_LEN] = new_pitch_token

    playNote(new_pitch_token, velocity) 
    visualizer.get_note(new_pitch_token, velocity)
    if is_joker:
        visualizer.get_joker(velocity)
    else:
        visualizer.get_button(but, velocity)

    # add (pitch, time, button, is_joker) to dictionary using original MIDI note as key
    noteOn_dict[note] = (new_pitch_token, timeNew, but, is_joker)
    i += 1

  else: # noteOff
    # Use original MIDI note as key to find corresponding noteOn
    if note in noteOn_dict:

      # get pitch, time, button, and joker flag from dictionary of accumulated notesOns without noteOff
      pitch, noteOn_time, but, was_joker = noteOn_dict[note]
      playNote(pitch, 0)
      visualizer.get_note(pitch, 0)
      if was_joker:
          visualizer.get_joker(0)
      else:
          visualizer.get_button(but, 0)
      # Remove from dictionary to allow the same note to be played again
      del noteOn_dict[note]
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
      visualizer.draw()
except (EOFError, KeyboardInterrupt, SystemExit):
    print("Bye.")

