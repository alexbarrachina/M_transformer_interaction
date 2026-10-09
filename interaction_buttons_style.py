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


import time
import sys
from tkinter.constants import FALSE
import fluidsynth
import os
import atexit
# pip install pyfluidsynth
from typing import Optional, List, Tuple
from rtmidi.midiconstants import NOTE_ON, NOTE_OFF
from rtmidi.midiutil import open_midiinput
import rtmidi
# pip install python-rtmidi
from threading import Lock, Event
from pynput import keyboard as pkeyboard

from sympy import false, true
import torch

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter
from visualizer import Visualizer
from timing import LiveTimingContext, TIME_UNIT

TRACES = False
USE_CACHE = False
CACHE_IDLE_TIMEOUT = 2.0  # seconds - clear KV cache after this idle gap
XINXE_INTERFACE = False

# Joker detection: fast alternation of exactly 2 keys triggers joker mode
JOKER_WINDOW_SIZE = 4       # minimum note-on events to detect the pattern
JOKER_MAX_INTERVAL_MS = 200.0  # max ms between consecutive notes to count as "fast"

JOKER_FORCED_KEYS = {36, 37}  # these MIDI keys are always joker buttons, regardless of alternation

TEMPERATURE = 1#0.0001
SHOW_EVENT_NUMBERS = True

''' DEVICE SPECIFIC PARAMETERS '''
if torch.backends.mps.is_available():
    # CASA
    device = torch.device('mps')
    CTX_LEN = 128 # num notes in context.
    TOTAL_GEN_LEN = 800 # num notes to generate
    if XINXE_INTERFACE:
        KEY_OFFSET = 60 # esmuc 34, casa 48 
    else:
        KEY_OFFSET = 48 # esmuc 34, casa 48 
else:
    # ESMUC
    device = torch.device('cuda')
    CTX_LEN = 512 # num notes in context. 
    TOTAL_GEN_LEN = 1024 # num notes to generate
    if XINXE_INTERFACE:
        KEY_OFFSET = 60 # esmuc 34, casa 48 
    else:
        KEY_OFFSET = 34 # esmuc 34, casa 48 

''' MODEL '''

#model_name = 'AE_style_jokerParam_v1' #AE_style_v1
model_name = os.environ.get('MODEL_NAME', 'AE_style_jokerParam_dtime_tester_v1')
cfg = get_model_hparams(model_name)
if os.environ.get('CHECKPOINT_PATH'):
    cfg['ckpt_file_name'] = os.environ['CHECKPOINT_PATH']
model = load_model(model_name=model_name, cfg=cfg )
model.to(device)
model.eval()
live_timing = None

''' PARAMS '''
# Get sample seed MIDI path
sample_midi_path1 = './samples/Bach_Prelude_and_Fugue_in_C_major.mid'
sample_midi_path2 = './samples/clairTester_to_end.midi'
sample_midi_path3 = './samples/Chopin_Nocturnes_Op9No1_In_B_Flat_Minor.mid'
sample_midi_path4 = './samples/Scott_Cyril_Lotus_Land.mid'
sample_midi_path5 = './samples/Satie_Gymnopedie_No1_no_melody.midi'

sample_midi_path_init = sample_midi_path5
STYLE_IDX_INIT = 5

# Style prompts for keys 1, 2, 3 — set each path to a different MIDI to transfer style on-the-fly.
style_prompt_midi_paths: List[str] = [
    sample_midi_path1,  # key 1
    sample_midi_path2,  # key 2
    sample_midi_path3,  # key 3
    sample_midi_path4,  # key 4
    sample_midi_path5,  # key 5
]
output_midi_name = './out/interactive_buttons_style'

# Audible primer preview: only the tail of the context (model + visualizer still use full CTX_LEN).
PRIMER_PLAYBACK_LAST_N = 40
# >1.0 shortens wall-clock waits during play_primer (musical spacing unchanged in tokens).
PRIMER_PLAYBACK_SPEED = 1

NUM_BUTTONS = cfg['num_buttons']
HIGHLIGHT_MOTIF_LEN = 30
MOTIF_STYLE_KEY = '6'

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


'''THREADING'''
# Add these at the global scope after your imports
buffer_lock = Lock()
save_lock = Lock()
# pygame/SDL must run on the main thread (macOS); MIDI callback is on another thread
reset_requested = Event()
space_joker_held = Event()
memory_key_held = Event()
joker_toggle_active = Event()
toggle_key_held = Event()
timing_key_held = Event()

def make_controls_legend() -> List[Tuple[str, str]]:
    joker_state = 'ON' if joker_toggle_active.is_set() else 'OFF'
    controls = [
        ('SPACE', 'Hold for Joker'),
        ('T', f'Joker {joker_state}'),
        ('M', 'Capture memory'),
        ('S', 'Save'),
        ('R', 'Reset'),
        ('1-5', 'Styles'),
        ('6', 'Memory'),
    ]
    if cfg.get('timing_enabled', False):
        enabled = live_timing is None or live_timing.enabled
        controls.append(('D', f"Timing {'ON' if enabled else 'OFF'}"))
    return controls

'''VISUALIZER'''
visualizer = Visualizer(
    button_slots=cfg['num_buttons'],
    controls_legend=make_controls_legend(),
    show_event_numbers=SHOW_EVENT_NUMBERS
)

'''MIDI IN CALLBACK'''
midi_event_time: Optional[float] = None

def midiin_callback(event, data=None):
    global midi_event_time
    message, deltatime = event
    # RtMidi reports elapsed time since the previous MIDI message. Advance the
    # clock for every message, including releases, before pitch inference runs.
    midi_event_time = (time.perf_counter() if midi_event_time is None
                       else midi_event_time + deltatime)
    note_time = midi_event_time if live_timing is not None else None

    if message[0] & 0xF0 == NOTE_ON:
        status, note, velocity = message
        #channel = (status & 0xF) + 1
        with buffer_lock: # lock to avoid race condition
            manageNote(note, velocity, note_time)

    if message[0] & 0xF0 == NOTE_OFF: 
        status, note, velocity = message
        with buffer_lock:
            manageNote(note, 0, note_time)
    
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

    # using xinxe interface
    if XINXE_INTERFACE:
        button = key
    else:
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

  detailed_stats = ms_SONG_to_MIDI_Converter(song_d, output_file_name = output_midi_name,
                                                            timings_multiplier=1
                                                            )
  print("saved performance")

def reset_context():
    global i
    global dict_output_tokens, dict_input_tokens
    global kv_cache
    global b

    with buffer_lock:
        if live_timing is not None:
            # Restart the bounded generation context, while retaining the full
            # performance and noteOn_dict so held notes can still be released.
            live_timing.reset()
            kv_cache = None
            joker_detector.reset()
            return
        i = 0
        kv_cache = None
        # Reset and extend dict_output_tokens to accommodate TOTAL_GEN_LEN + CTX_LEN tokens
        for key in dict_input_tokens.keys():
            extended_list = dict_input_tokens[key].copy()
            extended_list.extend([0] * (TOTAL_GEN_LEN + CTX_LEN - len(extended_list)))
            dict_output_tokens[key] = extended_list
        # Rebuild b from seed and extend to match TOTAL_GEN_LEN + CTX_LEN
        seed_context = {
            'dtime': torch.tensor(dict_input_tokens['dtime'], dtype=torch.long).unsqueeze(0).to(device),
            'pitch': torch.tensor(dict_input_tokens['pitch'], dtype=torch.long).unsqueeze(0).to(device),
            'dur':   torch.tensor(dict_input_tokens['dur'],   dtype=torch.long).unsqueeze(0).to(device),
        }
        with torch.inference_mode():
            e = model.encoder(seed_context)
            b = model.real_to_discrete(e).squeeze(0).clone().detach().tolist()
        b.extend([0] * (TOTAL_GEN_LEN + CTX_LEN - len(b)))
        #visualizer.play_primer(playNote, last_n=PRIMER_PLAYBACK_LAST_N,
        #                       playback_speed=PRIMER_PLAYBACK_SPEED)

        joker_detector.reset()

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
dict_input_tokens, num_notes = midi_to_dict(sample_midi_path_init) # tokens, without vel

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

style_seq_len = int(cfg.get('style_seq_len', CTX_LEN))
MOTIF_STYLE_IDX = len(style_prompt_midi_paths)

def _encode_style_pitch_list(pitch_list: List[int], repeat_to_style_len: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_len = min(len(pitch_list), style_seq_len)
    if valid_len <= 0:
        _spp = torch.full((1, style_seq_len), PAD_IDX, dtype=torch.long, device=device)
        _smask = torch.zeros((1, style_seq_len), dtype=torch.bool, device=device)
    elif repeat_to_style_len:
        _motif = torch.tensor(pitch_list[:valid_len], dtype=torch.long, device=device)
        _repeat_count = (style_seq_len + valid_len - 1) // valid_len
        _spp = _motif.repeat(_repeat_count)[:style_seq_len].unsqueeze(0)
        _smask = torch.ones((1, style_seq_len), dtype=torch.bool, device=device)
    else:
        _pitch_list = pitch_list[:style_seq_len]
        if valid_len < style_seq_len:
            _pitch_list = _pitch_list + [PAD_IDX] * (style_seq_len - valid_len)
        _mask_list = [True] * valid_len + [False] * (style_seq_len - valid_len)
        _spp = torch.tensor(_pitch_list, dtype=torch.long).unsqueeze(0).to(device)
        _smask = torch.tensor(_mask_list, dtype=torch.bool).unsqueeze(0).to(device)

    with torch.inference_mode():
        _sctx = model.encode_style(_spp, _smask)
    return _sctx, _smask

# Pre-encode all MIDI style prompts to avoid latency on style switch
style_contexts: List[torch.Tensor] = []
style_context_masks_list: List[torch.Tensor] = []
for _spath in style_prompt_midi_paths:
    _style_tokens, _ = midi_to_dict(_spath)
    _sctx, _smask = _encode_style_pitch_list(_style_tokens['pitch'][:style_seq_len])
    style_contexts.append(_sctx)
    style_context_masks_list.append(_smask)
    print(f"Style {len(style_contexts)} encoded: {_spath}")

active_style_idx: int = STYLE_IDX_INIT - 1 # -1 because list is 0-indexed
highlight_motif_pitch_tokens: List[int] = []
motif_style_ready = False
style_contexts.append(style_contexts[active_style_idx])
style_context_masks_list.append(style_context_masks_list[active_style_idx])
style_context = style_contexts[active_style_idx]
style_context_mask = style_context_masks_list[active_style_idx]

def capture_highlight_motif() -> None:
    global highlight_motif_pitch_tokens
    global motif_style_ready
    global kv_cache

    with buffer_lock:
        end_idx = CTX_LEN + i
        start_idx = max(CTX_LEN, end_idx - HIGHLIGHT_MOTIF_LEN)
        _pitch_list = list(dict_output_tokens['pitch'][start_idx:end_idx])

    if len(_pitch_list) <= 0:
        print("No generated pitch tokens yet for highlight motif")
        return

    _sctx, _smask = _encode_style_pitch_list(_pitch_list, repeat_to_style_len=True)
    with buffer_lock:
        highlight_motif_pitch_tokens = _pitch_list
        style_contexts[MOTIF_STYLE_IDX] = _sctx
        style_context_masks_list[MOTIF_STYLE_IDX] = _smask
        motif_style_ready = True
        kv_cache = None
    print(f"Highlight motif saved: {len(highlight_motif_pitch_tokens)} pitch tokens. Press {MOTIF_STYLE_KEY} to activate.")

with torch.inference_mode():
    #style_context = model.encode_style(style_prompt_pitch, style_context_mask)
    e = model.encoder(context) # encoder output (batch, seq_len)
    b = model.real_to_discrete(e).squeeze(0) # generate buttons (batch, seq_len)
    b = b.clone().detach().tolist()
    b.extend([0] * (TOTAL_GEN_LEN + CTX_LEN - len(b)))

if cfg.get('timing_enabled', False):
    live_timing = LiveTimingContext(
        dict_input_tokens['pitch'], b, dict_input_tokens['dtime'], CTX_LEN
    )
# Full CTX_LEN in the primer strip and in the model; audible tail runs after MIDI opens (main loop).
visualizer.primer(dict_input_tokens['pitch'][:CTX_LEN], dict_input_tokens['dtime'][:CTX_LEN],
                  b[:CTX_LEN], dict_input_tokens['dur'][:CTX_LEN])

def manageNote(note, velocity, event_time=None):
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
  global active_style_idx
  
  if TRACES:
    print("key", note)

  # Use the MIDI event clock for spacing and note duration. Direct calls can
  # still use the local clock; keep raw seconds for long phrase detection.
  now = time.perf_counter() if event_time is None else event_time
  timeNew = now / TIME_UNIT

  if velocity > 0: # noteOn
    if USE_CACHE and kv_cache is not None and (time.perf_counter() - last_gen_time) > CACHE_IDLE_TIMEOUT:
        kv_cache = None
    # Update position token

    dtime = max(0, int(timeNew) - int(timeLast))
    if live_timing is None:
        dtime = min(127, dtime)
    else:
        # The recording can grow across phrase resets and preserves long pauses.
        for values in (*dict_output_tokens.values(), b):
            if len(values) <= i + CTX_LEN:
                values.extend([0] * (i + CTX_LEN + 1 - len(values)))
    if first_note:
        dtime = 0
        first_note = False

    timeLast = timeNew
    dict_output_tokens['dtime'][i+CTX_LEN] = dtime
    # Keep tracking fast 2-key alternation while Space or T forces Joker notes.
    timestamp_ms = now * 1000
    repeated_pair_joker = joker_detector.update(note, timestamp_ms)
    is_joker = (space_joker_held.is_set() or joker_toggle_active.is_set()
                or repeated_pair_joker or note in JOKER_FORCED_KEYS)

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
    if live_timing is not None:
        # Features are computed once on arrival, aligned with this button and
        # target pitch. Older notes keep their original timing reference scale.
        context, restarted = live_timing.begin_note(now, but)
        if restarted:
            # The context now contains the primer and a new unknown interval.
            # Style, recording, timing toggle and held-note releases stay intact.
            kv_cache = None
            joker_detector.reset()
            joker_detector.update(note, now * 1000)
    else:
        context = {
          'dtime': torch.tensor(dict_output_tokens['dtime'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
          'pitch': torch.tensor(dict_output_tokens['pitch'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
          'dur': torch.tensor(dict_output_tokens['dur'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
          'button': torch.tensor(b[i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0)
        }
    context = to_device(context, device)
    if TRACES:
        print("ctx", i+CTX_LEN+1, "of", TOTAL_GEN_LEN+CTX_LEN)
    with torch.inference_mode():
        if USE_CACHE:
            new_pitch_token, kv_cache = model.gen_pitch_token(
                context,
                style_context=style_contexts[active_style_idx],
                style_context_mask=style_context_masks_list[active_style_idx],
                cache=kv_cache
            )
        else:
            new_pitch_token, _ = model.gen_pitch_token(
                context,
                style_context=style_contexts[active_style_idx],
                style_context_mask=style_context_masks_list[active_style_idx]
            , temperature=TEMPERATURE)
        last_gen_time = time.perf_counter()
    dict_output_tokens['pitch'][i+CTX_LEN] = new_pitch_token
    if live_timing is not None:
        # Replace the target placeholder so it becomes the next note's past pitch.
        live_timing.finish_note(new_pitch_token)

    playNote(new_pitch_token, velocity) 
    visualizer.get_note(new_pitch_token, velocity, is_joker=is_joker)
    #visualizer.get_button(but, velocity)

    if is_joker:
        visualizer.get_joker(velocity)
    else:
        visualizer.get_button(but, velocity)

    noteOn_dict[note] = (new_pitch_token, timeNew, but, is_joker)
    
    i += 1

  else: # noteOff
    # Use original MIDI note as key to find corresponding noteOn
    if note in noteOn_dict:

      # get pitch, time, and button from dictionary of accumulated notesOns without noteOff
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

"""# KEYBOARD LISTENER — Space holds Joker, T toggles Joker, M captures memory """
def _on_key_press(key: pkeyboard.Key) -> None:
    global active_style_idx
    global kv_cache
    global style_context
    global style_context_mask
    global motif_style_ready
    if key == pkeyboard.Key.space:
        space_joker_held.set()
        return
    try:
        c = key.char  # type: ignore[union-attr]
        if c is not None and c.lower() == 'd' and live_timing is not None:
            if not timing_key_held.is_set():
                timing_key_held.set()
                with buffer_lock:
                    # Keep collecting features in both modes. Cached activations
                    # encode the old mode, so rebuild from stored context next time.
                    live_timing.enabled = not live_timing.enabled
                    kv_cache = None
                visualizer.controls_legend = make_controls_legend()
                print(f"Timing {'ON' if live_timing.enabled else 'OFF'}")
            return
        if c is not None and c.lower() == 'm':
            if not memory_key_held.is_set():
                memory_key_held.set()
                capture_highlight_motif()
            return
        if c is not None and c.lower() == 't':
            if not toggle_key_held.is_set():
                toggle_key_held.set()
                if joker_toggle_active.is_set():
                    joker_toggle_active.clear()
                else:
                    joker_toggle_active.set()
                visualizer.controls_legend = make_controls_legend()
                print(f"Joker mode {'ON' if joker_toggle_active.is_set() else 'OFF'}")
            return
        if c in ('1', '2', '3', '4', '5'):
            active_style_idx = int(c) - 1
            style_context = style_contexts[active_style_idx]
            style_context_mask = style_context_masks_list[active_style_idx]
            kv_cache = None
            print(f"Style {c} active: {style_prompt_midi_paths[active_style_idx]}")
        if c == MOTIF_STYLE_KEY:
            if motif_style_ready:
                active_style_idx = MOTIF_STYLE_IDX
                style_context = style_contexts[active_style_idx]
                style_context_mask = style_context_masks_list[active_style_idx]
                kv_cache = None
                print("Highlight motif active")
            else:
                print("No highlight motif saved yet. Press M first.")
        if c in ('r'):
            print("resetting context")
            reset_context()
        if c in ('s'):
            save_performance()
            print("saved performance")
    except AttributeError:
        pass

def _on_key_release(key: pkeyboard.Key) -> None:
    if key == pkeyboard.Key.space:
        space_joker_held.clear()
    else:
        char = getattr(key, 'char', None)
        if char is not None and char.lower() == 'm':
            memory_key_held.clear()
        elif char is not None and char.lower() == 't':
            toggle_key_held.clear()
        elif char is not None and char.lower() == 'd':
            timing_key_held.clear()

_key_listener = pkeyboard.Listener(on_press=_on_key_press, on_release=_on_key_release)
_key_listener.start()

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
        for port_index, port in enumerate(available_ports):
            print(f"[{port_index}] {port}")
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
