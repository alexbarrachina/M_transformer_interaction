#===================================================================================================
# Monster Genie interaction_buttons_style_reins.py Python module (copied from interaction_buttons_style_chords.py)
# Interaction, generating buttons from MIDI keyboard,
# starting with a context extracted from a MIDI file
#
# Reins (pitch_reins.py): each button has an ideal pitch, the button-th note of the
# key's scale (buttons 0, 7, 14 = tonics), in an octave that follows the register.
# A continuous pedal (MIDI CC REINS_PEDAL_CC from its own device PEDAL_PORT_NAME, or Z/X) sets how far the model may move
# away from it: fully pressed plays the ideal pitch, barely pressed lets the model choose
# within +-7 semitones in the direction of the hand. The press reaches the model as a
# Joker note and the window as a logit bias, so the checkpoint is unchanged. Fully
# released (until 2026-10-27 the +-7 window too) sends the button itself: the learned
# map of interaction_buttons_style_chords.py, with no window. H/K move
# the tonic, N toggles major/minor; Return / J / fast alternation stay free Joker notes.
#
# Adds the held chord / change FiLM of AE_style_jokerParam_chords, used as in
# interaction_1_button_chords.py: hold 1,2,4,5,6,7 for the primer's fixed chords
# (shift+2 V/V; Satie also shift+5 V7; CC 16/17/18 of the keyboard toggle 1/4/5), hold Space to change chord (the model chooses
# it; AE_style_jokerParam_change only, otherwise Space does nothing). The last pressed
# control wins; releasing all of them returns to free generation.
# Styles move to q..t (motif y, capture m), reset to Backspace, Joker hold to Return
# and the Joker toggle to J. Keys are read from the visualizer window, so it must have focus.
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
import pygame

from sympy import false, true
import torch

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter
from visualizer import Visualizer
from timing import LiveTimingContext, TIME_UNIT
from harmony.chord_joker import ChordControls
from interaction_1_button_chords import FIXED_CHORDS
from pitch_reins import Reins, pitch_bias

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

# Reins pedal: 11 = expression, 4 = foot controller (64 only for a continuous sustain pedal);
# this setup's pedal sends 6.
REINS_PEDAL_CC = 6
PEDAL_PORT_NAME = 'beat bars controller'  # the pedal is its own MIDI device, opened by (part of) its name
# Keyboard (Axiom) CCs that toggle the chords of keys 1, 4, 5 (I, IV, V): a press turns its chord on
# (and the other CC chord off), pressing it again turns it off. They take over CC 17 / 18,
# until 2026-10-27 PLAY (reset) / REC (save and exit); Backspace / S still reset / save.
CHORD_CCS = {16: '1', 17: '4', 18: '5'}
PEDAL_INVERT = False
PEDAL_RELEASED_BELOW = 10  # raw CC values below this are fully released (the learned map); the
                           # rest is rescaled to (0, 1], so 10 already opens the full +-7 window
PEDAL_SNAP = 0.05   # pedal values this close to 1 snap to it, so "fully pressed" is exact
                    # (until 2026-10-27 values near 0 snapped to "released" too; now PEDAL_RELEASED_BELOW)
PEDAL_STEP = 0.1    # Z/X step, to test without a pedal

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
# Chord style joker (train_style_chords.py); interaction_buttons_style.py runs
# 'AE_style_jokerParam_dtime_tester_v1', which has no chord adapter. Stage 2,
# AE_style_jokerParam_change, adds the Space "change" adapter; until its checkpoint
# exists, stage 1 runs and Space does nothing.
model_name = 'AE_style_jokerParam_change'
#if not os.path.isfile(get_model_hparams(model_name)['ckpt_file_name']):
#    model_name = 'AE_style_jokerParam_chords'
model_name = os.environ.get('MODEL_NAME', model_name)
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

sample_midi_path_init = sample_midi_path2
STYLE_IDX_INIT = 2

# FIXED_CHORDS of interaction_1_button_chords.py follows this order (path5 is the same Satie piece).
sample_midi_paths = [sample_midi_path1, sample_midi_path2, sample_midi_path3,
                     sample_midi_path4, sample_midi_path5]
chord_bank = FIXED_CHORDS[sample_midi_paths.index(sample_midi_path_init)]
chord_legend = '  |  '.join(f'{key}: {chord.name}' for key, chord in chord_bank.items() if len(key) == 1)
chord_legend += '\n' + '  |  '.join(f'{key}: {chord.name}' for key, chord in chord_bank.items() if len(key) > 1)

# Style prompts for keys q..t (number keys hold chords) — set each path to a different MIDI to transfer style on-the-fly.
STYLE_KEYS = ('q', 'w', 'e', 'r', 't')
style_prompt_midi_paths: List[str] = [
    sample_midi_path1,  # key q
    sample_midi_path2,  # key w
    sample_midi_path3,  # key e
    sample_midi_path4,  # key r
    sample_midi_path5,  # key t
]
output_midi_name = './out/interactive_buttons_style_reins'

# Reins key per primer, (tonic pc, minor): Bach and Chopin from the titles, the others
# from harmony/key_track.py (which agrees with both titles). Other primers fall back to key_track.
SAMPLE_KEYS = {
    sample_midi_path1: (0, False),   # C major
    sample_midi_path2: (1, False),   # Db major
    sample_midi_path3: (10, True),   # Bb minor
    sample_midi_path4: (3, True),    # Eb minor
    sample_midi_path5: (2, False),   # D major
}

def primer_key(path: str) -> Tuple[int, bool]:
    if path in SAMPLE_KEYS:
        return SAMPLE_KEYS[path]
    from pathlib import Path
    from harmony.key_track import track_key
    track = track_key(Path(path))
    if track is None or track.global_key is None:
        print(f"No key found for {path}; Reins uses C major")
        return 0, False
    return track.global_key.tonic_pc, track.global_key.mode == 'm'

# Audible primer preview: only the tail of the context (model + visualizer still use full CTX_LEN).
PRIMER_PLAYBACK_LAST_N = 40
# >1.0 shortens wall-clock waits during play_primer (musical spacing unchanged in tokens).
PRIMER_PLAYBACK_SPEED = 1

NUM_BUTTONS = cfg['num_buttons']
HIGHLIGHT_MOTIF_LEN = 30
MOTIF_STYLE_KEY = 'y'
# pygame key names: Space now changes chord, 'r' and 't' are style keys.
JOKER_HOLD_KEY = 'return'
JOKER_TOGGLE_KEY = 'j'
RESET_KEY = 'backspace'
# Reins keys, listed in the status text (the one-row legend is full).
TONIC_DOWN_KEY, TONIC_UP_KEY, MODE_KEY = 'h', 'k', 'n'
PEDAL_LESS_KEY, PEDAL_MORE_KEY = 'z', 'x'
REINS_KEYS_TEXT = 'H/K tonic  |  N major/minor  |  Z/X pedal'

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
joker_key_held = Event()
memory_key_held = Event()
joker_toggle_active = Event()
toggle_key_held = Event()
timing_key_held = Event()

''' CHORD CONTROLS '''
# Held chord / change states aligned with the CTX_LEN+1 window of manageNote. Protect them
# with buffer_lock: keys change them on the main thread, notes read them in the MIDI callback.
chord_controls = ChordControls(CTX_LEN, CTX_LEN, chord_bank,
                               change_key='space' if cfg.get('chord_change_enabled') else None,
                               cc_chords=CHORD_CCS)

def make_controls_legend() -> List[Tuple[str, str]]:
    joker_state = 'ON' if joker_toggle_active.is_set() else 'OFF'
    # Paired entries keep the one-row legend within the 1200 px window.
    controls = [
        ('RETURN', 'Hold for Joker'),
        ('J', f'Joker {joker_state}'),
        ('1-7 / SPACE', 'Hold chord / change') if chord_controls.change_key else ('1-7', 'Hold chords'),
        ('M', 'Capture memory'),
        ('Q-T / Y', 'Styles / Memory'),
        ('S / BACKSPACE', 'Save / Reset'),
    ]
    if cfg.get('timing_enabled', False):
        enabled = live_timing is None or live_timing.enabled
        controls.append(('D', f"Timing {'ON' if enabled else 'OFF'}"))
    return controls

'''VISUALIZER'''
visualizer = Visualizer(
    button_slots=cfg['num_buttons'],
    controls_legend=make_controls_legend(),
    show_event_numbers=SHOW_EVENT_NUMBERS,
    status_text=chord_controls.status() + '\n' + chord_legend
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

        # Chord CCs first: while CHORD_CCS maps 17 / 18 they no longer reset / save below.
        with buffer_lock:  # chord states are read by manageNote
            chord_cc = chord_controls.handle_cc(*message)

        if not chord_cc and message[1] == 18 and message[2] > 0: # Using REC button as a trigger to save performance
          with save_lock:
            print("saving performance")
            save_performance()
            sys.exit(0)

        if not chord_cc and message[1] == 17 and message[2] > 0: # Using PLAY button as a trigger to reset the context
          print("resetting context")
          reset_requested.set()

        if message[1] == REINS_PEDAL_CC:
          set_reins_pedal(message[2])

def pedal_device_callback(event, data=None):
    """MIDI in of the pedal device (PEDAL_PORT_NAME). It reads only the pedal CC and leaves
    midi_event_time alone: RtMidi's deltatime is per port, so mixing it in would break note timing."""
    message, _ = event
    if message[0] & 0xF0 == 176 and message[1] == REINS_PEDAL_CC:
        set_reins_pedal(message[2])

def set_reins_pedal(value):
    if PEDAL_INVERT:
      value = 127 - value
    if value < PEDAL_RELEASED_BELOW:
      pedal = 0.0
    else:
      # PEDAL_RELEASED_BELOW..127 -> 1/118..1 (for 10): just above the released 0
      pedal = (value - PEDAL_RELEASED_BELOW + 1) / (128 - PEDAL_RELEASED_BELOW)
    if pedal > 1 - PEDAL_SNAP:
      pedal = 1.0
    reins.pedal = pedal

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
        # The window restarts at the seed: forget chord history and release held controls.
        chord_controls.reset(CTX_LEN)
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
        reins.reset(dict_input_tokens['pitch'][:CTX_LEN])

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

reins = Reins(*primer_key(sample_midi_path_init), dict_input_tokens['pitch'][:CTX_LEN])
print(reins.status())

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
    # Keep tracking fast 2-key alternation while Return or J forces Joker notes.
    timestamp_ms = now * 1000
    repeated_pair_joker = joker_detector.update(note, timestamp_ms)
    is_joker = (joker_key_held.is_set() or joker_toggle_active.is_set()
                or repeated_pair_joker or note in JOKER_FORCED_KEYS)

    # Reins presses reach the model as Joker notes; the button acts only through the window.
    # With the pedal released the button itself goes to the model (learned map, no window).
    pitch_logit_bias = None
    but = model.joker_button_idx
    if is_joker:
        if TRACES:
            print("JOKER detected")
    else:
        try:
            shown_but = key_to_button(note)
        except:
            shown_but = 0
            print("ERROR key_to_button", note)
        if reins.update_anchor():
            print(reins.status())
        ideal, width, allowed = reins.press(shown_but, timestamp_ms)
        if allowed is None:
            but = shown_but
        else:
            pitch_logit_bias = torch.tensor(pitch_bias(allowed, ideal, reins.pedal, VOCAB_SIZE_PITCH),
                                            dtype=torch.float, device=device)
        if TRACES:
            print("reins button", shown_but, "ideal", ideal, "width", width, "allowed", allowed)

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
    # Held chord / change controls over the same window: chroma [1, CTX_LEN+1, 12], others [1, CTX_LEN+1].
    context.update(chord_controls.next_tokens(device))
    if TRACES:
        print("ctx", i+CTX_LEN+1, "of", TOTAL_GEN_LEN+CTX_LEN)
    with torch.inference_mode():
        if USE_CACHE:
            # A held chord / change control recomputes the window and returns no cache.
            new_pitch_token, kv_cache = model.gen_pitch_token(
                context,
                style_context=style_contexts[active_style_idx],
                style_context_mask=style_context_masks_list[active_style_idx],
                temperature=TEMPERATURE,
                cache=kv_cache,
                logit_bias=pitch_logit_bias
            )
        else:
            new_pitch_token, _ = model.gen_pitch_token(
                context,
                style_context=style_contexts[active_style_idx],
                style_context_mask=style_context_masks_list[active_style_idx]
            , temperature=TEMPERATURE, logit_bias=pitch_logit_bias)
        last_gen_time = time.perf_counter()
    dict_output_tokens['pitch'][i+CTX_LEN] = new_pitch_token
    chord_controls.commit_note()
    reins.after_note(new_pitch_token)
    if live_timing is not None:
        # Replace the target placeholder so it becomes the next note's past pitch.
        live_timing.finish_note(new_pitch_token)

    playNote(new_pitch_token, velocity)
    # Faint marker above the note: the model's most probable pitch, before the pedal window.
    visualizer.get_note(new_pitch_token, velocity, is_joker=is_joker,
                        top_pitch=model.last_top_pitch[0].item())
    #visualizer.get_button(but, velocity)

    if is_joker:
        visualizer.get_joker(velocity)
    else:
        visualizer.get_button(shown_but, velocity)

    noteOn_dict[note] = (new_pitch_token, timeNew, None if is_joker else shown_but, is_joker)
    
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

"""# KEYBOARD — Return holds Joker, J toggles Joker, M captures memory. Chord keys and Space
   go to chord_controls first (main loop). pygame key events (instead of pynput) give the
   unshifted key name, which ChordControls needs for Shift+number chords. """
def _on_key_press(c: str) -> None:
    global active_style_idx
    global kv_cache
    global style_context
    global style_context_mask
    global motif_style_ready
    if c == JOKER_HOLD_KEY:
        joker_key_held.set()
        return
    if c == 'd' and live_timing is not None:
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
    if c == 'm':
        if not memory_key_held.is_set():
            memory_key_held.set()
            capture_highlight_motif()
        return
    if c == JOKER_TOGGLE_KEY:
        if not toggle_key_held.is_set():
            toggle_key_held.set()
            if joker_toggle_active.is_set():
                joker_toggle_active.clear()
            else:
                joker_toggle_active.set()
            visualizer.controls_legend = make_controls_legend()
            print(f"Joker mode {'ON' if joker_toggle_active.is_set() else 'OFF'}")
        return
    if c in STYLE_KEYS:
        active_style_idx = STYLE_KEYS.index(c)
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
    if c == RESET_KEY:
        print("resetting context")
        reset_context()
    if c == 's':
        save_performance()
        print("saved performance")
    if c in (TONIC_DOWN_KEY, TONIC_UP_KEY, MODE_KEY):
        with buffer_lock:
            step = {TONIC_DOWN_KEY: -1, TONIC_UP_KEY: 1}.get(c, 0)
            reins.set_key(reins.tonic_pc + step, reins.minor != (c == MODE_KEY))
        print(reins.status())
    if c in (PEDAL_LESS_KEY, PEDAL_MORE_KEY):
        step = PEDAL_STEP if c == PEDAL_MORE_KEY else -PEDAL_STEP
        reins.pedal = round(min(1.0, max(0.0, reins.pedal + step)), 2)

def _on_key_release(c: str) -> None:
    if c == JOKER_HOLD_KEY:
        joker_key_held.clear()
    elif c == 'm':
        memory_key_held.clear()
    elif c == JOKER_TOGGLE_KEY:
        toggle_key_held.clear()
    elif c == 'd':
        timing_key_held.clear()

def handle_pygame_events() -> None:
    """Keyboard events from the visualizer window (main thread). Chord keys and Space
    go to chord_controls; every other key keeps its previous action."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise SystemExit
        if event.type in (pygame.KEYDOWN, pygame.KEYUP):
            c, pressed = pygame.key.name(event.key), event.type == pygame.KEYDOWN
            with buffer_lock:
                handled = chord_controls.handle_key(c, pressed, shift=bool(event.mod & pygame.KMOD_SHIFT))
            if not handled:
                (_on_key_press if pressed else _on_key_release)(c)
        elif event.type == pygame.WINDOWFOCUSLOST:
            # A key-up may go to another app after focus changes.
            with buffer_lock:
                chord_controls.release_keyboard()
            joker_key_held.clear()

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

    # The reins pedal is a separate device: open it by name, since its port index changes.
    pedal_midiin = rtmidi.MidiIn(midiin.get_current_api())
    pedal_ports = [port_index for port_index, port in enumerate(available_ports) if PEDAL_PORT_NAME in port]
    if pedal_ports:
        pedal_midiin.open_port(pedal_ports[0])
        pedal_midiin.set_callback(pedal_device_callback)
        print(f"Using reins pedal port: {available_ports[pedal_ports[0]]}")
    else:
        print(f"Reins pedal port '{PEDAL_PORT_NAME}' not found: use Z/X")

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
      handle_pygame_events()
      visualizer.status_text = (chord_controls.status() + '\n' + chord_legend + '\n'
                                + reins.status() + '  |  ' + REINS_KEYS_TEXT)
      visualizer.pedal = reins.pedal
      visualizer.set_chord_chroma(chord_controls.chroma.float().tolist(), chord_controls.active)
      visualizer.draw(handle_events=False)
except (EOFError, KeyboardInterrupt, SystemExit):
    print("Bye.")
