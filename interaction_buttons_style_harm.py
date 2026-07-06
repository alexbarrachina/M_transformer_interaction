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

from sympy import false
import torch

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter
from visualizer import Visualizer
from harmony_extractor import RealtimeHarmonyExtractor, estimate_key_from_pitches, PITCH_NAMES

TRACES = False
USE_CACHE = False
CACHE_IDLE_TIMEOUT = 2.0  # seconds - clear KV cache after this idle gap
XINXE_INTERFACE = False

# Audible chord test: when True, each chord planned after a harmonic movement is
# played via playNote() so you can verify the model predicts plausible chords.
TEST_CHORD_PREDICTION = True
CHORD_TEST_OCTAVE = 5      # MIDI octave for the chord tones (5 -> C = 60)
CHORD_TEST_DURATION = 0.6  # seconds the test chord rings before note-off
CHORD_TEST_VELOCITY = 80

# Joker detection: fast alternation of exactly 2 keys triggers joker mode
JOKER_WINDOW_SIZE = 4       # minimum note-on events to detect the pattern
JOKER_MAX_INTERVAL_MS = 200.0  # max ms between consecutive notes to count as "fast"

TEMPERATURE = 1 #0.0001

''' HARMONY INFERENCE PARAMS '''
HARM_CFG_WEIGHT = 2.0# 1.0          # classifier-free guidance strength on harmony (1.0 = off)
HARM_PC_BIAS = 3.0 #0.0             # soft pitch-class logit bias toward chord tones (0.0 = off)
HARM_MOVE_WEIGHT = 4.0         # movement-compatibility weight in constrained chord planning. Reduces/increases the number of possible chords.
HARM_PLAN_HIST = 32            # pitch-history length fed to the chord planner
HARM_DECAY_STEP = 1.0 / 15.0   # transition_phase decay per generated note (harmony release span ~30 notes); intensity stays binary 1.0 while active
HARM_REPLAN_EVERY = 0          # while a movement is active, re-plan the chord every N generated notes so the FiLM chord tracks the evolving melody (0 = freeze for the whole span)
# Keep the runtime conditioner on the checkpoint's trained harmony manifold by
# default. The current AE_style_harm_tester_v1 training pickle contains no
# KEY_CHANNEL events and all chord-label events have FUNC_UNKNOWN, so non-neutral
# function/key/mode embeddings are effectively untrained for this model.
HARM_USE_PLANNER_ANALYSIS_FACTORS = False
# First test (intensity spatial-pattern diagnosis): force FiLM ON at EVERY position
# of the harm_window (context included), so the decoder sees the same constant
# intensity=1.0 regime it saw in training (where intensity was 1.0 over the whole
# sequence, never a partly-conditioned window). True = override the per-note
# intensity buffer with all-ones when building the window.
FORCE_FULL_INTENSITY = False
# Test 1 (FiLM gain sweep): scale the harmony AdaLN-Zero modulation. 1.0 = original
# strength; sweep {0.1, 0.25, 0.5, 1.0} to check whether weaker modulation stays
# coherent (=> magnitude blow-up). Applied as x*(1 + g*I*scale) + g*I*shift.
FILM_GAIN = 1.0
# Test 2 (instrumentation): print conditioner/FiLM norms at the conditioned (last)
# position each generated note, so the pre- vs post-keypress jump is visible.
HARM_FILM_DEBUG = False

''' HARMONY VISUALIZER PARAMS '''
VISUALIZER_WIDTH = 1400
HARMONY_PANEL_WIDTH = 650
HARMONY_TENSION_WINDOW_SEC = 1.0
HARMONY_BUFFER_SEC = 10.0
HARMONY_CHORD_THRESHOLD = 0.3

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

model_name = 'AE_style_harm_tester_v1'
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

# Style prompts for keys 1, 2, 3 — set each path to a different MIDI to transfer style on-the-fly.
style_prompt_midi_paths: List[str] = [
    sample_midi_path1,  # key 1
    sample_midi_path2,  # key 2
    sample_midi_path3,  # key 3
    sample_midi_path4,  # key 4
    sample_midi_path5,  # key 5
]
output_midi_name = './out/interactive_performance'

# Audible primer preview: only the tail of the context (model + visualizer still use full CTX_LEN).
PRIMER_PLAYBACK_LAST_N = 40
# >1.0 shortens wall-clock waits during play_primer (musical spacing unchanged in tokens).
PRIMER_PLAYBACK_SPEED = 1

NUM_BUTTONS = cfg['num_buttons']
HIGHLIGHT_MOTIF_LEN = 30
MOTIF_STYLE_KEY = '6'

# --- Harmonic movement keys (QWERTY): pressing one requests a harmonic movement.
# The chord planner picks the next chord realising that movement (JOKER = model
# decides) and the conditioning intensity decays back to "released" over time. ---
HARMONY_KEY_MAPPING = {
    'z': MOVE_STABILIZE,
    'x': MOVE_RECOLOR,
    'c': MOVE_PREPARE,
    'v': MOVE_TENSION,
    'b': MOVE_RESOLVE,
    'n': MOVE_EVADE,
    'm': MOVE_CHROMATIC,
    ',': MOVE_MODULATE,
    '.': MOVE_JOKER,
}
MOVEMENT_NAMES = {
    MOVE_STABILIZE: 'STABILIZE', MOVE_RECOLOR: 'RECOLOR', MOVE_PREPARE: 'PREPARE',
    MOVE_TENSION: 'TENSION', MOVE_RESOLVE: 'RESOLVE', MOVE_EVADE: 'EVADE',
    MOVE_CHROMATIC: 'CHROMATIC', MOVE_MODULATE: 'MODULATE', MOVE_JOKER: 'JOKER',
}

# Readable names for verifying the chord planner's output (point 5).
_PC_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_QUALITY_NAMES = {
    QUALITY_UNKNOWN: 'unknown', QUALITY_MAJOR: 'maj', QUALITY_MINOR: 'min',
    QUALITY_DIMINISHED: 'dim', QUALITY_AUGMENTED: 'aug',
    QUALITY_DOMINANT_SEVENTH: '7', QUALITY_MAJOR_SEVENTH: 'maj7',
    QUALITY_MINOR_SEVENTH: 'min7', QUALITY_MINOR_MAJOR_SEVENTH: 'minMaj7',
    QUALITY_DIMINISHED_SEVENTH: 'dim7', QUALITY_HALF_DIMINISHED_SEVENTH: 'm7b5',
    QUALITY_DOMINANT_EXT: '9/11/13', QUALITY_OTHER: 'other',
}


def _pc_name(pc: int) -> str:
    return _PC_NAMES[int(pc)] if 0 <= int(pc) < 12 else '?'


def _format_chord(factors: dict, chroma: List[float]) -> str:
    """Human-readable chord summary (root + quality + active chroma pitch-classes)
    so the performer can verify what the planner chose for each movement."""
    root = _pc_name(factors['root_pc'])
    qual = _QUALITY_NAMES.get(int(factors['quality_id']), str(factors['quality_id']))
    pcs = ' '.join(_PC_NAMES[pc] for pc in range(12) if chroma[pc] > 0.0)
    return f"{root} {qual}  chroma=[{pcs}]"


def _neutral_harm_factors():
    return {'root_pc': PC_UNKNOWN, 'quality_id': QUALITY_UNKNOWN,
            'function_id': FUNC_UNKNOWN, 'key_pc': PC_UNKNOWN, 'mode': MODE_UNKNOWN}


def _init_harm_buffers(n: int) -> dict:
    """Per-position harmony buffers (parallel to the button buffer `b`). The
    seed/context region stays neutral with intensity 0 (== unconditional)."""
    return {
        'harm_movement': [MOVE_STABILIZE] * n,
        'transition_phase': [0.0] * n,
        'bass_pc': [PC_UNKNOWN] * n,
        'root_pc': [PC_UNKNOWN] * n,
        'quality_id': [QUALITY_UNKNOWN] * n,
        'function_id': [FUNC_UNKNOWN] * n,
        'key_pc': [PC_UNKNOWN] * n,
        'mode': [MODE_UNKNOWN] * n,
        'intensity': [0.0] * n,
        'chroma': [[0.0] * 12 for _ in range(n)],
    }

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

'''VISUALIZER'''
visualizer = Visualizer(
    width=VISUALIZER_WIDTH,
    button_slots=cfg['num_buttons'],
    harmony_panel_width=HARMONY_PANEL_WIDTH,
)

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
    global harm_buffers
    global current_movement, current_intensity, current_phase, current_factors
    global current_bass_pc, current_chroma, pending_plan
    global harm_gen_hits, harm_gen_total
    global harmony_tracker

    with buffer_lock:
        i = 0
        kv_cache = None
        # Reset harmony state + buffers to neutral (released / unconditional)
        current_movement = MOVE_STABILIZE
        current_intensity = 0.0
        current_phase = 0.0
        harm_gen_hits = 0
        harm_gen_total = 0
        current_factors = _neutral_harm_factors()
        current_bass_pc = PC_UNKNOWN
        current_chroma = [0.0] * 12
        pending_plan = False
        harm_buffers = _init_harm_buffers(TOTAL_GEN_LEN + CTX_LEN)
        harmony_tracker = RealtimeHarmonyExtractor(
            global_key=harmony_global_key,
            tension_window_sec=HARMONY_TENSION_WINDOW_SEC,
            buffer_size_sec=HARMONY_BUFFER_SEC,
            visualize=False,
            chord_threshold=HARMONY_CHORD_THRESHOLD,
        )
        visualizer.clear_harmony_history()
        visualizer.set_chord_chroma(None, False)
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

''' VARIABLES '''
context = None
timeLast = 0
i = 0 # num current tokens in context after CTX_LEN
noteOn_dict = {} # note: (pitch, timeIn, button)
first_note = True
kv_cache = None
last_gen_time: float = 0.0
joker_detector = JokerDetector()

''' HARMONY STATE '''
current_movement = MOVE_STABILIZE
current_intensity = 0.0                      # binary: 1.0 == movement active, 0.0 == released (unconditional)
current_phase = 0.0                          # transition_phase: 1.0 at movement onset -> 0 across the span (matches training; decoupled from intensity)
current_factors = _neutral_harm_factors()    # active chord factors
current_bass_pc = PC_UNKNOWN
current_chroma: List[float] = [0.0] * 12
pending_plan = False                          # set by a movement key, consumed on the next note
harm_buffers: dict = {}
# TRACES diagnostics: chord-tone match stats for notes generated during the current transition (intensity>0)
harm_gen_hits = 0                             # generated notes whose pitch-class is in the FiLM chord chroma
harm_gen_total = 0                            # generated notes while a movement is active

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

# Real-time harmony analysis for the embedded Tonnetz panel. The key is fixed
# from the seed context, while x/y/tension are recomputed from generated notes.
harmony_global_key = estimate_key_from_pitches(dict_input_tokens['pitch'][:CTX_LEN])
harmony_key_root = harmony_global_key % 12
harmony_key_mode = 'major' if harmony_global_key < 12 else 'minor'
print(f"Harmony visualizer key: {PITCH_NAMES[harmony_key_root]} {harmony_key_mode}")
harmony_tracker = RealtimeHarmonyExtractor(
    global_key=harmony_global_key,
    tension_window_sec=HARMONY_TENSION_WINDOW_SEC,
    buffer_size_sec=HARMONY_BUFFER_SEC,
    visualize=False,
    chord_threshold=HARMONY_CHORD_THRESHOLD,
)
visualizer.set_harmony_state(
    0.0,
    0.0,
    0.0,
    key_root=harmony_key_root,
    key_mode=harmony_key_mode,
    active=False,
)

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

harm_buffers = _init_harm_buffers(TOTAL_GEN_LEN + CTX_LEN)


def _write_harm(idx: int) -> None:
    """Record the current harmony state at position idx (called per generated note)."""
    harm_buffers['harm_movement'][idx] = int(current_movement)
    harm_buffers['intensity'][idx] = float(current_intensity)
    harm_buffers['transition_phase'][idx] = float(current_phase)
    harm_buffers['bass_pc'][idx] = int(current_bass_pc)
    harm_buffers['root_pc'][idx] = int(current_factors['root_pc'])
    harm_buffers['quality_id'][idx] = int(current_factors['quality_id'])
    harm_buffers['function_id'][idx] = int(current_factors['function_id'])
    harm_buffers['key_pc'][idx] = int(current_factors['key_pc'])
    harm_buffers['mode'][idx] = int(current_factors['mode'])
    harm_buffers['chroma'][idx] = list(current_chroma)


def _harm_window(lo: int, hi: int) -> dict:
    """Build the harm_fields tensor window [1, hi-lo, ...] for gen_pitch_token."""
    win = {}
    for k in ('harm_movement', 'bass_pc', 'root_pc', 'quality_id',
              'function_id', 'key_pc', 'mode'):
        win[k] = torch.tensor(harm_buffers[k][lo:hi], dtype=torch.long).unsqueeze(0).to(device)
    for k in ('transition_phase', 'intensity'):
        win[k] = torch.tensor(harm_buffers[k][lo:hi], dtype=torch.float).unsqueeze(0).to(device)
    # First test: force FiLM ON across the whole window (context included) to match
    # the training distribution (constant intensity=1.0 over the sequence).
    if FORCE_FULL_INTENSITY:
        win['intensity'] = torch.ones_like(win['intensity'])
    win['chroma'] = torch.tensor(harm_buffers['chroma'][lo:hi], dtype=torch.float).unsqueeze(0).to(device)
    return win


def _plan_next_chord(idx: int) -> None:
    """Plan the next chord that realises current_movement from recent pitch history
    (movement-constrained; JOKER lets the model decide), then update the active
    chord factors + chroma. Runs in the MIDI/inference thread under buffer_lock."""
    global current_factors, current_chroma, current_bass_pc
    lo = max(0, idx - HARM_PLAN_HIST)
    hist = dict_output_tokens['pitch'][lo:idx]
    if len(hist) <= 0:
        return
    pitch_t = torch.tensor(hist, dtype=torch.long).unsqueeze(0).to(device)
    mvt = torch.full((1, pitch_t.shape[1]), int(current_movement), dtype=torch.long, device=device)
    cur_f = {k: torch.tensor([int(current_factors[k])], dtype=torch.long, device=device)
             for k in current_factors}
    with torch.inference_mode():
        planned = model.plan_factors_constrained(
            pitch_t, mvt, current_factors=cur_f, move_weight=HARM_MOVE_WEIGHT)
        ch = model.chroma_from_factors(planned['root_pc'], planned['quality_id'])
    current_factors = {k: int(planned[k].item()) for k in planned}
    if not HARM_USE_PLANNER_ANALYSIS_FACTORS:
        current_factors['function_id'] = FUNC_UNKNOWN
        current_factors['key_pc'] = PC_UNKNOWN
        current_factors['mode'] = MODE_UNKNOWN
    current_bass_pc = current_factors['root_pc']
    current_chroma = ch.squeeze(0).cpu().tolist()
    # Always print a readable summary so the performer can verify the planner.
    print(f"planned chord [{MOVEMENT_NAMES.get(current_movement, current_movement)}]: "
          f"{_format_chord(current_factors, current_chroma)}")
    if TRACES:
        print("planned chord", current_factors)


def _play_predicted_chord() -> None:
    """Audibly play the currently planned chord so the user can verify whether
    the model predicts plausible chords. Chroma (binary chord-tone mask) ->
    pitch-classes -> MIDI notes; blocks briefly for the chord to ring (test-only
    path, runs in the MIDI thread under buffer_lock)."""
    pcs = [pc for pc in range(12) if current_chroma[pc] > 0.0]
    if len(pcs) <= 0:
        print("predicted chord empty (unknown/other quality)")
        return
    base = CHORD_TEST_OCTAVE * 12  # MIDI note of C in the chosen octave
    notes: List[int] = []
    # Bass note from the planned root/bass pitch-class, one octave below.
    if 0 <= int(current_bass_pc) < 12:
        notes.append(base - 12 + int(current_bass_pc))
    for pc in pcs:
        notes.append(base + pc)
    if TRACES:
        print("predicted chord notes", notes)
    for n in notes:
        playNote(n, CHORD_TEST_VELOCITY)
    time.sleep(CHORD_TEST_DURATION)
    for n in notes:
        playNote(n, 0)


def _update_harmony_visualizer() -> None:
    """Push real-time generated-note harmony analysis into the pygame panel."""
    now_s = visualizer.current_time()
    harmony_tracker.update(now_s)
    cx, cy, mag, key_root, key_mode = harmony_tracker.extract_current_features(now_s)
    active_pcs = harmony_tracker._extract_active_pitch_classes(now_s)
    target_chroma = current_chroma if current_intensity > 0.0 else None
    movement_label = MOVEMENT_NAMES.get(current_movement, str(current_movement))
    chord_label = _format_chord(current_factors, current_chroma) if target_chroma is not None else ''
    has_harmony_content = bool(active_pcs) or mag > 0.01 or target_chroma is not None
    visualizer.set_harmony_state(
        cx,
        cy,
        mag,
        key_root=key_root,
        key_mode=key_mode,
        active_pcs=active_pcs,
        target_chroma=target_chroma,
        movement_label=movement_label,
        chord_label=chord_label,
        active=has_harmony_content,
    )

# Full CTX_LEN in the primer strip and in the model; audible tail runs after MIDI opens (main loop).
visualizer.primer(dict_input_tokens['pitch'][:CTX_LEN], dict_input_tokens['dtime'][:CTX_LEN],
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
  global active_style_idx
  global current_intensity
  global current_phase
  global pending_plan
  global harm_gen_hits, harm_gen_total
  global harmony_tracker

  if TRACES:
    print("key", note)

  timeNew = time.perf_counter()*1000 /32 # in miliseconds /32 as in midi_to_dict()

  if velocity > 0: # noteOn
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
    # MIDI note to button — joker detector overrides when fast 2-key alternation
    timestamp_ms = time.perf_counter() * 1000
    is_joker = joker_detector.update(note, timestamp_ms)

    if is_joker:
        but = getattr(model, 'joker_button_idx', NUM_BUTTONS - 1)
        if TRACES:
            print("JOKER detected")
    else:
        try:
            but = key_to_button(note)
        except:
            but = 0
            print("ERROR key_to_button", note)

    b[i+CTX_LEN] = but

    # --- Harmony: on a movement request, plan the next chord; then record the
    # harmony state for this position so the decoder is FiLM-conditioned.
    idx = i + CTX_LEN
    if pending_plan:
        _plan_next_chord(idx)
        if TEST_CHORD_PREDICTION:
            _play_predicted_chord()
        # Movement onset: intensity is binary (1.0 == active) to match the training
        # distribution (constant 1.0 over a span); transition_phase starts at 1.0
        # and decays across the span (see the release step below).
        current_intensity = 1.0
        current_phase = 1.0
        pending_plan = False
        harm_gen_hits = 0      # restart chord-tone match stats for this transition
        harm_gen_total = 0
    elif current_phase > 0.0 and HARM_REPLAN_EVERY > 0 and (i % HARM_REPLAN_EVERY == 0):
        # Re-plan while the movement is still active so the FiLM chord tracks the
        # evolving (autoregressive) melody instead of staying frozen for the span.
        _plan_next_chord(idx)
    _write_harm(idx)
    harm_window = _harm_window(i, i + CTX_LEN + 1)

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
            # CFG needs two passes (would corrupt the cache) -> no guidance when cached.
            new_pitch_token, kv_cache = model.gen_pitch_token(
                context,
                style_context=style_contexts[active_style_idx],
                style_context_mask=style_context_masks_list[active_style_idx],
                harm_fields=harm_window,
                cfg_weight=1.0,
                pc_bias_weight=HARM_PC_BIAS,
                cache=kv_cache,
                temperature=TEMPERATURE,
                film_gain=FILM_GAIN,
                film_debug=HARM_FILM_DEBUG,
            )
        else:
            new_pitch_token, _ = model.gen_pitch_token(
                context,
                style_context=style_contexts[active_style_idx],
                style_context_mask=style_context_masks_list[active_style_idx],
                harm_fields=harm_window,
                cfg_weight=HARM_CFG_WEIGHT,
                pc_bias_weight=HARM_PC_BIAS,
                temperature=TEMPERATURE,
                film_gain=FILM_GAIN,
                film_debug=HARM_FILM_DEBUG,
            )
        last_gen_time = time.perf_counter()
    dict_output_tokens['pitch'][i+CTX_LEN] = new_pitch_token

    # Overlay the predicted chord (white rows) on the visualizer pitch roll so the
    # performer can see whether generated pitches follow the chord (only while a
    # movement is active, i.e. intensity>0).
    visualizer.set_chord_chroma(current_chroma, current_intensity > 0.0)

    # --- TRACES: verify the FiLM chord influences generation during a transition.
    # For each note generated while intensity>0, check if its pitch-class (p%12)
    # is a chord tone (chroma>0) and compare the running chord-tone hit rate to the
    # chance rate (#chord pcs / 12). observed >> chance => FiLM is steering pitches.
    if TRACES and current_intensity > 0.0:
        _pc = int(new_pitch_token) % 12
        _chord_pcs = [p for p in range(12) if current_chroma[p] > 0.0]
        _in_chord = _pc in _chord_pcs
        harm_gen_total += 1
        if _in_chord:
            harm_gen_hits += 1
        _rate = harm_gen_hits / harm_gen_total
        _chance = (len(_chord_pcs) / 12.0) if len(_chord_pcs) > 0 else 0.0
        print(f"[harm] gen {new_pitch_token} ({_pc_name(_pc)}) "
              f"{'IN ' if _in_chord else 'OUT'} chord [{' '.join(_PC_NAMES[p] for p in _chord_pcs)}]  "
              f"int={current_intensity:.2f} phase={current_phase:.2f}  "
              f"hit_rate={_rate:.2f} ({harm_gen_hits}/{harm_gen_total}) vs chance={_chance:.2f}")

    # Release: transition_phase decays toward 0 across the movement span (matching
    # the training span-decay); intensity stays binary 1.0 while active and drops
    # to 0 (unconditional) once the span elapses.
    if current_phase > 0.0:
        current_phase = max(0.0, current_phase - HARM_DECAY_STEP)
        if current_phase <= 0.0:
            current_intensity = 0.0
    visualizer.set_chord_chroma(current_chroma, current_intensity > 0.0)
    if TRACES:
        print("intensity", current_intensity, "phase", current_phase)

    playNote(new_pitch_token, velocity) 
    harmony_tracker.add_note(new_pitch_token, visualizer.current_time(), velocity)
    visualizer.get_note(new_pitch_token, velocity)
    visualizer.get_button(but, velocity)
    _update_harmony_visualizer()

    # add (pitch, time, button) to dictionary using original MIDI note as key
    noteOn_dict[note] = (new_pitch_token, timeNew, but)
    i += 1

  else: # noteOff
    # Use original MIDI note as key to find corresponding noteOn
    if note in noteOn_dict:

      # get pitch, time, and button from dictionary of accumulated notesOns without noteOff
      pitch, noteOn_time, but = noteOn_dict[note]
      playNote(pitch, 0)
      harmony_tracker.note_off(pitch, visualizer.current_time())
      visualizer.get_note(pitch, 0)
      visualizer.get_button(but, 0)
      _update_harmony_visualizer()
      # Remove from dictionary to allow the same note to be played again
      del noteOn_dict[note]
      #visualizer.update(noteOn_time)


"""# KEYBOARD LISTENER — keys 1/2/3/4/5 switch MIDI style, space saves motif, 6 activates motif """
def _on_key_press(key: pkeyboard.Key) -> None:
    global active_style_idx
    global kv_cache
    global style_context
    global style_context_mask
    global motif_style_ready
    global current_movement
    global pending_plan
    if key == pkeyboard.Key.space:
        capture_highlight_motif()
        return
    try:
        c = key.char  # type: ignore[union-attr]
        if c in HARMONY_KEY_MAPPING:
            current_movement = HARMONY_KEY_MAPPING[c]
            pending_plan = True  # the next generated note plans + applies the chord
            print(f"Movement: {MOVEMENT_NAMES.get(current_movement, current_movement)}")
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
                print("No highlight motif saved yet. Press space first.")
        if c in ('r'):
            print("resetting context")
            reset_context()
        if c in ('s'):
            save_performance()
            print("saved performance")
    except AttributeError:
        pass

_key_listener = pkeyboard.Listener(on_press=_on_key_press)
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
      with buffer_lock:
          _update_harmony_visualizer()
          visualizer.draw()
except (EOFError, KeyboardInterrupt, SystemExit):
    print("Bye.")
