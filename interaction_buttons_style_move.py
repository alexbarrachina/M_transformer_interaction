#===================================================================================================
# Monster Genie interaction_buttons_style_move.py Python module
# Interaction, generating buttons from a MIDI keyboard, starting from a MIDI seed.
#
# Drives the TRAINED AE_style_move model: style cross-attention + button next-note
# control + a single BINARY "move now" harmonic-step channel (FiLM). There are no
# chord planners, movement classes or chroma targets here — the performer (or the
# stuck detector) only says WHEN to move; the model decides WHERE the harmony goes.
#
#   '.'        take one harmonic step now (pulse the move flag)
#   'a'        toggle auto stuck-detection (on by default)
#   '[' / ']'  decrease / increase CFG insistence (press-depth proxy)
#   1..5       switch MIDI style prompt      6  activate captured motif
#   space      capture highlight motif       r/s  reset / save
#
# Requires a trained checkpoint: run train_style_harm.py with
# model_name='AE_style_move_v1' (or 'AE_style_move_tester_v1') first.
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

import torch

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter
from visualizer import Visualizer
from harmonic_step import HarmonicStepTracker, triad_name

TRACES = False
USE_CACHE = False
CACHE_IDLE_TIMEOUT = 2.0  # seconds - clear KV cache after this idle gap
XINXE_INTERFACE = False

# Joker detection (button vocab): fast alternation of exactly 2 keys => joker button
JOKER_WINDOW_SIZE = 4       # minimum note-on events to detect the pattern
JOKER_MAX_INTERVAL_MS = 200.0  # max ms between consecutive notes to count as "fast"

TEMPERATURE = 1  # 0.0001

''' MOVE (harmonic-step) INFERENCE PARAMS '''
# The move flag is the whole control signal. A "move" is a pulse: move_flag=1 for
# MOVE_FLAG_SPAN generated notes with transition_phase ramping 1.0 -> ~0 across the
# span (exactly the training labelling in train_style_harm.py move_binary), then
# the flag drops back to 0 = "stable / hold".
MOVE_FLAG_SPAN = 8            # notes the "move now" pulse stays raised (match cfg['move_flag_span'])
MOVE_CFG_WEIGHT = 2.0         # classifier-free guidance strength = insistence / press depth (1.0 = off)
MOVE_CFG_STEP = 0.5          # '[' / ']' live adjustment step for the CFG weight
# Intensity gates the FiLM modulation (x*(1+I*scale)+I*shift). The model trains
# with intensity=1.0 UNIFORM over the whole sequence (dropout zeros the whole
# sample), and the binary flag — not intensity — distinguishes move vs stable. So
# the in-distribution inference regime is intensity=1.0 across the whole window;
# the unconditional CFG branch (intensity=0) is applied internally by
# gen_pitch_token. Do NOT toggle intensity per-note here (that window was never
# seen in training). Set 0.0 only to A/B against the pure unconditioned model.
MOVE_INTENSITY = 1.0
# Test knob (FiLM gain sweep): scale the AdaLN-Zero modulation. 1.0 = trained strength.
FILM_GAIN = 1.0
MOVE_FILM_DEBUG = False       # print conditioner/FiLM norms at the conditioned position

''' AUTO STUCK-DETECTION (reuses harmonic_step.HarmonicStepTracker as a DETECTOR only) '''
AUTO_MOVE_DEFAULT = True      # auto-pulse the flag when the generated stream is stuck on one triad
MOVE_STUCK_AFTER = 12         # consecutive same-triad notes that count as "stuck"
MOVE_COOLDOWN_NOTES = 8       # notes after a pulse ends before auto can re-arm
MOVE_SEED_NOTES = 32          # primer-tail notes used to seed the triad tracker
MOVE_KEY = '.'               # manual "take one harmonic step now"
MOVE_TOGGLE_KEY = 'a'        # toggle auto stuck-detection
CFG_DOWN_KEY = '['
CFG_UP_KEY = ']'

''' DEVICE SPECIFIC PARAMETERS '''
if torch.backends.mps.is_available():
    # CASA
    device = torch.device('mps')
    CTX_LEN = 128  # num notes in context.
    TOTAL_GEN_LEN = 800  # num notes to generate
    if XINXE_INTERFACE:
        KEY_OFFSET = 60  # esmuc 34, casa 48
    else:
        KEY_OFFSET = 48  # esmuc 34, casa 48
else:
    # ESMUC
    device = torch.device('cuda')
    CTX_LEN = 512  # num notes in context.
    TOTAL_GEN_LEN = 1024  # num notes to generate
    if XINXE_INTERFACE:
        KEY_OFFSET = 60  # esmuc 34, casa 48
    else:
        KEY_OFFSET = 34  # esmuc 34, casa 48

''' MODEL '''
# Trained binary move-flag model. Switch to 'AE_style_move_tester_v1' for the small
# (512-dim) variant. The checkpoint must exist (train it first with train_style_harm.py).
model_name = 'AE_style_move_tester_v1'
cfg = get_model_hparams(model_name)
model = load_model(model_name=model_name, cfg=cfg)
model.to(device)
model.eval()

''' PARAMS '''
# Get sample seed MIDI path
sample_midi_path1 = './samples/Bach_Prelude_and_Fugue_in_C_major.mid'
sample_midi_path2 = './samples/clairTester_to_end.midi'
sample_midi_path3 = './samples/Chopin_Nocturnes_Op9No1_In_B_Flat_Minor.mid'
sample_midi_path4 = './samples/Scott_Cyril_Lotus_Land.mid'
sample_midi_path5 = './samples/Satie_Gymnopedie_No1.mid'

sample_midi_path_init = sample_midi_path4
STYLE_IDX_INIT = 4

# Style prompts for keys 1..5 — set each path to a different MIDI to transfer style on-the-fly.
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


class JokerDetector:
    """Detects fast repetitive alternation of exactly 2 keys and tracks which
    specific keys are the current joker pair (button-vocab joker; orthogonal to
    the harmonic move flag). Only the 2 keys involved in the fast alternation
    become joker keys; the pair clears once the fast pattern stops."""

    def __init__(self, window_size: int = JOKER_WINDOW_SIZE,
                 max_interval_ms: float = JOKER_MAX_INTERVAL_MS):
        self.window_size: int = window_size
        self.max_interval_ms: float = max_interval_ms
        self.history: List[tuple] = []  # (key, timestamp_ms) sliding window
        self.joker_keys: set = set()    # the 2 MIDI keys currently acting as joker (empty = none)
        self.last_joker_time_ms: float = 0.0

    def update(self, key: int, timestamp_ms: float) -> bool:
        """Register a note-on event. Returns True if THIS key is a joker key."""
        if self.joker_keys and (timestamp_ms - self.last_joker_time_ms) > self.max_interval_ms:
            self.joker_keys = set()
        if key in self.joker_keys:
            self.last_joker_time_ms = timestamp_ms
            self._push_history(key, timestamp_ms)
            return True
        self._push_history(key, timestamp_ms)
        if self._detect_pattern():
            self.joker_keys = set(h[0] for h in self.history)
            self.last_joker_time_ms = timestamp_ms
            return True
        return False

    def _push_history(self, key: int, timestamp_ms: float) -> None:
        self.history.append((key, timestamp_ms))
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def _detect_pattern(self) -> bool:
        if len(self.history) < self.window_size:
            return False
        for idx in range(1, len(self.history)):
            if self.history[idx][1] - self.history[idx - 1][1] > self.max_interval_ms:
                return False
        keys = set(h[0] for h in self.history)
        return len(keys) == 2

    def reset(self) -> None:
        self.history.clear()
        self.joker_keys = set()
        self.last_joker_time_ms = 0.0


def _init_move_buffers(n: int) -> dict:
    """Per-position move buffers (parallel to the button buffer `b`). intensity is
    1.0 everywhere (uniform, in-distribution); the seed/context region carries
    flag 0 (no move requested) with phase 0 = stable."""
    return {
        'move_flag': [0] * n,
        'transition_phase': [0.0] * n,
        'intensity': [MOVE_INTENSITY] * n,
    }


'''THREADING'''
buffer_lock = Lock()
save_lock = Lock()
reset_requested = Event()  # pygame/SDL must run on the main thread (macOS); MIDI callback is on another thread

'''VISUALIZER'''
visualizer = Visualizer(button_slots=cfg['num_buttons'])

'''MIDI IN CALLBACK'''
def midiin_callback(event, data=None):
    message, deltatime = event

    if message[0] & 0xF0 == NOTE_ON:
        status, note, velocity = message
        with buffer_lock:  # lock to avoid race condition
            manageNote(note, velocity)

    if message[0] & 0xF0 == NOTE_OFF:
        status, note, velocity = message
        manageNote(note, 0)

    if message[0] & 0xF0 == 176:  # 176 is the status for control change
        if message[1] == 18 and message[2] > 0:  # REC button -> save performance
            with save_lock:
                print("saving performance")
                save_performance()
                sys.exit(0)
        if message[1] == 17 and message[2] > 0:  # PLAY button -> reset the context
            print("resetting context")
            reset_requested.set()


def key_to_button(key):
    key = key - KEY_OFFSET  # keyboard starts at C = 48
    button = key
    if XINXE_INTERFACE:
        button = key
    else:
        toWhite = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 10, 10, 11, 11, 12, 12, 13, 14, 14, 15, 15, 16, 17, 17, 18, 18, 19, 19, 20, 21, 21, 22, 22, 23, 23, 24]
        if key < 0 or key >= len(toWhite):
            button = 0
        else:
            button = toWhite[button]  # convert to white key index
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

    song_d = dict_to_song(context)
    ms_SONG_to_MIDI_Converter(song_d, output_file_name=output_midi_name, timings_multiplier=1)
    print("saved performance")


def reset_context():
    global i
    global dict_output_tokens, dict_input_tokens
    global kv_cache
    global b
    global move_buffers
    global move_notes_left, move_cooldown, pending_move, current_flag, current_phase
    global step_tracker

    with buffer_lock:
        i = 0
        kv_cache = None
        # Reset move state + buffers to neutral (stable / no pending move)
        move_notes_left = 0
        move_cooldown = 0
        pending_move = False
        current_flag = 0
        current_phase = 0.0
        move_buffers = _init_move_buffers(TOTAL_GEN_LEN + CTX_LEN)
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
        # Re-seed the triad tracker from the primer tail
        step_tracker.reset()
        step_tracker.seed([int(p) for p in dict_input_tokens['pitch'][:CTX_LEN][-MOVE_SEED_NOTES:]])


''' VARIABLES '''
context = None
timeLast = 0
i = 0  # num current tokens in context after CTX_LEN
noteOn_dict = {}  # note: (pitch, timeIn, button)
first_note = True
kv_cache = None
last_gen_time: float = 0.0
joker_detector = JokerDetector()

''' MOVE STATE '''
# A "move" is a pulse: move_notes_left counts down the raised-flag span; each note
# inside the span has move_flag=1 and transition_phase = 1 - k/SPAN (k from onset).
move_notes_left = 0
move_cooldown = 0
pending_move = False
current_flag = 0
current_phase = 0.0
auto_move_enabled = AUTO_MOVE_DEFAULT
move_cfg_weight = MOVE_CFG_WEIGHT
step_tracker = HarmonicStepTracker(stuck_after=MOVE_STUCK_AFTER)  # DETECTOR only (model chooses the chord)
move_buffers: dict = {}

''' BUILD CTX '''
# Load seed MIDI
dict_input_tokens, num_notes = midi_to_dict(sample_midi_path_init)  # tokens, without vel

# Extend dict_output_tokens to accommodate TOTAL_GEN_LEN + CTX_LEN tokens
dict_output_tokens = {}
for key in dict_input_tokens.keys():
    extended_list = dict_input_tokens[key].copy()
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

active_style_idx: int = STYLE_IDX_INIT - 1  # -1 because list is 0-indexed
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
    e = model.encoder(context)  # encoder output (batch, seq_len)
    b = model.real_to_discrete(e).squeeze(0)  # generate buttons (batch, seq_len)
    b = b.clone().detach().tolist()
    b.extend([0] * (TOTAL_GEN_LEN + CTX_LEN - len(b)))

move_buffers = _init_move_buffers(TOTAL_GEN_LEN + CTX_LEN)


def _write_move(idx: int) -> None:
    """Record the current move state at position idx (called per generated note)."""
    move_buffers['move_flag'][idx] = int(current_flag)
    move_buffers['transition_phase'][idx] = float(current_phase)
    move_buffers['intensity'][idx] = float(MOVE_INTENSITY)


def _move_window(lo: int, hi: int) -> dict:
    """Build the harm_fields tensor window [1, hi-lo, ...] for gen_pitch_token."""
    return {
        'move_flag': torch.tensor(move_buffers['move_flag'][lo:hi], dtype=torch.long).unsqueeze(0).to(device),
        'transition_phase': torch.tensor(move_buffers['transition_phase'][lo:hi], dtype=torch.float).unsqueeze(0).to(device),
        'intensity': torch.tensor(move_buffers['intensity'][lo:hi], dtype=torch.float).unsqueeze(0).to(device),
    }


# Full CTX_LEN in the primer strip and in the model; audible tail runs after MIDI opens (main loop).
visualizer.primer(dict_input_tokens['pitch'][:CTX_LEN], dict_input_tokens['dtime'][:CTX_LEN],
                  b[:CTX_LEN], dict_input_tokens['dur'][:CTX_LEN])

# Seed the triad tracker with the primer tail so the first generated notes already
# have a triad estimate to compare against for stuck detection.
step_tracker.seed([int(p) for p in dict_input_tokens['pitch'][:CTX_LEN][-MOVE_SEED_NOTES:]])
print(f"[move] tracker seeded: current triad = "
      f"{triad_name(step_tracker.current) if step_tracker.current else 'none yet'} | "
      f"auto={'ON' if auto_move_enabled else 'OFF'}, cfg={move_cfg_weight:.1f} | "
      f"'{MOVE_KEY}' = move now, '{MOVE_TOGGLE_KEY}' = toggle auto, "
      f"'{CFG_DOWN_KEY}'/'{CFG_UP_KEY}' = cfg -/+")


def manageNote(note, velocity):
    global context  # Access the global context
    global timeLast  # time of last note, global variable
    global b  # button array
    global i  # num current tokens in context after CTX_LEN
    global dict_output_tokens  # output tokens
    global noteOn_dict  # note: (pitch, timeIn, button)
    global first_note
    global visualizer
    global kv_cache
    global last_gen_time
    global active_style_idx
    global pending_move, move_notes_left, move_cooldown, current_flag, current_phase

    if TRACES:
        print("key", note)

    timeNew = time.perf_counter() * 1000 / 32  # in miliseconds /32 as in midi_to_dict()

    if velocity > 0:  # noteOn
        now = time.perf_counter()
        if USE_CACHE and kv_cache is not None and (now - last_gen_time) > CACHE_IDLE_TIMEOUT:
            kv_cache = None

        dtime = max(0, min(127, int(timeNew) - int(timeLast)))  # time difference from previous events, capped at 127
        if first_note:
            dtime = 0
            first_note = False

        timeLast = timeNew
        dict_output_tokens['dtime'][i + CTX_LEN] = dtime
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

        b[i + CTX_LEN] = but
        idx = i + CTX_LEN

        # --- Move flag: arm the pulse on manual request or on stuck detection, then
        # compute this position's (flag, phase) from the countdown and record it so
        # the decoder is FiLM-conditioned over the whole attended window. ---
        if move_notes_left <= 0:
            if pending_move:
                pending_move = False
                move_notes_left = MOVE_FLAG_SPAN
                print(f"[move] manual step: pulse the flag for {MOVE_FLAG_SPAN} notes "
                      f"(from {triad_name(step_tracker.current) if step_tracker.current else '?'}, cfg={move_cfg_weight:.1f})")
            elif auto_move_enabled and move_cooldown <= 0 and step_tracker.is_stuck():
                move_notes_left = MOVE_FLAG_SPAN
                print(f"[move] stuck for {step_tracker.held_notes} notes on "
                      f"{triad_name(step_tracker.current)} -> pulse the flag (cfg={move_cfg_weight:.1f})")

        if move_notes_left > 0:
            k = MOVE_FLAG_SPAN - move_notes_left        # 0 at onset
            current_flag = 1
            current_phase = max(0.0, 1.0 - k / MOVE_FLAG_SPAN)
        else:
            current_flag = 0
            current_phase = 0.0
        _write_move(idx)
        move_window = _move_window(i, i + CTX_LEN + 1)

        context = {
            'dtime': torch.tensor(dict_output_tokens['dtime'][i:i + CTX_LEN + 1], dtype=torch.long).unsqueeze(0),
            'pitch': torch.tensor(dict_output_tokens['pitch'][i:i + CTX_LEN + 1], dtype=torch.long).unsqueeze(0),
            'dur': torch.tensor(dict_output_tokens['dur'][i:i + CTX_LEN + 1], dtype=torch.long).unsqueeze(0),
            'button': torch.tensor(b[i:i + CTX_LEN + 1], dtype=torch.long).unsqueeze(0)
        }
        context = to_device(context, device)
        if TRACES:
            print("ctx", i + CTX_LEN + 1, "of", TOTAL_GEN_LEN + CTX_LEN)
        with torch.inference_mode():
            if USE_CACHE:
                # CFG needs two passes (would corrupt the cache) -> no guidance when cached.
                new_pitch_token, kv_cache = model.gen_pitch_token(
                    context,
                    style_context=style_contexts[active_style_idx],
                    style_context_mask=style_context_masks_list[active_style_idx],
                    harm_fields=move_window,
                    cfg_weight=1.0,
                    cache=kv_cache,
                    temperature=TEMPERATURE,
                    film_gain=FILM_GAIN,
                    film_debug=MOVE_FILM_DEBUG,
                )
            else:
                new_pitch_token, _ = model.gen_pitch_token(
                    context,
                    style_context=style_contexts[active_style_idx],
                    style_context_mask=style_context_masks_list[active_style_idx],
                    harm_fields=move_window,
                    cfg_weight=move_cfg_weight,
                    temperature=TEMPERATURE,
                    film_gain=FILM_GAIN,
                    film_debug=MOVE_FILM_DEBUG,
                )
            last_gen_time = time.perf_counter()
        dict_output_tokens['pitch'][i + CTX_LEN] = new_pitch_token

        # Feed the generated note to the stuck detector; advance / release the pulse.
        prev_triad = step_tracker.current
        step_tracker.add_note(new_pitch_token)
        if move_notes_left > 0:
            move_notes_left -= 1
            if move_notes_left <= 0:
                move_cooldown = MOVE_COOLDOWN_NOTES
                moved = step_tracker.current is not None and step_tracker.current != prev_triad
                print(f"[move] pulse ended -> {triad_name(step_tracker.current) if step_tracker.current else '?'} "
                      f"({'moved' if moved else 'still settling'})")
        elif move_cooldown > 0:
            move_cooldown -= 1

        playNote(new_pitch_token, velocity)
        visualizer.get_note(new_pitch_token, velocity)
        visualizer.get_button(but, velocity)

        # add (pitch, time, button) to dictionary using original MIDI note as key
        noteOn_dict[note] = (new_pitch_token, timeNew, but)
        i += 1

    else:  # noteOff
        if note in noteOn_dict:
            pitch, noteOn_time, but = noteOn_dict[note]
            playNote(pitch, 0)
            visualizer.get_note(pitch, 0)
            visualizer.get_button(but, 0)
            del noteOn_dict[note]


"""# KEYBOARD LISTENER — 1..5 switch MIDI style, space saves motif, 6 activates motif,
   '.' move now, 'a' toggle auto stuck-detection, '['/']' cfg -/+, r/s reset/save """
def _on_key_press(key: pkeyboard.Key) -> None:
    global active_style_idx
    global kv_cache
    global style_context
    global style_context_mask
    global motif_style_ready
    global pending_move, auto_move_enabled, move_cfg_weight
    if key == pkeyboard.Key.space:
        capture_highlight_motif()
        return
    try:
        c = key.char  # type: ignore[union-attr]
        if c == MOVE_KEY:
            pending_move = True
            print("[move] manual harmonic step requested (applies on next note)")
            return
        if c == MOVE_TOGGLE_KEY:
            auto_move_enabled = not auto_move_enabled
            print(f"[move] auto stuck-detection {'ON' if auto_move_enabled else 'OFF'}")
            return
        if c == CFG_DOWN_KEY:
            move_cfg_weight = max(1.0, move_cfg_weight - MOVE_CFG_STEP)
            print(f"[move] cfg insistence = {move_cfg_weight:.1f}")
            return
        if c == CFG_UP_KEY:
            move_cfg_weight = move_cfg_weight + MOVE_CFG_STEP
            print(f"[move] cfg insistence = {move_cfg_weight:.1f}")
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
        MIDI_PORT = 0  #  Axiom 0 for Mac
    else:
        midiin = rtmidi.MidiIn(rtmidi.API_LINUX_ALSA)  #  for Linux
        MIDI_PORT = 1  # minicontrol32 in linux

    # List available ports
    available_ports = midiin.get_ports()

    if available_ports:
        print("Available MIDI input ports:")
        for idx, port in enumerate(available_ports):
            print(f"[{idx}] {port}")
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
        visualizer.draw()
except (EOFError, KeyboardInterrupt, SystemExit):
    print("Bye.")
