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


from operator import truediv
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

TRACES = True
TRACE_BUTTONS = True
USE_CACHE = False
CACHE_IDLE_TIMEOUT = 2.0  # seconds - clear KV cache after this idle gap
XINXE_INTERFACE = False

TEMPERATURE = 1#0.0001

# Gap-phase: boost injected-pitch side-channel (try 1.5, 2.0, 3.0; >1.0 is mildly OOD).
ANTIC_STRENGTH = 2.0
# Post-injection echo: keep injected pitch visible + shorten history after SPACE release.
POST_INJ_STEPS = 8
POST_INJ_CTX_LEN = 64  # try 32 or 64
POST_INJ_ANTIC_STRENGTH = 1.5

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
  
model_name = 'AE_antic_style_joker_tester_v1' 
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

sample_midi_path_init = sample_midi_path3
STYLE_IDX_INIT = 3

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
    if key==36 or key==37: #  joker butttons
        print("button", model.joker_button_idx)
        return model.joker_button_idx
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
    global post_inj_steps_left, last_injected_anchor

    with buffer_lock:
        i = 0
        kv_cache = None
        post_inj_steps_left = 0
        last_injected_anchor = -1
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

delta: int = int(cfg.get('anticipation_delta', 4))
style_seq_len = int(cfg.get('style_seq_len', CTX_LEN))

def _encode_style_pitch_list(pitch_list: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_len = min(len(pitch_list), style_seq_len)
    if valid_len <= 0:
        _spp = torch.full((1, style_seq_len), PAD_IDX, dtype=torch.long, device=device)
        _smask = torch.zeros((1, style_seq_len), dtype=torch.bool, device=device)
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
style_context = style_contexts[active_style_idx]
style_context_mask = style_context_masks_list[active_style_idx]

with torch.inference_mode():
    e = model.encoder(context) # encoder output (batch, seq_len)
    b = model.real_to_discrete(e).squeeze(0) # generate buttons (batch, seq_len)
    b = b.clone().detach().tolist()
    b.extend([0] * (TOTAL_GEN_LEN + CTX_LEN - len(b)))

''' INJECTION MODE STATE '''
injection_mode: bool = False   # True while SPACE is held
need_gap: bool = False         # True after SPACE press, before first injection
n_injected: int = 0            # total injected notes in current span
i_before_injection: int = 0   # value of i when SPACE was pressed
# Incremental gap-fill buffers (pipelined: 1 inference per injected note)
gap_pitches_buf: List[int] = []   # bridge pitches generated so far (unplayed)
gap_buttons_cond: List[int] = []  # interpolated conditioning button per bridge
inj_pitches_buf: List[int] = []   # injected pitches played during the gap phase
inj_dtimes_buf: List[int] = []    # their dtimes
antic_target_pitch: int = -1      # first injected note = anticipation target
gap_start_button: int = 0         # last generated button before injection (interp start)
gap_target_button: int = 0        # first injected note's estimated button (interp end)
post_inj_steps_left: int = 0
last_injected_anchor: int = -1

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
  global injection_mode, need_gap, n_injected
  global active_style_idx
  global antic_target_pitch, gap_start_button, gap_target_button
  global post_inj_steps_left
  if TRACES:
    print("key", note)
    print("injection_mode", injection_mode)
    print("need_gap", need_gap)
    print("n_injected", n_injected)

  timeNew = time.perf_counter()*1000 /32 # in miliseconds /32 as in midi_to_dict()

  if velocity > 0: # noteOn
    now = time.perf_counter()
    if USE_CACHE and kv_cache is not None and (now - last_gen_time) > CACHE_IDLE_TIMEOUT:
        kv_cache = None

    dtime = max(0, min(127, int(timeNew) - int(timeLast))) # time difference from previous events, but trunk to maximum 127
    if first_note:
        dtime = 0
        first_note = False
    timeLast = timeNew

    if injection_mode:
      # Audio is always immediate (raw injected pitch); model work is pipelined.
      playNote(note, velocity)
      visualizer.get_note(note, velocity)
      visualizer.get_button(0, velocity)
      noteOn_dict[note] = (note, timeNew, 0)

      if need_gap:
        # GAP PHASE: spread bridge generation over the first `delta` injected
        # notes — exactly one inference per keypress. The token block is only
        # finalized once `delta` bridges exist (or on SPACE release).
        if len(inj_pitches_buf) == 0:
            # First injected note fixes the anticipation target and the button
            # interpolation endpoints (last generated button → injected button).
            antic_target_pitch = note
            gap_start_button = b[i + CTX_LEN - 1]
            try:
                gap_target_button = key_to_button(note)
            except:
                gap_target_button = gap_start_button
        inj_pitches_buf.append(note)
        inj_dtimes_buf.append(dtime)

        # Generate ONE bridge pitch (anticipating the first injected note).
        g_idx: int = len(gap_pitches_buf)
        ctx_pitch: torch.Tensor = torch.tensor(
            dict_output_tokens['pitch'][i:i+CTX_LEN] + gap_pitches_buf,
            dtype=torch.long
        ).unsqueeze(0).to(device)
        ctx_button: torch.Tensor = torch.tensor(
            b[i:i+CTX_LEN] + gap_buttons_cond,
            dtype=torch.long
        ).unsqueeze(0).to(device)
        frac: float = float(g_idx + 1) / float(delta)
        cond_btn: int = int(round(gap_start_button + frac * (gap_target_button - gap_start_button)))
        cond_btn = max(0, min(cfg['num_buttons'] - 1, cond_btn))
        with torch.inference_mode():
            g_pitch: int = model.gen_anticipation_bridge_step(
                ctx_pitch, ctx_button, antic_target_pitch, CTX_LEN - 1,
                cond_btn,
                style_context=style_contexts[active_style_idx],
                style_context_mask=style_context_masks_list[active_style_idx],
                antic_strength=ANTIC_STRENGTH,
                temperature=TEMPERATURE
            )
        gap_pitches_buf.append(g_pitch)
        gap_buttons_cond.append(cond_btn)

        # Once `delta` bridges exist, commit [history, gap, injected] + buttons.
        if len(gap_pitches_buf) >= delta:
            _commit_gap()
      else:
        # Steady-state injection (note beyond the gap): write pitch +
        # placeholder button. Full button recompute deferred to SPACE release.
        dict_output_tokens['dtime'][i + CTX_LEN] = dtime
        dict_output_tokens['pitch'][i + CTX_LEN] = note
        b[i + CTX_LEN] = b[i + CTX_LEN - 1]  # placeholder: copy previous button
        i += 1
        n_injected += 1

    else:
      # Button mode: key → button → gen_pitch_token
      dict_output_tokens['dtime'][i+CTX_LEN] = dtime
      try:
            but = key_to_button(note)
      except:
            but = 0
            print("ERROR key_to_button", note)
      b[i+CTX_LEN] = but
      ctx_len: int = POST_INJ_CTX_LEN if post_inj_steps_left > 0 else CTX_LEN
      end: int = i + CTX_LEN + 1
      start: int = max(0, end - (ctx_len + 1))
      context = {
        'pitch': torch.tensor(dict_output_tokens['pitch'][start:end], dtype=torch.long).unsqueeze(0),
        'button': torch.tensor(b[start:end], dtype=torch.long).unsqueeze(0)
      }
      context = to_device(context, device)
      if post_inj_steps_left > 0 and last_injected_anchor >= 0:
        L: int = context['pitch'].shape[1]
        antic_pitch: torch.Tensor = torch.zeros((1, L), dtype=torch.long, device=device)
        antic_mask: torch.Tensor = torch.zeros((1, L), dtype=torch.float, device=device)
        # -1 is the next-token placeholder, so put the echo on the last real token.
        antic_pitch[0, -2] = last_injected_anchor
        antic_mask[0, -2] = POST_INJ_ANTIC_STRENGTH
        context['antic_pitch'] = antic_pitch
        context['antic_mask'] = antic_mask
        context['mode'] = torch.ones(1, dtype=torch.long, device=device)
      if TRACES:
        print("ctx", i+CTX_LEN+1, "of", TOTAL_GEN_LEN+CTX_LEN)
      with torch.inference_mode():
        new_pitch_token = model.gen_pitch_token(
            context,
            style_context=style_contexts[active_style_idx],
            style_context_mask=style_context_masks_list[active_style_idx],
            temperature=TEMPERATURE
        )
        last_gen_time = time.perf_counter()
      dict_output_tokens['pitch'][i+CTX_LEN] = new_pitch_token
      playNote(new_pitch_token, velocity)
      visualizer.get_note(new_pitch_token, velocity)
      visualizer.get_button(but, velocity)
      # add (pitch, time, button) to dictionary using original MIDI note as key
      noteOn_dict[note] = (new_pitch_token, timeNew, but)
      i += 1
      if post_inj_steps_left > 0:
        post_inj_steps_left -= 1

  else: # noteOff
    # Use original MIDI note as key to find corresponding noteOn
    if note in noteOn_dict:

      # get pitch, time, and button from dictionary of accumulated notesOns without noteOff
      pitch, noteOn_time, but = noteOn_dict[note]
      playNote(pitch, 0)
      visualizer.get_note(pitch, 0)
      visualizer.get_button(but, 0)
      # Remove from dictionary to allow the same note to be played again
      del noteOn_dict[note]
      #visualizer.update(noteOn_time)


"""# KEYBOARD LISTENER — SPACE toggles injection mode; keys 1/2/3/4/5 switch MIDI style, r=reset, s=save """
def _recompute_buttons_for_injection() -> None:
    """Recompute encoder buttons for ALL gap + injected positions at once (caveat 1).
    Called once on SPACE release. Caller must hold buffer_lock.
    Encoder input length = CTX_LEN+delta+n_injected-1; slight overrun vs seq_len is
    accepted because USE_CACHE=False and rotary positions handle it (caveat 2)."""
    global b
    if n_injected == 0:
        return
    # Full pitch slice: pre-injection window (CTX_LEN) + gap (delta) + all injections
    total_len: int = CTX_LEN + delta + n_injected
    pitch_seq: torch.Tensor = torch.tensor(
        dict_output_tokens['pitch'][i_before_injection : i_before_injection + total_len],
        dtype=torch.long
    ).unsqueeze(0).to(device)
    # Training convention: encoder receives pitch[:, 1:]
    enc_input = {'pitch': pitch_seq[:, 1:]}
    with torch.inference_mode():
        all_buttons: torch.Tensor = model.gen_buttons(enc_input)  # [1, total_len-1]
    # Gap+injected buttons start at encoder index CTX_LEN-1 (= N-1 in gen_anticipation_gap)
    new_buttons: torch.Tensor = all_buttons[0, CTX_LEN - 1:]  # [delta + n_injected]
    write_start: int = i_before_injection + CTX_LEN
    for idx in range(delta + n_injected):
        b[write_start + idx] = new_buttons[idx].item()
    if TRACES:
        print(f"recomputed {delta + n_injected} buttons for injection span of {n_injected} notes")

def _commit_gap() -> None:
    """Commit the accumulated gap: write the `delta` bridge notes (dtime=0,
    unplayed) followed by the injected notes, advance `i`, then recompute
    coherent buttons over the whole block. Caller must hold buffer_lock."""
    global i, n_injected, need_gap
    n_bridges: int = len(gap_pitches_buf)
    m: int = len(inj_pitches_buf)
    # Bridges first (unplayed, dtime=0)
    for g in range(n_bridges):
        dict_output_tokens['dtime'][i + CTX_LEN + g] = 0
        dict_output_tokens['pitch'][i + CTX_LEN + g] = gap_pitches_buf[g]
        b[i + CTX_LEN + g] = gap_buttons_cond[g]
    # Injected notes after the bridges (placeholder buttons, recomputed below)
    for j in range(m):
        pos: int = i + CTX_LEN + n_bridges + j
        dict_output_tokens['dtime'][pos] = inj_dtimes_buf[j]
        dict_output_tokens['pitch'][pos] = inj_pitches_buf[j]
        b[pos] = b[pos - 1]
    i += n_bridges + m
    n_injected = m
    need_gap = False
    _recompute_buttons_for_injection()

def _finish_gap_fallback() -> None:
    """SPACE released before `delta` bridges were built: synchronously generate
    the remaining bridges (one-time cost, short spans only) then commit.
    Caller must hold buffer_lock."""
    global need_gap
    if len(inj_pitches_buf) == 0:
        need_gap = False          # SPACE tapped with no injected note
        return
    while len(gap_pitches_buf) < delta:
        g_idx: int = len(gap_pitches_buf)
        ctx_pitch: torch.Tensor = torch.tensor(
            dict_output_tokens['pitch'][i:i+CTX_LEN] + gap_pitches_buf,
            dtype=torch.long
        ).unsqueeze(0).to(device)
        ctx_button: torch.Tensor = torch.tensor(
            b[i:i+CTX_LEN] + gap_buttons_cond,
            dtype=torch.long
        ).unsqueeze(0).to(device)
        frac: float = float(g_idx + 1) / float(delta)
        cond_btn: int = int(round(gap_start_button + frac * (gap_target_button - gap_start_button)))
        cond_btn = max(0, min(cfg['num_buttons'] - 1, cond_btn))
        with torch.inference_mode():
            g_pitch: int = model.gen_anticipation_bridge_step(
                ctx_pitch, ctx_button, antic_target_pitch, CTX_LEN - 1,
                cond_btn,
                style_context=style_contexts[active_style_idx],
                style_context_mask=style_context_masks_list[active_style_idx],
                antic_strength=ANTIC_STRENGTH,
                temperature=TEMPERATURE
            )
        gap_pitches_buf.append(g_pitch)
        gap_buttons_cond.append(cond_btn)
    _commit_gap()

def _on_key_press(key: pkeyboard.Key) -> None:
    global injection_mode, need_gap, n_injected, i_before_injection
    global active_style_idx, kv_cache, style_context, style_context_mask
    if key == pkeyboard.Key.space:
        with buffer_lock:
            if not injection_mode:  # avoid re-triggering on key-repeat
                injection_mode = True
                need_gap = True
                n_injected = 0
                i_before_injection = i
                gap_pitches_buf.clear()
                gap_buttons_cond.clear()
                inj_pitches_buf.clear()
                inj_dtimes_buf.clear()
        return
    try:
        c = key.char  # type: ignore[union-attr]
        if c in ('1', '2', '3', '4', '5'):
            active_style_idx = int(c) - 1
            style_context = style_contexts[active_style_idx]
            style_context_mask = style_context_masks_list[active_style_idx]
            kv_cache = None
            print(f"Style {c} active: {style_prompt_midi_paths[active_style_idx]}")
        if c in ('r'):
            print("resetting context")
            reset_context()
        if c in ('s'):
            save_performance()
            print("saved performance")
    except AttributeError:
        pass

def _on_key_release(key: pkeyboard.Key) -> None:
    global injection_mode
    global post_inj_steps_left, last_injected_anchor
    if key == pkeyboard.Key.space:
        with buffer_lock:
            injection_mode = False
            if need_gap:
                _finish_gap_fallback()  # complete + commit a short/aborted gap
            else:
                _recompute_buttons_for_injection()
            if n_injected > 0:
                if len(inj_pitches_buf) > 0:
                    last_injected_anchor = inj_pitches_buf[-1]
                else:
                    last_injected_anchor = dict_output_tokens['pitch'][i + CTX_LEN - 1]
                post_inj_steps_left = POST_INJ_STEPS

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

