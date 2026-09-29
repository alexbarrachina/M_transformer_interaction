#===================================================================================================
# Monster Genie interaction_buttons_style_tensions.py Python module
# Real-time interaction for the AE_style_tensions model: buttons from a MIDI
# keyboard, style cross-attention, and CONTINUOUS tonal-tension FiLM conditioning
# driven by high-level performer commands (add tension / resolve / change tonal
# center / maintain).
#
# The tonal-tension state is tracked from the generated pitch stream by
# tension_extractor.RealtimeTensionExtractor and the command -> target trajectory
# is produced by tension_extractor.command_to_target (plan Module 9). No chord
# labels, no bars, no chord planner.
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
from typing import Optional, List, Tuple
from rtmidi.midiconstants import NOTE_ON, NOTE_OFF
from rtmidi.midiutil import open_midiinput
import rtmidi
from threading import Lock, Event
from pynput import keyboard as pkeyboard

from sympy.logic import false
import torch

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter
from visualizer import Visualizer
from tension_extractor import (
    RealtimeTensionExtractor, command_to_target, target_chroma_for_feature,
    TENSION_FEATURE_DIM, TF_KEY_POST, TF_TENSION_SCALAR,
    TF_KEY_CONF, TF_KEY_ENTROPY, TF_HARMONIC_CHANGE, TF_DIST_KEY, TF_DIST_TONIC,
    TF_DIST_SUBDOM, TF_DIST_DOM, TF_DISSONANCE, TF_MOD_PRESSURE, TF_TENSION_SLOPE,
    TF_RESOLUTION,
    CMD_MAINTAIN, CMD_ADD_TENSION, CMD_RESOLVE, CMD_CHANGE_TONAL_CENTER,
)

TRACES = False
USE_CACHE = True   # KV cache on: CFG is off by default so single-pass decoding is safe
CACHE_IDLE_TIMEOUT = 2.0
XINXE_INTERFACE = False

TEMPERATURE = 1

''' TENSION INFERENCE PARAMS '''
# Steering stack kept deliberately light: FiLM does the work; CFG off by
# default (it costs a 2nd decoder pass AND forces the KV cache off -> the
# latency you noticed); the tension critic rerank replaces it at zero decoder
# cost (top-K rescoring via the tension extractor only).
TENS_CFG_WEIGHT = 1.0          # classifier-free guidance strength (1.0 = off; >1 disables cache)
TENS_PC_BIAS = 1.0             # soft pitch-class logit bias toward target key tones (0.0 = off)
TENS_STRENGTH = 1.0           # command strength 0..1 (magnitude of the target adjustment)
TENS_DECAY_STEP = 1.0 / 15.0  # intensity release per generated note (~15-note span)
TENS_RERANK_K = 8             # top-K candidates rescored by the tension critic
TENS_RERANK_W = 2.0           # critic weight on the candidate log-probs (0 = off)
FILM_GAIN = 1.0
TENS_FILM_DEBUG = False

''' VISUALIZER PARAMS '''
VISUALIZER_WIDTH = 1400
HARMONY_PANEL_WIDTH = 650

_PC_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

''' DEVICE SPECIFIC PARAMETERS '''
if torch.backends.mps.is_available():
    device = torch.device('mps')
    CTX_LEN = 128
    TOTAL_GEN_LEN = 800
    KEY_OFFSET = 60 if XINXE_INTERFACE else 48
else:
    device = torch.device('cuda')
    CTX_LEN = 512
    TOTAL_GEN_LEN = 1024
    KEY_OFFSET = 60 if XINXE_INTERFACE else 34

''' MODEL '''
model_name = 'AE_style_tensions_tester_v1'
cfg = get_model_hparams(model_name)
model = load_model(model_name=model_name, cfg=cfg)
model.to(device)
model.eval()

''' PARAMS '''
sample_midi_path1 = './samples/Bach_Prelude_and_Fugue_in_C_major.mid'
sample_midi_path2 = './samples/clairTester_to_end.midi'
sample_midi_path3 = './samples/Chopin_Nocturnes_Op9No1_In_B_Flat_Minor.mid'
sample_midi_path4 = './samples/Scott_Cyril_Lotus_Land.mid'
sample_midi_path5 = './samples/Satie_Gymnopedie_No1.mid'

sample_midi_path_init = sample_midi_path1
STYLE_IDX_INIT = 1

style_prompt_midi_paths: List[str] = [
    sample_midi_path1, sample_midi_path2, sample_midi_path3,
    sample_midi_path4, sample_midi_path5,
]
output_midi_name = './out/interactive_performance'

PRIMER_PLAYBACK_LAST_N = 40
PRIMER_PLAYBACK_SPEED = 1

NUM_BUTTONS = cfg['num_buttons']
HIGHLIGHT_MOTIF_LEN = 30
MOTIF_STYLE_KEY = '6'

# Tonal-tension command keys (QWERTY). Pressing one requests a harmonic movement;
# the next generated note computes the target tension trajectory and FiLM-conditions
# the decoder, then the intensity decays back to the unconditional model (release).
TENSION_KEY_MAPPING = {
    'z': CMD_RESOLVE,
    'x': CMD_ADD_TENSION,
    'c': CMD_CHANGE_TONAL_CENTER,
    'v': CMD_MAINTAIN,
}
COMMAND_NAMES = {
    CMD_RESOLVE: 'RESOLVE', CMD_ADD_TENSION: 'ADD_TENSION',
    CMD_CHANGE_TONAL_CENTER: 'CHANGE_TONAL_CENTER', CMD_MAINTAIN: 'MAINTAIN',
}

# Tension-extractor hyperparameters (match training cfg).
T_SHORT = int(cfg.get('tension_short_window', 8))
T_ALPHA = float(cfg.get('tension_key_alpha', 2.0))
T_TRANS_ALPHA = float(cfg.get('tension_trans_alpha', 100.0))
T_TAU_LOW = float(cfg.get('tension_tau_low', 8.0))
T_TAU_HIGH = float(cfg.get('tension_tau_high', 3.0))
T_PROFILES = str(cfg.get('tension_profiles', 'genie'))
T_KEY_THETA = float(cfg.get('tension_key_theta', 3.0))
T_BASS_W = float(cfg.get('tension_bass_weight', 2.0))
T_CAD_GAP = float(cfg.get('tension_cadence_gap', 1.0))
T_CAD_BOOST = float(cfg.get('tension_cadence_boost', 2.0))


def _feature_key(feat: torch.Tensor) -> Tuple[int, str]:
    """Top key (root, mode) from a tension feature vector's key posterior."""
    post = feat[TF_KEY_POST]
    top = int(post.argmax().item())
    return top % 12, ('minor' if top >= 12 else 'major')


def _tension_detail_dict(feat: torch.Tensor) -> dict:
    """Full tension-component breakdown for the visualizer's detail meters."""
    return {
        'key_conf': float(feat[TF_KEY_CONF].item()),
        'key_entropy': float(feat[TF_KEY_ENTROPY].item()),
        'harmonic_change': float(feat[TF_HARMONIC_CHANGE].item()),
        'dist_key': float(feat[TF_DIST_KEY].item()),
        'dist_tonic': float(feat[TF_DIST_TONIC].item()),
        'dist_subdom': float(feat[TF_DIST_SUBDOM].item()),
        'dist_dom': float(feat[TF_DIST_DOM].item()),
        'dissonance': float(feat[TF_DISSONANCE].item()),
        'mod_pressure': float(feat[TF_MOD_PRESSURE].item()),
        'tension_slope': float(feat[TF_TENSION_SLOPE].item()),
        'resolution': float(feat[TF_RESOLUTION].item()),
    }


def _init_tension_buffers(n: int) -> dict:
    """Per-position tension buffers (parallel to the button buffer). The
    seed/context region stays neutral with intensity 0 (== unconditional)."""
    return {
        'tension_feat': [[0.0] * TENSION_FEATURE_DIM for _ in range(n)],
        'chroma': [[0.0] * 12 for _ in range(n)],
        'intensity': [0.0] * n,
    }


'''THREADING'''
buffer_lock = Lock()
save_lock = Lock()
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
        with buffer_lock:
            manageNote(note, velocity)
    if message[0] & 0xF0 == NOTE_OFF:
        status, note, velocity = message
        with buffer_lock:
            manageNote(note, 0)
    if message[0] & 0xF0 == 176:  # control change
        if message[1] == 18 and message[2] > 0:  # REC -> save
            with save_lock:
                print("saving performance")
                save_performance()
                sys.exit(0)
        if message[1] == 17 and message[2] > 0:  # PLAY -> reset
            print("resetting context")
            reset_requested.set()


def key_to_button(key):
    key = key - KEY_OFFSET
    button = key
    if XINXE_INTERFACE:
        button = key
    else:
        toWhite = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 10, 10, 11, 11, 12, 12, 13, 14, 14, 15, 15, 16, 17, 17, 18, 18, 19, 19, 20, 21, 21, 22, 22, 23, 23, 24]
        if key < 0 or key >= len(toWhite):
            button = 0
        else:
            button = toWhite[button]
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
    context = {
        'dtime': dict_output_tokens['dtime'][CTX_LEN:CTX_LEN + i],
        'pitch': dict_output_tokens['pitch'][CTX_LEN:CTX_LEN + i],
        'dur': dict_output_tokens['dur'][CTX_LEN:CTX_LEN + i],
    }
    song_d = dict_to_song(context)
    ms_SONG_to_MIDI_Converter(song_d, output_file_name=output_midi_name, timings_multiplier=1)
    print("saved performance")


def reset_context():
    global i
    global dict_output_tokens, dict_input_tokens
    global kv_cache
    global b
    global tension_buffers
    global current_command, current_intensity, current_phase
    global current_target_feat, current_target_chroma, pending_command
    global tension_tracker

    with buffer_lock:
        i = 0
        kv_cache = None
        current_command = CMD_MAINTAIN
        current_intensity = 0.0
        current_phase = 0.0
        current_target_feat = torch.zeros(TENSION_FEATURE_DIM, dtype=torch.float, device=device)
        current_target_chroma = [0.0] * 12
        pending_command = False
        tension_buffers = _init_tension_buffers(TOTAL_GEN_LEN + CTX_LEN)
        tension_tracker = RealtimeTensionExtractor(
            short_window=T_SHORT, key_alpha=T_ALPHA, trans_alpha=T_TRANS_ALPHA,
            tau_low=T_TAU_LOW, tau_high=T_TAU_HIGH, profiles=T_PROFILES,
            key_decision_theta=T_KEY_THETA, bass_weight=T_BASS_W,
            cadence_gap=T_CAD_GAP, cadence_boost=T_CAD_BOOST, device=device)
        visualizer.clear_harmony_history()
        visualizer.set_chord_chroma(None, False)
        for key in dict_input_tokens.keys():
            extended_list = dict_input_tokens[key].copy()
            extended_list.extend([0] * (TOTAL_GEN_LEN + CTX_LEN - len(extended_list)))
            dict_output_tokens[key] = extended_list
        seed_context = {
            'dtime': torch.tensor(dict_input_tokens['dtime'], dtype=torch.long).unsqueeze(0).to(device),
            'pitch': torch.tensor(dict_input_tokens['pitch'], dtype=torch.long).unsqueeze(0).to(device),
            'dur':   torch.tensor(dict_input_tokens['dur'],   dtype=torch.long).unsqueeze(0).to(device),
        }
        with torch.inference_mode():
            e = model.encoder(seed_context)
            b = model.real_to_discrete(e).squeeze(0).clone().detach().tolist()
        b.extend([0] * (TOTAL_GEN_LEN + CTX_LEN - len(b)))
        # Pre-analyze the context: global key prior + filter over the primer
        # notes, so the key posterior is already converged/stable at note one.
        tension_tracker.prime(dict_input_tokens['pitch'][:CTX_LEN])


''' VARIABLES '''
context = None
timeLast = 0
i = 0
noteOn_dict = {}
first_note = True
kv_cache = None
last_gen_time: float = 0.0

''' TENSION STATE '''
current_command = CMD_MAINTAIN
current_intensity = 0.0                       # 1.0 == command active, 0.0 == released
current_phase = 0.0                           # 1.0 at command onset -> 0 across the release span
current_target_feat = torch.zeros(TENSION_FEATURE_DIM, dtype=torch.float, device=device)
current_target_chroma: List[float] = [0.0] * 12
pending_command = False                        # set by a command key, consumed on the next note
tension_buffers: dict = {}

''' BUILD CTX '''
dict_input_tokens, num_notes = midi_to_dict(sample_midi_path_init)

dict_output_tokens = {}
for key in dict_input_tokens.keys():
    extended_list = dict_input_tokens[key].copy()
    extended_list.extend([0] * (TOTAL_GEN_LEN + CTX_LEN - len(extended_list)))
    dict_output_tokens[key] = extended_list

if TRACES:
    print("num_notes", num_notes)

# Real-time tension tracker over the generated pitch stream, primed with the
# context (whole-primer key pre-analysis + incremental filter over its notes).
tension_tracker = RealtimeTensionExtractor(
    short_window=T_SHORT, key_alpha=T_ALPHA, trans_alpha=T_TRANS_ALPHA,
    tau_low=T_TAU_LOW, tau_high=T_TAU_HIGH, profiles=T_PROFILES,
    key_decision_theta=T_KEY_THETA, bass_weight=T_BASS_W,
    cadence_gap=T_CAD_GAP, cadence_boost=T_CAD_BOOST, device=device)
tension_tracker.prime(dict_input_tokens['pitch'][:CTX_LEN])
_seed_feat = tension_tracker.current_features()
_seed_root, _seed_mode = _feature_key(_seed_feat)
print(f"Tension tracker primed key: {_PC_NAMES[_seed_root]} {_seed_mode} "
      f"(conf {float(_seed_feat[TF_KEY_CONF].item()):.2f})")
visualizer.set_harmony_state(
    0.0, 0.0, float(_seed_feat[TF_TENSION_SCALAR].item()),
    key_root=_seed_root, key_mode=_seed_mode,
    tension_detail=_tension_detail_dict(_seed_feat), active=False,
)

context = {
    'dtime': torch.tensor(dict_input_tokens['dtime'], dtype=torch.long).unsqueeze(0),
    'pitch': torch.tensor(dict_input_tokens['pitch'], dtype=torch.long).unsqueeze(0),
    'dur': torch.tensor(dict_input_tokens['dur'], dtype=torch.long).unsqueeze(0),
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


style_contexts: List[torch.Tensor] = []
style_context_masks_list: List[torch.Tensor] = []
for _spath in style_prompt_midi_paths:
    _style_tokens, _ = midi_to_dict(_spath)
    _sctx, _smask = _encode_style_pitch_list(_style_tokens['pitch'][:style_seq_len])
    style_contexts.append(_sctx)
    style_context_masks_list.append(_smask)
    print(f"Style {len(style_contexts)} encoded: {_spath}")

active_style_idx: int = STYLE_IDX_INIT - 1
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
    e = model.encoder(context)
    b = model.real_to_discrete(e).squeeze(0)
    b = b.clone().detach().tolist()
    b.extend([0] * (TOTAL_GEN_LEN + CTX_LEN - len(b)))

tension_buffers = _init_tension_buffers(TOTAL_GEN_LEN + CTX_LEN)


def _write_tension(idx: int) -> None:
    """Record the current tension target at position idx (per generated note)."""
    tension_buffers['tension_feat'][idx] = current_target_feat.detach().cpu().tolist()
    tension_buffers['chroma'][idx] = list(current_target_chroma)
    tension_buffers['intensity'][idx] = float(current_intensity)


def _tension_window(lo: int, hi: int) -> dict:
    """Build the harm_fields tensor window [1, hi-lo, ...] for gen_pitch_token."""
    win = {}
    win['tension_feat'] = torch.tensor(
        tension_buffers['tension_feat'][lo:hi], dtype=torch.float).unsqueeze(0).to(device)
    win['chroma'] = torch.tensor(
        tension_buffers['chroma'][lo:hi], dtype=torch.float).unsqueeze(0).to(device)
    win['intensity'] = torch.tensor(
        tension_buffers['intensity'][lo:hi], dtype=torch.float).unsqueeze(0).to(device)
    return win


def _apply_command() -> None:
    """Compute the target tension trajectory for the requested command from the
    current tracked tension state (plan Module 9). Runs under buffer_lock."""
    global current_target_feat, current_target_chroma
    cur_feat = tension_tracker.current_features().to(device)
    current_target_feat = command_to_target(cur_feat, current_command, TENS_STRENGTH)
    current_target_chroma = target_chroma_for_feature(current_target_feat).cpu().tolist()
    root, mode = _feature_key(current_target_feat)
    print(f"command [{COMMAND_NAMES.get(current_command, current_command)}]: "
          f"target key {_PC_NAMES[root]} {mode}  "
          f"tension={float(current_target_feat[TF_TENSION_SCALAR].item()):.2f}")


def _update_tension_visualizer() -> None:
    """Push the current tracked tension state into the visualizer panel.
    The displayed key uses the tracker's hysteretic decision (stable)."""
    feat = tension_tracker.current_features()
    top = tension_tracker.top_key()
    root, mode = top % 12, ('minor' if top >= 12 else 'major')
    mag = float(feat[TF_TENSION_SCALAR].item())
    active = current_intensity > 0.0
    target_chroma = current_target_chroma if active else None
    detail = _tension_detail_dict(feat)
    nd = tension_tracker.neighbor_distances()
    sig = tension_tracker.fifths_signature()
    for k in ('p_relative', 'p_parallel', 'p_dominant', 'p_subdominant'):
        detail[k] = nd.get(k, 0.0)
    detail['fifths_shift'] = sig['fifths_shift']
    detail['mode_axis'] = sig['mode_score']
    visualizer.set_harmony_state(
        0.0, 0.0, mag, key_root=root, key_mode=mode,
        target_chroma=target_chroma,
        movement_label=COMMAND_NAMES.get(current_command, ''),
        tension_detail=detail,
        active=active,
    )


def _tension_rerank(cands: List[int]) -> torch.Tensor:
    """Tension critic for gen_pitch_token's top-K rerank: score each candidate
    pitch by how its hypothetical realized tension state relates to the active
    command target (higher = better). Fully batched on the tension extractor
    (peek_features_batch, ~0.7 ms for K=8) — zero extra decoder passes, so it
    is KV-cache friendly, unlike CFG.

    Command-aware component weights (validated by ranking sanity checks):
      A: key-posterior alignment with the target posterior  (key changes)
      B: target-chroma bonus for the candidate's pitch class (triad tones)
      C: closeness on tension_scalar/dist_key/dissonance     (tension moves)
    """
    tgt = current_target_feat.detach().cpu()
    tch = torch.tensor(current_target_chroma, dtype=torch.float)
    feats = tension_tracker.peek_features_batch(cands)            # [K,48]
    A = feats[:, TF_KEY_POST] @ tgt[TF_KEY_POST]
    B = tch[torch.tensor(cands) % 12]
    C = -((feats[:, TF_TENSION_SCALAR] - tgt[TF_TENSION_SCALAR]).abs()
          + (feats[:, TF_DIST_KEY] - tgt[TF_DIST_KEY]).abs()
          + (feats[:, TF_DISSONANCE] - tgt[TF_DISSONANCE]).abs())

    def z(x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean()) / x.std().clamp_min(1e-8)

    if current_command == CMD_ADD_TENSION:
        wA, wB, wC = 0.0, 0.0, 1.0
    elif current_command == CMD_RESOLVE:
        wA, wB, wC = 0.3, 0.5, 1.0
    elif current_command == CMD_CHANGE_TONAL_CENTER:
        wA, wB, wC = 1.0, 1.0, 0.3
    else:  # maintain
        wA, wB, wC = 0.3, 0.3, 0.5
    return wA * z(A) + wB * B + wC * z(C)


visualizer.primer(dict_input_tokens['pitch'][:CTX_LEN], dict_input_tokens['dtime'][:CTX_LEN],
                  b[:CTX_LEN], dict_input_tokens['dur'][:CTX_LEN])


def manageNote(note, velocity):
    global context
    global timeLast
    global b
    global i
    global dict_output_tokens
    global noteOn_dict
    global first_note
    global visualizer
    global kv_cache
    global last_gen_time
    global active_style_idx
    global current_intensity
    global current_phase
    global pending_command
    global tension_tracker

    if TRACES:
        print("key", note)

    timeNew = time.perf_counter() * 1000 / 32

    if velocity > 0:  # noteOn
        now = time.perf_counter()
        if USE_CACHE and kv_cache is not None and (now - last_gen_time) > CACHE_IDLE_TIMEOUT:
            kv_cache = None

        dtime = max(0, min(127, int(timeNew) - int(timeLast)))
        if first_note:
            dtime = 0
            first_note = False

        timeLast = timeNew
        dict_output_tokens['dtime'][i + CTX_LEN] = dtime

        try:
            but = key_to_button(note)
        except:
            but = 0
            print("ERROR key_to_button", note)
        b[i + CTX_LEN] = but

        # --- Tension: on a command request, compute the target trajectory; then
        # record the tension state for this position so the decoder is conditioned.
        idx = i + CTX_LEN
        if pending_command:
            _apply_command()
            current_intensity = 1.0
            current_phase = 1.0
            pending_command = False
        _write_tension(idx)
        tension_window = _tension_window(i, i + CTX_LEN + 1)

        context = {
            'dtime': torch.tensor(dict_output_tokens['dtime'][i:i + CTX_LEN + 1], dtype=torch.long).unsqueeze(0),
            'pitch': torch.tensor(dict_output_tokens['pitch'][i:i + CTX_LEN + 1], dtype=torch.long).unsqueeze(0),
            'dur': torch.tensor(dict_output_tokens['dur'][i:i + CTX_LEN + 1], dtype=torch.long).unsqueeze(0),
            'button': torch.tensor(b[i:i + CTX_LEN + 1], dtype=torch.long).unsqueeze(0),
        }
        context = to_device(context, device)
        if TRACES:
            print("ctx", i + CTX_LEN + 1, "of", TOTAL_GEN_LEN + CTX_LEN)

        # Tension critic rerank: only while a command is active (zero cost when
        # inactive; no extra decoder passes when active).
        rr_active = TENS_RERANK_W > 0.0 and current_intensity > 0.0
        rr_fn = _tension_rerank if rr_active else None
        rr_w = TENS_RERANK_W * current_intensity if rr_active else 0.0

        with torch.inference_mode():
            if USE_CACHE:
                # CFG needs two passes (would corrupt the cache) -> no guidance when cached.
                new_pitch_token, kv_cache = model.gen_pitch_token(
                    context,
                    style_context=style_contexts[active_style_idx],
                    style_context_mask=style_context_masks_list[active_style_idx],
                    harm_fields=tension_window,
                    cfg_weight=1.0,
                    pc_bias_weight=TENS_PC_BIAS,
                    cache=kv_cache,
                    temperature=TEMPERATURE,
                    film_gain=FILM_GAIN,
                    film_debug=TENS_FILM_DEBUG,
                    rerank_fn=rr_fn, rerank_topk=TENS_RERANK_K, rerank_weight=rr_w,
                )
            else:
                new_pitch_token, _ = model.gen_pitch_token(
                    context,
                    style_context=style_contexts[active_style_idx],
                    style_context_mask=style_context_masks_list[active_style_idx],
                    harm_fields=tension_window,
                    cfg_weight=TENS_CFG_WEIGHT,
                    pc_bias_weight=TENS_PC_BIAS,
                    temperature=TEMPERATURE,
                    film_gain=FILM_GAIN,
                    film_debug=TENS_FILM_DEBUG,
                    rerank_fn=rr_fn, rerank_topk=TENS_RERANK_K, rerank_weight=rr_w,
                )
            last_gen_time = time.perf_counter()
        dict_output_tokens['pitch'][i + CTX_LEN] = new_pitch_token

        visualizer.set_chord_chroma(current_target_chroma, current_intensity > 0.0)

        # Release: intensity ramps linearly toward 0 across the command span
        # (same shape as the training intensity bursts), returning the model to
        # the unconditional (tension-released) state.
        if current_phase > 0.0:
            current_phase = max(0.0, current_phase - TENS_DECAY_STEP)
            current_intensity = current_phase
        if TRACES:
            print("intensity", current_intensity, "phase", current_phase)

        playNote(new_pitch_token, velocity)
        tension_tracker.add_pitch(new_pitch_token)
        visualizer.get_note(new_pitch_token, velocity)
        visualizer.get_button(but, velocity)
        _update_tension_visualizer()

        noteOn_dict[note] = (new_pitch_token, timeNew, but)
        i += 1

    else:  # noteOff
        if note in noteOn_dict:
            pitch, noteOn_time, but = noteOn_dict[note]
            playNote(pitch, 0)
            visualizer.get_note(pitch, 0)
            visualizer.get_button(but, 0)
            _update_tension_visualizer()
            del noteOn_dict[note]


"""# KEYBOARD LISTENER — tension command keys z/x/c/v, 1..5 style, space motif, 6 activate, r reset, s save """
def _on_key_press(key: pkeyboard.Key) -> None:
    global active_style_idx
    global kv_cache
    global style_context
    global style_context_mask
    global motif_style_ready
    global current_command
    global pending_command
    if key == pkeyboard.Key.space:
        capture_highlight_motif()
        return
    try:
        c = key.char  # type: ignore[union-attr]
        if c in TENSION_KEY_MAPPING:
            current_command = TENSION_KEY_MAPPING[c]
            pending_command = True
            print(f"Command: {COMMAND_NAMES.get(current_command, current_command)}")
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
    if torch.backends.mps.is_available():
        midiin = rtmidi.MidiIn(rtmidi.API_MACOSX_CORE)
        MIDI_PORT = 0
    else:
        midiin = rtmidi.MidiIn(rtmidi.API_LINUX_ALSA)
        MIDI_PORT = 1

    available_ports = midiin.get_ports()
    if available_ports:
        print("Available MIDI input ports:")
        for i, port in enumerate(available_ports):
            print(f"[{i}] {port}")
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
        with buffer_lock:
            _update_tension_visualizer()
            visualizer.draw()
except (EOFError, KeyboardInterrupt, SystemExit):
    print("Bye.")
