#===================================================================================================
# Monster Genie interaction_buttons_style_tensions_tester.py Python module
# Tonal-tension TESTER: no model, no buttons, no pitch generation. Notes played
# on a MIDI keyboard are passed straight through to the synth and fed to
# tension_extractor.RealtimeTensionExtractor so the live tonal-tension state
# (key, dissonance, tension scalar, sounding chroma) can be visualised in real
# time via the Visualizer's harmony (Tonnetz) panel.
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
import atexit
from typing import Dict, List, Tuple
from rtmidi.midiconstants import NOTE_ON, NOTE_OFF
import rtmidi
from threading import Lock
from pynput import keyboard as pkeyboard

import torch

from midiUtils import dict_to_song, ms_SONG_to_MIDI_Converter
from visualizer import Visualizer
from tension_extractor import (
    RealtimeTensionExtractor, TF_KEY_POST, TF_TENSION_SCALAR,
    TF_KEY_CONF, TF_KEY_ENTROPY, TF_HARMONIC_CHANGE, TF_DIST_KEY, TF_DIST_TONIC,
    TF_DIST_SUBDOM, TF_DIST_DOM, TF_DISSONANCE, TF_MOD_PRESSURE, TF_TENSION_SLOPE,
    TF_RESOLUTION,
)

TRACES = True

''' TENSION TRACKER PARAMS (match the AE_style_tensions training defaults) '''
T_SHORT = 8
T_ALPHA = 2.0         # key-correlation emission gain
T_TRANS_ALPHA = 100.0 # circle-of-fifths ring transition ratio (per-note)
T_TAU_LOW = 8.0      # integrator decay [s], bass end
T_TAU_HIGH = 3.0     # integrator decay [s], treble end
T_PROFILES = 'genie'

''' VISUALIZER PARAMS '''
VISUALIZER_WIDTH = 1400
HARMONY_PANEL_WIDTH = 650

_PC_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

output_midi_name = './out/tension_test_performance'

device = torch.device('cpu')

'''THREADING'''
buffer_lock = Lock()

'''VISUALIZER'''
visualizer = Visualizer(width=VISUALIZER_WIDTH, harmony_panel_width=HARMONY_PANEL_WIDTH)

'''TENSION TRACKER'''
tension_tracker = RealtimeTensionExtractor(
    short_window=T_SHORT, key_alpha=T_ALPHA, trans_alpha=T_TRANS_ALPHA,
    tau_low=T_TAU_LOW, tau_high=T_TAU_HIGH, profiles=T_PROFILES, device=device)

''' PERFORMANCE RECORDING (optional: 's' key / REC CC saves what was played) '''
recorded_pitches: List[int] = []
recorded_dtimes: List[int] = []
recorded_durs: List[int] = []
noteOn_dict: Dict[int, Tuple[int, float]] = {}  # midi note -> (index in recorded_*, perf-counter time-units at note-on)
timeLast = 0
first_note = True


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

def _update_tension_visualizer() -> None:
    """Push the live tracked tension state (key, tension scalar, sounding
    chroma) into the visualizer's harmony panel. The displayed key uses the
    tracker's hysteretic decision (stable at near-ties)."""
    feat = tension_tracker.current_features()
    top = tension_tracker.top_key()
    root, mode = top % 12, ('minor' if top >= 12 else 'major')
    mag = float(feat[TF_TENSION_SCALAR].item())
    chroma = tension_tracker.current_chroma().tolist()
    active_pcs = [pc for pc in range(12) if chroma[pc] > 0.0]
    detail = _tension_detail_dict(feat)
    nd = tension_tracker.neighbor_distances()
    sig = tension_tracker.fifths_signature()
    for k in ('p_relative', 'p_parallel', 'p_dominant', 'p_subdominant'):
        detail[k] = nd.get(k, 0.0)
    detail['fifths_shift'] = sig['fifths_shift']
    detail['mode_axis'] = sig['mode_score']
    visualizer.set_harmony_state(
        0.0, 0.0, mag, key_root=root, key_mode=mode,
        active_pcs=active_pcs, tension_detail=detail, active=True,
    )
    if TRACES:
        print(f"tension={mag:.2f} key={_PC_NAMES[root]} {mode}  "
              f"pull rel={nd.get('p_relative', 0):.2f} par={nd.get('p_parallel', 0):.2f} "
              f"dom={nd.get('p_dominant', 0):.2f} sub={nd.get('p_subdominant', 0):.2f}  "
              f"fifths={sig['fifths_shift']:+.1f} mode={sig['mode_score']:+.1f}")


def save_performance() -> None:
    if len(recorded_pitches) <= 0:
        print("nothing recorded to save")
        return
    context = {'dtime': list(recorded_dtimes), 'pitch': list(recorded_pitches), 'dur': list(recorded_durs)}
    song_d = dict_to_song(context)
    ms_SONG_to_MIDI_Converter(song_d, output_file_name=output_midi_name, timings_multiplier=1)
    print("saved performance")


def reset_tension() -> None:
    with buffer_lock:
        tension_tracker.reset()
        visualizer.clear_harmony_history()
    print("tension tracker reset")


def toggle_key_lock() -> None:
    """Pin/unpin the tracked key to the current top key ('k'). While locked the
    posterior is one-hot, so all key-derived features are perfectly stable."""
    with buffer_lock:
        if tension_tracker.locked_key is None:
            top = tension_tracker.top_key()
            tension_tracker.lock_key(top)
            print(f"key LOCKED to {_PC_NAMES[top % 12]} {'minor' if top >= 12 else 'major'}")
        else:
            tension_tracker.lock_key(None)
            print("key unlocked")


def manageNote(note, velocity) -> None:
    global timeLast, first_note

    if TRACES:
        print("key", note)

    timeNew = time.perf_counter() * 1000 / 32

    if velocity > 0:  # noteOn
        dtime = max(0, min(127, int(timeNew) - int(timeLast)))
        if first_note:
            dtime = 0
            first_note = False
        timeLast = timeNew

        playNote(note, velocity)
        tension_tracker.add_pitch(note, velocity)
        _update_tension_visualizer()
        visualizer.get_note(note, velocity)

        recorded_pitches.append(int(note))
        recorded_dtimes.append(int(dtime))
        recorded_durs.append(0)
        noteOn_dict[note] = (len(recorded_durs) - 1, timeNew)

    else:  # noteOff
        if note in noteOn_dict:
            idx, onTime = noteOn_dict[note]
            recorded_durs[idx] = max(0, min(127, int(timeNew) - int(onTime)))
            playNote(note, 0)
            visualizer.get_note(note, 0)
            del noteOn_dict[note]


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
            print("saving performance")
            save_performance()
            sys.exit(0)
        if message[1] == 17 and message[2] > 0:  # PLAY -> reset tension tracker
            reset_tension()


"""# KEYBOARD LISTENER — r reset tracker, k toggle key lock, s save performance """
def _on_key_press(key: pkeyboard.Key) -> None:
    try:
        c = key.char  # type: ignore[union-attr]
        if c == 'r':
            reset_tension()
        if c == 'k':
            toggle_key_lock()
        if c == 's':
            save_performance()
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

    while True:
        time.sleep(0.0001)
        with buffer_lock:
            visualizer.draw()
except (EOFError, KeyboardInterrupt, SystemExit):
    print("Bye.")
