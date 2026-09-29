#===================================================================================================
# Monster Genie interaction_chord_cond.py Python module
# Interaction with chord-conditioned residual adapter (AE_buttons_p_residual).
# Lower MIDI octave → melodic-shape buttons (each press generates a note).
# Upper MIDI octave → chord notes (held notes define the active chord conditioning).
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
from typing import Optional, List
from rtmidi.midiconstants import NOTE_ON, NOTE_OFF
from rtmidi.midiutil import open_midiinput
import rtmidi
from threading import Lock

import torch

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter
from visualizer import Visualizer


TRACES = False
USE_CACHE = False
CACHE_IDLE_TIMEOUT = 2.0  # seconds - clear KV cache after this idle gap

# --- Keyboard split ---
# Lower octave: buttons (each key press triggers note generation)
BUTTON_OCTAVE_START = 69   # A4
BUTTON_OCTAVE_END   = 79   # G5
# Upper octave: chord notes (held notes define active chord conditioning)
CHORD_OCTAVE_START  = 48   # C3
CHORD_OCTAVE_END    = 60   # C4

# White-key mapping for buttons: 12 chromatic → condensed button indices
# C=0,C#=0,D=1,D#=1,E=2,F=3,F#=3,G=4,G#=4,A=5,A#=5,B=6
TO_BUTTON = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6]

''' DEVICE '''
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda')


''' MODEL '''
'''
model_name = 'AE_residual_v1'
cfg = get_model_hparams(model_name)


# Load frozen base model, then wrap in residual adapter
base_model_name = cfg['base_model_name']
base_cfg = get_model_hparams(base_model_name)
base_model = load_model(model_name=base_model_name, cfg=base_cfg, set_only=False)
from x_transformer import AE_buttons_p_residual
model = AE_buttons_p_residual(base_model=base_model, cfg=cfg)

# Load residual adapter weights if checkpoint exists
if cfg.get('ckpt_file_name', ''):
    ckpt_path = cfg['ckpt_file_name']
    if os.path.exists(ckpt_path):
        map_loc = torch.device('cpu') if not torch.cuda.is_available() else None
        model.load_state_dict(torch.load(ckpt_path, map_location=map_loc), strict=False)
        print(f"Loaded adapter checkpoint: {ckpt_path}")
'''
model_name = 'AE_residual_v1' # 8 buttons
#model_name = 'no_dtime_joker_v1' # 8 buttons
cfg = get_model_hparams(model_name)
model = load_model(model_name=model_name, cfg=cfg )

model.to(device)
model.eval()

''' PARAMS '''

sample_midi_path = './samples/Bach_Prelude_and_Fugue_in G_minor.mid'
#sample_midi_path = './samples/Chopin_Nocturnes_Op9No1_In_B_Flat_Minor.mid'
sample_midi_path2 = './samples/clara.mid'
output_midi_name = './out/interactive_performance'

CTX_LEN = 128 # num notes in context. tokens = CTX_LENGTH * 3
TOTAL_GEN_LEN = 512 # num notes to generate

'''THREADING'''
buffer_lock = Lock()
save_lock = Lock()

'''VISUALIZER'''
visualizer = Visualizer(button_slots=cfg['num_buttons'])

# --- Chord state: tracks currently held chord notes ---
held_chord_notes = set()   # set of MIDI note numbers currently held in chord octave

def get_chord_pcs() -> List[float]:
    """Return 12-dim pitch-class multi-hot from currently held chord notes."""
    pcs = [0.0] * 12
    for note in held_chord_notes:
        pcs[note % 12] = 1.0
    return pcs

def get_bass_pc() -> int:
    """Return bass pitch class (lowest held chord note), 0 if none."""
    if len(held_chord_notes) == 0:
        return 0
    return min(held_chord_notes) % 12


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
    global pitch_buffer
    global dict_input_tokens
    global i

    context = {
        'dtime': dict_input_tokens['dtime'][:i+CTX_LEN+1],
        'pitch': pitch_buffer[:i+CTX_LEN+1],
        'dur': dict_input_tokens['dur'][:i+CTX_LEN+1],
    }

    song_d = dict_to_song(context)
    detailed_stats = ms_SONG_to_MIDI_Converter(song_d, output_file_name=output_midi_name,
                                                timings_multiplier=2)
    print("saved performance")

def reset_context(dict_input):
    global i
    global pitch_buffer, button_buffer
    global chord_pcs_buffer, bass_pc_buffer
    global first_note, kv_cache
    global held_chord_notes

    PRESERVE_LEN = 16
    current_end_idx = i + CTX_LEN
    preserved_pitch = []
    
    if i > 0:
        start_slice = max(0, current_end_idx - PRESERVE_LEN)
        preserved_pitch = pitch_buffer[start_slice:current_end_idx]

    i = 0
    first_note = True
    kv_cache = None
    held_chord_notes = set()

    pitch_buffer = dict_input['pitch'].copy()
    if len(preserved_pitch) > 0:
        splice_start = CTX_LEN - len(preserved_pitch)
        pitch_buffer[splice_start : CTX_LEN] = preserved_pitch

    if len(pitch_buffer) < TOTAL_GEN_LEN:
         pitch_buffer += [0] * (TOTAL_GEN_LEN - len(pitch_buffer))

    # Recalculate buttons from encoder
    current_pitch_tensor = torch.tensor(pitch_buffer[:TOTAL_GEN_LEN], dtype=torch.long).unsqueeze(0).to(device)
    new_buttons = model.gen_buttons({'pitch': current_pitch_tensor}).squeeze(0).cpu().tolist()
    button_buffer = new_buttons

    # Reset chord buffers (no active chord)
    chord_pcs_buffer = [[0.0]*12] * TOTAL_GEN_LEN
    bass_pc_buffer = [0] * TOTAL_GEN_LEN

    print("RESET_CONTEXT")


''' VARIABLES '''
context = None
timeLast = 0
i = 0
noteOn_dict = {}
first_note = True
kv_cache = None
last_gen_time: float = 0.0

''' BUILD CTX '''
dict_input_tokens, num_notes = midi_to_dict(sample_midi_path)
dict_input_tokens2, _ = midi_to_dict(sample_midi_path2)

# Derive initial buttons from encoder
original_pitch_tensor = torch.tensor(dict_input_tokens['pitch'], dtype=torch.long).unsqueeze(0).to(device)
original_buttons = model.gen_buttons({'pitch': original_pitch_tensor}).squeeze(0).cpu().tolist()

button_buffer = original_buttons.copy()
pitch_buffer = dict_input_tokens['pitch'].copy()

# Initialize chord buffers as unconditioned (no chord)
chord_pcs_buffer = [[0.0]*12 for _ in range(len(pitch_buffer))]
bass_pc_buffer = [0] * len(pitch_buffer)

# Ensure buffers are long enough for generation
if len(pitch_buffer) < TOTAL_GEN_LEN:
    pitch_buffer += [0] * (TOTAL_GEN_LEN - len(pitch_buffer))
if len(button_buffer) < TOTAL_GEN_LEN:
    button_buffer += [0] * (TOTAL_GEN_LEN - len(button_buffer))
if len(chord_pcs_buffer) < TOTAL_GEN_LEN:
    chord_pcs_buffer += [[0.0]*12] * (TOTAL_GEN_LEN - len(chord_pcs_buffer))
if len(bass_pc_buffer) < TOTAL_GEN_LEN:
    bass_pc_buffer += [0] * (TOTAL_GEN_LEN - len(bass_pc_buffer))

visualizer.primer(
    pitch_buffer[:CTX_LEN+1], 
    dict_input_tokens['dtime'][:CTX_LEN+1], 
    button_buffer[:CTX_LEN+1]
)


'''MIDI IN CALLBACK'''
def midiin_callback(event, data=None):
    message, deltatime = event
    status_type = message[0] & 0xF0
    note = message[1] if len(message) > 1 else 0
    velocity = message[2] if len(message) > 2 else 0

    if status_type == NOTE_ON and velocity > 0:
        with buffer_lock:
            if CHORD_OCTAVE_START <= note <= CHORD_OCTAVE_END:
                manage_chord_input(note, velocity)
            elif BUTTON_OCTAVE_START <= note <= BUTTON_OCTAVE_END:
                manage_button_input(note, velocity)

    elif status_type == NOTE_OFF or (status_type == NOTE_ON and velocity == 0):
        with buffer_lock:
            if CHORD_OCTAVE_START <= note <= CHORD_OCTAVE_END:
                manage_chord_input(note, 0)
            elif BUTTON_OCTAVE_START <= note <= BUTTON_OCTAVE_END:
                manage_button_input(note, 0)

    if status_type == 0xB0:  # Control Change
        if message[1] == 18 and message[2] > 0:  # REC → save
            with save_lock:
                print("saving performance")
                save_performance()
                sys.exit(0)
        if message[1] == 17 and message[2] > 0:  # PLAY → reset
            with save_lock:
                print("resetting context")
                reset_context(dict_input_tokens2)


def manage_chord_input(note, velocity):
    """Add/remove chord notes. No note generation happens here."""
    global held_chord_notes

    if velocity > 0:
        held_chord_notes.add(note)
    else:
        held_chord_notes.discard(note)

    if TRACES:
        pcs = get_chord_pcs()
        active = [i for i, v in enumerate(pcs) if v > 0]
        print(f"Chord: {active}  bass_pc={get_bass_pc()}")


def manage_button_input(note, velocity):
    """Map MIDI note to button, generate a note conditioned on the active chord."""
    global context, timeLast, i
    global pitch_buffer, button_buffer
    global chord_pcs_buffer, bass_pc_buffer
    global noteOn_dict, first_note
    global kv_cache, last_gen_time

    but = TO_BUTTON[note - BUTTON_OCTAVE_START]
    if TRACES:
        print("button", but, "from note", note)

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

        # Store dtime for saving
        if i + CTX_LEN < len(dict_input_tokens['dtime']):
            dict_input_tokens['dtime'][i + CTX_LEN] = dtime
        else:
            dict_input_tokens['dtime'].append(dtime)

        gen_pos = i + CTX_LEN

        # Update buffers at generation position
        if gen_pos < len(button_buffer):
            button_buffer[gen_pos] = but
        else:
            button_buffer.append(but)

        # Snapshot the current chord state into the buffer
        current_pcs = get_chord_pcs()
        current_bass = get_bass_pc()
        if gen_pos < len(chord_pcs_buffer):
            chord_pcs_buffer[gen_pos] = current_pcs
        else:
            chord_pcs_buffer.append(current_pcs)
        if gen_pos < len(bass_pc_buffer):
            bass_pc_buffer[gen_pos] = current_bass
        else:
            bass_pc_buffer.append(current_bass)

        # Build context window
        sl = slice(i, i + CTX_LEN + 1)
        context = {
            'pitch':     torch.tensor(pitch_buffer[sl], dtype=torch.long).unsqueeze(0),
            'button':    torch.tensor(button_buffer[sl], dtype=torch.long).unsqueeze(0),
            'chord_pcs': torch.tensor(chord_pcs_buffer[sl], dtype=torch.float).unsqueeze(0),
            'bass_pc':   torch.tensor(bass_pc_buffer[sl], dtype=torch.long).unsqueeze(0),
        }
        context = to_device(context, device)

        if TRACES:
            pcs_now = context['chord_pcs'][0, -1]
            active = [j for j in range(12) if pcs_now[j] > 0]
            print(f"  gen chord_pcs={active}  bass={context['bass_pc'][0,-1].item()}")

        with torch.inference_mode():
            new_pitch_token = model.gen_pitch_token(context)
        last_gen_time = time.perf_counter()

        if gen_pos < len(pitch_buffer):
            pitch_buffer[gen_pos] = new_pitch_token
        else:
            pitch_buffer.append(new_pitch_token)

        playNote(new_pitch_token, velocity)
        visualizer.get_note(new_pitch_token, velocity)
        visualizer.get_button(but, velocity)

        noteOn_dict[note] = (new_pitch_token, timeNew, but)
        i += 1

    else:  # noteOff
        if note in noteOn_dict:
            pitch, noteOn_time, but = noteOn_dict[note]
            del noteOn_dict[note]
            playNote(pitch, 0)
            visualizer.get_note(pitch, 0)
            visualizer.get_button(but, 0)


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
        for idx, port in enumerate(available_ports):
            print(f"[{idx}] {port}")
        midiin.open_port(MIDI_PORT) 
        print(f"Using MIDI input port: {available_ports[MIDI_PORT]}")

    midiin.set_callback(midiin_callback)

    print("=" * 50)
    print(f"Button octave: MIDI {BUTTON_OCTAVE_START}-{BUTTON_OCTAVE_END} (C3-B3)")
    print(f"Chord octave:  MIDI {CHORD_OCTAVE_START}-{CHORD_OCTAVE_END} (C4-B4)")
    print("Hold chord notes to condition, press button keys to generate.")
    print("CC18 = Save, CC17 = Reset context")
    print("=" * 50)

    while True:
        time.sleep(0.0001)
        visualizer.draw()
except (EOFError, KeyboardInterrupt, SystemExit):
    print("Bye.")
