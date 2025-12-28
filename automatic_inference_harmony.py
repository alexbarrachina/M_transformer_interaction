#==================================================================================================
# Monster Genie automatic_inference_harmony.py Python module
# Automatic inference, from a MIDI file as context,
# guided with harmony features (Tonnetz x, y, r) extracted from the same MIDI file
# By default, the context len is fixed to 128 notes. Once reached 128 notes, the first ones are discarded.
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
import torch
import numpy as np
from typing import List, Tuple, Optional

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter
from harmony_extractor import extract_harmony_for_tokens

temperature = 1.0

''' DEVICE '''
#device = torch.device('cpu')
device = torch.device('mps') 


''' MODEL '''
model_name = 'autoenc_no_dtime_harmony_v1'
cfg = get_model_hparams(model_name)
model = load_model(model_name=model_name, cfg=cfg)
model.to(device)
model.eval()

#print(model)

''' PARAMS '''
# Get sample seed MIDI path
sample_midi_path = './samples/clairTester_to_end.midi'
output_midi_name = './out/continuator_harmony_clairTester'
output_butt_midi_name = './out/continuator_harmony_clairTester_buttons'
CTX_LEN = 128  # num notes in context (shorter due to harmony model)
HARMONY_INTERVAL_MS = 500  # harmony sampling interval in ms (from harmony_extractor)


def extract_note_timings_ms(dict_tokens: dict, timing_multiplier: int = 32) -> List[int]:
    """
    Extract absolute note onset times in milliseconds from dict tokens.
    
    Args:
        dict_tokens: Dictionary with 'dtime' key containing delta times
        timing_multiplier: Multiplier to convert quantized time to ms
    
    Returns:
        List of absolute onset times in ms for each note
    """
    dtimes: List[int] = dict_tokens['dtime']
    abs_times: List[int] = []
    current_time: int = 0
    
    for dtime in dtimes:
        current_time += dtime * timing_multiplier
        abs_times.append(current_time)
    
    return abs_times


def interpolate_harmony_to_notes(
    harmony_events: List[Tuple[int, int, int, int]],
    note_times_ms: List[int]
) -> Tuple[List[int], List[int], List[int]]:
    """
    Interpolate harmony events (sampled at regular intervals) to note positions.
    
    For each note, finds the closest harmony event by time and assigns its values.
    
    Args:
        harmony_events: List of (time_ms, x_bin, y_bin, r_bin) tuples
        note_times_ms: List of absolute note onset times in ms
    
    Returns:
        Tuple of (harm_x_list, harm_y_list, harm_r_list) for each note
    """
    harm_x_list: List[int] = []
    harm_y_list: List[int] = []
    harm_r_list: List[int] = []
    
    if not harmony_events:
        # Default to center values if no harmony events
        default_val = 64
        for _ in note_times_ms:
            harm_x_list.append(default_val)
            harm_y_list.append(default_val)
            harm_r_list.append(default_val)
        return harm_x_list, harm_y_list, harm_r_list
    
    # Sort harmony events by time
    sorted_events = sorted(harmony_events, key=lambda x: x[0])
    
    for note_time in note_times_ms:
        # Find closest harmony event
        best_idx = 0
        best_dist = abs(sorted_events[0][0] - note_time)
        
        for i, event in enumerate(sorted_events):
            dist = abs(event[0] - note_time)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        # Assign harmony values from closest event
        _, x_bin, y_bin, r_bin = sorted_events[best_idx]
        harm_x_list.append(x_bin)
        harm_y_list.append(y_bin)
        harm_r_list.append(r_bin)
    
    return harm_x_list, harm_y_list, harm_r_list


''' BUILD CTX '''
# Load seed MIDI
dict_input_tokens, num_notes = midi_to_dict(sample_midi_path)

dict_output_tokens = dict_input_tokens.copy()

print("num_notes", num_notes)

# Extract harmony features from the MIDI file
print("Extracting harmony features...")
harmony_events = extract_harmony_for_tokens(
    sample_midi_path,
    harmony_interval_ms=HARMONY_INTERVAL_MS,
    n_bins=128
)

if harmony_events is None:
    print("Error: Could not extract harmony features from MIDI")
    exit(1)

print(f"Extracted {len(harmony_events)} harmony events")

# Get note timings and interpolate harmony to notes
note_times_ms = extract_note_timings_ms(dict_input_tokens)
harm_x_list, harm_y_list, harm_r_list = interpolate_harmony_to_notes(harmony_events, note_times_ms)

print(f"Interpolated harmony features to {len(harm_x_list)} notes")

''' GENERATE 10 files'''

for j in range(0, 10):  # generate 10 continuation files
    # Build context tokens for encoder (to get buttons)
    context_encoder = {
        'pitch': torch.tensor(dict_input_tokens['pitch'], dtype=torch.long).unsqueeze(0),
    }
    context_encoder = to_device(context_encoder, device)
    
    with torch.inference_mode():
        e = model.encoder(context_encoder)  # encoder output (batch, seq_len)
        b = model.real_to_discrete(e).squeeze(0)  # generate buttons (batch, seq_len)
        e = e.squeeze(0)
    
    # Copy original pitches for output (will be replaced by generated ones)
    pitch_buffer = dict_input_tokens['pitch'].copy()
    
    # Build output sequence
    dict_output_tokens_gen = {
        'dtime': [],
        'pitch': [],
        'dur': []
    }
    
    print(f"Generating continuation file {j}")
    timeStart = time.perf_counter()
    
    # Generate pitches autoregressively
    for i in range(0, num_notes - 1 - CTX_LEN):
        # Build context for decoder
        # Context pitch: use GENERATED pitches from rolling buffer (autoregressive)
        # Context button: from encoder
        # Context harmony: from ORIGINAL melody (extracted features)
        
        # Harmony features: use original extracted features for guidance
        # Note: harm features are shifted by 1 relative to pitch (they guide the NEXT pitch)        
        decoder_context = {
            'pitch': torch.tensor(pitch_buffer[i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
            'button': torch.tensor(b[i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
            'harm_x': torch.tensor(harm_x_list[i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
            'harm_y': torch.tensor(harm_y_list[i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
            'harm_r': torch.tensor(harm_r_list[i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
        }
        
        decoder_context = to_device(decoder_context, device)
  
        with torch.inference_mode():
            new_pitch_token = model.gen_pitch_token(decoder_context, temperature=temperature)
           
        # Update rolling buffer (so next iteration uses this generated pitch in context)
        pitch_buffer[i + CTX_LEN+1] = new_pitch_token
        
        # Build output sequence
        dict_output_tokens_gen['pitch'].append(new_pitch_token)
        dict_output_tokens_gen['dtime'].append(dict_input_tokens['dtime'][i + CTX_LEN+1])
        dict_output_tokens_gen['dur'].append(dict_input_tokens['dur'][i + CTX_LEN+1])
        # print(new_pitch_token)
    
    timeEnd = time.perf_counter()
    num_generated = len(dict_output_tokens_gen['pitch'])
    if num_generated > 0:
        print(f"  Generation time: {(timeEnd - timeStart) * 1000 / num_generated:.2f} ms/note")
    
    # Output context
    context_out = {
        'dtime': dict_output_tokens_gen['dtime'],
        'pitch': dict_output_tokens_gen['pitch'],
        'dur': dict_output_tokens_gen['dur'],
    }
    
    # Generate a midi file from generated pitches
    song_d = dict_to_song(context_out)
    detailed_stats = ms_SONG_to_MIDI_Converter(
        song_d,
        output_file_name=output_midi_name + str(j),
        timings_multiplier=2
    )
    
    # Convert buttons to values similar to pitch, just to represent the melodic contour
    b_shifted = torch.add(b, 60)
    
    button_context = {
        'dtime': dict_output_tokens_gen['dtime'],
        'pitch': b_shifted[CTX_LEN:CTX_LEN + num_generated].tolist(),
        'dur': dict_output_tokens_gen['dur'],
    }
    song_d = dict_to_song(button_context)
    
    detailed_stats = ms_SONG_to_MIDI_Converter(
        song_d,
        output_file_name=output_butt_midi_name + str(j),
        timings_multiplier=2
    )
    
    # Calculate accuracy: compare generated pitches with original
    original_pitches = dict_input_tokens['pitch'][CTX_LEN:CTX_LEN + num_generated]
    generated_pitches = dict_output_tokens_gen['pitch']
    
    if num_generated > 0:
        matches = sum(1 for o, g in zip(original_pitches, generated_pitches) if o == g)
        accuracy = matches / num_generated * 100
        print(f"  Pitch accuracy: {accuracy:.2f}% ({matches}/{num_generated})")
        
        # Also compute pitch proximity (within semitone)
        close_matches = sum(1 for o, g in zip(original_pitches, generated_pitches) if abs(o - g) <= 1)
        proximity = close_matches / num_generated * 100
        print(f"  Pitch proximity (±1 semitone): {proximity:.2f}%")

print("\nDone!")
