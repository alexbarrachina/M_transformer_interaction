#===================================================================================================
# Monster Genie automatic_inference_harm.py Python module
# Automatic inference for the dual-conditioned model (buttons + harmony movements).
# Loads a pickle containing note + harmony events, extracts oracle buttons from the encoder,
# and generates 3 versions:
#   1/ Zero harmony guidance (harm_strength=0 everywhere)
#   2/ Oracle harmony (original movements from the data)
#   3/ Random harmony (random movement types at the same onset positions)
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

from params import *
from model_loader import load_model
from models import get_model_hparams
from midiUtils import Any_Pickle_File_Reader, dict_to_song, ms_SONG_to_MIDI_Converter, to_device

# Harmony channel marker in the pickle (MIDI channel 4 = stored as channel 3)
HARMONY_CHANNEL = 3

temperature = 0.0001

''' DEVICE '''
device = torch.device('mps') 

''' MODEL '''
model_name = 'AE_dual_tester_v1'
cfg = get_model_hparams(model_name)
model = load_model(model_name=model_name, cfg=cfg)
model.to(device)
model.eval()

''' PARAMS '''
pickle_path = './Training-Data/giantmidi_full_harmony_test'
output_base = './out/harm_inference'
CTX_LEN = 450
# Use a subset of notes from the pickle for testing
MAX_NOTES = 2000


def parse_pickle_events(data_tensor):
    """Parse flat pickle data into note events + harmony regime/strength tensors."""
    num_events = len(data_tensor) // 5
    events = data_tensor[:num_events * 5].view(num_events, 5).long()

    is_harmony = (events[:, 4] == HARMONY_CHANNEL)
    is_note = ~is_harmony
    note_indices = torch.where(is_note)[0]

    if len(note_indices) == 0:
        return None

    note_events = events[note_indices]
    pitches = note_events[:, 2]
    dtimes = note_events[:, 0]
    durs = note_events[:, 1]

    # --- Vectorized harm_regime + harm_strength (same logic as train_harm.py) ---
    harm_cumsum = torch.cumsum(is_harmony.long(), dim=0)
    harm_event_indices = torch.where(is_harmony)[0]
    num_harm = harm_event_indices.shape[0]

    move_lookup = torch.zeros(num_harm + 1, dtype=torch.long)
    if num_harm > 0:
        move_lookup[1:] = torch.clamp(events[harm_event_indices, 2] - 60, min=0, max=7)

    harm_group_notes = harm_cumsum[note_indices]
    harm_regime_notes = move_lookup[harm_group_notes]

    group_changes = torch.cat([
        torch.tensor([True]),
        harm_group_notes[1:] != harm_group_notes[:-1]
    ])
    group_start_indices = torch.where(group_changes)[0]
    note_arange = torch.arange(len(harm_group_notes))
    group_of_note = torch.searchsorted(group_start_indices, note_arange, side='right') - 1
    position_in_group = note_arange - group_start_indices[group_of_note]

    span_lengths = torch.diff(group_start_indices, append=torch.tensor([len(harm_group_notes)]))
    span_length_per_note = span_lengths[group_of_note]

    # Linear decay from 1.0 at onset to 0.0 at next onset
    harm_strength_notes = 1.0 - position_in_group.float() / span_length_per_note.float()

    unguided = (harm_group_notes == 0)
    harm_strength_notes = harm_strength_notes.masked_fill(unguided, 0.0)

    return {
        'pitches': pitches,
        'dtimes': dtimes,
        'durs': durs,
        'harm_regime': harm_regime_notes,
        'harm_strength': harm_strength_notes,
    }


def build_random_harmony(harm_regime, harm_strength):
    """Randomize movement types at onset positions, keep the same strength pattern."""
    regime_random = harm_regime.clone()
    onsets = (harm_strength > 0.95)
    if not onsets.any():
        return regime_random

    harm_group = torch.cumsum(onsets.long(), dim=0)
    num_groups = int(harm_group.max().item())
    if num_groups == 0:
        return regime_random

    onset_indices = torch.where(onsets)[0]
    random_types = torch.randint(0, 8, (len(onset_indices),))

    # Build lookup and broadcast (same cumsum approach)
    group_type_lookup = torch.zeros(num_groups + 1, dtype=torch.long)
    group_type_lookup[1:] = random_types
    regime_random = group_type_lookup[harm_group]

    return regime_random


def generate_sequence(model, pitches, buttons, harm_regime, harm_strength,
                      ctx_len, device, temperature):
    """Generate pitch sequence using oracle buttons and specified harmony conditioning."""
    num_notes = len(pitches)
    output_pitches = pitches.clone()

    for i in range(0, num_notes - 1 - ctx_len):
        sl = slice(i, i + ctx_len + 1)
        context = {
            'pitch': pitches[sl].unsqueeze(0).to(device),
            'button': buttons[sl].unsqueeze(0).to(device),
            'harm_regime': harm_regime[sl].unsqueeze(0).to(device),
            'harm_strength': harm_strength[sl].unsqueeze(0).float().to(device),
        }

        with torch.inference_mode():
            new_pitch = model.gen_pitch_token(context, temperature=temperature)
        output_pitches[i] = new_pitch

    return output_pitches


def save_midi(pitches, dtimes, durs, num_notes, ctx_len, filename):
    """Save generated pitches with original timing as MIDI."""
    length = num_notes - ctx_len
    context = {
        'dtime': dtimes[:length].tolist(),
        'pitch': pitches[:length].tolist(),
        'dur': durs[:length].tolist(),
    }
    song_d = dict_to_song(context)
    ms_SONG_to_MIDI_Converter(song_d, output_file_name=filename, timings_multiplier=2)
    print(f"  Saved: {filename}")


''' MAIN '''
print("Loading pickle data...")
raw_data = Any_Pickle_File_Reader(pickle_path)
data_tensor = torch.Tensor(raw_data)

parsed = parse_pickle_events(data_tensor)
if parsed is None:
    print("No note events found in pickle.")
    exit()

pitches = parsed['pitches'][:MAX_NOTES]
dtimes = parsed['dtimes'][:MAX_NOTES]
durs = parsed['durs'][:MAX_NOTES]
harm_regime = parsed['harm_regime'][:MAX_NOTES]
harm_strength = parsed['harm_strength'][:MAX_NOTES]
num_notes = len(pitches)

print(f"Total notes: {num_notes}")
print(f"Harmony onsets: {(harm_strength > 0.95).sum().item()}")
print(f"Guided notes (strength > 0): {(harm_strength > 0).sum().item()}")

# Extract oracle buttons from encoder
print("Extracting oracle buttons from encoder...")
with torch.inference_mode():
    pitch_tensor = pitches.unsqueeze(0).to(device)
    buttons = model.gen_buttons({'pitch': pitch_tensor}).squeeze(0).cpu()

print(f"Buttons range: [{buttons.min().item()}, {buttons.max().item()}]")

# --- Condition 1: Zero harmony guidance ---
print("\n[1/3] Generating with ZERO harmony guidance...")
harm_regime_zero = torch.zeros_like(harm_regime)
harm_strength_zero = torch.zeros_like(harm_strength)

out_1 = generate_sequence(
    model, pitches, buttons, harm_regime_zero, harm_strength_zero,
    CTX_LEN, device, temperature
)
save_midi(out_1, dtimes, durs, num_notes, CTX_LEN, output_base + '_1_no_harmony')

# --- Condition 2: Oracle harmony (original movements) ---
print("\n[2/3] Generating with ORACLE harmony guidance...")
out_2 = generate_sequence(
    model, pitches, buttons, harm_regime, harm_strength,
    CTX_LEN, device, temperature
)
save_midi(out_2, dtimes, durs, num_notes, CTX_LEN, output_base + '_2_oracle_harmony')

# --- Condition 3: Random harmony (random movement types at original onsets) ---
print("\n[3/3] Generating with RANDOM harmony guidance...")
harm_regime_random = build_random_harmony(harm_regime, harm_strength)
out_3 = generate_sequence(
    model, pitches, buttons, harm_regime_random, harm_strength,
    CTX_LEN, device, temperature
)
save_midi(out_3, dtimes, durs, num_notes, CTX_LEN, output_base + '_3_random_harmony')

# --- Also save oracle buttons as MIDI for visual inspection ---
buttons_as_pitch = torch.add(buttons[:num_notes - CTX_LEN], 60)
context_buttons = {
    'dtime': dtimes[:num_notes - CTX_LEN].tolist(),
    'pitch': buttons_as_pitch.tolist(),
    'dur': durs[:num_notes - CTX_LEN].tolist(),
}
song_d = dict_to_song(context_buttons)
ms_SONG_to_MIDI_Converter(song_d, output_file_name=output_base + '_buttons', timings_multiplier=2)

print("\nDone. Compare the 3 MIDI files to assess harmony conditioning effect.")
