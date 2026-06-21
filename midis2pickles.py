#===================================================================================================
# Monster Genie midis2pickles.py Python module
# Converts MIDI files into a pickle file
# 
# Copyright 2025 Alex Barrachina
#
# Based on Project Los Angeles / Tegridy Code 2025
# https://github.com/asigalov61/monsterpianotransformer
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


import random
import os
from tqdm import tqdm

from midiUtils import midi2ms_score, Any_Pickle_File_Writer
from params import *

# NO OFFSETS: Each token type is stored in its raw range (0-127)
# The pickle format stores 5 tokens per event: [dtime, dur, pitch, vel, chan]
# All values are in range 0-127 (or 0-15 for channel)
#
# HARMONY MOVEMENT EVENTS (from MIDI channel 4, stored as channel 3 in pickle):
#   Stored as [0, 0, movement_type, 0, 3] following the [dtime, dur, pitch, vel, chan] layout.
#   Harmony events do NOT update pe, so subsequent notes compute dtime from the last playable note.
#
# CHORD REFERENCE EVENTS (from MIDI channel 5, stored as channel 4 in pickle):
#   Stored as [0, dur, pitch, vel, 4] following the normal layout but with dtime=0.
#   Chord events do NOT update pe, so subsequent notes compute dtime from the last playable note.
#   Multiple simultaneous chord notes form one chord group (consecutive chan=4 events).
#   When all chord notes expire (abs_time >= onset + max_dur), a chord-off marker
#   [0, 0, 0, 0, 4] (vel=0) is inserted so the dataset knows the chord is no longer active.


melody_only = False # if True, only process melody notes (channel 0)
sorted_or_random_file_loading_order = False # Sorted order is NOT usually recommended
dataset_ratio = 1 # Change this if you need more or less % of the dataset

#train_and_test_ratio = 1. # 100% for training
train_and_test_ratio = 0.8 # 80% for training, 20% for testing


MELODY_ONLY = 0 # melody only (harmony not used for dtime)
ACCOMP_ONLY = 1 # accompaniment only (harmony not used for dtime)
MELODY_AND_ACCOMP = 2 # melody and accompaniment (harmony not used for dtime)
MELODY_AND_ACCOMP_NOT_HARMONY = 3 # melody, accompaniment. Not harmony no chords
ALL_CHANNELS = 4 # we will use all channels in the pickle (harmony not used for dtime)  

useful_channels = MELODY_AND_ACCOMP 

###########

files_count = 0

gfiles = []

train_data1 = []
test_data1 = []

# Channel statistics
total_notes = 0
channel_0_notes = 0
channel_10_notes = 0
channel_4_movements = 0
channel_5_chords = 0
chord_label_events = 0   # one packed (root, quality, function) event per chord
key_events = 0           # one (key_pc, mode) event per key change
chords_with_label = 0    # chords whose factors came from a parsed text label
chords_derived = 0       # chords whose factors were derived from the notes (no label)

###########

# dataset_addr = "./Samples"  # when testing
dataset_addr = "../../../DataSets/MIDI/giantMIDI/all_harmony_labels"
# Output file names
output_name = 'giantmidi_full_harmony_labels'

# Process MIDIs

# first quantize the time and duration, then calculate the time difference and maximum duration
def time2quant(time):
    return int(time/10)

def dur2quant(dur):
    return int(dur/20)


def parse_harmony_label(text):
    """Parse an analyzer harmony-label string into a tuple
    (root_pc, quality_id, key_pc, mode, function_id).

    Expected format (German functional analysis), e.g.:
        'C# DOMINANT_SEVENTH:SEPT(7):F# MOLL:D (V)'
        'C# DIMINISHED_MINOR_SEVENTH:TERZQUART(3,4):null:null:'
    Layout: '<root> <QUALITY>:<figbass>(...):<keyroot> <KEYMODE>:<FUNC> (<roman>)'
    Trailing key/function fields may be 'null'. The figured-bass field is ignored
    (inversion is recoverable from the chord-tone bass). Returns None if the text
    does not look like a harmony label.
    """
    if not text:
        return None
    text = text.strip()
    if ':' not in text:
        return None
    parts = text.split(':')

    # parts[0] = '<root> <QUALITY>'
    head = parts[0].strip().split(None, 1)
    if len(head) < 2:
        return None
    root_tok, quality_tok = head[0], head[1].strip().upper()
    root_pc = note_name_to_pc(root_tok)
    if root_pc == PC_UNKNOWN:
        return None  # not a real chord label
    quality_id = QUALITY_NAME_MAP.get(quality_tok, QUALITY_OTHER)

    # parts[2] = '<keyroot> <KEYMODE>' or 'null'
    key_pc, mode = PC_UNKNOWN, MODE_UNKNOWN
    if len(parts) >= 3:
        kf = parts[2].strip()
        if kf and kf.lower() != 'null':
            ktoks = kf.split(None, 1)
            key_pc = note_name_to_pc(ktoks[0])
            if len(ktoks) > 1:
                km = ktoks[1].strip().upper()
                if km.startswith('MOLL'):
                    mode = MODE_MINOR
                elif km.startswith('DUR'):
                    mode = MODE_MAJOR

    # parts[3] = '<FUNC> (<roman>)' or 'null'
    function_id = FUNC_UNKNOWN
    if len(parts) >= 4:
        ff = parts[3].strip()
        if ff and ff.lower() != 'null':
            fsym = ff.split('(')[0].strip()
            if fsym:
                function_id = FUNCTION_NAME_MAP.get(
                    fsym, FUNCTION_NAME_MAP.get(fsym[:1], FUNC_UNKNOWN)
                )

    return (root_pc, quality_id, key_pc, mode, function_id)


def identify_chord(pitch_classes):
    """Notes-derived fallback when no label string is present.
    Returns (root_pc, quality_id) from a set of pitch classes via interval
    templates. Falls back to (lowest pc, QUALITY_UNKNOWN) when nothing matches."""
    pcs = sorted({int(p) % 12 for p in pitch_classes})
    if not pcs:
        return PC_UNKNOWN, QUALITY_UNKNOWN
    for root in pcs:
        intervals = frozenset((p - root) % 12 for p in pcs)
        q = QUALITY_TEMPLATES.get(intervals)
        if q is not None:
            return root, q
    return pcs[0], QUALITY_UNKNOWN


filez = list()
for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
    filez += [os.path.join(dirpath, file) for file in filenames]
print('=' * 70)

if filez == []:
    print('Could not find any MIDI files. Please check Dataset dir...')
    print('=' * 70)

if sorted_or_random_file_loading_order:
    print('Sorting files...')
    filez.sort()
    print('Done!')
    print('=' * 70)
else:
    print('Randomizing file list...')
    random.shuffle(filez)


print('Processing MIDI files. Please wait...')
for f in tqdm(filez[:int(len(filez) * dataset_ratio)]):
    try:
        fn = os.path.basename(f)
        fn1 = fn.split('.')[0]

        #print('Loading MIDI file...')
        score = midi2ms_score(open(f, 'rb').read())

        events_matrix = []
        label_events_ms = []  # (time_ms, parsed_label) from text/marker meta-events

        itrack = 1

        while itrack < len(score):
            for event in score[itrack]:         
                if event[0] == 'note' and event[3] != 9: # skip percussion notes
                    events_matrix.append(event)
                elif event[0] in ('marker', 'text_event', 'lyric', 'cue_point'):
                    # Chord-name + harmonic-function labels are carried as text meta-events.
                    raw = event[2]
                    try:
                        s = raw.decode('latin-1', 'ignore') if isinstance(raw, (bytes, bytearray)) else str(raw)
                    except Exception:
                        s = ''
                    parsed = parse_harmony_label(s)
                    if parsed is not None:
                        label_events_ms.append((event[1], parsed))
            itrack += 1
        
        if len(events_matrix) > 0:

          # Sorting by pitch (descending) then by time (ascending). when the second sort reorders by time, 
          # notes with the same start time (i.e., chords) retain their relative order from the first sort (by pitch descending).
          events_matrix.sort(key=lambda x: x[4], reverse=True) # pitch
          events_matrix.sort(key=lambda x: x[1]) # time

          # Include channel 4 (harmony movements) and channel 5 (chord references) in the event stream
          if useful_channels == MELODY_ONLY:
            filtered_events_matrix = [e for e in events_matrix if e[3] in (MELODY_CHANNEL, HARMONY_CHANNEL, CHORDS_CHANNEL)]
          elif useful_channels == ACCOMP_ONLY:
            filtered_events_matrix = [e for e in events_matrix if e[3] in (ACCOMP_CHANNEL, HARMONY_CHANNEL, CHORDS_CHANNEL)]
          elif useful_channels == MELODY_AND_ACCOMP:
            filtered_events_matrix = [e for e in events_matrix if e[3] in (MELODY_CHANNEL, ACCOMP_CHANNEL, HARMONY_CHANNEL, CHORDS_CHANNEL)]
          elif useful_channels == MELODY_AND_ACCOMP_NOT_HARMONY:
            filtered_events_matrix = [e for e in events_matrix if e[3] in (MELODY_CHANNEL, ACCOMP_CHANNEL)]
          else:
            filtered_events_matrix = events_matrix

          # Skip files with no playable notes
          if not any(e[3] not in (HARMONY_CHANNEL, CHORDS_CHANNEL) for e in filtered_events_matrix):
              continue

          # Quantize timings for all events
          for e in filtered_events_matrix:
              e[1] = time2quant(e[1])
              e[2] = dur2quant(e[2])

          # Determine if this file goes to train or test
          is_train = random.random() < train_and_test_ratio
          target_data = train_data1 if is_train else test_data1

          # Intro/Zero seq (5 tokens) - no offsets, raw values
          target_data.extend([126, 126, 0, 0, 0])  # dtime, dur, pitch, vel, chan

          # pe tracks the last PLAYABLE note for dtime calculation
          # Harmony events (channel 3) and chord events (channel 4) do NOT update pe
          pe = next(e for e in filtered_events_matrix if e[3] not in (HARMONY_CHANNEL, CHORDS_CHANNEL))

          # Track when the current chord expires (abs time, quantized)
          chord_end_abs_time = -1

          # --- Harmony-label lookups (quantized-time indexed) ---
          # Text labels: quantized onset time -> (root_pc, quality_id, key_pc, mode, function_id)
          label_by_qtime = {}
          for t_ms, parsed in label_events_ms:
              label_by_qtime[time2quant(t_ms)] = parsed
          # Chord-tone pitch classes per onset, for the notes-derived fallback
          chord_group_pcs = {}
          for e in filtered_events_matrix:
              if e[3] == CHORDS_CHANNEL:
                  chord_group_pcs.setdefault(e[1], set()).add(e[4] % 12)

          current_key = (PC_UNKNOWN, MODE_UNKNOWN)
          last_chord_label_qtime = None

          for e in filtered_events_matrix:

              if e[3] == HARMONY_CHANNEL:
                  # Harmony movement: [0, 0, movement_type, 0, chan=3], pe NOT updated
                  movement_type = max(0, min(126, e[4]))
                  target_data.extend([0, 0, movement_type, 0, HARMONY_CHANNEL])
                  channel_4_movements += 1
              elif e[3] == CHORDS_CHANNEL:
                  # On the FIRST tone of each chord group, emit compact harmony
                  # conditioning: a key event (only on change) + a chord-label event.
                  onset_q = e[1]
                  if onset_q != last_chord_label_qtime:
                      last_chord_label_qtime = onset_q
                      lbl = label_by_qtime.get(onset_q)
                      if lbl is None:
                          lbl = label_by_qtime.get(onset_q - 1) or label_by_qtime.get(onset_q + 1)
                      if lbl is not None:
                          root_pc, quality_id, key_pc, mode, function_id = lbl
                          chords_with_label += 1
                      else:
                          root_pc, quality_id = identify_chord(chord_group_pcs.get(onset_q, set()))
                          key_pc, mode, function_id = PC_UNKNOWN, MODE_UNKNOWN, FUNC_UNKNOWN
                          chords_derived += 1
                      # Key event only when the key actually changes (very sparse)
                      if key_pc != PC_UNKNOWN and (key_pc, mode) != current_key:
                          target_data.extend([0, key_pc, mode, 0, KEY_CHANNEL])
                          current_key = (key_pc, mode)
                          key_events += 1
                      # Chord-label event: [0, root_pc, quality_id, function_id, 120], pe NOT updated
                      target_data.extend([0, root_pc, quality_id, function_id, CHORD_LABEL_CHANNEL])
                      chord_label_events += 1

                  # Chord reference: [0, dur, pitch, vel, chan=4], pe NOT updated
                  dur = max(1, min(126, e[2]))
                  ptc = max(1, min(126, e[4]))
                  vel = max(1, min(126, e[5]))
                  target_data.extend([0, dur, ptc, vel, CHORDS_CHANNEL])
                  channel_5_chords += 1
                  # Track chord expiry: onset + duration (whichever note lasts longest)
                  this_end = e[1] + e[2]
                  if this_end > chord_end_abs_time:
                      chord_end_abs_time = this_end
              else:
                  # Before writing the playable note, check if the chord has expired
                  if chord_end_abs_time >= 0 and e[1] >= chord_end_abs_time:
                      # Insert chord-off marker: [0, 0, 0, 0, CHORDS_CHANNEL]
                      # vel=0 distinguishes it from real chord notes in the dataset
                      target_data.extend([0, 0, 0, 0, CHORDS_CHANNEL])
                      chord_end_abs_time = -1

                  # Playable note: compute dtime from last playable note
                  time = max(0, min(126, e[1]-pe[1]))
                  dur = max(1, min(126, e[2]))
                  chan = max(0, min(15, e[3]))
                  ptc = max(1, min(126, e[4]))
                  vel = max(1, min(126, e[5]))

                  target_data.extend([time, dur, ptc, vel, chan])

                  total_notes += 1
                  if e[3] == MELODY_CHANNEL:
                      channel_0_notes += 1
                  elif e[3] == ACCOMP_CHANNEL:
                      channel_10_notes += 1

                  pe = e

          files_count += 1
        
    except KeyboardInterrupt:
        print('Quitting...')
        break  

    except Exception as ex:
        print(f'Bad MIDI: {f} - {ex}')
        continue

# Save training data
output_path = './Training-Data/' + output_name + '_train'
print('Saving training data...')
Any_Pickle_File_Writer(train_data1, output_path)        
print(f'Training data saved to {output_path}.pickle')   
print(f'{len(train_data1)} tokens ({len(train_data1)//5} notes)')
print('=' * 70)

# Save test data
if train_and_test_ratio < 1.:
    print('Saving test data...')
    output_path = './Training-Data/' + output_name + '_test'
    Any_Pickle_File_Writer(test_data1, output_path)
    print(f'Test data saved to {output_path}.pickle')
    print(f'{len(test_data1)} tokens ({len(test_data1)//5} notes)')
    print('=' * 70)

# Display channel statistics
print('Channel Statistics:')
print(f'Total notes processed: {total_notes}')
if total_notes > 0:
    channel_0_pct = (channel_0_notes / total_notes) * 100
    channel_10_pct = (channel_10_notes / total_notes) * 100
    print(f'Channel 0 (melody) notes: {channel_0_notes} ({channel_0_pct:.2f}%)')
    print(f'Channel 10 (accompaniment) notes: {channel_10_notes} ({channel_10_pct:.2f}%)')
print(f'Channel 4 harmony movement events inserted: {channel_4_movements}')
print(f'Channel 5 chord reference events inserted: {channel_5_chords}')
print(f'Chord-label events inserted: {chord_label_events} '
      f'(from text label: {chords_with_label}, notes-derived: {chords_derived})')
print(f'Key-change events inserted: {key_events}')
if total_notes > 0:
    print(f'Movement events per note ratio: {channel_4_movements / total_notes:.4f}')
    print(f'Chord events per note ratio: {channel_5_chords / total_notes:.4f}')
    print(f'Chord-label events per note ratio: {chord_label_events / total_notes:.4f}')
print('=' * 70)

print('Done!')
