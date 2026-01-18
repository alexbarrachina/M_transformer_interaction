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
from harmony_extractor import extract_harmony_for_tokens

# PICKLE FORMAT: Single flat list of integers (5 tokens per event)
# 
# EVENT TYPES (distinguished by first token = channel marker):
#   - Note event:     [channel, dtime, dur, pitch, vel]  where channel = 0-15
#   - Harmony event:  [17, harm_x, harm_y, harm_r, key_root]  channel = 17 marks harmony
#   - New song event: [18, key, 0, 0, 0]  channel = 18 marks new song
#
# All values are in range 0-127 (except channel markers 17, 18)
#
# HARMONY FEATURES:
#   - harm_x: Tonnetz X coordinate (fifths axis), quantized 0-127
#   - harm_y: Tonnetz Y coordinate (thirds axis), quantized 0-127  
#   - harm_r: Tension magnitude (mean distance from key), quantized 0-127
#   - key_root: Root of the key (0-11 major, 12-23 minor, 24 unknown)
#
# Harmony events are inserted at regular intervals (every HARMONY_INTERVAL_MS).
# Note dtime is computed only between consecutive notes (harmony events don't affect it).

HARMONY_INTERVAL_MS = 500  # Insert harmony event every 500ms
HARMONY_N_BINS = 128  # Quantization bins for x, y, r (0-127)

# Channel markers for event types
CHANNEL_HARMONY = 17  # Marks a harmony event
CHANNEL_NEW_SONG = 18  # Marks a new song event

# Process MIDIs

# first quantize the time and duration, then calculate the time difference and maximum duration
def time2quant(time):
    return int(time/10)

def dur2quant(dur):
    return int(dur/20)

MELODY_ONLY = 0
ACCOMP_ONLY = 1
MELODY_AND_ACCOMP = 2
ALL_CHANNELS = 3 # we will use all channels

useful_channels = ALL_CHANNELS 

sorted_or_random_file_loading_order = False # Sorted order is NOT usually recommended
dataset_ratio = 1 # Change this if you need more or less % of the dataset

train_and_test_ratio = 1. # 100% for training
#train_and_test_ratio = 0.8 # 80% for training, 20% for testing

# Melody channel filter
MELODY_CHANNEL = 0  # Channel 0 is melody
ACCOMP_CHANNEL = 10  # Channel 10 is accompaniment
MELODY_2nd_CHANNEL = 1
ACCOMP_2nd_CHANNEL = 11

###########

files_count = 0

gfiles = []

# Flat token lists (single list of integers, 5 tokens per event)
train_data1 = []
test_data1 = []

###########

# dataset_addr = "./Samples"  # when testing
dataset_addr = "../../../DataSets/MIDI/giantMIDI/all_train"
# Output file names
output_name = 'giantmidi_full_harmony_train'


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
harmony_success = 0
harmony_fail = 0

for f in tqdm(filez[:int(len(filez) * dataset_ratio)]):
    try:
        fn = os.path.basename(f)
        fn1 = fn.split('.')[0]

        #print('Loading MIDI file...')
        score = midi2ms_score(open(f, 'rb').read())

        events_matrix = []

        itrack = 1

        while itrack < len(score):
            for event in score[itrack]:         
                if event[0] == 'note' and event[3] != 9: # skip percussion notes
                    events_matrix.append(event)
            itrack += 1
        
        if len(events_matrix) > 0:

          # Sorting by pitch (descending) then by time (ascending). when the second sort reorders by time, 
          # notes with the same start time (i.e., chords) retain their relative order from the first sort (by pitch descending).
          events_matrix.sort(key=lambda x: x[4], reverse=True) # pitch
          events_matrix.sort(key=lambda x: x[1]) # time

          # Filter for melody notes BEFORE timing recalculation
          # This ensures delta times are calculated between consecutive melody notes
          if useful_channels == MELODY_ONLY:
            # event format: ['note', start_time, duration, channel, pitch, velocity]
            filtered_events_matrix = [e for e in events_matrix if e[3] == MELODY_CHANNEL]
          elif useful_channels == ACCOMP_ONLY:
            # event format: ['note', start_time, duration, channel, pitch, velocity]
            filtered_events_matrix = [e for e in events_matrix if e[3] == ACCOMP_CHANNEL]
          elif useful_channels == MELODY_AND_ACCOMP:
            # event format: ['note', start_time, duration, channel, pitch, velocity]
            filtered_events_matrix = [e for e in events_matrix if (e[3] == MELODY_CHANNEL or e[3] == ACCOMP_CHANNEL)]
          else:
            # event format: ['note', start_time, duration, channel, pitch, velocity]
            filtered_events_matrix = events_matrix
          
          # Skip files with no notes after filtering
          if len(filtered_events_matrix) == 0:
              continue

          # --- EXTRACT HARMONY FEATURES ---
          harmony_events = extract_harmony_for_tokens(
              midi_path=f,
              harmony_interval_ms=HARMONY_INTERVAL_MS,
              n_bins=HARMONY_N_BINS
          )
          if harmony_events:
              harmony_success += 1
          else:
              harmony_fail += 1
              harmony_events = []  # Empty list if extraction failed

          # --- MERGE NOTE AND HARMONY EVENTS BY TIME ---
          # Create unified event list:
          #   ('note', time_ms, dur_ms, chan, pitch, vel) 
          #   ('harmony', time_ms, x_val, y_val, r_val, key_root)  where x,y,r are floats in [0,1]
          unified_events = []
          
          # Add note events (time is in ms before quantization)
          for e in filtered_events_matrix:
              # event format: ['note', start_time_ms, duration_ms, channel, pitch, velocity]
              unified_events.append(('note', e[1], e[2], e[3], e[4], e[5]))
          
          # Add harmony events
          for h in harmony_events:
              # h = (time_ms, x_bin, y_bin, r_bin, key_root)
              unified_events.append(('harmony', h[0], h[1], h[2], h[3], h[4]))
          
          # Sort by time (index 1), harmony events come after notes at same time
          unified_events.sort(key=lambda x: (x[1], 0 if x[0] == 'note' else 1))

          # Determine if this file goes to train or test
          is_train = random.random() < train_and_test_ratio
          target_data = train_data1 if is_train else test_data1

          # Get the global key for this song (from first harmony event, or unknown)
          global_key = 24  # unknown by default
          for h in harmony_events:
              # h = (time_ms, x_bin, y_bin, r_bin, key_root)
              global_key = max(0, min(24, h[4]))
              break  # Use first harmony event's key
          
          # Add new_song token: [CHANNEL_NEW_SONG, key, 0, 0, 0]
          target_data.extend([CHANNEL_NEW_SONG, global_key, 0, 0, 0])
          
          # Track previous note time for dtime calculation (only updated by notes)
          prev_note_time_quant = None
          
          for evt in unified_events:
              if evt[0] == 'note':
                  # Note event: ('note', time_ms, dur_ms, chan, pitch, vel)
                  _, time_ms, dur_ms, chan, ptc, vel = evt
                  
                  time_quant = time2quant(time_ms)
                  dur_quant = dur2quant(dur_ms)
                  
                  # dtime is computed strictly from previous note (or 0 for first note)
                  if prev_note_time_quant is None:
                      dtime = 0  # First note has dtime = 0
                  else:
                      dtime = max(0, min(126, time_quant - prev_note_time_quant))
                  
                  dur = max(1, min(126, dur_quant))
                  chan = max(0, min(15, chan))
                  ptc = max(1, min(126, ptc))
                  vel = max(1, min(126, vel))
                  
                  # Note event: [channel, dtime, dur, pitch, vel]
                  target_data.extend([chan, dtime, dur, ptc, vel])
                  
                  # Update previous note time for next note's dtime calculation
                  prev_note_time_quant = time_quant
                  
              else:
                  # Harmony event: ('harmony', time_ms, x_val, y_val, r_val, key_root)
                  # x_val, y_val, r_val are floats in [0, 1], convert to [0, 127]
                  _, time_ms, x_val, y_val, r_val, key_root = evt
                  
                  # Quantize from [0,1] to [0,127] and clamp
                  harm_x = max(0, min(127, int(x_val * 127)))
                  harm_y = max(0, min(127, int(y_val * 127)))
                  harm_r = max(0, min(127, int(r_val * 127)))
                  key_rt = max(0, min(24, key_root))
                  
                  # Harmony event: [CHANNEL_HARMONY, harm_x, harm_y, harm_r, key_root]
                  target_data.extend([CHANNEL_HARMONY, harm_x, harm_y, harm_r, key_rt])

          files_count += 1
        
    except KeyboardInterrupt:
        print('Quitting...')
        break  

    except Exception as ex:
        print(f'Bad MIDI: {f} - {ex}')
        continue

print(f'Harmony extraction: {harmony_success} success, {harmony_fail} failed')

# Save training data
output_path = './Training-Data/' + output_name + '_train'
print('Saving training data...')
Any_Pickle_File_Writer(train_data1, output_path)        
print(f'Training data saved to {output_path}.pickle')   
num_tokens = len(train_data1)
num_events = num_tokens // 5
print(f'{num_tokens} tokens, {num_events} events (5 tokens each)')
print('=' * 70)

# Save test data
if train_and_test_ratio < 1.:
    print('Saving test data...')
    output_path = './Training-Data/' + output_name + '_test'
    Any_Pickle_File_Writer(test_data1, output_path)
    print(f'Test data saved to {output_path}.pickle')
    num_tokens = len(test_data1)
    num_events = num_tokens // 5
    print(f'{num_tokens} tokens, {num_events} events (5 tokens each)')
    print('=' * 70)

print('Done!')
