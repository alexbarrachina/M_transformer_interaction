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

# NO OFFSETS: Each token type is stored in its raw range (0-127)
# The pickle format stores 5 tokens per note: [dtime, dur, pitch, vel, chan]
# All values are in range 0-127 (or 0-15 for channel)

# Process MIDIs

# first quantize the time and duration, then calculate the time difference and maximum duration
def time2quant(time):
    return int(time/10)

def dur2quant(dur):
    return int(dur/20)

melody_only = False # if True, only process melody notes (channel 0)
sorted_or_random_file_loading_order = False # Sorted order is NOT usually recommended
dataset_ratio = 1 # Change this if you need more or less % of the dataset

#train_and_test_ratio = 1. # 100% for training
train_and_test_ratio = 0.8 # 80% for training, 20% for testing

# Melody channel filter
MELODY_CHANNEL = 0  # Channel 0 is melody
ACCOMP_CHANNEL = 10  # Channel 10 is accompaniment
MELODY_ONLY = 0
ACCOMP_ONLY = 1
MELODY_AND_ACCOMP = 2
ALL_CHANNELS = 3 # we will use all channels

useful_channels = ACCOMP_ONLY 

###########

files_count = 0

gfiles = []

train_data1 = []
test_data1 = []

# Channel statistics
total_notes = 0
channel_0_notes = 0
channel_10_notes = 0

###########

# dataset_addr = "./Samples"  # when testing
dataset_addr = "../../../DataSets/MIDI/giantMIDI/all_chan_segmented"
# Output file names
output_name = 'giantmidi_full_melody'


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
            filtered_events_matrix = events_matrix
          
          # Skip files with no notes after filtering
          if len(filtered_events_matrix) == 0:
              continue

          # Recalculating timings (quantize) 
          for e in filtered_events_matrix:
              e[1] = time2quant(e[1])
              e[2] = dur2quant(e[2])

          # Determine if this file goes to train or test
          is_train = random.random() < train_and_test_ratio
          target_data = train_data1 if is_train else test_data1

          # Intro/Zero seq (5 tokens) - no offsets, raw values
          target_data.extend([126, 126, 0, 0, 0])  # dtime, dur, pitch, vel, chan

          pe = filtered_events_matrix[0]
          for e in filtered_events_matrix:

              time = max(0, min(126, e[1]-pe[1]))
              dur = max(1, min(126, e[2]))
              chan = max(0, min(15, e[3]))  # Channel 0-15
              ptc = max(1, min(126, e[4]))
              vel = max(1, min(126, e[5]))

              # 5 tokens per note: dtime, dur, pitch, vel, chan (no offsets)
              target_data.extend([time, dur, ptc, vel, chan])

              # Update channel statistics
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
else:
    print('No notes processed')
print('=' * 70)

print('Done!')
