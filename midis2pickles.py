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
#   Channel 5 (chord references) is discarded entirely.

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

###########

# dataset_addr = "./Samples"  # when testing
dataset_addr = "../../../DataSets/MIDI/giantMIDI/all_harmony"
# Output file names
output_name = 'giantmidi_full_harmony'


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

          # Include channel 4 (harmony movements) in the event stream
          # Channel 5 (chord references) is discarded entirely
          if useful_channels == MELODY_ONLY:
            filtered_events_matrix = [e for e in events_matrix if e[3] in (MELODY_CHANNEL, HARMONY_CHANNEL)]
          elif useful_channels == ACCOMP_ONLY:
            filtered_events_matrix = [e for e in events_matrix if e[3] in (ACCOMP_CHANNEL, HARMONY_CHANNEL)]
          elif useful_channels == MELODY_AND_ACCOMP:
            filtered_events_matrix = [e for e in events_matrix if e[3] in (MELODY_CHANNEL, ACCOMP_CHANNEL, HARMONY_CHANNEL)]
          elif useful_channels == MELODY_AND_ACCOMP_NOT_HARMONY:
            filtered_events_matrix = [e for e in events_matrix if e[3] in (MELODY_CHANNEL, ACCOMP_CHANNEL)]
          else:
            filtered_events_matrix = events_matrix

          # Skip files with no playable notes
          if not any(e[3] != HARMONY_CHANNEL for e in filtered_events_matrix):
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
          # Harmony events (channel 3) do NOT update pe
          pe = next(e for e in filtered_events_matrix if e[3] != HARMONY_CHANNEL)
          for e in filtered_events_matrix:

              if e[3] == HARMONY_CHANNEL:
                  # Harmony movement: [0, 0, movement_type, 0, chan=3], pe NOT updated
                  movement_type = max(0, min(126, e[4]))
                  target_data.extend([0, 0, movement_type, 0, HARMONY_CHANNEL])
                  channel_4_movements += 1
              else:
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
if total_notes > 0:
    print(f'Movement events per note ratio: {channel_4_movements / total_notes:.4f}')
print('=' * 70)

print('Done!')
