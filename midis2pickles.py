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

from midiUtils import midi2ms_score, Any_Pickle_File_Writer #, DUR_OFF, PITCH_OFF, VEL_OFF, time2quant, dur2quant

# Offsets create non-overlapping ranges for each token type
# DTIME 0-127
DUR_OFF = 128
# DUR 128-255
PITCH_OFF = 256
# PITCH 256-383
VEL_OFF = 384
# VEL 384-511

# Process MIDIs

# first quantize the time and duration, then calculate the time difference and maximum duration
def time2quant(time):
    return int(time/10)

def dur2quant(dur):
    return int(dur/20)


sorted_or_random_file_loading_order = False # Sorted order is NOT usually recommended
dataset_ratio = 1 # Change this if you need more data

###########

files_count = 0

gfiles = []

train_data1 = []

###########

# dataset_addr = "./Samples"  # when testing
dataset_addr = "../../../DataSets/MIDI/giantMIDI/test"

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
                if event[0] == 'note' and event[3] != 9:
                    events_matrix.append(event)
            itrack += 1
        
        if len(events_matrix) > 0:

          # Sorting...
          events_matrix.sort(key=lambda x: x[4], reverse=True)
          events_matrix.sort(key=lambda x: x[1])

          # recalculating timings
          for e in events_matrix:
              e[1] = time2quant(e[1])
              e[2] = dur2quant(e[2])
          
          # final processing...

            # TODO comprovar l'ordre correcte
          #train_data1.extend([0+PITCH_OFF, 126+0, 126+DUR_OFF, 0+VEL_OFF]) # Intro/Zero seq
          train_data1.extend([126+0, 126+DUR_OFF, 0+PITCH_OFF, 0+VEL_OFF]) # Intro/Zero seq

          pe = events_matrix[0]
          for e in events_matrix:

              time = max(0, min(126, e[1]-pe[1]))
              dur = max(1, min(126, e[2]))
              ptc = max(1, min(126, e[4]))
              vel = max(1, min(126, e[5]))

              #train_data1.extend([ptc+PITCH_OFF, time+0, dur+DUR_OFF, vel+VEL_OFF]) # re-order to priorize pitch output first
              train_data1.extend([time+0, dur+DUR_OFF, ptc+PITCH_OFF, vel+VEL_OFF]) # re-order to priorize pitch output first

              pe = e

          files_count += 1
        
    except KeyboardInterrupt:
        print('Quitting...')
        break  

    except:
        print('Bad MIDI:', f)
        continue

print('=' * 70)
Any_Pickle_File_Writer(train_data1, './Training-Data/processedMIDIs')        
print('Done!')   
print(str(len(train_data1)) + ' tokens')
print('=' * 70)

