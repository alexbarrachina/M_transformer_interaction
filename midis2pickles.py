import random
import os
from tqdm import tqdm
from params import *
from midiUtils import midi2ms_score, Tegridy_Any_Pickle_File_Writer, time2quant, dur2quant, vel2quant

# Process MIDIs

sorted_or_random_file_loading_order = False # Sorted order is NOT usually recommended
dataset_ratio = 1 # Change this if you need more data

###########

files_count = 0

gfiles = []

train_data1 = []

###########

# dataset_addr = "./Samples"  # when testing
dataset_addr = "../../../DataSets/MIDI/giantMIDI/sel"

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
              # e[3] channel, e[4] pitch
              e[5] = vel2quant(e[5]) # event velocity / 4 -> 128/4 = 32

          # final processing...
          # (dtime, vel, pitch, dur)
          train_data1.extend([RANGE_DTIME_SHIFT, 0, 0, RANGE_DUR_SHIFT]) # Intro/Zero seq

          pe = events_matrix[0]
          for e in events_matrix:

            assert e[4] < PIANO_NUM_KEYS + PIANO_LOWEST_KEY_MIDI_PITCH, "pitch must be less than 88+21"
            time = max(0, min(RANGE_DTIME_SHIFT, e[1]-pe[1])) # time difference from previous events, but trunk to maximum 126
            dur = max(1, min(RANGE_DUR_SHIFT, e[2])) # maximum duration 126
            ptc = max(1, min(PIANO_NUM_KEYS, e[4] - PIANO_LOWEST_KEY_MIDI_PITCH)) # maximum pitch 88
            vel = max(1, min(RANGE_VEL, e[5])) # maximum velocity 32

            #train_data1.extend([ptc+PITCH_OFF, time+0, dur+DUR_OFF, vel+VEL_OFF]) # re-order to priorize pitch output first
            train_data1.extend([time, vel, ptc, dur]) # re-order with duration at the end

            pe = e

          files_count += 1
        
    except KeyboardInterrupt:
        print('Quitting...')
        break  

    except:
        print('Bad MIDI:', f)
        continue

print('=' * 70)
Tegridy_Any_Pickle_File_Writer(train_data1, './Training-Data/processedMIDIs')        
print('Done!')   
print(str(len(train_data1)) + ' tokens')
print('=' * 70)

