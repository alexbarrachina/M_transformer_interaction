import random
import os
from tqdm import tqdm

from midiUtils import midi2ms_score, Tegridy_Any_Pickle_File_Writer

# Process MIDIs

sorted_or_random_file_loading_order = False # Sorted order is NOT usually recommended
dataset_ratio = 1 # Change this if you need more data

###########

files_count = 0

gfiles = []

train_data1 = []

###########

# dataset_addr = "./Samples"  # when testing
dataset_addr = "../../../DataSets/MIDI/piano_jazz/sel"

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
              e[1] = int(e[1] / 10)
              e[2] = int(e[2] / 20)
          
          # final processing...

          train_data1.extend([0+256, 126+0, 126+128, 0+384]) # Intro/Zero seq

          pe = events_matrix[0]
          for e in events_matrix:

              time = max(0, min(126, e[1]-pe[1]))
              dur = max(1, min(126, e[2]))
              ptc = max(1, min(126, e[4]))
              vel = max(1, min(126, e[5]))

              train_data1.extend([ptc+256, time+0, dur+128, vel+384]) # re-order to priorize pitch output first

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

