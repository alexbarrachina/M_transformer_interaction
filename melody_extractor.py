#==================================================================================================
# Monster Genie melody extractor test.py Python module
# Extract a monophonic melody from a MIDI file, avoiding chord notes.
# notes from a channel (default 0) are selected.
# notes of the same chord (contiguous notes with same dtime) are re-ordered using pitch
# notes of the same chord (notes with same dtime) are reduced to 1:
#   -  will pick the closest to the previous note and discard the rest. if the previous note is not in a chord
#   - If the previous note is in a chord, will pick the note with same order in the chord. 
#     (if the 3rd note of a chord is selected, consecutive chords will select the 3rd note of each chord)
# Output is a MIDI file with the selected monophonic melody.
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
import os
from tqdm import tqdm

from midiUtils import midi_to_dict, dict_to_song, ms_SONG_to_MIDI_Converter
from preprocessUtils import monophonic_melody_mask

''' DEVICE '''
#device = torch.device('cpu')
device = torch.device('mps') 

''' PARAMS '''
# Get sample seed MIDI path
file_name='Baines_hkNo8ESFFZU'
sample_midi_path = './samples/2hands/' + file_name + '.midi'
output_midi_name = './out/' + file_name
CHANNEL = 0

''' GET MIDIFILES '''

###########

# dataset_addr = "./Samples"  # when testing

#dataset_addr = "../../../DataSets/MIDI/giantMIDI/sel_hannds/_post-process"
#dataout_addr = "../../../DataSets/MIDI/giantMIDI/sel_hannds/_post-process/melody_segmented"

dataset_addr = "../PROCESS/hannds/out"
dataout_addr = "../PROCESS/hannds/out/melody_segmented"


filez = list()
for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
    filez += [os.path.join(dirpath, file) for file in filenames]
print('=' * 70)

if filez == []:
    print('Could not find any MIDI files. Please check Dataset dir...')
    print('=' * 70)

''' PROCESS MIDIFILES '''

print('Processing MIDI files. Please wait...')
for sample_midi_path in tqdm(filez[:int(len(filez))]):
    try:
        fn = os.path.basename(sample_midi_path)
        #fn1 = fn.split('.')[0]

        ''' BUILD CTX '''
        # Load seed MIDI
        input_tokens, _ = midi_to_dict(sample_midi_path) # tokens, with vel and chan

        # Build context tokens (tensors)
        features = {
            'dtime': torch.tensor(input_tokens['dtime'], dtype=torch.long),
            'pitch': torch.tensor(input_tokens['pitch'], dtype=torch.long),
            'dur': torch.tensor(input_tokens['dur'], dtype=torch.long),
            'vel': torch.tensor(input_tokens['vel'], dtype=torch.long),
            'chan': torch.tensor(input_tokens['chan'], dtype=torch.long)
                }
        # build monophonic melody mask, discarding notes of the same chord (notes with same dtime)
        melody_mask, chord_mask = monophonic_melody_mask(features, channel=CHANNEL, dtime_threshold=1)

        # keep all notes, but change channel of melody notes to CHANNEL+1
        modified_features = {
            'dtime': features['dtime'].clone(),
            'pitch': features['pitch'].clone(),
            'dur': features['dur'].clone(),
            'vel': features['vel'].clone(),
            'chan': features['chan'].clone()
        }

        # change channel of non-melody chord notes to CHANNEL+1
        modified_features['chan'][chord_mask] = CHANNEL + 1

        # convert to lists for MIDI generation
        output_tokens = {
            'dtime': modified_features['dtime'].tolist(),
            'pitch': modified_features['pitch'].tolist(),
            'dur': modified_features['dur'].tolist(),
            'vel': modified_features['vel'].tolist(),
            'chan': modified_features['chan'].tolist()
        }

        # generate a midi file from generated pitches
        #print('Generating MIDI file...')
        #print(f'melody notes on channel {CHANNEL}, chord notes on channel {CHANNEL+1}')

        song_d = dict_to_song(output_tokens, force_chan=False)
        detailed_stats = ms_SONG_to_MIDI_Converter(song_d, output_file_name = dataout_addr+'/'+fn,
                                                timings_multiplier=2, add_extension=False
                                          )

    except KeyboardInterrupt:
        print('Quitting...')
        break  

    except:
        print('Bad MIDI:', sample_midi_path)
        continue