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

from midiUtils import midi_to_dict, dict_to_song, ms_SONG_to_MIDI_Converter, monophonic_melody_mask

''' DEVICE '''
#device = torch.device('cpu')
device = torch.device('mps') 

''' PARAMS '''
# Get sample seed MIDI path
sample_midi_path = './samples/test4.midi'
output_midi_name = './out/test4'

CHANNEL = 0

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

# select notes from channel 0
'''features['pitch'] = features['pitch'][features['chan'] == CHANNEL]
features['dtime'] = features['dtime'][features['chan'] == CHANNEL]
features['dur'] = features['dur'][features['chan'] == CHANNEL]
features['vel'] = features['vel'][features['chan'] == CHANNEL]
features['chan'] = features['chan'][features['chan'] == CHANNEL]
'''

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

# change channel of selected melody notes to CHANNEL+1
modified_features['chan'][melody_mask] = CHANNEL + 1

# convert to lists for MIDI generation
output_tokens = {
    'dtime': modified_features['dtime'].tolist(),
    'pitch': modified_features['pitch'].tolist(),
    'dur': modified_features['dur'].tolist(),
    'vel': modified_features['vel'].tolist(),
    'chan': modified_features['chan'].tolist()
}

# generate a midi file from generated pitches
print('Generating MIDI file...')
print(f'Original notes on channel {CHANNEL}, monophonic melody on channel {CHANNEL+1}')

song_d = dict_to_song(output_tokens, force_chan=False)
detailed_stats = ms_SONG_to_MIDI_Converter(song_d, output_file_name = output_midi_name,
                                          timings_multiplier=2
                                          )

 