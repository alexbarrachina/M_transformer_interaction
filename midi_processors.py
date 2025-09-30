#===================================================================================================
# Monster Genie midi_processors.py Python module
# Converts MIDI files <-> tokens
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


import midiUtils

#===================================================================================================

def midi_to_tokens(input_midi,
                   encode_velocity=False,
                   verbose=False
                   ):

    if verbose:
        print('=' * 70)
        print('Encoding MIDI...')

    raw_score = midiUtils.midi2single_track_ms_score(input_midi)
    
    escore_notes = midiUtils.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]
    escore_notes = midiUtils.augment_enhanced_score_notes(escore_notes, timings_divider=32)
    
    sp_escore_notes = midiUtils.solo_piano_escore_notes(escore_notes, keep_drums=False)
    zscore = midiUtils.recalculate_score_timings(sp_escore_notes)
    
    cscore = midiUtils.chordify_score([1000, zscore])
    
    score = []
    
    pc = cscore[0]
    
    notes_counter = 0
    
    for i, c in enumerate(cscore): # c[0][1] absolute time in miliseconds /32 -> dtime 0-127
        score.append(max(0, min(127, c[0][1]-pc[0][1]))) # calculate dtime: the time difference between the current chord and the previous note
    
        for n in c: # tokens in note event
            if encode_velocity:
                score.extend([max(1, min(127, n[2]))+128, max(1, min(127, n[4]))+256, max(1, min(127, n[5]))+384])

            else:
                score.extend([max(1, min(127, n[2]))+128, max(1, min(127, n[4]))+256])
                
            notes_counter += 1
    
        pc = c
        
    if verbose:
        print('Done!')
        print('=' * 70)
        
        print('Source MIDI composition has', len(zscore), 'notes')
        print('Source MIDI composition has', len(cscore), 'chords')
        print('-' * 70)
        print('Encoded sequence has', notes_counter, 'pitches')
        print('Encoded sequence has', i+1, 'chords')
        print('-' * 70)
        print('Final encoded sequence has', len(score), 'tokens')
        print('=' * 70)
        
    return score

#===================================================================================================

def tokens_to_midi(tokens,
                   custom_channel=-1,
                   custom_velocity=-1,
                   custom_patch=-1,
                   output_signature = 'Monster Piano Transformer',
                   track_name='Project Los Angeles',
                   output_midi_name='Monster-Piano-Transformer-Composition',
                   return_ms_score=False,
                   verbose=False
                   ):
    
    if verbose:
        print('=' * 70)
        print('Decoding tokens...')
    
    if [t for t in tokens if 384 < t < 512]:
        model_with_velocity = True
        
    else:
        model_with_velocity = False
    
    song = tokens
    song_f = []

    time = 0
    dur = 8
    vel = 90
    pitch = 60
    channel = 0
    patch = 0

    patches = [0] * 16
    
    if -1 < custom_channel < 16:
        channel = custom_channel
        
    if -1 < custom_patch < 128:
        patch = custom_patch
        patches[channel] = patch

    for m in song:

        if 0 <= m < 128:
            time += m * 32

        elif 128 < m < 256:
            dur = (m-128) * 32

        elif 256 < m < 384:
            pitch = (m-256)
            
            if not model_with_velocity:
                
                if 0 < custom_velocity < 128:
                    vel = custom_velocity
                    
                else:
                    if not model_with_velocity:
                        vel = max(40, pitch)
                
                song_f.append(['note', time, dur, channel, pitch, vel, patch])

        elif 384 < m < 512:
            vel = (m-384)

            if model_with_velocity:
                
                if 0 < custom_velocity < 128:
                    vel = custom_velocity
                    
                song_f.append(['note', time, dur, channel, pitch, vel, patch])
                
    if verbose:
        print('Done!')
        print('=' * 70)
                
    detailed_stats = midiUtils.ms_SONG_to_MIDI_Converter(song_f,
                                                                output_signature=output_signature,
                                                                output_file_name=output_midi_name,
                                                                track_name=track_name,
                                                                list_of_MIDI_patches=patches,
                                                                verbose=verbose
                                                            )
    
    if verbose:
        print('=' * 70)
    
    if return_ms_score:
        return song_f

    else:
        return detailed_stats
    
#===================================================================================================
