#===================================================================================================
# Monster Genie midis2pickles.py Python module
# Converts MIDI files into a pickle file
# 
# Copyright 2025 Alex Barrachina
#
# Based on Project Los Angeles / Tegridy Code 2025
#   https://github.com/asigalov61/monsterpianotransformer
# 
# previously based upon MIDI.py module v.6.7. by Peter Billam / pjb.com.au
#   https://pjb.com.au/
#   https://peterbillam.gitlab.io/miditools/
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


import sys, struct, copy
Version = '6.7'
VersionDate = '20201120'

_previous_warning = ''  # 5.4
_previous_times = 0     # 5.4
_no_warning = False

#------------------------------- Encoding stuff --------------------------

def opus2midi(opus=[], text_encoding='ISO-8859-1'):
    r'''The argument is a list: the first item in the list is the "ticks"
parameter, the others are the tracks. Each track is a list
of midi-events, and each event is itself a list; see above.
opus2midi() returns a bytestring of the MIDI, which can then be
written either to a file opened in binary mode (mode='wb'),
or to stdout by means of:   sys.stdout.buffer.write()

my_opus = [
    96, 
    [   # track 0:
        ['patch_change', 0, 1, 8],   # and these are the events...
        ['note_on',   5, 1, 25, 96],
        ['note_off', 96, 1, 25, 0],
        ['note_on',   0, 1, 29, 96],
        ['note_off', 96, 1, 29, 0],
    ],   # end of track 0
]
my_midi = opus2midi(my_opus)
sys.stdout.buffer.write(my_midi)
'''
    if len(opus) < 2:
        opus=[1000, [],]
    tracks = copy.deepcopy(opus)
    ticks = int(tracks.pop(0))
    ntracks = len(tracks)
    if ntracks == 1:
        format = 0
    else:
        format = 1

    my_midi = b"MThd\x00\x00\x00\x06"+struct.pack('>HHH',format,ntracks,ticks)
    for track in tracks:
        events = _encode(track, text_encoding=text_encoding)
        my_midi += b'MTrk' + struct.pack('>I',len(events)) + events
    _clean_up_warnings()
    return my_midi


def score2opus(score=None, text_encoding='ISO-8859-1'):
    r'''
The argument is a list: the first item in the list is the "ticks"
parameter, the others are the tracks. Each track is a list
of score-events, and each event is itself a list.  A score-event
is similar to an opus-event (see above), except that in a score:
 1) the times are expressed as an absolute number of ticks
from the track's start time
 2) there are no 'note_off' events, only 'note_on' events with
velocity=0
 3) the times are sorted (this is the only format that
score2opus() accepts, so you may have to call sort() on
the score before calling score2opus())
'''
    if score == None:
        return [1000, [],]
    if len(score) < 2:
        return [1000, [],]
    tracks = copy.deepcopy(score)
    ticks = int(tracks.pop(0))
    opus_tracks = []
    for score_track in tracks:
        opus_track = []
        for score_event in score_track:
            opus_event = copy.deepcopy(score_event)
            if opus_event[0] == 'note_on' and opus_event[4] == 0:
                opus_event[0] = 'note_off'
                del opus_event[4]
            opus_track.append(opus_event)
        opus_tracks.append(opus_track)
    opus = [ticks] + opus_tracks
    _clean_up_warnings()
    return opus


def score2midi(score=None, text_encoding='ISO-8859-1'):
    r'''
Translates a score into MIDI, using score2opus() then opus2midi()
'''
    return opus2midi(score2opus(score, text_encoding), text_encoding)


def midi2opus(midi=b'', do_not_check_MIDI_signature=False):
    r'''
Translates MIDI into a "opus".  For a discussion of the
"opus" format, see opus2midi().
The opus format is essentially the same as MIDI, but:
 1) the times are expressed as an absolute number of ticks
from the track's start time
 2) the tracks are just lists of midi events.  No further
structure is assumed.
 3) the fourth item of each note_on event is
guaranteed to be a positive integer; this is some
programs' "note on velocity", and 100 is often the
default value.  The "note off velocity" is taken from
the fifth item of a note_off event.
'''
    my_midi = bytearray(midi)
    if len(my_midi) < 4:
        _warn('midi2opus: midi file is too short')
        return [1000, [],]
    if not do_not_check_MIDI_signature:
        if my_midi[0:4] != b'MThd':
            _warn('midi2opus: midi file does not start with MThd')
            return [1000, [],]
    [length, format, ntracks, ticks] = struct.unpack('>IHHH', my_midi[4:14])
    my_midi = my_midi[14:]
    opus_tracks = []
    for i in range(ntracks):
        track = _decode(my_midi)
        opus_tracks.append(track[2])
        my_midi = track[1]
    opus = [ticks] + opus_tracks
    _clean_up_warnings()
    return opus


def opus2score(opus=[]):
    r'''
The argument is a list: the first item in the list is the "ticks"
parameter, the others are the tracks. Each track is a list
of midi-events, and each event is itself a list.  For a
discussion of the "opus" format, see opus2midi().
opus2score() returns a "score", which is similar to an
"opus", except that:
 1) all times are expressed as an absolute number of ticks
from the track's start time
 2) there are no 'note_off' events, only 'note_on' events with
velocity=0
 3) the times are sorted
'''
    if len(opus) < 2:
        return [1000, [],]
    tracks = copy.deepcopy(opus)
    ticks = int(tracks.pop(0))
    score = [ticks]
    for opus_track in tracks:
        score_track = []
        for opus_event in opus_track:
            score_event = copy.deepcopy(opus_event)
            if score_event[0] == 'note_off':
                score_event[0] = 'note_on'
                score_event.append(0)
            score_track.append(score_event)
        score.append(score_track)
    _clean_up_warnings()
    return score


def midi2score(midi=b'', do_not_check_MIDI_signature=False):
    r'''
Translates MIDI into a "score" and sorts each track.
For a discussion of the "score" format, see opus2score().
'''
    return opus2score(midi2opus(midi, do_not_check_MIDI_signature))


def midi2ms_score(midi=b'', do_not_check_MIDI_signature=False):
    r'''
Translates MIDI into a "ms_score" and sorts each track.
For a discussion of the "ms_score" format, see opus2score().
'''
    return to_millisecs(midi2score(midi, do_not_check_MIDI_signature))


def midi2single_track_ms_score(midi_path_or_bytes, 
                                recalculate_channels = False, 
                                pass_old_timings_events= False, 
                                verbose = False, 
                                do_not_check_MIDI_signature=False
                                ):
    r'''
Translates MIDI into a single track "ms_score" and sorts it.
For a discussion of the "ms_score" format, see opus2score().
'''
    if isinstance(midi_path_or_bytes, str):
        with open(midi_path_or_bytes, 'rb') as f:
            midi_data = f.read()
    else:
        midi_data = midi_path_or_bytes
    
    if verbose:
        print('Processing MIDI file...')
    
    # Convert to score format
    score = midi2score(midi_data, do_not_check_MIDI_signature)
    
    if len(score) < 2:
        if verbose:
            print('No tracks found in MIDI file')
        return [1000, []]
    
    # Merge all tracks into one
    merged_track = []
    for track in score[1:]:
        merged_track.extend(track)
    
    # Sort by time
    merged_track.sort(key=lambda x: x[1])
    
    # Recalculate channels if requested
    if recalculate_channels:
        for event in merged_track:
            if len(event) > 3:
                event[3] = 0  # Set all to channel 0
    
    # Convert to milliseconds
    ms_score = to_millisecs([score[0], merged_track], pass_old_timings_events=pass_old_timings_events)
    
    if verbose:
        print(f'Processed {len(merged_track)} events')
    
    _clean_up_warnings()
    return ms_score


def to_millisecs(old_opus=None, desired_time_in_ms=1, pass_old_timings_events = False):
    r'''
Translates the "absolute time" fields in a score to milliseconds.
For a discussion of the "score" format, see opus2score().
'''
    if old_opus == None:
        return [1000, [],]
    if len(old_opus) < 2:
        return [1000, [],]
    tracks = copy.deepcopy(old_opus)
    ticks = int(tracks.pop(0))
    ms_tracks = []
    for track in tracks:
        ms_track = []
        for event in track:
            ms_event = copy.deepcopy(event)
            if not pass_old_timings_events:
                ms_event[1] = int(ms_event[1] * desired_time_in_ms / ticks)
            ms_track.append(ms_event)
        ms_tracks.append(ms_track)
    ms_opus = [desired_time_in_ms] + ms_tracks
    _clean_up_warnings()
    return ms_opus


def _twobytes2int(byte_a):
    r'''decode a 16 bit quantity from two bytes,'''
    return (byte_a[1] | (byte_a[0] << 8))


def _int2twobytes(int_16bit):
    r'''encode a 16 bit quantity into two bytes,'''
    return bytes([(int_16bit>>8) & 0xFF, int_16bit & 0xFF])


def _read_14_bit(byte_a):
    r'''decode a 14 bit quantity from two bytes,'''
    return (byte_a[0] | (byte_a[1] << 7))


def _write_14_bit(int_14bit):
    r'''encode a 14 bit quantity into two bytes,'''
    return bytes([int_14bit & 0x7F, (int_14bit>>7) & 0x7F])


def _ber_compressed_int(integer):
    r'''BER compressed integer (not an ASN.1 BER, see perlpacktut for
details).  Its bytes represent an unsigned integer in base 128,
most significant digit first, with as few digits as possible.
Bit eight (the high bit) is set on each byte except the last.
'''
    ber = bytearray(b'')
    while True:
        ber.insert(0, (integer & 0x7F))
        if integer < 128:
            break
        integer = integer >> 7
    ber[-1] |= 0x80
    return bytes(ber)


def _unshift_ber_int(ba):
    r'''This takes a bytearray, and returns a tuple of (n, rest_of_ba).
n is the BER compressed integer at the start of ba.
rest_of_ba is ba without the first n bytes.
'''
    i = 0
    n = 0
    while True:
        if i >= len(ba):
            _warn('_unshift_ber_int: unexpected end of data')
            return (0, ba)
        byte = ba[i]
        i += 1
        n = (n << 7) + (byte & 0x7F)
        if byte & 0x80:
            break
    return (n, ba[i:])


def _clean_up_warnings():  # 5.4
    # Call this before returning from any publicly callable function
    # whenever there's a possibility that a warning might have been printed
    # by the function, or by any private functions it might have called.
    global _previous_warning, _previous_times, _no_warning
    if _no_warning:
        return
    if _previous_times > 1:
        print('  previous warning was given %d more times' % (_previous_times - 1))
    _previous_warning = ''
    _previous_times = 0


def _warn(s=''):
    r'''
Warn the user about something.
'''
    global _previous_warning, _previous_times, _no_warning
    if _no_warning:
        return
    if s == _previous_warning:
        _previous_times += 1
        return
    _previous_times = 0
    print('WARNING: %s' % s)
    _previous_warning = s


def _some_text_event(which_kind=0x01, text=b'some_text', text_encoding='ISO-8859-1'):
    r'''
Returns a list of the form ['text_event', delta_time, text]
'''
    if isinstance(text, str):
        text = text.encode(text_encoding)
    return ['text_event', 0, which_kind, text]


def _consistentise_ticks(scores):  # 3.6
    # used by mix_scores, merge_scores, concatenate_scores
    if len(scores) == 0:
        return []
    if len(scores) == 1:
        return scores
    ticks = scores[0][0]
    for score in scores[1:]:
        if score[0] != ticks:
            _warn('_consistentise_ticks: ticks were inconsistent; using the first score\'s ticks')
            break
    return scores


def _decode(trackdata=b'', exclude=None, include=None,
            event_callback=None, exclusive_event_callback=None, no_eot_magic=False):
    r'''
Decodes MIDI track data into a list of events.
'''
    if exclude is None:
        exclude = set()
    if include is None:
        include = set()
    
    trackdata = bytearray(trackdata)
    events = []
    i = 0
    time = 0
    last_status = 0
    
    while i < len(trackdata):
        # Read delta time
        delta_time, trackdata = _unshift_ber_int(trackdata)
        time += delta_time
        
        # Read status byte
        if i >= len(trackdata):
            break
        status = trackdata[i]
        i += 1
        
        # Handle running status
        if status < 0x80:
            status = last_status
            i -= 1
        
        last_status = status
        
        # Parse event based on status
        if status == 0xFF:  # Meta event
            if i >= len(trackdata):
                break
            meta_type = trackdata[i]
            i += 1
            length, trackdata = _unshift_ber_int(trackdata)
            if i + length > len(trackdata):
                break
            data = trackdata[i:i+length]
            i += length
            events.append(['meta_event', time, meta_type, data])
            
        elif status == 0xF0 or status == 0xF7:  # SysEx
            length, trackdata = _unshift_ber_int(trackdata)
            if i + length > len(trackdata):
                break
            data = trackdata[i:i+length]
            i += length
            events.append(['sysex_event', time, data])
            
        elif status >= 0x80 and status <= 0xEF:  # Channel message
            channel = status & 0x0F
            message_type = status & 0xF0
            
            if message_type == 0x80:  # Note off
                if i + 2 > len(trackdata):
                    break
                note = trackdata[i]
                velocity = trackdata[i+1]
                i += 2
                events.append(['note_off', time, channel, note, velocity])
                
            elif message_type == 0x90:  # Note on
                if i + 2 > len(trackdata):
                    break
                note = trackdata[i]
                velocity = trackdata[i+1]
                i += 2
                events.append(['note_on', time, channel, note, velocity])
                
            elif message_type == 0xA0:  # Aftertouch
                if i + 2 > len(trackdata):
                    break
                note = trackdata[i]
                pressure = trackdata[i+1]
                i += 2
                events.append(['aftertouch', time, channel, note, pressure])
                
            elif message_type == 0xB0:  # Controller
                if i + 2 > len(trackdata):
                    break
                controller = trackdata[i]
                value = trackdata[i+1]
                i += 2
                events.append(['controller', time, channel, controller, value])
                
            elif message_type == 0xC0:  # Program change
                if i + 1 > len(trackdata):
                    break
                program = trackdata[i]
                i += 1
                events.append(['patch_change', time, channel, program])
                
            elif message_type == 0xD0:  # Channel pressure
                if i + 1 > len(trackdata):
                    break
                pressure = trackdata[i]
                i += 1
                events.append(['channel_pressure', time, channel, pressure])
                
            elif message_type == 0xE0:  # Pitch bend
                if i + 2 > len(trackdata):
                    break
                lsb = trackdata[i]
                msb = trackdata[i+1]
                i += 2
                pitch_bend = (msb << 7) | lsb
                events.append(['pitch_wheel_change', time, channel, pitch_bend])
    
    return (len(trackdata), trackdata, events)


def _encode(events_lol, unknown_callback=None, never_add_eot=False,
  no_eot_magic=False, no_running_status=False, text_encoding='ISO-8859-1'):
    # encode an event structure, presumably for writing to a file
    # Calling format:
    #   $data_r = MIDI::Event::encode( \@event_lol, { options } );
    # Takes a REFERENCE to an event structure (a LoL)
    # Returns an (unblessed) REFERENCE to track data.

    # If you want to use this to encode a /single/ event,
    # you still have to do it as a reference to an event structure (a LoL)
    # that just happens to have just one event.  I.e.,
    #   encode( [ $event ] ) or encode( [ [ 'note_on', 100, 5, 42, 64] ] )
    # If you're doing this, consider the never_add_eot track option, as in
    #   print MIDI ${ encode( [ $event], { 'never_add_eot' => 1} ) };

    if not isinstance(events_lol, list):
        _warn('_encode: events_lol is not a list')
        return b''
    
    if len(events_lol) == 0:
        return b''
    
    # Convert absolute times to delta times
    delta_events = []
    last_time = 0
    for event in events_lol:
        if not isinstance(event, list) or len(event) < 2:
            continue
        delta_time = event[1] - last_time
        delta_event = [event[0], delta_time] + event[2:]
        delta_events.append(delta_event)
        last_time = event[1]
    
    # Encode events
    data = []
    last_status = 0
    
    for event in delta_events:
        if not isinstance(event, list) or len(event) < 2:
            continue
            
        event_type = event[0]
        delta_time = event[1]
        
        # Encode delta time
        data.append(_ber_compressed_int(delta_time))
        
        # Encode event
        if event_type == 'note_on':
            if len(event) >= 5:
                channel = event[2]
                note = event[3]
                velocity = event[4]
                status = 0x90 | (channel & 0x0F)
                data.append(bytes([status, note & 0x7F, velocity & 0x7F]))
                last_status = status
                
        elif event_type == 'note_off':
            if len(event) >= 5:
                channel = event[2]
                note = event[3]
                velocity = event[4]
                status = 0x80 | (channel & 0x0F)
                data.append(bytes([status, note & 0x7F, velocity & 0x7F]))
                last_status = status
                
        elif event_type == 'patch_change':
            if len(event) >= 4:
                channel = event[2]
                program = event[3]
                status = 0xC0 | (channel & 0x0F)
                data.append(bytes([status, program & 0x7F]))
                last_status = status
                
        elif event_type == 'controller':
            if len(event) >= 5:
                channel = event[2]
                controller = event[3]
                value = event[4]
                status = 0xB0 | (channel & 0x0F)
                data.append(bytes([status, controller & 0x7F, value & 0x7F]))
                last_status = status
                
        elif event_type == 'meta_event':
            if len(event) >= 4:
                meta_type = event[2]
                meta_data = event[3]
                if isinstance(meta_data, str):
                    meta_data = meta_data.encode(text_encoding)
                data.append(bytes([0xFF, meta_type & 0x7F]))
                data.append(_ber_compressed_int(len(meta_data)))
                data.append(meta_data)
                
        elif event_type == 'end_of_track':
            data.append(bytes([0xFF, 0x2F, 0x00]))
    
    # Add end of track if not present and not disabled
    if not never_add_eot and not no_eot_magic:
        # Check if last event is end_of_track
        if len(events_lol) == 0 or events_lol[-1][0] != 'end_of_track':
            data.append(_ber_compressed_int(0))
            data.append(bytes([0xFF, 0x2F, 0x00]))

    return b''.join(data)

###################################################################################
###################################################################################
###################################################################################
#
#	Tegridy MIDI X Module (TMIDI X / tee-midi eks)
#	Version 1.0
#
#	Based upon and includes the amazing MIDI.py module v.6.7. by Peter Billam
#	pjb.com.au
#
#	Project Los Angeles
#	Tegridy Code 2021
# https://github.com/Tegridy-Code/Project-Los-Angeles
#
###################################################################################
###################################################################################
###################################################################################

import os
import datetime
import copy
from datetime import datetime
import secrets
import random
import pickle
import csv
import tqdm
from itertools import zip_longest
from itertools import groupby
from collections import Counter
from operator import itemgetter
import sys
from abc import ABC, abstractmethod
from difflib import SequenceMatcher as SM
import statistics
import math
import matplotlib.pyplot as plt

###################################################################################

def ms_SONG_to_MIDI_Converter(ms_SONG,
                                      output_signature = 'Tegridy TMIDI Module', 
                                      track_name = 'Composition Track',
                                      list_of_MIDI_patches = [0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 0, 0, 0, 0, 0, 0],
                                      output_file_name = 'TMIDI-Composition',
                                      text_encoding='ISO-8859-1',
                                      timings_multiplier=1,
                                      verbose=True
                                      ):
    r'''
    Converts ms_SONG to MIDI file
    '''
    if verbose:
        print('Converting ms_SONG to MIDI...')
    
    if len(ms_SONG) < 2:
        if verbose:
            print('Empty ms_SONG, creating empty MIDI')
        return score2midi([1000, []], text_encoding)
    
    ticks = ms_SONG[0]
    tracks = ms_SONG[1:]
    
    # Convert to score format
    score = [ticks]
    for track in tracks:
        score_track = []
        for event in track:
            if isinstance(event, list) and len(event) >= 4:
                if event[0] == 'note':
                    # Convert note event to MIDI events
                    start_time = int(event[1] * timings_multiplier)
                    duration = int(event[2] * timings_multiplier)
                    channel = event[3] if len(event) > 3 else 0
                    pitch = event[4] if len(event) > 4 else 60
                    velocity = event[5] if len(event) > 5 else 100
                    patch = event[6] if len(event) > 6 else 0
                    
                    # Add patch change
                    score_track.append(['patch_change', start_time, channel, patch])
                    
                    # Add note on
                    score_track.append(['note_on', start_time, channel, pitch, velocity])
                    
                    # Add note off
                    score_track.append(['note_off', start_time + duration, channel, pitch, 0])
                else:
                    score_track.append(event)
        score.append(score_track)
    
    # Convert to MIDI
    midi_data = score2midi(score, text_encoding)
    
    # Write to file if filename provided
    if output_file_name:
        with open(output_file_name + '.mid', 'wb') as f:
            f.write(midi_data)
        if verbose:
            print(f'MIDI file saved as {output_file_name}.mid')
    
    return midi_data


def Any_Pickle_File_Writer(Data, input_file_name='TMIDI_Pickle_File'):
    r'''
    Writes any data to a pickle file
    '''
    try:
        with open(input_file_name + '.pickle', 'wb') as f:
            pickle.dump(Data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Data successfully saved to {input_file_name}.pickle')
        return True
    except Exception as e:
        print(f'Error saving pickle file: {e}')
        return False


def Any_Pickle_File_Reader(input_file_name='TMIDI_Pickle_File', ext='.pickle', verbose=True):
    r'''
    Reads any data from a pickle file
    '''
    try:
        with open(input_file_name + ext, 'rb') as f:
            data = pickle.load(f)
        if verbose:
            print(f'Data successfully loaded from {input_file_name}{ext}')
        return data
    except Exception as e:
        if verbose:
            print(f'Error loading pickle file: {e}')
        return None


def chordify_score(score,
                  return_choridfied_score=True,
                  return_detected_score_information=False
                  ):
    r'''
    Chordifies a score by grouping simultaneous notes
    '''
    if len(score) < 2:
        return score
    
    ticks = score[0]
    tracks = score[1:]
    
    chordified_tracks = []
    for track in tracks:
        chordified_track = []
        current_time = 0
        current_notes = []
        
        for event in track:
            if isinstance(event, list) and len(event) >= 4:
                if event[0] == 'note_on':
                    if event[1] != current_time:
                        # Process current chord
                        if current_notes:
                            chordified_track.append(['chord', current_time, 0, current_notes])
                        current_time = event[1]
                        current_notes = []
                    current_notes.append(event[3])  # pitch
                elif event[0] == 'note_off':
                    # End of chord
                    if current_notes:
                        chordified_track.append(['chord', current_time, event[1] - current_time, current_notes])
                        current_notes = []
        
        # Add final chord if exists
        if current_notes:
            chordified_track.append(['chord', current_time, 0, current_notes])
        
        chordified_tracks.append(chordified_track)
    
    result = [ticks] + chordified_tracks
    
    if return_choridfied_score:
        return result
    else:
        return None


def advanced_score_processor(raw_score, 
                              patches_to_analyze=list(range(129)), 
                              return_score_analysis=False,
                              return_enhanced_score=False,
                              return_enhanced_score_notes=False,
                              return_enhanced_monophonic_melody=False,
                              return_chordified_enhanced_score=False,
                              return_chordified_enhanced_score_with_lyrics=False,
                              return_score_tones_chords=False,
                              return_text_and_lyric_events=False
                            ):
    r'''
    Advanced score processor that extracts various representations
    '''
    if len(raw_score) < 2:
        return [None] * 9
    
    ticks = raw_score[0]
    tracks = raw_score[1:]
    
    # Extract notes
    enhanced_score_notes = []
    for track in tracks:
        for event in track:
            if isinstance(event, list) and len(event) >= 4:
                if event[0] == 'note':
                    enhanced_score_notes.append(event)
    
    # Sort by time
    enhanced_score_notes.sort(key=lambda x: x[1])
    
    # Create enhanced score
    enhanced_score = [ticks, enhanced_score_notes]
    
    # Chordify if requested
    chordified_enhanced_score = None
    if return_chordified_enhanced_score:
        chordified_enhanced_score = chordify_score(enhanced_score)
    
    # Extract monophonic melody (first track)
    enhanced_monophonic_melody = None
    if return_enhanced_monophonic_melody and tracks:
        melody_notes = []
        for event in tracks[0]:
            if isinstance(event, list) and len(event) >= 4:
                if event[0] == 'note':
                    melody_notes.append(event)
        enhanced_monophonic_melody = [ticks, melody_notes]
    
    # Extract tones chords
    score_tones_chords = []
    if return_score_tones_chords:
        for event in enhanced_score_notes:
            if len(event) > 4:
                pitch = event[4]
                tone = pitch % 12
                if tone not in score_tones_chords:
                    score_tones_chords.append(tone)
    
    # Extract text and lyric events
    text_and_lyric_events = []
    if return_text_and_lyric_events:
        for track in tracks:
            for event in track:
                if isinstance(event, list) and len(event) >= 3:
                    if event[0] in ['text_event', 'lyric']:
                        text_and_lyric_events.append(event)
    
    # Score analysis
    score_analysis = None
    if return_score_analysis:
        score_analysis = {
            'num_notes': len(enhanced_score_notes),
            'num_tracks': len(tracks),
            'ticks': ticks,
            'tones_chords': score_tones_chords
        }
    
    return [
        enhanced_score_notes,
        enhanced_score,
        enhanced_monophonic_melody,
        chordified_enhanced_score,
        chordified_enhanced_score_with_lyrics,
        score_tones_chords,
        text_and_lyric_events,
        score_analysis,
        None  # Placeholder for additional return value
    ]


def check_and_fix_tones_chord(tones_chord, use_full_chords=True):
    r'''
    Checks and fixes tones chord
    '''
    if not isinstance(tones_chord, list):
        return [0, 4, 7]  # Default C major chord
    
    # Remove duplicates and sort
    unique_tones = sorted(list(set(tones_chord)))
    
    # Ensure all values are in range 0-11
    fixed_tones = [t % 12 for t in unique_tones]
    
    # Remove duplicates again after modulo
    fixed_tones = sorted(list(set(fixed_tones)))
    
    # If empty, return default chord
    if not fixed_tones:
        return [0, 4, 7]
    
    return fixed_tones


def augment_enhanced_score_notes(enhanced_score_notes,
                                  timings_divider=16,
                                  full_sorting=True,
                                  timings_shift=0,
                                  pitch_shift=0,
                                  ceil_timings=False,
                                  round_timings=False,
                                  legacy_timings=True,
                                  sort_drums_last=False
                                ):
    r'''
    Augments enhanced score notes with various transformations
    '''
    if not enhanced_score_notes:
        return []
    
    augmented_notes = []
    for note in enhanced_score_notes:
        if isinstance(note, list) and len(note) >= 5:
            augmented_note = note.copy()
            
            # Apply timing transformations
            if len(augmented_note) > 1:
                timing = augmented_note[1]
                if timings_divider != 1:
                    timing = timing / timings_divider
                if ceil_timings:
                    timing = math.ceil(timing)
                elif round_timings:
                    timing = round(timing)
                timing += timings_shift
                augmented_note[1] = int(timing)
            
            # Apply pitch transformations
            if len(augmented_note) > 4:
                pitch = augmented_note[4]
                pitch += pitch_shift
                augmented_note[4] = pitch % 128  # Keep in MIDI range
            
            augmented_notes.append(augmented_note)
    
    # Sort if requested
    if full_sorting:
        augmented_notes.sort(key=lambda x: (x[1], x[4]) if len(x) > 4 else (x[1], 0))
    
    return augmented_notes


def recalculate_score_timings(score, 
                              start_time=0, 
                              timings_index=1
                              ):
    r'''
    Recalculates score timings to be absolute
    '''
    if len(score) < 2:
        return score
    
    ticks = score[0]
    tracks = score[1:]
    
    recalculated_tracks = []
    for track in tracks:
        recalculated_track = []
        current_time = start_time
        
        for event in track:
            if isinstance(event, list) and len(event) > timings_index:
                recalculated_event = event.copy()
                current_time += event[timings_index]
                recalculated_event[timings_index] = current_time
                recalculated_track.append(recalculated_event)
            else:
                recalculated_track.append(event)
        
        recalculated_tracks.append(recalculated_track)
    
    return [ticks] + recalculated_tracks


def flatten(list_of_lists):
    r'''
    Flattens a list of lists
    '''
    result = []
    for item in list_of_lists:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def solo_piano_escore_notes(escore_notes,
                            channels_index=3,
                            pitches_index=4,
                            patches_index=6,
                            keep_drums=False,
                            ):
    r'''
    Extracts solo piano notes from enhanced score notes
    '''
    if not escore_notes:
        return []
    
    piano_notes = []
    for note in escore_notes:
        if isinstance(note, list) and len(note) > max(channels_index, pitches_index, patches_index):
            channel = note[channels_index] if len(note) > channels_index else 0
            patch = note[patches_index] if len(note) > patches_index else 0
            
            # Keep piano notes (channel 0, patch 0) and optionally drums (channel 9)
            if channel == 0 and patch == 0:
                piano_notes.append(note)
            elif keep_drums and channel == 9:
                piano_notes.append(note)
    
    return piano_notes


def tokens_to_dict(data):
    r'''
    Converts tokens to dictionary format
    '''
    if not isinstance(data, list):
        return {}, 0
    
    # Count total tokens
    total_tokens = len(data)
    
    # Create dictionary with token counts
    token_dict = {}
    for token in data:
        if isinstance(token, (int, float)):
            token_dict[token] = token_dict.get(token, 0) + 1
    
    return token_dict, total_tokens


def midi_tokens_to_dict(score):
    r'''
    Converts MIDI tokens to dictionary format
    '''
    if not isinstance(score, list):
        return {}, 0
    
    # Extract all tokens from score
    tokens = []
    for item in score:
        if isinstance(item, list):
            tokens.extend(item)
        else:
            tokens.append(item)
    
    # Count tokens
    total_tokens = len(tokens)
    
    # Create dictionary with token counts
    token_dict = {}
    for token in tokens:
        if isinstance(token, (int, float)):
            token_dict[token] = token_dict.get(token, 0) + 1
    
    return token_dict, total_tokens


def dict_to_song(dict_data, add_vel=True):
    r'''
    Converts dictionary data to song format
    '''
    if not isinstance(dict_data, dict):
        return []
    
    song = []
    for key, value in dict_data.items():
        if isinstance(key, (int, float)) and isinstance(value, (int, float)):
            # Create note event
            note = ['note', int(key), int(value), 0, 60]  # time, duration, channel, pitch
            if add_vel:
                note.append(100)  # velocity
            song.append(note)
    
    return song


def to_device(tensor_or_dict, device):
    r'''
    Moves tensor or dictionary to device
    '''
    if hasattr(tensor_or_dict, 'to'):
        return tensor_or_dict.to(device)
    elif isinstance(tensor_or_dict, dict):
        return {k: to_device(v, device) for k, v in tensor_or_dict.items()}
    else:
        return tensor_or_dict
