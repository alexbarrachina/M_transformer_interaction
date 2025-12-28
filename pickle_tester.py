#===================================================================================================
# Monster Genie pickle_tester.py Python module
# Training with harmony-augmented dataset (Tonnetz conditioning)
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


import os
import tempfile
import numpy as np

import time
import tqdm
from random import randint, random
from typing import Dict, Tuple, List, Optional

from midiUtils import Any_Pickle_File_Reader, ms_SONG_to_MIDI_Converter, dict_to_song
from tonnetz import visualize_tonnetz

dataset_path = './Training-Data/giantmidi_full_harmony_test.pickle'
tmp_midi_path = 'out.mid'

def get_info(songs):
    """
    Get a song from the pickle file and convert it to feature_data dictionary.
    
    Args:
        songs: List of songs from the pickle file
        
    Returns:
        feature_data: Dictionary with 'pitch', 'harm_x', 'harm_y', 'harm_r', 'key_root'
        song: The original song events
        song_idx: Index of the selected song
    """
    # Pick a random song
    #song_idx = randint(0, len(songs) - 1)
    song_idx = 7
    song = songs[song_idx]
    
    # Initialize feature data lists
    pitches = []
    dtimes = []
    durs = []
    vels = []
    channels = []
    harm_x_list = []
    harm_y_list = []
    harm_r_list = []
    key_root_list = []
    
    # Initialize temporal harmonic information (will be updated by harmony events)
    last_harm_x = 0
    last_harm_y = 0
    last_harm_r = 0
    last_key_root = 0
    
    # Process events sequentially
    for event in song:
        if event[0] == 'note':
            # Note event: ['note', dtime, dur, pitch, vel, channel]
            _, dtime, dur, pitch, vel, channel = event
            
            # Add note to feature_data with current harmonic information
            pitches.append(pitch)
            dtimes.append(dtime)
            durs.append(dur)
            vels.append(vel)
            channels.append(channel)
            harm_x_list.append(last_harm_x)
            harm_y_list.append(last_harm_y)
            harm_r_list.append(last_harm_r)
            key_root_list.append(last_key_root)
            
        elif event[0] == 'harm':
            # Harmony event: ['harm', dtime, harm_x, harm_y, harm_r, key_root]
            _, dtime, harm_x, harm_y, harm_r, key_root = event
            
            # Update temporal harmonic information for next note events
            last_harm_x = harm_x
            last_harm_y = harm_y
            last_harm_r = harm_r
            last_key_root = key_root
    
    # Build feature_data dictionary
    feature_data = {
        'pitch': pitches,       # List of note pitches only (no harmony events)
        'dtimes': dtimes,
        'durs': durs,
        'vels': vels,
        'channels': channels,
        'harm_x': harm_x_list,  # Tonnetz X bin (forward-filled, aligned to notes)
        'harm_y': harm_y_list,  # Tonnetz Y bin (forward-filled, aligned to notes)
        'harm_r': harm_r_list,  # Tension magnitude bin (forward-filled, aligned to notes)
        'key_root': key_root_list,  # Key root (forward-filled, aligned to notes)
    }
    
    return feature_data, song, song_idx


def transpose_feature_data(feature_data: Dict, transpose_semitones: int, 
                          pitch_min: int = 1, pitch_max: int = 126,
                          harm_bin_min: int = 0, harm_bin_max: int = 127) -> Dict:
    """
    Transpose pitch and harmonic features by the given number of semitones.
    
    Args:
        feature_data: Dictionary with 'pitch', 'harm_x', 'harm_y', 'harm_r', 'key_root'
        transpose_semitones: Number of semitones to transpose (can be negative)
        pitch_min: Minimum pitch value (default 1)
        pitch_max: Maximum pitch value (default 126)
        harm_bin_min: Minimum harmony bin value (default 0)
        harm_bin_max: Maximum harmony bin value (default 127)
        
    Returns:
        Transposed feature_data dictionary
    """
    import numpy as np
    
    # Convert lists to numpy arrays for easier manipulation
    pitches = np.array(feature_data['pitch'], dtype=int)
    #harm_x = np.array(feature_data['harm_x'], dtype=int)
    #harm_y = np.array(feature_data['harm_y'], dtype=int)
    #harm_r = np.array(feature_data['harm_r'], dtype=int)  # harm_r doesn't transpose
    key_root = np.array(feature_data['key_root'], dtype=int)
    
    # Check if all pitches will remain in valid range after transposition
    pitches_after_transpose = pitches + transpose_semitones
    if np.any(pitches_after_transpose < pitch_min) or np.any(pitches_after_transpose > pitch_max):
        # Skip transposition if any pitch would go out of range
        print(f'Warning: Transposition by {transpose_semitones} semitones would put some pitches out of range.')
        print(f'Pitch range would be: [{pitches_after_transpose.min()}, {pitches_after_transpose.max()}]')
        print(f'Valid range is: [{pitch_min}, {pitch_max}]. Transposition skipped.')
        return feature_data  # Return original feature_data unchanged
    
    # All pitches are in valid range, proceed with transposition
    pitches_transposed = pitches_after_transpose
    
    # Transpose key_root
    # key_root encoding: 0-11 = major keys, 12-23 = minor keys, 24 = unknown
    key_root_transposed = np.zeros_like(key_root)
    for i, kr in enumerate(key_root):
        if kr == 24:  # Unknown key, keep as is
            key_root_transposed[i] = 24
        elif kr < 12:  # Major key (0-11)
            new_root = (kr + transpose_semitones) % 12
            key_root_transposed[i] = new_root
        else:  # Minor key (12-23)
            minor_root = kr - 12  # Get the minor root (0-11)
            new_minor_root = (minor_root + transpose_semitones) % 12
            key_root_transposed[i] = new_minor_root + 12
    
    # Transpose harmonic features (harm_x, harm_y)
    # These are Tonnetz coordinates, so we transpose them as well
    # Clamp to valid bin range
    #harm_x_transposed = np.clip(harm_x + transpose_semitones, harm_bin_min, harm_bin_max)
    #harm_y_transposed = np.clip(harm_y + transpose_semitones, harm_bin_min, harm_bin_max)
    
    # harm_r stays the same (tension magnitude doesn't change with transposition)
    
    return {
        'pitch': pitches_transposed.tolist(),
        'dtimes': feature_data['dtimes'],
        'durs': feature_data['durs'],
        'vels': feature_data['vels'],
        'channels': feature_data['channels'],
        'harm_x': feature_data['harm_x'],
        'harm_y': feature_data['harm_y'],
        'harm_r': feature_data['harm_r'],  # Unchanged
        'key_root': key_root_transposed.tolist(),
    }


def feature_data_to_song_dict(feature_data: Dict) -> Dict:
    """
    Convert feature_data dictionary to format expected by dict_to_song.
    
    Args:
        feature_data: Dictionary with 'dtimes', 'durs', 'vels', 'channels', 'pitch'
        
    Returns:
        Dictionary with 'dtime', 'dur', 'vel', 'chan', 'pitch' (singular keys)
    """
    return {
        'dtime': feature_data['dtimes'],
        'dur': feature_data['durs'],
        'vel': feature_data['vels'],
        'chan': feature_data['channels'],
        'pitch': feature_data['pitch'],
    }


def song_to_midi(feature_data: Dict, output_path: str):
    """
    Convert feature_data to MIDI file.
    
    Args:
        feature_data: Dictionary with note features (dtimes, durs, vels, channels, pitch)
        output_path: Path to save MIDI file
    """
    # Time quantization: time2quant(time_ms) = int(time_ms / 10)
    # So to convert back: dtime * 10 = time_ms
    # Duration quantization: dur2quant(dur_ms) = int(dur_ms / 20)
    # So to convert back: dur * 20 = dur_ms
    
    song_f = []
    time_ms = 0
    
    dtimes = feature_data['dtimes']
    durs = feature_data['durs']
    pitches = feature_data['pitch']
    vels = feature_data['vels']
    channels = feature_data['channels']
    
    song_dict = feature_data_to_song_dict(feature_data) # dtime, dur, vel, chan, pitch
    song_f = dict_to_song(song_dict)
    
    # Remove .mid extension if present (ms_SONG_to_MIDI_Converter adds it)
    if output_path.endswith('.mid'):
        output_path_no_ext = output_path[:-4]
    else:
        output_path_no_ext = output_path
    
    # Convert to MIDI
    ms_SONG_to_MIDI_Converter(
        song_f,
        output_signature='Pickle Tester',
        track_name='Extracted Song',
        output_file_name=output_path_no_ext,
        timings_multiplier=1,
        verbose=False,
        add_extension=True  # Explicitly add .mid extension
    )
    
    # Verify file was created
    final_path = output_path_no_ext + '.mid'
    if not os.path.exists(final_path):
        raise RuntimeError(f"Failed to create MIDI file: {final_path}")


def feature_data_to_tonnetz(feature_data: Dict, song: List, fs: int = 10) -> Tuple[np.ndarray, np.ndarray, Tuple[int, str]]:
    """
    Convert feature_data to tonnetz visualization format with proper timing.
    
    Args:
        feature_data: Dictionary with 'harm_x', 'harm_y', 'harm_r', 'key_root'
        song: Original song events to compute timing
        fs: Sampling frequency (samples per second)
        
    Returns:
        tension_xy: Array of shape (N, 2) with [x, y] Tonnetz coordinates
        tension_magnitude: Array of tension magnitudes
        global_key: (root, mode) tuple
    """
    # Dequantization parameters (from harmony_pickle_format.md)
    n_bins = 128
    x_range = (-3.0, 3.0)
    y_range = (-2.0, 2.0)
    r_max = 3.0
    
    harm_x_bins = np.array(feature_data['harm_x'], dtype=float)
    harm_y_bins = np.array(feature_data['harm_y'], dtype=float)
    harm_r_bins = np.array(feature_data['harm_r'], dtype=float)
    key_root_list = feature_data['key_root']
    
    # Compute note times from song events
    note_times_ms = []
    time_ms = 0
    
    for event in song:
        if event[0] == 'note':
            # Note event: ['note', dtime, dur, pitch, vel, channel]
            _, dtime, dur_quant, pitch, vel, channel = event
            time_ms += dtime * 10  # dtime was quantized by /10
            note_times_ms.append(time_ms)
    
    if len(note_times_ms) == 0:
        # No notes, return empty arrays
        return np.array([]).reshape(0, 2), np.array([]), (0, 'major')
    
    # Dequantize harmony coordinates for notes
    x_note = x_range[0] + (harm_x_bins / (n_bins - 1)) * (x_range[1] - x_range[0])
    y_note = y_range[0] + (harm_y_bins / (n_bins - 1)) * (y_range[1] - y_range[0])
    r_note = (harm_r_bins / (n_bins - 1)) * r_max
    
    # Convert to seconds
    note_times_sec = np.array(note_times_ms) / 1000.0
    
    # Create time frames at fs Hz
    total_duration = note_times_sec[-1]
    frame_times_sec = np.arange(0, total_duration, 1.0 / fs)
    
    # Interpolate harmony values to frame times using linear interpolation
    x_frames = np.interp(frame_times_sec, note_times_sec, x_note, left=x_note[0], right=x_note[-1])
    y_frames = np.interp(frame_times_sec, note_times_sec, y_note, left=y_note[0], right=y_note[-1])
    r_frames = np.interp(frame_times_sec, note_times_sec, r_note, left=r_note[0], right=r_note[-1])
    
    # Create tension_xy array
    tension_xy = np.column_stack([x_frames, y_frames])
    tension_magnitude = r_frames
    
    # Extract global key from key_root (most common key_root value)
    # key_root encoding: 0-11 = major keys, 12-23 = minor keys, 24 = unknown
    if len(key_root_list) > 0:
        # Get most common key_root (excluding unknown=24)
        valid_keys = [k for k in key_root_list if k != 24]
        if valid_keys:
            from collections import Counter
            most_common_key = Counter(valid_keys).most_common(1)[0][0]
            
            if most_common_key < 12:
                # Major key
                global_key = (most_common_key, 'major')
            elif most_common_key < 24:
                # Minor key
                global_key = (most_common_key - 12, 'minor')
            else:
                # Unknown, default to C major
                global_key = (0, 'major')
        else:
            global_key = (0, 'major')
    else:
        global_key = (0, 'major')
    
    return tension_xy, tension_magnitude, global_key


def play_and_visualize(feature_data: Dict, song: List, song_idx: int, fs: int = 10, enable_midi_playback: bool = False):
    """
    Play the song and visualize it using tonnetz.
    
    Args:
        feature_data: Dictionary with harmony features
        song: Original song events
        song_idx: Index of the song
        fs: Sampling frequency for visualization
        enable_midi_playback: If True, attempt MIDI playback (may fail on some systems)
    """
    print(f'\n--- Playing and visualizing song {song_idx} ---')
    
    # Convert feature_data to tonnetz format
    tension_xy, tension_magnitude, global_key = feature_data_to_tonnetz(feature_data, song, fs)
    
    print(f'Global Key: {global_key[0]} {global_key[1]}')
    print(f'Number of notes: {len(feature_data["pitch"])}')
    print(f'Tension XY shape: {tension_xy.shape}')
    
    # Create temporary MIDI file (always create for potential playback, even if disabled)
    #with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp_midi:
    #    tmp_midi_path = tmp_midi.name
    
    try:
        # Convert feature_data to MIDI
        song_to_midi(feature_data, tmp_midi_path)
        print(f'Temporary MIDI file created: {tmp_midi_path}')
        
        # Visualize with tonnetz
        title = f"Tonnetz: Song {song_idx}"
        if not enable_midi_playback:
            print('Note: MIDI playback is disabled (pygame.mixer does not support MIDI format)')
            print('The visualization will still work without audio.')
        
        visualize_tonnetz(
            tension_xy=tension_xy,
            tension_magnitude=tension_magnitude,
            global_key=global_key,
            fs=fs,
            title=title,
            animated=True,
            save_path=None,
            midi_path=tmp_midi_path if enable_midi_playback else None,
            play_midi=enable_midi_playback,
            active_pitches=None,  # We don't have per-frame pitch information in this format
            mouse_indicator_enabled=True,
            mouse_indicator_spread=0.3
        )
    finally:
        # Clean up temporary file
        #if os.path.exists(tmp_midi_path):
        #    os.remove(tmp_midi_path)
            print(f'Cleaned up temporary MIDI file: {tmp_midi_path}')


def main():

    ''' DATA '''

    """ LOAD TRAINING DATA """

    # Loading dataset from harmony-augmented pickles (new format: list of song dicts)
    train_songs = Any_Pickle_File_Reader(dataset_path)
    
    print(f'Loaded {len(train_songs)} songs from {dataset_path}')

    # Get feature data from a song
    feature_data, song, song_idx = get_info(train_songs)
    
    # Display information about the extracted features
    num_notes = len(feature_data['pitch'])
    
    # Transpose the features (example: transpose by +5 semitones)
    transpose_semitones = 5  # Change this to desired transposition amount
    print(f'\nTransposing by {transpose_semitones} semitones...')
    feature_data = transpose_feature_data(feature_data, transpose_semitones)
    print(f'Transposition complete.')
    
    # Play and visualize the song
    # Note: MIDI playback is disabled by default because pygame.mixer doesn't support MIDI format
    # The visualization will work without audio
    play_and_visualize(feature_data, song, song_idx, enable_midi_playback=True)

 
if __name__ == '__main__':
    main()
