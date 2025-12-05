#===================================================================================================
# Monster Genie segmentation_test.py Python module
# Load a MIDI, detect structure boundaries and export MIDI with marked notes
# at boundary points (by changing channel) similar to melody_extractor_test.py
#
# Copyright 2025 Alex Barrachina
# Licensed under the Apache License, Version 2.0
#===================================================================================================

import torch
from midiUtils import midi_to_dict, dict_to_song, ms_SONG_to_MIDI_Converter
from loss_funcs import detect_structure_boundaries

if __name__ == "__main__":

    device = torch.device('cpu')

    # Parameters
    sample_midi_path = './samples/test_mono8.midi'
    output_midi_name = './out/_segmented'
    CHANNEL = 0
    CHANNEL_OFFSET = 1

    # Load MIDI
    tokens, _ = midi_to_dict(sample_midi_path)

    # Build tensors
    features = {
        'dtime': torch.tensor(tokens['dtime'], dtype=torch.long, device=device),
        'pitch': torch.tensor(tokens['pitch'], dtype=torch.long, device=device),
        'dur': torch.tensor(tokens['dur'], dtype=torch.long, device=device),
        'vel': torch.tensor(tokens['vel'], dtype=torch.long, device=device),
        'chan': torch.tensor(tokens['chan'], dtype=torch.long, device=device)
    }

    # Detect boundaries (global across all notes; you can also filter by channel)
    boundaries = detect_structure_boundaries(
        pitch = features['pitch'],
        dtime = features['dtime'],
        window = 16,
        jump_semitones = 14 ,
        dtime_abs_threshold = 35,
        range_semitones = 24,
        curvature_semitones = 2,
        min_segment_len = 1,
        include_start = True
    )

    # Mark boundary notes by changing channel to CHANNEL+2 (distinct from melody extractor)
    modified = {
        'dtime': features['dtime'].clone(),
        'pitch': features['pitch'].clone(),
        'dur': features['dur'].clone(),
        'vel': features['vel'].clone(),
        'chan': features['chan'].clone()
    }

    # Only mark boundary indices that belong to CHANNEL (if desired), otherwise mark all
    mask = boundaries
    if CHANNEL is not None:
        mask = mask & (modified['chan'] == CHANNEL)

    modified['chan'][mask] = CHANNEL + CHANNEL_OFFSET

    output_tokens = {
        'dtime': modified['dtime'].tolist(),
        'pitch': modified['pitch'].tolist(),
        'dur': modified['dur'].tolist(),
        'vel': modified['vel'].tolist(),
        'chan': modified['chan'].tolist()
    }

    # Write MIDI
    print('Generating segmented MIDI file...')
    print(f'Original notes on channel {CHANNEL}, boundary notes on channel {CHANNEL+2}')
    song_d = dict_to_song(output_tokens, force_chan=False)
    _ = ms_SONG_to_MIDI_Converter(song_d, output_file_name = output_midi_name,
                                  timings_multiplier=2)
