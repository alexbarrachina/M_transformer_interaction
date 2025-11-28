#===================================================================================================
# Monster Genie preprocessUtils.py Python module
# Helper functions for melodic segmentation, automatic melody extraction, chord re-order
#
# Copyright 2025 Alex Barrachina
# Licensed under the Apache License, Version 2.0
#===================================================================================================

import torch
from torch import Tensor

@torch.no_grad()
def detect_structure_boundaries(
    pitch: torch.Tensor,
    dtime: torch.Tensor,
    *,
    window: int = 16,
    jump_semitones: int = 12,
    dtime_abs_threshold: float = 8,
    dtime_quantile: float = 0.95,
    range_semitones: int = 24,
    curvature_semitones: int = 2,
    min_segment_len: int = 6,
    include_start: bool = True
) -> torch.Tensor:
    """
    Returns a boolean mask of shape [T] with True at boundary indices.
    Boundary at index i means a segment can start at i (i.e., a cut between i-1 and i).

    Parameters:
      pitch: int tensor [T] MIDI pitches
      dtime: int tensor [T] delta times
      window: rolling window for local range
      jump_semitones: threshold for |Δpitch|
      dtime_abs_threshold: if provided, uses this absolute dtime threshold; otherwise dtime_quantile
      dtime_quantile: quantile for long IOI if abs threshold not given (0..1)
      range_semitones: threshold for rolling (max-min)
      curvature_semitones: abs second-difference threshold at peaks/troughs
      min_segment_len: minimum number of notes between consecutive boundaries
      include_start: mark index 0 as a boundary (start of sequence)
    """
    assert pitch.ndim == 1 and dtime.ndim == 1 and pitch.shape == dtime.shape
    T = pitch.shape[0]
    if T == 0:
        return torch.zeros(0, dtype=torch.bool, device=pitch.device)

    pitch_f = pitch.to(torch.float32)
    dtime_f = dtime.to(torch.float32)


    # 1) Long IOI boundaries when dtime >= threshold
    dthr = float(dtime_abs_threshold)
    b_ioi = dtime_f >= dthr

    # Δpitch, Δ²pitch
    dp = torch.zeros_like(pitch_f)
    dp[1:] = pitch_f[1:] - pitch_f[:-1]
    d2 = torch.zeros_like(pitch_f)
    d2[2:] = dp[2:] - dp[1:-1]

    # 2) Large pitch jump boundaries at current index when |Δpitch| >= jump
    b_jump = torch.zeros(T, dtype=torch.bool, device=pitch.device)
    b_jump[1:] = dp[1:].abs() >= float(jump_semitones)


    # 3+4) Segment-based range+peak: use windows delimited by (1) and (2)
    # Precompute directional peaks/troughs once
    sign_prev = torch.sign(dp.roll(1))
    sign_curr = torch.sign(dp)
    sign_change = (sign_prev != 0) & (sign_curr != 0) & (sign_prev != sign_curr)
    b_peak = sign_change & (d2.abs() >= float(curvature_semitones))

    # Seed boundaries from conditions (1) and (2)
    boundaries = torch.zeros(T, dtype=torch.bool, device=pitch.device)
    if include_start and T > 0:
        boundaries[0] = True
    boundaries |= (b_jump | b_ioi)

    '''
    # Build segments from seeds
    seed_idx = torch.nonzero(boundaries, as_tuple=False).flatten()
    if seed_idx.numel() == 0 and T > 0:
        seed_idx = torch.tensor([0], device=pitch.device)
    if seed_idx.numel() > 0 and seed_idx[0] != 0:
        seed_idx = torch.cat([torch.tensor([0], device=pitch.device), seed_idx])
    # append end sentinel
    if seed_idx.numel() > 0:
        ends = torch.cat([seed_idx[1:], torch.tensor([T], device=pitch.device)])
        for s, e in zip(seed_idx.tolist(), ends.tolist()):
            if e - s <= 1:
                continue
            seg = pitch_f[s:e]
            # if segment span large enough, add boundaries at peaks inside segment
            if (seg.max() - seg.min()) >= float(range_semitones):
                peaks_seg = b_peak[s:e]
                if s < T:
                    peaks_seg[0] = False  # don't duplicate at start
                boundaries[s:e] |= peaks_seg

    # Enforce minimum segment length
    if min_segment_len > 1 and boundaries.any():
        idx = torch.nonzero(boundaries, as_tuple=False).flatten()
        if idx.numel() > 1:
            keep = torch.ones_like(idx, dtype=torch.bool)
            last_keep_pos = idx[0]
            for k in range(1, idx.numel()):
                if (idx[k] - last_keep_pos) < min_segment_len:
                    keep[k] = False
                else:
                    last_keep_pos = idx[k]
            filtered = torch.zeros_like(boundaries)
            filtered[idx[keep]] = True
            boundaries = filtered
    '''
    return boundaries

@torch.no_grad()
def re_order_chord_notes(features):
    '''
    Re-order chord notes from lower to higher pitch
    Chord notes are identified by dtime=0 (same time as previous note)
    features: dictionary with keys: dtime, dur, pitch, vel, chan (all tensors)
    output: dictionary with keys: dtime, dur, pitch, vel, chan (all tensors)
    '''
    
    # Work directly with tensors
    dtime_tensor = features['dtime']
    dur_tensor = features['dur']
    pitch_tensor = features['pitch']
    vel_tensor = features['vel']
    chan_tensor = features['chan']
    
    # Find chord boundaries: notes with dtime > 0 start new chords
    # Notes with dtime = 0 belong to the same chord as the previous note
    chord_starts = torch.cat([torch.tensor([0]), torch.where(dtime_tensor > 0)[0]])
    chord_ends = torch.cat([chord_starts[1:], torch.tensor([len(dtime_tensor)])])
    
    # List to store reordered indices
    reordered_indices = []
    
    # Process each chord
    for start, end in zip(chord_starts, chord_ends):
        chord_indices = torch.arange(start, end)
        
        if len(chord_indices) > 1:
            # Multiple notes in chord - sort by pitch
            chord_pitches = pitch_tensor[chord_indices]
            sorted_pitch_indices = torch.argsort(chord_pitches)
            sorted_chord_indices = chord_indices[sorted_pitch_indices]
        else:
            # Single note
            sorted_chord_indices = chord_indices
        
        reordered_indices.extend(sorted_chord_indices.tolist())
    
    # Convert to tensor for indexing
    reordered_indices = torch.tensor(reordered_indices, dtype=torch.long)
    
    # Reorder all features using the sorted indices
    new_features = {
        'dtime': dtime_tensor,
        'dur': dur_tensor[reordered_indices],
        'pitch': pitch_tensor[reordered_indices],
        'vel': vel_tensor[reordered_indices],
        'chan': chan_tensor[reordered_indices]
    }
    
    return new_features

@torch.no_grad()
def monophonic_melody_mask(features, channel=None, dtime_threshold=0):
    '''
    Create boolean masks for monophonic melody selection and chord identification
    Chord notes are identified by dtime=0 (same time as previous note) across ALL channels
    notes of the same chord are reduced to 1:
    - If the previous note is not in a chord, will pick the closest to the tendency of previous notes and discard the rest. 
    - If the previous note is in a chord, will pick the note with same order in the chord. 
      (if the 3rd note of a chord is selected, consecutive chords will select the 3rd note of each chord)
    features: dictionary with keys: dtime, dur, pitch, vel, chan (all tensors)
    channel: int or None. If not None, only notes from this channel will be considered for selection
    dtime_threshold: int, notes with dtime <= threshold are considered part of the same chord
    output: tuple of two boolean tensor masks:
        - melody_mask: True for selected monophonic melody notes, False otherwise
        - chord_mask: True for notes that are part of multi-note chords (>1 note), False for isolated notes
                     (only considers notes from the selected channel)
    '''
    
    # Work directly with tensors
    dtime_tensor = features['dtime']
    pitch_tensor = features['pitch']
    chan_tensor = features['chan']
    
    # Initialize masks - all False initially
    melody_mask = torch.zeros(len(dtime_tensor), dtype=torch.bool)
    chord_mask = torch.zeros(len(dtime_tensor), dtype=torch.bool)
    
    # Apply channel filter if specified
    if channel is not None:
        channel_mask = (chan_tensor == channel)
        # If no notes in the specified channel, return all False masks
        if not channel_mask.any():
            return melody_mask, chord_mask
    else:
        channel_mask = torch.ones(len(dtime_tensor), dtype=torch.bool)
    
    # Find chord boundaries across ALL channels first
    # This gives us the true chord structure regardless of channel filtering
    if len(dtime_tensor) == 0:
        return melody_mask, chord_mask
    
    # chord boundary detection, considering all channels
    # melody notes ar considered as 1 no†e chords
    chord_starts = [0]  # First note always starts a chord
    cumulative_time = 0
    
    for i in range(1, len(dtime_tensor)):
        current_dtime = dtime_tensor[i].item()
        
        if current_dtime == 0:
            # dtime=0 means part of current chord (unless it's the first note or all other notes are different channels)
            continue
        elif current_dtime > dtime_threshold:
            # Large dtime definitely starts a new chord
            chord_starts.append(i)
            cumulative_time = 0
        else:
            # Small but non-zero dtime: check cumulative time from chord start
            cumulative_time += current_dtime
            if cumulative_time > dtime_threshold:
                # Cumulative time exceeds threshold, start new chord
                chord_starts.append(i)
                cumulative_time = 0
            # else: still part of current chord
    
    chord_starts = torch.tensor(chord_starts, dtype=torch.long)
    chord_ends = torch.cat([chord_starts[1:], torch.tensor([len(dtime_tensor)])])
    
    # Lists to store selected indices for monophonic melody
    selected_original_indices = []
    selected_pitches = []  # Track pitches for tendency calculation
    
    previous_pitch = None
    previous_chord_position = None
    previous_chord_num_notes = None
    
    # Process each chord (defined across all channels)
    for start, end in zip(chord_starts, chord_ends):
        # Get all notes in this chord (across all channels)
        chord_indices = torch.arange(start, end)
        
        # Filter to only notes in the target channel within this chord
        chord_channel_mask = channel_mask[chord_indices]
        target_channel_indices_in_chord = chord_indices[chord_channel_mask]
        
        if len(target_channel_indices_in_chord) == 0:
            # No notes from target channel in this chord, skip
            continue
        elif len(target_channel_indices_in_chord) == 1:
            # Single note from target channel - use it
            selected_original_idx = target_channel_indices_in_chord[0]
            selected_original_indices.append(selected_original_idx.item())
            current_pitch = pitch_tensor[selected_original_idx].item()
            selected_pitches.append(current_pitch)
            previous_pitch = current_pitch
            previous_chord_position = 0  # Single note is at position 0
            chord_size_in_target_channel = 1
            # This is an isolated note (not part of multi-note chord), chord_mask stays False
        else:
            # Multiple notes from target channel in this chord - sort by pitch to determine positions
            chord_pitches = pitch_tensor[target_channel_indices_in_chord]
            
            # Sort chord notes by pitch to establish consistent ordering
            sorted_pitch_indices = torch.argsort(chord_pitches)
            sorted_chord_pitches = chord_pitches[sorted_pitch_indices]
            sorted_chord_original_indices = target_channel_indices_in_chord[sorted_pitch_indices]
            
            if previous_chord_num_notes == 1:  # previous note was a single note chord
                # Calculate melodic tendency from last 4 selected notes (same as for chord case)
                if len(selected_pitches) >= 4:
                    # Calculate mean pitch difference over last 4 notes
                    recent_pitches = selected_pitches[-4:]
                    pitch_diffs = []
                    for i in range(1, len(recent_pitches)):
                        pitch_diffs.append(recent_pitches[i] - recent_pitches[i-1])
                    
                    # Calculate mean tendency (average pitch change per step)
                    mean_tendency = sum(pitch_diffs) / len(pitch_diffs)
                    
                    # Predict expected pitch based on tendency
                    expected_pitch = previous_pitch + mean_tendency
                    
                    # Find closest pitch to expected pitch (considering tendency)
                    pitch_distances = torch.abs(sorted_chord_pitches - expected_pitch)
                    closest_idx = torch.argmin(pitch_distances)
                    selected_original_idx = sorted_chord_original_indices[closest_idx]
                    previous_chord_position = closest_idx.item()
                else:
                    # Not enough history, find closest pitch to previous note
                    pitch_distances = torch.abs(sorted_chord_pitches - previous_pitch)
                    closest_idx = torch.argmin(pitch_distances)
                    selected_original_idx = sorted_chord_original_indices[closest_idx]
                    previous_chord_position = closest_idx.item()
            elif previous_chord_num_notes is not None and previous_chord_num_notes > 1:  # previous note was a chord
                # Calculate melodic tendency from last 4 selected notes
                if len(selected_pitches) >= 4:
                    # Calculate mean pitch difference over last 4 notes
                    recent_pitches = selected_pitches[-4:]
                    pitch_diffs = []
                    for i in range(1, len(recent_pitches)):
                        pitch_diffs.append(recent_pitches[i] - recent_pitches[i-1])
                    
                    # Calculate mean tendency (average pitch change per step)
                    mean_tendency = sum(pitch_diffs) / len(pitch_diffs)
                    
                    # Predict expected pitch based on tendency
                    expected_pitch = previous_pitch + mean_tendency
                    
                    # Find closest pitch to expected pitch (considering tendency)
                    pitch_distances = torch.abs(sorted_chord_pitches - expected_pitch)
                    closest_idx = torch.argmin(pitch_distances)
                    selected_original_idx = sorted_chord_original_indices[closest_idx]
                    previous_chord_position = closest_idx.item()
                else:
                    # Not enough history, use the same position as previous chord
                    if previous_chord_position >= len(sorted_chord_pitches):
                        # If previous position doesn't exist, use the highest note
                        selected_original_idx = sorted_chord_original_indices[-1]
                        previous_chord_position = len(sorted_chord_pitches) - 1
                    else:
                        selected_original_idx = sorted_chord_original_indices[previous_chord_position]
            else:      
                # No previous note, use the highest note (highest pitch)
                selected_original_idx = sorted_chord_original_indices[-1]
                previous_chord_position = 0
            
            selected_original_indices.append(selected_original_idx.item())
            # Get the pitch of the selected note for next iteration
            selected_pos_in_sorted = previous_chord_position
            current_pitch = sorted_chord_pitches[selected_pos_in_sorted].item()
            selected_pitches.append(current_pitch)
            previous_pitch = current_pitch
            chord_size_in_target_channel = len(target_channel_indices_in_chord)
            
            # Mark discarted notes in this multi-note chord in the chord_mask
            for idx in target_channel_indices_in_chord:
                if idx != selected_original_idx.item():
                    chord_mask[idx.item()] = True
        
        previous_chord_num_notes = chord_size_in_target_channel
    
    # Set melody_mask to True for selected positions (using original indices)
    for original_idx in selected_original_indices:
        melody_mask[original_idx] = True
    
    return melody_mask, chord_mask

