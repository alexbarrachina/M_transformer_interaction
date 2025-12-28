import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import math

# --- 1. CONFIGURATION & PROFILES ---
# Krumhansl-Schmuckler Key Profiles (Major and Minor)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

#midi_path = './samples/test.mid'
#midi_path = './samples/Bach_Prelude_and_Fugue_in_C_major.mid'
#midi_path = './samples/Satie_Gymnopedie_No1.mid'
#midi_path = './samples/Kuula_6_Piano_Pieces_Op26.mid'
#midi_path = './samples/Scott_Cyril_Lotus_Land.mid'
midi_path = './samples/Chopin_Nocturnes_Op9No1_In_B_Flat_Minor.mid'
#midi_path = './samples/Adams_The_Bells_of_St_Mary.mid'
#midi_path = './samples/Beach_5_Improvisations_ Op148.mid'
#midi_path = './samples/Chopin_Nocturnes_Op.9_No2.mid'
#midi_path = './samples/Debussy_Clair_de_Lune.mid'
#midi_path = './samples/Bach_Prelude_and_Fugue_in G_minor.mid'

save_path = ''

# Tonnetz 2D Coordinates: maps semitone interval (0-11) to (x_fifths, y_thirds)
# X-axis: Circle of Fifths (+1 = dominant direction, -1 = subdominant)
# Y-axis: Major Thirds (+1 = major third up, -1 = major third down)
# Coordinates satisfy: semitone ≡ 7*x + 4*y (mod 12)
# Using minimal |x| + |y| representation for each interval
TONNETZ_COORDS = [
    (0, 0),    # 0: Unison/Root
    (-1, 2),   # 1: Minor second
    (2, 0),    # 2: Major second (2 fifths up)
    (1, -1),   # 3: Minor third
    (0, 1),    # 4: Major third
    (-1, 0),   # 5: Perfect fourth (subdominant)
    (2, -2),   # 6: Tritone
    (1, 0),    # 7: Perfect fifth (dominant)
    (0, -1),   # 8: Minor sixth
    (-1, 1),   # 9: Major sixth
    (-2, 0),   # 10: Minor seventh
    (1, 1),    # 11: Major seventh
]

# Pitch class names for display
PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

from typing import Tuple, List, Optional


def pitch_to_tonnetz(pitch_class: int, key_root: int) -> Tuple[float, float]:
    """
    Maps a pitch class to 2D Tonnetz coordinates relative to the key root.
    
    Args:
        pitch_class: Absolute pitch class (0-11, where 0=C)
        key_root: Root of the local key (0-11, where 0=C)
    
    Returns:
        Tuple[float, float]: (x, y) coordinates where:
            x = position on Circle of Fifths (positive = dominant direction)
            y = position on Thirds axis (positive = major third direction)
    """
    # Calculate interval relative to key root
    interval = (pitch_class - key_root) % 12
    return TONNETZ_COORDS[interval]


def chroma_to_tonnetz_centroid(chroma_vector: np.ndarray, key_root: int) -> Tuple[float, float]:
    """
    Computes the weighted centroid of a chroma vector on the Tonnetz grid,
    relative to the detected key root (tonal center).
    
    Args:
        chroma_vector: 12-element array of pitch class weights
        key_root: Root of the local key (0-11, where 0=C)
    
    Returns:
        Tuple[float, float]: (x, y) weighted centroid coordinates RELATIVE to key_root
    """
    total_weight = np.sum(chroma_vector)
    if total_weight == 0:
        return (0.0, 0.0)
    
    x_sum = 0.0
    y_sum = 0.0
    
    for pitch_class in range(12):
        weight = chroma_vector[pitch_class]
        if weight > 0:
            # Compute coordinates relative to the key root (tonal center)
            coords = pitch_to_tonnetz(pitch_class, key_root)
            x_sum += weight * coords[0]
            y_sum += weight * coords[1]
    
    return (x_sum / total_weight, y_sum / total_weight)


def chroma_to_tonnetz_tension(chroma_vector: np.ndarray, key_root: int) -> Tuple[float, float, float]:
    """
    Computes the weighted centroid AND tension magnitude of a chroma vector.
    
    The centroid can be misleading when notes cancel out (e.g., two notes in
    opposite directions from the tonal center). The tension magnitude (mean
    distance from center) captures the true harmonic tension regardless of
    cancellation.
    
    Args:
        chroma_vector: 12-element array of pitch class weights
        key_root: Root of the local key (0-11, where 0=C)
    
    Returns:
        Tuple[float, float, float]: (centroid_x, centroid_y, tension_magnitude)
            - centroid_x, centroid_y: weighted mean position (can cancel out)
            - tension_magnitude: weighted mean distance from origin (never cancels)
    """
    total_weight = np.sum(chroma_vector)
    if total_weight == 0:
        return (0.0, 0.0, 0.0)
    
    x_sum = 0.0
    y_sum = 0.0
    dist_sum = 0.0
    
    for pitch_class in range(12):
        weight = chroma_vector[pitch_class]
        if weight > 0:
            # Compute coordinates relative to the key root (tonal center)
            coords = pitch_to_tonnetz(pitch_class, key_root)
            x, y = coords[0], coords[1]
            
            x_sum += weight * x
            y_sum += weight * y
            
            # Distance from origin (tonal center)
            dist = math.sqrt(x * x + y * y)
            dist_sum += weight * dist
    
    centroid_x = x_sum / total_weight
    centroid_y = y_sum / total_weight
    tension_magnitude = dist_sum / total_weight
    
    return (centroid_x, centroid_y, tension_magnitude)


def get_key_tonnetz_position(key_root: int) -> Tuple[float, float]:
    """
    Get the absolute Tonnetz position of a key root (relative to C).
    
    Args:
        key_root: The key root (0-11, where 0=C)
    
    Returns:
        Tuple[float, float]: (x, y) position of the key on the C-centered Tonnetz
    """
    return TONNETZ_COORDS[key_root]


def get_key_profile(root, mode):
    """Rotates the base profile to the target root."""
    base = MAJOR_PROFILE if mode == 'major' else MINOR_PROFILE
    return np.roll(base, root)

def estimate_key(chroma_vector):
    """
    Correlates a chroma vector against all 24 key profiles.
    Returns: (root_index, mode_string)
    """
    if np.sum(chroma_vector) == 0:
        return None # Silence
        
    best_corr = -1
    best_key = None
    
    # Normalize the input vector
    chroma_vector = chroma_vector / np.max(chroma_vector) if np.max(chroma_vector) > 0 else chroma_vector

    for root in range(12):
        for mode in ['major', 'minor']:
            profile = get_key_profile(root, mode)
            corr, _ = pearsonr(chroma_vector, profile)
            if corr > best_corr:
                best_corr = corr
                best_key = (root, mode)
    return best_key

def get_circle_of_fifths_angle(chroma_vector):
    """
    Maps a chroma vector to a single angle on the Circle of Fifths.
    Returns the angle in radians.
    """
    # X and Y coordinates for the "center of gravity" on the circle
    x, y = 0.0, 0.0
    total_weight = np.sum(chroma_vector)
    
    if total_weight == 0:
        return None

    for pitch_class, weight in enumerate(chroma_vector):
        # Map pitch class to Circle of Fifths Angle
        # C=0 -> 0 rads. G=7 -> 1 step. 
        # Angle = (pitch * 7) * (2pi / 12)
        angle = (pitch_class * 7) * (2 * np.pi / 12)
        x += weight * np.cos(angle)
        y += weight * np.sin(angle)
    
    # Calculate the angle of the resulting vector
    return np.arctan2(y, x)

# --- 2. MAIN PROCESSING LOOP ---

def analyze_tension(midi_path, fs=10):
    """
    fs: Sampling frequency (10 means 10 samples per second, i.e., 0.1s steps)
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI: {e}")
        return

    # Get chroma matrix (Shape: 12 x Total_Time_Steps)
    chroma_matrix = pm.get_chroma(fs=fs)
    total_steps = chroma_matrix.shape[1]
    
    tension_curve = []
    keys_detected = []
    
    # Define window sizes in 'steps' (fs=10 -> 10 steps = 1 second)
    context_window_rad = 20  # +/- 2 seconds for Key Detection
    focus_window_rad = 10     # +/- 1.0 seconds for Current Tension
    
    print(f"Processing {total_steps} time steps...")

    for t in range(total_steps):
        # 1. Define Context Window (handle edges)
        start_ctx = max(0, t - context_window_rad)
        end_ctx = min(total_steps, t + context_window_rad)
        
        # Sum chroma over context window to get stable key profile
        context_chroma = np.sum(chroma_matrix[:, start_ctx:end_ctx], axis=1)
        
        # 2. Estimate Local Key
        local_key = estimate_key(context_chroma)
        
        if local_key is None:
            tension_curve.append(0)
            continue
            
        key_root, key_mode = local_key
        
        # 3. Define Focus Window (Current Note content)
        start_focus = max(0, t - focus_window_rad)
        end_focus = min(total_steps, t + focus_window_rad)
        focus_chroma = np.sum(chroma_matrix[:, start_focus:end_focus], axis=1)

        # 4. Calculate Distance (Angular distance on Circle of Fifths)
        # Angle of the Key
        # (We multiply root by 7 because 1 semitone = 7 steps on the circle)
        key_angle = (key_root * 7) * (2 * np.pi / 12) 
        
        # Angle of the Current Content
        current_angle = get_circle_of_fifths_angle(focus_chroma)
        
        if current_angle is None:
            tension_curve.append(0) # Silence
        else:
            # Angular difference (shortest path)
            diff = abs(current_angle - key_angle)
            if diff > np.pi:
                diff = (2 * np.pi) - diff
            
            # Normalize to 0.0 - 1.0 (Pi is max distance)
            tension = diff / np.pi 
            tension_curve.append(tension)

    return np.array(tension_curve), fs


def get_active_pitches(chroma_vector: np.ndarray, threshold_ratio: float = 0.3) -> List[int]:
    """
    Extract active pitch classes from a chroma vector.
    
    Args:
        chroma_vector: 12-element array of pitch class weights
        threshold_ratio: Minimum ratio of max value to be considered active
    
    Returns:
        List of active pitch classes (0-11)
    """
    if np.max(chroma_vector) == 0:
        return []
    
    threshold = np.max(chroma_vector) * threshold_ratio
    active = [i for i in range(12) if chroma_vector[i] >= threshold]
    return active


# --- ROBUST HARMONIC REGION EXTRACTION ---

def ema_chroma(chroma_seq: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """
    Apply exponential moving average smoothing to a chroma sequence.
    
    Args:
        chroma_seq: Array of shape (N, 12) with chroma vectors per frame
        alpha: Smoothing factor (0 < alpha <= 1). Lower = smoother.
    
    Returns:
        Smoothed chroma sequence of same shape
    """
    if len(chroma_seq) == 0:
        return chroma_seq
    
    smoothed = np.zeros_like(chroma_seq, dtype=float)
    smoothed[0] = chroma_seq[0]
    
    for t in range(1, len(chroma_seq)):
        smoothed[t] = alpha * chroma_seq[t] + (1 - alpha) * smoothed[t - 1]
    
    return smoothed


def select_harmonic_region(
    chroma_vec: np.ndarray, 
    energy_threshold: float = 0.90, 
    k_max: int = 5, 
    min_k: int = 2
) -> List[int]:
    """
    Select the harmonic region pitch classes from a chroma vector.
    
    Uses cumulative energy threshold capped by k_max to select the most
    salient pitch classes without including weak melody/ornament notes.
    
    Args:
        chroma_vec: 12-element array of pitch class weights
        energy_threshold: Keep PCs until this fraction of total energy (0.85-0.95)
        k_max: Maximum number of pitch classes to return
        min_k: Minimum number of pitch classes to return (if available)
    
    Returns:
        List of pitch classes (0-11) forming the harmonic region
    """
    total_energy = np.sum(chroma_vec)
    if total_energy == 0:
        return []
    
    # Sort pitch classes by energy (descending)
    sorted_pcs = np.argsort(chroma_vec)[::-1]
    
    selected = []
    cumulative_energy = 0.0
    
    for pc in sorted_pcs:
        if chroma_vec[pc] <= 0:
            break
        
        selected.append(int(pc))
        cumulative_energy += chroma_vec[pc]
        
        # Stop if we've captured enough energy AND have at least min_k
        if len(selected) >= min_k and cumulative_energy / total_energy >= energy_threshold:
            break
        
        # Hard cap at k_max
        if len(selected) >= k_max:
            break
    
    return selected


def median_filter_pitch_sets(
    pitch_sets: List[List[int]], 
    window_size: int = 5
) -> List[List[int]]:
    """
    Apply median-like smoothing to a sequence of pitch sets using voting.
    
    For each frame, looks at surrounding frames and keeps pitch classes
    that appear in at least half of the window.
    
    Args:
        pitch_sets: List of pitch class lists per frame
        window_size: Size of the smoothing window (should be odd)
    
    Returns:
        Smoothed pitch sets per frame
    """
    if len(pitch_sets) == 0:
        return pitch_sets
    
    half_win = window_size // 2
    n_frames = len(pitch_sets)
    smoothed = []
    
    for t in range(n_frames):
        # Collect votes from window
        votes = np.zeros(12, dtype=int)
        
        start = max(0, t - half_win)
        end = min(n_frames, t + half_win + 1)
        window_frames = end - start
        
        for frame_idx in range(start, end):
            for pc in pitch_sets[frame_idx]:
                votes[pc] += 1
        
        # Keep pitch classes that appear in at least half of the window
        threshold = window_frames / 2.0
        result = [pc for pc in range(12) if votes[pc] >= threshold]
        
        # If result is empty but current frame has notes, keep strongest from current
        if len(result) == 0 and len(pitch_sets[t]) > 0:
            result = pitch_sets[t][:3]  # Keep up to 3 from original
        
        smoothed.append(result)
    
    return smoothed


def extract_robust_harmonic_regions(
    chroma_matrix: np.ndarray,
    fs: int,
    focus_window_sec: float = 0.5,
    ema_alpha: float = 0.25,
    energy_threshold: float = 0.90,
    k_max: int = 5,
    median_window: int = 5
) -> List[List[int]]:
    """
    Extract robust harmonic region pitch sets from a chroma matrix.
    
    Applies:
    1. Focus window aggregation
    2. EMA smoothing on chroma
    3. Energy-based pitch selection (capped by k_max)
    4. Median filtering on pitch sets
    
    Args:
        chroma_matrix: Shape (12, T) chroma features
        fs: Sample rate of chroma
        focus_window_sec: Focus window size in seconds
        ema_alpha: EMA smoothing factor
        energy_threshold: Cumulative energy threshold for pitch selection
        k_max: Maximum pitch classes per frame
        median_window: Window size for median filtering
    
    Returns:
        List of pitch class lists per frame
    """
    total_steps = chroma_matrix.shape[1]
    focus_window_rad = int(focus_window_sec * fs / 2)
    
    # Step 1: Compute focus-window aggregated chroma per frame
    focus_chromas = []
    for t in range(total_steps):
        start = max(0, t - focus_window_rad)
        end = min(total_steps, t + focus_window_rad)
        focus_chroma = np.sum(chroma_matrix[:, start:end], axis=1)
        focus_chromas.append(focus_chroma)
    
    focus_chromas = np.array(focus_chromas)  # Shape: (T, 12)
    
    # Step 2: Apply EMA smoothing on chroma
    smoothed_chromas = ema_chroma(focus_chromas, alpha=ema_alpha)
    
    # Step 3: Select harmonic region for each frame
    raw_regions = []
    for t in range(total_steps):
        region = select_harmonic_region(
            smoothed_chromas[t], 
            energy_threshold=energy_threshold,
            k_max=k_max
        )
        raw_regions.append(region)
    
    # Step 4: Apply median filtering on pitch sets
    final_regions = median_filter_pitch_sets(raw_regions, window_size=median_window)
    
    return final_regions


def extract_chord_aware_regions(
    pm: 'pretty_midi.PrettyMIDI',
    fs: int = 10,
    window_sec: float = 1.0,
    min_note_duration: float = 0.05,
    chord_tolerance_sec: float = 0.15,
    repetition_bonus: float = 1.5,
    chord_bonus: float = 6.0,
    bass_boost: float = 2.0,
    k_max: int = 5,
    ema_alpha: float = 0.15,
    median_window: int = 7
) -> List[List[int]]:
    """
    Extract harmonic regions from MIDI using chord-aware note analysis.
    
    Key principles:
    1. PENALIZE very short notes (likely grace notes / passing tones)
    2. PRIORITIZE chords (notes attacked simultaneously or near-simultaneously)
    3. PRIORITIZE repeated notes (notes that appear multiple times in window)
    4. PRIORITIZE lower notes (bass defines harmony more than treble)
    5. Use longer windows and heavy smoothing for stability (no sudden jumps)
    
    Args:
        pm: PrettyMIDI object
        fs: Output sample rate (frames per second)
        window_sec: Analysis window size in seconds (longer = more stable)
        min_note_duration: Notes shorter than this (seconds) are heavily penalized
        chord_tolerance_sec: Notes within this time are considered "simultaneous"
        repetition_bonus: Weight multiplier for each additional occurrence
        chord_bonus: Weight multiplier for notes in chord clusters
        bass_boost: Extra weight for low notes (1.0 = no boost, 2.0 = bass gets 2x)
        k_max: Maximum pitch classes per frame
        ema_alpha: EMA smoothing factor (lower = smoother)
        median_window: Window size for median filtering
    
    Returns:
        List of pitch class lists per frame
    """
    # Get total duration and create time grid
    end_time = pm.get_end_time()
    total_frames = int(end_time * fs) + 1
    
    # Collect all notes from all instruments
    all_notes = []
    for instrument in pm.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                all_notes.append({
                    'pitch': note.pitch,
                    'pc': note.pitch % 12,  # pitch class
                    'start': note.start,
                    'end': note.end,
                    'duration': note.end - note.start,
                    'velocity': note.velocity
                })
    
    if len(all_notes) == 0:
        return [[] for _ in range(total_frames)]
    
    # Sort by start time for efficient windowing
    all_notes.sort(key=lambda n: n['start'])
    
    # --- PRECOMPUTE CHORD CLUSTERS ---
    # Group notes by onset time (within chord_tolerance)
    onset_clusters = []
    current_cluster = []
    cluster_start = -999.0
    
    for note in all_notes:
        if note['start'] - cluster_start <= chord_tolerance_sec:
            current_cluster.append(note)
        else:
            if current_cluster:
                onset_clusters.append(current_cluster)
            current_cluster = [note]
            cluster_start = note['start']
    if current_cluster:
        onset_clusters.append(current_cluster)
    
    # Mark notes that are part of chords (cluster size >= 2)
    chord_notes = set()
    for cluster in onset_clusters:
        if len(cluster) >= 2:
            for note in cluster:
                chord_notes.add(id(note))
    
    # --- COMPUTE WEIGHTED PITCH CLASS SCORES PER FRAME ---
    half_window = window_sec / 2.0
    pc_scores_per_frame = []
    
    for frame_idx in range(total_frames):
        frame_time = frame_idx / fs
        window_start = frame_time - half_window
        window_end = frame_time + half_window
        
        # Score each pitch class
        pc_scores = np.zeros(12, dtype=float)
        pc_counts = np.zeros(12, dtype=int)  # For repetition tracking
        
        for note in all_notes:
            # Check if note overlaps with window
            # Note is relevant if it sounds during window OR starts in window
            note_in_window = (
                (note['start'] < window_end and note['end'] > window_start) or
                (window_start <= note['start'] < window_end)
            )
            
            if not note_in_window:
                # Early exit if we've passed the window (notes sorted by start)
                if note['start'] > window_end:
                    break
                continue
            
            pc = note['pc']
            duration = note['duration']
            
            # --- BASE WEIGHT: duration-based (penalize very short notes) ---
            if duration < min_note_duration:
                # Heavy penalty for very short notes (grace notes, etc.)
                dur_weight = 0.1
            else:
                # Soft saturation: longer notes get more weight, but with diminishing returns
                # Cap the benefit at ~0.5 seconds to avoid over-weighting held notes
                dur_weight = min(1.0, duration / 0.3)
            
            # --- CHORD BONUS: notes in chords get boosted ---
            is_chord = id(note) in chord_notes
            chord_mult = chord_bonus if is_chord else 1.0
            
            # --- BASS WEIGHT: lower notes define harmony more than treble ---
            # MIDI pitch 36 (C2) = full bass_boost, pitch 84 (C6) = 1.0, higher = less
            # Linear interpolation: bass_weight = bass_boost at low, 1.0 at mid, 0.5 at high
            pitch = note['pitch']
            if pitch <= 48:  # C3 and below: full bass boost
                bass_weight = bass_boost
            elif pitch >= 72:  # C5 and above: reduced weight (melody range)
                bass_weight = max(0.5, 1.0 - (pitch - 72) / 48.0)
            else:  # Middle range: linear interpolation from bass_boost to 1.0
                bass_weight = bass_boost - (bass_boost - 1.0) * (pitch - 48) / 24.0
            
            # --- VELOCITY (optional soft factor) ---
            vel_weight = note['velocity'] / 127.0
            
            # Combine weights
            weight = dur_weight * chord_mult * bass_weight * vel_weight
            pc_scores[pc] += weight
            pc_counts[pc] += 1
        
        # --- REPETITION BONUS: multiply by (1 + bonus * (count - 1)) ---
        for pc in range(12):
            if pc_counts[pc] > 1:
                rep_mult = 1.0 + repetition_bonus * (pc_counts[pc] - 1)
                pc_scores[pc] *= rep_mult
        
        pc_scores_per_frame.append(pc_scores)
    
    pc_scores_matrix = np.array(pc_scores_per_frame)  # Shape: (T, 12)
    
    # --- APPLY EMA SMOOTHING ---
    smoothed_scores = ema_chroma(pc_scores_matrix, alpha=ema_alpha)
    
    # --- SELECT TOP PITCH CLASSES PER FRAME ---
    raw_regions = []
    for t in range(total_frames):
        region = select_harmonic_region(
            smoothed_scores[t],
            energy_threshold=0.85,
            k_max=k_max,
            min_k=2
        )
        raw_regions.append(region)
    
    # --- APPLY MEDIAN FILTERING ---
    final_regions = median_filter_pitch_sets(raw_regions, window_size=median_window)
    
    return final_regions


def analyze_tension_2d(
    midi_path: str, 
    fs: int = 10,
    focus_window_sec: float = 1.0,
    region_window_sec: float = 1.5,
    ema_alpha: float = 0.15,
    k_max: int = 5,
    median_window: int = 9
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, str], List[List[int]], int]:
    """
    Analyzes harmonic tension using 2D Tonnetz coordinates with chord-aware region detection.
    
    The KEY is detected globally from the entire piece (stable, represents tonality).
    The TENSION is computed per-frame relative to this global key.
    The HARMONIC REGION uses chord-aware analysis (prioritizes chords, penalizes short notes).
    
    Args:
        midi_path: Path to the MIDI file
        fs: Sampling frequency (samples per second)
        focus_window_sec: Focus window for tension computation (seconds)
        region_window_sec: Longer window for chord region detection (seconds)
        ema_alpha: EMA smoothing factor (lower = smoother, 0.1-0.2 recommended)
        k_max: Maximum pitch classes in harmonic region (4-6 recommended)
        median_window: Window size for median filtering pitch sets (7-11 for stability)
    
    Returns:
        Tuple containing:
            - tension_xy: np.ndarray of shape (N, 2) with [x, y] Tonnetz coordinates (centroid)
            - tension_magnitude: np.ndarray of shape (N,) with tension magnitudes (mean distance)
            - global_key: Single (root, mode) tuple for the entire piece
            - region_pitches: Chord-aware harmonic region pitch classes per frame
            - fs: The sampling frequency used
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI: {e}")
        return None, None, None, None, fs

    # Get chroma matrix (Shape: 12 x Total_Time_Steps)
    chroma_matrix = pm.get_chroma(fs=fs)
    total_steps = chroma_matrix.shape[1]
    
    # --- GLOBAL KEY DETECTION ---
    # Sum all chroma over the entire piece to get the global pitch distribution
    global_chroma = np.sum(chroma_matrix, axis=1)
    global_key = estimate_key(global_chroma)
    
    if global_key is None:
        global_key = (0, 'major')  # Default to C major if empty
    
    key_root, key_mode = global_key
    print(f"Detected Global Key: {PITCH_NAMES[key_root]} {key_mode}")
    
    # Convert focus window size to steps
    focus_window_rad = int(focus_window_sec * fs / 2)
    
    print(f"Processing {total_steps} time steps (fs={fs}, duration={total_steps/fs:.1f}s)...")
    print(f"Tension window: {focus_window_sec}s | Region window: {region_window_sec}s | EMA α={ema_alpha}")

    # --- TENSION COMPUTATION (centroid + magnitude) ---
    tension_xy = []
    tension_magnitude = []
    for t in range(total_steps):
        # Define Focus Window (Current harmony content)
        start_focus = max(0, t - focus_window_rad)
        end_focus = min(total_steps, t + focus_window_rad)
        focus_chroma = np.sum(chroma_matrix[:, start_focus:end_focus], axis=1)

        # Calculate 2D Tonnetz Centroid AND tension magnitude
        cx, cy, mag = chroma_to_tonnetz_tension(focus_chroma, key_root)
        tension_xy.append([cx, cy])
        tension_magnitude.append(mag)
    
    # --- CHORD-AWARE HARMONIC REGION EXTRACTION ---
    # Uses note-level analysis: penalizes short notes, prioritizes chords & repetition
    region_pitches = extract_chord_aware_regions(
        pm=pm,
        fs=fs,
        window_sec=region_window_sec,
        min_note_duration=0.05,       # Penalize notes < 50ms
        chord_tolerance_sec=0.15,     # Notes within 80ms = chord
        repetition_bonus=1.5,         # Bonus for repeated notes
        chord_bonus=6.0,              # Bonus for chord notes
        k_max=k_max,
        ema_alpha=ema_alpha,
        median_window=median_window
    )
    
    print(f"Chord-aware region extraction complete (avg region size: {np.mean([len(r) for r in region_pitches]):.1f} PCs)")
    print(f"Tension magnitude range: [{min(tension_magnitude):.2f}, {max(tension_magnitude):.2f}]")

    return np.array(tension_xy), np.array(tension_magnitude), global_key, region_pitches, fs


def format_key_name(key_root: int, key_mode: str) -> str:
    """Formats a key as a readable string (e.g., 'C major', 'F# minor')."""
    return f"{PITCH_NAMES[key_root]} {key_mode}"


def extract_harmony_for_tokens(
    midi_path: str,
    harmony_interval_ms: int = 500,
    x_range: Tuple[float, float] = (-3.0, 3.0),
    y_range: Tuple[float, float] = (-2.0, 2.0),
    r_max: float = 3.0,
    n_bins: int = 128
) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Extract harmony features for token-based training data.
    
    Returns harmony events at regular time intervals, quantized to discrete bins
    suitable for insertion into a token stream.
    
    Args:
        midi_path: Path to the MIDI file
        harmony_interval_ms: Time between harmony events in milliseconds
        x_range: (min, max) range for x coordinate (fifths axis)
        y_range: (min, max) range for y coordinate (thirds axis)
        r_max: Maximum tension magnitude
        n_bins: Number of bins for quantization (0 to n_bins-1), default 128
    
    Returns:
        List of (time_ms, x_bin, y_bin, r_bin, key_root) tuples, or None if error
        - time_ms: absolute time in milliseconds
        - x_bin, y_bin, r_bin: quantized values in range [0, n_bins-1]
        - key_root: root of the key (0-11) major (12-23) minor, 24 for unknown
    """
    # Use low sample rate since we only need samples at harmony_interval_ms
    fs = max(2, 1000 // harmony_interval_ms)  # e.g., 2 Hz for 500ms intervals
    
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        return None
    
    # Get chroma and key
    chroma_matrix = pm.get_chroma(fs=fs)
    total_steps = chroma_matrix.shape[1]
    
    if total_steps == 0:
        return None
    
    global_chroma = np.sum(chroma_matrix, axis=1)
    global_key = estimate_key(global_chroma)
    
    if global_key is None:
        global_key = (0, 'major')
    
    key_root = global_key[0]
    
    # Compute tension at each frame
    focus_window_rad = max(1, fs // 2)  # ~0.5s window
    
    harmony_events = []
    
    for t in range(total_steps):
        # Focus window
        start = max(0, t - focus_window_rad)
        end = min(total_steps, t + focus_window_rad)
        focus_chroma = np.sum(chroma_matrix[:, start:end], axis=1)
        
        # Get centroid and magnitude
        cx, cy, mag = chroma_to_tonnetz_tension(focus_chroma, key_root)
        
        # Quantize to bins
        x_bin = int(np.clip((cx - x_range[0]) / (x_range[1] - x_range[0]) * n_bins, 0, n_bins - 1))
        y_bin = int(np.clip((cy - y_range[0]) / (y_range[1] - y_range[0]) * n_bins, 0, n_bins - 1))
        r_bin = int(np.clip(mag / r_max * n_bins, 0, n_bins - 1))
        
        # Time in milliseconds
        time_ms = int(t * 1000 / fs)

        key_root = global_key[0] if global_key is not None else 24
        if key_root != 24 and global_key[1] == 'minor':
            key_root += 12
        
        harmony_events.append((time_ms, x_bin, y_bin, r_bin, key_root))
    
    return harmony_events


# --- 3. VISUALIZATION ---

def plot_tension(midi_path):
    tension, fs = analyze_tension(midi_path)
    
    if tension is None: return

    time_axis = np.arange(len(tension)) / fs
    
    plt.figure(figsize=(14, 6))
    
    # Plot the curve
    plt.plot(time_axis, tension, label='Harmonic Distance (Tension)', color='#2c3e50', linewidth=1.5)
    
    # Fill under the curve for better visual
    plt.fill_between(time_axis, tension, color='#3498db', alpha=0.3)
    
    plt.title(f"Tonal Tension Curve: {midi_path}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Distance from Local Tonal Center (0=Stable, 1=Far)")
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Add some threshold lines
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='High Tension Threshold')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- 4. MAIN ENTRY POINT ---

def extract_and_visualize(
    fs: int = 10,
    focus_window_sec: float = 1.0,
    region_window_sec: float = 1.5,
    animated: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, str], List[List[int]], int]]:
    """
    Main entry point: Extract 2D Tonnetz features from a MIDI file and visualize.
    
    Args:
        fs: Sampling frequency (samples per second)
        focus_window_sec: Focus window for tension computation (seconds)
        region_window_sec: Longer window for chord region detection (seconds)
        animated: If True, show animated visualization
    
    Returns:
        Tuple of (tension_xy, tension_magnitude, global_key, region_pitches, fs) or None if error
    """
    from tonnetz import visualize_tonnetz
    
    print(f"\n{'='*60}")
    print(f"Harmonic Tension Extractor - Chord-Aware Tonnetz Analysis")
    print(f"{'='*60}")
    print(f"MIDI File: {midi_path}")
    print(f"Sample Rate: {fs} Hz")
    print(f"Tension Window: {focus_window_sec}s | Region Window: {region_window_sec}s")
    print(f"{'='*60}\n")
    
    # Extract features with chord-aware region detection
    result = analyze_tension_2d(
        midi_path=midi_path,
        fs=fs,
        focus_window_sec=focus_window_sec,
        region_window_sec=region_window_sec
    )
    
    if result[0] is None:
        print("Error: Could not extract features from MIDI file.")
        return None
    
    tension_xy, tension_magnitude, global_key, region_pitches, fs = result
    
    # Print summary statistics
    print(f"\n--- Feature Summary ---")
    print(f"Global Key: {format_key_name(*global_key)}")
    print(f"Total frames: {len(tension_xy)}")
    print(f"Duration: {len(tension_xy) / fs:.1f} seconds")
    print(f"Centroid X range: [{tension_xy[:, 0].min():.2f}, {tension_xy[:, 0].max():.2f}]")
    print(f"Centroid Y range: [{tension_xy[:, 1].min():.2f}, {tension_xy[:, 1].max():.2f}]")
    print(f"Tension magnitude range: [{tension_magnitude.min():.2f}, {tension_magnitude.max():.2f}]")
    
    print(f"\n--- Launching Tonnetz Visualizer ---\n")
    
    # Extract filename for title
    import os
    filename = os.path.basename(midi_path)
    title = f"Tonnetz: {filename}"
    
    # Visualize with MIDI playback, chord shapes, and tension magnitude
    visualize_tonnetz(
        tension_xy=tension_xy,
        tension_magnitude=tension_magnitude,
        global_key=global_key,
        active_pitches=region_pitches,
        fs=fs,
        title=title,
        animated=animated,
        save_path=None,  # Don't save, just play
        midi_path=midi_path,
        play_midi=True,
        mouse_indicator_enabled=True,
        mouse_indicator_spread=0.3,
    )
    
    return tension_xy, tension_magnitude, global_key, region_pitches, fs


def notes_to_chroma(
    notes: List[dict],
    current_time: float,
    window_sec: float = 2.0
) -> np.ndarray:
    """
    Convert a list of recent notes to a chroma vector.
    
    Args:
        notes: List of note dictionaries with keys:
               - 'pitch': MIDI pitch (0-127)
               - 'start': start time in seconds
               - 'end': end time in seconds (or None if still active)
               - 'velocity': velocity (0-127)
        current_time: Current time in seconds
        window_sec: Time window to consider (seconds before current_time)
    
    Returns:
        12-element chroma vector
    """
    chroma = np.zeros(12, dtype=float)
    window_start = current_time - window_sec
    
    for note in notes:
        note_start = note['start']
        note_end = note['end'] if note.get('end') is not None else current_time  # If still active, use current time
        
        # Check if note overlaps with window
        if note_end > window_start and note_start < current_time:
            # Calculate overlap duration
            overlap_start = max(note_start, window_start)
            overlap_end = min(note_end, current_time)
            duration = overlap_end - overlap_start
            
            # Weight by duration and velocity
            pitch_class = note['pitch'] % 12
            velocity = note.get('velocity', 64) / 127.0
            weight = duration * velocity
            
            chroma[pitch_class] += weight
    
    return chroma


class RealtimeHarmonyExtractor:
    """
    Real-time harmony feature extraction and visualization.
    
    Processes a rolling buffer of recent notes and extracts:
    - Global key (updated periodically)
    - Current tension (x, y, magnitude)
    - Current harmonic region
    
    Usage:
        extractor = RealtimeHarmonyExtractor()
        
        # Add notes as they occur
        extractor.add_note(pitch=60, start_time=0.0, velocity=80)
        extractor.note_off(pitch=60, end_time=0.5)
        
        # Update visualization
        extractor.update(current_time=0.5)
    """
    
    def __init__(
        self,
        key_window_sec: float = 4.0,
        tension_window_sec: float = 1.0,
        key_update_interval_sec: float = 2.0,
        buffer_size_sec: float = 10.0,
        visualize: bool = True,
        fs: int = 10
    ):
        """
        Initialize the real-time harmony extractor.
        
        Args:
            key_window_sec: Window size for key detection (seconds)
            tension_window_sec: Window size for tension computation (seconds)
            key_update_interval_sec: How often to update key detection (seconds)
            buffer_size_sec: How long to keep notes in buffer (seconds)
            visualize: If True, enable real-time visualization
            fs: Sampling frequency for visualization data storage
        """
        self.key_window_sec = key_window_sec
        self.tension_window_sec = tension_window_sec
        self.key_update_interval_sec = key_update_interval_sec
        self.buffer_size_sec = buffer_size_sec
        self.fs = fs
        
        # Note buffer: list of {pitch, start, end, velocity}
        self.notes: List[dict] = []
        
        # Current state
        self.current_key: Tuple[int, str] = (0, 'major')  # Default to C major
        self.last_key_update: float = 0.0
        
        # Visualization data (for trail history)
        self.tension_history: List[Tuple[float, float]] = []
        self.magnitude_history: List[float] = []
        self.time_history: List[float] = []
        
        # Visualization setup
        self.visualize_enabled = visualize
        if self.visualize_enabled:
            self._setup_visualization()
    
    def _setup_visualization(self):
        """Setup matplotlib for real-time plotting."""
        import matplotlib.pyplot as plt
        plt.ion()  # Enable interactive mode
        plt.style.use('dark_background')
        
        self.fig, self.ax = plt.subplots(figsize=(12, 10), facecolor='#1a1a2e')
        self.ax.set_facecolor('#16213e')
        
        # Set fixed limits
        grid_min, grid_max = -3, 3
        from tonnetz import lattice_to_display
        
        corners = [
            lattice_to_display(grid_min, grid_min),
            lattice_to_display(grid_min, grid_max),
            lattice_to_display(grid_max, grid_min),
            lattice_to_display(grid_max, grid_max),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        padding = 0.8
        self.ax.set_xlim(min(xs) - padding, max(xs) + padding)
        self.ax.set_ylim(min(ys) - padding, max(ys) + padding)
        
        self.ax.set_xlabel('Circle of Fifths (← Subdominant | Dominant →)', 
                          fontsize=12, color='#e8e8e8', fontfamily='monospace')
        self.ax.set_ylabel('Major Thirds (↓ Minor | Major ↑)', 
                          fontsize=12, color='#e8e8e8', fontfamily='monospace')
        self.ax.set_title('Real-Time Harmonic Tension', fontsize=14, 
                         color='#f8f8f8', fontfamily='monospace', pad=20)
        
        self.ax.set_aspect('equal')
        self.ax.grid(False)
        self.ax.tick_params(colors='#a0a0a0', labelsize=9)
        
        for spine in self.ax.spines.values():
            spine.set_color('#4a4a6a')
        
        # Draw static Tonnetz grid
        self._draw_tonnetz_grid()
        
        # Initialize plot elements
        self.current_point, = self.ax.plot(
            [], [], 'o', markersize=18, 
            color='#ff9f43', markeredgecolor='#fff', 
            markeredgewidth=2, zorder=10
        )
        
        self.inner_point, = self.ax.plot(
            [], [], 'o', markersize=8,
            color='#fff', zorder=11
        )
        
        self.trail_line, = self.ax.plot(
            [], [], '-', linewidth=2, color='#ff9f43', 
            alpha=0.6, zorder=5
        )
        
        # Tension magnitude circle
        from matplotlib.patches import Circle
        self.tension_circle = Circle(
            (0, 0), 0.0,
            fill=False, edgecolor='#ff6b6b', linewidth=2.5,
            linestyle='--', alpha=0.8, zorder=6
        )
        self.ax.add_patch(self.tension_circle)
        
        # Text displays
        self.key_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            fontsize=14, color='#ffd700', fontfamily='monospace',
            fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', 
                     edgecolor='#ffd700', alpha=0.8)
        )
        
        self.coord_text = self.ax.text(
            0.02, 0.02, '', transform=self.ax.transAxes,
            fontsize=10, color='#7a7a9a', fontfamily='monospace',
            va='bottom'
        )
        
        self.time_text = self.ax.text(
            0.98, 0.98, '', transform=self.ax.transAxes,
            fontsize=11, color='#a0a0a0', fontfamily='monospace',
            ha='right', va='top'
        )
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)
    
    def _draw_tonnetz_grid(self):
        """Draw the static Tonnetz grid (C at origin)."""
        from tonnetz import lattice_to_display, get_pitch_at_tonnetz_coord
        from matplotlib.patches import Circle
        
        grid_min, grid_max = -3, 3
        
        # Draw lattice lines
        line_color = '#3a3a5a'
        line_alpha = 0.3
        
        # Horizontal lines
        for y in range(grid_min, grid_max + 1):
            x1, y1 = lattice_to_display(grid_min, y)
            x2, y2 = lattice_to_display(grid_max, y)
            self.ax.plot([x1, x2], [y1, y2], '-', color=line_color, 
                        alpha=line_alpha, linewidth=0.5, zorder=1)
        
        # Vertical lines
        for x in range(grid_min, grid_max + 1):
            x1, y1 = lattice_to_display(x, grid_min)
            x2, y2 = lattice_to_display(x, grid_max)
            self.ax.plot([x1, x2], [y1, y2], '-', color=line_color, 
                        alpha=line_alpha, linewidth=0.5, zorder=1)
        
        # Diagonal lines
        for d in range(2 * grid_min, 2 * grid_max + 1):
            points = []
            for x in range(grid_min, grid_max + 1):
                y = d - x
                if grid_min <= y <= grid_max:
                    points.append(lattice_to_display(x, y))
            if len(points) >= 2:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                self.ax.plot(xs, ys, '-', color=line_color, 
                            alpha=line_alpha, linewidth=0.5, zorder=1)
        
        # Draw pitch nodes
        for x in range(grid_min, grid_max + 1):
            for y in range(grid_min, grid_max + 1):
                pitch_name = get_pitch_at_tonnetz_coord(x, y, key_root=0)
                X, Y = lattice_to_display(float(x), float(y))
                
                # Color coding
                if x == 0 and y == 0:
                    color = '#ffd700'  # Gold for C
                    fontweight = 'bold'
                    fontsize = 11
                elif x == 1 and y == 0:
                    color = '#ff6b6b'  # Red for G
                    fontweight = 'normal'
                    fontsize = 10
                elif x == -1 and y == 0:
                    color = '#4ecdc4'  # Teal for F
                    fontweight = 'normal'
                    fontsize = 10
                elif y == 1:
                    color = '#95e1d3'  # Light green for major third
                    fontweight = 'normal'
                    fontsize = 9
                elif y == -1:
                    color = '#f38181'  # Coral for minor
                    fontweight = 'normal'
                    fontsize = 9
                else:
                    color = '#7a7a9a'
                    fontweight = 'normal'
                    fontsize = 9
                
                circle = Circle((X, Y), 0.08, color=color, alpha=0.3)
                self.ax.add_patch(circle)
                
                self.ax.text(
                    X, Y + 0.18, pitch_name,
                    ha='center', va='bottom',
                    fontsize=fontsize, fontweight=fontweight,
                    color=color, fontfamily='monospace'
                )
    
    def add_note(self, pitch: int, start_time: float, velocity: int = 64):
        """
        Add a note-on event.
        
        Args:
            pitch: MIDI pitch (0-127)
            start_time: Start time in seconds
            velocity: MIDI velocity (0-127)
        """
        self.notes.append({
            'pitch': pitch,
            'start': start_time,
            'end': None,  # Will be set on note-off
            'velocity': velocity
        })
    
    def note_off(self, pitch: int, end_time: float):
        """
        Add a note-off event.
        
        Args:
            pitch: MIDI pitch (0-127)
            end_time: End time in seconds
        """
        # Find the most recent active note with this pitch
        for note in reversed(self.notes):
            if note['pitch'] == pitch and note['end'] is None:
                note['end'] = end_time
                break
    
    def _clean_buffer(self, current_time: float):
        """Remove old notes from buffer."""
        cutoff_time = current_time - self.buffer_size_sec
        # Keep notes that either have no end time (still active) or ended after cutoff
        self.notes = [n for n in self.notes if (n['end'] if n.get('end') is not None else current_time) > cutoff_time]
    
    def _update_key(self, current_time: float) -> bool:
        """
        Update key detection if enough time has passed.
        
        Returns:
            True if key was updated
        """
        if current_time - self.last_key_update < self.key_update_interval_sec:
            return False
        
        # Get chroma from key detection window
        chroma = notes_to_chroma(self.notes, current_time, self.key_window_sec)
        
        if np.sum(chroma) > 0:
            new_key = estimate_key(chroma)
            if new_key is not None:
                self.current_key = new_key
                self.last_key_update = current_time
                return True
        
        return False
    
    def extract_current_features(self, current_time: float) -> Tuple[float, float, float, int, str]:
        """
        Extract harmony features at the current time.
        
        Args:
            current_time: Current time in seconds
        
        Returns:
            Tuple of (centroid_x, centroid_y, magnitude, key_root, key_mode)
        """
        # Update key if needed
        self._update_key(current_time)
        
        # Get chroma from tension window
        tension_chroma = notes_to_chroma(self.notes, current_time, self.tension_window_sec)
        
        # Extract key
        key_root, key_mode = self.current_key
        
        # Compute tension
        cx, cy, mag = chroma_to_tonnetz_tension(tension_chroma, key_root)
        
        return cx, cy, mag, key_root, key_mode
    
    def update(self, current_time: float):
        """
        Update the real-time visualization with current harmony state.
        
        Args:
            current_time: Current time in seconds
        """
        # Clean old notes
        self._clean_buffer(current_time)
        
        # Extract features
        cx, cy, mag, key_root, key_mode = self.extract_current_features(current_time)
        
        # Store in history
        self.tension_history.append((cx, cy))
        self.magnitude_history.append(mag)
        self.time_history.append(current_time)
        
        # Limit history length
        max_history = 500
        if len(self.tension_history) > max_history:
            self.tension_history = self.tension_history[-max_history:]
            self.magnitude_history = self.magnitude_history[-max_history:]
            self.time_history = self.time_history[-max_history:]
        
        # Update visualization if enabled
        if self.visualize_enabled:
            self._update_visualization(cx, cy, mag, key_root, key_mode, current_time)
    
    def _update_visualization(self, cx: float, cy: float, mag: float, 
                            key_root: int, key_mode: str, current_time: float):
        """Update the real-time visualization."""
        from tonnetz import lattice_to_display
        
        # Convert relative to absolute lattice coords
        key_x, key_y = TONNETZ_COORDS[key_root]
        lat_x = key_x + cx
        lat_y = key_y + cy
        
        # Transform to display coordinates
        X, Y = lattice_to_display(float(lat_x), float(lat_y))
        
        # Update current position
        self.current_point.set_data([X], [Y])
        self.inner_point.set_data([X], [Y])
        
        # Update tension circle
        key_disp_x, key_disp_y = lattice_to_display(float(key_x), float(key_y))
        self.tension_circle.set_center((key_disp_x, key_disp_y))
        self.tension_circle.set_radius(mag)
        
        # Update trail (last N positions)
        trail_length = 50
        if len(self.tension_history) > 1:
            trail_start = max(0, len(self.tension_history) - trail_length)
            trail_rel = np.array(self.tension_history[trail_start:])
            trail_lattice = trail_rel + np.array([key_x, key_y])
            trail_disp = np.array([lattice_to_display(p[0], p[1]) for p in trail_lattice])
            self.trail_line.set_data(trail_disp[:, 0], trail_disp[:, 1])
        
        # Update text
        self.key_text.set_text(f"Key: {format_key_name(key_root, key_mode)}")
        self.coord_text.set_text(f"Centroid: ({cx:.2f}, {cy:.2f}) | Mag: {mag:.2f}")
        self.time_text.set_text(f"Time: {current_time:.1f}s | Notes: {len(self.notes)}")
        
        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def close(self):
        """Close the visualization."""
        if self.visualize_enabled:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
    
    def get_history(self) -> Tuple[np.ndarray, np.ndarray, Tuple[int, str]]:
        """
        Get the accumulated history data.
        
        Returns:
            Tuple of (tension_xy, tension_magnitude, current_key)
        """
        return (
            np.array(self.tension_history),
            np.array(self.magnitude_history),
            self.current_key
        )


def demo_realtime_from_midi(
    midi_path: str,
    playback_speed: float = 1.0,
    update_interval_sec: float = 0.1
):
    """
    Demo: Load a MIDI file and simulate real-time processing with visualization.
    
    Args:
        midi_path: Path to MIDI file
        playback_speed: Speed multiplier (1.0 = normal, 2.0 = 2x speed)
        update_interval_sec: How often to update (seconds)
    """
    import pretty_midi
    import time as time_module
    
    print(f"Loading MIDI file: {midi_path}")
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    # Collect all note events
    events: List[Tuple[float, str, int, int]] = []  # (time, type, pitch, velocity)
    
    for instrument in pm.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                events.append((note.start, 'on', note.pitch, note.velocity))
                events.append((note.end, 'off', note.pitch, 0))
    
    # Sort by time
    events.sort(key=lambda x: x[0])
    
    if not events:
        print("No notes found in MIDI file!")
        return
    
    print(f"Found {len(events)} note events")
    print("Starting real-time simulation...")
    
    # Create extractor
    extractor = RealtimeHarmonyExtractor(
        key_window_sec=4.0,
        tension_window_sec=1.0,
        key_update_interval_sec=2.0,
        visualize=True
    )
    
    # Simulate real-time playback
    start_wall_time = time_module.time()
    start_midi_time = events[0][0]
    
    event_idx = 0
    last_update = 0.0
    
    try:
        while event_idx < len(events):
            # Current time in MIDI coordinates
            wall_elapsed = (time_module.time() - start_wall_time) * playback_speed
            current_midi_time = start_midi_time + wall_elapsed
            
            # Process all events up to current time
            while event_idx < len(events) and events[event_idx][0] <= current_midi_time:
                event_time, event_type, pitch, velocity = events[event_idx]
                
                if event_type == 'on':
                    extractor.add_note(pitch, event_time, velocity)
                else:
                    extractor.note_off(pitch, event_time)
                
                event_idx += 1
            
            # Update visualization periodically
            if current_midi_time - last_update >= update_interval_sec:
                extractor.update(current_midi_time)
                last_update = current_midi_time
            
            # Small sleep to avoid busy-waiting
            time_module.sleep(0.01)
        
        # Final update
        extractor.update(events[-1][0])
        
        print("\nPlayback complete! Close the window to exit.")
        
        # Keep window open
        import matplotlib.pyplot as plt
        plt.show(block=True)
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        extractor.close()


# --- RUN IT ---
if __name__ == "__main__":
    
    # Real-time harmony extraction from MIDI file
    demo_realtime_from_midi(midi_path, playback_speed=1.0, update_interval_sec=0.1)
    # Default: full analysis and visualization
    #extract_and_visualize()