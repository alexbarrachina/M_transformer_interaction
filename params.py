#===================================================================================================
# Monster Genie params.py Python module
# Global parameters 
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

import torch
from typing import Final

########################################################

''' TRAINING/TESTING FLAGS '''

TESTING = False # minimal configuration, just for fast testing
UPF = False # setting for UPF cluster
USE_TOPK = False # setting for topk sampling

''' DEFAULT MODEL PARAMETERS '''

DEFAULT_HPARAMS = {
    # specific model parameters
    "model_name": 'no_name',
    "description": 'no description',
    "seq_len": 2048,
    "emb_dim": 2048,
    "num_layers": 4,
    "heads": 32,

    # loss components
    "loss_recons": 1., # 1., original # Reconstruction loss

    "loss_margin": 0.01, #0.01 # encourage values to be closer to [-1, 1] range
    "loss_deviate": 0.01, # 0.01 # enalize button changes when notes are held (same notes)
    "loss_contour": 0.1, # 0.1, but 0.01 # melody shape, in direction (-1,+1)
    "loss_button_held": 0., #0.01 # # Penalizes same button values when consecutive notes are different
    "loss_norm_pos": 0.0, #0.01 #  between normalized positions of pitches and buttons.
    "loss_pitch_button": 0.0, #0.01 #  correlates pitch tendencies with button concentrations
    "loss_button_concentration": 0.0, #0.01 # Multiplier for button concentration loss
    "loss_window_corr": 0.0, # weight for windowed Pearson correlation loss (1-corr)
    "loss_saturated_contour": 0.0, #0.1 # Saturated contour loss (allows button saturation at extremes)
    "loss_pitch_extreme_anchoring": 0.0, #0.01 # Anchors extreme pitches to extreme buttons
    "loss_nonlinear_compression": 0.0, #0.1 # Non-linear compression: more control in middle, less at extremes
    "loss_latent_velocity": 0.0, #0.1 # Latent→velocity coupling: makes buttons control pitch direction
    "loss_drift": 0.0, #0.1 # Drift regularization: rewards cumulative motion in latent direction
    # % of every component in loss contour
    "loss_contour_perc": 0., # 0.4, original genie # encourage button intervals to match piano note intervals (in direction, not magnitude, -1,+1)
    "loss_multi_step_perc": 1., # 0.3, original # considers relationships between the current note and multiple previous notes (in directions, not magnitude, -1,+1)
    "loss_interval_perc": 0., # 0.2, original # Encourages the relative magnitudes of intervals to be preserved between pitches and buttons
    "loss_shape_perc": 0., # 0.1, original # Preserves the overall shape of melodic phrases by comparing the pattern of ups and downs within sliding windows.

    'loss_arrow_consistency': 0.1, # Weight for arrow consistency loss (soft arrows + KL divergence)
    'arrow_soft_temp': 2.0, # Temperature for soft arrow boundaries (lower = sharper)

    # training parameters
    'learning_rate': 1e-4,
    'grad_clip': 1.5,
    'num_workers': 10,
    'num_val_batches_per_step': 1,
    "batch_size": 20,
    "epochs": 6000,

    # output parameters
    "save_every": 25000,
    "validate_every": 500,
    "generate_every": 10000,
    "generate_length": 512,
    "print_stats_every": 500,

    # dataset parameters
    "data%": 20, # % of dataset to use
    "data_augment_time_stretch_max": 0.05, # Max time stretch for data augmentation (+- 5%)
    "data_augment_transpose_max": 6, # Max transpose for data augmentation (+- 6 semitones, tritone)
    "data_augment_chord_threshold": 2, # Define chord threshold (e.g., notes within 2 time units are considered part of same chord)
    'pitch_history_dropout': 0.0,
    'dataset': 'giantmidi_full',
    "dataset_train_path": "./Training-Data/giantMIDI_sel", # './Training-Data/asigalov_train'
    "dataset_val_path": "./Training-Data/giantMIDI_sel_test", # './Training-Data/asigalov_val'

    # output parameters
    "save_dir": "./saved_checkpoints",

    # vocabulary parameters
    "num_buttons": 12,
    "num_arrows": 7,

    # Activation flags 
    "use_logs": False,
    "use_topk": False,

    # Zero out pitch embeddings to force arrow/buttons reliance
    'pitch_history_dropout': 0.0,

    # Freeze encoder for first N steps
    "unfreeze_encoder_after_n_epochs": 30, # after N epochs
}

''' VOCABULARY '''
VOCAB_SIZE_PITCH:Final[int] = 128
PAD_IDX:Final[int] = 128

RANGE_DTIME_SHIFT:Final[int] = 127
VOCAB_SIZE_DTIME:Final[int] = RANGE_DTIME_SHIFT + 1

RANGE_DUR_SHIFT:Final[int] = 127
VOCAB_SIZE_DUR:Final[int] = RANGE_DUR_SHIFT + 1

RANGE_VEL_SHIFT:Final[int] = 127 
VOCAB_SIZE_VEL:Final[int] = RANGE_VEL_SHIFT + 1

VOCAB_SIZE_ARROWS:Final[int] = 8
ARROW_NA:Final[int] = 7

''' TRAINING '''
# Taken from the paper
if torch.cuda.is_available(): 
    DEFAULT_HPARAMS['num_workers'] = 10
    DEFAULT_HPARAMS['use_logs'] = True
    #print("using CUDA")
else: # on macbook pro  
    DEFAULT_HPARAMS['num_workers'] = 1
    print("using CPU/MPS")
    DEFAULT_HPARAMS['use_logs'] = False

if TESTING:
    DEFAULT_HPARAMS['batch_size'] = 1
    DEFAULT_HPARAMS['seq_len'] = 16 
    DEFAULT_HPARAMS['use_logs'] = True
    DEFAULT_HPARAMS['data%'] = 1


  
''' DATASET '''

# ASIGALOV DATASET path
if torch.cuda.is_available(): 
    # Path to your locally saved dataset
    DEFAULT_HPARAMS['local_dataset_path'] = "../Datasets/asigalov61___monster-piano"
else:
    DEFAULT_HPARAMS['local_dataset_path'] = "../../../Datasets/MIDI/asigalov61___monster-piano"

# MIDI CHANNELS
# Melody channel filter
MELODY_CHANNEL = 0  # Channel 0 is melody (channel 1 in the midifile)
ACCOMP_CHANNEL = 10  # Channel 10 is accompaniment (channel 11 in the midifile)
HARMONY_CHANNEL = 3 # harmony movement info (channel 4 in the midifile)
CHORDS_CHANNEL = 4 # harmony chords (channel 5 in the midifile)

MELODY_CHORDS_CHANNEL = 1 # channel 2 in the midifile
EXTRA_ACCOMP_CHANNEL = 11 # channel 12 in the midifile

MOVE_STABILIZE = 0 #60
MOVE_RECOLOR = 1 #61
MOVE_PREPARE = 2 #62
MOVE_TENSION = 3 #63
MOVE_RESOLVE = 4 #64
MOVE_EVADE = 5 #65
MOVE_CHROMATIC = 6 #66
MOVE_MODULATE = 7 #67
MOVE_JOKER = 8        # "move now, model decides which" (not a stored MIDI pitch; used in training/inference)
NUM_MOVEMENTS = 9     # 0..7 real movements + 8 = joker
MOVE_PITCH_BASE = 60  # channel-3 movement notes encode move as pitch (60 + move)

# ---------------------------------------------------------------------------
#  HARMONY V3: compact chord-label + key conditioning (stage-1 pickle format)
# ---------------------------------------------------------------------------
# Rich harmony info (root, chord quality, harmonic function, local key) is
# stored as SPARSE pseudo-events (one per chord / one per key change), keeping
# the flat "5 tokens per event" layout so the dataset stays memory-light.
# Chroma / bass are NOT stored: they are derived at load time from the existing
# channel-4 chord-tone events. The marker values are >15 (outside the MIDI
# channel range 0..15) so they never collide with real note channels.

CHORD_LABEL_CHANNEL = 120  # pseudo-event: [0, root_pc, quality_id, function_id, 120]
KEY_CHANNEL = 121          # pseudo-event: [0, key_pc, mode, 0, 121] (emitted on key change)

# "unknown" sentinels (used when the analyzer provides null / no label)
PC_UNKNOWN = 12            # pitch class 0..11, 12 = unknown
MODE_MAJOR = 0
MODE_MINOR = 1
MODE_UNKNOWN = 2

# Chord-quality enum (0 = unknown). Kept small/coarse on purpose.
QUALITY_UNKNOWN = 0
QUALITY_MAJOR = 1
QUALITY_MINOR = 2
QUALITY_DIMINISHED = 3
QUALITY_AUGMENTED = 4
QUALITY_DOMINANT_SEVENTH = 5
QUALITY_MAJOR_SEVENTH = 6
QUALITY_MINOR_SEVENTH = 7
QUALITY_MINOR_MAJOR_SEVENTH = 8
QUALITY_DIMINISHED_SEVENTH = 9          # fully-diminished 7th (°7)
QUALITY_HALF_DIMINISHED_SEVENTH = 10    # half-diminished 7th (ø7 / m7b5)
QUALITY_DOMINANT_EXT = 11               # 9/11/13 etc. over a dominant
QUALITY_OTHER = 12
NUM_QUALITIES = 13

# Maps the analyzer's chord-quality names (see docs examples like
# "DOMINANT_SEVENTH", "MINOR_MAJOR_SEVENTH", "DIMINISHED_MINOR_SEVENTH") to ids.
# NOTE: "DIMINISHED_MINOR_SEVENTH" = half-diminished 7th (dim triad + minor 7th).
QUALITY_NAME_MAP = {
    'MAJOR': QUALITY_MAJOR,
    'MINOR': QUALITY_MINOR,
    'DIMINISHED': QUALITY_DIMINISHED,
    'AUGMENTED': QUALITY_AUGMENTED,
    'DOMINANT_SEVENTH': QUALITY_DOMINANT_SEVENTH,
    'MAJOR_SEVENTH': QUALITY_MAJOR_SEVENTH,
    'MINOR_SEVENTH': QUALITY_MINOR_SEVENTH,
    'MINOR_MAJOR_SEVENTH': QUALITY_MINOR_MAJOR_SEVENTH,
    'DIMINISHED_SEVENTH': QUALITY_DIMINISHED_SEVENTH,
    'DIMINISHED_MINOR_SEVENTH': QUALITY_HALF_DIMINISHED_SEVENTH,
    'HALF_DIMINISHED_SEVENTH': QUALITY_HALF_DIMINISHED_SEVENTH,
    'DOMINANT_NINTH': QUALITY_DOMINANT_EXT,
    'DOMINANT_ELEVENTH': QUALITY_DOMINANT_EXT,
    'DOMINANT_THIRTEENTH': QUALITY_DOMINANT_EXT,
}

# Interval-template -> quality id, for the notes-derived fallback when no label
# string is present (intervals are pitch classes relative to the candidate root).
QUALITY_TEMPLATES = {
    frozenset({0, 4, 7}): QUALITY_MAJOR,
    frozenset({0, 3, 7}): QUALITY_MINOR,
    frozenset({0, 3, 6}): QUALITY_DIMINISHED,
    frozenset({0, 4, 8}): QUALITY_AUGMENTED,
    frozenset({0, 4, 7, 10}): QUALITY_DOMINANT_SEVENTH,
    frozenset({0, 4, 7, 11}): QUALITY_MAJOR_SEVENTH,
    frozenset({0, 3, 7, 10}): QUALITY_MINOR_SEVENTH,
    frozenset({0, 3, 7, 11}): QUALITY_MINOR_MAJOR_SEVENTH,
    frozenset({0, 3, 6, 9}): QUALITY_DIMINISHED_SEVENTH,
    frozenset({0, 3, 6, 10}): QUALITY_HALF_DIMINISHED_SEVENTH,
}

# Harmonic-function enum (0 = unknown). Coarse German functional categories.
FUNC_UNKNOWN = 0
FUNC_TONIC = 1          # T / t
FUNC_SUBDOMINANT = 2    # S / s
FUNC_DOMINANT = 3       # D / d
FUNC_DOUBLE_DOMINANT = 4  # DD
NUM_FUNCTIONS = 5

FUNCTION_NAME_MAP = {
    'T': FUNC_TONIC, 't': FUNC_TONIC,
    'S': FUNC_SUBDOMINANT, 's': FUNC_SUBDOMINANT,
    'D': FUNC_DOMINANT, 'd': FUNC_DOMINANT,
    'DD': FUNC_DOUBLE_DOMINANT,
}

# ---------------------------------------------------------------------------
#  TONAL TENSION conditioning (AE_style_tensions)
# ---------------------------------------------------------------------------
# The AE_style_tensions model conditions FiLM on a CONTINUOUS tonal-tension
# feature vector (TIV/TIS-based) computed on the fly by tension_extractor.py,
# instead of the discrete chord factors / movements used by AE_style_harm.
# TENSION_FEATURE_DIM must match tension_extractor.TENSION_FEATURE_DIM.
TENSION_FEATURE_DIM = 48   # [tiv12, key_post24, conf, entropy, harm_change,
                           #  dist_key, dist_tonic, dist_subdom, dist_dom,
                           #  dissonance, mod_pressure, tension, slope, resolution]

# High-level performer commands mapped to a target tension trajectory at
# inference (see tension_extractor.command_to_target).
TENSION_CMD_MAINTAIN = 'maintain'
TENSION_CMD_ADD_TENSION = 'add_tension'
TENSION_CMD_RESOLVE = 'resolve'
TENSION_CMD_CHANGE_TONAL_CENTER = 'change_tonal_center'

# Note-name (English spelling, e.g. "C#", "Gb") -> pitch class 0..11.
_NOTE_BASE_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def note_name_to_pc(name: str) -> int:
    """Convert a note-name token ('C', 'C#', 'Db', 'F##', 'Gb') to pitch class 0..11.
    Returns PC_UNKNOWN for null/empty/unparseable input."""
    if name is None:
        return PC_UNKNOWN
    name = name.strip()
    if name == '' or name.lower() == 'null':
        return PC_UNKNOWN
    base = name[0].upper()
    if base not in _NOTE_BASE_PC:
        return PC_UNKNOWN
    pc = _NOTE_BASE_PC[base]
    for ch in name[1:]:
        if ch == '#':
            pc += 1
        elif ch == 'b' or ch == 'B':
            pc -= 1
        else:
            break
    return pc % 12
