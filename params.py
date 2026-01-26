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

