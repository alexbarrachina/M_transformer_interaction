#===================================================================================================
# Monster Genie model_loader Python module
# Loads the model
# 
# Copyright 2025 Alex Barrachina
#
# Based on Project Los Angeles / Tegridy Code 2025
# https://github.com/asigalov61/monsterpianotransformer
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
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

import torch

#from models import *
from models import get_model_hparams
from x_transformer import *

#===================================================================================================

def load_model(model_name='default',
               cfg=None,
               compile_mode='max-autotune',
               set_only=False,
               ):
    """
    Load and initialize Monster Piano Transformer model with specified parameters.

    Parameters:
    model_name (str): The name of the model to load from MODELS_INFO dictionary. Default and the best model is 'without velocity - 7 epochs'.
    device (str): The computing device to use. Options include 'cpu' or 'cuda'. Default is 'cuda'.
    compile_mode (str): The torch.compile mode for the model. Options include 'default', 'reduce-overhead', 'max-autotune'. Default is 'max-autotune'.
    verbose (bool): Whether to print detailed information during the loading process. Default is False.

    Returns:
    model: The initialized Monster Piano Transformer model configured with the specified parameters.

    Example use:
    
    import x_transformer as mpt
    
    mpt_model = mpt.load_model('models')
    """
    
    if cfg is None:
        cfg = get_model_hparams(model_name)
        
    if cfg is None:
        print('=' * 70)
        print('Available models in models.py')
        print('=' * 70)
        return []

    print(cfg['model_type'])
    
    if cfg['model_type'] == 'autoencoder':
        mpt_model = AutoregressiveAutoencoder(
           cfg = cfg,
           decoder = Decoder(
           max_seq_len = cfg['seq_len'],
           dim = cfg['emb_dim'],
           depth = cfg['num_layers'],
           heads = cfg['heads'],
           rotary_pos_emb = True,
           attn_flash = True
        ),
        encoder = Encoder(
           max_seq_len = cfg['seq_len'],
           dim = cfg['emb_dim'],
           depth = cfg['num_layers'],
           heads = cfg['heads'],
           rotary_pos_emb = True,
           attn_flash = True
        )
        )
    elif cfg['model_type'] == 'autoencoder_no_dtime':
        mpt_model = AutoregressiveAutoencoder_no_dtime(
           cfg = cfg,
        decoder = Decoder_no_dtime(
           max_seq_len = cfg['seq_len'],
           dim = cfg['emb_dim'],
           depth = cfg['num_layers'],
           heads = cfg['heads'],
           rotary_pos_emb = True,
           attn_flash = True
        ),
        encoder = Encoder_no_dtime(
           max_seq_len = cfg['seq_len'],
           dim = cfg['emb_dim'],
           depth = cfg['num_layers'],
           heads = cfg['heads'],
           rotary_pos_emb = True,
           attn_flash = True
        )
        )  
    elif cfg['model_type'] == 'autoencoder_w_encoder_antic':
        mpt_model = AutoregressiveAutoencoder(
           cfg = cfg,
           decoder = Decoder(
           max_seq_len = cfg['seq_len'],
           dim = cfg['emb_dim'],
           depth = cfg['num_layers'],
           heads = cfg['heads'],
           rotary_pos_emb = True,
           attn_flash = True
        ),
        encoder = Encoder_antic(
           max_seq_len = cfg['seq_len'],
           dim = cfg['emb_dim'],
           depth = cfg['num_layers'],
           heads = cfg['heads'],
           rotary_pos_emb = True,
           attn_flash = True
        )
        )  
    elif cfg['model_type'] == 'decoder_only':
        mpt_model = DecoderOnly(
            cfg = cfg,
            # Use Decoder instead of DecoderSimple to match the saved model architecture
            decoder = DecoderSimple(
               max_seq_len = cfg['seq_len'],
               dim = cfg['emb_dim'],
               depth = cfg['num_layers'],
               heads = cfg['heads'],
               rotary_pos_emb = True,
               attn_flash = True
            )
        )
    elif cfg['model_type'] == 'encoder_only':
        mpt_model = EncoderOnly(
            cfg = cfg,
            encoder = Encoder(
               max_seq_len = cfg['seq_len'],
               dim = cfg['emb_dim'],
               depth = cfg['num_layers'],
               heads = cfg['heads'],
               rotary_pos_emb = True,
               attn_flash = True
            )
        )
    elif cfg['model_type'] == 'encoder_only_antic':
        mpt_model = EncoderOnly(
            cfg = cfg,
            encoder = Encoder_antic(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            )
        )
    elif cfg['model_type'] == 'autoencoder_melody':
        # Melody autoencoder with arrow guidance (no encoder needed)
        mpt_model = AutoregressiveAutoencoder_melody(
            cfg = cfg,
            decoder = Decoder_melody(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                pitch_history_dropout = cfg['pitch_history_dropout'],  # Zero out pitch embeddings to force arrow reliance
                rotary_pos_emb = True,
                attn_flash = True
            )
        )
    elif cfg['model_type'] == 'AE_melody_w_coarse_arrows':
        # Melody autoencoder with arrow guidance (no encoder needed)
        mpt_model = AE_melody_w_coarse_arrows(
            cfg = cfg,
            decoder = Decoder_melody_w_coarse_arrows(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                pitch_history_dropout = cfg['pitch_history_dropout'],  # Zero out pitch embeddings to force arrow reliance
                rotary_pos_emb = True,
                attn_flash = True
            )
        )
    elif cfg['model_type'] == 'AE_arrows_and_buttons':
        # Autoencoder with arrows and buttons guidance 
        mpt_model = AE_arrows_and_buttons(
            cfg = cfg,
            decoder = Decoder_arrows_and_buttons(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                pitch_history_dropout = cfg['pitch_history_dropout'],  # Zero out pitch embeddings to force arrow reliance
                rotary_pos_emb = True,
                attn_flash = True
            ),
            encoder = Encoder_no_dtime(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            )
        )
    elif cfg['model_type'] == 'AE_arrows_and_buttons_concatenated':
        # Autoencoder with arrows and buttons guidance 
        mpt_model = AE_arrows_and_buttons_concatenated(
            cfg = cfg,
            decoder = Decoder_arrows_and_buttons_concatenated(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                pitch_history_dropout = cfg['pitch_history_dropout'],  # Zero out pitch embeddings to force arrow reliance
                rotary_pos_emb = True,
                attn_flash = True
            ),
            encoder = Encoder_no_dtime(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            )
        )
    elif cfg['model_type'] == 'AE_arrows_and_buttons_simpler':
        # Autoencoder with arrows and buttons guidance 
        mpt_model = AE_arrows_and_buttons_simpler(
            cfg = cfg,
            decoder = Decoder_arrows_and_buttons_simpler(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                #pitch_history_dropout = cfg['pitch_history_dropout'],  # Zero out pitch embeddings to force arrow reliance
                rotary_pos_emb = True,
                attn_flash = True
            ),
            encoder = Encoder_no_dtime(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            )
        )
    elif cfg['model_type'] == 'AE_mixed_vocab':
        # Mixed vocabulary autoencoder: arrows (1-7) + buttons (8-31) in single key sequence
        mpt_model = AE_mixed_vocab(
            cfg = cfg,
            decoder = Decoder_mixed_vocab(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                num_buttons = cfg.get('num_buttons', 12),
                num_arrows = cfg.get('num_arrows', 7),
                pitch_history_dropout = cfg.get('pitch_history_dropout', 0.0),
                rotary_pos_emb = True,
                attn_flash = True
            ),
            encoder = Encoder_no_dtime(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            )
        )
    elif cfg['model_type'] == 'autoencoder_no_dtime_harmony':
        # Autoencoder with Tonnetz harmony conditioning (decoder-only)
        mpt_model = AutoregressiveAutoencoder_no_dtime_harmony(
            cfg = cfg,
            decoder = Decoder_no_dtime_harmony(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            ),
            encoder = Encoder_no_dtime(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            )
        )
    elif cfg['model_type'] == 'autoencoder_just_harmony':
        # Autoencoder with Tonnetz harmony conditioning (decoder-only)
        mpt_model = AutoregressiveAutoencoder_just_harmony(
            cfg = cfg,
            decoder = Decoder_just_harmony(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            ),

        )
    elif cfg['model_type'] == 'AE_no_conditioning':
        # Autoencoder with Tonnetz harmony conditioning (decoder-only)
        mpt_model = AE_no_conditioning(
            cfg = cfg,
            decoder = Decoder_no_conditioning(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            ),

        )
    if set_only == False:
        model_path = cfg['ckpt_file_name']

        if not torch.cuda.is_available():
            map_location = torch.device('cpu')
        else:
            map_location = None

        mpt_model.load_state_dict(torch.load(model_path, map_location=map_location), strict=False) # weights_only=True not compatible cpu

        if compile_mode != 'none':
            mpt_model = torch.compile(mpt_model, mode=compile_mode)


    return mpt_model

