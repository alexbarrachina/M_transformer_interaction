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

from models import *
from x_transformer import *

#===================================================================================================

def load_model(model_name='default',
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
    
    if model_name not in MODELS_PARAMETERS:
        print('=' * 70)
        print('Available models:')
        
        for n, d in MODELS_INFO.items():
            print('=' * 70)
            print('MODEL NAME:', n)
            print('-' * 70)
            print('MODEL INFO:', d)

        print('=' * 70)
        return []

    if model_name in MODELS_TYPES:
       model_type = MODELS_TYPES[model_name]
    else:
        model_type = 'autoencoder'

    print(model_type)
    
    if model_type == 'autoencoder':
        mpt_model = AutoregressiveAutoencoder(
        ignore_index = MODELS_PARAMETERS[model_name]['pad_idx'], 
        #pad_value=MODELS_PARAMETERS[model_name]['pad_idx'],
        decoder = Decoder(
        num_tokens = MODELS_PARAMETERS[model_name]['pad_idx']+1,
        max_seq_len = MODELS_PARAMETERS[model_name]['seq_len'],
        dim = MODELS_PARAMETERS[model_name]['emb_dim'],
        depth = MODELS_PARAMETERS[model_name]['num_layers'],
        heads = MODELS_PARAMETERS[model_name]['heads'],
        rotary_pos_emb = True,
        attn_flash = True
        ),
        encoder = Encoder(
        num_tokens = MODELS_PARAMETERS[model_name]['pad_idx']+1,
        max_seq_len = MODELS_PARAMETERS[model_name]['seq_len'],
        dim = MODELS_PARAMETERS[model_name]['emb_dim'],
        depth = MODELS_PARAMETERS[model_name]['num_layers'],
        heads = MODELS_PARAMETERS[model_name]['heads'],
        rotary_pos_emb = True,
        attn_flash = True
        )
        )
    elif model_type == 'autoencoder_no_dtime':
        mpt_model = AutoregressiveAutoencoder_no_dtime(
        ignore_index = MODELS_PARAMETERS[model_name]['pad_idx'], 
        #pad_value=MODELS_PARAMETERS[model_name]['pad_idx'],
        decoder = Decoder_no_dtime(
        num_tokens = MODELS_PARAMETERS[model_name]['pad_idx']+1,
        max_seq_len = MODELS_PARAMETERS[model_name]['seq_len'],
        dim = MODELS_PARAMETERS[model_name]['emb_dim'],
        depth = MODELS_PARAMETERS[model_name]['num_layers'],
        heads = MODELS_PARAMETERS[model_name]['heads'],
        rotary_pos_emb = True,
        attn_flash = True
        ),
        encoder = Encoder_no_dtime(
        num_tokens = MODELS_PARAMETERS[model_name]['pad_idx']+1,
        max_seq_len = MODELS_PARAMETERS[model_name]['seq_len'],
        dim = MODELS_PARAMETERS[model_name]['emb_dim'],
        depth = MODELS_PARAMETERS[model_name]['num_layers'],
        heads = MODELS_PARAMETERS[model_name]['heads'],
        rotary_pos_emb = True,
        attn_flash = True
        )
        )  
    elif model_type == 'autoencoder_w_encoder_antic':
        mpt_model = AutoregressiveAutoencoder(
        ignore_index = MODELS_PARAMETERS[model_name]['pad_idx'], 
        #pad_value=MODELS_PARAMETERS[model_name]['pad_idx'],
        decoder = Decoder(
        num_tokens = MODELS_PARAMETERS[model_name]['pad_idx']+1,
        max_seq_len = MODELS_PARAMETERS[model_name]['seq_len'],
        dim = MODELS_PARAMETERS[model_name]['emb_dim'],
        depth = MODELS_PARAMETERS[model_name]['num_layers'],
        heads = MODELS_PARAMETERS[model_name]['heads'],
        rotary_pos_emb = True,
        attn_flash = True
        ),
        encoder = Encoder_antic(
        num_tokens = MODELS_PARAMETERS[model_name]['pad_idx']+1,
        max_seq_len = MODELS_PARAMETERS[model_name]['seq_len'],
        dim = MODELS_PARAMETERS[model_name]['emb_dim'],
        depth = MODELS_PARAMETERS[model_name]['num_layers'],
        heads = MODELS_PARAMETERS[model_name]['heads'],
        rotary_pos_emb = True,
        attn_flash = True
        )
        )  
    elif model_type == 'decoder_only':
        mpt_model = DecoderOnly(
            ignore_index = MODELS_PARAMETERS[model_name]['pad_idx'], 
            # Use Decoder instead of DecoderSimple to match the saved model architecture
            decoder = DecoderSimple(
            num_tokens = MODELS_PARAMETERS[model_name]['pad_idx']+1,
            max_seq_len = MODELS_PARAMETERS[model_name]['seq_len'],
            dim = MODELS_PARAMETERS[model_name]['emb_dim'],
            depth = MODELS_PARAMETERS[model_name]['num_layers'],
            heads = MODELS_PARAMETERS[model_name]['heads'],
            rotary_pos_emb = True,
            attn_flash = True
            )
        )
    elif model_type == 'encoder_only':
        mpt_model = EncoderOnly(
            ignore_index = MODELS_PARAMETERS[model_name]['pad_idx'], 
            #pad_value=MODELS_PARAMETERS[model_name]['pad_idx'],
            encoder = Encoder(
            num_tokens = MODELS_PARAMETERS[model_name]['pad_idx']+1,
            max_seq_len = MODELS_PARAMETERS[model_name]['seq_len'],
            dim = MODELS_PARAMETERS[model_name]['emb_dim'],
            depth = MODELS_PARAMETERS[model_name]['num_layers'],
            heads = MODELS_PARAMETERS[model_name]['heads'],
            rotary_pos_emb = True,
            attn_flash = True
            )
        )
    elif model_type == 'encoder_only_antic':
        mpt_model = EncoderOnly(
            ignore_index = MODELS_PARAMETERS[model_name]['pad_idx'], 
            #pad_value=MODELS_PARAMETERS[model_name]['pad_idx'],
            encoder = Encoder_antic(
            num_tokens = MODELS_PARAMETERS[model_name]['pad_idx']+1,
            max_seq_len = MODELS_PARAMETERS[model_name]['seq_len'],
            dim = MODELS_PARAMETERS[model_name]['emb_dim'],
            depth = MODELS_PARAMETERS[model_name]['num_layers'],
            heads = MODELS_PARAMETERS[model_name]['heads'],
            rotary_pos_emb = True,
            attn_flash = True
            )
        )
    elif model_type == 'decoder_only_2_buttons':
        mpt_model = Decoder_only_2_buttons(
            ignore_index = MODELS_PARAMETERS[model_name]['pad_idx'], 
            # Use Decoder instead of DecoderSimple to match the saved model architecture
            decoder = Decoder_no_dtime(
            num_tokens = MODELS_PARAMETERS[model_name]['pad_idx']+1,
            max_seq_len = MODELS_PARAMETERS[model_name]['seq_len'],
            dim = MODELS_PARAMETERS[model_name]['emb_dim'],
            depth = MODELS_PARAMETERS[model_name]['num_layers'],
            heads = MODELS_PARAMETERS[model_name]['heads'],
            rotary_pos_emb = True,
            attn_flash = True
            )
        )
    if set_only == False:
        model_path = MODELS_FILE_NAMES[model_name]

        if not torch.cuda.is_available():
            map_location = torch.device('cpu')
        else:
            map_location = None

        mpt_model.load_state_dict(torch.load(model_path, map_location=map_location)) # weights_only=True not compatible cpu

        if compile_mode != 'none':
            mpt_model = torch.compile(mpt_model, mode=compile_mode)


    return mpt_model

