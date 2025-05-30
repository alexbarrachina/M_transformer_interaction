#===================================================================================================
# Monster Piano Transformer model_loader Python module
#===================================================================================================
# Project Los Angeles
# Tegridy Code 2025
#===================================================================================================
# License: Apache 2.0
#===================================================================================================

import os

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

#===================================================================================================

from models import *

import torch

from x_transformer_1_23_2 import AutoregressiveAutoencoder, Decoder, Encoder, EncoderOnly, DecoderOnly, DecoderSimple, DecoderSimple_continuous_dtime, Decoder_no_dtime, Encoder_no_dtime, AutoregressiveAutoencoder_no_dtime, Encoder_antic

#===================================================================================================

def load_model(model_name='default',
               device='cuda',
               compile_mode='max-autotune',
               verbose=False
               ):
    """
    Load and initialize Giant Music Transformer model with specified parameters.

    Parameters:
    model_name (str): The name of the model to load from MODELS_INFO dictionary. Default and the best model is 'without velocity - 7 epochs'.
    device (str): The computing device to use. Options include 'cpu' or 'cuda'. Default is 'cuda'.
    compile_mode (str): The torch.compile mode for the model. Options include 'default', 'reduce-overhead', 'max-autotune'. Default is 'max-autotune'.
    verbose (bool): Whether to print detailed information during the loading process. Default is False.

    Returns:
    model: The initialized Monster Piano Transformer model configured with the specified parameters.

    Example use:
    
    import monsterpianotransformer as mpt
    
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
    if verbose:
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
        
        print('=' * 70)
        print('Selected model:', model_name.title(), '/', MODELS_PARAMETERS[model_name]['params'], 'M params')
        print('=' * 70)
        print('Model info:')
        print('-' * 70)
        print(MODELS_INFO[model_name])

        print('=' * 70)
        print('Downloading model...')

    else:
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

    model_path = MODELS_FILE_NAMES[model_name]

    if verbose:
        print('Done!')
        print('=' * 70)
        
        print('Instantiating model...')
        # Check if CUDA is available
    
    if not torch.cuda.is_available():
        map_location = torch.device('cpu')
    else:
        map_location = None
    
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
    if verbose:
        print('Done!')
        print('=' * 70)
        
        print('Loading model...')
    
    mpt_model.load_state_dict(torch.load(model_path, map_location=map_location)) # weights_only=True not compatible cpu

    if verbose:
        print('Done!')
        print('=' * 70)
    
        print('Compiling model...')

    mpt_model = torch.compile(mpt_model, mode=compile_mode)

    mpt_model.to(device)
    mpt_model.eval()  

    return mpt_model

