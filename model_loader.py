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
#os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

import torch

#from models import *
from models import get_model_hparams
from x_transformer import *

#===================================================================================================

def warm_start_tension(model: D_Tension_Joker, checkpoint_path: str) -> None:
    """Load only an exact base decoder; never silently train a random frozen prior."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Base joker checkpoint missing: {checkpoint_path}. '
                                'Provide INIT_FROM_CKPT or train D_Base_Joker_little_v1 first.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint.get('model_state_dict', checkpoint)
    adapter = ('decoder.tension_conditioner.', 'decoder.harm_film.')
    expected = {name for name in model.state_dict() if not name.startswith(adapter)}
    missing, unexpected = expected - state.keys(), state.keys() - expected
    if missing or unexpected:
        raise ValueError(f'Not a matching base joker checkpoint: missing={sorted(missing)}, '
                         f'unexpected={sorted(unexpected)}')
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not name.startswith(adapter) for name in result.missing_keys):
        raise ValueError(f'Unexpected warm-start mismatch: {result}')


def load_model(model_name='default',
               cfg=None,
               compile_mode='max-autotune',
               set_only=False,
               initialize_only=False,
               ):
    """
    Load and initialize Monster Piano Transformer model with specified parameters.

    Parameters:
    model_name (str): The name of the model to load from MODELS_INFO dictionary. Default and the best model is 'without velocity - 7 epochs'.
    device (str): The computing device to use. Options include 'cpu' or 'cuda'. Default is 'cuda'.
    compile_mode (str): The torch.compile mode for the model. Options include 'default', 'reduce-overhead', 'max-autotune'. Default is 'max-autotune'.
    verbose (bool): Whether to print detailed information during the loading process. Default is False.
    initialize_only (bool): Build the architecture, including nested base models,
        without loading any checkpoint. Used by strict checkpoint evaluation.

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
    elif cfg['model_type'] == 'autoencoder_no_dtime_joker':
        mpt_model = AutoregressiveAutoencoder_no_dtime_joker(
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
    elif cfg['model_type'] == 'AE_antic':
        mpt_model = AE_antic(
           cfg = cfg,
           decoder = Decoder_no_dtime_anticipation(
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
    elif cfg['model_type'] == 'autoencoder_no_dtime_dual':
        # Dual-conditioned autoencoder: melodic shape buttons + harmony movement decay
        mpt_model = AutoregressiveAutoencoder_no_dtime_dual(
            cfg = cfg,
            decoder = Decoder_no_dtime_dual(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                num_harmony_movements = cfg.get('num_harmony_movements', 8),
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
    elif cfg['model_type'] in ('AutoregressiveDecoder_no_conditioning', 'D_Base_Joker'):
        # Pitch-only causal decoder; the joker is an external generation trigger.
        mpt_model = AutoregressiveDecoder_no_conditioning(
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
    elif cfg['model_type'] == 'D_Tension_Joker':
        mpt_model = D_Tension_Joker(cfg=cfg, decoder=Decoder_tension_joker(
            max_seq_len=cfg['seq_len'], dim=cfg['emb_dim'], depth=cfg['num_layers'],
            heads=cfg['heads'], rotary_pos_emb=True, attn_flash=True,
            tens_cond_dim=cfg.get('tens_cond_dim', 128),
            tens_film_start_frac=cfg.get('tens_film_start_frac', 0.5),
            tens_film_scale_limit=cfg.get('tens_film_scale_limit', 0.5),
            tens_film_shift_limit=cfg.get('tens_film_shift_limit', 0.5)))
    elif cfg['model_type'] == 'AE_style':
        # Style-conditioned autoencoder (no harmony): buttons + cross-attention to style reference
        mpt_model = AE_style(
            cfg = cfg,
            decoder = Decoder_no_dtime_style(
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
            ),
            style_encoder = StyleEncoder(
                max_seq_len = cfg.get('style_seq_len', 256),
                dim = cfg['emb_dim'],
                depth = cfg.get('style_encoder_depth', 4),
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            ),
        )
    elif cfg['model_type'] == 'AE_style_harm':
        # Style + button + harmony (movement) conditioning via AdaLN-Zero FiLM,
        # hidden chord planner and auxiliary chord-factor supervision (stage 2).
        mpt_model = AE_style_harm(
            cfg = cfg,
            decoder = Decoder_no_dtime_style_harm(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                harm_cond_dim = cfg.get('harm_cond_dim', 256),
                harm_film_start_frac = cfg.get('harm_film_start_frac', 0.5),
                harm_film_scale_limit = cfg.get('harm_film_scale_limit', 1.0),
                harm_film_shift_limit = cfg.get('harm_film_shift_limit', 1.0),
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
            ),
            style_encoder = StyleEncoder(
                max_seq_len = cfg.get('style_seq_len', 256),
                dim = cfg['emb_dim'],
                depth = cfg.get('style_encoder_depth', 4),
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            ),
        )
    elif cfg['model_type'] == 'AE_style_tensions':
        # Style + button + CONTINUOUS tonal-tension conditioning via AdaLN-Zero
        # FiLM (TIV/TIS tension features from tension_extractor.py). Resumable
        # from AE_style_v2 (zero-init FiLM / aux head => identity at start).
        mpt_model = AE_style_tensions(
            cfg = cfg,
            decoder = Decoder_no_dtime_style_tensions(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                harm_cond_dim = cfg.get('harm_cond_dim', 256),
                harm_film_start_frac = cfg.get('harm_film_start_frac', 0.5),
                harm_film_scale_limit = cfg.get('harm_film_scale_limit', 1.0),
                harm_film_shift_limit = cfg.get('harm_film_shift_limit', 1.0),
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
            ),
            style_encoder = StyleEncoder(
                max_seq_len = cfg.get('style_seq_len', 256),
                dim = cfg['emb_dim'],
                depth = cfg.get('style_encoder_depth', 4),
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            ),
        )
    elif cfg['model_type'] == 'AE_style_move':
        # Style + button + BINARY move-flag conditioning via AdaLN-Zero FiLM
        # ("move now, model decides which" — boundary positions only, no chord
        # factors / planner). Resumable from AE_style_v2 (zero-init FiLM =>
        # identity at start).
        mpt_model = AE_style_move(
            cfg = cfg,
            decoder = Decoder_no_dtime_style_move(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                harm_cond_dim = cfg.get('harm_cond_dim', 256),
                harm_film_start_frac = cfg.get('harm_film_start_frac', 0.5),
                harm_film_scale_limit = cfg.get('harm_film_scale_limit', 1.0),
                harm_film_shift_limit = cfg.get('harm_film_shift_limit', 1.0),
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
            ),
            style_encoder = StyleEncoder(
                max_seq_len = cfg.get('style_seq_len', 256),
                dim = cfg['emb_dim'],
                depth = cfg.get('style_encoder_depth', 4),
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            ),
        )
    elif cfg['model_type'] == 'AE_style_chords':
        # Style + button + SOUNDING-CHORD conditioning via AdaLN-Zero FiLM
        # (channel-4 chroma + bass only — a simpler AE_style_harm with no chord
        # factors / planner / aux heads). Resumable from AE_style_v2 (zero-init
        # FiLM => identity at start).
        mpt_model = AE_style_chords(
            cfg = cfg,
            decoder = Decoder_no_dtime_style_chords(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                harm_cond_dim = cfg.get('harm_cond_dim', 256),
                harm_film_start_frac = cfg.get('harm_film_start_frac', 0.5),
                harm_film_scale_limit = cfg.get('harm_film_scale_limit', 1.0),
                harm_film_shift_limit = cfg.get('harm_film_shift_limit', 1.0),
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
            ),
            style_encoder = StyleEncoder(
                max_seq_len = cfg.get('style_seq_len', 256),
                dim = cfg['emb_dim'],
                depth = cfg.get('style_encoder_depth', 4),
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            ),
        )
    elif cfg['model_type'] == 'AE_style_roman':
        # Style + button + ROMAN-MOVEMENT conditioning via AdaLN-Zero FiLM: a
        # variation of AE_style_chords conditioned on the roman-derived 5-class
        # harmony movement (+ NULL) instead of channel-4 chroma + bass. Resumable
        # from AE_style_v2 (zero-init FiLM => identity at start).
        mpt_model = AE_style_roman(
            cfg = cfg,
            decoder = Decoder_no_dtime_style_roman(
                max_seq_len = cfg['seq_len'],
                dim = cfg['emb_dim'],
                depth = cfg['num_layers'],
                heads = cfg['heads'],
                harm_cond_dim = cfg.get('harm_cond_dim', 256),
                harm_film_start_frac = cfg.get('harm_film_start_frac', 0.5),
                harm_film_scale_limit = cfg.get('harm_film_scale_limit', 1.0),
                harm_film_shift_limit = cfg.get('harm_film_shift_limit', 1.0),
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
            ),
            style_encoder = StyleEncoder(
                max_seq_len = cfg.get('style_seq_len', 256),
                dim = cfg['emb_dim'],
                depth = cfg.get('style_encoder_depth', 4),
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            ),
        )
    elif cfg['model_type'] == 'AE_antic_style':
        # Style-conditioned autoencoder + anticipation: buttons + cross-attention to
        # style reference + anticipated-pitch signal for user-injected notes
        mpt_model = AE_antic_style(
            cfg = cfg,
            decoder = Decoder_no_dtime_antic_style(
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
            ),
            style_encoder = StyleEncoder(
                max_seq_len = cfg.get('style_seq_len', 256),
                dim = cfg['emb_dim'],
                depth = cfg.get('style_encoder_depth', 4),
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            ),
        )
    elif cfg['model_type'] == 'AE_antic_style_joker':
        # Style-conditioned autoencoder + anticipation: buttons + cross-attention to
        # style reference + anticipated-pitch signal for user-injected notes
        mpt_model = AE_antic_style_joker(
            cfg = cfg,
            decoder = Decoder_no_dtime_antic_style(
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
            ),
            style_encoder = StyleEncoder(
                max_seq_len = cfg.get('style_seq_len', 256),
                dim = cfg['emb_dim'],
                depth = cfg.get('style_encoder_depth', 4),
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            ),
        )
    elif cfg['model_type'] == 'AE_style_joker':
        # Style-conditioned autoencoder (no harmony): buttons + cross-attention to style reference
        mpt_model = AE_style_joker(
            cfg = cfg,
            decoder = Decoder_no_dtime_style(
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
            ),
            style_encoder = StyleEncoder(
                max_seq_len = cfg.get('style_seq_len', 256),
                dim = cfg['emb_dim'],
                depth = cfg.get('style_encoder_depth', 4),
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            ),
        )
    elif cfg['model_type'] == 'AE_style_jokerParam':
        # Style-conditioned autoencoder with a zero-initialised categorical
        # joker parameter and neutral (0.0) scalar button input.
        mpt_model = AE_style_jokerParam(
            cfg = cfg,
            decoder = Decoder_no_dtime_style_jokerParam(
                # The opt-in branch creates no timing weights in legacy models.
                timing_enabled = cfg.get('timing_enabled', False),
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
            ),
            style_encoder = StyleEncoder(
                max_seq_len = cfg.get('style_seq_len', 256),
                dim = cfg['emb_dim'],
                depth = cfg.get('style_encoder_depth', 4),
                heads = cfg['heads'],
                rotary_pos_emb = True,
                attn_flash = True
            ),
        )
    elif cfg['model_type'] == 'ae_buttons_p_residual':
        # Residual chord adapter: load base model with pretrained weights (no compile
        # to avoid _orig_mod. prefix mismatch), then wrap in residual adapter.
        base_model_name = cfg['base_model_name']
        base_cfg = get_model_hparams(base_model_name)
        base_model = load_model(model_name=base_model_name, cfg=base_cfg,
                                set_only=initialize_only, initialize_only=initialize_only,
                                compile_mode='none')
        mpt_model = AE_buttons_p_residual(base_model=base_model, cfg=cfg)
    else:
        raise ValueError(f"Unknown model_type {cfg['model_type']!r}; choose a supported training configuration")

    if set_only == False and not initialize_only:
        model_path = cfg['ckpt_file_name']

        if not torch.cuda.is_available():
            map_location = torch.device('cpu')
        else:
            map_location = None

        state = torch.load(model_path, map_location=map_location)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        # Trained timing checkpoints must contain the full branch. Loading the
        # pitch-only base is handled separately by warm_start_timing in training.
        mpt_model.load_state_dict(
            state, strict=cfg.get('timing_enabled', False)
            or cfg['model_type'] in ('D_Base_Joker', 'D_Tension_Joker'))

        if compile_mode != 'none':
            mpt_model = torch.compile(mpt_model, mode=compile_mode)


    return mpt_model
