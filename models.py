#===================================================================================================
# Monster Genie models Python module
# info about models available 
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

MODELS_HF_REPO_LINK = 'asigalov61/Monster-Piano-Transformer'
MODELS_HF_REPO_URL = 'https://huggingface.co/asigalov61/Monster-Piano-Transformer'

#===================================================================================================

MODELS_INFO = {'tester': 'tester. light model not trained',
               'light': ' light test version (without velocity) trained for 40 epochs on full Monster Piano dataset.',
               'full': 'the full model, trained for 40 epochs on full Monster Piano dataset.',
               'encoder_light_mar14': 'encoder only, for testing melodic contour loss., light dataset, 39_eps_82603_steps_0.0038_loss_0.0_acc',
               'encoder full': 'encoder only, for testing melodic contour loss., full dataset',
                'encoder_orig': 'encoder only, original genie implemetation, full dataset, light model, 5_eps_422096_steps_0.0021_loss_0.0_acc',
                'encoder_new_loss_mar17': 'encoder only, new loss functionalities, full dataset, light model, 9_eps_82603_steps_0.0114_loss_0.0_acc',
                'full_mar18': 'full model, new loss functionalities, full dataset, 2_eps_42335_steps_0.0129_loss_0.0_acc.',
                'light_mar24': 'light model, new loss functionalities, full dataset, 3_eps_253258_steps_2.4025_loss_0.8244_acc.',
                'light_mar26_only_butt_loss': 'light model, only button loss, no recons, light dataset, 39_eps_82603_steps_0.0107_loss_0.0158_acc',
                'light_mar26_only_recon_loss': 'light model, only recons loss, no button loss , light dataset, 34_eps_72013_steps_0.3459_loss_0.8914_acc',
                'light_mar27_decoder_only': 'light model, decoder only, no encoder, no button loss, light dataset, 20_eps_1738381_steps_0.7658_loss_0.7768_acc',
                'big_mar28_decoder_only': 'big model, decoder only, no encoder, no button loss, big dataset, early interruption on 7_eps_153170_steps_0.8395_loss_0.7523_acc',
                'light_apr3_decoder_only_no_dur_cont_dtime': 'light model, decoder only, no encoder, no button loss, no duration, light dataset 1%, interruption before overfitting on checkpoint_12_eps_25417_steps_0.8986_loss_0.7389_acc',
                'full_apr4_decoder_only_no_dur': 'full model, decoder only, no encoder, no button loss, no duration, full dataset 20%, model_checkpoint_1_eps_26233_steps_1.1316_loss_0.6802_acc',
                'light__apr4_autoencoder': 'light model, autoencoder, no duration, light dataset 20%,  losses de button 0 per error',
                'full_apr7_autoencoder': 'full model, autoencoder, no duration, full dataset 20%, autoenc_apr7_full_0_eps_5001_steps_2.2775_loss_0.6992_acc',
                'light_apr7_autoencoder': 'light model, autoencoder, no duration, light dataset, more deviate 0.1, margin 0.1, contour 0.1, autoenc_apr7_light_deviate_14_eps_29653_steps_1.1294_loss_0.8396_acc.pth',
                'full_apr24_hi_losses': 'full model, autoencoder, more deviate 0.1, margin 0.1, contour 0.1, full dataset 20%, autoenc_apr24_full_hi_losses_11_eps_121112_steps_0.359_loss_0.8954_acc.pth',
                'encoder_only': 'encoder only, full dataset, light model, for testing',
                'full_giantmidi': 'full model, autoencoder, no duration, giantMIDI dataset, multi-step contour 0.1, button held margin 0.1, deviate 0.1, data augmentation',
                'ultra_full': 'ultra hi full model, autoencoder, no duration, giantMIDI dataset, multi-step contour 0.1, button held margin 0.1, deviate 0.1, data augmentation',
                'no_dtime': 'no dtime model, autoencoder, no duration, full dataset, no dtime',
                'encoder_only_antic': 'encoder only, with an error in dtime embedding'
            }     

#===================================================================================================

MODELS_FILE_NAMES = {'tester': './save_models/tester.pth',
                     'light': './save_models/model_checkpoint_39_eps_3292342_steps_0.6787_loss_0.783_acc.pth',
                     'full': './save_models/model_checkpoint_39_eps_3292342_steps_0.6787_loss_0.783_acc.pth',
                     'encoder_light_mar14': './save_models/encoder_light_mar14.pth',
                     'encoder full': './save_models/model_checkpoint_9_eps_759772_steps_0.0039_loss_0.0_acc.pth',
                     'encoder_orig': './save_models/encoder_orig_mar15.pth',
                     'encoder_new_loss_mar17': './save_models/encoder_new_loss_mar17.pth',
                     'full_mar18': './save_models/full_mar18.pth',
                     'light_mar24': './save_models/model_checkpoint_4_eps_337677_steps_3.2628_loss_0.8016_acc.pth',
                     'light_mar26_only_butt_loss': './save_models/model_checkpoint_39_eps_82603_steps_0.0107_loss_0.0158_acc.pth',
                     'light_mar26_only_recon_loss': './save_models/model_checkpoint_34_eps_72013_steps_0.3459_loss_0.8914_acc.pth',
                     'light_mar27_decoder_only': './save_models/light_mar27_decoder_only.pth',
                     'big_mar28_decoder_only': './save_models/big_mar28_decoder_only.pth',
                     'light_apr3_decoder_only_no_dur_cont_dtime': './save_models/light_apr3_decoder_only_no_dur_cont_dtime.pth',
                     'full_apr4_decoder_only_no_dur': './save_models/full_apr4_decoder_only_no_dur.pth',
                     'light__apr4_autoencoder': './save_models/light__apr4_autoencoder.pth',
                     'full_apr7_autoencoder': './save_models/autoenc_apr7_full_0_eps_30001_steps_0.6359_loss_0.8169_acc.pth',
                     'light_apr7_autoencoder': './save_models/autoenc_apr7_light_deviate_39_eps_82603_steps_0.4157_loss_0.9494_acc.pth',
                     'full_apr24_hi_losses': './save_models/autoenc_apr24_full_hi_losses_11_eps_121112_steps_0.359_loss_0.8954_acc.pth',
                     'full_tester': './save_models/mai6_original_m_lo_losses_full_dataset_7_eps_2129_steps_0.6687_loss_0.817_acc.pth',
                     'encoder_only': './save_models/encoder/mai27_encoder_button_concent_x10_97_eps_9798_steps_0.0013_loss_0.0_acc.pth',
                     'full_giantmidi': './save_models/mai14_hi_m_giantMIDI_data_990_eps_5946_steps_0.1842_loss_0.9458_acc.pth',
                     'ultra_full': './save_models/mai24_no_dtime_ultra_m_multi_held_650_eps_4557_steps_0.3407_loss_0.8965_acc.pth',
                     'no_dtime_good_reference': './save_models/mai21_no_dtime_hi_multi_held_2860_eps_20027_steps_0.13_loss_0.959_acc.pth',
                     'no_dtime_19_buttons': './save_models/mai27_19but_pitch_button_correlation_1220_eps_159951_steps_0.191_loss.pth',
                     'encoder_only_antic': './save_models/encoder/mai13_encoder_only_lo_m_multi_&_held_margin_1.0_deviate_1.0_16_eps_657_steps_0.1721_loss_0.0_acc.pth',
                     'mai27_big_m_5buttons_original_loss': './save_models/mai27_big_m_5buttons_original_loss_3.pth',
                   }

#===================================================================================================

MODELS_PARAMETERS = {
    'tester': {
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.01,
        'loss_deviate': 0.01
    },
    'light': {
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },
    'full': {
        'seq_len': 2048,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.01,
        'loss_deviate': 0.01
        },
    'encoder_light_mar14': {
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },
    'encoder full': {
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },
    'encoder_orig': {
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },
    'encoder_new_loss_mar17': {
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },
    'full_mar18': {
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },
    'light_mar24': {
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },
    'light_mar26_only_butt_loss': {
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },
    'light_mar26_only_recon_loss': {
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },
    'light_mar27_decoder_only': {
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },
    'big_mar28_decoder_only': {
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },
    'light_apr3_decoder_only_no_dur_cont_dtime': {
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },
    'full_apr4_decoder_only_no_dur': {
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },          
    'light__apr4_autoencoder': {
        'seq_len': 2048,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },
    'full_apr7_autoencoder': {
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },
    'light_apr7_autoencoder': {
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01,
        'dataset': 'light'
        },
    'full_apr24_hi_losses': {
        'seq_len': 512,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'full'
        },
    'full_tester': {
        'seq_len': 512,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'full'
        },
    'encoder_only': {
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'full'
        },
    'full_giantmidi': {
        'seq_len': 512,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'full'
        },
    'no_dtime_good_reference': {
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'full'
        },
    'no_dtime_19_buttons': {
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'full'
        },    
    'encoder_only_antic': {
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'full'
        },
    'mai27_big_m_5buttons_original_loss': {
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'full'
    },    
    'ultra_full': {
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 6,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'full'
    },   
    'decoder_only_2_buttons': {
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'giantmidi_full',
        'num_buttons': 2
        },    
}


MODELS_TYPES = {'tester': 'autoencoder_w_encoder_antic',
               'light': 'autoencoder_w_encoder_antic',
               'full': 'autoencoder_w_encoder_antic',
               'encoder_light_mar14': 'encoder_only',
               'encoder full': 'encoder_only',
                'encoder_orig': 'encoder_only',
                'encoder_new_loss_mar17': 'encoder_only',
                'full_mar18': 'autoencoder_w_encoder_antic',
                'light_mar24': 'autoencoder_w_encoder_antic',
                'light_mar26_only_butt_loss': 'autoencoder_w_encoder_antic',
                'light_mar26_only_recon_loss': 'autoencoder_w_encoder_antic',
                'light_mar27_decoder_only': 'decoder_only',
                'big_mar28_decoder_only': 'decoder_only',
                'light_apr3_decoder_only_no_dur_cont_dtime': 'decoder_only',
                'full_apr4_decoder_only_no_dur': 'decoder_only',
                'light__apr4_autoencoder': 'autoencoder_w_encoder_antic',
                'full_apr7_autoencoder': 'autoencoder_w_encoder_antic',
                'light_apr7_autoencoder': 'autoencoder_w_encoder_antic',
                'full_apr24_hi_losses': 'autoencoder_w_encoder_antic',
                'encoder_only': 'encoder_only',
                'full_giantmidi': 'autoencoder_w_encoder_antic',
                'ultra_full': 'autoencoder_no_dtime',
                'full_tester': 'autoencoder_w_encoder_antic',
                'no_dtime_good_reference': 'autoencoder_no_dtime',
                'no_dtime_19_buttons': 'autoencoder_no_dtime',
                'encoder_only_antic': 'encoder_only_antic',
                'mai27_big_m_5buttons_original_loss': 'autoencoder_no_dtime',
                'decoder_only_2_buttons': 'decoder_only_2_buttons',
            }  
#===================================================================================================

def detect_model_type(model):

    seq_len = model.max_seq_len
    pad_idx = model.pad_value

    model_type = 'unknown'
    model_idx = -1

    for i, np in enumerate(MODELS_PARAMETERS.items()):
        if np[1]['seq_len'] == seq_len and np[1]['pad_idx'] == pad_idx:
            model_type = np[0]
            model_idx = i
            break

    return model_type, model_idx

#===================================================================================================
# This is the end of models Python module
#===================================================================================================