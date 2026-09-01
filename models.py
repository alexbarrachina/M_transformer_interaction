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


#===================================================================================================

MODELS_PARAMETERS = {
'tester': {
        'model_type': 'autoencoder_w_encoder_antic',
        'description': 'tester. light model not trained',
        'ckpt_file_name': './save_models/tester.pth',
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
        'model_type': 'autoencoder_w_encoder_antic',
        'description': ' light test version (without velocity) trained for 40 epochs on full Monster Piano dataset.',
        'ckpt_file_name': './save_models/model_checkpoint_39_eps_3292342_steps_0.6787_loss_0.783_acc.pth',
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
        'model_type': 'autoencoder_w_encoder_antic',
        'description': 'the full model, trained for 40 epochs on full Monster Piano dataset.',
        'ckpt_file_name': './save_models/model_checkpoint_39_eps_3292342_steps_0.6787_loss_0.783_acc.pth',
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
        'model_type': 'encoder_only',
        'description': 'encoder only, for testing melodic contour loss., light dataset, 39_eps_82603_steps_0.0038_loss_0.0_acc',
        'train_log': '39_eps_82603_steps_0.0038_loss_0.0_acc',
        'ckpt_file_name': './save_models/encoder_light_mar14.pth',
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },
    'encoder_full': {
        'model_type': 'encoder_only',
        'description': 'encoder only, for testing melodic contour loss., full dataset',
        'ckpt_file_name': './save_models/model_checkpoint_9_eps_759772_steps_0.0039_loss_0.0_acc.pth',
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
        'model_type': 'encoder_only',
        'description': 'encoder only, original genie implemetation, full dataset, light model',
        'train_log': '5_eps_422096_steps_0.0021_loss_0.0_acc',
        'ckpt_file_name': './save_models/encoder_orig_mar15.pth',
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
        'model_type': 'encoder_only',
        'description': 'encoder only, new loss functionalities, full dataset, light model',
        'train_log': '9_eps_82603_steps_0.0114_loss_0.0_acc',
        'ckpt_file_name': './save_models/encoder_new_loss_mar17.pth',
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
        'model_type': 'autoencoder_w_encoder_antic',
        'description': 'full model, new loss functionalities, full dataset',
        'train_log': '2_eps_42335_steps_0.0129_loss_0.0_acc',
        'ckpt_file_name': './save_models/full_mar18.pth',
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
        'model_type': 'autoencoder_w_encoder_antic',
        'description': 'light model, new loss functionalities, full dataset',
        'train_log': '3_eps_253258_steps_2.4025_loss_0.8244_acc',
        'ckpt_file_name': './save_models/model_checkpoint_4_eps_337677_steps_3.2628_loss_0.8016_acc.pth',
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
        'model_type': 'autoencoder_w_encoder_antic',
        'description': 'light model, only button loss, no recons, light dataset, 39_eps_82603_steps_0.0107_loss_0.0158_acc',
        'train_log': '39_eps_82603_steps_0.0107_loss_0.0158_acc',
        'ckpt_file_name': './save_models/model_checkpoint_39_eps_82603_steps_0.0107_loss_0.0158_acc.pth',
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
        'model_type': 'autoencoder_w_encoder_antic',
        'description': 'light model, only recons loss, no button loss , light dataset, 34_eps_72013_steps_0.3459_loss_0.8914_acc',
        'train_log': '34_eps_72013_steps_0.3459_loss_0.8914_acc',
        'ckpt_file_name': './save_models/model_checkpoint_34_eps_72013_steps_0.3459_loss_0.8914_acc.pth',
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
        'model_type': 'decoder_only',
        'description': 'light model, decoder only, no encoder, no button loss, light dataset',
        'train_log': '20_eps_1738381_steps_0.7658_loss_0.7768_acc',
        'ckpt_file_name': './save_models/light_mar27_decoder_only.pth',
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
        'model_type': 'decoder_only',
        'description': 'big model, decoder only, no encoder, no button loss, big dataset, early interruption on',
        'train_log': '7_eps_153170_steps_0.8395_loss_0.7523_acc',
        'ckpt_file_name': './save_models/big_mar28_decoder_only.pth',
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
        'model_type': 'decoder_only',
        'description': 'light model, decoder only, no encoder, no button loss, no duration, light dataset 1%, interruption before overfitting on',
        'train_log': '12_eps_25417_steps_0.8986_loss_0.7389_acc',
        'ckpt_file_name': './save_models/light_apr3_decoder_only_no_dur_cont_dtime.pth',
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
        'model_type': 'decoder_only',
        'description': 'full model, decoder only, no encoder, no button loss, no duration, full dataset 20%',
        'train_log': '1_eps_26233_steps_1.1316_loss_0.6802_acc',
        'ckpt_file_name': './save_models/full_apr4_decoder_only_no_dur.pth',
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
        'model_type': 'autoencoder_w_encoder_antic',
        'description': 'light model, autoencoder, no encoder, no button loss, light dataset',
        'train_log': '2_eps_1738381_steps_0.7658_loss_0.7768_acc',
        'ckpt_file_name': './save_models/light__apr4_autoencoder.pth',
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
        'model_type': 'autoencoder_w_encoder_antic',
        'description': 'full model, autoencoder, no encoder, no button loss, full dataset',
        'train_log': '1_eps_26233_steps_1.1316_loss_0.6802_acc',
        'ckpt_file_name': './save_models/autoenc_apr7_full_0_eps_30001_steps_0.6359_loss_0.8169_acc.pth',
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
        'model_type': 'autoencoder_w_encoder_antic',
        'description': 'light model, autoencoder, no encoder, no button loss, light dataset',
        'train_log': '39_eps_82603_steps_0.4157_loss_0.9494_acc',
        'ckpt_file_name': './save_models/autoenc_apr7_light_deviate_39_eps_82603_steps_0.4157_loss_0.9494_acc.pth',
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
        'model_type': 'autoencoder_w_encoder_antic',
        'description': 'full model, autoencoder, no encoder, no button loss, full dataset',
        'train_log': '11_eps_121112_steps_0.359_loss_0.8954_acc',
        'ckpt_file_name': './save_models/autoenc_apr24_full_hi_losses_11_eps_121112_steps_0.359_loss_0.8954_acc.pth',
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
        'model_type': 'autoencoder_w_encoder_antic',
        'description': 'full model, autoencoder, no encoder, no button loss, full dataset',
        'train_log': '7_eps_2129_steps_0.6687_loss_0.817_acc',
        'ckpt_file_name':'./save_models/mai6_original_m_lo_losses_full_dataset_7_eps_2129_steps_0.6687_loss_0.817_acc.pth',
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
        'model_type': 'encoder_only',
        'description': 'encoder only, full dataset, light model, for testing',
        'train_log': '97_eps_9798_steps_0.0013_loss_0.0_acc',
        'ckpt_file_name': './save_models/encoder/mai27_encoder_button_concent_x10_97_eps_9798_steps_0.0013_loss_0.0_acc.pth',
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
        'model_type': 'autoencoder_w_encoder_antic',
        'description': 'full model, autoencoder, no encoder, no button loss, full dataset',
        'train_log': '990_eps_5946_steps_0.1842_loss_0.9458_acc',
        'ckpt_file_name': './save_models/mai14_hi_m_giantMIDI_data_990_eps_5946_steps_0.1842_loss_0.9458_acc.pth',
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
        'model_type': 'autoencoder_no_dtime',
        'description': 'full model, autoencoder, no encoder, no button loss, full dataset',
        'train_log': '2860_eps_20027_steps_0.13_loss_0.959_acc',
        'ckpt_file_name': './save_models/mai21_no_dtime_hi_multi_held_2860_eps_20027_steps_0.13_loss_0.959_acc.pth',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'full',
        'num_buttons': 12,
        },
    'no_dtime_19_buttons': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'full model, autoencoder, no encoder, no button loss, full dataset',
        'train_log': '19_eps_159951_steps_0.191_loss_0.959_acc',
        'ckpt_file_name': './save_models/mai27_19but_pitch_button_correlation_1220_eps_159951_steps_0.191_loss.pth',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'full',
        'num_buttons': 19,
        },   
    'no_dtime_8_buttons': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'full model, autoencoder, no encoder, no button loss, full dataset',
        'train_log': '',
        'ckpt_file_name': '',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'full',
        'num_buttons': 8,
        },   
    'encoder_only_antic': {
        'model_type': 'encoder_only_antic',
        'description': 'encoder only, with an error in dtime embedding, full dataset, light model',
        'train_log': '16_eps_657_steps_0.1721_loss_0.0_acc',
        'ckpt_file_name': './save_models/encoder/mai13_encoder_only_lo_m_multi_&_held_margin_1.0_deviate_1.0_16_eps_657_steps_0.1721_loss_0.0_acc.pth',
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
        'model_type': 'autoencoder_no_dtime',
        'description': 'full model, autoencoder, no encoder, no button loss, full dataset',
        'train_log': '3_eps_1220_steps_0.013_loss_0.959_acc',
        'ckpt_file_name': './save_models/model_checkpoint_39_eps_82603_steps_0.0047_loss_0.0_acc.pth',
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
        'model_type': 'autoencoder_no_dtime',
        'description': 'full model, autoencoder, no encoder, no button loss, full dataset',
        'train_log': '3_eps_1220_steps_0.013_loss_0.959_acc',
        'ckpt_file_name': './save_models/model_checkpoint_39_eps_82603_steps_0.0038_loss_0.0_acc.pth',
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

    'autoencoder_button_held': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'autoencoder, button held loss',
        'train_log': 'not yet trained',
        'ckpt_file_name': './save_models/autoencoder_button_held_900_eps_1802_steps_7.8132_loss_0.9062_acc.pth',
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01,
        'loss_button_held': 0.1,
        "loss_multi_step_perc": 0.,
        "loss_contour_perc": 1., # original genie contour loss
        "save_every": 15, # in epochs
        },  

    'win_correlation_loss': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'autoencoder, windowed correlation loss, no other contour losses',
        'train_log': 'not yet trained',
        'ckpt_file_name': './save_models/win_correlation_loss_120_eps_242_steps_0.8973_loss_0.5875_acc.pth',
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.,
        'loss_deviate': 0.01,
        'loss_button_held': 0.,
        "loss_multi_step_perc": 0.,
        "loss_contour_perc": 1., # original genie contour loss
        "loss_window_corr": 1.,
        "loss_recons": 0.5, # 1., original # Reconstruction loss
        "save_every": 15, # in epochs
        },   

    'original_genie': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'autoencoder, original genie loss, no other contour losses',
        'train_log': 'not yet trained',
        'ckpt_file_name': './save_models/original_genie_270_eps_542_steps_0.6156_loss_0.7336_acc.pth',
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 1.0,
        'loss_contour': 1.,
        'loss_deviate': 1.0,
        'loss_button_held': 0.,
        "loss_multi_step_perc": 0.,
        "loss_contour_perc": 1., # original genie contour loss
        "loss_window_corr": 0.,
        "loss_recons": 0.5, # 1., original # Reconstruction loss
        "save_every": 15, # in epochs
        },  
    'original_genie+loss_button_held': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'autoencoder, original genie loss, no other contour losses',
        'train_log': 'not yet trained',
        'ckpt_file_name': './save_models/original_genie+loss_button_held_105_eps_212_steps_0.7841_loss_0.6609_acc.pth',
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 1.0,
        'loss_contour': 1.,
        'loss_deviate': 1.0,
        'loss_button_held': 10.0,
        "loss_multi_step_perc": 0.,
        "loss_contour_perc": 1., # original genie contour loss
        "loss_window_corr": 0.,
        "loss_recons": 0.5, # 1., original # Reconstruction loss
        "save_every": 15, # in epochs
        },  
    'original_genie+loss_norm_pos': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'autoencoder, original genie loss, and normalized position loss',
        'train_log': 'not yet trained',
        'ckpt_file_name': './save_models/original_genie+loss_norm_pos_285_eps_572_steps_1.3977_loss_0.6414_acc.pth',
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 1.0,
        'loss_contour': 1.,
        'loss_deviate': 1.0,
        'loss_button_held': 1.0,
        "loss_multi_step_perc": 0.,
        "loss_contour_perc": 1., # original genie contour loss
        "loss_window_corr": 0.,
        "loss_recons": 1., # 1., original # Reconstruction loss
        "loss_norm_pos": 1.0,
        "save_every": 15, # in epochs
        },  
    'loss_norm_pos_10x': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'autoencoder, normalized position loss only',
        'train_log': 'not yet trained',
        'ckpt_file_name': './save_models/loss_norm_pos_150_eps_302_steps_9.7449_loss_0.6945_acc.pth',
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.,
        'loss_deviate': 0.0,
        'loss_button_held': 0.0,
        "loss_multi_step_perc": 0.,
        "loss_contour_perc": 1., # original genie contour loss
        "loss_window_corr": 0.,
        "loss_recons": 1., # 1., original # Reconstruction loss
        "loss_norm_pos": 10.0,
        "save_every": 15, # in epochs
        },  
    'test_w_dtime': {
        'model_type': 'autoencoder',
        'description': 'autoencoder, using dtime embedding, small model, original genie loss',
        'train_log': 'not yet trained',
        'ckpt_file_name': './save_models/test_w_dtime_390_eps_391_steps_0.6523_loss_0.7891_acc.pth',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'loss_button_held': 0.1,
        "loss_multi_step_perc": 0.,
        "loss_contour_perc": 1., # original genie contour loss
        "loss_window_corr": 0.,
        "loss_recons": 1., # 1., original # Reconstruction loss
        "save_every": 15, # in epochs
        },
    'contour+loss_norm_pos': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'autoencoder, normalized position loss only',
        'train_log': 'not yet trained',
        'ckpt_file_name': './save_models/contour+loss_norm_pos_585_eps_1172_steps_2.3291_loss_0.8602_acc.pth',
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'loss_button_held': 0.1,
        "loss_multi_step_perc": 0.,
        "loss_contour_perc": 1., # original genie contour loss
        "loss_window_corr": 0.,
        "loss_recons": 1., # 1., original # Reconstruction loss
        "loss_norm_pos": 5.0,
        "save_every": 15, # in epochs
        },
    'no_dtime_2buttons': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'full model, autoencoder, only 2 buttons',
        'train_log': '',
        'ckpt_file_name': './save_models/no_dtime_2buttons_135_eps_136_steps_1.8422_loss_0.4674_acc.pth',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'num_buttons': 2,
        'save_every': 15, # in epochs
        },
    'melody_arrow_v1': {
        'model_type': 'autoencoder_melody',
        'description': 'Melody autoencoder with arrow guidance. Uses melody-only pickles (channel 0). Old implementation, as arrows were treated as continuous scalars like buttons in original',
        'train_log': '',
        'ckpt_file_name': './save_models/melody_arrow_v1_75_eps_152_steps_2.2351_loss_0.8008_acc.pth',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_recons': 1.0,
        'loss_arrow_consistency': 0.1,  # Weight for arrow consistency loss (soft arrows + KL divergence)
        'arrow_soft_temp': 2.0,  # Temperature for soft arrow boundaries (lower = sharper)
        "dataset_train_path": "./Training-Data/giantmidi_full_melody_train", # './Training-Data/asigalov_train'
        "dataset_val_path": "./Training-Data/giantmidi_full_melody_test", # './Training-Data/asigalov_val'

        "save_every": 15, # in epochs
        },
    'melody_arrow_v2': {
        'model_type': 'autoencoder_melody',
        'description': 'Melody autoencoder with arrow guidance. Uses melody-only pickles (channel 0). New implementation, as arrows are treated as discrete embedded values, not continuous scalars like buttons in original',
        'train_log': '',
        'ckpt_file_name': './save_models/melody_arrow_v2_375_eps_376_steps_0.8684_loss_0.9911_acc.pth',
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_recons': 1.0,
        'loss_arrow_consistency': 0.1,  # Weight for arrow consistency loss (soft arrows + KL divergence)
        'arrow_soft_temp': 2.0,  # Temperature for soft arrow boundaries (lower = sharper)
        'pitch_history_dropout': 0.0,  # Dropout rate for pitch embeddings (0.0-1.0) to force arrow reliance
        "dataset_train_path": "./Training-Data/giantmidi_full_melody_train", # './Training-Data/asigalov_train'
        "dataset_val_path": "./Training-Data/giantmidi_full_melody_test", # './Training-Data/asigalov_val'

        "save_every": 15, # in epochs
        },
    'tester_arrow': {
        'model_type': 'autoencoder_melody',
        'description': 'Tester model for arrow consistency loss inspection.',
        'train_log': '',
        'ckpt_file_name': '',
        'seq_len': 32,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_recons': 1.0,
        'loss_arrow_consistency': 0.1,  # Weight for arrow consistency loss (soft arrows + KL divergence)
        'arrow_soft_temp': 2.0,  # Temperature for soft arrow boundaries (lower = sharper)
        "dataset_train_path": "./Training-Data/tester_train", # './Training-Data/asigalov_train'
        "dataset_val_path": "./Training-Data/tester_train", # './Training-Data/asigalov_val'
        "batch_size": 1,
        "save_every": 15, # in epochs
        },
    'melody_arrow_v3': {
        'model_type': 'autoencoder_melody',
        'description': 'Melody autoencoder with arrow guidance. Uses melody-only pickles (channel 0). New implementation, as arrows are treated as discrete embedded values, not continuous scalars like buttons in original. Soft_temp 1.0',
        'train_log': '',
        'ckpt_file_name': './save_models/melody_arrow_v3_795_eps_796_steps_14.3731_loss_0.8396_acc.pth',
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_recons': 1.0,
        'loss_arrow_consistency': 1.0,  # Weight for arrow consistency loss (soft arrows + KL divergence)
        'arrow_soft_temp': 1.0,  # Temperature for soft arrow boundaries (lower = sharper)
        'pitch_history_dropout': 0.0,  # Dropout rate for pitch embeddings (0.0-1.0) to force arrow reliance
        "dataset_train_path": "./Training-Data/giantmidi_full_melody_train", # './Training-Data/asigalov_train'
        "dataset_val_path": "./Training-Data/giantmidi_full_melody_test", # './Training-Data/asigalov_val'

        "save_every": 15, # in epochs
        },
    'melody_arrow_v4': {
        'model_type': 'autoencoder_melody',
        'description': 'Melody autoencoder with arrow guidance + pitch history dropout (30%) to force arrow reliance',
        'train_log': '',
        'ckpt_file_name': './save_models/melody_arrow_v4_540_eps_1084_steps_3.2021_loss_0.9431_acc.pth',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_recons': 1.0,
        'loss_arrow_consistency': 0.5,  # Weight for arrow consistency loss
        'arrow_soft_temp': 2.0,  # Temperature for soft arrow boundaries
        'pitch_history_dropout': 0.3,  # 30% of pitch embeddings are zeroed during training
        "dataset_train_path": "./Training-Data/giantmidi_full_melody_train",
        "dataset_val_path": "./Training-Data/giantmidi_full_melody_test",
        "save_every": 15,
        },
    'melody_arrow_v5': {
        'model_type': 'autoencoder_melody',
        'description': 'Melody autoencoder with fine+coarse arrow guidance (30percent coarse in contiguous spans)',
        'train_log': '',
        'ckpt_file_name': '',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_recons': 1.0,
        'loss_arrow_consistency': 0.2,  # Weight for arrow consistency loss (fine arrows)
        'arrow_soft_temp': 1.0,  # Temperature for soft arrow boundaries
        'pitch_history_dropout': 0.1,  # 30% of pitch embeddings are zeroed during training
        'coarse_arrow_ratio': 0.1,  # 30% of sequence uses coarse arrows in contiguous spans
        "dataset_train_path": "./Training-Data/giantmidi_full_melody_train",
        "dataset_val_path": "./Training-Data/giantmidi_full_melody_test",
        "save_every": 15,
        },
'melody_arrow_v6': {
    'model_type': 'autoencoder_melody',
    'description': 'High accuracy attempt: No dropout, fine arrows only',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 1024,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 6,           # Increased depth
    'heads': 32,
    'loss_recons': 1.0,
    'loss_arrow_consistency': 0.1,  # Minimal arrow constraint
    'arrow_soft_temp': 1.0,         # Sharper boundaries
    'pitch_history_dropout': 0.0,   # No dropout (full context)
    'coarse_arrow_ratio': 0.0,      # Fine arrows only (easier task)
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_test",
    "save_every": 15,
    },
'melody_arrow_v7': {
    'model_type': 'autoencoder_melody',
    'description': 'High accuracy attempt: dropout 10%, fine arrows only',
    'train_log': '',
    'ckpt_file_name': './save_models/melody_arrow_v7_375_eps_752_steps_0.5624_loss_0.969_acc.pth',
    'seq_len': 1024,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 6,           # Increased depth
    'heads': 32,
    'loss_recons': 1.0,
    'loss_arrow_consistency': 0.1,  # Minimal arrow constraint
    'arrow_soft_temp': 1.0,         # Sharper boundaries
    'pitch_history_dropout': 0.1,   # 10% dropout (full context)
    'coarse_arrow_ratio': 0.0,      # Fine arrows only (easier task)
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_test",
    "save_every": 15,
    },
'melody_arrow_v8': {
    'model_type': 'autoencoder_melody',
    'description': 'High accuracy attempt: dropout 10%, coarse arrows 10%',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 1024,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 6,           # Increased depth
    'heads': 32,
    'loss_recons': 1.0,
    'loss_arrow_consistency': 0.1,  # Minimal arrow constraint
    'arrow_soft_temp': 1.0,         # Sharper boundaries
    'pitch_history_dropout': 0.1,   # 10% dropout (full context)
    'coarse_arrow_ratio': 0.1,      # 10% coarse arrows (easier task)
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_test",
    "save_every": 15,
    },
 'melody_arrow_v9': {
    'model_type': 'autoencoder_melody',
    'description': 'High accuracy attempt: dropout 10%, coarse arrows 100%',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 1024,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 6,           # Increased depth
    'heads': 32,
    'loss_recons': 1.0,
    'loss_arrow_consistency': 0.1,  # Minimal arrow constraint
    'arrow_soft_temp': 1.0,         # Sharper boundaries
    'pitch_history_dropout': 0.1,   # 10% dropout (full context)
    'coarse_arrow_ratio': 1.0,      # 100% coarse arrows (easier task)
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_test",
    "save_every": 15,
    },
 'melody_arrow_v10': {
    'model_type': 'AE_melody_w_coarse_arrows',
    'description': 'High accuracy attempt: dropout 0%, coarse arrows 30%',
    'train_log': '',
    'ckpt_file_name': './save_models/melody_arrow_v10_135_eps_272_steps_0.4089_loss_0.8658_acc.pth',
    'seq_len': 1024,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,           
    'heads': 32,
    'loss_recons': 1.0,
    'loss_arrow_consistency': 0.1,  # Minimal arrow constraint
    'arrow_soft_temp': 2.0,         # Sharp boundaries
    'pitch_history_dropout': 0.0,   # 10% dropout (full context)
    'coarse_arrow_ratio': 0.3,      # 100% coarse arrows (easier task)
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_test",
    "save_every": 15,
    },
'melody_arrow_v11': {
    'model_type': 'AE_melody_w_coarse_arrows',
    'description': 'Options A+B+D: semantic init, direction loss, coarse direction aux loss',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 1024,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,           
    'heads': 32,
    'loss_recons': 1.0,
    'loss_arrow_consistency': 0.1,  # Direction-based comparison (Option B)
    'loss_coarse_direction': 0.5,   # Explicit coarse direction supervision (Option D)
    'arrow_soft_temp': 2.0,         # Sharp boundaries
    'pitch_history_dropout': 0.0,   # Full context
    'coarse_arrow_ratio': 0.3,      # 30% coarse arrows
    'no_influence_ratio': 0.0,      # No "no influence" arrows
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_test",
    "save_every": 15,
    },

'melody_arrow_v12': {
    'model_type': 'AE_melody_w_coarse_arrows',
    'description': 'Full arrow palette: 45% fine, 30% coarse, 25% no-influence',
    'train_log': '',
    'ckpt_file_name': './save_models/melody_arrow_v12_225_eps_452_steps_0.3688_loss_0.967_acc.pth',
    'seq_len': 1024,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,           
    'heads': 32,
    'loss_recons': 1.0,
    'loss_arrow_consistency': 0.1,  # Direction-based comparison (Option B)
    'loss_coarse_direction': 0.1,   # Explicit coarse direction supervision (Option D)
    'arrow_soft_temp': 2.0,         # Sharp boundaries
    'pitch_history_dropout': 0.0,   # Full context
    'coarse_arrow_ratio': 0.3,      # 30% coarse arrows
    'no_influence_ratio': 0.25,     # 25% no-influence arrows (model decides freely)
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_test",
    "save_every": 15,
    },
# Harmony-conditioned autoencoder (Tonnetz conditioning)
'autoenc_no_dtime_harmony_v1': {
    'model_type': 'autoencoder_no_dtime_harmony',
    'description': 'Autoencoder with Tonnetz harmony conditioning (decoder-only). Tester light model.',
    'train_log': '',
    'ckpt_file_name': './save_models/autoenc_no_dtime_harmony_v1_40_eps_533_steps_0.846_loss_0.7605_acc.pth',
    'seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,
    'heads': 32,
    'num_buttons': 12,
    # Loss weights
    'loss_recons': 1.0,
    'loss_margin': 0.1,
    'loss_deviate': 0.1,
    'loss_contour': 0.1,
    'loss_button_held': 0.0,
    'loss_norm_pos': 0.0,
    'loss_pitch_button': 0.0,
    'loss_button_concentration': 0.0,
    'loss_window_corr': 0.0,
    # Contour loss components
    'loss_contour_perc': 0.0,
    'loss_multi_step_perc': 1.0,
    'loss_interval_perc': 0.0,
    'loss_shape_perc': 0.0,
    # Dataset paths (harmony-augmented pickles)
    "dataset_train_path": "./Training-Data/giantmidi_full_harmony_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_harmony_test",
    # Training settings
    "save_every": 5,
    "batch_size": 8,
    },
# Harmony-conditioned autoencoder (Tonnetz conditioning)
'autoenc_just_harmony_v1b': {
    'model_type': 'autoencoder_just_harmony',
    'description': 'no harmony conditioning. With pitch augmentation, chord augmentation.',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 2048, 
    'num_layers': 4,
    'heads': 32,
    # Loss weights
    'loss_recons': 1.0,
    # Dataset paths (harmony-augmented pickles)
    "dataset_train_path": "./Training-Data/giantmidi_full_harmony_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_harmony_test",
    # Training settings
    "save_every": 10,
    "batch_size": 24,
    "num_workers": 10,
    },

'autoenc_no_dtime_harmony_v1': {
    'model_type': 'autoencoder_no_dtime_harmony',
    'description': ' with harmony conditioning and button guidance.',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 2048, 
    'num_layers': 4,
    'heads': 32,
    # Loss weights
    'loss_recons': 1.0,
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    "loss_multi_step_perc": 1.,
    # Dataset paths (harmony-augmented pickles)
    "dataset_train_path": "./Training-Data/giantmidi_full_harmony_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_harmony_test",
    # Training settings
    "save_every": 5,
    "batch_size": 24,
    "num_workers": 10,
    },

'AE_arrows_and_buttons_v1': {
    'model_type': 'AE_arrows_and_buttons',
    'description': ' with arrows and buttons guidance.',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 2048, 
    'num_layers': 4,
    'heads': 32,
    # Loss weights
    'loss_recons': 1.0,
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    'loss_arrow_consistency': 0.1,  # Arrow consistency loss
    'arrow_soft_temp': 2.0,         # Sharp boundaries
    'pitch_history_dropout': 0.0,   # Full context
    "loss_multi_step_perc": 1.,
    # Dataset paths (harmony-augmented pickles)
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_acc_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_acc_test",
    # Training settings
    "save_every": 5,
    "batch_size": 24,
    "num_workers": 10,
    },
'AE_arrows_and_buttons_v2': {
    'model_type': 'AE_arrows_and_buttons',
    'description': ' with arrows and buttons guidance. 8 buttons.',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 512,
    'num_buttons': 8,
    # Loss weights
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    'loss_button_held': 0.,
    "loss_multi_step_perc": 0.,
    "loss_contour_perc": 1., # original genie contour loss
    "loss_window_corr": 0.,
    "loss_recons": 1., # 1., original # Reconstruction loss
    'loss_arrow_consistency': 0.1,  # Arrow consistency loss
    'arrow_soft_temp': 2.0,         # Sharp boundaries
    'pitch_history_dropout': 0.0,   # Full context
    "loss_multi_step_perc": 1.,
    # Dataset paths (harmony-augmented pickles)
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_acc_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_acc_test",
    # Training settings
    "save_every": 5,
    "batch_size": 20,
    "num_workers": 8,
    },

'AE_arrows_and_buttons_light_v1': {
    'model_type': 'AE_arrows_and_buttons',
    'description': 'just reconstruction loss, light model',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 512,
    'emb_dim': 512, 
    'num_buttons': 8,
    # Loss weights
    'loss_margin': 0.0,
    'loss_contour': 0.0,
    'loss_deviate': 0.0,
    "loss_recons": 1., # 1., original # Reconstruction loss
    'loss_arrow_consistency': 0.0,  # Arrow consistency loss
    'arrow_soft_temp': 2.0,         # Sharp boundaries
    'pitch_history_dropout': 0.0,   # Full context
    # Dataset paths (harmony-augmented pickles)
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_acc_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_acc_test",
    # Training settings
    "save_every": 5,
    "batch_size": 24,
    "num_workers": 8,
    },
'AE_arrows_and_buttons_light_v2': {
    'model_type': 'AE_arrows_and_buttons',
    'description': ' recons+contour loss, light model',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 512,
    'emb_dim': 512, 
    'num_buttons': 8,
    # Loss weights
    'loss_margin': 0.0,
    'loss_contour': 0.1,
    'loss_deviate': 0.0,
    "loss_recons": 0.5, # 1., original # Reconstruction loss
    'loss_arrow_consistency': 0.0,  # Arrow consistency loss
    'arrow_soft_temp': 2.0,         # Sharp boundaries
    'pitch_history_dropout': 0.0,   # Full context
    # Dataset paths (harmony-augmented pickles)
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_acc_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_acc_test",
    # Training settings
    "save_every": 105, # just testing. No savings needed.
    "batch_size": 24,
    "num_workers": 8,
    },
'AE_arrows_and_buttons_light_v3': {
    'model_type': 'AE_arrows_and_buttons',
    'description': ' recons+contour+arrow consistency loss, light model',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 512,
    'emb_dim': 512, 
    'num_buttons': 8,
    # Loss weights
    'loss_margin': 0.0,
    'loss_contour': 0.1,
    'loss_deviate': 0.0,
    "loss_recons": 1.0, # 1., original # Reconstruction loss
    'loss_arrow_consistency': 0.1,  # Arrow consistency loss
    'arrow_soft_temp': 2.0,         # Sharp boundaries
    'pitch_history_dropout': 0.0,   # Full context
    # Dataset paths (harmony-augmented pickles)
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_acc_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_acc_test",
    # Training settings
    "save_every": 105, # just testing. No savings needed.
    "batch_size": 24,
    "num_workers": 8,
    },
'AE_arrows_and_buttons_light_v4': {
    'model_type': 'AE_arrows_and_buttons',
    'description': ' recons+contour+arrow consistency loss, light model',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 1024,
    'emb_dim': 2048, 
    'num_buttons': 8,
    # Loss weights
    'loss_margin': 0.0,
    'loss_contour': 0.1,
    'loss_deviate': 0.0,
    "loss_recons": 1.0, # 1., original # Reconstruction loss
    'loss_arrow_consistency': 0.1,  # Arrow consistency loss
    'arrow_soft_temp': 2.0,         # Sharp boundaries
    'pitch_history_dropout': 0.0,   # Full context
    # Dataset paths (harmony-augmented pickles)
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_acc_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_acc_test",
    # Training settings
    "save_every": 105, # just testing. No savings needed.
    "batch_size": 16,
    "num_workers": 8,
    },

'AE_arrows_and_buttons_simpler_v1': {
    'model_type': 'AE_arrows_and_buttons_simpler',
    'description': ' simpler model',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 512,
    'emb_dim': 512, 
    'num_buttons': 8,
    # Loss weights
    'loss_margin': 0.0,
    'loss_contour': 0.1,
    'loss_deviate': 0.0,
    "loss_recons": 1.0, # 1., original # Reconstruction loss
    'loss_arrow_consistency': 0.1,  # Arrow consistency loss
    'arrow_soft_temp': 2.0,         # Sharp boundaries
    'pitch_history_dropout': 0.0,   # Full context
    # Dataset paths (harmony-augmented pickles)
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_acc_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_acc_test",
    # Training settings
    "save_every": 105, # just testing. No savings needed.
    "batch_size": 24,
    "num_workers": 8,
    },
'no_dtime_good_reference_pretrain': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'full model, autoencoder, no encoder, no button loss, full dataset',
        'train_log': '2860_eps_20027_steps_0.13_loss_0.959_acc',
        'ckpt_file_name': './save_models/no_dtime_good_reference_pretrain_85_eps_344_steps_0.5602_loss_0.8237_acc.pth',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        # Training settings
        "save_every": 5, 
        "batch_size": 16,
        "num_workers": 8,
        "num_buttons": 24,
    },
'AE_no_conditioning_tester': {
        'model_type': 'AE_no_conditioning',
        'description': 'full model, autoencoder, no encoder, no button loss, full dataset',
        'train_log': '',
        'ckpt_file_name': '',
        'seq_len': 512,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        "dataset_train_path": "./Training-Data/giantmidi_full_melody_acc_train",
        "dataset_val_path": "./Training-Data/giantmidi_full_melody_acc_test",
        # Training settings
        "save_every": 10005, # just testing. No savings needed.
        "batch_size": 24,
        "num_workers": 10,
    },

'melody_arrow_v7_light': {
    'model_type': 'autoencoder_melody',
    'description': 'High accuracy attempt: dropout 10%, fine arrows only light model',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 512,
    'num_layers': 6,           # Increased depth
    'heads': 32,
    'loss_recons': 1.0,
    'loss_arrow_consistency': 0.1,  # Minimal arrow constraint
    'arrow_soft_temp': 1.0,         # Sharper boundaries
    'pitch_history_dropout': 0.1,   # 10% dropout (full context)
    'coarse_arrow_ratio': 0.0,      # Fine arrows only (easier task)
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_only_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_only_test",
    "save_every": 10005, # just testing. No savings needed.
    "batch_size": 24,
    "num_workers": 10,
    },
'AE_arrows_and_buttons_concatenated': {
    'model_type': 'AE_arrows_and_buttons_concatenated',
    'description': 'concatenated arrows and buttons model',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 1048,
    'pad_idx': 128,
    'emb_dim': 2028,
    'num_layers': 4,           # Increased depth
    'heads': 32,
    'loss_recons': 1.0,
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    'loss_arrow_consistency': 0.1,  # Minimal arrow constraint
    'arrow_soft_temp': 2.0,         # Sharper boundaries
    'pitch_history_dropout': 0.1,   # 10% dropout (full context)
    'coarse_arrow_ratio': 0.0,      # Fine arrows only (easier task)
    "dataset_train_path": "./Training-Data/giantmidi_full_accom_only_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_accom_only_test",
    "save_every": 10, # just testing. No savings needed.
    "batch_size": 8,
    "num_workers": 4,
    },
'AE_arrows_and_buttons_pretrained_v1': {
    'model_type': 'AE_arrows_and_buttons',
    'description': ' recons+contour+arrow consistency loss, full model, pretrained encoder',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_arrows_and_buttons_pretrained_v1_162_eps_652_steps_0.3209_loss_0.9013_acc.pth',
    'seq_len': 1024,
    'emb_dim': 2048, 
    'num_layers': 4,           # Increased depth
    'heads': 32,
    # Loss weights
    'loss_margin': 0.0,
    'loss_contour': 0.1,
    'loss_deviate': 0.0,
    "loss_recons": 1.0, # 1., original # Reconstruction loss
    'loss_arrow_consistency': 0.1,  # Arrow consistency loss
    'arrow_soft_temp': 2.0,         # Sharp boundaries
    'pitch_history_dropout': 0.0,   # Full context
    # Dataset paths (harmony-augmented pickles)
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_acc_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_acc_test",
    # Training settings
    "save_every": 3, 
    "batch_size": 16,
    "num_workers": 8,
    "num_buttons": 24,
    "unfreeze_encoder_after_n_epochs": 3000000, # never unfreeze
    },

'AE_mixed_vocab_v1': {
    'model_type': 'AE_mixed_vocab',
    'description': 'Mixed vocabulary: arrows (1-7) + buttons (8-19) in single key sequence',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_mixed_vocab_v1_70_eps_284_steps_1.2954_loss_0.7118_acc.pth',
    'seq_len': 1024,
    'emb_dim': 2048,
    'num_layers': 4,
    'heads': 32,
    'num_buttons': 12,  # Buttons will use indices 8-19
    'num_arrows': 7,
    # Loss weights
    'loss_recons': 1.0,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    'loss_arrow_consistency': 0.1,
    'contour_max_steps': 5,  # Multi-step contour loss lookback
    'pitch_history_dropout': 0.1,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_acc_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_acc_test",
    # Training settings
    "save_every": 5,
    "batch_size": 16,
    "num_workers": 8,
    "unfreeze_encoder_after_n_epochs": 3000000,  # never unfreeze (encoder pretrained)
    },
'AE_mixed_vocab_tester_v1': {
    'model_type': 'AE_mixed_vocab',
    'description': 'Mixed vocabulary: light model, smart initialization for buttons',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_mixed_vocab_tester_v1_135_eps_952_steps_1.4543_loss_0.7025_acc.pth',
    'seq_len': 512,
    'emb_dim': 512,
    'num_layers': 4,
    'heads': 32,
    'num_buttons': 12,  # Buttons will use indices 8-19
    'num_arrows': 7,
    # Loss weights
    'loss_recons': 1.0,
    'loss_contour': 0,  # Reduced (encoder-based, constant)
    'loss_deviate': 0,
    'loss_pred_contour': 2.0,  # NEW: trains decoder to follow button shape
    'loss_arrow_consistency': 0.1,
    'contour_max_steps': 5,  # Multi-step contour loss lookback
    'pitch_history_dropout': 0.3,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_acc_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_acc_test",
    # Training settings
    "save_every": 5,
    "batch_size": 16,
    "num_workers": 8,
    "unfreeze_encoder_after_n_epochs": 3000000,  # never unfreeze (encoder pretrained)
    },
'AE_mixed_vocab_v2': {
    'model_type': 'AE_mixed_vocab',
    'description': 'Mixed vocabulary: more contour loss + predicted contour',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_mixed_vocab_v2_70_eps_284_steps_1.8114_loss_0.7069_acc.pth',
    'seq_len': 1024,
    'emb_dim': 2048,
    'num_layers': 4,
    'heads': 32,
    'num_buttons': 12,  # Buttons will use indices 8-19
    'num_arrows': 7,
    # Loss weights
    'loss_recons': 1.0,
    'loss_contour': 0,  # Reduced (encoder-based, constant)
    'loss_deviate': 0,
    'loss_pred_contour': 2.0,  # NEW: trains decoder to follow button shape
    'loss_arrow_consistency': 0.1,
    'contour_max_steps': 5,  # Multi-step contour loss lookback
    'pitch_history_dropout': 0.3,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_melody_acc_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_melody_acc_test",
    # Training settings
    "save_every": 5,
    "batch_size": 16,
    "num_workers": 8,
    "unfreeze_encoder_after_n_epochs": 3000000,  # never unfreeze (encoder pretrained)
    },
'no_dtime_good_reference_pretrain_tester': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'full model, autoencoder, light model, to extract encoder weights',
        'train_log': '',
        'ckpt_file_name': './save_models/no_dtime_good_reference_pretrain_tester_45_eps_230_steps_1.844_loss_0.7385_acc.pth',
        'seq_len': 512,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        "dataset_train_path": "./Training-Data/giantmidi_full_accom_only_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_accom_only_test.pickle",
        # Training settings
        "save_every": 15, 
        "batch_size": 24,
        "num_workers": 10,
        "num_buttons": 12,
    },
'no_dtime_button_concentration_tester': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'light model, autoencoder, button concentration loss',
        'train_log': '',
        'ckpt_file_name': './save_models/no_dtime_button_concentration_tester_15_eps_96_steps_1.7993_loss_0.7272_acc.pth',
        'seq_len': 512,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'loss_button_concentration': 0.1,
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        # Training settings
        "save_every": 15, 
        "batch_size": 24,
        "num_workers": 10,
        "num_buttons": 12,
    },
'no_dtime_button_concentration_tester_v2': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'light model, autoencoder, button concentration loss 2.0',
        'train_log': '',
        'ckpt_file_name': './save_models/no_dtime_button_concentration_tester_v2_15_eps_96_steps_1.2784_loss_0.7177_acc.pth',
        'seq_len': 512,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'loss_button_concentration': 2.0,
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        # Training settings
        "save_every": 15, 
        "batch_size": 24,
        "num_workers": 10,
        "num_buttons": 12,
    },
'no_dtime_button_concentration_tester_v3': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'light model, autoencoder, button concentration extremes',
        'train_log': '',
        'ckpt_file_name': './save_models/no_dtime_button_concentration_tester_v3_30_eps_186_steps_1.7027_loss_0.7318_acc.pth',
        'seq_len': 512,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'loss_button_concentration': 1.,
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        # Training settings
        "save_every": 15, 
        "batch_size": 24,
        "num_workers": 10,
        "num_buttons": 12,
    },

'no_dtime_button_concentration_v1': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'full model, autoencoder, button concentration extremes',
        'train_log': '',
        'ckpt_file_name': './save_models/no_dtime_button_concentration_v1_4_eps_27_steps_1.0645_loss_0.6621_acc.pth',
        'seq_len': 1048,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'loss_button_concentration': 1.,
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        # Training settings
        "save_every": 2, 
        "batch_size": 16,
        "num_workers": 8,
        "num_buttons": 12,
    },

'Dec_no_conditioning_v1': {
        'model_type': 'AutoregressiveDecoder_no_conditioning',
        'description': 'decoder only, no conditioning, full dataset',
        'train_log': '',
        'ckpt_file_name': './save_models/Dec_no_conditioning_v1_42_eps_172_steps_1.7524_loss_0.4802_acc.pth',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        # Training settings
        "save_every": 2, 
        "batch_size": 16,
        "num_workers": 8,
    },
'Dec_no_conditioning_tester_v1': {
        'model_type': 'AutoregressiveDecoder_no_conditioning',
        'description': 'light model decoder only, no conditioning, full dataset',
        'train_log': '',
        'ckpt_file_name': '',
        'seq_len': 512,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        # Training settings
        "save_every": 2, 
        "batch_size": 16,
        "num_workers": 8,
    },
'AE_no_dtime_24_buttons': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'full model, autoencoder, light model, 24 buttons',
        'train_log': '',
        'ckpt_file_name': '',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        "dataset_train_path": "./Training-Data/giantmidi_full_accom_only_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_accom_only_test.pickle",
        # Training settings
        "save_every": 15, 
        "batch_size": 16,
        "num_workers": 8,
        "num_buttons": 24,
    },
'AE_no_dtime_4_buttons': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'full model, autoencoder, light model, 24 buttons',
        'train_log': '',
        'ckpt_file_name': '',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        "dataset_train_path": "./Training-Data/giantmidi_full_accom_only_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_accom_only_test.pickle",
        # Training settings
        "save_every": 15, 
        "batch_size": 16,
        "num_workers": 8,
        "num_buttons": 4,
    },
'AE_no_dtime_saturation_v1': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'full model, autoencoder, light model, 12 buttons',
        'train_log': '',
        'ckpt_file_name': './save_models/AE_no_dtime_saturation_v1_240_eps_964_steps_0.7869_loss_0.7435_acc.pth',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.0,
        'loss_deviate': 0.1,
        "loss_saturated_contour": 0.1, #0.1 # Saturated contour loss (allows button saturation at extremes)
        "loss_pitch_extreme_anchoring": 0.0, #0.01 # Anchors extreme pitches to extreme buttons
        "dataset_train_path": "./Training-Data/giantmidi_full_accom_only_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_accom_only_test.pickle",
        # Training settings
        "save_every": 15, 
        "batch_size": 16,
        "num_workers": 8,
        "num_buttons": 12,
    },
    'AE_no_dtime_saturation_tester_v1': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'light model, autoencoder, light model, 12 buttons',
        'train_log': '',
        'ckpt_file_name': '',
        'seq_len': 512,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.0,
        'loss_deviate': 0.1,
        "loss_saturated_contour": 0.1, #0.1 # Saturated contour loss (allows button saturation at extremes)
        "loss_pitch_extreme_anchoring": 0.0, #0.01 # Anchors extreme pitches to extreme buttons
        "dataset_train_path": "./Training-Data/giantmidi_full_accom_only_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_accom_only_test.pickle",
        # Training settings
        "save_every": 15, 
        "batch_size": 16,
        "num_workers": 8,
        "num_buttons": 12,
    },
   'AE_non_linear_compression_v1': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'light model, autoencoder, light model, 12 buttons',
        'train_log': '',
        'ckpt_file_name': './save_models/AE_non_linear_compression_v1_120_eps_484_steps_0.8579_loss_0.7218_acc.pth',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        "loss_nonlinear_compression": 0.1, #0.1 # Non-linear compression: more control in middle, less at extremes
        #"loss_saturated_contour": 0.0, #0.1 # Saturated contour loss (allows button saturation at extremes)
        "loss_pitch_extreme_anchoring": 0.0, #0.01 # Anchors extreme pitches to extreme buttons
        "dataset_train_path": "./Training-Data/giantmidi_full_accom_only_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_accom_only_test.pickle",
        # Training settings
        "save_every": 2, 
        "batch_size": 16,
        "num_workers": 8,
        "num_buttons": 18,
    },
   'AE_non_linear_compression_tester_v1': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'light model, autoencoder, 5-highest-button companded warp, 18 buttons',
        'train_log': '',
        'ckpt_file_name': './save_models/AE_non_linear_compression_tester_v1_20_eps_126_steps_2.6403_loss_0.8075_acc.pth',
        'seq_len': 512,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        "loss_nonlinear_compression": 0.1, #0.1 # Non-linear compression: more control in middle, less at extremes
        #"loss_saturated_contour": 0.0, #0.1 # Saturated contour loss (allows button saturation at extremes)
        "loss_pitch_extreme_anchoring": 0.0, #0.01 # Anchors extreme pitches to extreme buttons
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        # Training settings
        "save_every": 4, 
        "batch_size": 24,
        "num_workers": 10,
        "num_buttons": 18,
    },

   'AE_non_linear_compression_12but_tester_v1': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'light model, autoencoder, 5-highest-button companded warp, 12 buttons',
        'train_log': '',
        'ckpt_file_name': './save_models/AE_non_linear_compression_12but_tester_v1_28_eps_174_steps_1.5786_loss_0.732_acc.pth',
        'seq_len': 512,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        "loss_nonlinear_compression": 0.1, #0.1 # Non-linear compression: more control in middle, less at extremes
        #"loss_saturated_contour": 0.0, #0.1 # Saturated contour loss (allows button saturation at extremes)
        "loss_pitch_extreme_anchoring": 0.0, #0.01 # Anchors extreme pitches to extreme buttons
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        # Training settings
        "save_every": 4, 
        "batch_size": 24,
        "num_workers": 10,
        "num_buttons": 12,
    },
   'AE_non_linear_compression_6but_tester_v1': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'light model, autoencoder, 5-highest-button companded warp, 12 buttons',
        'train_log': '',
        'ckpt_file_name': './save_models/AE_non_linear_compression_12but_tester_v1_28_eps_174_steps_1.5786_loss_0.732_acc.pth',
        'seq_len': 512,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        "loss_nonlinear_compression": 0.1, #0.1 # Non-linear compression: more control in middle, less at extremes
        #"loss_saturated_contour": 0.0, #0.1 # Saturated contour loss (allows button saturation at extremes)
        "loss_pitch_extreme_anchoring": 0.0, #0.01 # Anchors extreme pitches to extreme buttons
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        # Training settings
        "save_every": 4, 
        "batch_size": 24,
        "num_workers": 10,
        "num_buttons": 6,
    },
    'AE_LSTM_behavior_tester_v1': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'light model, autoencoder, light model, 12 buttons',
        'train_log': '',
        'ckpt_file_name': './save_models/AE_LSTM_behavior_tester_v1_375_eps_2632_steps_1.3732_loss_0.6465_acc.pth',
        'seq_len': 512,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        "loss_contour_perc": 1., # 0.4, original genie # encourage button intervals to match piano note intervals (in direction, not magnitude, -1,+1)
        "loss_multi_step_perc": 0., # 0.3, original # considers relationships between the current note and multiple previous notes (in directions, not magnitude, -1,+1)
        'loss_deviate': 0.1,
        "loss_latent_velocity": 0.1, #0.1 # LSTM behavior loss
        "loss_drift": 0.1, #0.1 # LSTM behavior loss
        "dataset_train_path": "./Training-Data/giantmidi_full_accom_only_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_accom_only_test.pickle",
        # Training settings
        "save_every": 15, 
        "batch_size": 16,
        "num_workers": 8,
        "num_buttons": 12,
    },
    'good_ref_127buttons_tester': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'full model, autoencoder 127 butttons',
        'train_log': '',
        'ckpt_file_name': './save_models/good_ref_127buttons_tester_15_eps_96_steps_3.6484_loss_0.9473_acc.pth',
        'seq_len': 512,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'full',
        "num_buttons": 127,
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        "save_every": 15, 
        "batch_size": 24,
        "num_workers": 10,
        },
    'good_ref_5buttons_tester': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'full model, autoencoder 5 butttons',
        'train_log': '',
        'ckpt_file_name': './save_models/good_ref_5buttons_tester_90_eps_546_steps_1.3976_loss_0.6243_acc.pth',
        'seq_len': 512,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'full',
        "num_buttons":5,
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        "save_every": 15, 
        "batch_size": 24,
        "num_workers": 10,
        },
    'good_ref_5buttons_UPF': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'full model, autoencoder 5 butttons',
        'train_log': '',
        'ckpt_file_name': '',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'full',
        "num_buttons":5,
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        # Training settings
        "save_every": 5, 
        "batch_size": 1,
        "num_workers": 1,
        },
    'good_ref_5buttons': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'full model, autoencoder 5 butttons',
        'train_log': '',
        'ckpt_file_name': './save_models/good_ref_5buttons_25_eps_104_steps_1.3811_loss_0.5778_acc.pth',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'full',
        "num_buttons":5,
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        # Training settings
        "save_every": 5, 
        "batch_size": 16,
        "num_workers": 8,
        },
    'anticipation_5buttons': {
        'model_type': 'AE_antic',
        'description': 'Piano Genie with anticipation for user-injected MIDI notes. Same arch as good_ref_5buttons but decoder has antic_pitch_emb + mode_emb.',
        'train_log': '',
        'ckpt_file_name': '',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'num_buttons': 5,
        # Anticipation parameters
        'anticipation_delta': 4,
        'anticipation_rate': 0.15,
        'ar_prob': 0.5,
        'random_prob': 0.25,
        'anticipation_min_span': 5,
        'anticipation_max_span': 20,
        # Loss weights
        'loss_recons': 1.0,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        # Dataset (same as good_ref_5buttons)
        'dataset': 'full',
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        # Training settings
        "save_every": 5,
        "batch_size": 16,
        "num_workers": 8,
        },
    'anticipation_tester_v1': {
        'model_type': 'AE_antic',
        'description': 'Light tester for anticipation model.',
        'train_log': '',
        'ckpt_file_name': './save_models/anticipation_tester_v1_145_eps_2592_steps_0.8195_loss_0.8071_acc.pth',
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'num_buttons': 19,
        # Anticipation parameters
        'anticipation_delta': 4,
        'anticipation_rate': 0.15,
        'ar_prob': 0.5, # probability of using AR mode, no anticipation
        'random_prob': 0.25, # probability of individual tokens being controls
        'anticipation_min_span': 5,
        'anticipation_max_span': 20,
        # Loss weights
        'loss_recons': 1.0,
        'loss_margin': 0.1,
        'loss_contour': 0.3,
        'loss_deviate': 0.2,
        # Dataset
        'dataset': 'full',
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        # Training settings
        "save_every": 5,
        "batch_size": 16,
        "num_workers": 8,
        },
    'anticipation_tester_v2': {
        'model_type': 'AE_antic',
        'description': 'Only span anticipation. delta=12, reinforce contour and deviate loss',
        'train_log': '',
        'ckpt_file_name': './save_models/anticipation_tester_v2_5_eps_1792_steps_2.6901_loss_0.8691_acc.pth',
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'num_buttons': 19,
        # Anticipation parameters
        'anticipation_delta': 12,
        'anticipation_rate': 0.15,
        'ar_prob': 0.5,
        'random_prob': 0.,
        'anticipation_min_span': 5,
        'anticipation_max_span': 50,
        # Loss weights
        'loss_recons': 1.0,
        'loss_margin': 0.1,
        'loss_contour': 0.3,
        'loss_deviate': 0.2,
        # Dataset
        'dataset': 'full',
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        # Training settings
        "save_every": 5,
        "batch_size": 16,
        "num_workers": 8,
        },
    'anticipation_v1': {
        'model_type': 'AE_antic',
        'description': 'Light tester for anticipation model.',
        'train_log': '',
        'ckpt_file_name': './save_models/anticipation_v1_14_eps_496_steps_0.7573_loss_0.8945_acc.pth',
        'seq_len': 512,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'num_buttons': 19,
        # Anticipation parameters
        'anticipation_delta': 4,
        'anticipation_rate': 0.15,
        'ar_prob': 0.5,
        'random_prob': 0.25,
        'anticipation_min_span': 5,
        'anticipation_max_span': 20,
        # Loss weights
        'loss_recons': 1.0,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        # Dataset
        'dataset': 'full',
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        # Training settings
        "save_every": 1,
        "batch_size": 8,
        "num_workers": 8,
        },
    'good_ref_88buttons': {
        'model_type': 'autoencoder_no_dtime',
        'description': 'full model, autoencoder 127 butttons',
        'train_log': '',
        'ckpt_file_name': './save_models/good_ref_88buttons_120_eps_484_steps_0.7065_loss_0.9663_acc.pth',
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.1,
        'loss_contour': 0.1,
        'loss_deviate': 0.1,
        'dataset': 'full',
        "num_buttons": 88,
        "dataset_train_path": "./Training-Data/giantmidi_full_train.pickle",
        "dataset_val_path": "./Training-Data/giantmidi_full_test.pickle",
        # Training settings
        "save_every": 5, 
        "batch_size": 16,
        "num_workers": 8,
        },
# Dual-conditioned autoencoder: melodic shape buttons + harmony movement with exponential decay
'AE_dual_v1': {
    'model_type': 'autoencoder_no_dtime_dual',
    'description': 'Dual conditioning: melodic shape buttons + harmony movement events with exponential decay. Full model.',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,
    'heads': 32,
    'num_buttons': 12,
    'num_harmony_movements': 8,  # 0=STAY, 1=TENSION, 2=STABILIZE, 3=COLORIZE, ...
    'harmony_span_dropout': 0.6, # Drop 60% of harmony spans during training
    'harmony_decay_residual': 0.05, # Target residual at span end (5%)
    # Loss weights
    #'loss_recons': 1.0,
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths (harmony-movement-augmented pickles)
    "dataset_train_path": "./Training-Data/giantmidi_full_harmony_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_harmony_test",
    # Training settings
    "save_every": 5,
    "batch_size": 16,
    "num_workers": 8,
    },
'AE_dual_tester_v4': {
    'model_type': 'autoencoder_no_dtime_dual',
    'description': '0.0 dropout, concatenation instead of addition, variable decay rate' ,
    'train_log': '',
    'ckpt_file_name': './save_models/AE_dual_tester_v4_75_eps_380_steps_1.6323_loss_0.7606_acc.pth',
    'seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 512,
    'num_layers': 4,
    'heads': 32,
    'num_buttons': 12,
    'num_harmony_movements': 8,
    'harmony_span_dropout': 0.0,
    'harmony_decay_residual': 0.05,
    # Loss weights
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_harmony_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_harmony_test",
    # Training settings
    "save_every": 15,
    "batch_size": 24,
    "num_workers": 10,
    },
# Style-conditioned autoencoder: buttons + cross-attention to style reference
'AE_style_tester': {
    'model_type': 'AE_style',
    'description': 'Style-conditioned: buttons + cross-attention to a style reference MIDI sequence',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_style_tester_30_eps_434_steps_0.5793_loss_0.8586_acc.pth',
    'seq_len': 512,
    'style_seq_len': 256,
    'pad_idx': 128,
    'emb_dim': 512,
    'num_layers': 4,
    'style_encoder_depth': 2,
    'heads': 32,
    'num_buttons': 19,
    # Loss weights
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_test",
    # Training settings
    "save_every": 5,
    "batch_size": 16,
    "num_workers": 0,
    },
'AE_style_tester_v2': {
    'model_type': 'AE_style',
    'description': 'more style context window',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_style_tester_v2_5_eps_294_steps_0.6602_loss_0.7962_acc.pth',
    'seq_len': 512,
    'style_seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 512,
    'num_layers': 4,
    'style_encoder_depth': 2,
    'heads': 32,
    'num_buttons': 19,
    # Loss weights
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_test",
    # Training settings
    "save_every": 1,
    "batch_size": 16,
    "num_workers": 0,
    },

# Style-conditioned autoencoder: buttons + cross-attention to style reference
'AE_style_v1': {
    'model_type': 'AE_style_joker',
    'description': 'Style-conditioned: 19 buttons + joker button + cross-attention',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_style_v1_0_eps_1092_steps_0.773_loss_0.7658_acc.pth',
    'seq_len': 512,
    'style_seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,
    'style_encoder_depth': 4,
    'heads': 32,
    'num_buttons': 19,
    # Loss weights
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_test",
    # Training settings
    "save_every": 5,
    "batch_size": 8,
    "num_workers": 2,
    },
'AE_style_v2': {
    'model_type': 'AE_style',
    'description': 'Style-conditioned: no joker, less seq_len, more style_seq_len, more deviate',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_style_v2_20_eps_567_steps_0.6873_loss_0.7982_acc.pth',
    'seq_len': 256,
    'style_seq_len': 512, 
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,
    'style_encoder_depth': 4,
    'heads': 32,
    'num_buttons': 19,
    # Loss weights
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.2,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_test",
    # Training settings
    "save_every": 5,
    "batch_size": 8,
    "num_workers": 0,
    },
'AE_style_harm_v1': {
    'model_type': 'AE_style_harm',
    'description': 'Style + button + harmony movement (AdaLN-Zero FiLM, hidden chord planner). Resumable from AE_style_v2.',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_style_harm_v1_10_eps_780_steps_1.392_loss_0.9414_acc.pth',
    # Optional warm-start: path to an AE_style_v1 checkpoint. Loaded strict=False
    # so the zero-initialised harmony/planner/aux params start as identity and the
    # model reproduces the base style model until harmony training kicks in.
    'init_from_ckpt': './save_models/AE_style_v1_0_eps_1092_steps_0.773_loss_0.7658_acc.pth',
    'seq_len': 512,
    'style_seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,
    'style_encoder_depth': 4,
    'heads': 32,
    'num_buttons': 19,
    # Harmony conditioning (stage 2)
    'harm_cond_dim': 256,
    'harm_film_start_frac': 0.5,   # modulate the upper half of decoder blocks
    # FiLM stabilisation (fix for the non-neutral-chord modulation blow-up):
    'harm_film_scale_limit': 0.5,  # Rec 1: bound |scale| (tanh*limit) per element
    'harm_film_shift_limit': 0.5,  # Rec 1: bound |shift| (tanh*limit) per element
    'loss_film_reg': 0.01,         # Rec 2: penalty on mean squared FiLM modulation
    'planner_dim': 256,
    'planner_depth': 2,
    'planner_heads': 8,
    'harmony_drop_prob': 0.3,      # train the unconditional branch (CFG / release)
    'joker_prob': 0.1,             # "move now, model decides which"
    'loss_chord_plan': 0.5,        # hidden chord planner supervision
    'loss_aux_chord': 0.5,         # decoder-hidden chord-factor supervision
    # Stage 3: movement-constrained planner sampling + soft PC-bias
    'movement_clf_dim': 128,
    'loss_move_recover': 0.5,      # learn chord-transition -> movement compatibility
    'pc_bias_weight': 0.0,         # inference soft pitch-class logit bias (0 = off)
    # Loss weights (style/button, inherited from AE_style)
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths (regenerate with midis2pickles.py harmony output)
    "dataset_train_path": "./Training-Data/giantmidi_full_harmony_labels_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_harmony_labels_test",
    # Training settings
    "save_every": 1,
    "batch_size": 4,
    "num_workers": 0,
    },
'AE_style_harm_tester_v1': {
    'model_type': 'AE_style_harm',
    'description': 'Style + button + harmony movement (AdaLN-Zero FiLM, hidden chord planner). Resumable from AE_style_v2.',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_style_harm_tester_v1_17_eps_144_steps_2.0324_loss_1.7237_val_loss_0.7767_acc.pth',
    # Optional warm-start: path to an AE_style_v2 checkpoint. Loaded strict=False
    # so the zero-initialised harmony/planner/aux params start as identity and the
    # model reproduces the base style model until harmony training kicks in.
    'init_from_ckpt': './save_models/AE_style_tester_v2_5_eps_294_steps_0.6602_loss_0.7962_acc.pth',
    'seq_len': 512,
    'style_seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 512,
    'num_layers': 4,
    'style_encoder_depth': 2,
    'heads': 32,
    'num_buttons': 19,
    # Harmony conditioning (stage 2)
    'harm_cond_dim': 256,
    'harm_film_start_frac': 0.5,   # modulate the upper half of decoder blocks
    # FiLM stabilisation (fix for the non-neutral-chord modulation blow-up):
    'harm_film_scale_limit': 0.5,  # Rec 1: bound |scale| (tanh*limit) per element
    'harm_film_shift_limit': 0.5,  # Rec 1: bound |shift| (tanh*limit) per element
    'loss_film_reg': 0.01,         # Rec 2: penalty on mean squared FiLM modulation
    'planner_dim': 256,
    'planner_depth': 2,
    'planner_heads': 8,
    'harmony_drop_prob': 0.3,      # train the unconditional branch (CFG / release)
    'joker_prob': 0.1,             # "move now, model decides which"
    'loss_chord_plan': 0.5,        # hidden chord planner supervision
    'loss_aux_chord': 0.5,         # decoder-hidden chord-factor supervision
    # Stage 3: movement-constrained planner sampling + soft PC-bias
    'movement_clf_dim': 128,
    'loss_move_recover': 0.5,      # learn chord-transition -> movement compatibility
    'pc_bias_weight': 0.0,         # inference soft pitch-class logit bias (0 = off)
    # Loss weights (style/button, inherited from AE_style)
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths (regenerate with midis2pickles.py harmony output)
    "dataset_train_path": "./Training-Data/giantmidi_full_harmony_labels_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_harmony_labels_test",
    # Training settings
    "save_every": 1,
    "batch_size": 16,
    "num_workers": 0,
    },
'AE_style_tensions_v1': {
    'model_type': 'AE_style_tensions',
    'description': 'Style + button + continuous tonal-tension (TIV/TIS) AdaLN-Zero FiLM. Self-supervised tension from pitch; resumable from AE_style_v1.',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_style_tensions_v1_0_eps_0_steps_0.0_loss_0.0_acc.pth',
    # Optional warm-start from a base AE_style checkpoint (strict=False): the
    # zero-initialised tension FiLM / aux head start as identity, so the
    # warm-started model reproduces the base style+button model exactly.
    'init_from_ckpt': './save_models/AE_style_v1_0_eps_1092_steps_0.773_loss_0.7658_acc.pth',
    'seq_len': 512,
    'style_seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,
    'style_encoder_depth': 4,
    'heads': 32,
    'num_buttons': 19,
    # Tension conditioning
    'harm_cond_dim': 256,
    'harm_film_start_frac': 0.5,   # modulate the upper half of decoder blocks
    'harm_film_scale_limit': 0.5,  # bound |scale| (tanh*limit) per element
    'harm_film_shift_limit': 0.5,  # bound |shift| (tanh*limit) per element
    'loss_film_reg': 0.01,         # penalty on mean squared FiLM modulation
    'tension_drop_prob': 0.0,      # OFF: the dataset now samples per-position intensity
    'tension_cond_horizon': 12,    # future-aggregate conditioning window (notes, leak-free)
    'loss_aux_tension': 0.2,       # future-tension planning auxiliary (masked tail, block-weighted)
    'pc_bias_weight': 0.0,         # inference soft pitch-class logit bias (0 = off)
    # Tonal-tension feature extraction (tension_extractor.py)
    'tension_short_window': 8,     # local-sonority TIV window (notes)
    'tension_key_alpha': 2.0,      # gain of the per-note key-correlation likelihood
    'tension_trans_alpha': 100.0,  # circle-of-fifths ring transition ratio (HMM, per-note)
    'tension_key_theta': 3.0,      # hysteretic key-decision margin (inference display)
    'tension_tau_low': 8.0,        # leaky-integrator decay [s] at the bass end
    'tension_tau_high': 3.0,       # leaky-integrator decay [s] at the treble end
    'tension_profiles': 'genie',   # tuned Temperley/Sapp variant (see tension_extractor)
    'tension_bass_weight': 2.0,    # bass-register weight in the key evidence
    'tension_cadence_gap': 1.0,    # inter-onset gap [s] marking a phrase-final note
    'tension_cadence_boost': 2.0,  # evidence boost for phrase-final (cadence) notes
    # Loss weights (style/button, inherited from AE_style)
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths (plain pitch pickles; tension is derived self-supervised)
    "dataset_train_path": "./Training-Data/giantmidi_full_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_test",
    # Training settings
    "save_every": 1,
    "batch_size": 4,
    "num_workers": 0,
    },
'AE_style_tensions_tester_v1': {
    'model_type': 'AE_style_tensions',
    'description': 'Small tester for the tonal-tension FiLM model. Resumable from AE_style_tester_v2.',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_style_tensions_tester_v1_32_eps_231_steps_0.2656_loss_0.7377_val_loss_0.9564_acc.pth',
    'init_from_ckpt': './save_models/AE_style_tester_v2_5_eps_294_steps_0.6602_loss_0.7962_acc.pth',
    'seq_len': 512,
    'style_seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 512,
    'num_layers': 4,
    'style_encoder_depth': 2,
    'heads': 32,
    'num_buttons': 19,
    # Tension conditioning
    'harm_cond_dim': 256,
    'harm_film_start_frac': 0.5,
    'harm_film_scale_limit': 0.5,
    'harm_film_shift_limit': 0.5,
    'loss_film_reg': 0.01,
    'tension_drop_prob': 0.0,      # OFF: dataset samples per-position intensity
    'tension_cond_horizon': 12,    # future-aggregate conditioning window (notes)
    'loss_aux_tension': 0.2,       # future-tension planning auxiliary (masked tail, block-weighted)
    'pc_bias_weight': 0.0,
    # Tonal-tension feature extraction (tension_extractor.py)
    'tension_short_window': 8,     # local-sonority TIV window (notes)
    'tension_key_alpha': 2.0,      # gain of the per-note key-correlation likelihood
    'tension_trans_alpha': 100.0,  # circle-of-fifths ring transition ratio (HMM, per-note)
    'tension_key_theta': 3.0,      # hysteretic key-decision margin (inference display)
    'tension_tau_low': 8.0,        # leaky-integrator decay [s] at the bass end
    'tension_tau_high': 3.0,       # leaky-integrator decay [s] at the treble end
    'tension_profiles': 'genie',   # tuned Temperley/Sapp variant (see tension_extractor)
    'tension_bass_weight': 2.0,    # bass-register weight in the key evidence
    'tension_cadence_gap': 1.0,    # inter-onset gap [s] marking a phrase-final note
    'tension_cadence_boost': 2.0,  # evidence boost for phrase-final (cadence) notes
    # Loss weights (style/button)
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_test",
    # Training settings
    "save_every": 1,
    "batch_size": 16,
    "num_workers": 0,
    },
'AE_style_move_v1': {
    'model_type': 'AE_style_move',
    'description': 'Style + button + BINARY move-flag AdaLN-Zero FiLM ("move now, model decides which"). Boundary positions only — no chord factors, no planner. Resumable from AE_style_v2.',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_style_move_v1_0_eps_0_steps_0.0_loss_0.0_acc.pth',
    # Warm-start from the base AE_style_v2 checkpoint (strict=False): the
    # zero-initialised move FiLM starts as identity, so the warm-started model
    # reproduces the base style+button model exactly.
    'init_from_ckpt': './save_models/AE_style_tester_v2_5_eps_294_steps_0.6602_loss_0.7962_acc.pth',
    'seq_len': 256,
    'style_seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,
    'style_encoder_depth': 4,
    'heads': 32,
    'num_buttons': 19,
    # Move-flag conditioning
    'harm_cond_dim': 256,
    'harm_film_start_frac': 0.5,   # modulate the upper half of decoder blocks
    'harm_film_scale_limit': 0.5,  # bound |scale| (tanh*limit) per element
    'harm_film_shift_limit': 0.5,  # bound |shift| (tanh*limit) per element
    'loss_film_reg': 0.01,         # penalty on mean squared FiLM modulation
    'harmony_drop_prob': 0.3,      # train the unconditional branch (CFG / release)
    'move_binary': True,           # dataset: binarize ch-3 boundaries into move_flag
    'move_flag_span': 8,           # notes of "transition happening" after a boundary
    'pc_bias_weight': 0.0,         # inference soft pitch-class logit bias (0 = off)
    # Loss weights (style/button, inherited from AE_style)
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths (harmony-labelled pickles: only the ch-3 boundary POSITIONS are used)
    "dataset_train_path": "./Training-Data/giantmidi_full_harmony_labels_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_harmony_labels_test",
    # Training settings
    "save_every": 1,
    "batch_size": 8,
    "num_workers": 0,
    },
'AE_style_move_tester_v1': {
    'model_type': 'AE_style_move',
    'description': 'Small tester for the binary move-flag FiLM model. Resumable from AE_style_tester_v2.',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_style_move_tester_v1_1_eps_16_steps_0.7978_loss_0.6807_val_loss_0.765_acc.pth',
    'init_from_ckpt': './save_models/AE_style_tester_v2_5_eps_294_steps_0.6602_loss_0.7962_acc.pth',
    'seq_len': 512,
    'style_seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 512,
    'num_layers': 4,
    'style_encoder_depth': 2,
    'heads': 32,
    'num_buttons': 19,
    # Move-flag conditioning
    'harm_cond_dim': 256,
    'harm_film_start_frac': 0.5,
    'harm_film_scale_limit': 0.5,
    'harm_film_shift_limit': 0.5,
    'loss_film_reg': 0.01,
    'harmony_drop_prob': 0.3,
    'move_binary': True,
    'move_flag_span': 8,
    'pc_bias_weight': 0.0,
    # Loss weights (style/button)
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths (harmony-labelled pickles)
    "dataset_train_path": "./Training-Data/giantmidi_full_harmony_labels_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_harmony_labels_test",
    # Training settings
    "save_every": 1,
    "batch_size": 16,
    "num_workers": 0,
    },
'AE_style_chords_v1': {
    'model_type': 'AE_style_chords',
    'description': 'Style + button + SOUNDING-CHORD AdaLN-Zero FiLM (channel-4 chroma + bass only). A simpler AE_style_harm: no chord factors, no planner, no aux heads. Anticipation-jitter augmented. Resumable from AE_style_v2.',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_style_chords_v1_0_eps_0_steps_0.0_loss_0.0_acc.pth',
    # Warm-start from the base AE_style_v2 checkpoint (strict=False): the
    # zero-initialised chord FiLM starts as identity, so the warm-started model
    # reproduces the base style+button model exactly.
    'init_from_ckpt': './save_models/AE_style_tester_v2_5_eps_294_steps_0.6602_loss_0.7962_acc.pth',
    'seq_len': 256,
    'style_seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,
    'style_encoder_depth': 4,
    'heads': 32,
    'num_buttons': 19,
    # Chord conditioning
    'harm_cond_dim': 256,
    'harm_film_start_frac': 0.5,   # modulate the upper half of decoder blocks
    'harm_film_scale_limit': 0.5,  # bound |scale| (tanh*limit) per element
    'harm_film_shift_limit': 0.5,  # bound |shift| (tanh*limit) per element
    'loss_film_reg': 0.01,         # penalty on mean squared FiLM modulation
    'harmony_drop_prob': 0.3,      # train the unconditional branch (CFG / release)
    'pc_bias_weight': 0.0,         # inference soft pitch-class logit bias (0 = off)
    # Anticipation jitter (dataset augmentation): shift each chord-span onset
    # earlier so the chord "arrives" before the notes it labels (models the user
    # pressing before wanting to hear the effect; also prevents overfitting to
    # exact boundaries). Training-only, re-randomised per draw.
    'chords_jitter': True,
    'chords_jitter_mode': 'ms',    # 'ms' | 'fraction'
    'chords_jitter_ms_min': 100.0,
    'chords_jitter_ms_max': 500.0,
    'chords_jitter_frac_min': 0.10,
    'chords_jitter_frac_max': 0.25,
    # Loss weights (style/button, inherited from AE_style)
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths (harmony-labelled pickles: only the channel-4 chord tones are used)
    "dataset_train_path": "./Training-Data/giantmidi_full_harmony_labels_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_harmony_labels_test",
    # Training settings
    "save_every": 1,
    "batch_size": 8,
    "num_workers": 0,
    },
'AE_style_chords_tester_v1': {
    'model_type': 'AE_style_chords',
    'description': 'Small tester for the sounding-chord FiLM model (channel-4 chroma + bass). Resumable from AE_style_tester_v2.',
    'train_log': '',
    'ckpt_file_name': '',
    'init_from_ckpt': './save_models/AE_style_tester_v2_5_eps_294_steps_0.6602_loss_0.7962_acc.pth',
    'seq_len': 512,
    'style_seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 512,
    'num_layers': 4,
    'style_encoder_depth': 2,
    'heads': 32,
    'num_buttons': 19,
    # Chord conditioning
    'harm_cond_dim': 256,
    'harm_film_start_frac': 0.5,
    'harm_film_scale_limit': 0.5,
    'harm_film_shift_limit': 0.5,
    'loss_film_reg': 0.01,
    'harmony_drop_prob': 0.3,
    'pc_bias_weight': 0.0,
    # Anticipation jitter (dataset augmentation)
    'chords_jitter': True,
    'chords_jitter_mode': 'ms',
    'chords_jitter_ms_min': 100.0,
    'chords_jitter_ms_max': 500.0,
    'chords_jitter_frac_min': 0.10,
    'chords_jitter_frac_max': 0.25,
    # Loss weights (style/button)
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths (harmony-labelled pickles)
    "dataset_train_path": "./Training-Data/giantmidi_full_roman_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_roman_test",
    # Training settings
    "save_every": 1,
    "batch_size": 16,
    "num_workers": 0,
    },
'AE_style_roman_v1': {
    'model_type': 'AE_style_roman',
    'description': 'Style + button + ROMAN-MOVEMENT AdaLN-Zero FiLM: 5 harmony movements (STABLE/TENSION/RESOLVE/MODULATE/COLOR) + NULL, reduced from roman-numeral labels per chord transition. A variation of AE_style_chords (discrete movement embedding instead of chroma+bass). Minimal cut: no aux heads / NULL-span sampling yet. Resumable from AE_style_v2.',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_style_roman_v1_0_eps_0_steps_0.0_loss_0.0_acc.pth',
    # Warm-start from the base AE_style_v2 checkpoint (strict=False): the
    # zero-initialised roman-movement FiLM starts as identity, so the warm-started
    # model reproduces the base style+button model exactly.
    'init_from_ckpt': './save_models/AE_style_tester_v2_5_eps_294_steps_0.6602_loss_0.7962_acc.pth',
    'seq_len': 256,
    'style_seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,
    'style_encoder_depth': 4,
    'heads': 32,
    'num_buttons': 19,
    # Roman-movement conditioning
    'harm_cond_dim': 256,
    'harm_film_start_frac': 0.5,   # modulate the upper half of decoder blocks
    'harm_film_scale_limit': 0.5,  # bound |scale| (tanh*limit) per element
    'harm_film_shift_limit': 0.5,  # bound |shift| (tanh*limit) per element
    'loss_film_reg': 0.01,         # penalty on mean squared FiLM modulation
    'harmony_drop_prob': 0.3,      # train the unconditional branch (CFG / release)
    'pc_bias_weight': 0.0,         # inference soft pitch-class logit bias (0 = off)
    # Anticipation jitter operates on chroma/bass (unused by the movement
    # conditioner); roman-onset jitter is a follow-on, so keep it off here.
    'chords_jitter': False,
    # Loss weights (style/button, inherited from AE_style)
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths (roman-movement pickles: channel-122 movement + channel-4 tones)
    "dataset_train_path": "./Training-Data/giantmidi_roman_move_train",
    "dataset_val_path": "./Training-Data/giantmidi_roman_move_test",
    # Training settings
    "save_every": 1,
    "batch_size": 8,
    "num_workers": 0,
    },
'AE_style_roman_tester_v1': {
    'model_type': 'AE_style_roman',
    'description': 'Small tester for the roman-movement FiLM model (5 movements + NULL). Resumable from AE_style_tester_v2.',
    'train_log': '',
    'ckpt_file_name': '',
    'init_from_ckpt': './save_models/AE_style_tester_v2_5_eps_294_steps_0.6602_loss_0.7962_acc.pth',
    'seq_len': 512,
    'style_seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 512,
    'num_layers': 4,
    'style_encoder_depth': 2,
    'heads': 32,
    'num_buttons': 19,
    # Roman-movement conditioning
    'harm_cond_dim': 256,
    'harm_film_start_frac': 0.5,
    'harm_film_scale_limit': 0.5,
    'harm_film_shift_limit': 0.5,
    'loss_film_reg': 0.01,
    'harmony_drop_prob': 0.3,
    'pc_bias_weight': 0.0,
    'chords_jitter': False,
    # Loss weights (style/button)
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths (roman-movement pickles)
    "dataset_train_path": "./Training-Data/giantmidi_roman_move_train",
    "dataset_val_path": "./Training-Data/giantmidi_roman_move_test",
    # Training settings
    "save_every": 1,
    "batch_size": 16,
    "num_workers": 0,
    },
'AE_antic_style_v1': {
    'model_type': 'AE_antic_style',
    'description': 'Style cross-attention + anticipation for user-injected notes (AE_style_v2 base).',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_antic_style_v1_1_eps_2322_steps_0.8459_loss_0.7852_acc.pth',
    'seq_len': 256,
    'style_seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,
    'style_encoder_depth': 4,
    'heads': 32,
    'num_buttons': 19,
    # Anticipation parameters
    'anticipation_delta': 4,
    'anticipation_rate': 0.15,
    'ar_prob': 0.5,
    'random_prob': 0.25,
    'anticipation_min_span': 5,
    'anticipation_max_span': 20,
    # Loss weights
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.2,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_test",
    # Training settings
    "save_every": 1,
    "batch_size": 8,
    "num_workers": 0,
    },
'AE_antic_style_tester_v1': {
    'model_type': 'AE_antic_style',
    'description': 'Style cross-attention + anticipation for user-injected notes (AE_style_v2 base).',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_antic_style_tester_v1_35_eps_252_steps_1.6014_loss_1.1014_val_loss_0.8672_acc.pth',
    'seq_len': 512,
    'style_seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 512,
    'num_layers': 4,
    'style_encoder_depth': 4,
    'heads': 32,
    'num_buttons': 19,
    # Anticipation parameters
    'anticipation_delta': 8,
    'anticipation_rate': 0.15, # per-position control probability for random mode 
    'ar_prob': 0.5,
    'random_prob': 0., # probability of random-control strategy  (default 0.25) 
    # span_prob = 0.5 = 1 - ar_prob - random_prob
    'anticipation_min_span': 8,
    'anticipation_max_span': 50,
    # Loss weights
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.15,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_test",
    # Training settings
    "save_every": 1,
    "batch_size": 16,
    "num_workers": 10,
    },
'AE_antic_style_joker_tester_v1': {
    'model_type': 'AE_antic_style_joker',
    'description': 'Style cross-attention + anticipation for user-injected notes (AE_style_v2 base).',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_antic_style_joker_tester_v1_76_eps_658_steps_0.8148_loss_0.7022_val_loss_0.8047_acc.pth',
    'seq_len': 512,
    'style_seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 512,
    'num_layers': 4,
    'style_encoder_depth': 4,
    'heads': 32,
    'num_buttons': 19,
    # Anticipation parameters
    'anticipation_delta': 8,
    'anticipation_rate': 0.15, # per-position control probability for random mode 
    'ar_prob': 0.5,
    'random_prob': 0., # probability of random-control strategy  (default 0.25) 
    # span_prob = 0.5 = 1 - ar_prob - random_prob
    'anticipation_min_span': 8,
    'anticipation_max_span': 50,
    'joker_prob': 0.15,
    # Loss weights
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.15,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_test",
    # Training settings
    "save_every": 2,
    "batch_size": 16,
    "num_workers": 0,
    },
'AE_antic_style_joker_v1': {
    'model_type': 'AE_antic_style_joker',
    'description': 'Style cross-attention + anticipation + joker button full model.',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 512,
    'style_seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,
    'style_encoder_depth': 4,
    'heads': 32,
    'num_buttons': 19,
    # Anticipation parameters
    'anticipation_delta': 8,
    'anticipation_rate': 0.15, # per-position control probability for random mode 
    'ar_prob': 0.5,
    'random_prob': 0., # probability of random-control strategy  (default 0.25) 
    # span_prob = 0.5 = 1 - ar_prob - random_prob
    'anticipation_min_span': 8,
    'anticipation_max_span': 50,
    'joker_prob': 0.15,
    # Loss weights
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.15,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_test",
    # Training settings
    "save_every": 1,
    "batch_size": 8,
    "num_workers": 0,
    },
'AE_antic_style_v2': {
    'model_type': 'AE_antic_style',
    'description': 'Style cross-attention + anticipation for user-injected notes (AE_style_v2 base).',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 512,
    'style_seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,
    'style_encoder_depth': 4,
    'heads': 32,
    'num_buttons': 19,
    # Anticipation parameters
    'anticipation_delta': 4,
    'anticipation_rate': 0.15,
    'ar_prob': 0.5,
    'random_prob': 0.25,
    'anticipation_min_span': 5,
    'anticipation_max_span': 20,
    # Loss weights
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.2,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_test",
    # Training settings
    "save_every": 5,
    "batch_size": 8,
    "num_workers": 0,
    },
'no_dtime_joker_tester_v1': {
    'model_type': 'autoencoder_no_dtime_joker',
    'description': 'full model, autoencoder, no encoder, no button loss, full dataset',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 512,
    'num_layers': 4,
    'heads': 32,
    'num_buttons': 8,
    # Loss weights
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_test",
    # Training settings
    "save_every": 15,
    "batch_size": 16,
    "num_workers": 10,
    },
'no_dtime_joker_v1': {
    'model_type': 'autoencoder_no_dtime_joker',
    'description': 'full model',
    'train_log': '',
    'ckpt_file_name': './save_models/no_dtime_joker_v1_75_eps_1176_steps_1.5117_loss_0.5527_acc.pth',
    'seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,
    'heads': 32,
    'num_buttons': 8,
    # Loss weights
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_test",
    # Training settings
    "save_every": 5,
    "batch_size": 16,
    "num_workers": 8,
    },

'no_dtime_joker_18buttons_v1': {
    'model_type': 'autoencoder_no_dtime_joker',
    'description': 'full model',
    'train_log': '',
    'ckpt_file_name': './save_models/no_dtime_joker_18buttons_v1_6_eps_280_steps_0.9987_loss_0.7017_acc.pth',
    'seq_len': 1024,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,
    'heads': 32,
    'num_buttons': 18,
    # Loss weights
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_chann_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_chann_test",
    # Training settings
    "save_every": 2,
    "batch_size": 16,
    "num_workers": 6,
    },

'no_dtime_21buttons_v1': {
    'model_type': 'autoencoder_no_dtime',
    'description': 'full model, 21 buttons',
    'train_log': '',
    'ckpt_file_name': '',
    'seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,
    'heads': 32,
    'num_buttons': 21,
    # Loss weights
    'loss_margin': 0.1,
    'loss_contour': 0.1,
    'loss_deviate': 0.1,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_test",
    # Training settings
    "save_every": 5,
    "batch_size": 16,
    "num_workers": 8,
    },
# Residual harmony steering: frozen base + GRU logit correction
# Residual chord adapter: frozen base + MLP hidden-state correction
'AE_residual_v1': {
    'model_type': 'ae_buttons_p_residual',
    'description': 'Chord-conditioned MLP residual adapter on frozen base autoencoder',
    'train_log': '',
    'ckpt_file_name': './save_models/AE_residual_v1_105_eps_954_steps_1.397_loss_0.5571_acc.pth',
    'base_model_name': 'no_dtime_joker_v1',
    'seq_len': 512,
    'pad_idx': 128,
    'emb_dim': 2048,
    'num_layers': 4,
    'heads': 32,
    'num_buttons': 8,
    # Chord adapter params
    'chord_emb_dim': 256,
    'use_bass': True,
    'adapter_hidden': 1024,
    'adapter_lambda': 5.0,
    'logit_bias_scale': 1.0,
    'chord_dropout': 0.,
    # Loss weights (only recons for residual training)
    'loss_margin': 0,
    'loss_contour': 0,
    'loss_deviate': 0,
    # Dataset paths
    "dataset_train_path": "./Training-Data/giantmidi_full_w_chords_train",
    "dataset_val_path": "./Training-Data/giantmidi_full_w_chords_test",
    # Training settings
    "save_every": 15,
    "batch_size": 16,
    "num_workers": 10,
    },
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

def get_model_hparams(model_name, base_hparams=None):
    """
    Get hyperparameters for a specific model by merging model-specific parameters
    with default hyperparameters.
    
    Parameters:
    -----------
    model_name : str
        The name of the model (must exist in MODELS_PARAMETERS)
    base_hparams : dict, optional
        Base hyperparameters dictionary to merge with. If None, will import from params.py
        
    Returns:
    --------
    dict
        Merged hyperparameters dictionary with model-specific values overriding defaults
    """
    if base_hparams is None:
        from params import DEFAULT_HPARAMS
        base_hparams = DEFAULT_HPARAMS
    
    # Start with a copy of base hyperparameters
    hparams = base_hparams.copy()
    
    # Check if model exists
    if model_name not in MODELS_PARAMETERS:
        print(f"Warning: Model '{model_name}' not found in MODELS_PARAMETERS")
        print("Available models:", list(MODELS_PARAMETERS.keys()))
        return hparams
    
    # Merge model-specific parameters
    model_params = MODELS_PARAMETERS[model_name]
    hparams.update(model_params)
    hparams['model_name'] = model_name

    return hparams

#===================================================================================================
# This is the end of models Python module
#===================================================================================================