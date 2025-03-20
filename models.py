#===================================================================================================
# Monster Piano Transformer models Python module
#===================================================================================================
# Project Los Angeles
# Tegridy Code 2025
#===================================================================================================
# License: Apache 2.0
#===================================================================================================

MODELS_HF_REPO_LINK = 'asigalov61/Monster-Piano-Transformer'
MODELS_HF_REPO_URL = 'https://huggingface.co/asigalov61/Monster-Piano-Transformer'

#===================================================================================================

MODELS_INFO = {'tester': 'tester. light model not trained',
               'light': ' light test version (without velocity) trained for 40 epochs on full Monster Piano dataset.',
               'full': 'the full model, trained for 40 epochs on full Monster Piano dataset.',
               'encoder light': 'encoder only, for testing melodic contour loss., light dataset',
               'encoder full': 'encoder only, for testing melodic contour loss., full dataset',
                'encoder original': 'encoder only, original genie implemetation, full dataset',
                'encoder new loss': 'encoder only, new loss functionalities, full dataset',
                'full w new loss': 'full model, new loss functionalities, full dataset'
              }     

#===================================================================================================

MODELS_FILE_NAMES = {'tester': './save_models/tester.pth',
                     'light': './save_models/model_checkpoint_39_eps_3292342_steps_0.6787_loss_0.783_acc.pth',
                     'full': './save_models/model_checkpoint_39_eps_3292342_steps_0.6787_loss_0.783_acc.pth',
                     'encoder light': './save_models/model_checkpoint_39_eps_82603_steps_0.0038_loss_0.0_acc.pth',
                     'encoder full': './save_models/model_checkpoint_9_eps_759772_steps_0.0039_loss_0.0_acc.pth',
                     'encoder original': './save_models/model_checkpoint_5_eps_422096_steps_0.0021_loss_0.0_acc.pth',
                     'encoder new loss': './save_models/model_checkpoint_39_eps_82603_steps_0.0114_loss_0.0_acc.pth',
                     'full w new loss': './save_models/model_checkpoint_2_eps_42335_steps_0.0129_loss_0.0_acc.pth'
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
    'encoder light': {
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
    'encoder original': {
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },
    'encoder new loss': {
        'seq_len': 256,
        'pad_idx': 128,
        'emb_dim': 512,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        },
    'full w new loss': {
        'seq_len': 1024,
        'pad_idx': 128,
        'emb_dim': 2048,
        'num_layers': 4,
        'heads': 32,
        'loss_margin': 0.01,
        'loss_contour': 0.1,
        'loss_deviate': 0.01
        }
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