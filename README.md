# Transformer-based Piano Genie Implementation

A PyTorch implementation of Piano Genie using transformer architecture with teacher forcing mode, supporting both training and real-time inference for musical interaction.

## Requirements

Python 3.9.19

### Dependencies

```bash
torch
tqdm 
datasets
tensorboard
einops
wandb
matplotlib
python-rtmidi
pyfluidsynth
keyboard
pythonosc
```

Install with:
```bash
pip install torch tqdm datasets tensorboard einops wandb matplotlib python-rtmidi pyfluidsynth keyboard python-osc
```

## Project Structure

```
monster_genie/
├── train_selection.py      # Training with GIANTsel dataset
├── train_resume.py         # Resume training from checkpoint
├── train.py               # Standard training script
├── params.py              # Configuration parameters
├── models.py              # Model definitions
├── x_transformer_1_23_2.py # Transformer implementation
├── model_loader.py        # Model loading utilities
├── midi_processors.py     # MIDI processing functions
├── inference_continuator.py # Offline MIDI continuation
├── interaction_midi.py    # Real-time MIDI keyboard interaction
├── interaction_osc.py     # OSC-based interaction
├── interaction_kbrd.py    # Computer keyboard interaction
├── visualizer.py          # Real-time visualization
├── Training-Data/         # Training datasets
├── save_models/           # Saved model checkpoints
└── samples/               # Sample MIDI files
```

## Training

### 1. Dataset Preparation

Place your training data in the `Training-Data/` directory:

- `giantMIDI_sel.pickle` - training dataset (train_selection.py)
- `giantMIDI_test.pickle` - Validation dataset (train.selection)
- `asigalov_train.pickle` - Alternative big training dataset (train.py)
- `asigalov_val.pickle` - Alternative big validation dataset (train.py)

### 2. Configuration

Edit `params.py` to configure training parameters:


### 3. Training Commands

#### Train with GIANTsel dataset:
```bash
python train_selection.py
```

#### Resume training from checkpoint:
```bash
python train_resume.py
```
This automatically finds the latest checkpoint in `save_models/` and resumes training.

#### Using SLURM (for cluster computing):
```bash
sbatch train_genie.sh
```

## Inference 

### 1. Inference Continuators 

Automatic inferences, from 9 MIDI files as context, 
guided with buttons extracted from the same MIDI file 
By default, the context len is fixed to 120 notes. Once reached 120 notes, the first ones are discarded.

from all 9 tests
```bash
python inference_continuator_all.py
```
1 test
```bash
python inference_continuator.py
```

encoder only (to visualize button structure)
```bash
python inference_encoder_all.py
```

### 2. Real-time Interaction

Interaction, generating buttons from MIDI keyboard, 
starting with a context extracted from a MIDI file

```bash
python interaction_continuator.py
```

Interaction from osc messages
```bash
python interaction_osc.py
```

OSC Message Format:
```
/button <button_number> <state>  # state: 1=noteon, 0=noteoff
/save                            # save performance
/reset                           # reset context
```

Default OSC settings:
- Listen IP: `0.0.0.0` (all interfaces)
- Port: `3003`


### 2. Model loaders

Available model types:
- `autoencoder` - Standard encoder-decoder
- `autoencoder_no_dtime` - Without delta-time tokens
- `autoencoder_w_encoder_antic` - With dtime in the encoder
- `decoder_only` - Decoder-only model
- `encoder_only` - Encoder-only model

## Visualization

Real-time visualization is available during interactive modes:
- Note activity display
- Button state visualization  
- Melodic contour representation

## Output

Generated files are saved in the `out/` directory:
- `*.mid` - Generated MIDI files
- `*_buttons.mid` - Button sequences as MIDI
- `*_e.mid` - Encoder outputs as MIDI






