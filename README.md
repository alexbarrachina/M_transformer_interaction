# Transformer-based Piano Genie Implementation

A PyTorch implementation of Piano Genie using transformer architecture with teacher forcing mode, supporting both training and real-time inference for musical interaction.

## Checkpoint analysis

Run `python checkpoint_analyzer.py` in the `tgenie` environment and open
http://127.0.0.1:8765 for cached checkpoint comparisons, six metric tabs, and
harmonic-evolution experiments. The Oracle tab includes ascending and descending
button sweeps. See the [analyzer guide](metrics/analyzer/README.md).
For plain-language explanations of every score, see
[Understanding the checkpoint metrics](CHECKPOINT_ANALYZER_METRICS.md).

## Requirements

Python 3.9.19

### Dependencies

```bash
torch
tqdm 
datasets
einops
wandb
matplotlib
python-rtmidi
pyfluidsynth
scipy
pretty_midi
```

Install with:
```bash
pip install torch tqdm datasets einops wandb matplotlib python-rtmidi pyfluidsynth
```

## Project Structure

```
monster_genie/
├── train_selection.py     # Training with GIANTsel dataset
├── train_resume.py        # Resume training from checkpoint
├── train.py               # Standard training script
├── params.py              # Configuration parameters
├── models.py              # Model definitions
├── x_transformer.py       # Transformer implementation
├── model_loader.py        # Model loading utilities
├── midi_processors.py     # MIDI processing functions
├── inference_continuator.py # Offline MIDI continuation
├── visualizer.py          # Real-time visualization
├── Training-Data/         # Training datasets
├── save_models/           # Saved model checkpoints
└── samples/               # Sample MIDI files
```

## Training

### Optional delta-time conditioning

`AE_style_jokerParam_dtime_tester_v1` extends the existing style/Joker model;
the default model is unchanged. It conditions pitch predictions on causal timing
features without adding sequence positions or predicting timing/duration.

In the `tgenie` environment, from the project root:

```bash
MODEL_NAME=AE_style_jokerParam_dtime_tester_v1 python train_style.py
```

The first run starts from the checkpoint configured for
`AE_style_jokerParam_tester_v2`. Subsequent runs resume
`save_models/AE_style_jokerParam_dtime_tester_v1_latest.pth` automatically.
`RESUME=0` starts again from the base checkpoint and replaces that timing
checkpoint when saving. `BATCH_SIZE` and `NUM_WORKERS` remain available.

Training stops after 10,000 optimizer updates: 1,000 updates of the timing MLP
at `1e-4`, followed by 9,000 also adapting the last two decoder blocks'
self-attention/feed-forward paths and final norm/pitch head at `1e-5`.
Encoders, style cross-attention, original input projection/embeddings, and Joker
parameter stay frozen. Checkpoints save every 500 updates and at completion,
including optimizer/scaler state, stage, configuration, base-checkpoint hash,
and Python/PyTorch random states. Resume restores training progress but does not
replay the exact shuffled data position or worker-local random states.
The timing preset runs no validation passes; quality evaluation is deferred.

`timing.py` shares causal feature calculation between training and performance:
absolute log interval, signed log interval/recent-scale ratio, log recent scale,
and an observed-zero indicator. The scale uses the previous 16 onsets, excluding
the current onset. Model intervals cap at 4.064 seconds; missing timing is masked.
Training keeps 50% original timing and mixes tempo/jitter (25%), grid snapping
(10%), steady tapping (5%), and pauses (10%); 25% of excerpts disable timing.
The decoder's optional inputs are `timing_features [B,T,4]` and
`timing_mask [B,T]`. Training/live wrappers accept them aligned with their
`[B,T+1]` pitch/event history and shift them to match each target pitch.

After training, use the checkpoint in the existing live application:

```bash
MODEL_NAME=AE_style_jokerParam_dtime_tester_v1 python interaction_buttons_style.py
```

Optionally set `CHECKPOINT_PATH` to another trained timing checkpoint. **D** toggles
timing (on by default). Timing history continues while disabled; switching
invalidates the cache. A gap of at least 10 seconds between onsets, or **R**,
reseeds the timing model's generation context from the primer while preserving
the recording, chosen style, timing-toggle state, and held-note releases.
Timing-off uses the adapted model, not the original checkpoint. Caching remains
disabled by default.

Run correctness checks and the four-update trainer smoke test:

```bash
python -m unittest test_timing test_train_style_joker tester.test_style_joker_param
```

The smoke test uses a small model, synthetic notes, and temporary checkpoints.
It does not train the production preset or run musical-quality/latency benchmarks.

### 1. Dataset Preparation

Place your training data in the `Training-Data/` directory:

- `giantMIDI_sel.pickle` - training dataset (train_selection.py)
- `giantMIDI_sel_test.pickle` - Validation dataset (train.selection)
- `asigalov_train.pickle` - Alternative big training dataset (train.py)
- `asigalov_val.pickle` - Alternative big validation dataset (train.py)


### 3. Training Commands

#### Train with GIANTsel dataset:
Edit `train_selection.py` to choose a model (models in `models.py`) in load_model() and load_hyperparameters()
By default, model_name='no_dtime_good_reference'

Then, execute:
```bash
python train_selection.py
```

#### Resume training from checkpoint:
Edit `train_resume.py` to choose a model (models in `models.py`) in load_model() and load_hyperparameters()
By default, model_name='no_dtime_good_reference'

Then, execute:
```bash
python train_resume.py
```
This automatically finds the latest checkpoint in `save_models/` and resumes training.

#### Using SLURM (for cluster computing):
```bash
sbatch train_genie.sh
```

### Checkpoints

To use the default `no_dtime_good_reference` model
1. download the checkpoint file:
https://drive.google.com/file/d/1CR90pEQwYupaEnG91ZI7Rd9iKVzbiXG8/view?usp=drive_link
2. put the file in the ./save_models folder

## Inference 

### 1. Inference Continuators 

Automatic inferences, from 9 MIDI files as context, 
guided with buttons extracted from the original MIDI files 
By default, the context len is fixed to 120 notes. Once reached 120 notes, the first ones are discarded.

from all 9 tests
```bash
python inference_continuator_all.py
```
1 test only
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
python interaction_dtime_only.py
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

Use `no_dtime_good_reference` model by default

Available model types:
- `autoencoder` - Standard encoder-decoder
- `autoencoder_no_dtime` - Without delta-time tokens
- `autoencoder_w_encoder_antic` - With dtime in the encoder
- `decoder_only` - Decoder-only model (decorder tester)
- `encoder_only` - Encoder-only model (button-compression tester)

## Visualization

Real-time visualization is available during interactive modes:
- Note activity display
- Button state visualization  
- Melodic contour representation

Set `Visualizer(show_event_numbers=True)` to display MIDI note and button numbers at the start of each live event. This option defaults to `False`.

## Output

Generated files are saved in the `out/` directory:
- `*.mid` - Generated MIDI files
- `*_buttons.mid` - Button sequences as MIDI
- `*_e.mid` - continuous-value Encoder outputs as MIDI


## Format conversion


    MIDI File ──────────────► midi_to_tokens() ──────► Flat Tokens ◄──── (pickles)
        │                                                     │
        │                                                     │                  
        │                                                     │                  
        │                                                     ▼
        │                                          tokens_to_dict()              
        │                                                     │
        │                                                     │                  
        │                                                     ▼                  
        │                                                  ┌─────────────────────────┐
        └──────────► midi_to_dict() ─────────────────────► │         DICT            │
                     (no offsets)                          │  {'dtime':[], 'dur':[], │
                                                           │   'pitch':[], ...}      │    
                                                           └─────────────────────────┘
                                                                      │
                                                                      ▼
                                                              dict_to_song()
                                                                      │
                                                                      ▼
                                                           ┌─────────────────────┐
                                                           │   Song (ms_score)   │
                                                           │ [['note', t, d, c,  │
                                                           │   p, v, patch], ...] │
                                                           └─────────────────────┘
                                                                      │
                                                                      ▼
                                                         ms_SONG_to_MIDI_Converter()
                                                                      │
                                                                      ▼
                                                                 MIDI File

