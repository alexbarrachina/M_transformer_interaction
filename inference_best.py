

#@title Import all needed modules

import os
import random
import copy
import statistics
from collections import OrderedDict

from tqdm import tqdm

import numpy as np


import TMIDIX
from GPT2RGAX import *


"""# (LOAD MODEL)"""
full_path_to_model_checkpoint = "./gpt2_rpr_checkpoint_1_epoch_176000_steps_1.7523_loss.pth" #@param {type:"string"}

#@markdown Model precision option
model_precision = "bfloat16" # @param ["bfloat16", "float16", "float32"]

#@markdown bfloat16 == Third precision/triple speed (if supported, otherwise the model will default to float16)
#@markdown float16 == Half precision/double speed
#@markdown float32 == Full precision/normal speed

plot_tokens_embeddings = False # @param {type:"boolean"}

print('=' * 70)
print('Loading GIGA-Piano XL model...')

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

if torch.backends.mps.is_available():
  device_type = 'cpu'
  #mps_device = torch.device("mps")
else:
  device_type = 'cuda'

if model_precision == 'bfloat16':
  dtype = 'bfloat16'

if device_type == 'cuda':
  if model_precision == 'bfloat16' and torch.cuda.is_bf16_supported():
    dtype = 'bfloat16'
  else:
    dtype = 'float16'

if model_precision == 'float16':
  dtype = 'float16'

if model_precision == 'float32':
  dtype = 'float32'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

config = GPTConfig(512,
                  2048,
                  dim_feedforward=2048,
                  n_layer=24,
                  n_head=8,
                  n_embd=1024,
                  enable_rpr=True,
                  er_len=2048)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT(config)

model = torch.nn.DataParallel(model)

model.load_state_dict(torch.load(full_path_to_model_checkpoint, map_location=device))

model.to(device)

model.eval()

print('Done!')
print('=' * 70)
print('Model will use', dtype, 'precision...')

# summary(model)
#print(model)
#print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

"""# (CUSTOM MIDI)"""

#@title Upload your own seed MIDI
midi_file = './Samples/clairTester.midi'
uploaded_MIDI = open(midi_file, 'rb').read()

print('=' * 70)

if True:

  score = TMIDIX.midi2ms_score(uploaded_MIDI)
  #f = list(uploaded_MIDI.keys())[0]

  print('Loading custom MIDI file...')

  events_matrix = []

  itrack = 1

  while itrack < len(score):
      for event in score[itrack]:
          if event[0] == 'note' and event[3] != 9:
              events_matrix.append(event)
      itrack += 1

  # Sorting...
  events_matrix.sort(key=lambda x: x[4], reverse=True)
  events_matrix.sort(key=lambda x: x[1])

  # recalculating timings
  for e in events_matrix:
      e[1] = int(e[1] / 10)
      e[2] = int(e[2] / 20)

  # final processing...

  melody = []
  melody_chords = []
  memory_melody_chords = []

  pe = events_matrix[0]
  for e in events_matrix:

      time = max(0, min(126, e[1]-pe[1]))
      dur = max(1, min(126, e[2]))

      ptc = max(1, min(126, e[4]))
      vel = max(1, min(126, e[5]))

      melody_chords.append([time, dur+128, ptc+256, vel+384])

      if time != 0:
        if ptc < 60:
          ptc = (ptc % 12) + 60
        melody.append([time, dur+128, ptc+256, vel+384])

      pe = e

  inputs = []

  for m in melody_chords:
    inputs.extend(m)

  memory_chunks_train = []

  for i in range(0, (len(inputs) // 128)*128, 128): # 32 notes
    memory_chunks_train.append(inputs[i:i+128])

  memory_chunks_test = []

  for i in range(64, (((len(inputs) // 128)-1)*128), 128): # 32 notes
    memory_chunks_test.append(inputs[i:i+128])

  memory_chunks = memory_chunks_train + memory_chunks_test

  print('Done!')
  print('=' * 70)
  print('Composition has', len(melody_chords), 'notes')
  print('Composition has', len(inputs), 'tokens')
  print('Composition has', len(melody), 'melody notes')
  print('Composition has', len(melody)*4, 'melody tokens')
  print('=' * 70)

  out1 = inputs

  if len(out1) != 0:

      song = out1
      song_f = []
      time = 0
      dur = 0
      vel = 0
      pitch = 0
      channel = 0

      son = []

      song1 = []

      for s in song:
        if s > 127:
          son.append(s)

        else:
          if len(son) == 4:
            song1.append(son)
          son = []
          son.append(s)

      for s in song1:

          time += s[0] * 10

          dur = (s[1]-128) * 20

          channel = 0

          pitch = s[2]-256

          vel = s[3]-384

          song_f.append(['note', time, dur, channel, pitch, vel ])

      detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                                output_signature = 'GIGA-Piano XL',
                                                                output_file_name = './Out/prior',
                                                                track_name='Project Los Angeles',
                                                                list_of_MIDI_patches=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                                                )



"""# ************************************ (GENERATE)**************************************************
# (SINGLE BLOCK CONTINUATION)
******************************************************************************************************+
"""

# Play with the settings to get different results

custom_MIDI_or_improvisation = True 
number_of_prime_notes = 128 # min:32, max:256
number_of_tokens_to_generate = 1024 # min:512, max:1920
temperature = 1.0 # min:0.1, max:1
number_of_batches = 2 # min:1, max:8
show_stats = True 

#===================================================================
print('=' * 70)
print('GIGA-Piano XL Music Model Continuation Generator')
print('=' * 70)

print('Generation settings:')
print('=' * 70)
print('Number of prime notes:', number_of_prime_notes)
print('Model temperature:', temperature)
print('Number of batches:', number_of_batches)
print('=' * 70)

for i in range(0,9):
  print('file',i)
  if custom_MIDI_or_improvisation:
    inp = inputs[:number_of_prime_notes*4]
  else:
    inp = [126, 126+128, 0+256, 0+384]

  with ctx:
    rand_seq = model.module.generate_batches(torch.Tensor(inp),
                                              target_seq_length=number_of_tokens_to_generate,
                                              temperature=temperature,
                                              num_batches=number_of_batches,
                                              verbose=show_stats)

  out1 = rand_seq[0].cpu().tolist()

  print('Done!')

  #@title Explore generated continuations
  batch_number = 0 #@param {type:"slider", min:0, max:7, step:1}
  render_MIDI_to_audio = True # @param {type:"boolean"}

  if batch_number >= number_of_batches:
    bn = 0
  else:
    bn = batch_number

  print('=' * 70)
  print('Displaying batch #:',bn )
  print('=' * 70)

  out1 = rand_seq[bn].cpu().tolist()

  if len(out1) != 0:

      song = out1
      song_f = []
      time = 0
      dur = 0
      vel = 0
      pitch = 0
      channel = 0

      son = []

      song1 = []

      for s in song:
        if s > 127:
          son.append(s)

        else:
          if len(son) == 4:
            song1.append(son)
          son = []
          son.append(s)

      for s in song1:

          time += s[0] * 10

          dur = (s[1]-128) * 20

          channel = 0

          pitch = s[2]-256

          vel = s[3]-384

          song_f.append(['note', time, dur, channel, pitch, vel ])

      nameOut = './Out/singleBlockContinuator'+str(i)
      detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                                output_signature = 'GIGA-Piano XL',
                                                                output_file_name = nameOut,
                                                                track_name='Project Los Angeles',
                                                                list_of_MIDI_patches=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                                                )

