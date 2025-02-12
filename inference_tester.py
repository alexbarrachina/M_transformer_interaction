import time

#@title Import all needed modules


import TMIDIX
from model import *


"""# (LOAD MODEL)"""
full_path_to_model_checkpoint = "./SaveModel/giantMIDI_sel_436000_steps_0.0495_loss.pth" 

if torch.backends.mps.is_available(): 
  model_precision = "bfloat16" # @param ["bfloat16", "float16", "float32"]
else:
  model_precision = "bfloat16"

print('=' * 70)
print('Loading GIGA-Piano XL model...')

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

if torch.backends.mps.is_available():
    device_type = 'cpu'
    device = torch.device("cpu")
else:
    device_type = 'cuda'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
#ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

config = GPTConfig(512, # vocab_size
                  2048, # block_size
                  dim_feedforward=2048,
                  n_layer=24,
                  n_head=8,
                  n_embd=1024,
                  enable_rpr=False,
                  er_len=2048)

model = GPT(config)

#model = torch.nn.DataParallel(model)

model.load_state_dict(torch.load(full_path_to_model_checkpoint, map_location=device))

model.to(device)

model.eval()

"""# (CUSTOM MIDI)"""
# Upload your own seed MIDI
midi_file = './Samples/clairTester.midi'
uploaded_MIDI = open(midi_file, 'rb').read()

if True:

  print('Loading custom MIDI file...')
  score = TMIDIX.midi2ms_score(uploaded_MIDI)

  events_matrix = []

  itrack = 1

  while itrack < len(score):
      for event in score[itrack]:
          if event[0] == 'note' and event[3] != 9: # I suspect channel=9 are drums
              events_matrix.append(event)
      itrack += 1

  # Sorting...
  events_matrix.sort(key=lambda x: x[4], reverse=True)
  events_matrix.sort(key=lambda x: x[1])

  # recalculating timings
  for e in events_matrix:
      e[1] = int(e[1] / 10) # event time / 10
      e[2] = int(e[2] / 20) # event duration / 20

  # final processing...

  melody = []
  melody_chords = []
  memory_melody_chords = []

  pe = events_matrix[0] # the first event
  for e in events_matrix:

      time_ = max(0, min(126, e[1]-pe[1])) # time difference from previous events, but trunk to maximum 126
      dur = max(1, min(126, e[2])) # maximum duration 126

      ptc = max(1, min(126, e[4])) # maximum pitch 126
      vel = max(1, min(126, e[5])) # maximum velocity 126

      melody_chords.append([time_, dur+128, ptc+256, vel+384]) # 384 = 128*3 all events, melodies + chords

      if time_ != 0: # not a chord -> melody
        if ptc < 60:
          ptc = (ptc % 12) + 60 # avoid pitches < 60 by transposition
        melody.append([time_, dur+128, ptc+256, vel+384]) # suma 128s imagino que per fer one Hot vectors

      pe = e

  inputs = []

  for m in melody_chords:
    inputs.extend(m) # add to the end of the input list

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
      time_ = 0
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

          time_ += s[0] * 10

          dur = (s[1]-128) * 20

          channel = 0

          pitch = s[2]-256

          vel = s[3]-384





"""# ************************************ (GENERATE)**************************************************
# (SINGLE BLOCK CONTINUATION)
******************************************************************************************************+
"""

# Play with the settings to get different results

number_of_prime_notes = 128 # min:32, max:256
temperature = 1.0 # min:0.1, max:1


#===================================================================

print('Generation settings:')
print('=' * 70)
print('Number of prime notes:', number_of_prime_notes)
print('Model temperature:', temperature)
print('=' * 70)

seq_len = number_of_prime_notes*4

inp = inputs[:seq_len]
inp = range(0,512)
inp = torch.Tensor(inp)
inp = inp.unsqueeze(dim=0).type(torch.long)

#inp = torch.zeros(1,512).long()
inp = inp.to(device)  # Move input tensor to the correct device


  #with ctx:

timeStart = time.perf_counter()

for i in range(0,100):     
    next_token = model.generate_single(inp.to(device),
                                              temperature=temperature)
  
timeEnd = time.perf_counter()
print((timeEnd-timeStart) * 1000 / 100) # in miliseconds, promig#if (i%10==0):
      #print(i)