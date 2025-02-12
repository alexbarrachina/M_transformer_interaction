
from model import *
from midiUtils import get_midifile_data, generate_midifile_from_list

"""# SETTINGS """
# Play with the settings to get different results
full_path_to_model_checkpoint = "./SaveModel/giantMIDI_Sel_latency_412000_steps_0.0564_loss.pth" 
midi_file = './Samples/clairTester.midi'
nameOut = './Out/singleBlockContinuator'

number_of_prime_notes = 128 # min:32, max:256
number_of_tokens_to_generate = 1024 # min:512, max:1920
temperature = 0.8 # min:0.1, max:1
show_stats = True 

"""# SET MODEL PRECISION""" 
# Model precision option
if torch.backends.mps.is_available(): 
  model_precision = "float32" # @param ["bfloat16", "float16", "float32"]
  device = torch.device("cpu")
else:
  model_precision = "bfloat16"
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# bfloat16 == Third precision/triple speed (if supported, otherwise the model will default to float16)
# float16 == Half precision/double speed
# float32 == Full precision/normal speed

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


'''
if model_precision == 'bfloat16':
  dtype = 'bfloat16'

if model_precision == 'float32':
  dtype = 'float32'

if device_type == 'cuda':
  if model_precision == 'bfloat16' and torch.cuda.is_bf16_supported():
    dtype = 'bfloat16'
  else:
    dtype = 'float16'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
#ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
'''

"""# (LOAD MODEL)"""

print('=' * 70)
print('Loading GIGA-Piano XL model...')
config = GPTConfig(512,
                  2048,
                  dim_feedforward=2048,
                  n_layer=24,
                  n_head=8,
                  n_embd=1024,
                  enable_rpr=True,
                  er_len=2048)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT(config)

#model = torch.nn.DataParallel(model)

model.load_state_dict(torch.load(full_path_to_model_checkpoint, map_location=device))
model.to(device)
model.eval()

"""# GET PRIMER FROM MIDI FILE """

inputs = get_midifile_data(midi_file)


"""# INFERENCE  """

inp = inputs[:number_of_prime_notes*4] # 128*4 = 512
# If you want to generate from a blank state
#inp = [126, 126+128, 0+256, 0+384]

  #with ctx:
rand_seq = model.generate_batches(torch.Tensor(inp).to(device),
                                              target_seq_length=number_of_tokens_to_generate,
                                              temperature=temperature,
                                              num_batches=1,
                                              verbose=show_stats)

out1 = rand_seq[0].cpu().tolist()

generate_midifile_from_list(out1, nameOut)