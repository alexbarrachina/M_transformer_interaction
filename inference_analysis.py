import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from midiUtils import midi2ms_score
import time

from midiUtils import get_midifile_data
from model import *


""" PATHS """
full_path_to_model_checkpoint = "./SaveModel/giantMIDI_Sel_latency_412000_steps_0.0564_loss.pth" 
midi_file = './Samples/clairTester.midi'


""" MODEL PRECISION """
if torch.backends.mps.is_available(): 
  model_precision = "bfloat16" # @param ["bfloat16", "float16", "float32"]
else:
  model_precision = "bfloat16"

print('=' * 70)
print('Loading GIGA-Piano XL model...')

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

if torch.backends.mps.is_available():
    device_type = 'mps'
    device = torch.device("mps")
else:
    device_type = 'cuda'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_precision == 'bfloat16':
  dtype = 'bfloat16'


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

inputs = get_midifile_data(midi_file)

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
inp = inp.contiguous()
inp = inp.to(device, non_blocking=True)  # Move input tensor to the correct device



with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.MPS],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs/inference_profile'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with torch.no_grad():
        for i in range(0, 100):
            with record_function("model_inference"):
                next_token = model.generate_single(inp, temperature=temperature)
            prof.step()

print("Profiling complete. Check TensorBoard for detailed reports.")

# tensorboard --logdir=./runs/inference_profile