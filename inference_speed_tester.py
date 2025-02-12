import time

from midiUtils import get_midifile_data
from model import *

""" SETTINGS """
full_path_to_quantized_model = "./model_scripted.ts"
full_path_to_model_checkpoint = "./SaveModel/giantMIDI_Sel_latency_412000_steps_0.0564_loss.pth" 

midi_file = './Samples/clairTester.midi'
number_of_prime_notes = 128 # min:32, max:256
seq_len = number_of_prime_notes*4
temperature = 1.0 # min:0.1, max:1


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

#model = torch.jit.load(full_path_to_quantized_model, map_location=device)
config = GPTConfig(512,
                  2048,
                  dim_feedforward=2048,
                  n_layer=24,
                  n_head=8,
                  n_embd=1024,
                  enable_rpr=True,
                  er_len=2048)

model = GPT(config)
model.load_state_dict(torch.load(full_path_to_model_checkpoint, map_location=device))
model.to(device)
model.eval()

"""# (CUSTOM MIDI)"""
# Upload your own seed MIDI


inputs = get_midifile_data(midi_file)

inp = inputs[:seq_len]
inp = torch.Tensor(inp)
inp = inp.unsqueeze(dim=0).type(torch.long)
inp = inp.to(device)  # Move input tensor to the correct device

timeStart = time.perf_counter()

for i in range(0,100):     
    next_token = model.generate_single(inp.to(device),
                                              temperature=temperature)
  
timeEnd = time.perf_counter()
print((timeEnd-timeStart) * 1000 / 100) # in miliseconds, promig#if (i%10==0):
