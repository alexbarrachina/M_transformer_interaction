
from model import *
from midiUtils import get_midifile_data, generate_midifile_from_list

"""# SETTINGS """
# Play with the settings to get different results
full_path_to_model_checkpoint = "./SaveModel/giantMIDI_Sel_latency_412000_steps_0.0564_loss.pth" 
midi_file = './Samples/clairTester.midi'
nameOut = './Out/generate_single'

number_of_prime_notes = 128 # min:32, max:256
number_of_tokens_to_generate = 1024 # min:512, max:1920
temperature = 0.8 # min:0.1, max:1
show_stats = True 

"""# RECURSIVE FUNCTION """
def generate_batches(primer=None, target_seq_length=1024, temperature=1, num_batches=1, verbose=True):

        #assert (not self.training), "Cannot generate while in training mode"

            if verbose: print("Generating sequence of max length:", target_seq_length) 
            gen_seq = torch.full((num_batches,target_seq_length), TOKEN_PAD, dtype=torch.long, device=device) # shape(2,1024)

            num_primer = len(primer) # primer shape (512)
            gen_seq[..., :num_primer] = primer.type(torch.long).to(device) # shape(2, 512)

            cur_i = num_primer
            while(cur_i < target_seq_length):
                next_token = model.generate_single(gen_seq[..., :cur_i])
                gen_seq[:, cur_i] = next_token

                cur_i += 1
                if(cur_i % 50 == 0):
                    if verbose: print(cur_i, "/", target_seq_length)

            return gen_seq[:, :cur_i] #shape(2,513)


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
rand_seq = generate_batches(torch.Tensor(inp).to(device),
                                              target_seq_length=number_of_tokens_to_generate,
                                              temperature=temperature,
                                              num_batches=1,
                                              verbose=show_stats)



out1 = rand_seq[0].cpu().tolist()

generate_midifile_from_list(out1, nameOut)

