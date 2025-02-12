from model import *
from midiUtils import get_midifile_data, generate_midifile_from_list

"""# SETTINGS """
# Play with the settings to get different results
full_path_to_model_checkpoint = "./Model/GIGA_Piano_XL_Trained_Model_138600_steps_1.259_loss.pth" 
midi_file = './Samples/clairTester.midi'
outFileName = './Out/single_original_model'

number_of_prime_notes = 128 # min:32, max:256
number_of_tokens_to_generate = 1024 # min:512, max:1920
temperature = 1.0 # min:0.1, max:1
num_generations = 10
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

model = GPT(config)

# Load and remove 'module.' prefix from state dict keys
state_dict = torch.load(full_path_to_model_checkpoint, map_location=device)
new_state_dict = {}
for k, v in state_dict.items():
    name = k.replace('module.', '') # Remove 'module.' prefix
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

"""# GET PRIMER FROM MIDI FILE """

inputs = get_midifile_data(midi_file)


"""# INFERENCE  """

inp = inputs[:number_of_prime_notes*4] # 128*4 = 512
# If you want to generate from a blank state
#inp = [126, 126+128, 0+256, 0+384]

for i in range(0,num_generations):

  #with ctx:
  rand_seq = model.generate_batches(torch.Tensor(inp).to(device),
                                    target_seq_length=number_of_tokens_to_generate,
                                    temperature=temperature,
                                    num_batches=1,
                                    verbose=show_stats)

  out1 = rand_seq[0].cpu().tolist()

  nameOut = outFileName +str(i+1)

  generate_midifile_from_list(out1, nameOut)


      
