from model import *


#@markdown Model precision option
if torch.backends.mps.is_available(): 
  model_precision = "bfloat16" # @param ["bfloat16", "float16", "float32"]
else:
  model_precision = "bfloat16"

# bfloat16 == Third precision/triple speed (if supported, otherwise the model will default to float16)
# float16 == Half precision/double speed
# float32 == Full precision/normal speed

plot_tokens_embeddings = False 

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

config = GPTConfig(512, # vocab_size
                  2048, # block_size
                  dim_feedforward=2048,
                  n_layer=24,
                  n_head=8,
                  n_embd=1024,
                  enable_rpr=False,
                  er_len=2048)

model = GPT(config)

"""# (LOAD MODEL)"""
full_path_to_model_checkpoint = "./SaveModel/giantMIDI_Sel_latency_412000_steps_0.0564_loss.pth" 
checkpoint = torch.load(full_path_to_model_checkpoint, map_location=device)

model.eval()
model.load_state_dict(checkpoint, strict=False)

# Move the model to the appropriate device
model = model.to(device)

# Apply dynamic quantization
'''quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear},  # Layers to quantize
    dtype=torch.qint8
)'''

# Create a dummy input tensor
dummy_input = torch.zeros((1, config.block_size), dtype=torch.long, device=device)

# save the model with TorchScript
MODEL_NAME = 'model_scripted.ts'
# Use torch.jit.trace instead of torch.jit.script
#model_scripted = torch.jit.trace(model, dummy_input)
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save(MODEL_NAME) # Save
print('Model converted and saved as model_scripted.ts')

# Print available methods
print("Available methods:", model_scripted.code)


# load: test
model = torch.jit.load(MODEL_NAME)
#print('Model loaded: ', model)

print("Model converted to TorchScript and saved as model_scripted.ts")
