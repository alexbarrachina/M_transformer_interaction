
from model import *
from midiUtils import get_midifile_data, generate_midifile_from_list, validate_token, what_token, PITCH_OFF, VEL_OFF, DUR_OFF, VOCAB_SIZE

"""# SETTINGS """
# Play with the settings to get different results
full_path_to_model_checkpoint = "./SaveModel/giantMIDI_Sel_latency_412000_steps_0.0564_loss.pth" 
midi_file = './Samples/clairTester.midi'
nameOut = './Out/generate_batches'

number_of_prime_notes = 128 # min:32, max:256
number_of_tokens_to_generate = 1024 # min:512, max:1920
temperature = 1.0 # min:0.1, max:1
show_stats = True 


"""# RECURSIVE FUNCTION """
def generate_batches(primer=None, target_seq_length=1024, temperature=1, num_batches=1, verbose=True):

        #assert (not self.training), "Cannot generate while in training mode"

            if verbose: print("Generating sequence of max length:", target_seq_length) 
            gen_seq = torch.full((num_batches,target_seq_length), TOKEN_PAD, dtype=torch.long, device=device) # shape(2,1024)

            num_primer = len(primer) # primer shape (512)
            gen_seq[..., :num_primer] = torch.tensor(primer, dtype=torch.long, device=device) # shape(2, 512)

            cur_i = num_primer
            while(cur_i < target_seq_length):
                next_token = model.generate_single_mps(gen_seq[..., :cur_i])
                gen_seq[:, cur_i] = next_token

                print(next_token.item())
                cur_i += 1
                if(cur_i % 50 == 0):
                    if verbose: print(cur_i, "/", target_seq_length)

            return gen_seq[:, :cur_i] #shape(2,513)

def generate_batches_validate(primer=None, target_seq_length=1024, temperature=1, num_batches=1, verbose=True):

        #assert (not self.training), "Cannot generate while in training mode"

            if verbose: print("Generating sequence of max length:", target_seq_length) 
            gen_seq = torch.full((num_batches,target_seq_length), TOKEN_PAD, dtype=torch.long, device=device) # shape(2,1024)


            num_primer = len(primer) # primer shape (512)
            gen_seq[..., :num_primer] = torch.tensor(primer, dtype=torch.long, device=device)
            i = num_primer
            for i in range(num_primer, target_seq_length):
            #for i in range(num_primer, target_seq_length): # Generate next token using current buffer state
                next_token = model.generate_single(gen_seq[..., :i])
                validate_token(next_token.item(), i)
                if(i % 50 == 0):
                    if verbose: print(i, "/", target_seq_length)

                gen_seq[:, i] = next_token
                i+=1
            return gen_seq[:, :i] #shape(2,513)

def generate_circular(
    primer: torch.Tensor, 
    target_seq_length: int = 1024,
    num_batches: int = 1,
    temperature: float = 1.0,
    verbose=True
) -> torch.Tensor:
    """
    Generates a sequence using a circular buffer approach with efficient updates.
    
    Args:
        primer: Initial sequence to start generation
        temperature: Sampling temperature
        verbose: Whether to print progress
    
    Returns:
        Generated sequence of tokens
    """

    #device = primer.device
    buffer_size = len(primer)

    # Initialize circular buffer, with 4 extra tokens to store the last 4 generated tokens
    circular_buffer = torch.full((num_batches, buffer_size+4,), 0, dtype=torch.long, device=device)
    circular_buffer[...,:buffer_size] = torch.tensor(primer, dtype=torch.long, device=device)

    for i in range(buffer_size, target_seq_length, 4): 
      # Generate next 4 tokens using the last 4 tokens of the buffer
      # last 4 tokens reserved for 4 new tokens
      next_token = model.generate_single_by_type(circular_buffer[...,:-4], position=i, temperature=temperature) 
      circular_buffer[...,-4] = next_token
      next_token = model.generate_single_by_type(circular_buffer[...,:-3], position=i+1, temperature=temperature)
      circular_buffer[...,-3] = next_token
      next_token = model.generate_single_by_type(circular_buffer[...,:-2], position=i+2, temperature=temperature)
      circular_buffer[...,-2] = next_token
      next_token = model.generate_single_by_type(circular_buffer[...,:-1], position=i+3, temperature=temperature)
      circular_buffer[...,-1] = next_token

      # Update circular buffer: shift left 4 tokens
      circular_buffer = torch.roll(circular_buffer, shifts=-4, dims=1)
            
    return circular_buffer

def generate_circular_when_full(
    primer: torch.Tensor, 
    max_buf_len: int = 2048,
    total_seq_len: int = 2048,
    num_batches: int = 1,
    temperature: float = 1.0,
    verbose=True
) -> torch.Tensor:
    """
    Starting from a primer, new tokens are added to a buffer until the buffer size is reached. (target_seq_length)
    Then the buffer is updated in a circular way, so the last token is always the last generated token
    and the first token is the first removed token. tokens are added and removed in blocks of 4 tokens.
    Args:
        primer: Initial sequence to start generation
        max_buf_len: maximum context length
        total_seq_len: total performance length
        temperature: Sampling temperature
        verbose: Whether to print progress
    
    Returns:
        Generated sequence of tokens
    """

    num_primer = len(primer)
    
    assert max_buf_len > num_primer, "Target length must be greater than primer length"
    assert max_buf_len % 4 == 0, "Target length should be multiple of 4 for MIDI token structure"
    assert num_primer % 4 == 0, "Primer length should be multiple of 4 for MIDI token structure"
   
    saved_perf_list = []

    # Initialize circular buffer, with 4 extra tokens to store the last 4 generated tokens
    buffer = torch.full((num_batches, max_buf_len,), 0, dtype=torch.long, device=device)
    buffer[...,:num_primer] = torch.tensor(primer, dtype=torch.long, device=device)
    saved_perf_list[:num_primer] = primer

    for i in range(num_primer, max_buf_len-4):
        # Generate next token using all previous context
        next_token = model.generate_single(buffer[...,:i], temperature=temperature)            
        # Add the new token
        buffer[...,i] = next_token
        print(next_token.item())
        saved_perf_list.append(next_token.item())

    for i in range(max_buf_len, total_seq_len, 4): 
      # Generate next 4 tokens using the last 4 tokens of the buffer
      # last 4 tokens reserved for 4 new tokens
      next_token = model.generate_single_by_type(buffer[...,:-4], position=i, temperature=temperature) 
      buffer[...,-4] = next_token
      saved_perf_list.append(next_token.item())
      next_token = model.generate_single_by_type(buffer[...,:-3], position=i+1, temperature=temperature)
      buffer[...,-3] = next_token
      saved_perf_list.append(next_token.item())
      next_token = model.generate_single_by_type(buffer[...,:-2], position=i+2, temperature=temperature)
      buffer[...,-2] = next_token
      saved_perf_list.append(next_token.item())
      next_token = model.generate_single_by_type(buffer[...,:-1], position=i+3, temperature=temperature)
      buffer[...,-1] = next_token
      saved_perf_list.append(next_token.item())

      print(buffer[...,-4:])

      # Update circular buffer: shift left 4 tokens
      buffer = torch.roll(buffer, shifts=-4, dims=1)
            
    return saved_perf_list

def generate_circular_mantain_primer(
    primer: torch.Tensor, 
    max_buf_len: int = 716,
    total_seq_len: int = 2048,
    temperature: float = 1.0,
    num_batches: int = 1,
    verbose=True
) -> torch.Tensor:
    """
    Generates a sequence using a circular buffer approach mantaining the primer at the beginning.
    The primer is not generated, only the new tokens are generated.
    The buffer starts with the primer and then the new tokens are added, until the target_seq_length is reached.
    Thenm the buffer (except the primer) is updated in a circular way, so the last token is always the last generated token
    and the first token after the primer is the first removed token.
    Args:
        primer: Initial sequence to start generation
        target_seq_length: Number of tokens to generate (primer + new tokens)
        temperature: Sampling temperature
        verbose: Whether to print progress
    
    Returns:
        Generated sequence of tokens
    """

    num_primer = len(primer)
    # Validation
    assert max_buf_len > num_primer, "Target length must be greater than primer length"
    assert max_buf_len % 4 == 0, "Target length should be multiple of 4 for MIDI token structure"
    assert num_primer % 4 == 0, "Primer length should be multiple of 4 for MIDI token structure"
   
    saved_perf_list = []

    # Initialize circular buffer, with 4 extra tokens to store the last 4 generated tokens
    buffer = torch.full((num_batches, max_buf_len,), 0, dtype=torch.long, device=device)
    buffer[...,:num_primer] = torch.tensor(primer, dtype=torch.long, device=device)
    saved_perf_list[:num_primer] = primer


    # Generate tokens until we reach target_seq_length
    for i in range(num_primer, max_buf_len-4):
        # Generate next token using all previous context
        next_token = model.generate_single(buffer[...,:i], temperature=temperature)            
        # Add the new token
        buffer[...,i] = next_token
        print(next_token.item())
        saved_perf_list.append(next_token.item())
         
    # Phase 2: Circular updates after primer
    for i in range(max_buf_len, total_seq_len, 4): 
     # Circular update: shift tokens after primer one position left and add new token at the end
      # last 4 tokens reserved for 4 new tokens
      next_token = model.generate_single_by_type(buffer[...,:-4], position=i, temperature=temperature) 
      buffer[...,-4] = next_token
      saved_perf_list.append(next_token.item())
      next_token = model.generate_single_by_type(buffer[...,:-3], position=i+1, temperature=temperature)
      buffer[...,-3] = next_token
      saved_perf_list.append(next_token.item())
      next_token = model.generate_single_by_type(buffer[...,:-2], position=i+2, temperature=temperature)
      buffer[...,-2] = next_token
      saved_perf_list.append(next_token.item())
      next_token = model.generate_single_by_type(buffer[...,:-1], position=i+3, temperature=temperature)
      buffer[...,-1] = next_token
      saved_perf_list.append(next_token.item())

      print(buffer[...,-4:])
      # Keep primer untouched and shift left the rest 4 tokens
      buffer[..., num_primer:] = torch.roll(buffer[..., num_primer:], shifts=-4, dims=-1)

            
    return saved_perf_list


"""# SET MODEL PRECISION""" 
# Model precision option
if torch.backends.mps.is_available(): 
  model_precision = "float32" # @param ["bfloat16", "float16", "float32"]
  device = torch.device("mps")
else:
  model_precision = "bfloat16"
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device ",device)

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

rand_seq = generate_batches(inp)

rand_seq = rand_seq[0].cpu().tolist()

generate_midifile_from_list(rand_seq, nameOut) 
