
import time
from pythonosc import udp_client
from pythonosc import osc_message_builder
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server


from model import *
from midiUtils import get_midifile_data, generate_midifile_from_list


"""# SETTINGS """
# Play with the settings to get different results
full_path_to_model_checkpoint = "./SaveModel/giantMIDI_Sel_latency_412000_steps_0.0564_loss.pth" 
midi_file = './Samples/clairTester.midi'
nameOut = './Out/inference_interaction'

number_of_prime_notes = 128 # min:32, max:256
number_of_tokens_to_generate = 1024 # min:512, max:1920
primer_len = number_of_prime_notes * 4 # 128*4 = 512 4 tokens per note
seq_len = number_of_tokens_to_generate + primer_len
temperature = 1.0 # min:0.1, max:1

SAVE_PERFORMANCE = True
saved_perf_list = [] # list that save the improvisation

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

model.load_state_dict(torch.load(full_path_to_model_checkpoint, map_location=device))
model.to(device)
model.eval()

"""# GET PRIMER FROM MIDI FILE """

inputs = get_midifile_data(midi_file)
inputs = torch.tensor(inputs, dtype=torch.long)

#gen_seq = torch.full((1,seq_len), TOKEN_PAD, dtype=torch.long,) # shape(2,1024)
# Add this at the global scope after model initialization
input_buffer = torch.full((primer_len,), TOKEN_PAD, dtype=torch.long)
input_buffer[:primer_len] = inputs[:primer_len]
# test
#primer_len = 10
#input_buffer = torch.tensor([3,4,1,2,3,4,1,2,3,4]) # add duration token

timeLast = time.perf_counter()
noteOn_dict = {}

"""# UPDATE DURATION FUNCTION """

def update_duration_in_buffer(input_buffer, pitch, duration):
    """Helper function to update the duration before the last occurrence of pitch"""
    # Convert duration from milliseconds to appropriate token value (0-127)
#    duration_token = max(0, min(127, int(duration/10)))  # Assuming 10ms per duration unit
    duration_tensor = torch.tensor([duration + 128], dtype=torch.long)  # Add 128 for duration tokens
    
    # Find the last occurrence of this pitch in the buffer
    buffer_list = input_buffer.tolist()
    for i in reversed(range(len(buffer_list))):
        if buffer_list[i] == pitch: 
            # Update the duration token that comes before this pitch
            input_buffer[i-1] = duration_tensor[0]
            break
    return input_buffer

""" OSC SENDER """
client = udp_client.SimpleUDPClient("127.0.0.1", 3333)

def sendOSC(note):    
    msg = osc_message_builder.OscMessageBuilder(address = '/note')
    msg.add_arg(note, arg_type='i')
    msg = msg.build()
    client.send(msg)


def oscReceiverNote(unused_addr, args, note, velocity): 
  global input_buffer  # Access the global buffer
  global timeLast # time of last note, global variable
  global noteOn_dict # button: (pitch, timeIn)

  #print("noteOn:", noteOn)
  timeNew = time.perf_counter()*1000
  
  if velocity > 0: # noteOn
    # time, dur+128, ptc+256, vel+384]

    # Update position token
    #position = timeNew - timeLast * 1000 # in miliseconds, promig#if (i%10==0):
    position = max(0, min(126, int(timeNew/10) - int(timeLast/10))) # time difference from previous events, but trunk to maximum 126
    timeLast = timeNew
    position_tensor = torch.tensor([position], dtype=torch.long)
    if SAVE_PERFORMANCE:
      saved_perf_list.append(input_buffer[0]) # save 1rst token before delete in input_buffer
    input_buffer = torch.cat([input_buffer[1:], position_tensor])

    # Add provisional duration using the last duration in buffer
    duration_tensor = input_buffer[-4].unsqueeze(0)  # Convert to 1D tensor, [-4] is the last duration token
    if SAVE_PERFORMANCE:
      saved_perf_list.append(input_buffer[0]) # save 1rst token before delete in input_buffer
    input_buffer = torch.cat([input_buffer[1:], duration_tensor]) 

    # Update pitch and velocity differently if it's a button or a keyborard note 
    if(note < 10): # it's a button
      # Reshape input_buffer for model input (add batch dimension)
      inp = input_buffer.unsqueeze(0)  # Shape: (1, 512)
      next_token = model.generate_single(inp.to(device), temperature=temperature) 
      pitch = next_token.item()
      sendOSC(pitch-256)  # -256 Convert from token to MIDI pitch
      # Update input buffer: remove first token and append new token, circular buffer
      pitch_tensor = next_token.squeeze().unsqueeze(0)  # Ensure 1D tensor
      if SAVE_PERFORMANCE:
        saved_perf_list.append(input_buffer[0]) # save 1rst token before delete in input_buffer
      input_buffer = torch.cat([input_buffer[1:], pitch_tensor])
      
      # Update velocity with a fixed value ¿¿¿???? TODO: inference velocity or use keyboard velocity
      velocity_tensor = input_buffer[-4].unsqueeze(0)  # Convert to 1D tensor, [-4] is the last velocity token
      if SAVE_PERFORMANCE:
        saved_perf_list.append(input_buffer[0]) # save 1rst token before delete in input_buffer
      input_buffer = torch.cat([input_buffer[1:], velocity_tensor]) 
    else: # it's a keyboard note
      if SAVE_PERFORMANCE:
        saved_perf_list.append(input_buffer[0]) # save 1rst token before delete in input_buffer
      input_buffer = torch.cat([input_buffer[1:], note+256])
      if SAVE_PERFORMANCE:
        saved_perf_list.append(input_buffer[0]) # save 1rst token before delete in input_buffer
      input_buffer = torch.cat([input_buffer[1:], velocity+384]) 
       
    # add (user_note, pitch, time) to dictionary
    noteOn_dict[note] = (pitch, timeNew)

  else: # noteOff
    # check noteoffs in noteOn list
    notes_to_update = []

    if note in noteOn_dict:
      # get pitch and time in dictionary of accumulated notesOns without noteOff
      pitch, noteOn_time = noteOn_dict[note]
      # Calculate actual duration
      duration = int((timeNew - noteOn_time) / 20)
      duration = max(1, min(126, duration)) # maximum duration 126
      # Update the duration in the buffer
      input_buffer = update_duration_in_buffer(input_buffer, pitch, duration)
      # Mark this note for removal from dictionary
      del noteOn_dict[note]
    
    # Update duration of notes that have been held for too long
    for button, (pitch, noteOn_time) in list(noteOn_dict.items()):
      if (timeNew - noteOn_time)  > 2520: # 2520 = 126*20 max duration
        # Update duration in buffer
        input_buffer = update_duration_in_buffer(input_buffer, pitch, 126) # force duration to max
        # Mark this note for removal from dictionary
        notes_to_update.append(button)

    # Remove updated notes from dictionary
    for button in notes_to_update:
        del noteOn_dict[button]

     
def oscReceiverSave(unused_addr, args):
  global input_buffer  # Access the global buffer

  for token in input_buffer:
    saved_perf_list.append(token)

    generate_midifile_from_list(saved_perf_list, nameOut)

'''def oscReceiverOff(unused_addr, args, noteOff): # A susbtituir per receives OSC
  global input_buffer  # Access the global buffer
  global timeLast # time of last note, global variable
  global noteOn_dict

  #print("noteOff:", noteOff)
  timeNew = time.perf_counter()*1000
  
  with buffer_lock:     
    # check noteoffs in noteOn list
    notes_to_update = []

    if noteOff in noteOn_dict:
      # get pitch and time in dictionary of accumulated notesOns without noteOff
      pitch, noteOn_time = noteOn_dict[noteOff]
      # Calculate actual duration
      duration = int((timeNew - noteOn_time) / 20)
      duration = max(1, min(126, duration)) # maximum duration 126
      # Update the duration in the buffer
      input_buffer = update_duration_in_buffer(input_buffer, pitch, duration)
      # Mark this note for removal from dictionary
      del noteOn_dict[noteOff]
    
    # Update duration of notes that have been held for too long
    for button, (pitch, noteOn_time) in list(noteOn_dict.items()):
      if (timeNew - noteOn_time)  > 2520: # 2520 = 126*20 max duration
        # Update duration in buffer
        input_buffer = update_duration_in_buffer(input_buffer, pitch, 126) # force duration to max
        # Mark this note for removal from dictionary
        notes_to_update.append(button)

    # Remove updated notes from dictionary
    for button in notes_to_update:
        del noteOn_dict[button]

    #print('out:', input_buffer)  '''

dispatcher = Dispatcher()
dispatcher.map("/note", oscReceiverNote, "note", "velocity")
dispatcher.map("/save", oscReceiverSave)  

server = osc_server.ThreadingOSCUDPServer(
      ("127.0.0.1", 3334), dispatcher)
print("Serving on {}".format(server.server_address))
server.serve_forever()


