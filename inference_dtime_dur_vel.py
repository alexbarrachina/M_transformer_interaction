
import time
import sys
import fluidsynth
import os 
# pip install pyfluidsynth
from typing import Optional, List
from rtmidi.midiconstants import NOTE_ON, NOTE_OFF
from rtmidi.midiutil import open_midiinput
import rtmidi
# pip install python-rtmidi
from threading import Lock, Thread

from model import *
from midiUtils import get_midifile_data, generate_midifile_from_list, DUR_OFF, PITCH_OFF, VEL_OFF, time2quant, log, dur2quant

"""# SETTINGS """
# Play with the settings to get different results
full_path_to_model_checkpoint = "./SaveModel/giantMIDI_Sel_latency_412000_steps_0.0564_loss.pth" 
midi_file = './Samples/clairTester.midi'
nameOut = './Out/inference_dtime'

number_of_prime_notes = 128 # min:32, max:256
number_of_tokens_to_play = 2048 
target_seq_length = 2048
temperature = 1.0 # min:0.1, max:1

saved_perf_list = []
SAVE_PERFORMANCE = True

'''THREADING'''
# Add these at the global scope after your imports
buffer_lock = Lock()
save_lock = Lock()
threads = []

'''MIDI IN CALLBACK'''
def midiin_callback(event, data=None):
    message, deltatime = event

    if message[0] & 0xF0 == NOTE_ON:
        status, note, velocity = message
        #channel = (status & 0xF) + 1
        with buffer_lock: # lock to avoid race condition
            manageNote(note, velocity)

    if message[0] & 0xF0 == NOTE_OFF: 
        status, note, velocity = message
        #with buffer_lock: # lock to avoid race condition, temporary disabled for debugging
        manageNote(note, 0)
    
    if message[0] & 0xF0 == 176:  # 176 is the status for control change
        if message[1] == 7: # Using volume slider as a trigger to save performance
          with save_lock:
            generate_midifile_from_list(saved_perf_list, nameOut)
            os._exit(1)


"""# FLUIDSYNTH INIT """

fs = fluidsynth.Synth()
fs.start()

sfid = fs.sfload("./piano.sf2")
fs.program_select(0, sfid, 0, 0)


def playNote(note, velocity=100):
    print("fluidNote", note, velocity)
    if velocity > 0:
        fs.noteon(0, note, velocity)
    else:
        fs.noteoff(0, note) 

"""# SET MODEL PRECISION""" 
# Model precision option
if torch.backends.mps.is_available(): 
  model_precision = "float32" # @param ["bfloat16", "float16", "float32"]
  device = torch.device("cpu")
else:
  model_precision = "bfloat16"
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model.load_state_dict(torch.load(full_path_to_model_checkpoint, map_location=device))
model.to(device)
model.eval()

"""# GET PRIMER FROM MIDI FILE """

primer = get_midifile_data(midi_file)

"""# BUILD BUFFER  """

num_primer = len(primer)
circular_size = target_seq_length -4 - num_primer

assert target_seq_length > num_primer, "Target length must be greater than primer length"
assert target_seq_length % 4 == 0, "Target length should be multiple of 4 for MIDI token structure"
assert circular_size % 4 == 0, "Primer length should be multiple of 4 for MIDI token structure"

 # Initialize circular buffer, with 4 extra tokens to store the last 4 generated tokens
buffer = torch.full((1, target_seq_length,), 0, dtype=torch.long, device=device)
buffer[...,:num_primer] = torch.tensor(primer, dtype=torch.long, device=device)
noteOn_dict = {} # dictionary to store noteOns

startTime = time.time() # time of init session
timeLast = 0

# current position in buffer
curr_i = num_primer

"""# INFERENCE  """
def manageNote( note, velocity):
    """
    Starting from a primer, new tokens are added to a buffer until the buffer size is reached. (target_seq_length)
    Then the buffer is updated in a circular way, so the last token is always the last generated token
    and the first token is the first removed token. tokens are added and removed in blocks of 4 tokens.
    Args:
        primer: Initial sequence to start generation
        target_seq_length: maximum context length
        temperature: Sampling temperature
        verbose: Whether to print progress
    
    Returns:
        Generated sequence of tokens
    """

    global buffer  # Access the global buffer
    global timeLast # time of last note, global variable 
    global curr_i # current position in buffer

    timeNew = (time.time() - startTime)*1000 # time since init session in miliseconds

    #if curr_i < (target_seq_length-4): TODO: circular buffer
    if velocity > 0: # NOTE_ON
        # time, dur+ DUR_OFF, ptc+PITCH_OFF, vel+VEL_OFF]
        # Update dtime token
        dtime = max(0, min(126, time2quant(timeNew) - time2quant(timeLast))) # time difference from previous events, but trunk to maximum 126
        timeLast = timeNew        # Generate next token using all previous context
        buffer[...,curr_i] = torch.tensor(dtime, dtype=torch.long, device=device)

        # Add provisional duration copying last duration
        buffer[...,curr_i+1] = buffer[...,curr_i+1-4]

        # inference pitch
        if(note < 20): # it's a button
            next_token = model.generate_single_by_type(buffer[...,:curr_i+2], position=curr_i+2, temperature=temperature)
            buffer[...,curr_i+2] = next_token
            pitch = next_token.item() - PITCH_OFF
        else: # it's a keyboard note
            pitch = note
            buffer[...,curr_i+2] = torch.tensor(note + PITCH_OFF, dtype=torch.long, device=device)
        playNote(pitch, velocity) 

        # update velocity
        buffer[...,curr_i+3] =  torch.tensor(velocity + VEL_OFF, dtype=torch.long, device=device)
    
        # add (user_note, pitch, time) to dictionary
        noteOn_dict[note] = (pitch+PITCH_OFF, timeNew)
        
        curr_i += 4

    else: # NOTE_OFF
        # check noteoffs in noteOn list
        notes_to_update = []
        if note in noteOn_dict:
            # get pitch and time in dictionary of accumulated notesOns without noteOff
            pitch, noteOn_time = noteOn_dict[note]
            # Calculate actual duration
            duration = dur2quant(timeNew - noteOn_time)
            duration = max(1, min(126, duration)) # maximum duration 126
            # Update the duration in the buffer
            last_pos_pitch = find_pos_in_buffer(buffer, pitch)
            if last_pos_pitch > 0:
                duration_tensor = torch.tensor([duration + DUR_OFF], dtype=torch.long, device=device) 
                buffer[...,last_pos_pitch-1] = duration_tensor # last duration is last_pitch-1

                # Mark this note for removal from dictionary
                del noteOn_dict[note]
                playNote(pitch-PITCH_OFF, 0)
        
    notes_to_update = []
    # Update duration of notes that have been held for too long
    for button, (pitch, noteOn_time) in list(noteOn_dict.items()):
        # Update duration in buffer
        last_pos_pitch = find_pos_in_buffer(buffer, pitch)
        if last_pos_pitch > 0:
            duration = dur2quant(timeNew - noteOn_time)
            duration = max(1, min(126, duration)) # maximum duration 126
            if (duration == 126): # 2520 = 126*20 max duration
                # Mark this note for removal from dictionary
                notes_to_update.append(button)
            duration_tensor =  torch.tensor([126 + DUR_OFF], dtype=torch.long, device=device) 
            buffer[...,last_pos_pitch-1] = duration_tensor # last duration is last_pitch-1

    # Remove updated notes from dictionary
    for button in notes_to_update:
        del noteOn_dict[button]
        playNote(pitch-PITCH_OFF, 0)
    
    print(buffer[...,curr_i-8:curr_i+4])


"""# UPDATE DURATION FUNCTION """

def find_pos_in_buffer(input_buffer, pitch):
    """Helper function to find the last occurrence of pitch"""
    pos = -1
    # [0] input_buffer shape (1, 1024), get the 1st batch
    buffer_list = input_buffer[0].tolist()
    for i in reversed(range(len(buffer_list))):
        if buffer_list[i] == pitch: 
            pos = i
            break
    return pos


"""# TESTING manageNote """
'''
for i in range(20):
  manageNote(1, 100)
  time.sleep(1)
  manageNote(1, 100)
  time.sleep(1)
  manageNote(1, 100)
  time.sleep(1)
  manageNote(1, 100)
  time.sleep(1)'''

"""# MIDI IN """

try:
   # Create MIDI input object
    if torch.backends.mps.is_available(): 
        midiin = rtmidi.MidiIn(rtmidi.API_MACOSX_CORE)  #  for Mac
        MIDI_PORT = 0 #  Axiom 0 for Mac
    else:
        midiin = rtmidi.MidiIn(rtmidi.API_LINUX_ALSA)  #  for Linux
        MIDI_PORT = 1 # minicontrol32 in linux

    
    # List available ports
    available_ports = midiin.get_ports()
    
    if available_ports:
        print("Available MIDI input ports:")
        for i, port in enumerate(available_ports):
            print(f"[{i}] {port}")
        # Open first available port
        midiin.open_port(MIDI_PORT) 
      
        print(f"Using MIDI input port: {available_ports[MIDI_PORT]}")

    midiin.set_callback(midiin_callback)

    while True:
      time.sleep(0.0001)
except (EOFError, KeyboardInterrupt):
    print("Bye.")

