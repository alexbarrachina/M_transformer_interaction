''' Interaction, generating buttons from MIDI keyboard, 
starting with no context, generating from scratch '''

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
import threading
from threading import Lock

# Import Monster Piano Transformer as mpt
from model_loader import load_model
from midi_processors import midi_to_tokens, tokens_to_midi
#from monsterpianotransformer import generate
import torch
import TMIDIX

''' DEVICE '''
#device = torch.device('cpu')
device = torch.device('mps') 

''' MODEL '''
model = load_model(model_name='full_apr7_autoencoder', device='cpu')
model.to(device)
model.eval()

''' PARAMS '''
# Get sample seed MIDI path
#sample_midi_path = './seed_midis/Monster-Piano-Transformer-Piano-Seed-3.mid'
sample_midi_path = './samples/clairTester_to_end.midi'
output_midi_name = './out/continuator_clairTester_to_end'
output_butt_midi_name = './out/continuator_clairTester_buttons'
CTX_LEN = 120 # num notes in context. tokens = CTX_LENGTH * 3
TOTAL_GEN_LEN = 500 # num notes to generate

'''THREADING'''
# Add these at the global scope after your imports
buffer_lock = Lock()
save_lock = Lock()

'''MIDI IN CALLBACK'''
def midiin_callback(event, data=None):
    message, deltatime = event

    #print("message",message[0]  & 0xF0)
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
            print("saving performance")
            #save_performance()
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

''' GLOBAL VARIABLES '''
context = None
timeLast = 0

''' BUILD CTX '''
# Load seed MIDI
input_tokens = midi_to_tokens(sample_midi_path) # tokens, without vel

output_tokens = input_tokens.copy()

dict_input_tokens, num_notes = TMIDIX.midi_tokens_to_dict(input_tokens) # vel already filtered out
dict_output_tokens, num_notes = TMIDIX.midi_tokens_to_dict(output_tokens) # vel already filtered out

# Build context tokens
context = {
    'dtime': torch.tensor(dict_input_tokens['dtime'], dtype=torch.long).unsqueeze(0),
    'pitch': torch.tensor(dict_input_tokens['pitch'], dtype=torch.long).unsqueeze(0),
    'dur': torch.tensor(dict_input_tokens['dur'], dtype=torch.long).unsqueeze(0)
          }
context = TMIDIX.to_device(context, device)
  
with torch.inference_mode():
    e = model.encoder(context) # encoder output (batch, seq_len)
    b = model.real_to_discrete(e).squeeze(0) # generate buttons (batch, seq_len)
    e = e.squeeze(0)

def manageNote(note, velocity): 
  global context  # Access the global context
  global timeLast # time of last note, global variable
  #global noteOn_dict # button: (pitch, timeIn)

  print("manageNote", note, velocity)
  timeNew = time.perf_counter()*1000

  if velocity > 0: # noteOn
    # time, dur+128, ptc+256, vel+384]

    # Update position token
    #position = timeNew - timeLast * 1000 # in miliseconds, promig#if (i%10==0):
    dtimeNew = max(0, min(126, int(timeNew/10) - int(timeLast/10))) # time difference from previous events, but trunk to maximum 126
    timeLast = timeNew
    position_tensor = torch.tensor([position], dtype=torch.long)
    context['dtime'] = torch.cat([context['dtime'][1:], position_tensor])











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
    # duration_token = max(0, min(127, int(duration/10)))  # Assuming 10ms per duration unit
    duration_tensor = torch.tensor([duration + 128], dtype=torch.long)  # Add 128 for duration tokens
    
    # Find the last occurrence of this pitch in the buffer
    buffer_list = input_buffer.tolist()
    for i in reversed(range(len(buffer_list))):
        if buffer_list[i] == pitch: 
            # Update the duration token that comes before this pitch
            input_buffer[i-1] = duration_tensor[0]
            break
    return input_buffer


def manageNote(note, velocity): 
  global input_buffer  # Access the global buffer
  global timeLast # time of last note, global variable
  global noteOn_dict # button: (pitch, timeIn)

  print("manageNote", note, velocity)
  timeNew = time.perf_counter()*1000
  
  if velocity > 0: # noteOn
    # time, dur+128, ptc+256, vel+384]

    # Update position token
    #position = timeNew - timeLast * 1000 # in miliseconds, promig#if (i%10==0):
    position = max(0, min(126, int(timeNew/10) - int(timeLast/10))) # time difference from previous events, but trunk to maximum 126
    timeLast = timeNew
    position_tensor = torch.tensor([position], dtype=torch.long)
    save_token()
    input_buffer = torch.cat([input_buffer[1:], position_tensor])

    # Add provisional duration using the last duration in buffer
    duration_tensor = input_buffer[-4].unsqueeze(0)  # Convert to 1D tensor, [-4] is the last duration token
    save_token()
    input_buffer = torch.cat([input_buffer[1:], duration_tensor]) 

    # Update pitch and velocity differently if it's a button or a keyborard note 
    if(note < 20): # it's a button
      # Reshape input_buffer for model input (add batch dimension)
      inp = input_buffer.unsqueeze(0)  # Shape: (1, 512)
      next_token = model.generate_single(inp.to(device), temperature=temperature) 
      pitch = next_token.item()
      print("gen_pitch",pitch)
      playNote(pitch-256, velocity)  # -256 Convert from token to MIDI pitch
      # Update input buffer: remove first token and append new token, circular buffer
      #pitch_tensor = next_token.squeeze().unsqueeze(0)  # Ensure 1D tensor
      save_token()
      pitch_tensor = torch.tensor([pitch], dtype=torch.long)  
      input_buffer = torch.cat([input_buffer[1:], pitch_tensor])
      
      # Update velocity with a fixed value ¿¿¿???? TODO: inference velocity or use keyboard velocity
      #velocity_tensor = input_buffer[-4].unsqueeze(0)  # Convert to 1D tensor, [-4] is the last velocity token
      velocity_tensor = torch.tensor([velocity+384], dtype=torch.long)  
      save_token()
      input_buffer = torch.cat([input_buffer[1:], velocity_tensor]) 
    else: # it's a keyboard note
      pitch = note
      save_token()
      note_tensor = torch.tensor([note + 256], dtype=torch.long)
      input_buffer = torch.cat([input_buffer[1:], note_tensor])
      save_token()

      velocity_tensor = torch.tensor([velocity+384], dtype=torch.long)
      input_buffer = torch.cat([input_buffer[1:], velocity_tensor]) 
      playNote(note, velocity)
   
    # add (user_note, pitch, time) to dictionary
    noteOn_dict[note] = (pitch+256, timeNew)

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
      playNote(pitch-256, 0)
    
    # Update duration of notes that have been held for too long
    for button, (pitch, noteOn_time) in list(noteOn_dict.items()):
      if (timeNew - noteOn_time)  > 2520: # 2520 = 126*20 max duration
        # Update duration in buffer
        input_buffer = update_duration_in_buffer(input_buffer, pitch, 126) # force duration to max
        # Mark this note for removal from dictionary
        notes_to_update.append(button)
        playNote(pitch-256, 0)

    # Remove updated notes from dictionary
    for button in notes_to_update:
        del noteOn_dict[button]
        
  #print(noteOn_dict)


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
