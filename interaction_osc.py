''' Interaction, generating buttons from MIDI keyboard, 
starting with a context extracted from a MIDI file '''

import time
import sys
import fluidsynth
import os 
# pip install pyfluidsynth
from typing import Optional, List
# OSC imports instead of MIDI
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
# pip install python-osc
from threading import Lock

from visualizer import Visualizer

# Import Monster Piano Transformer as mpt
from model_loader import load_model
from midi_processors import midi_to_tokens, tokens_to_midi
#from monsterpianotransformer import generate
import torch
import TMIDIX
from params import NUM_BUTTONS

TRACES = True

''' DEVICE '''
#device = torch.device('cpu')
device = torch.device('mps') 

''' MODEL '''
model = load_model(model_name='no_dtime', device='cpu')
model.to(device)
model.eval()

''' PARAMS '''
# Get sample seed MIDI path
#sample_midi_path = './seed_midis/Monster-Piano-Transformer-Piano-Seed-3.mid'
sample_midi_path = './samples/clairTester_to_end.midi'
output_midi_name = './out/interactive_performance'

CTX_LEN = 256 # num notes in context. tokens = CTX_LENGTH * 3
TOTAL_GEN_LEN = 1024 # num notes to generate

'''THREADING'''
# Add these at the global scope after your imports
buffer_lock = Lock()
save_lock = Lock()

'''VISUALIZER'''
visualizer = Visualizer()

'''OSC MESSAGE HANDLERS'''
def osc_button_handler(unused_addr, button_num, state):
    """Handle OSC messages with button number and state (1=noteon, 0=noteoff)"""
    # Convert state to velocity: 1 -> 127, 0 -> 0
    velocity = 127 if state == 1 else 0
    
    if TRACES:
        print(f"OSC button: {button_num}, state: {state}, velocity: {velocity}")
    
    with buffer_lock:  # lock to avoid race condition
        manageNote(button_num, velocity)

def osc_save_handler(unused_addr):
    """Handle OSC save message"""
    with save_lock:
        print("saving performance")
        save_performance()
        os._exit(1)

def osc_reset_handler(unused_addr):
    """Handle OSC reset message"""
    with save_lock:
        print("resetting context")
        reset_context()


"""# FLUIDSYNTH INIT """
fs = fluidsynth.Synth()
fs.start()
sfid = fs.sfload("./piano.sf2")
fs.program_select(0, sfid, 0, 0)

def playNote(note, velocity=100):
    if velocity > 0:
        fs.noteon(0, note, velocity)
    else:
        fs.noteoff(0, note) 

def save_performance():
  global dict_output_tokens
  global i

  context = {
      'dtime': dict_output_tokens['dtime'][:i+CTX_LEN+1],
      'pitch': dict_output_tokens['pitch'][:i+CTX_LEN+1],
      'dur': dict_output_tokens['dur'][:i+CTX_LEN+1],
    }

  # generate a midi file from generated pitches
  song_d = TMIDIX.dict_to_song(context)

  detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_d, output_file_name = output_midi_name,
                                                            timings_multiplier=2
                                                            )
  print("saved performance")

def reset_context():
    global i
    global dict_output_tokens, dict_input_tokens

    i = 0
    dict_output_tokens['dtime'] = dict_input_tokens['dtime'] 
    dict_output_tokens['pitch'] = dict_input_tokens['pitch'] 
    dict_output_tokens['dur'] = dict_input_tokens['dur'] 
    dict_output_tokens['button'] = dict_input_tokens['button'] 

''' VARIABLES '''
context = None
timeLast = 0
i = 0 # num current tokens in context after CTX_LEN
noteOn_dict = {}
first_note = True

''' BUILD CTX '''
# Load seed MIDI
input_tokens = midi_to_tokens(sample_midi_path) # tokens, without vel

output_tokens = input_tokens.copy()

dict_input_tokens, num_notes = TMIDIX.midi_tokens_to_dict(input_tokens) # vel already filtered out
dict_output_tokens, num_notes = TMIDIX.midi_tokens_to_dict(output_tokens) # vel already filtered out

if TRACES:  
    print("num_notes", num_notes)
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
    b = b.clone().detach().tolist()

visualizer.primer(dict_input_tokens['pitch'][:CTX_LEN], dict_input_tokens['dtime'][:CTX_LEN ], b[:CTX_LEN])

def manageNote(button_num, velocity): 
  global context  # Access the global context
  global timeLast # time of last note, global variable
  global b # button array
  global i # num current tokens in context after CTX_LEN
  global dict_output_tokens # output tokens
  global noteOn_dict # button: (pitch, timeIn)
  global first_note
  global visualizer
  
  button_num = button_num -1 # [0-11]
  if TRACES:
    print("button", button_num)

  timeNew = time.perf_counter()*1000 /32 # in miliseconds /32 as in midi_to_tokens()

  if velocity > 0: # noteOn
    # Update position token
    dtime = max(0, min(127, int(timeNew) - int(timeLast))) # time difference from previous events, but trunk to maximum 127
    if first_note:
        dtime = 0
        first_note = False

    timeLast = timeNew
    dict_output_tokens['dtime'][i+CTX_LEN] = dtime
    # Use button number directly
    but = button_num
    #b[i+CTX_LEN] = but
    
    context = {
      'dtime': torch.tensor(dict_output_tokens['dtime'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'pitch': torch.tensor(dict_output_tokens['pitch'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'dur': torch.tensor(dict_output_tokens['dur'][i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0),
      'button': torch.tensor(b[i:i+CTX_LEN+1], dtype=torch.long).unsqueeze(0)
    }
    context = TMIDIX.to_device(context, device)
                 
    with torch.inference_mode():
        new_pitch_token = model.gen_pitch_token(context)
    dict_output_tokens['pitch'][i+CTX_LEN] = new_pitch_token

    playNote(new_pitch_token, velocity) 
    visualizer.get_note(new_pitch_token, velocity)
    visualizer.get_button(but, velocity)

    # add (user_note, pitch, time) to dictionary
    noteOn_dict[but] = (new_pitch_token, timeNew)
    i += 1

  else: # noteOff
    if TRACES:
        print("noteOff", button_num)
    but = button_num
    #print("but", but)
    if but in noteOn_dict:
      if TRACES:
        print("in_Noteon_dict")
      # get pitch and time in dictionary of accumulated notesOns without noteOff
      pitch, noteOn_time = noteOn_dict[but]
      playNote(pitch, 0)
      visualizer.get_note(pitch, 0)
      visualizer.get_button(but, 0)
      #visualizer.update(noteOn_time)


"""# OSC SERVER """

try:
    # Create OSC dispatcher and server
    dispatcher = Dispatcher()
    
    # Map OSC addresses to handler functions
    # Expected message format: /button <button_number> <state>
    # where state is 1 for noteon, 0 for noteoff
    dispatcher.map("/button", osc_button_handler)
    dispatcher.map("/save", osc_save_handler)
    dispatcher.map("/reset", osc_reset_handler)
    
    # Create OSC server
    # Use "0.0.0.0" to listen on all network interfaces (accept from any IP)
    # Use "127.0.0.1" to only accept from localhost
    OSC_IP = "0.0.0.0"  # Listen on all interfaces
    OSC_PORT = 3003
    
    server = osc_server.ThreadingOSCUDPServer((OSC_IP, OSC_PORT), dispatcher)
    
    print(f"OSC Server listening on {OSC_IP}:{OSC_PORT}")
    print("Expected message format:")
    print("  /button <button_number> <state>  (state: 1=noteon, 0=noteoff)")
    print("  /save                            (save performance)")
    print("  /reset                           (reset context)")
    
    # Start server in a separate thread
    import threading
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    # Main loop for visualization
    while True:
        time.sleep(0.0001)
        #visualizer.get_note(60, 100)
        #visualizer.get_button(0, 100)
        visualizer.draw()
        
except (EOFError, KeyboardInterrupt):
    print("Bye.")
    if 'server' in locals():
        server.shutdown()

