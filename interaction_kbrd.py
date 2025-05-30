''' Interaction, generating buttons from laptop keyboard, 
starting with a context extracted from a MIDI file '''

import time
import sys
import os
# Try to import fluidsynth with better error handling
try:
    import fluidsynth
except ImportError as e:
    print(f"FluidSynth import error: {e}")
    print("Trying to locate FluidSynth library...")
    # Try common paths where FluidSynth might be installed
    import sys
    import os
    import platform
    
    if platform.system() == "Darwin":  # macOS
        # Common FluidSynth paths on macOS
        possible_paths = [
            "/opt/homebrew/lib",
            "/usr/local/lib", 
            "/Library/Frameworks",
            "/System/Library/Frameworks"
        ]
        for path in possible_paths:
            if path not in os.environ.get("DYLD_LIBRARY_PATH", ""):
                os.environ["DYLD_LIBRARY_PATH"] = os.environ.get("DYLD_LIBRARY_PATH", "") + ":" + path
    
    # Try importing again
    try:
        import fluidsynth
        print("FluidSynth successfully imported after setting library paths")
    except ImportError:
        print("Still cannot import FluidSynth. Please check your FluidSynth installation.")
        print("On macOS, try: brew install fluidsynth")
        sys.exit(1) 
# pip install pyfluidsynth
from typing import Optional, List
import keyboard  # pip install keyboard
from threading import Lock

from visualizer import Visualizer

# Import Monster Piano Transformer as mpt
from model_loader import load_model
from midi_processors import midi_to_tokens, tokens_to_midi
#from monsterpianotransformer import generate
import torch
import TMIDIX
from params import NUM_BUTTONS

TRACES = False

''' DEVICE '''
#device = torch.device('cpu')
device = torch.device('mps') 

''' MODEL '''
model = load_model(model_name='no_dtime_good_reference', device='cpu')
model.to(device)
model.eval()

''' PARAMS '''
# Get sample seed MIDI path
#sample_midi_path = './seed_midis/Monster-Piano-Transformer-Piano-Seed-3.mid'
sample_midi_path = './samples/clairTester_to_end.midi'
output_midi_name = './out/interactive_performance'

CTX_LEN = 120 # num notes in context. tokens = CTX_LENGTH * 3
TOTAL_GEN_LEN = 1024 # num notes to generate

'''THREADING'''
# Add these at the global scope after your imports
buffer_lock = Lock()
save_lock = Lock()

'''VISUALIZER'''
visualizer = Visualizer()

'''KEYBOARD MAPPING'''
# Map keyboard scan codes to button numbers (0-11)
# Using scan codes because key names are not reliable on macOS
# Scan codes for number row: 1 2 3 4 5 6 7 8 9 0 - =
SCANCODE_TO_BUTTON = {
    18: 0,   # 1
    19: 1,   # 2
    20: 2,   # 3
    21: 3,   # 4
    23: 4,   # 5
    22: 5,   # 6
    26: 6,   # 7
    28: 7,   # 8
    25: 8,   # 9
    29: 9,   # 0
    27: 10,  # - (dash/minus)
    24: 11,  # = (equals)
}

# Fallback mapping for key names (if they work)
KEY_TO_BUTTON = {
    '1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5,
    '7': 6, '8': 7, '9': 8, '0': 9, "-": 10, "=": 11
}

# Track which keys are currently pressed to avoid key repeat
pressed_keys = set()

'''KEYBOARD EVENT HANDLERS'''
def on_key_event(event):
    """Handle keyboard press/release events"""
    global pressed_keys
    
    if TRACES:
        print(f"Key event: scan_code={event.scan_code}, name='{event.name}', event_type={event.event_type}")
    
    # First try scan code mapping
    button_num = None
    key_id = None
    
    if event.scan_code in SCANCODE_TO_BUTTON:
        button_num = SCANCODE_TO_BUTTON[event.scan_code]
        key_id = f"scan_{event.scan_code}"
    elif event.name and event.name in KEY_TO_BUTTON:
        button_num = KEY_TO_BUTTON[event.name]
        key_id = f"name_{event.name}"
    
    # Handle special keys (by name if available, otherwise by scan code)
    if button_num is None:
        # ESC key scan code is usually 53 on macOS
        if event.scan_code == 53 and event.event_type == keyboard.KEY_DOWN:
            print("ESC pressed - exiting...")
            os._exit(0)
        # S key scan code is usually 1 on macOS  
        elif event.scan_code == 1 and event.event_type == keyboard.KEY_DOWN:
            with save_lock:
                print("S pressed - saving performance")
                save_performance()
                os._exit(1)
        # R key scan code is usually 15 on macOS
        elif event.scan_code == 15 and event.event_type == keyboard.KEY_DOWN:
            with save_lock:
                print("R pressed - resetting context")
                reset_context()
        # Also try by name for special keys
        elif event.name == 'esc' and event.event_type == keyboard.KEY_DOWN:
            print("ESC pressed - exiting...")
            os._exit(0)
        elif event.name == 's' and event.event_type == keyboard.KEY_DOWN:
            with save_lock:
                print("S pressed - saving performance")
                save_performance()
                os._exit(1)
        elif event.name == 'r' and event.event_type == keyboard.KEY_DOWN:
            with save_lock:
                print("R pressed - resetting context")
                reset_context()
        return
    
    if event.event_type == keyboard.KEY_DOWN:
        # Avoid key repeat - only process if key wasn't already pressed
        if key_id not in pressed_keys:
            pressed_keys.add(key_id)
            velocity = 127  # noteon
            if TRACES:
                print(f"Button {button_num} pressed (scan_code: {event.scan_code})")
            with buffer_lock:
                manageNote(button_num, velocity)
                
    elif event.event_type == keyboard.KEY_UP:
        # Key released
        if key_id in pressed_keys:
            pressed_keys.remove(key_id)
            velocity = 0  # noteoff
            if TRACES:
                print(f"Button {button_num} released (scan_code: {event.scan_code})")
            with buffer_lock:
                manageNote(button_num, velocity)

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
  
  button_num = button_num # keep button numbers 0-11 as received
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


"""# KEYBOARD INPUT """

print("Keyboard interaction started!")
print("Key mapping (number row):")
print("  1 2 3 4 5 6 7 8 9 0 - =  ->  Buttons 0-11")
print("Special keys:")
print("  S: Save performance")
print("  R: Reset context")
print("  ESC: Exit")
print("NOTE: Using scan codes for key detection on macOS")
print("-" * 50)

# Set up keyboard hook
keyboard.hook(on_key_event)

try:
    while True:
        time.sleep(0.0001)
        #visualizer.get_note(60, 100)
        #visualizer.get_button(0, 100)
        visualizer.draw()
except (EOFError, KeyboardInterrupt):
    print("Bye.")
    keyboard.unhook_all()  # Clean up keyboard hooks 