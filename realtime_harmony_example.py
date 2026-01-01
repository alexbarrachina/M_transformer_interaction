#!/usr/bin/env python3
"""
Real-Time Harmony Extraction Example

This script demonstrates how to use the RealtimeHarmonyExtractor class
to process live MIDI input or simulate real-time processing from a file.

Usage:
    # Simulate real-time from MIDI file (visualization enabled)
    python realtime_harmony_example.py --file samples/test.mid
    
    # Process MIDI file at 2x speed
    python realtime_harmony_example.py --file samples/test.mid --speed 2.0
    
    # Use with live MIDI input (requires pygame.midi)
    python realtime_harmony_example.py --live --device 0
"""

import sys
import time
import fluidsynth
import argparse
from typing import List, Tuple
from harmony_extractor import RealtimeHarmonyExtractor
from threading import Lock

TRACES = False

'''THREADING'''
# Add these at the global scope after your imports
buffer_lock = Lock()
save_lock = Lock()

"""# FLUIDSYNTH INIT """
fs = fluidsynth.Synth()
fs.start()
sfid = fs.sfload("./piano.sf2")
fs.program_select(0, sfid, 0, 0)

def playNote(note, velocity=100):
    if TRACES:
        print("fluidNote", note, velocity)
    if velocity > 0:
        fs.noteon(0, note, velocity)
    else:
        fs.noteoff(0, note) 


def simulate_from_midi_file(
    midi_path: str,
    global_key: int = -1,
    playback_speed: float = 1.0,
    update_interval_sec: float = 0.1,
    visualize: bool = True,
    chord_threshold: float = 0.3
):
    """
    Simulate real-time processing from a MIDI file.
    
    Args:
        midi_path: Path to MIDI file
        global_key: Fixed global key (0-11 for major, 12-23 for minor, -1 for auto-detect)
        playback_speed: Speed multiplier (1.0 = normal, 2.0 = 2x speed)
        update_interval_sec: How often to update visualization (seconds)
        visualize: Enable real-time visualization
        chord_threshold: Threshold ratio for chord detection (0.0-1.0)
    """
    import pretty_midi
    from harmony_extractor import estimate_key, format_key_name
    import numpy as np
    
    print(f"Loading MIDI file: {midi_path}")
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI: {e}")
        return
    
    # Auto-detect key if global_key is -1
    if global_key == -1:
        print("Auto-detecting key from MIDI file...")
        chroma_matrix = pm.get_chroma(fs=10)
        global_chroma = np.sum(chroma_matrix, axis=1)
        detected_key = estimate_key(global_chroma)
        if detected_key is None:
            detected_key = (0, 'major')
        key_root, key_mode = detected_key
        global_key = key_root if key_mode == 'major' else key_root + 12
        print(f"Detected key: {format_key_name(key_root, key_mode)} (encoded as {global_key})")
    
    # Collect all note events
    events: List[Tuple[float, str, int, int]] = []  # (time, type, pitch, velocity)
    
    for instrument in pm.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                events.append((note.start, 'on', note.pitch, note.velocity))
                events.append((note.end, 'off', note.pitch, 0))
    
    # Sort by time
    events.sort(key=lambda x: x[0])
    
    if not events:
        print("No notes found in MIDI file!")
        return
    
    total_duration = events[-1][0]
    print(f"Found {len(events)} note events")
    print(f"Duration: {total_duration:.1f} seconds")
    print(f"Playback speed: {playback_speed}x")
    print(f"Using global key: {global_key} ({format_key_name(global_key % 12, 'major' if global_key < 12 else 'minor')})")
    print("Starting real-time simulation...\n")
    
    # Create extractor with fixed global key
    extractor = RealtimeHarmonyExtractor(
        global_key=global_key,
        tension_window_sec=1.0,
        visualize=visualize,
        chord_threshold=chord_threshold
    )
    
    # Simulate real-time playback
    start_wall_time = time.time()
    start_midi_time = events[0][0]
    
    event_idx = 0
    last_update = 0.0
    
    try:
        while event_idx < len(events):
            # Current time in MIDI coordinates
            wall_elapsed = (time.time() - start_wall_time) * playback_speed
            current_midi_time = start_midi_time + wall_elapsed
            
            # Process all events up to current time
            while event_idx < len(events) and events[event_idx][0] <= current_midi_time:
                event_time, event_type, pitch, velocity = events[event_idx]
                
                if event_type == 'on':
                    extractor.add_note(pitch, event_time, velocity)
                else:
                    extractor.note_off(pitch, event_time)
                
                event_idx += 1
            
            # Update visualization periodically
            if current_midi_time - last_update >= update_interval_sec:
                extractor.update(current_midi_time)
                last_update = current_midi_time
                
                # Print current state
                cx, cy, mag, key_root, key_mode = extractor.extract_current_features(current_midi_time)
                progress = (current_midi_time - start_midi_time) / (total_duration - start_midi_time) * 100
                print(f"[{progress:5.1f}%] Time: {current_midi_time:6.1f}s | "
                      f"Key: {key_mode[0].upper()}{key_root:2d} | "
                      f"Tension: ({cx:+.2f}, {cy:+.2f}) mag={mag:.2f} | "
                      f"Active notes: {len(extractor.notes):3d}", end='\r')
            
            # Small sleep to avoid busy-waiting
            time.sleep(0.01 / playback_speed)
        
        # Final update
        extractor.update(events[-1][0])
        print("\n\nPlayback complete!")
        
        # Print summary
        tension_xy, tension_mag, final_key = extractor.get_history()
        print(f"\nSummary:")
        print(f"  Final key: {final_key}")
        print(f"  Tension range: [{tension_mag.min():.2f}, {tension_mag.max():.2f}]")
        print(f"  Total frames captured: {len(tension_xy)}")
        
        if visualize:
            print("\nClose the window to exit.")
            import matplotlib.pyplot as plt
            plt.show(block=True)
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        extractor.close()


def process_live_midi(device_id: int = 0, global_key: int = 0, visualize: bool = True, chord_threshold: float = 0.3):
    """
    Process live MIDI input from a device.
    
    Args:
        device_id: MIDI input device ID
        global_key: Fixed global key (0-11 for major, 12-23 for minor)
        visualize: Enable real-time visualization
        chord_threshold: Threshold ratio for chord detection (0.0-1.0)
    """
    try:
        import pygame.midi
    except ImportError:
        print("Error: pygame not installed. Install with: pip install pygame")
        return
    
    from harmony_extractor import format_key_name
    
    print(f"Initializing MIDI input device {device_id}...")
    
    pygame.midi.init()
    
    # List available devices
    print("\nAvailable MIDI devices:")
    for i in range(pygame.midi.get_count()):
        info = pygame.midi.get_device_info(i)
        name = info[1].decode()
        is_input = info[2]
        print(f"  [{i}] {name} {'(INPUT)' if is_input else '(OUTPUT)'}")
    
    # Open device
    try:
        midi_input = pygame.midi.Input(device_id)
    except Exception as e:
        print(f"Error opening device {device_id}: {e}")
        pygame.midi.quit()
        return
    
    print(f"\nListening to device {device_id}...")
    print(f"Using global key: {global_key} ({format_key_name(global_key % 12, 'major' if global_key < 12 else 'minor')})")
    print("Press Ctrl+C to stop.\n")
    
    # Create extractor with fixed global key
    extractor = RealtimeHarmonyExtractor(
        global_key=global_key,
        tension_window_sec=1.0,
        visualize=visualize,
        chord_threshold=chord_threshold
    )
    
    start_time = time.time()
    last_update = 0.0
    update_interval = 0.1
    
    try:
        while True:
            current_time = time.time() - start_time
            
            # Read MIDI events
            if midi_input.poll():
                events = midi_input.read(10)
                for event in events:
                    # event = [[status, data1, data2, data3], timestamp]
                    status = event[0][0]
                    data1 = event[0][1]  # note number or CC number
                    data2 = event[0][2]  # velocity or CC value
                    
                    playNote(data1, data2)
                    # Note On (0x90-0x9F) with velocity > 0
                    if 0x90 <= status <= 0x9F and data2 > 0:
                        extractor.add_note(data1, current_time, data2)
                        print(f"Note ON:  {data1:3d} vel={data2:3d}")
                    
                    # Note Off (0x80-0x8F) or Note On with velocity 0
                    elif (0x80 <= status <= 0x8F) or (0x90 <= status <= 0x9F and data2 == 0):
                        extractor.note_off(data1, current_time)
                        print(f"Note OFF: {data1:3d}")
            
            # Update visualization periodically
            if current_time - last_update >= update_interval:
                extractor.update(current_time)
                last_update = current_time
            
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        midi_input.close()
        pygame.midi.quit()
        extractor.close()


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Harmony Extraction Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Global Key Encoding:
  0-11:  Major keys (C=0, C#=1, D=2, ..., B=11)
  12-23: Minor keys (C=12, C#=13, D=14, ..., B=23)
  -1:    Auto-detect from file (only for --file mode)

Examples:
  # Auto-detect key from file
  python realtime_harmony_example.py --file samples/test.mid

  # Use C major (key=0)
  python realtime_harmony_example.py --file samples/test.mid --key 0

  # Use A minor (key=21)
  python realtime_harmony_example.py --file samples/test.mid --key 21

  # Live MIDI in C major
  python realtime_harmony_example.py --live --device 0 --key 0
        """
    )
    
    parser.add_argument(
        '--file', type=str,
        help='MIDI file to process (simulates real-time)'
    )
    
    parser.add_argument(
        '--live', action='store_true',
        help='Use live MIDI input'
    )
    
    parser.add_argument(
        '--key', type=int, default=-1,
        help='Global key (0-11 major, 12-23 minor, -1 auto-detect). Default: -1'
    )
    
    parser.add_argument(
        '--device', type=int, default=0,
        help='MIDI input device ID (for --live mode). Default: 0'
    )
    
    parser.add_argument(
        '--speed', type=float, default=1.0,
        help='Playback speed multiplier (for --file mode). Default: 1.0'
    )
    
    parser.add_argument(
        '--no-viz', action='store_true',
        help='Disable visualization'
    )
    
    parser.add_argument(
        '--chord-threshold', type=float, default=0.3,
        help='Chord detection threshold (0.0-1.0). Lower = more notes. Default: 0.3'
    )
    
    args = parser.parse_args()
    
    if args.live:
        # Live MIDI input mode
        if args.key == -1:
            print("Warning: --key not specified for live mode, defaulting to C major (0)")
            args.key = 0
        process_live_midi(
            device_id=args.device, 
            global_key=args.key, 
            visualize=not args.no_viz,
            chord_threshold=args.chord_threshold
        )
    
    elif args.file:
        # File simulation mode
        simulate_from_midi_file(
            midi_path=args.file,
            global_key=args.key,
            playback_speed=args.speed,
            visualize=not args.no_viz,
            chord_threshold=args.chord_threshold
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

