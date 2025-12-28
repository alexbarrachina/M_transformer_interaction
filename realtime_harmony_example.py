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
import argparse
from typing import List, Tuple
from harmony_extractor import RealtimeHarmonyExtractor


def simulate_from_midi_file(
    midi_path: str,
    playback_speed: float = 1.0,
    update_interval_sec: float = 0.1,
    visualize: bool = True
):
    """
    Simulate real-time processing from a MIDI file.
    
    Args:
        midi_path: Path to MIDI file
        playback_speed: Speed multiplier (1.0 = normal, 2.0 = 2x speed)
        update_interval_sec: How often to update visualization (seconds)
        visualize: Enable real-time visualization
    """
    import pretty_midi
    
    print(f"Loading MIDI file: {midi_path}")
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI: {e}")
        return
    
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
    print("Starting real-time simulation...\n")
    
    # Create extractor
    extractor = RealtimeHarmonyExtractor(
        key_window_sec=4.0,
        tension_window_sec=1.0,
        key_update_interval_sec=2.0,
        visualize=visualize
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


def process_live_midi(device_id: int = 0, visualize: bool = True):
    """
    Process live MIDI input from a device.
    
    Args:
        device_id: MIDI input device ID
        visualize: Enable real-time visualization
    """
    try:
        import pygame.midi
    except ImportError:
        print("Error: pygame not installed. Install with: pip install pygame")
        return
    
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
    print("Press Ctrl+C to stop.\n")
    
    # Create extractor
    extractor = RealtimeHarmonyExtractor(
        key_window_sec=4.0,
        tension_window_sec=1.0,
        key_update_interval_sec=2.0,
        visualize=visualize
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
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        '--device', type=int, default=0,
        help='MIDI input device ID (for --live mode)'
    )
    
    parser.add_argument(
        '--speed', type=float, default=1.0,
        help='Playback speed multiplier (for --file mode)'
    )
    
    parser.add_argument(
        '--no-viz', action='store_true',
        help='Disable visualization'
    )
    
    args = parser.parse_args()
    
    if args.live:
        # Live MIDI input mode
        process_live_midi(device_id=args.device, visualize=not args.no_viz)
    
    elif args.file:
        # File simulation mode
        simulate_from_midi_file(
            midi_path=args.file,
            playback_speed=args.speed,
            visualize=not args.no_viz
        )
    
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python realtime_harmony_example.py --file samples/test.mid")
        print("  python realtime_harmony_example.py --file samples/test.mid --speed 2.0")
        print("  python realtime_harmony_example.py --live --device 0")


if __name__ == "__main__":
    main()

