#!/usr/bin/env python3
"""
Test script for real-time harmony extraction.
Runs basic tests to verify the implementation.
"""

import sys
import numpy as np
from harmony_extractor import RealtimeHarmonyExtractor, notes_to_chroma


def test_basic_functionality():
    """Test basic initialization and functionality."""
    print("="*60)
    print("Test 1: Basic Functionality")
    print("="*60)
    
    # Test C major (key=0)
    print("\n1.1: Creating extractor for C major (key=0)...")
    extractor = RealtimeHarmonyExtractor(global_key=0, visualize=False)
    assert extractor.key_root == 0
    assert extractor.key_mode == 'major'
    print("  ✓ C major extractor created")
    
    # Test A minor (key=21: 12 + 9)
    print("\n1.2: Creating extractor for A minor (key=21)...")
    extractor_am = RealtimeHarmonyExtractor(global_key=21, visualize=False)
    assert extractor_am.key_root == 9  # A = 9
    assert extractor_am.key_mode == 'minor'
    print("  ✓ A minor extractor created")
    
    # Test adding notes
    print("\n1.3: Adding notes (C major triad)...")
    extractor.add_note(pitch=60, start_time=0.0, velocity=80)  # C
    extractor.add_note(pitch=64, start_time=0.0, velocity=70)  # E
    extractor.add_note(pitch=67, start_time=0.0, velocity=70)  # G
    assert len(extractor.notes) == 3
    print(f"  ✓ Added 3 notes, buffer size: {len(extractor.notes)}")
    
    # Test extracting features
    print("\n1.4: Extracting features...")
    cx, cy, mag, key_root, key_mode = extractor.extract_current_features(0.1)
    assert key_root == 0
    assert key_mode == 'major'
    print(f"  ✓ Tension: ({cx:.2f}, {cy:.2f}) magnitude={mag:.2f}")
    print(f"  ✓ Key: {key_root} {key_mode}")
    
    # Test note off
    print("\n1.5: Note off...")
    extractor.note_off(pitch=60, end_time=0.5)
    extractor.note_off(pitch=64, end_time=0.5)
    extractor.note_off(pitch=67, end_time=0.5)
    assert all(n['end'] is not None for n in extractor.notes)
    print(f"  ✓ Notes off, buffer size: {len(extractor.notes)}")
    
    # Test update
    print("\n1.6: Update...")
    extractor.update(current_time=0.5)
    print("  ✓ Update complete")
    
    # Test history
    print("\n1.7: Get history...")
    tension_xy, tension_mag, final_key = extractor.get_history()
    assert len(tension_xy) > 0
    assert final_key == (0, 'major')
    print(f"  ✓ History length: {len(tension_xy)} frames")
    print(f"  ✓ Final key: {final_key}")
    
    extractor.close()
    print("\n✅ Test 1 passed!\n")


def test_key_encoding():
    """Test key encoding for all 24 keys."""
    print("="*60)
    print("Test 2: Key Encoding (All 24 Keys)")
    print("="*60)
    
    pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Test major keys (0-11)
    print("\nMajor keys (0-11):")
    for i in range(12):
        extractor = RealtimeHarmonyExtractor(global_key=i, visualize=False)
        assert extractor.key_root == i
        assert extractor.key_mode == 'major'
        print(f"  key={i:2d} -> {pitch_names[i]} major ✓")
        extractor.close()
    
    # Test minor keys (12-23)
    print("\nMinor keys (12-23):")
    for i in range(12, 24):
        extractor = RealtimeHarmonyExtractor(global_key=i, visualize=False)
        root = i - 12
        assert extractor.key_root == root
        assert extractor.key_mode == 'minor'
        print(f"  key={i:2d} -> {pitch_names[root]} minor ✓")
        extractor.close()
    
    print("\n✅ Test 2 passed!\n")


def test_notes_to_chroma():
    """Test the notes_to_chroma helper function."""
    print("="*60)
    print("Test 3: notes_to_chroma Helper")
    print("="*60)
    
    # Test with C major triad
    print("\n3.1: C major triad...")
    notes = [
        {'pitch': 60, 'start': 0.0, 'end': 1.0, 'velocity': 80},  # C
        {'pitch': 64, 'start': 0.0, 'end': 1.0, 'velocity': 70},  # E
        {'pitch': 67, 'start': 0.0, 'end': 1.0, 'velocity': 70},  # G
    ]
    chroma = notes_to_chroma(notes, current_time=0.5, window_sec=2.0)
    assert chroma.shape == (12,)
    assert chroma[0] > 0  # C
    assert chroma[4] > 0  # E
    assert chroma[7] > 0  # G
    active_pcs = np.where(chroma > 0)[0].tolist()
    print(f"  ✓ Chroma vector shape: {chroma.shape}")
    print(f"  ✓ Active pitch classes: {active_pcs}")
    
    # Test with active notes (end=None)
    print("\n3.2: Active notes (end=None)...")
    notes_active = [
        {'pitch': 60, 'start': 0.0, 'end': None, 'velocity': 80},  # C (still playing)
        {'pitch': 64, 'start': 0.0, 'end': 0.3, 'velocity': 70},   # E (finished)
    ]
    chroma = notes_to_chroma(notes_active, current_time=0.5, window_sec=2.0)
    assert chroma[0] > 0  # C should be active
    assert chroma[4] > 0  # E should have some weight
    print(f"  ✓ Active note (C) present: {chroma[0] > 0}")
    print(f"  ✓ Recent note (E) present: {chroma[4] > 0}")
    
    print("\n✅ Test 3 passed!\n")


def test_buffer_cleaning():
    """Test buffer cleaning logic."""
    print("="*60)
    print("Test 4: Buffer Cleaning")
    print("="*60)
    
    extractor = RealtimeHarmonyExtractor(
        global_key=0,
        buffer_size_sec=2.0,  # Small buffer for testing
        visualize=False
    )
    
    # Add some notes
    print("\n4.1: Adding notes at different times...")
    extractor.add_note(pitch=60, start_time=0.0, velocity=80)
    extractor.note_off(pitch=60, end_time=0.5)
    
    extractor.add_note(pitch=62, start_time=1.0, velocity=80)
    extractor.note_off(pitch=62, end_time=1.5)
    
    extractor.add_note(pitch=64, start_time=2.0, velocity=80)
    extractor.note_off(pitch=64, end_time=2.5)
    
    assert len(extractor.notes) == 3
    print(f"  ✓ Buffer has {len(extractor.notes)} notes")
    
    # Clean buffer at time=3.0 (buffer_size_sec=2.0)
    # Should remove notes that ended before time=1.0
    print("\n4.2: Cleaning buffer at time=3.0...")
    extractor._clean_buffer(current_time=3.0)
    # First note ended at 0.5, should be removed (cutoff=1.0)
    # Second and third notes should remain
    assert len(extractor.notes) == 2
    print(f"  ✓ Buffer cleaned, now has {len(extractor.notes)} notes")
    
    extractor.close()
    print("\n✅ Test 4 passed!\n")


def test_chord_extraction():
    """Test chord shape extraction."""
    print("="*60)
    print("Test 5: Chord Shape Extraction")
    print("="*60)
    
    extractor = RealtimeHarmonyExtractor(global_key=0, visualize=False)
    
    # Add C major triad
    print("\n5.1: C major triad (C-E-G)...")
    extractor.add_note(pitch=60, start_time=0.0, velocity=80)  # C
    extractor.add_note(pitch=64, start_time=0.0, velocity=70)  # E
    extractor.add_note(pitch=67, start_time=0.0, velocity=70)  # G
    
    active_pcs = extractor._extract_active_pitch_classes(0.1)
    assert len(active_pcs) >= 3
    assert 0 in active_pcs  # C
    assert 4 in active_pcs  # E
    assert 7 in active_pcs  # G
    print(f"  ✓ Extracted {len(active_pcs)} active pitch classes: {active_pcs}")
    
    # Add a passing tone (shouldn't be as prominent)
    print("\n5.2: Adding passing tone (D)...")
    extractor.add_note(pitch=62, start_time=0.1, velocity=40)  # D (weaker)
    extractor.note_off(pitch=62, end_time=0.15)  # Short duration
    
    active_pcs = extractor._extract_active_pitch_classes(0.2)
    # C, E, G should still be the main notes
    assert 0 in active_pcs  # C
    assert 4 in active_pcs  # E
    assert 7 in active_pcs  # G
    print(f"  ✓ Main chord notes still dominant: {active_pcs}")
    
    extractor.close()
    print("\n✅ Test 5 passed!\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Real-Time Harmony Extraction - Test Suite")
    print("="*60 + "\n")
    
    try:
        test_basic_functionality()
        test_key_encoding()
        test_notes_to_chroma()
        test_buffer_cleaning()
        test_chord_extraction()
        
        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        return 0
    
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

