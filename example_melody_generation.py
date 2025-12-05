#!/usr/bin/env python3
"""
Example script demonstrating melody generation with AutoregressiveAutoencoder_melody.
Shows how to use arrow guidance for interactive melody creation.
"""

import torch
from x_transformer import Decoder_melody, AutoregressiveAutoencoder_melody
from params import DEFAULT_HPARAMS

def create_model():
    """Create and return a melody autoencoder model."""
    print("Creating Decoder_melody...")
    decoder = Decoder_melody(
        max_seq_len=2048,
        dim=512,  # Smaller for demo
        depth=4,
        heads=4,
        emb_dropout=0.1,
        rotary_pos_emb=True,
        attn_flash=False,  # Set True if you have flash attention
        causal=True
    )
    
    print("Creating AutoregressiveAutoencoder_melody...")
    cfg = DEFAULT_HPARAMS.copy()
    model = AutoregressiveAutoencoder_melody(decoder=decoder, cfg=cfg)
    model.eval()
    
    return model

def generate_melody_example():
    """Example 1: Generate a melody with a specific arrow pattern."""
    print("\n" + "="*60)
    print("Example 1: Generate melody with arrow guidance")
    print("="*60)
    
    model = create_model()
    
    # Start with a simple C major scale prompt
    prompt_pitches = torch.tensor([[60, 62, 64, 65]])  # C, D, E, F
    prompts = {'pitch': prompt_pitches}
    
    # Define arrow sequence for melodic contour
    # Pattern: gradually ascend, stay, then descend
    arrow_names = ['small_up', 'small_up', 'medium_up', 'stay', 
                   'stay', 'small_down', 'medium_down', 'small_down']
    arrow_mapping = {
        'large_down': 0,
        'medium_down': 1,
        'small_down': 2,
        'stay': 3,
        'small_up': 4,
        'medium_up': 5,
        'large_up': 6
    }
    
    arrows = [arrow_mapping[name] for name in arrow_names]
    arrow_sequence = torch.tensor([arrows])
    
    print(f"\nPrompt pitches (MIDI): {prompt_pitches.tolist()[0]}")
    print(f"Arrow sequence: {arrow_names}")
    print(f"Arrow indices: {arrows}")
    
    # Generate
    print("\nGenerating melody...")
    generated = model.generate(
        prompts=prompts,
        seq_len=len(arrows),
        arrow_sequence=arrow_sequence,
        temperature=0.9,
        verbose=False
    )
    
    print(f"\nGenerated pitches (MIDI): {generated.tolist()[0]}")
    
    # Show pitch differences
    full_sequence = torch.cat([prompt_pitches, generated], dim=1)
    diffs = full_sequence[:, 1:] - full_sequence[:, :-1]
    print(f"Pitch differences: {diffs.tolist()[0]}")
    
    return generated

def analyze_melody_example():
    """Example 2: Analyze an existing melody to extract arrows."""
    print("\n" + "="*60)
    print("Example 2: Analyze melody to extract arrows")
    print("="*60)
    
    model = create_model()
    
    # A simple melody: C major arpeggio up and down
    melody = torch.tensor([[60, 64, 67, 72, 67, 64, 60]])
    
    print(f"\nInput melody (MIDI): {melody.tolist()[0]}")
    
    # Extract arrows
    arrows = model.gen_arrows({'pitch': melody})
    
    # Calculate pitch differences for comparison
    diffs = melody[:, 1:] - melody[:, :-1]
    
    print(f"Pitch differences: {diffs.tolist()[0]}")
    print(f"Extracted arrows: {arrows.tolist()[0]}")
    
    # Interpret arrows
    arrow_names = ['large_down', 'medium_down', 'small_down', 'stay',
                   'small_up', 'medium_up', 'large_up']
    arrow_interpretation = [arrow_names[a] for a in arrows.tolist()[0]]
    print(f"Arrow meanings: {arrow_interpretation}")
    
    return arrows

def interactive_generation_example():
    """Example 3: Interactive melody generation with user-defined arrows."""
    print("\n" + "="*60)
    print("Example 3: Interactive melody generation")
    print("="*60)
    
    model = create_model()
    
    # Start with middle C
    current_melody = torch.tensor([[60]])
    
    print(f"\nStarting pitch: {current_melody.tolist()[0][0]} (Middle C)")
    print("\nGenerating melody step by step with different arrows...")
    
    # Define a sequence of arrows to try
    arrow_sequence = [4, 4, 5, 3, 2, 2, 1, 3, 4, 5, 6, 2]  # Varied pattern
    arrow_names = ['large_down', 'medium_down', 'small_down', 'stay',
                   'small_up', 'medium_up', 'large_up']
    
    for i, arrow_idx in enumerate(arrow_sequence):
        # Prepare context
        prompts = {'pitch': current_melody}
        arrows = torch.tensor([[arrow_idx]])
        
        # Generate next note
        generated = model.generate(
            prompts=prompts,
            seq_len=1,
            arrow_sequence=arrows,
            temperature=0.8,
            verbose=False
        )
        
        # Append to melody
        current_melody = torch.cat([current_melody, generated], dim=1)
        
        # Show progress
        new_pitch = generated.item()
        prev_pitch = current_melody[0, -2].item()
        diff = new_pitch - prev_pitch
        
        print(f"Step {i+1}: Arrow={arrow_names[arrow_idx]:12s} → "
              f"Generated pitch={new_pitch:3d} (diff={diff:+3d})")
    
    print(f"\nFinal melody (MIDI): {current_melody.tolist()[0]}")
    
    return current_melody

def main():
    print("="*60)
    print("Melody Generation with Arrow Guidance Examples")
    print("="*60)
    
    # Run examples
    try:
        generate_melody_example()
        analyze_melody_example()
        interactive_generation_example()
        
        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


