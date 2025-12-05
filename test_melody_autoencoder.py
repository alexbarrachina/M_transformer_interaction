#!/usr/bin/env python3
"""
Test script for AutoregressiveAutoencoder_melody implementation.
Verifies that the arrow-based melody autoencoder works correctly.
"""

import torch
from x_transformer import pitch_to_arrow, Decoder_melody, AutoregressiveAutoencoder_melody
from params import DEFAULT_HPARAMS, PAD_IDX

def test_pitch_to_arrow():
    """Test the pitch_to_arrow conversion function."""
    print("Testing pitch_to_arrow function...")
    
    # Create a simple pitch sequence
    # Pitches: [60, 60, 62, 65, 72, 71, 69, 69, 70]
    # dPitch:     [0,  2,  3,  7, -1, -2,  0,  1]
    # Expected arrows: [3, 4, 5, 5, 2, 2, 3, 4]
    # Note: dPitch=7 maps to arrow 5 (range 3-7), not 6 (>=8)
    pitch_seq = torch.tensor([[60, 60, 62, 65, 72, 71, 69, 69, 70]])
    
    arrows = pitch_to_arrow(pitch_seq)
    
    print(f"Input pitches: {pitch_seq}")
    print(f"Pitch differences: {pitch_seq[:, 1:] - pitch_seq[:, :-1]}")
    print(f"Output arrows: {arrows}")
    
    expected = torch.tensor([[3, 4, 5, 5, 2, 2, 3, 4]])
    assert torch.all(arrows == expected), f"Expected {expected}, got {arrows}"
    print("✓ pitch_to_arrow test passed!\n")

def test_decoder_melody():
    """Test the Decoder_melody class."""
    print("Testing Decoder_melody class...")
    
    # Create a small decoder
    decoder = Decoder_melody(
        max_seq_len=128,
        dim=256,
        depth=2,
        heads=4,
        emb_dropout=0.1,
        rotary_pos_emb=True,
        attn_flash=False,  # Disable flash attention for testing
        causal=True
    )
    
    # Create dummy input
    batch_size = 2
    seq_len = 16
    past_tokens = {
        'pitch': torch.randint(0, 88, (batch_size, seq_len)),
        'arrow': torch.randint(0, 7, (batch_size, seq_len))
    }
    
    # Forward pass
    logits = decoder(past_tokens)
    
    print(f"Input shape: pitch={past_tokens['pitch'].shape}, arrow={past_tokens['arrow'].shape}")
    print(f"Output logits shape: {logits.shape}")
    
    assert logits.shape == (batch_size, seq_len, 128), \
        f"Expected shape ({batch_size}, {seq_len}, 128), got {logits.shape}"
    print("✓ Decoder_melody test passed!\n")

def test_autoencoder_melody():
    """Test the AutoregressiveAutoencoder_melody class."""
    print("Testing AutoregressiveAutoencoder_melody class...")
    
    # Create decoder
    decoder = Decoder_melody(
        max_seq_len=128,
        dim=256,
        depth=2,
        heads=4,
        emb_dropout=0.1,
        rotary_pos_emb=True,
        attn_flash=False,
        causal=True
    )
    
    # Create autoencoder
    cfg = DEFAULT_HPARAMS.copy()
    autoencoder = AutoregressiveAutoencoder_melody(
        decoder=decoder,
        cfg=cfg
    )
    
    # Create dummy training data
    batch_size = 2
    seq_len = 16
    note_tokens = {
        'pitch': torch.randint(0, 88, (batch_size, seq_len + 1))
    }
    
    # Forward pass (training)
    loss_dict, acc = autoencoder(note_tokens)
    
    print(f"Input pitch shape: {note_tokens['pitch'].shape}")
    print(f"Loss total: {loss_dict['loss_total'].item():.4f}")
    print(f"Loss recons: {loss_dict['loss_recons'].item():.4f}")
    print(f"Accuracy: {acc.item():.4f}")
    
    assert 'loss_total' in loss_dict, "loss_total not in loss dict"
    assert 'loss_recons' in loss_dict, "loss_recons not in loss dict"
    assert loss_dict['loss_total'].item() >= 0, "Loss should be non-negative"
    print("✓ AutoregressiveAutoencoder_melody forward test passed!\n")

def test_generation():
    """Test the generation method."""
    print("Testing generation method...")
    
    # Create decoder
    decoder = Decoder_melody(
        max_seq_len=128,
        dim=256,
        depth=2,
        heads=4,
        emb_dropout=0.0,
        rotary_pos_emb=True,
        attn_flash=False,
        causal=True
    )
    
    # Create autoencoder
    cfg = DEFAULT_HPARAMS.copy()
    autoencoder = AutoregressiveAutoencoder_melody(
        decoder=decoder,
        cfg=cfg
    )
    autoencoder.eval()
    
    # Create prompt and arrow sequence
    batch_size = 1
    prompt_len = 8
    gen_len = 16
    
    prompts = {
        'pitch': torch.randint(40, 80, (batch_size, prompt_len))
    }
    
    # Create arrow sequence (user would provide this)
    # For testing, create a simple pattern: up, up, stay, down, down, stay, ...
    arrow_pattern = [4, 4, 3, 2, 2, 3]  # small up, small up, stay, small down, small down, stay
    arrow_list = (arrow_pattern * (gen_len // len(arrow_pattern) + 1))[:gen_len]
    arrow_sequence = torch.tensor([arrow_list])  # [1, gen_len]
    
    print(f"Prompt pitches: {prompts['pitch']}")
    print(f"Arrow sequence: {arrow_sequence}")
    
    # Generate
    generated = autoencoder.generate(
        prompts=prompts,
        seq_len=gen_len,
        arrow_sequence=arrow_sequence,
        temperature=1.0,
        verbose=False
    )
    
    print(f"Generated pitch shape: {generated.shape}")
    print(f"Generated pitches: {generated}")
    
    assert generated.shape == (batch_size, gen_len), \
        f"Expected shape ({batch_size}, {gen_len}), got {generated.shape}"
    print("✓ Generation test passed!\n")

def test_gen_arrows():
    """Test the gen_arrows method."""
    print("Testing gen_arrows method...")
    
    # Create minimal autoencoder
    decoder = Decoder_melody(
        max_seq_len=128,
        dim=256,
        depth=1,
        heads=2,
    )
    
    cfg = DEFAULT_HPARAMS.copy()
    autoencoder = AutoregressiveAutoencoder_melody(decoder=decoder, cfg=cfg)
    autoencoder.eval()
    
    # Create pitch sequence
    note_tokens = {
        'pitch': torch.tensor([[60, 62, 65, 64, 60, 60, 67]])
    }
    
    arrows = autoencoder.gen_arrows(note_tokens)
    
    print(f"Input pitches: {note_tokens['pitch']}")
    print(f"Generated arrows: {arrows}")
    print(f"Pitch differences: {note_tokens['pitch'][:, 1:] - note_tokens['pitch'][:, :-1]}")
    
    assert arrows.shape == (1, 6), f"Expected shape (1, 6), got {arrows.shape}"
    print("✓ gen_arrows test passed!\n")

if __name__ == "__main__":
    print("="*60)
    print("Testing AutoregressiveAutoencoder_melody Implementation")
    print("="*60 + "\n")
    
    try:
        test_pitch_to_arrow()
        test_decoder_melody()
        test_autoencoder_melody()
        test_gen_arrows()
        test_generation()
        
        print("="*60)
        print("✓ All tests passed successfully!")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

