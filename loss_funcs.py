#===================================================================================================
# Monster Genie loss_funcs.py Python module
# Loss functions
#
#
# Copyright 2025 Alex Barrachina
# Licensed under the Apache License, Version 2.0
#===================================================================================================

import torch
from torch import Tensor

''' LOSS FUNCTIONS '''

def margin_loss(e):
    """
    # Improved margin loss: encourage values to be closer to [-1, 1] range
    # Instead of only penalizing values outside [-1, 1], also encourage 
    # values to use the full range effectively
    """
    margin_penalty = torch.square(
                torch.maximum(torch.abs(e) - 1, torch.zeros_like(e))
            )
            
    # Add a term to encourage using the full range (prevent collapse to center)
    range_utilization = 1.0 - torch.var(e, dim=1).mean()  # Penalize low variance
            
    loss_margin = margin_penalty.mean() + 0.1 * range_utilization
    return loss_margin

def deviate_loss(pitches, e):
    """
    Penalize button changes when consecutive pitches are the same
    """
    # Identify held notes (where consecutive pitches are the same)
    notes_held = (pitches[:, 1:-1] == pitches[:, :-2]).float()
            
    # Only apply loss if there are actually held notes in the batch
    if notes_held.sum() > 0:
        # Penalize button changes when notes are held
        button_changes = torch.diff(e, dim=1)
        held_button_changes = button_changes * notes_held
        loss_deviate = torch.square(held_button_changes).sum() / notes_held.sum().clamp(min=1e-6)
    else:
        # If no held notes, add small penalty to encourage stability
        loss_deviate = 0.01 * torch.square(torch.diff(e, dim=1)).mean()
    return loss_deviate

def simple_contour_loss(pitches, e):
    """
    Computes contour preservation loss that considers simple relationships between consecutive notes, just tendencies in direction (-1,+1)

    Args:
        pitches: Tensor of shape [batch, seq_len] containing pitch values
        e: Tensor of shape [batch, seq_len] containing e values (encoder output)
    
    Returns:
        A differentiable loss tensor
    """    
    pitch_diff = torch.diff(pitches[:,1:], dim=1)
    e_diff = torch.diff(e, dim=1) # [:, :-1]    
    loss_contour_perc = torch.square(
        torch.maximum(
            1 - pitch_diff.float() * e_diff,
            torch.zeros_like(pitch_diff, dtype=torch.float)
        )
    )
    return loss_contour_perc

def multi_step_contour_loss(pitches, buttons, max_steps=5):
    """
    Computes a contour preservation loss that considers relationships
    between the current note and multiple previous notes.
    
    Args:
        pitches: Tensor of shape [batch, seq_len] containing pitch values
        buttons: Tensor of shape [batch, seq_len] containing button values (encoder output)
        max_steps: Maximum number of steps back to consider
    
    Returns:
        A differentiable loss tensor
    """
    batch_size, seq_len = pitches.shape
    total_loss = torch.zeros(1, device=pitches.device)
    
    # Convert to float for calculations
    pitches = pitches.float()
    buttons = buttons.float()
    
    # For each step size (1 to max_steps)
    for step in range(1, min(max_steps + 1, seq_len)):
        # Calculate differences with notes 'step' positions back
        pitch_diffs = pitches[:, step:] - pitches[:, :-step]  # [batch, seq_len-step]
        button_diffs = buttons[:, step:] - buttons[:, :-step]  # [batch, seq_len-step]
        
        # Normalize the importance by step size (closer relationships matter more)
        step_weight = 1.0 / step
        
        # Calculate directional agreement
        # When pitch_diffs and button_diffs have the same sign, their product is positive
        # When they have opposite signs, their product is negative
        agreement = pitch_diffs * button_diffs
        
        # Penalize disagreements (when the product is <= 0)
        # The penalty increases with the magnitude of the disagreement
        disagreement_penalty = torch.square(
            torch.maximum(
                1 - agreement,  # 1 minus the agreement (higher for disagreements)
                torch.zeros_like(agreement)  # Zero floor to avoid penalizing agreements
            )
        )
        
        # Weight by step size and add to total loss
        step_loss = step_weight * disagreement_penalty.mean()
        total_loss += step_loss
        
    return total_loss

def interval_preservation_loss(pitches, buttons, max_steps=5):
    """
    Encourages the relative magnitudes of intervals to be preserved
    between pitches and buttons.
    """
    batch_size, seq_len = pitches.shape
    total_loss = torch.zeros(1, device=pitches.device)
    
    # Normalize both to [0,1] range for fair comparison
    pitch_range = (pitches.max(dim=1, keepdim=True)[0] - pitches.min(dim=1, keepdim=True)[0]).clamp(min=1e-5)
    button_range = (buttons.max(dim=1, keepdim=True)[0] - buttons.min(dim=1, keepdim=True)[0]).clamp(min=1e-5)
    
    norm_pitches = (pitches - pitches.min(dim=1, keepdim=True)[0]) / pitch_range
    norm_buttons = (buttons - buttons.min(dim=1, keepdim=True)[0]) / button_range
    
    for step in range(1, min(max_steps + 1, seq_len)):
        # Calculate normalized intervals
        pitch_intervals = torch.abs(norm_pitches[:, step:] - norm_pitches[:, :-step])
        button_intervals = torch.abs(norm_buttons[:, step:] - norm_buttons[:, :-step])
        
        # Compute difference between normalized intervals
        interval_diff = torch.abs(pitch_intervals - button_intervals)
        
        # Weight by step size (closer relationships matter more)
        step_weight = 1.0 / step
        step_loss = step_weight * interval_diff.mean()
        
        total_loss += step_loss
        
    return total_loss

def melodic_shape_loss(pitches, buttons, window_size=5):
    """
    Preserves the overall shape of melodic phrases by comparing
    the pattern of ups and downs within sliding windows.
    Vectorized with unfold + broadcasting for better gradient flow and speed.
    """
    batch_size, seq_len = pitches.shape
    if seq_len < window_size:
        return torch.zeros(1, device=pitches.device)

    pad = window_size // 2
    # B x T -> B x (T) x W windows centered at each position
    p_win = F.pad(pitches.float(), (pad, pad), mode='reflect').unfold(1, window_size, 1)
    b_win = F.pad(buttons.float(), (pad, pad), mode='reflect').unfold(1, window_size, 1)

    # Pairwise differences within each window: B x T x W x W
    p_pairs = p_win.unsqueeze(-1) - p_win.unsqueeze(-2)
    b_pairs = b_win.unsqueeze(-1) - b_win.unsqueeze(-2)

    # Soft sign to keep gradients
    scale = 5.0
    p_signs = torch.tanh(scale * p_pairs)
    b_signs = torch.tanh(scale * b_pairs)

    sign_agree = p_signs * b_signs  # 1 when same direction
    disagreement = (1 - sign_agree).clamp_min(0.0)

    # Mean over pairwise dims W x W, then over positions and batch
    window_loss = disagreement.mean(dim=(-1, -2))  # B x T
    return window_loss.mean()

def normalized_position_loss(pitches, buttons, num_buttons, window_size=15):
    """
    Calculates loss between normalized positions of pitches and buttons.
    
    Args:
        pitches: Tensor of shape [batch, seq_len] containing pitch values
        buttons: Tensor of shape [batch, seq_len] containing button values
        window_size: Size of window to calculate local min/max for pitches
    
    Returns:
        A differentiable loss tensor
    """
    batch_size, seq_len = pitches.shape
    
    # Convert to float for calculations
    pitches = pitches.float()
    buttons = buttons.float()
    
    # Calculate normalized button positions (0 to 1)
    # If 'buttons' is continuous encoder output e in [-1, 1], map to [0, 1].
    # If it is discrete indices [0..num_buttons-1], scale accordingly.
    norm_buttons = (buttons + 1.0) * 0.5
    norm_buttons = norm_buttons.clamp(0.0, 1.0)
    
    # Calculate normalized pitch positions using sliding window (vectorized)
    pad = window_size // 2
    frames = F.pad(pitches, (pad, pad), mode='reflect').unfold(1, window_size, 1)  # B x T x W
    local_min = frames.min(dim=2, keepdim=False)[0]
    local_max = frames.max(dim=2, keepdim=False)[0]
    range_size = (local_max - local_min).clamp(min=1e-6)
    norm_pitches = (pitches - local_min) / range_size
    
    # Calculate quadratic difference between normalized positions
    position_diff = torch.square(norm_pitches.clamp(0.0, 1.0) - norm_buttons)
    
    return position_diff.mean()



def pitch_button_correlation_loss(pitches, e, window_size=15, tendency_distance=40):
    """
    Calculates loss that correlates pitch tendencies with button concentrations.
    
    Args:
        pitches: Tensor of shape [batch, seq_len] containing pitch values
        e: Tensor of shape [batch, seq_len] containing encoder outputs in [-1,1] range
        window_size: Size of window to calculate local pitch means
        tendency_distance: Distance between tokens to calculate pitch tendency
    
    Returns:
        A differentiable loss tensor
    """
    batch_size, seq_len = pitches.shape
    
    # Convert to float for calculations
    pitches = pitches.float()
    e = e.float()
    
    # Calculate pitch means for each position using sliding window
    # Sliding window means via unfold (keeps gradients and is efficient)
    pad = window_size // 2
    padded = F.pad(pitches, (pad, pad), mode='reflect')
    frames = padded.unfold(1, window_size, 1)
    pitch_means = frames.mean(dim=2)
    
    # Calculate pitch tendencies by comparing current pitch_mean with earlier pitch_mean
    pitch_tendencies = torch.zeros_like(pitches)
    
    for i in range(tendency_distance, seq_len):
        # Compare current pitch_mean with pitch_mean from tendency_distance steps ago
        current_pitch_mean = pitch_means[:, i:i+1]
        earlier_pitch_mean = pitch_means[:, i-tendency_distance:i-tendency_distance+1]
        pitch_tendencies[:, i:i+1] = current_pitch_mean - earlier_pitch_mean
    
    # Calculate button concentrations (mean of e values in sliding window)
    # Button concentrations with unfold
    padded_e = F.pad(e, (pad, pad), mode='reflect')
    frames_e = padded_e.unfold(1, window_size, 1)
    button_concentrations = frames_e.mean(dim=2)
    
    # Calculate correlation loss
    # We want:
    # - High pitch tendencies (positive) to correlate with high button concentrations (>0)
    # - Low pitch tendencies (negative) to correlate with low button concentrations (<0)
    # So their product should be positive in both cases
    
    # Only consider positions where we have valid pitch tendencies
    valid_mask = torch.zeros_like(pitch_tendencies)
    valid_mask[:, tendency_distance:] = 1.0
    
    # Calculate correlation only for valid positions
    correlation = pitch_tendencies * button_concentrations * valid_mask
    
    # Penalize when correlation is negative (opposite tendencies)
    loss = torch.square(
        torch.maximum(
            -correlation,  # Negative when tendencies are opposite
            torch.zeros_like(correlation)
        )
    ).sum() / valid_mask.sum().clamp(min=1e-6)  # Average only over valid positions
    
    return loss

def button_concentration_loss(e, note_tokens, num_buttons, window_size=15, tendency_distance=40, button_concentration_window_size=12):
    """
    Calculates loss that enforces button concentration in a BUTTON_CONCENTRATION_WINDOW_SIZE window
    that shifts based on pitch tendency.
    
    Args:
        e: Tensor of shape [batch, seq_len] containing encoder outputs in [-1,1] range
        note_tokens: Dictionary containing pitch tokens
        window_size: Size of window to calculate local pitch means
        tendency_distance: Distance between tokens to calculate pitch tendency
    
    Returns:
        A differentiable loss tensor
    """

    batch_size, seq_len = e.shape
    pitches = note_tokens['pitch'][:, 1:]  # Use same pitch slice as in other functions
    
    # Convert to float for calculations
    pitches = pitches.float()
    e = e.float()
    
    # Calculate pitch means for each position using sliding window (unfold keeps grad)
    pad = window_size // 2
    padded = F.pad(pitches, (pad, pad), mode='reflect')
    frames = padded.unfold(1, window_size, 1)
    pitch_means = frames.mean(dim=2)
    
    # Calculate pitch tendencies by comparing current pitch_mean with earlier pitch_mean
    pitch_tendencies = torch.zeros_like(pitches)
    
    for i in range(tendency_distance, seq_len):
        # Compare current pitch_mean with pitch_mean from tendency_distance steps ago
        current_pitch_mean = pitch_means[:, i:i+1]
        earlier_pitch_mean = pitch_means[:, i-tendency_distance:i-tendency_distance+1]
        pitch_tendencies[:, i:i+1] = current_pitch_mean - earlier_pitch_mean
    
    # Scale pitch tendencies to control window shift
    # Use tanh to bound the tendencies and scale appropriately
    scaled_tendency = torch.tanh(pitch_tendencies / 10.0)  # Normalize pitch differences
    
    # Calculate target center for button concentration window
    # e range [-1,1] equivalent to button range [0-18]
    # BUTTON_CONCENTRATION_WINDOW_SIZE (12) in e space = 12/19*2 = 1.263
    window_size_e = button_concentration_window_size / num_buttons * 2  # 1.263
    max_shift = (2 - window_size_e) / 2  # Maximum shift from center = (2-1.263)/2 = 0.368
    
    # For positive tendency: shift toward +1 (high buttons)
    # For negative tendency: shift toward -1 (low buttons)
    # For neutral tendency: center at 0
    target_center = scaled_tendency * max_shift
    
    # Calculate how far e values are from their target center
    distance_from_center = torch.abs(e - target_center)
    
    # Penalize when e values are too far from their target center
    # Allow for half the window size on each side
    allowed_distance = window_size_e / 2  # 1.263/2 = 0.632
    
    # Only consider positions where we have valid pitch tendencies
    valid_mask = torch.zeros_like(pitch_tendencies)
    valid_mask[:, tendency_distance:] = 1.0
    
    # Calculate loss only for valid positions
    violations = torch.maximum(
        distance_from_center - allowed_distance,
        torch.zeros_like(distance_from_center)
    ) * valid_mask
    
    loss = torch.square(violations).sum() / valid_mask.sum().clamp(min=1e-6)
    
    return loss

#===================================================
def windowed_correlation_loss(
    pitches: Tensor,
    e: Tensor,
    window_size: int = 11,
    eps: float = 1e-6
) -> Tensor:
    """
    Sliding-window Pearson correlation loss between normalized pitch windows
    and encoder output windows. Maximizes correlation by minimizing (1 - corr).

    Args:
        pitches: [B, T] float tensor (e.g., MIDI semitones)
        e:       [B, T] float tensor in [-1, 1]
        window_size: odd window size for local correlation (>=3)
        eps: small constant for numerical stability
        use_abs_corr: if True, maximize |corr| (reward negative correlation too)

    Returns:
        Scalar tensor loss.
    """
    B, T = pitches.shape
    if T < window_size:
        # Return differentiable zero tensor on correct device/dtype
        return torch.zeros((), device=pitches.device, dtype=e.dtype, requires_grad=True)

    p = pitches.float()
    x = e.float()

    # Extract sliding windows [B, n_windows, window_size]
    pw = p.unfold(1, window_size, 1)
    xw = x.unfold(1, window_size, 1)

    # Z-score normalize per window (mean 0, std 1)
    pw = pw - pw.mean(dim=2, keepdim=True)
    xw = xw - xw.mean(dim=2, keepdim=True)
    pw = pw / (pw.std(dim=2, keepdim=True).clamp(min=eps))
    xw = xw / (xw.std(dim=2, keepdim=True).clamp(min=eps))

    # Compute Pearson correlation per window
    corr = (pw * xw).mean(dim=2).clamp(min=-1.0, max=1.0)  # [B, n_windows]

    #  Define loss (maximize correlation → minimize 1 - corr)
    loss = (1.0 - corr).clamp_min(0.0)  # focus on positive correlation only

    # 5️⃣ Return mean loss across batch and windows
    return loss.mean()

#===================================================
def button_held_loss(
    pitches: Tensor,
    e: Tensor,
    num_buttons: int,
) -> Tensor:
    """
     When the melody moves to a different note, the latent/button trajectory should also change meaningfully (roughly a bin’s worth). 
    This loss nudges e to move at least ~0.8 of one bin on pitch changes, while not penalizing positions where the pitch is held (mask is 0 there).
    It’s “soft” and fully differentiable (uses the continuous e and ReLU-like clamping), so it’s training-friendly.
    Args:
        pitches: [B, T] float tensor (e.g., MIDI semitones)
        e:       [B, T] float tensor in [-1, 1]
        window_size: odd window size for local correlation (>=3)
        eps: small constant for numerical stability
        use_abs_corr: if True, maximize |corr| (reward negative correlation too)

    Returns:
        Scalar tensor loss.
    """
    # Soft button-held penalty using continuous e (keeps gradients)
    # Identifies when consecutive notes are different (pitch change)
    # Penalizes same button values when consecutive notes are different
    # Helps maintain consistency in the mapping
    notes_diff = (pitches[:, 1:] != pitches[:, :-1]).float() # a mask notes_diff that is 1 where the pitch changes between time t−1 and t, and 0 where it stays the same.
    delta_e = torch.abs(torch.diff(e, dim=1)) #  the absolute step size in e between consecutive steps: |e_t − e_{t-1}|
    bin_size = 2.0 / (num_buttons - 1) # the size of one button bin, which is 2/(num_buttons−1)
    margin = 0.8 * bin_size #  minimum desired movement threshold 80% of one bin
    loss = ((margin - delta_e).clamp_min(0.0) * notes_diff).mean() #Penalizes steps where the pitch changed but e moved less than margin. Averages over batch and time to get a scalar.
    return loss
