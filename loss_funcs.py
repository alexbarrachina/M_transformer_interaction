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
from torch.nn import functional as F

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



def pitch_button_correlation_loss(pitches, e, window_size=6, tendency_distance=1):
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

def button_concentration_loss(
    e: Tensor,
    note_tokens: dict,
    num_buttons: int,
    window_size: int = 16,
    tendency_distance: int = 32,
    eps: float = 1e-6
) -> Tensor:
    """
    Calculates loss that correlates average button position with pitch tendency.
    
    When the performer plays buttons close to the higher button (e close to +1),
    force the model to generate higher pitches (ascending tendency).
    When buttons are low (e close to -1), force descending pitch tendency.
    
    Uses sliding windows to compute:
    - Average button position (mean of e values in window)
    - Pitch tendency (difference between current and earlier pitch window means)
    
    Penalizes when:
    - High average buttons don't correspond with ascending pitch tendencies
    - Low average buttons don't correspond with descending pitch tendencies
    
    Args:
        e: Tensor of shape [batch, seq_len] containing encoder outputs in [-1,1] range
        note_tokens: Dictionary containing pitch tokens
        num_buttons: Number of buttons (unused, kept for API compatibility)
        window_size: Size of sliding window for averaging
        tendency_distance: Distance between windows to calculate pitch tendency
        eps: Small constant for numerical stability
    
    Returns:
        A differentiable loss tensor
    """
    batch_size, seq_len = e.shape
    pitches = note_tokens['pitch'][:, 1:]  # Use same pitch slice as in other functions
    
    # Ensure shapes match
    min_len = min(seq_len, pitches.size(1))
    pitches = pitches[:, :min_len].float()
    e = e[:, :min_len].float()
    
    # Need enough length for windowed operations
    if min_len < window_size + tendency_distance:
        return torch.zeros((), device=e.device, dtype=e.dtype, requires_grad=True)
    
    # Calculate sliding window means for buttons (e values)
    # Vectorized using unfold
    pad = window_size // 2
    padded_e = F.pad(e, (pad, pad), mode='reflect')
    e_windows = padded_e.unfold(1, window_size, 1)  # [B, T, W]
    button_means = e_windows.mean(dim=2)  # [B, T] average button position per step
    
    # Calculate sliding window means for pitches
    padded_pitches = F.pad(pitches, (pad, pad), mode='reflect')
    pitch_windows = padded_pitches.unfold(1, window_size, 1)  # [B, T, W]
    pitch_means = pitch_windows.mean(dim=2)  # [B, T] average pitch per step
    
    # Calculate pitch tendency: difference between current and earlier pitch means
    # Positive tendency = pitch going up, negative = pitch going down
    # Vectorized: compare windows separated by tendency_distance
    current_pitch_means = pitch_means[:, tendency_distance:]  # [B, T-tendency_distance]
    earlier_pitch_means = pitch_means[:, :-tendency_distance]  # [B, T-tendency_distance]
    pitch_tendency = current_pitch_means - earlier_pitch_means  # [B, T-tendency_distance]
    
    # Get corresponding button means (aligned with the current window position)
    button_means_aligned = button_means[:, tendency_distance:]  # [B, T-tendency_distance]
    
    # Normalize pitch tendency to [-1, 1] range using tanh
    # Scale factor controls sensitivity (smaller = more sensitive to small pitch changes)
    pitch_tendency_normalized = torch.tanh(pitch_tendency / 6.0)
    
    # The core idea:
    # - button_means_aligned is in [-1, 1]: high values = high buttons, low = low buttons
    # - pitch_tendency_normalized is in [-1, 1]: positive = ascending, negative = descending
    # 
    # We want them to agree in sign and roughly in magnitude:
    # - High buttons (positive e) should correlate with ascending pitch (positive tendency)
    # - Low buttons (negative e) should correlate with descending pitch (negative tendency)
    #
    # The product button_means * pitch_tendency should be positive when they agree.
    # We penalize when the product is negative (disagreement) or when there's a mismatch.
    
    # Calculate extremeness weight: buttons near ±1 get disproportionately stronger force
    # Use squared absolute value to create quadratic weighting:
    # - At e=0 (center): weight ≈ 0
    # - At e=±0.5: weight = 0.25
    # - At e=±1.0 (extremes): weight = 1.0
    # This gives 4x more weight at extremes compared to halfway points
    extremeness_weight = torch.square(torch.abs(button_means_aligned))  # [B, T-tendency_distance]
    
    # Alternative: even more aggressive cubic weighting (uncomment to use)
    # extremeness_weight = torch.pow(torch.abs(button_means_aligned), 3)
    
    agreement = button_means_aligned * pitch_tendency_normalized  # [B, T-tendency_distance]
    
    # Penalize disagreement (when signs don't match)
    # Apply extremeness weighting: violations at extreme buttons are penalized much more
    disagreement_loss = torch.relu(-agreement) * (1.0 + 2.0 * extremeness_weight)
    
    # Additional loss: penalize when button magnitude doesn't correlate with tendency magnitude
    # High |button| should correspond to high |tendency|
    magnitude_diff = torch.abs(button_means_aligned) - torch.abs(pitch_tendency_normalized)
    # Penalize only when buttons are extreme but pitch tendency is weak
    # Apply stronger extremeness weighting here since this specifically targets extreme positions
    magnitude_mismatch = torch.relu(magnitude_diff) * (1.0 + 3.0 * extremeness_weight)
    
    # Combine losses
    # Weight disagreement more heavily than magnitude mismatch
    total_loss = disagreement_loss.mean() + 0.5 * magnitude_mismatch.mean()
    
    return total_loss

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

def expected_pitch_from_logits(logits):
    """
    Compute expected pitch value from logits (differentiable soft argmax).
    
    Args:
        logits: [B, T, vocab_size] pitch prediction logits
    
    Returns:
        [B, T] expected pitch values
    """
    probs = F.softmax(logits, dim=-1)
    values = torch.arange(logits.size(-1), device=logits.device, dtype=probs.dtype)
    return (probs * values).sum(dim=-1)

def predicted_contour_loss(predicted_pitch, buttons):
    """
    Contour loss on predicted pitches vs button controls.
    Encourages predicted pitch intervals to follow button intervals in direction.
    
    Args:
        predicted_pitch: [B, T] predicted/expected pitch values (differentiable)
        buttons: [B, T] button values (continuous)
    
    Returns:
        Scalar loss
    """
    if predicted_pitch.size(1) < 2:
        return torch.tensor(0.0, device=predicted_pitch.device)
    
    # Compute differences
    dp = torch.diff(predicted_pitch, dim=1)  # [B, T-1]
    db = torch.diff(buttons, dim=1)  # [B, T-1]
    
    # Penalize when directions don't match (sign disagreement)
    # Loss is 0 when dp*db > 1, increases when they disagree
    loss = torch.square(torch.relu(1.0 - dp * db)).mean()
    return loss

def saturated_contour_loss(
    pitches: Tensor,          # [B, T+1] like you pass to simple_contour_loss
    e: Tensor,                # [B, T] encoder output (should be roughly in [-1, 1])
    num_buttons: int,
    sat_bin_frac: float = 0.5,    # how deep into the extreme bin before we "allow saturation"
    tau: float = 0.05,            # softness of the saturation gate
    hold_weight: float = 0.05,    # optional: discourage changing e when pitch is flat
    min_dp: float = 0.5           # minimum pitch interval for scaling (prevents division issues)
) -> Tensor:
    """
    Saturated contour loss that scales with pitch interval magnitude (like simple_contour_loss)
    but allows saturation at button extremes.
    
    In the middle range: behaves like simple_contour_loss, wanting dp * de >= 1
    Near extremes: relaxes the requirement, allowing buttons to saturate (repeat top/bottom)
    
    Args:
        pitches: [B, T+1] pitch sequence
        e: [B, T] encoder output in [-1, 1]
        num_buttons: number of discrete buttons
        sat_bin_frac: how deep into extreme bin before saturation is allowed
        tau: softness of saturation gate transition
        hold_weight: weight for penalizing button changes when pitch is held
        min_dp: minimum pitch interval for scaling (prevents division by very small values)
    """
    # Align pitches to e length: pitches[:, 1:] is [B, T]
    p = pitches[:, 1:].float()

    # Differences: [B, T-1]
    dp = torch.diff(p, dim=1)
    de = torch.diff(e.float(), dim=1)

    dir_up = (dp > 0).float()
    dir_dn = (dp < 0).float()
    dir_eq = 1.0 - dir_up - dir_dn

    # One discrete button bin width in e-space ([-1, 1] split into num_buttons bins)
    bin_size = 2.0 / float(num_buttons - 1)

    # Use midpoint between consecutive e's to decide "near extreme"
    e_mid = 0.5 * (e[:, 1:] + e[:, :-1])

    # Thresholds: consider "saturated" when already inside the top/bottom bin region
    hi_thresh = 1.0 - sat_bin_frac * bin_size
    lo_thresh = -1.0 + sat_bin_frac * bin_size

    # Soft gates in [0,1]
    sat_up = torch.sigmoid((e_mid - hi_thresh) / tau)     # high when near +1
    sat_dn = torch.sigmoid((lo_thresh - e_mid) / tau)     # high when near -1

    # Only allow saturation when pitch is trying to go outward
    sat_mask = dir_up * sat_up + dir_dn * sat_dn

    # Scale required movement with pitch interval (like simple_contour_loss)
    # simple_contour_loss wants: dp * de >= 1, i.e., de >= 1/dp
    # We use the same scaling but apply saturation mask to reduce/eliminate at extremes
    dp_abs = dp.abs().clamp(min=min_dp)
    req = (1.0 / dp_abs) * (1.0 - sat_mask)

    # Upward pitch: want de >= req
    loss_up = F.relu(req - de) * dir_up

    # Downward pitch: want de <= -req  <=>  req + de <= 0
    loss_dn = F.relu(req + de) * dir_dn

    # Square the losses for smoother gradients (like simple_contour_loss)
    loss_contour = torch.square(loss_up + loss_dn)

    # Optional stability when pitch is held
    loss_hold = torch.square(de) * dir_eq

    return loss_contour.mean() + hold_weight * loss_hold.mean()


def pitch_extreme_anchoring_loss(
    pitches: Tensor,          # [B, T+1] or [B, T] pitch sequence
    e: Tensor,                # [B, T] encoder output (should be in [-1, 1])
    high_pitch_thresh: float = 84.0,    # MIDI pitch above which is considered "high" (C6)
    low_pitch_thresh: float = 36.0,     # MIDI pitch below which is considered "low" (C2)
    transition_width: float = 12.0,     # Semitones over which the sigmoid transition occurs
    target_extreme: float = 0.95,       # Target e value at extremes (±0.95 instead of ±1.0 for stability)
) -> Tensor:
    """
    Explicitly anchors extreme pitches to extreme button values.
    
    - Very high pitches (above high_pitch_thresh) should map to e ≈ +target_extreme
    - Very low pitches (below low_pitch_thresh) should map to e ≈ -target_extreme
    
    This helps the model learn the semantics of button 0 (lowest) and button N-1 (highest)
    faster, and complements the saturated_contour_loss.
    
    Args:
        pitches: [B, T+1] or [B, T] pitch sequence (MIDI note numbers 0-127)
        e: [B, T] encoder output in [-1, 1] range
        high_pitch_thresh: Pitch value above which we want e ≈ +1
        low_pitch_thresh: Pitch value below which we want e ≈ -1
        transition_width: Width of sigmoid transition (in semitones)
        target_extreme: Target value at extremes (use <1.0 for margin safety)
    
    Returns:
        Scalar loss tensor
    """
    # Align shapes: if pitches is [B, T+1], use pitches[:, 1:] to match e's [B, T]
    if pitches.size(1) == e.size(1) + 1:
        p = pitches[:, 1:].float()
    else:
        p = pitches.float()
    
    # Soft weights for "how much is this pitch in the high/low extreme?"
    # sigmoid((p - thresh) / width) gives smooth 0→1 transition
    tau = transition_width / 6.0  # Scale factor for sigmoid steepness
    
    # High pitch weight: 0 for p << high_thresh, 1 for p >> high_thresh
    w_high = torch.sigmoid((p - high_pitch_thresh) / tau)
    
    # Low pitch weight: 1 for p << low_thresh, 0 for p >> low_thresh
    w_low = torch.sigmoid((low_pitch_thresh - p) / tau)
    
    # Target e values for high/low pitches
    # For high pitches: want e ≈ +target_extreme
    # For low pitches: want e ≈ -target_extreme
    target_high = target_extreme
    target_low = -target_extreme
    
    # Weighted squared errors
    # Only penalize high pitches that don't map to high e values
    loss_high = w_high * torch.square(e - target_high)
    
    # Only penalize low pitches that don't map to low e values
    loss_low = w_low * torch.square(e - target_low)
    
    # Total loss: mean over batch and time
    # Only positions with significant weights contribute meaningfully
    total_loss = (loss_high + loss_low).mean()
    
    return total_loss

def non_linear_compression_loss_vectorized(
    pitches: Tensor,          # [B, T+1] or [B, T] pitch sequence
    e: Tensor,                # [B, T] encoder output in [-1, 1]
    steepness: float = 6.0,   # Controls compression: higher = more compression at extremes
    window_size: int = 128,   # Window size for local normalization
    contour_weight: float = 0.5,  # Weight for contour matching
) -> Tensor:
    """
    Vectorized version of non_linear_compression_loss (faster, no Python loop).
    Uses unfold for efficient sliding window operations.
    """
    # Align shapes
    if pitches.size(1) == e.size(1) + 1:
        p = pitches[:, 1:].float()
    else:
        p = pitches.float()
    
    batch_size, seq_len = p.shape
    
    # Pad for sliding window
    pad_size = window_size // 2
    p_padded = F.pad(p, (pad_size, pad_size), mode='replicate')
    
    # Unfold to get all windows: [B, num_windows, window_size]
    windows = p_padded.unfold(dimension=1, size=window_size, step=1)
    
    # Unfold produces num_windows = padded_len - window_size + 1 = seq_len + 1
    # We need to slice to match seq_len exactly
    if windows.size(1) > seq_len:
        windows = windows[:, :seq_len, :]
    
    # Get min/max per window
    p_min = windows.min(dim=2, keepdim=True)[0].squeeze(-1)  # [B, seq_len]
    p_max = windows.max(dim=2, keepdim=True)[0].squeeze(-1)  # [B, seq_len]
    range_size = (p_max - p_min).clamp(min=1.0)
    
    # Normalize
    p_norm = (p - p_min) / range_size
    
    # Apply non-linear compression
    p_compressed = torch.sigmoid((p_norm - 0.5) * steepness)
    
    # Renormalize to full [0, 1] range
    p_comp_min = torch.sigmoid(torch.tensor(-0.5 * steepness, device=p.device))
    p_comp_max = torch.sigmoid(torch.tensor(0.5 * steepness, device=p.device))
    p_compressed = (p_compressed - p_comp_min) / (p_comp_max - p_comp_min)
    
    # Map to e-space [-1, 1]
    target_e = p_compressed * 2.0 - 1.0
    
    # Position loss
    loss_position = torch.square(e - target_e)
    
    # Contour loss
    if contour_weight > 0:
        de = torch.diff(e, dim=1)
        dt = torch.diff(target_e, dim=1)
        loss_contour = torch.square(torch.relu(1.0 - de * dt))
        total_loss = loss_position.mean() + contour_weight * loss_contour.mean()
    else:
        total_loss = loss_position.mean()
    
    return total_loss


def latent_velocity_loss(
    pitches: Tensor,          # [B, T+1] or [B, T] pitch sequence (MIDI note numbers)
    e: Tensor,                # [B, T] encoder output (latent z) in [-1, 1]
    alpha: float = 12.0,      # Slope strength: how much pitch change per unit latent
    normalize_pitch: bool = True,  # Normalize pitch changes to similar scale as e
    pitch_range: float = 88.0,     # Range of pitches for normalization (88 piano keys)
) -> Tensor:
    """
    Latent-to-velocity coupling loss.
    
    This loss forces the encoder latent (z/e) to act as a pitch velocity controller:
    - High latent (e ≈ +1) → should produce positive pitch changes (ascending)
    - Low latent (e ≈ -1) → should produce negative pitch changes (descending)
    - Middle latent (e ≈ 0) → should produce small/no pitch changes
    
    Loss: L_vel = ||Δx_t - α * z_t||²
    
    Where:
    - Δx_t is the pitch change (x_t - x_{t-1})
    - z_t is the encoder latent (e value)
    - α controls the expected slope magnitude
    
    This recreates the LSTM's implicit "button = velocity" behavior that emerges
    naturally in recurrent networks but not in Transformers.
    
    Args:
        pitches: [B, T+1] or [B, T] pitch sequence
        e: [B, T] encoder latent values in [-1, 1]
        alpha: Expected pitch change per unit latent. Higher = steeper slopes.
               Default 12.0 means e=+1 expects pitch to rise by ~12 semitones per step,
               e=-1 expects pitch to fall by ~12 semitones.
        normalize_pitch: If True, normalize pitch changes to [-1, 1] range
        pitch_range: Range for normalization (default 88 for piano)
    
    Returns:
        Loss tensor (scalar)
    """
    # Align shapes: we need pitches to have same length as e for diff calculation
    if pitches.size(1) == e.size(1) + 1:
        # pitches is [B, T+1], e is [B, T]
        # Use pitches[:, 1:] to align with e, then compute diff
        p = pitches.float()
        # Pitch changes: Δx_t = x_t - x_{t-1}, where t aligns with e indices
        # pitches[:, 1:] corresponds to e positions, pitches[:, :-1] is previous
        delta_pitch = p[:, 1:] - p[:, :-1]  # [B, T]
    else:
        # pitches is [B, T], same as e
        p = pitches.float()
        # We can only compute T-1 deltas
        delta_pitch = torch.diff(p, dim=1)  # [B, T-1]
        e = e[:, :-1]  # Align e to match delta_pitch length
    
    # Optionally normalize pitch changes to similar scale as e (which is in [-1, 1])
    if normalize_pitch:
        # Normalize: a change of pitch_range maps to 2.0 (full -1 to +1 swing)
        delta_pitch_norm = delta_pitch / (pitch_range / 2.0)
    else:
        delta_pitch_norm = delta_pitch
    
    # Target: pitch velocity should match latent * alpha
    # If e = +1, we want delta_pitch_norm ≈ +alpha (in normalized space)
    # If normalize_pitch is True, alpha should be in normalized space too
    if normalize_pitch:
        # Alpha in normalized space: alpha=1.0 means e=+1 expects max pitch change
        target_velocity = e * alpha
    else:
        target_velocity = e * alpha
    
    # Loss: squared difference between actual and target velocity
    loss = torch.square(delta_pitch_norm - target_velocity)
    
    return loss.mean()


def drift_regularization_loss(
    pitches: Tensor,          # [B, T+1] or [B, T] pitch sequence
    e: Tensor,                # [B, T] encoder latent in [-1, 1]
    drift_window: int = 8,    # How far back to look for cumulative drift
    normalize_pitch: bool = True,
    pitch_range: float = 88.0,
) -> Tensor:
    """
    Drift regularization loss.
    
    Encourages cumulative pitch motion in the direction of the latent.
    This rewards long-term movement, not just local step-to-step changes.
    
    Loss: L_drift = E[ReLU(-z_t * drift_t)]
    
    Where drift_t = (x_t - x_{t-k}) is the cumulative pitch change.
    
    This penalizes MISALIGNMENT between:
    - The latent direction (z_t)
    - The cumulative pitch change over the last k steps
    
    When z_t and drift_t have the same sign (aligned), loss = 0
    When z_t and drift_t have opposite signs (misaligned), loss > 0
    
    So if z_t > 0 (high button), we want (x_t - x_{t-k}) > 0 (pitch went up)
    If z_t < 0 (low button), we want (x_t - x_{t-k}) < 0 (pitch went down)
    
    Args:
        pitches: [B, T+1] or [B, T] pitch sequence
        e: [B, T] encoder latent
        drift_window: How many steps back to measure cumulative drift (k)
        normalize_pitch: Normalize pitch drift to [-1, 1] scale
        pitch_range: Range for normalization
    
    Returns:
        Loss tensor (scalar, always >= 0)
    """
    # Align shapes
    if pitches.size(1) == e.size(1) + 1:
        p = pitches[:, 1:].float()  # [B, T]
    else:
        p = pitches.float()
    
    batch_size, seq_len = p.shape
    
    # We need at least drift_window + 1 elements to compute drift
    if seq_len <= drift_window:
        return torch.tensor(0.0, device=e.device)
    
    # Cumulative drift: x_t - x_{t-k}
    # For each position t >= drift_window, compute p[:, t] - p[:, t - drift_window]
    current_pitch = p[:, drift_window:]  # [B, T - drift_window]
    past_pitch = p[:, :-drift_window]     # [B, T - drift_window]
    drift = current_pitch - past_pitch     # [B, T - drift_window]
    
    # Align e to match drift positions
    e_aligned = e[:, drift_window:]  # [B, T - drift_window]
    
    # Normalize drift if requested
    if normalize_pitch:
        # A drift of pitch_range maps to 2.0
        drift_norm = drift / (pitch_range / 2.0)
    else:
        drift_norm = drift
    
    # Reformulated as a standard positive loss (squared error encouraging alignment)
    # We want drift_norm to align with e_aligned (both in similar scales)
    # When e > 0, we want drift > 0; when e < 0, we want drift < 0
    # 
    # Method 1: Penalize misalignment using hinge loss (always positive)
    # When e * drift < 0 (opposite signs), penalize
    # When e * drift > 0 (same signs), no penalty
    misalignment = -e_aligned * drift_norm  # Negative when aligned, positive when misaligned
    loss = torch.relu(misalignment)  # Only penalize misalignment
    
    # Method 2 (alternative): Squared alignment loss
    # This encourages drift to be proportional to latent
    # Uncomment below if you prefer this approach:
    # loss = torch.square(drift_norm - e_aligned)
    
    return loss.mean()


def latent_velocity_and_drift_loss(
    pitches: Tensor,
    e: Tensor,
    alpha: float = 1.0,           # Velocity coupling strength
    drift_window: int = 8,        # Drift lookback window
    drift_weight: float = 0.5,    # Weight for drift term relative to velocity term
    normalize_pitch: bool = True,
    pitch_range: float = 88.0,
) -> Tensor:
    """
    Combined latent velocity and drift loss.
    
    This combines both:
    1. Local velocity coupling: pitch changes should match latent direction
    2. Cumulative drift: long-term pitch movement should align with latent
    
    Together, these recreate the LSTM's implicit stateful dynamics in Transformers,
    making extreme buttons (1 and 8) act as pitch velocity controllers:
    - Button 8 (high latent) → ascending pitch sequences
    - Button 1 (low latent) → descending pitch sequences
    
    Args:
        pitches: [B, T+1] or [B, T] pitch sequence
        e: [B, T] encoder latent
        alpha: Velocity coupling strength (how much pitch change per unit latent)
        drift_window: Steps to look back for cumulative drift
        drift_weight: Weight of drift term relative to velocity term
        normalize_pitch: Normalize pitch to [-1, 1] scale
        pitch_range: Range for normalization
    
    Returns:
        Combined loss tensor
    """
    loss_vel = latent_velocity_loss(
        pitches, e, alpha=alpha, 
        normalize_pitch=normalize_pitch, pitch_range=pitch_range
    )
    
    loss_drift = drift_regularization_loss(
        pitches, e, drift_window=drift_window,
        normalize_pitch=normalize_pitch, pitch_range=pitch_range
    )
    
    return loss_vel + drift_weight * loss_drift


def companded_warp_loss_tanh(
    pitches: Tensor,
    e: Tensor,
    k: float = 2.5,                 # warp strength: >1 expands low pitches, compresses high pitches
    low_emphasis: float = 2.0,       # weighting strength: >1 emphasizes low pitches more
    w_min: float = 0.10,             # minimum weight at high pitches (keeps some signal everywhere)
    contour_weight: float = 0.25,    # 0 disables contour; otherwise adds warped contour agreement
    huber_delta: float = 0.10,
    eps: float = 1e-6
) -> Tensor:
    """
    Asymmetric non-uniform "control density" loss:

      pitch (0..127) -> x in [0,1]
      warp: u01 = 1 - (1-x)^k  (concave, expands low, compresses high)
      u = 2*u01 - 1 in [-1,1]

    Then:
      - Align encoder output e (in [-1,1]) to u (Huber)
      - Weight alignment more for low pitches, less for high pitches
      - Optional contour agreement in warped space

    Expected shapes:
      pitches: [B, T+1] or [B, T]
      e:       [B, T]
    """

    # --- Align pitch length with e length (your code often has pitches [B, T+1]) ---
    if pitches.size(1) == e.size(1) + 1:
        p = pitches[:, 1:]
    else:
        p = pitches
        T = min(p.size(1), e.size(1))
        p = p[:, :T]
        e = e[:, :T]

    if e.size(1) < 2:
        return torch.zeros(1, device=e.device)

    # --- Normalize MIDI pitch (0..127) to x in [0,1] ---
    p = p.float()
    e_f = e.float()

    x = (p / 127.0).clamp(0.0, 1.0)  # [B, T]

    # --- Asymmetric companding warp: expand low, compress high ---
    # u01 in [0,1]
    k_t = torch.tensor(float(k), device=e.device, dtype=x.dtype)
    u01 = 1.0 - torch.pow((1.0 - x).clamp(min=eps), k_t)  # [B, T]
    # map to [-1,1]
    u = 2.0 * u01 - 1.0

    # --- Weighting: emphasize low pitches (lowest compression), relax high pitches (highest compression) ---
    # w ~ 1 at x=0, w ~ w_min at x=1
    le = torch.tensor(float(low_emphasis), device=e.device, dtype=x.dtype)
    w = torch.pow((1.0 - x).clamp(min=0.0), le)
    w = float(w_min) + (1.0 - float(w_min)) * w  # [B, T]

    # --- Huber (smooth L1) position loss, elementwise then weighted ---
    d = e_f - u
    ad = torch.abs(d)
    delta = float(huber_delta)

    huber = torch.where(
        ad < delta,
        0.5 * (d * d) / max(delta, eps),
        ad - 0.5 * delta
    )
    loss_pos = (w * huber).mean()

    # --- Optional contour agreement in warped space ---
    if contour_weight > 0.0:
        de = torch.diff(e_f, dim=1)  # [B, T-1]
        du = torch.diff(u, dim=1)    # [B, T-1]

        contour = torch.square(
            torch.maximum(
                1.0 - de * du,
                torch.zeros_like(de)
            )
        )

        w_mid = 0.5 * (w[:, 1:] + w[:, :-1])  # [B, T-1]
        loss_contour = (w_mid * contour).mean()

        return loss_pos + float(contour_weight) * loss_contour

    return loss_pos

def companded_warp_loss_knee(
    pitches: Tensor,
    e: Tensor,
    # --- shape of the non-uniform mapping ---
    high_k: float = 5,             # >1 : stronger compression in the HIGH region (after knee)
    knee_x0: float = 0.75,           # where "high compression" starts (0..1). LOWER => more high buttons compress
    knee_width: float = 0.02,        # smoothness of the knee (smaller = sharper transition)
    low_gamma: float = 1.0,          # <=1 : expand LOW region (1.0 = linear; 0.7 gives more resolution at low)
    # --- weighting & extras ---
    low_emphasis: float = 2.0,       # >1 : weight low pitches more, weight high pitches less
    w_min: float = 0.10,
    contour_weight: float = 0.25,
    huber_delta: float = 0.10,
    eps: float = 1e-6
) -> Tensor:
    """
    Asymmetric "control density" loss with a smooth knee:

      pitch (0..127) -> x in [0,1]
      LOW part (x <= x0): u01_low = x0 * (x/x0)^low_gamma   (more resolution at low if low_gamma<1)
      HIGH part (x > x0): u01_high = x0 + (1-x0) * (1 - (1-t)^high_k),  t=(x-x0)/(1-x0)
      Blend both with a sigmoid mask around x0 for smoothness.

    Then:
      - Align e to target u in [-1,1] with weighted Huber
      - Optional contour agreement in the warped space
    """

    # --- Align pitch length with e length (often pitches [B, T+1], e [B, T]) ---
    if pitches.size(1) == e.size(1) + 1:
        p = pitches[:, 1:]
    else:
        p = pitches
        T = min(p.size(1), e.size(1))
        p = p[:, :T]
        e = e[:, :T]

    if e.size(1) < 2:
        return torch.zeros(1, device=e.device)

    p = p.float()
    e_f = e.float()

    # --- Normalize MIDI pitch (0..127) to x in [0,1] ---
    x = (p / 127.0).clamp(0.0, 1.0)  # [B, T]

    # --- parameters as tensors on correct device/dtype ---
    x0 = torch.tensor(float(knee_x0), device=e.device, dtype=x.dtype).clamp(min=eps, max=1.0 - eps)
    k_t = torch.tensor(float(high_k), device=e.device, dtype=x.dtype)
    g_t = torch.tensor(float(low_gamma), device=e.device, dtype=x.dtype).clamp(min=eps)
    tau = torch.tensor(float(knee_width), device=e.device, dtype=x.dtype).clamp(min=eps)

    one_minus_x0 = (1.0 - x0).clamp(min=eps)

    # --- LOW branch (scaled so u01_low(x0)=x0) ---
    # u01_low = x0 * (x/x0)^gamma  for x<=x0
    x_over_x0 = (x / x0).clamp(0.0, 1.0)
    u01_low = x0 * torch.pow(x_over_x0, g_t)

    # --- HIGH branch (also continuous at x0, u01_high(x0)=x0, u01_high(1)=1) ---
    t = ((x - x0) / one_minus_x0).clamp(0.0, 1.0)
    u01_high = x0 + one_minus_x0 * (1.0 - torch.pow((1.0 - t).clamp(min=eps), k_t))

    # --- smooth blend around the knee ---
    m = torch.sigmoid((x - x0) / tau)  # ~0 below x0, ~1 above x0
    u01 = (1.0 - m) * u01_low + m * u01_high

    # map to [-1, 1] target for e
    u = 2.0 * u01 - 1.0  # [B, T]

    # --- Weighting: emphasize low (more control), relax high (more freedom) ---
    le = torch.tensor(float(low_emphasis), device=e.device, dtype=x.dtype)
    w = torch.pow((1.0 - x).clamp(min=0.0), le)          # 1 at low, -> 0 at high
    w = float(w_min) + (1.0 - float(w_min)) * w          # keep a floor

    # --- Weighted Huber (smooth L1) position loss ---
    d = e_f - u
    ad = torch.abs(d)
    delta = float(huber_delta)

    huber = torch.where(
        ad < delta,
        0.5 * (d * d) / max(delta, eps),
        ad - 0.5 * delta
    )
    loss_pos = (w * huber).mean()

    # --- Optional contour agreement in warped space ---
    if contour_weight > 0.0:
        de = torch.diff(e_f, dim=1)  # [B, T-1]
        du = torch.diff(u, dim=1)    # [B, T-1]

        contour = torch.square(torch.relu(1.0 - de * du))
        w_mid = 0.5 * (w[:, 1:] + w[:, :-1])  # [B, T-1]
        loss_contour = (w_mid * contour).mean()

        return loss_pos + float(contour_weight) * loss_contour

    return loss_pos

def companded_warp_loss(
    pitches: Tensor,
    e: Tensor,
    num_buttons: int = 18,         # total buttons (e quantized into these)
    high_buttons: int = 5,         # how many TOP buttons should cover the "high register"
    knee_x0: float = 0.70,         # pitch knee in x=[0,1] where high register begins (LOWER -> more pitches compressed)
    knee_width: float = 0.02,      # smoothness of transition around knee_x0
    low_gamma: float = 0.85,       # <1 expands low region more (less compression at low)
    high_gamma: float = 2.0,       # >1 spreads high pitches across top buttons (prevents only-top saturation)
    low_emphasis: float = 1.5,     # >0 weights low pitches more; higher => more low control
    w_min: float = 0.20,           # floor weight for high region (DON'T set too low or high bins collapse)
    contour_weight: float = 0.25,  # contour agreement in warped space
    huber_delta: float = 0.10,
    eps: float = 1e-6
) -> Tensor:
    """
    Asymmetric non-uniform mapping with explicit bin allocation for the high register.

    pitch p (0..127) -> x in [0,1]
    Define knee_x0 in pitch space, and knee_y0 in latent space so that x >= knee_x0
    is mapped only into the top 'high_buttons' bins.

    This makes compression apply to multiple top buttons, not only the highest one.
    """

    # Align shapes: if pitches is [B, T+1], use pitches[:, 1:] to match e [B, T]
    if pitches.size(1) == e.size(1) + 1:
        p = pitches[:, 1:].float()
    else:
        p = pitches.float()
        min_len = min(p.size(1), e.size(1))
        p = p[:, :min_len]
        e = e[:, :min_len]

    if e.size(1) < 2:
        return torch.zeros(1, device=e.device)

    e_f = e.float()

    # Normalize MIDI pitch to x in [0,1]
    x = (p / 127.0).clamp(0.0, 1.0)

    # -------------------------
    # Latent knee y0: boundary of top 'high_buttons' buttons in u01 space
    # Quantizer centers in u01 are at i/(num_buttons-1). Boundaries at (i+0.5)/(num_buttons-1).
    # Boundary below the top 'high_buttons' starts between buttons (N-high_buttons-1) and (N-high_buttons).
    # -------------------------
    N = float(max(int(num_buttons), 2))
    hb = float(max(min(int(high_buttons), int(num_buttons) - 1), 1))
    denom = max(N - 1.0, 1.0)

    knee_y0 = (N - hb - 0.5) / denom
    knee_y0 = float(max(min(knee_y0, 1.0 - 1e-4), 1e-4))

    # Params as tensors
    x0 = torch.tensor(float(knee_x0), device=e.device, dtype=x.dtype).clamp(min=eps, max=1.0 - eps)
    y0 = torch.tensor(float(knee_y0), device=e.device, dtype=x.dtype).clamp(min=eps, max=1.0 - eps)
    tau = torch.tensor(float(knee_width), device=e.device, dtype=x.dtype).clamp(min=eps)

    gL = torch.tensor(float(low_gamma), device=e.device, dtype=x.dtype).clamp(min=eps)
    gH = torch.tensor(float(high_gamma), device=e.device, dtype=x.dtype).clamp(min=eps)

    # LOW branch: map [0, x0] -> [0, y0]
    x_over_x0 = (x / x0).clamp(0.0, 1.0)
    u01_low = y0 * torch.pow(x_over_x0, gL)

    # HIGH branch: map [x0, 1] -> [y0, 1]
    t = ((x - x0) / (1.0 - x0).clamp(min=eps)).clamp(0.0, 1.0)
    u01_high = y0 + (1.0 - y0) * torch.pow(t, gH)

    # Smooth blend around knee
    m = torch.sigmoid((x - x0) / tau)  # ~0 below knee, ~1 above
    u01 = (1.0 - m) * u01_low + m * u01_high

    # Map to [-1, 1] target
    target_e = u01 * 2.0 - 1.0

    # Weighting: emphasize low pitches, de-emphasize high pitches (but keep floor w_min)
    le = torch.tensor(float(low_emphasis), device=e.device, dtype=x.dtype)
    w = torch.pow((1.0 - x).clamp(min=0.0), le)           # 1 at low, 0 at high
    w = float(w_min) + (1.0 - float(w_min)) * w           # keep floor

    # Weighted Huber (smooth L1) position loss
    d = e_f - target_e
    ad = torch.abs(d)
    delta = float(huber_delta)

    huber = torch.where(
        ad < delta,
        0.5 * (d * d) / max(delta, eps),
        ad - 0.5 * delta
    )
    loss_pos = (w * huber).mean()

    # Optional contour in warped space
    if contour_weight > 0:
        de = torch.diff(e_f, dim=1)
        dt = torch.diff(target_e, dim=1)
        loss_contour = torch.square(torch.relu(1.0 - de * dt))

        w_mid = 0.5 * (w[:, 1:] + w[:, :-1])
        loss_contour = (w_mid * loss_contour).mean()

        return loss_pos + float(contour_weight) * loss_contour

    return loss_pos
