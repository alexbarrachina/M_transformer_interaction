#===================================================================================================
# Monster Genie tension_extractor.py Python module
#
# Beatless tonal-tension feature extraction for the AE_style_tensions model.
#
# This module implements the TIV/TIS (Tonal Interval Space / Tonal Interval Vector)
# harmonic-state extractor described in docs/Tonal_tension_FiLM_architecture_plan.md
# (Modules 3-7 and 10). The tonal-tension formulation follows:
#
#   Ebrahimzadeh, Bernardes, Stober (2025), "Explicit Tonal Tension Conditioning
#   via Dual-Level Beam Search for Symbolic Music Generation", CMMR 2025
#   (see docs/tension-beamsearch-main/tension/tonal_tension.py), which itself
#   adapts Bernardes et al. (2016) and Navarro-Caceres et al. (2020).
#
# The extractor is intentionally beatless / chord-label-free: it works from the
# raw pitch stream alone (short note-count window for the local sonority TIV;
# leaky-integrator chroma + profile-correlation HMM for the key, per docs
# "Real-Time MIDI Key Detection"), so it needs no bars, no beat grid and no
# external chord/key annotations (the GiantMIDI-style corpus this project
# targets has none of these reliably).
#
# It is used by BOTH training (train_style_tensions.py: per-note features over the
# target window) and inference (interaction_buttons_style_tensions.py: the
# RealtimeTensionExtractor + command_to_target mapper).
#
# Copyright 2025 Alex Barrachina
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.'''
#===================================================================================================

import math
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
#  TIV / TIS constants                                                         #
# --------------------------------------------------------------------------- #
# Bernardes et al. (2016) TIV weights for interval classes k = 1..6.
_TIV_W: List[float] = [2.0, 11.0, 17.0, 16.0, 19.0, 7.0]
# Norm of the TIV of a single pitch class = sqrt(sum w_k^2) (== 32.8633..., the
# c_max used in the reference get_dissonance). We normalise realified TIVs by it
# so every TIV component / norm lies in [-1, 1] and the dissonance is in [0, 1].
TIV_MAX_NORM: float = math.sqrt(sum(w * w for w in _TIV_W))

# Krumhansl-Kessler key profiles (major / minor), rotated per tonic to build the
# 24 key chroma templates (used for the key/function TIV prototypes).
_MAJOR_PROFILE: List[float] = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE: List[float] = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# Key-detection emission profiles (docs "Real-Time MIDI Key Detection"):
# Temperley's modified profiles boost the 4th/7th scale degrees so closely
# related keys sharing 6 of 7 notes (e.g. C vs G major) separate cleanly, and
# remove the K-S minor-mode bias; per Napoles Lopez's symbolic-key HMM the best
# emission combination is Temperley for MAJOR keys and Sapp for MINOR keys.
_TEMPERLEY_MAJOR: List[float] = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]
_SAPP_MINOR: List[float] = [2.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.5, 0.5]
# 'genie': project-tuned variant widening the two weakest neighbor contrasts —
# major: lower the flat-seventh weight (deg 10) so the subdominant (7th-vs-b7th)
# margin widens; minor: raise the harmonic-minor leading tone (deg 11, chromatic
# in the relative major) and lower the subtonic, sharpening relative-mode
# discrimination. Verified against the margin/validation harness before adoption.
_GENIE_MAJOR: List[float] = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.0, 4.0]
_GENIE_MINOR: List[float] = [2.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.25, 1.25]
_PROFILE_SETS: Dict[str, Tuple[List[float], List[float]]] = {
    'kk': (_MAJOR_PROFILE, _MINOR_PROFILE),
    'temperley': (_TEMPERLEY_MAJOR, _SAPP_MINOR),
    'genie': (_GENIE_MAJOR, _GENIE_MINOR),
}

# Key-class mapping (documented, matches harmony_extractor.estimate_key_from_pitches):
#   0..11  -> major keys by tonic pitch class (0 = C major)
#   12..23 -> minor keys by tonic pitch class (12 = C minor, 21 = A minor)
NUM_KEYS: int = 24

# Triad templates (semitone offsets from the key tonic) for the tonal functions.
# Major key: I / IV / V. Minor key: i / iv / V (harmonic-minor major dominant).
_MAJOR_TONIC: List[int] = [0, 4, 7]
_MAJOR_SUBDOM: List[int] = [5, 9, 0]
_MAJOR_DOM: List[int] = [7, 11, 2]
_MINOR_TONIC: List[int] = [0, 3, 7]
_MINOR_SUBDOM: List[int] = [5, 8, 0]
_MINOR_DOM: List[int] = [7, 11, 2]


# --------------------------------------------------------------------------- #
#  Feature-vector layout (kept in sync with params.TENSION_FEATURE_DIM)        #
# --------------------------------------------------------------------------- #
# The per-note tension feature is a single flat real vector so it can drive the
# FiLM conditioner directly. Indices are exported so the model / inference agree.
TF_TIV = slice(0, 12)            # normalised realified short-window TIV
TF_KEY_POST = slice(12, 36)      # 24-key posterior
TF_KEY_CONF = 36                 # top key probability
TF_KEY_ENTROPY = 37              # normalised key entropy
TF_HARMONIC_CHANGE = 38          # euclidean TIV change (same-level)
TF_DIST_KEY = 39                 # angular distance sonority -> expected key
TF_DIST_TONIC = 40               # angular distance sonority -> tonic function
TF_DIST_SUBDOM = 41              # angular distance sonority -> subdominant
TF_DIST_DOM = 42                 # angular distance sonority -> dominant
TF_DISSONANCE = 43               # 1 - ||TIV|| / ||TIV_max||
TF_MOD_PRESSURE = 44             # modulation pressure heuristic
TF_TENSION_SCALAR = 45           # weighted tension scalar
TF_TENSION_SLOPE = 46            # first difference of the tension scalar
TF_RESOLUTION = 47               # resolution (movement toward stability)
TENSION_FEATURE_DIM: int = 48

# Default tension-scalar component weights (plan Module 7; tune on the corpus).
_DEFAULT_WEIGHTS: Dict[str, float] = {
    'harmonic_change': 1.0,
    'distance_to_key': 1.0,
    'tiv_dissonance': 1.0,
    'key_entropy': 0.5,
    'modulation_pressure': 0.8,
}


# --------------------------------------------------------------------------- #
#  Precomputed projection / template tensors (built once, float32, on CPU)     #
# --------------------------------------------------------------------------- #
def _build_dft_matrices() -> tuple:
    """Real/imag DFT bases for the six weighted TIV coefficients (k = 1..6)."""
    cos = torch.zeros(6, 12)
    neg_sin = torch.zeros(6, 12)
    for k in range(6):          # interval class k+1
        for n in range(12):
            ang = 2.0 * math.pi * (k + 1) * n / 12.0
            cos[k, n] = math.cos(ang)
            neg_sin[k, n] = -math.sin(ang)   # imag part of exp(-j*ang)
    return cos, neg_sin


_COS, _NEG_SIN = _build_dft_matrices()
_TIV_WEIGHTS = torch.tensor(_TIV_W, dtype=torch.float32)   # [6]


def chroma_to_tiv(chroma: Tensor) -> Tensor:
    """Project a (batch of) 12-d chroma vector(s) into the normalised realified
    Tonal Interval Vector space. Input [..., 12] -> output [..., 12] where the
    12 dims are [re(T1..T6), im(T1..T6)] / TIV_MAX_NORM (norm <= 1). Pure tensor
    ops (no python loop over the sequence)."""
    modc = chroma.sum(dim=-1, keepdim=True).clamp_min(1e-8)          # [...,1]
    re = torch.matmul(chroma, _COS.to(chroma).t()) * _TIV_WEIGHTS.to(chroma)  # [...,6]
    im = torch.matmul(chroma, _NEG_SIN.to(chroma).t()) * _TIV_WEIGHTS.to(chroma)
    tiv = torch.cat([re, im], dim=-1) / modc                         # [...,12]
    return tiv / TIV_MAX_NORM


def _build_key_and_function_tivs() -> tuple:
    """Precompute the 24-key TIVs and the per-key tonic/subdominant/dominant
    function TIVs (all in the normalised realified TIV space)."""
    key_chroma = torch.zeros(NUM_KEYS, 12)
    tonic_chroma = torch.zeros(NUM_KEYS, 12)
    subdom_chroma = torch.zeros(NUM_KEYS, 12)
    dom_chroma = torch.zeros(NUM_KEYS, 12)
    maj = torch.tensor(_MAJOR_PROFILE)
    mnr = torch.tensor(_MINOR_PROFILE)
    for root in range(12):
        # major key `root`
        key_chroma[root] = torch.roll(maj, shifts=root)
        for iv in _MAJOR_TONIC:
            tonic_chroma[root, (root + iv) % 12] = 1.0
        for iv in _MAJOR_SUBDOM:
            subdom_chroma[root, (root + iv) % 12] = 1.0
        for iv in _MAJOR_DOM:
            dom_chroma[root, (root + iv) % 12] = 1.0
        # minor key `root` (index root + 12)
        m = root + 12
        key_chroma[m] = torch.roll(mnr, shifts=root)
        for iv in _MINOR_TONIC:
            tonic_chroma[m, (root + iv) % 12] = 1.0
        for iv in _MINOR_SUBDOM:
            subdom_chroma[m, (root + iv) % 12] = 1.0
        for iv in _MINOR_DOM:
            dom_chroma[m, (root + iv) % 12] = 1.0
    return (chroma_to_tiv(key_chroma), chroma_to_tiv(tonic_chroma),
            chroma_to_tiv(subdom_chroma), chroma_to_tiv(dom_chroma))


_KEY_TIV, _TONIC_TIV, _SUBDOM_TIV, _DOM_TIV = _build_key_and_function_tivs()  # each [24,12]


@lru_cache(maxsize=4)
def _profile_matrix_z(profiles: str = 'genie') -> Tensor:
    """Z-scored key-emission profiles [24,12] for correlation-based evidence.
    Profile correlation is markedly more reliable on noisy chromas than
    TIV-angle template matching (which loses mode discrimination as the chroma
    flattens), so the key *detector* uses correlation while every tension
    distance stays in TIV space."""
    maj_p, min_p = _PROFILE_SETS[profiles]
    prof = torch.zeros(NUM_KEYS, 12)
    maj = torch.tensor(maj_p)
    mnr = torch.tensor(min_p)
    for root in range(12):
        prof[root] = torch.roll(maj, shifts=root)
        prof[root + 12] = torch.roll(mnr, shifts=root)
    return (prof - prof.mean(dim=-1, keepdim=True)) / prof.std(dim=-1, keepdim=True)


def _key_correlation(chroma: Tensor, profiles: str = 'genie') -> Tensor:
    """Pearson correlation of (a batch of) chroma vectors [...,12] against the
    24 z-scored key profiles -> [...,24] in [-1, 1] (0 for empty chroma)."""
    c = chroma - chroma.mean(dim=-1, keepdim=True)
    std = c.std(dim=-1, keepdim=True).clamp_min(1e-8)
    cz = c / std
    return torch.matmul(cz, _profile_matrix_z(profiles).to(chroma).t()) / 12.0


# --------------------------------------------------------------------------- #
#  Neighbor keys + Signature of Fifths (docs "Real-Time MIDI Key Detection")   #
# --------------------------------------------------------------------------- #
def neighbor_keys(key: int) -> Dict[str, int]:
    """The four ring-1 neighbors of a key class (0..23): relative, parallel,
    dominant, subdominant (mode-preserving for dom/subdom)."""
    root, minor = key % 12, key >= 12
    if minor:
        return {'relative': (root + 3) % 12,
                'parallel': root,
                'dominant': (root + 7) % 12 + 12,
                'subdominant': (root + 5) % 12 + 12}
    return {'relative': (root + 9) % 12 + 12,
            'parallel': root + 12,
            'dominant': (root + 7) % 12,
            'subdominant': (root + 5) % 12}


# Position of each pitch class on the circle of fifths (C=0, G=1, ..., F=11)
# and its angle (30 deg per fifth step).
_FIFTHS_INDEX: List[int] = [(7 * j) % 12 for j in range(12)]   # pc -> fifths pos
_FIFTHS_ANGLE = torch.tensor([(7 * j) % 12 for j in range(12)], dtype=torch.float32) \
    * (math.pi / 6.0)                                           # [12] radians
# Fifths position of each key class's tonic (0..23).
_KEY_FIFTHS: List[int] = [_FIFTHS_INDEX[k % 12] for k in range(NUM_KEYS)]


def signature_of_fifths(chroma: Tensor) -> Dict[str, float]:
    """Signature-of-Fifths descriptors of a chroma vector [12] (reference doc):
    pitch-class masses are placed as polar vectors on the circle of fifths.

      - MDASF (main directed axis): the direction that maximizes the mass
        difference between its two hemispheres. Because the 7 diatonic notes of
        any key occupy a contiguous arc of fifths, this axis points at the
        center of the diatonic mass; a dominant/subdominant shift moves it by
        exactly +/-1 fifth step — independent of key profiles.
      - CVSF (characteristic vector): the plain vector sum. Its signed angular
        deviation from the MDASF separates the two relative keys sharing the
        signature: tonic/dominant emphasis of the MAJOR key pulls the CVSF
        flat-ward of the diatonic center (negative), the relative MINOR's
        tonic/dominant pull it sharp-ward (positive).

    Returns dict with:
      fifths_center: diatonic-mass center in fifths steps 0..11 (float, from
                     the MDASF direction; e.g. 2.0 = 'D' = center of C major)
      major_key / minor_key: the two candidate key classes for that signature
      mode_score: signed CVSF deviation in fifths steps (<0 major, >0 minor)
      cvsf_norm: CVSF magnitude / total mass (0..1, signature sharpness)
    """
    r = chroma.float().clamp_min(0.0)
    total = float(r.sum())
    if total <= 0.0:
        return {'fifths_center': 0.0, 'major_key': 0, 'minor_key': 21,
                'mode_score': 0.0, 'cvsf_norm': 0.0}
    ang = _FIFTHS_ANGLE.to(r)
    x = float((r * torch.cos(ang)).sum())
    y = float((r * torch.sin(ang)).sum())
    # MDASF via candidate directed axes every 15 deg: maximize right-vs-left
    # hemisphere mass difference.
    best_theta, best_score = 0.0, -1e9
    for a in range(24):
        theta = a * (math.pi / 12.0)
        score = float((r * torch.cos(ang - theta)).sign().mul(r).sum())
        # (sign(cos) * r summed == right-hemisphere mass - left-hemisphere mass)
        if score > best_score:
            best_score, best_theta = score, theta
    fifths_center = (best_theta / (math.pi / 6.0)) % 12.0
    # candidate keys: major tonic sits 2 fifths flat-ward of the diatonic
    # center (C major spans F..B centered on D); relative minor 1 sharp-ward.
    maj_f = int(round(fifths_center - 2.0)) % 12
    min_f = int(round(fifths_center + 1.0)) % 12
    major_pc = (7 * maj_f) % 12
    minor_pc = (7 * min_f) % 12
    # signed angular deviation of the CVSF from the axis, in fifths steps
    cvsf_theta = math.atan2(y, x)
    dev = cvsf_theta - best_theta
    while dev > math.pi:
        dev -= 2.0 * math.pi
    while dev < -math.pi:
        dev += 2.0 * math.pi
    mode_score = dev / (math.pi / 6.0)
    return {'fifths_center': fifths_center,
            'major_key': major_pc,
            'minor_key': minor_pc + 12,
            'mode_score': mode_score,
            'cvsf_norm': math.hypot(x, y) / total}


# --------------------------------------------------------------------------- #
#  Circle-of-fifths key-transition prior (HMM forward filter)                  #
# --------------------------------------------------------------------------- #
# The key is tracked with a Bayesian forward filter over the 24 keys. The
# transition prior follows the ring-of-neighbors topology from the reference
# document ("Real-Time MIDI Key Detection"): relative to any key, the other 23
# keys are grouped into 9 concentric proximity rings on the circle of fifths
# (ring 1 = dominant/subdominant/relative/parallel, ..., ring 8 = the tritone
# key), and p(target in ring s_g) = trans_alpha^((G-1) - s_g), G = 9. With the
# default trans_alpha = 10 this yields p(stay) ~= 0.69 and ~0.069 per direct
# neighbor: modulations to related keys are cheap (the filter follows a real
# key change within a phrase) while distant keys are exponentially penalized
# (ornamental chromaticism cannot drag the state across the circle).
#
# Ring tables (semitone offset target_root - ref_root, relative to a MAJOR ref;
# from the reference document's proximity-group table for C major):
_RING_MAJ_TO_MAJ: Dict[int, int] = {0: 0, 7: 1, 5: 1, 2: 3, 10: 3, 9: 3, 3: 3,
                                    4: 4, 8: 4, 11: 5, 1: 5, 6: 8}
_RING_MAJ_TO_MIN: Dict[int, int] = {9: 1, 0: 1, 2: 2, 4: 2, 5: 2, 7: 2,
                                    10: 4, 11: 4, 3: 6, 6: 6, 1: 7, 8: 7}
_NUM_RINGS: int = 9


def _key_ring(i: int, j: int) -> int:
    """Proximity ring (0..8) between key classes i, j (0..23). Symmetric.
    Major->major and minor->minor use the fifths table on the root offset;
    major<->minor uses the mixed table (relative/parallel = ring 1, etc.)."""
    i_min, j_min = i >= 12, j >= 12
    off = (j % 12 - i % 12) % 12
    if not i_min and not j_min:
        return _RING_MAJ_TO_MAJ[off]
    if i_min and j_min:
        return _RING_MAJ_TO_MAJ[off]
    if not i_min and j_min:
        return _RING_MAJ_TO_MIN[off]
    return _RING_MAJ_TO_MIN[(-off) % 12]   # minor -> major (symmetric)


@lru_cache(maxsize=8)
def _key_transition_matrix(trans_alpha: float = 10.0) -> Tensor:
    """[24,24] row-stochastic key-transition prior from the ring topology:
    trans[i,j] ~ trans_alpha^((G-1) - ring(i,j)), rows normalized. Symmetric
    (doubly stochastic), so the uniform posterior is invariant in silence."""
    w = torch.zeros(NUM_KEYS, NUM_KEYS)
    for i in range(NUM_KEYS):
        for j in range(NUM_KEYS):
            w[i, j] = float(trans_alpha) ** ((_NUM_RINGS - 1) - _key_ring(i, j))
    return w / w.sum(dim=-1, keepdim=True)


def _key_filter_step(posterior: Tensor, lik: Tensor, trans: Tensor) -> Tensor:
    """One HMM forward-filter step: predict with the transition prior, then
    update with the (unnormalized) per-frame key likelihood [24]."""
    pred = torch.matmul(posterior, trans)
    post = pred * lik
    return post / post.sum(dim=-1, keepdim=True).clamp_min(1e-12)


# --------------------------------------------------------------------------- #
#  Distance helpers                                                            #
# --------------------------------------------------------------------------- #
def tiv_euclidean(a: Tensor, b: Tensor) -> Tensor:
    """Same-level distance (sonority-to-sonority). Inputs [...,12]."""
    return (a - b).norm(dim=-1)


def tiv_angle(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """Cross-level angular distance (sonority-to-key / function). Inputs [...,12]."""
    dot = (a * b).sum(dim=-1)
    denom = a.norm(dim=-1) * b.norm(dim=-1) + eps
    cos = (dot / denom).clamp(-1.0, 1.0)
    return torch.arccos(cos)


# --------------------------------------------------------------------------- #
#  Core feature extraction                                                     #
# --------------------------------------------------------------------------- #
def _windowed_chroma(cum: Tensor, window: int) -> Tensor:
    """Causal moving-sum chroma over the last `window` notes from a cumulative
    one-hot pitch-class sum `cum` [T,12]. Vectorised (no per-note loop)."""
    T = cum.shape[0]
    if window >= T:
        return cum
    shifted = torch.zeros_like(cum)
    shifted[window:] = cum[:-window]
    return cum - shifted


def _note_decay_rate(pitches: Tensor, tau_low: float, tau_high: float) -> Tensor:
    """Per-note decay rate 1/tau(p) [1/s]: bass notes persist longer than
    treble ones (pitch-parametrized 'virtual sustain', reference doc a(p))."""
    p_norm = ((pitches.float() - 21.0) / 87.0).clamp(0.0, 1.0)   # A0..C8 -> 0..1
    tau = tau_low + (tau_high - tau_low) * p_norm
    return 1.0 / tau.clamp_min(1e-3)


def _register_weight(pitches: Tensor, bass_weight: float) -> Tensor:
    """Per-note register weight: `bass_weight` at/below C3 (48) ramping to 1.0
    at/above C5 (72). Tonics and chord roots live in the bass, so weighting the
    low register sharpens the tonal-hierarchy evidence (the only thing that
    separates relative major/minor, which share every pitch class)."""
    ramp = ((72.0 - pitches.float()) / 24.0).clamp(0.0, 1.0)
    return 1.0 + (float(bass_weight) - 1.0) * ramp


def _leaky_chroma(pitches: Tensor, onsets: Tensor, velocities: Tensor,
                  tau_low: float, tau_high: float,
                  bass_weight: float = 2.0,
                  cadence_gap: float = 1.0,
                  cadence_boost: float = 2.0) -> Tensor:
    """Leaky-integrator pitch-class profile [T,12] ('virtual sustain'): each
    note contributes velocity-weighted energy that decays exponentially in
    time, v(t) = vel * exp(-a(p) * (t - onset)), summed per pitch class. This
    replaces the hard note-count window for KEY evidence: recent/loud notes
    dominate (responsive to modulation) while older context fades smoothly
    (stable against ornaments). Two tonal-hierarchy weightings on top:
      - register: bass notes weighted by up to `bass_weight` (_register_weight);
      - cadence: a note followed by an inter-onset gap > `cadence_gap` seconds
        (phrase-final ~ cadence note) is boosted by `cadence_boost` from the
        NEXT note onward (i.e. once the gap is observed — causal, matching the
        realtime tracker exactly).
    Fully vectorized ([T,T] weight matrix)."""
    T = int(pitches.shape[0])
    valid = (pitches >= 0) & (pitches < 128)
    pc = torch.where(valid, pitches % 12, torch.zeros_like(pitches))
    onehot = torch.zeros(T, 12, dtype=torch.float32, device=pitches.device)
    onehot.scatter_(1, pc.long().unsqueeze(1), valid.float().unsqueeze(1))
    rate = _note_decay_rate(pitches, tau_low, tau_high)              # [T]
    dt = (onsets.unsqueeze(0) - onsets.unsqueeze(1)).clamp_min(0.0)  # [i,t]
    base_w = velocities.float() * _register_weight(pitches, bass_weight)
    w = base_w.unsqueeze(1) * torch.exp(-rate.unsqueeze(1) * dt)
    idx = torch.arange(T, device=pitches.device)
    causal = idx.unsqueeze(1) <= idx.unsqueeze(0)
    w = w * causal                                                   # note i active for t >= i
    if T > 1 and cadence_boost != 1.0:
        gaps = onsets[1:] - onsets[:-1]                              # [T-1]
        b = torch.ones(T, device=pitches.device)
        b[:-1] = torch.where(gaps > float(cadence_gap),
                             torch.full_like(gaps, float(cadence_boost)),
                             torch.ones_like(gaps))
        strict_upper = idx.unsqueeze(1) < idx.unsqueeze(0)           # t > i
        w = w * (1.0 + (b - 1.0).unsqueeze(1) * strict_upper)
    return torch.matmul(w.t(), onehot)                               # [T,12]


@torch.no_grad()
def extract_tension_features(
    pitches: Tensor,
    short_window: int = 8,
    key_alpha: float = 2.0,
    trans_alpha: float = 100.0,
    tau_low: float = 8.0,
    tau_high: float = 3.0,
    profiles: str = 'genie',
    onsets: Optional[Tensor] = None,
    velocities: Optional[Tensor] = None,
    note_dt: float = 0.25,
    bass_weight: float = 2.0,
    cadence_gap: float = 1.0,
    cadence_boost: float = 2.0,
    weights: Optional[Dict[str, float]] = None,
    init_posterior: Optional[Tensor] = None,
) -> Tensor:
    """Compute the per-note tonal-tension feature matrix from a pitch sequence.

    Args:
        pitches: LongTensor [T] of MIDI pitches (values >= 128, e.g. PAD, are
                 ignored for chroma via pitch-class 0..11 masking).
        short_window: note-count window for the local sonority TIV.
        key_alpha: gain of the per-frame key-correlation likelihood
                 exp(key_alpha * correlation).
        trans_alpha: ring-decay ratio of the circle-of-fifths transition prior
                 (higher = stronger penalty per proximity ring; the reference doc\n                 uses 10 at ~10-20 Hz tick updates -- per-note updates need ~100).
        tau_low / tau_high: leaky-integrator decay time constants [s] at the
                 bottom / top of the piano range (bass persists longer).
        profiles: key-emission profile set ('temperley' = Temperley major +
                 Sapp minor, per the reference doc; 'kk' = Krumhansl-Kessler).
        onsets: optional [T] note-onset times in seconds (monotonic). Defaults
                 to a uniform note_dt grid when timing is unavailable.
        velocities: optional [T] MIDI velocities (defaults to 64).
        note_dt: fallback inter-onset spacing [s] when onsets is None.
        bass_weight: register weight of bass notes in the key evidence
                 (tonal-hierarchy cue; separates relative major/minor).
        cadence_gap / cadence_boost: notes followed by an inter-onset gap
                 > cadence_gap [s] (phrase-final ~ cadence notes) get their key
                 evidence boosted by cadence_boost (causally, from the next
                 note onward).
        weights: optional override for the tension-scalar component weights.
        init_posterior: optional [NUM_KEYS] key posterior to start the filter
                 from (e.g. a pre-analyzed prompt prior); uniform if None.

    Returns:
        FloatTensor [T, TENSION_FEATURE_DIM] with the layout documented above.
    """
    w = dict(_DEFAULT_WEIGHTS)
    if weights is not None:
        w.update(weights)

    device = pitches.device
    T = int(pitches.shape[0])
    out = torch.zeros(T, TENSION_FEATURE_DIM, dtype=torch.float32, device=device)
    if T == 0:
        return out

    # --- chroma (only real MIDI pitches 0..127 contribute a pitch class) ---
    valid = (pitches >= 0) & (pitches < 128)
    pc = torch.where(valid, pitches % 12, torch.zeros_like(pitches))
    onehot = torch.zeros(T, 12, dtype=torch.float32, device=device)
    onehot.scatter_(1, pc.long().unsqueeze(1), valid.float().unsqueeze(1))
    cum = onehot.cumsum(dim=0)

    short_chroma = _windowed_chroma(cum, short_window)

    # --- leaky-integrator chroma for the key evidence (time/velocity aware) ---
    if onsets is None:
        onsets = torch.arange(T, dtype=torch.float32, device=device) * float(note_dt)
    else:
        onsets = onsets.float().to(device)
    if velocities is None:
        velocities = torch.full((T,), 64.0, dtype=torch.float32, device=device)
    else:
        velocities = velocities.float().to(device)
    energy_chroma = _leaky_chroma(pitches, onsets, velocities, tau_low, tau_high,
                                  bass_weight, cadence_gap, cadence_boost)

    tiv_short = chroma_to_tiv(short_chroma)                          # [T,12]

    key_tiv = _KEY_TIV.to(tiv_short)                                 # [24,12]
    tonic_tiv = _TONIC_TIV.to(tiv_short)
    subdom_tiv = _SUBDOM_TIV.to(tiv_short)
    dom_tiv = _DOM_TIV.to(tiv_short)

    # --- key posterior: HMM forward filter over the 24 keys ---
    # Per-frame evidence is the profile correlation of the leaky-integrator
    # chroma (Temperley/Sapp emissions by default); the circle-of-fifths ring
    # transition prior makes related-key modulations cheap while exponentially
    # penalizing distant jumps. Stability comes from the integrator smoothing
    # + discriminative profiles; responsiveness from the soft ring transitions.
    # The recursion is inherently sequential; T is small and this runs on the
    # CPU dataloader, so a short python scan is acceptable and clearest.
    key_corr = _key_correlation(energy_chroma, profiles)               # [T,24]
    lik = torch.exp(float(key_alpha) * key_corr)                       # [T,24]
    trans = _key_transition_matrix(float(trans_alpha)).to(lik)
    posterior = torch.empty_like(lik)
    if init_posterior is not None:
        p = init_posterior.to(lik).clamp_min(0.0)
        p = p / p.sum().clamp_min(1e-12)
    else:
        p = torch.full((NUM_KEYS,), 1.0 / NUM_KEYS, dtype=lik.dtype, device=lik.device)
    for t in range(T):
        p = _key_filter_step(p, lik[t], trans)
        posterior[t] = p

    key_conf = posterior.max(dim=-1).values                           # [T]
    key_entropy = -(posterior * (posterior + 1e-8).log()).sum(dim=-1) / math.log(NUM_KEYS)

    # --- same-level harmonic change (short TIV vs previous short TIV) ---
    harmonic_change = torch.zeros(T, dtype=torch.float32, device=device)
    harmonic_change[1:] = tiv_euclidean(tiv_short[1:], tiv_short[:-1])

    # --- cross-level distances to key and tonal functions (posterior-weighted) ---
    expected_key_tiv = torch.matmul(posterior, key_tiv)               # [T,12]
    dist_key = tiv_angle(tiv_short, expected_key_tiv)                 # [T]

    ang_tonic = tiv_angle(tiv_short.unsqueeze(1), tonic_tiv.unsqueeze(0))    # [T,24]
    ang_subdom = tiv_angle(tiv_short.unsqueeze(1), subdom_tiv.unsqueeze(0))
    ang_dom = tiv_angle(tiv_short.unsqueeze(1), dom_tiv.unsqueeze(0))
    dist_tonic = (posterior * ang_tonic).sum(dim=-1)
    dist_subdom = (posterior * ang_subdom).sum(dim=-1)
    dist_dom = (posterior * ang_dom).sum(dim=-1)

    # --- dissonance (norm-based, direction per Navarro-Caceres 2020) ---
    dissonance = (1.0 - tiv_short.norm(dim=-1)).clamp(0.0, 1.0)

    # --- modulation pressure heuristic (plan Module 6) ---
    mod_pressure = (1.0 - key_conf) + key_entropy

    # --- tension scalar (weighted sum of interpretable components) ---
    tension_scalar = (
        w['harmonic_change'] * harmonic_change
        + w['distance_to_key'] * dist_key
        + w['tiv_dissonance'] * dissonance
        + w['key_entropy'] * key_entropy
        + w['modulation_pressure'] * mod_pressure
    )

    tension_slope = torch.zeros(T, dtype=torch.float32, device=device)
    tension_slope[1:] = tension_scalar[1:] - tension_scalar[:-1]

    # --- resolution score (movement toward stability; plan Module 7) ---
    d_conf = torch.zeros(T, dtype=torch.float32, device=device)
    d_conf[1:] = key_conf[1:] - key_conf[:-1]
    d_ent = torch.zeros(T, dtype=torch.float32, device=device)
    d_ent[1:] = key_entropy[1:] - key_entropy[:-1]
    d_tonic = torch.zeros(T, dtype=torch.float32, device=device)
    d_tonic[1:] = dist_tonic[1:] - dist_tonic[:-1]
    resolution = -tension_slope + d_conf - d_ent - d_tonic

    # --- assemble ---
    out[:, TF_TIV] = tiv_short
    out[:, TF_KEY_POST] = posterior
    out[:, TF_KEY_CONF] = key_conf
    out[:, TF_KEY_ENTROPY] = key_entropy
    out[:, TF_HARMONIC_CHANGE] = harmonic_change
    out[:, TF_DIST_KEY] = dist_key
    out[:, TF_DIST_TONIC] = dist_tonic
    out[:, TF_DIST_SUBDOM] = dist_subdom
    out[:, TF_DIST_DOM] = dist_dom
    out[:, TF_DISSONANCE] = dissonance
    out[:, TF_MOD_PRESSURE] = mod_pressure
    out[:, TF_TENSION_SCALAR] = tension_scalar
    out[:, TF_TENSION_SLOPE] = tension_slope
    out[:, TF_RESOLUTION] = resolution
    return out


def future_aggregate_features(feat: Tensor, pitches: Tensor,
                              horizon: int = 12) -> Tuple[Tensor, Tensor]:
    """Leak-free FiLM conditioning targets from realized per-note features.

    Conditioning the decoder on the realized feature of the *next note* leaks
    the answer (its chroma/TIV contain the predicted pitch class -> ~1.0 train
    accuracy and a useless steering channel at inference). Instead, position k
    is conditioned on the AGGREGATE over the next `horizon` notes
    [k .. k+horizon-1]: the mean of the scalar feature dims, the mean key
    posterior, the TIV of the summed window chroma, and the normalized window
    chroma. That is informative (where the harmony is heading over the coming
    phrase — not derivable from the past) but not decodable to a single note,
    and it matches the semantics of command_to_target exactly.

    Args:
        feat: [T, TENSION_FEATURE_DIM] realized per-note features.
        pitches: [T] the same pitch sequence the features came from.
        horizon: aggregation window in notes (truncated at the sequence end).

    Returns:
        (cond_feat [T, TENSION_FEATURE_DIM], cond_chroma [T, 12]).
    """
    T = int(feat.shape[0])
    if T == 0:
        return feat.clone(), torch.zeros(0, 12, dtype=torch.float32, device=feat.device)
    H = max(1, int(horizon))
    # forward-window mean of the features: rev-cumsum trick, fully vectorized
    csum = torch.cat([torch.zeros(1, feat.shape[1], device=feat.device),
                      feat.float().cumsum(dim=0)], dim=0)              # [T+1,F]
    idx_hi = torch.arange(T, device=feat.device) + H
    idx_hi = idx_hi.clamp(max=T)
    idx_lo = torch.arange(T, device=feat.device)
    counts = (idx_hi - idx_lo).clamp_min(1).unsqueeze(-1).float()
    cond_feat = (csum[idx_hi] - csum[idx_lo]) / counts                 # [T,F]
    # forward-window chroma (unnormalized sum -> TIV; normalized -> cond chroma)
    valid = (pitches >= 0) & (pitches < 128)
    pc = torch.where(valid, pitches % 12, torch.zeros_like(pitches))
    onehot = torch.zeros(T, 12, dtype=torch.float32, device=feat.device)
    onehot.scatter_(1, pc.long().unsqueeze(1), valid.float().unsqueeze(1))
    ccsum = torch.cat([torch.zeros(1, 12, device=feat.device),
                       onehot.cumsum(dim=0)], dim=0)                   # [T+1,12]
    win_chroma = ccsum[idx_hi] - ccsum[idx_lo]                         # [T,12]
    cond_feat[:, TF_TIV] = chroma_to_tiv(win_chroma)
    cond_chroma = win_chroma / win_chroma.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return cond_feat, cond_chroma


def short_chroma_from_pitches(pitches: Tensor, short_window: int = 8) -> Tensor:
    """Return the causal short-window chroma [T,12] (used for the soft PC-bias /
    visualization). Normalised to sum 1 per frame (0 where silent)."""
    T = int(pitches.shape[0])
    if T == 0:
        return torch.zeros(0, 12, dtype=torch.float32, device=pitches.device)
    valid = (pitches >= 0) & (pitches < 128)
    pc = torch.where(valid, pitches % 12, torch.zeros_like(pitches))
    onehot = torch.zeros(T, 12, dtype=torch.float32, device=pitches.device)
    onehot.scatter_(1, pc.long().unsqueeze(1), valid.float().unsqueeze(1))
    cum = onehot.cumsum(dim=0)
    ch = _windowed_chroma(cum, short_window)
    s = ch.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return ch / s


# --------------------------------------------------------------------------- #
#  Command -> target trajectory mapper (plan Module 9)                         #
# --------------------------------------------------------------------------- #
# High-level performer commands mapped to a target tension feature vector that
# drives FiLM at inference. Deterministic, configurable strength.
CMD_MAINTAIN = 'maintain'
CMD_ADD_TENSION = 'add_tension'
CMD_RESOLVE = 'resolve'
CMD_CHANGE_TONAL_CENTER = 'change_tonal_center'
TENSION_COMMANDS = (CMD_MAINTAIN, CMD_ADD_TENSION, CMD_RESOLVE, CMD_CHANGE_TONAL_CENTER)


def command_to_target(
    current_feat: Tensor,
    command: str,
    strength: float = 1.0,
    target_key: Optional[int] = None,
) -> Tensor:
    """Map a current tension feature vector [TENSION_FEATURE_DIM] + a high-level
    command to a target tension feature vector to condition the generator.

    The mapping nudges the interpretable components (plan Module 9):
      - add_tension: raise tension / distance-to-key / dissonance, positive slope.
      - resolve: lower tension, sharpen key posterior, raise confidence / drop
        entropy, negative slope, positive resolution.
      - change_tonal_center: shift the key posterior toward `target_key`.
      - maintain: keep the current state.
    Returns a new tensor (the input is not modified in place).
    """
    tgt = current_feat.clone().float()
    s = float(strength)

    if command == CMD_ADD_TENSION:
        tgt[TF_TENSION_SCALAR] = current_feat[TF_TENSION_SCALAR] + 0.5 * s
        tgt[TF_DIST_KEY] = (current_feat[TF_DIST_KEY] + 0.4 * s).clamp(0.0, math.pi)
        tgt[TF_DIST_TONIC] = (current_feat[TF_DIST_TONIC] + 0.4 * s).clamp(0.0, math.pi)
        tgt[TF_DISSONANCE] = (current_feat[TF_DISSONANCE] + 0.3 * s).clamp(0.0, 1.0)
        tgt[TF_HARMONIC_CHANGE] = current_feat[TF_HARMONIC_CHANGE] + 0.3 * s
        tgt[TF_TENSION_SLOPE] = 0.5 * s
        tgt[TF_RESOLUTION] = -0.5 * s

    elif command == CMD_RESOLVE:
        tgt[TF_TENSION_SCALAR] = (current_feat[TF_TENSION_SCALAR] - 0.5 * s).clamp_min(0.0)
        tgt[TF_DIST_KEY] = (current_feat[TF_DIST_KEY] - 0.4 * s).clamp_min(0.0)
        tgt[TF_DIST_TONIC] = (current_feat[TF_DIST_TONIC] - 0.5 * s).clamp_min(0.0)
        tgt[TF_DISSONANCE] = (current_feat[TF_DISSONANCE] - 0.3 * s).clamp_min(0.0)
        tgt[TF_HARMONIC_CHANGE] = (current_feat[TF_HARMONIC_CHANGE] - 0.3 * s).clamp_min(0.0)
        tgt[TF_KEY_CONF] = (current_feat[TF_KEY_CONF] + 0.3 * s).clamp(0.0, 1.0)
        tgt[TF_KEY_ENTROPY] = (current_feat[TF_KEY_ENTROPY] - 0.3 * s).clamp_min(0.0)
        tgt[TF_MOD_PRESSURE] = (current_feat[TF_MOD_PRESSURE] - 0.3 * s).clamp_min(0.0)
        tgt[TF_TENSION_SLOPE] = -0.5 * s
        tgt[TF_RESOLUTION] = 0.5 * s
        # sharpen the key posterior toward its current top key
        post = current_feat[TF_KEY_POST]
        top = int(post.argmax().item())
        sharp = torch.zeros_like(post)
        sharp[top] = 1.0
        tgt[TF_KEY_POST] = (1.0 - 0.5 * s) * post + 0.5 * s * sharp

    elif command == CMD_CHANGE_TONAL_CENTER:
        post = current_feat[TF_KEY_POST]
        if target_key is None:
            # default: move to the dominant of the current top key (up a fifth,
            # same mode), a musically related modulation.
            top = int(post.argmax().item())
            mode_off = 12 if top >= 12 else 0
            target_key = ((top % 12) + 7) % 12 + mode_off
        onehot = torch.zeros_like(post)
        onehot[int(target_key) % NUM_KEYS] = 1.0
        tgt[TF_KEY_POST] = (1.0 - 0.6 * s) * post + 0.6 * s * onehot
        tgt[TF_KEY_CONF] = (current_feat[TF_KEY_CONF] * 0.5 + 0.5).clamp(0.0, 1.0)
        tgt[TF_TENSION_SLOPE] = 0.3 * s   # temporary rise while modulating

    # CMD_MAINTAIN falls through (target == current)
    return tgt


def target_chroma_for_feature(target_feat: Tensor) -> Tensor:
    """Derive a soft chord-tone chroma [12] from a target feature vector for the
    inference PC-bias: the tonic triad of the target top key. Returns a
    binary-ish chroma (all-zero if the posterior is empty)."""
    post = target_feat[TF_KEY_POST]
    top = int(post.argmax().item())
    root = top % 12
    triad = _MINOR_TONIC if top >= 12 else _MAJOR_TONIC
    chroma = torch.zeros(12, dtype=torch.float32, device=target_feat.device)
    for iv in triad:
        chroma[(root + iv) % 12] = 1.0
    return chroma


# --------------------------------------------------------------------------- #
#  Prompt key pre-analysis                                                     #
# --------------------------------------------------------------------------- #
def estimate_key_prior_from_pitches(
    pitches: Tensor,
    key_alpha: float = 5.0,
    velocities: Optional[Tensor] = None,
    profiles: str = 'genie',
    bass_weight: float = 2.0,
) -> Tensor:
    """Estimate a [NUM_KEYS] key posterior from a whole pitch sequence at once
    (global velocity-weighted chroma histogram, key-profile correlation).
    Used to pre-analyze a prompt/primer so the realtime key filter starts from
    a reliable prior instead of the unstable few-notes regime. Uniform when
    the sequence contains no valid MIDI pitches."""
    pitches = pitches.long()
    valid = (pitches >= 0) & (pitches < 128)
    if int(valid.sum()) == 0:
        return torch.full((NUM_KEYS,), 1.0 / NUM_KEYS, dtype=torch.float32)
    vp = pitches[valid]
    w = _register_weight(vp, bass_weight)
    if velocities is not None:
        w = w * velocities.float()[valid]
    chroma = torch.bincount(vp % 12, weights=w, minlength=12).float()
    corr = _key_correlation(chroma, profiles)                          # [24]
    return torch.softmax(float(key_alpha) * corr, dim=-1)


# --------------------------------------------------------------------------- #
#  Real-time extractor (inference)                                             #
# --------------------------------------------------------------------------- #
class RealtimeTensionExtractor:
    """Persistent, incremental tonal-tension tracker for inference. Exposes the
    current tension feature vector for command_to_target + FiLM at play time.

    Key detection follows docs "Real-Time MIDI Key Detection": a leaky-
    integrator chroma (velocity-weighted, pitch-dependent exponential decay —
    'virtual sustain') provides the evidence, Temperley/Sapp profile
    correlation the emission, and an HMM forward filter with a circle-of-fifths
    ring transition prior the temporal stabilization. Related-key modulations
    are followed within a phrase; distant jumps require sustained evidence.
    Use `prime()` to pre-analyze a prompt/primer, and `lock_key()` to hard-pin
    the key when full reliability is required. Feature formulas match
    extract_tension_features step for step."""

    def __init__(
        self,
        short_window: int = 8,
        key_alpha: float = 2.0,
        trans_alpha: float = 100.0,
        tau_low: float = 8.0,
        tau_high: float = 3.0,
        profiles: str = 'genie',
        note_dt: float = 0.25,
        key_decision_theta: float = 3.0,
        bass_weight: float = 2.0,
        cadence_gap: float = 1.0,
        cadence_boost: float = 2.0,
        weights: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
    ):
        self.short_window = int(short_window)
        self.key_alpha = float(key_alpha)
        self.trans_alpha = float(trans_alpha)
        self.tau_low = float(tau_low)
        self.tau_high = float(tau_high)
        self.profiles = str(profiles)
        self.note_dt = float(note_dt)
        self.key_decision_theta = float(key_decision_theta)
        self.bass_weight = float(bass_weight)
        self.cadence_gap = float(cadence_gap)
        self.cadence_boost = float(cadence_boost)
        self.weights = dict(_DEFAULT_WEIGHTS)
        if weights is not None:
            self.weights.update(weights)
        self.device = device if device is not None else torch.device('cpu')
        self.locked_key: Optional[int] = None
        self.reset()

    def reset(self) -> None:
        """Clear all evidence (buffer, key posterior, deltas). Keeps a key lock."""
        self.pitch_buffer: List[int] = []
        # active integrator notes: [pitch_class, weight, onset_time, decay_rate]
        # (weight = velocity * register weight; mutable for the cadence boost)
        self._notes: List[List[float]] = []
        self._last_time: Optional[float] = None
        self._last_energy: Optional[Tensor] = None     # cached integrator chroma
        self._last_corr: Optional[Tensor] = None       # cached key correlations [24]
        self.posterior: Tensor = torch.full(
            (NUM_KEYS,), 1.0 / NUM_KEYS, dtype=torch.float32, device=self.device)
        # Hysteretic key decision (Temperley "change penalty" at decision level):
        # the reported key only switches when the challenger's posterior exceeds
        # the incumbent's by a factor key_decision_theta, so near-ties between
        # related keys don't flip the display/command target back and forth.
        self.current_key: Optional[int] = self.locked_key
        self._prev_tiv_short: Optional[Tensor] = None
        self._prev_scalar: float = 0.0
        self._prev_conf: float = 0.0
        self._prev_entropy: float = 0.0
        self._prev_dist_tonic: float = 0.0
        self._feat: Tensor = torch.zeros(
            TENSION_FEATURE_DIM, dtype=torch.float32, device=self.device)
        self._has_notes: bool = False

    # -- key control ------------------------------------------------------- #
    def lock_key(self, key: Optional[int]) -> None:
        """Pin the tracked key to `key` (0..23, see key-class mapping) until
        unlocked with lock_key(None). While locked the posterior is one-hot, so
        every key-derived feature is perfectly stable."""
        self.locked_key = None if key is None else int(key) % NUM_KEYS
        if self.locked_key is not None:
            self.posterior = torch.zeros(
                NUM_KEYS, dtype=torch.float32, device=self.device)
            self.posterior[self.locked_key] = 1.0
            self.current_key = self.locked_key

    def top_key(self) -> int:
        """Current key class (0..23) after the hysteretic decision (stable:
        near-ties between related keys don't flip it back and forth)."""
        if self.current_key is not None:
            return self.current_key
        return int(self.posterior.argmax().item())

    def neighbor_distances(self) -> Dict[str, float]:
        """Directional key distances relative to the current (hysteretic) key.

        Returns, for each ring-1 neighbor (relative / parallel / dominant /
        subdominant) plus the key itself:
          p_<name>       posterior mass on that key (where the filter thinks
                         the tonality is pulling);
          margin_<name>  contrastive evidence margin corr(key) - corr(neighbor)
                         from the current integrator chroma — a two-hypothesis
                         score concentrated on exactly the scale degrees that
                         distinguish the pair (e.g. 7th vs b7th for the
                         subdominant), so it reacts before the 24-way
                         posterior does. Positive = current key still wins.
        """
        key = self.top_key()
        nbrs = neighbor_keys(key)
        out: Dict[str, float] = {'p_same': float(self.posterior[key])}
        for name, idx in nbrs.items():
            out[f'p_{name}'] = float(self.posterior[idx])
        if self._last_corr is not None:
            ck = float(self._last_corr[key])
            for name, idx in nbrs.items():
                out[f'margin_{name}'] = ck - float(self._last_corr[idx])
        return out

    def fifths_signature(self) -> Dict[str, float]:
        """Signature-of-fifths view of the current integrator chroma (see
        signature_of_fifths), plus `fifths_shift`: the signed offset (in fifth
        steps, wrapped to [-6, 6)) between the observed diatonic-mass center
        and the one expected for the current key. +1 ~= drifting toward the
        dominant, -1 ~= toward the subdominant; `mode_score` < 0 leans major,
        > 0 leans the relative minor — both independent of the key profiles."""
        if self._last_energy is None:
            return {'fifths_center': 0.0, 'major_key': 0, 'minor_key': 21,
                    'mode_score': 0.0, 'cvsf_norm': 0.0, 'fifths_shift': 0.0}
        sig = signature_of_fifths(self._last_energy)
        key = self.top_key()
        # expected diatonic center: f(tonic)+2 for major keys, f(tonic)-1 for
        # minor (same signature as the relative major).
        expected = (_KEY_FIFTHS[key] + (2.0 if key < 12 else -1.0)) % 12.0
        sig['fifths_shift'] = ((sig['fifths_center'] - expected + 6.0) % 12.0) - 6.0
        return sig

    def prime(self, pitches: List[int], velocities: Optional[List[int]] = None) -> None:
        """Pre-analyze a prompt/primer: estimate a global key prior from the
        whole sequence at once, then run the incremental filter over the notes
        (synthetic note_dt spacing ending 'now'). The tracker ends up with a
        converged key posterior and warm integrator/delta state, so real-time
        tracking never starts from the unreliable few-notes regime."""
        self.reset()
        n = len(pitches)
        if n == 0:
            return
        pt = torch.tensor(list(pitches), dtype=torch.long)
        vt = None if velocities is None else torch.tensor(list(velocities))
        if self.locked_key is None:
            # A whole-prompt histogram is strong evidence: use a sharper gain
            # than the per-note emission so the filter starts well anchored.
            self.posterior = estimate_key_prior_from_pitches(
                pt, 5.0, vt, self.profiles, self.bass_weight).to(self.device)
            self.current_key = int(self.posterior.argmax().item())
        now = time.monotonic()
        for i, p in enumerate(pitches):
            v = 64 if velocities is None else int(velocities[i])
            self.add_pitch(int(p), velocity=v, t=now - (n - 1 - i) * self.note_dt)

    # -- incremental update ------------------------------------------------ #
    def _integrator_chroma(self, now: float) -> Tensor:
        """Leaky-integrator chroma [12] at time `now`; prunes dead notes."""
        self._notes = [nt for nt in self._notes
                       if (now - nt[2]) * nt[3] < 12.0]   # weight >= ~6e-6 kept
        chroma = torch.zeros(12, dtype=torch.float32, device=self.device)
        for pc, w, onset, rate in self._notes:
            chroma[int(pc)] += w * math.exp(-rate * max(0.0, now - onset))
        return chroma

    def add_pitch(self, pitch: int, velocity: int = 64, t: Optional[float] = None) -> None:
        """Feed one (generated or played) pitch and update the tracked state.
        `t` is the note-onset time in seconds (wall clock by default; any
        monotonic clock works as long as it is used consistently). Non-MIDI
        pitch values (e.g. PAD >= 128) are ignored."""
        pitch = int(pitch)
        if pitch < 0 or pitch >= 128:
            return
        if t is None:
            t = time.monotonic()
        if self._last_time is not None and t < self._last_time:
            t = self._last_time                    # keep onsets monotonic
        self.pitch_buffer.append(pitch)
        if len(self.pitch_buffer) > self.short_window:
            self.pitch_buffer.pop(0)
        self._has_notes = True

        # cadence boost: the previous note turned out to be phrase-final
        # (followed by a long inter-onset gap) -> boost its key evidence.
        if (self._notes and self._last_time is not None
                and (t - self._last_time) > self.cadence_gap):
            self._notes[-1][1] *= self.cadence_boost
        self._last_time = t

        rate = float(_note_decay_rate(
            torch.tensor([pitch], dtype=torch.float32), self.tau_low, self.tau_high)[0])
        reg_w = float(_register_weight(
            torch.tensor([pitch], dtype=torch.float32), self.bass_weight)[0])
        self._notes.append([pitch % 12, float(velocity) * reg_w, float(t), rate])

        short_chroma = self._buffer_chroma(self.short_window)
        tiv_short = chroma_to_tiv(short_chroma)

        energy = self._integrator_chroma(t)
        corr = _key_correlation(energy, self.profiles)
        self._last_energy, self._last_corr = energy, corr
        if self.locked_key is not None:
            posterior = self.posterior      # one-hot, set by lock_key
        else:
            lik = torch.exp(self.key_alpha * corr)
            trans = _key_transition_matrix(self.trans_alpha).to(lik)
            posterior = _key_filter_step(self.posterior, lik, trans)
            self.posterior = posterior
            # hysteretic key decision (see reset() comment)
            j = int(posterior.argmax().item())
            if self.current_key is None:
                self.current_key = j
            elif (j != self.current_key and float(posterior[j])
                    > self.key_decision_theta * float(posterior[self.current_key])):
                self.current_key = j

        feat = self._feature_from_state(tiv_short, posterior)
        self._feat = feat
        self._prev_tiv_short = tiv_short
        self._prev_scalar = float(feat[TF_TENSION_SCALAR])
        self._prev_conf = float(feat[TF_KEY_CONF])
        self._prev_entropy = float(feat[TF_KEY_ENTROPY])
        self._prev_dist_tonic = float(feat[TF_DIST_TONIC])

    def _feature_from_state(self, tiv_short: Tensor, posterior: Tensor) -> Tensor:
        """Assemble the [TENSION_FEATURE_DIM] feature vector for a sonority TIV
        + key posterior against the tracker's previous-state deltas. Pure: no
        state mutation (shared by add_pitch and peek_features_batch)."""
        return self._features_from_state_batch(
            tiv_short.unsqueeze(0), posterior.unsqueeze(0))[0]

    def _features_from_state_batch(self, tiv_short: Tensor, posterior: Tensor) -> Tensor:
        """Vectorized feature assembly for K hypothetical states: tiv_short
        [K,12], posterior [K,24] -> [K, TENSION_FEATURE_DIM]. Pure."""
        w = self.weights
        K = tiv_short.shape[0]
        key_tiv = _KEY_TIV.to(tiv_short)
        key_conf = posterior.max(dim=-1).values                              # [K]
        key_entropy = (-(posterior * (posterior + 1e-8).log()).sum(dim=-1)
                       / math.log(NUM_KEYS))                                 # [K]

        if self._prev_tiv_short is not None:
            harmonic_change = tiv_euclidean(
                tiv_short, self._prev_tiv_short.unsqueeze(0))                # [K]
        else:
            harmonic_change = torch.zeros(K, device=tiv_short.device)

        expected_key_tiv = torch.matmul(posterior, key_tiv)                  # [K,12]
        dist_key = tiv_angle(tiv_short, expected_key_tiv)                    # [K]
        ang_tonic = tiv_angle(tiv_short.unsqueeze(1), _TONIC_TIV.to(tiv_short).unsqueeze(0))
        ang_subdom = tiv_angle(tiv_short.unsqueeze(1), _SUBDOM_TIV.to(tiv_short).unsqueeze(0))
        ang_dom = tiv_angle(tiv_short.unsqueeze(1), _DOM_TIV.to(tiv_short).unsqueeze(0))
        dist_tonic = (posterior * ang_tonic).sum(dim=-1)                     # [K]
        dist_subdom = (posterior * ang_subdom).sum(dim=-1)
        dist_dom = (posterior * ang_dom).sum(dim=-1)

        dissonance = (1.0 - tiv_short.norm(dim=-1)).clamp(0.0, 1.0)
        mod_pressure = (1.0 - key_conf) + key_entropy

        tension_scalar = (
            w['harmonic_change'] * harmonic_change
            + w['distance_to_key'] * dist_key
            + w['tiv_dissonance'] * dissonance
            + w['key_entropy'] * key_entropy
            + w['modulation_pressure'] * mod_pressure
        )
        if self._prev_tiv_short is not None:
            tension_slope = tension_scalar - self._prev_scalar
            d_conf = key_conf - self._prev_conf
            d_ent = key_entropy - self._prev_entropy
            d_tonic = dist_tonic - self._prev_dist_tonic
        else:
            tension_slope = torch.zeros(K, device=tiv_short.device)
            d_conf = d_ent = d_tonic = torch.zeros(K, device=tiv_short.device)
        resolution = -tension_slope + d_conf - d_ent - d_tonic

        feat = torch.zeros(K, TENSION_FEATURE_DIM, dtype=torch.float32,
                           device=self.device)
        feat[:, TF_TIV] = tiv_short
        feat[:, TF_KEY_POST] = posterior
        feat[:, TF_KEY_CONF] = key_conf
        feat[:, TF_KEY_ENTROPY] = key_entropy
        feat[:, TF_HARMONIC_CHANGE] = harmonic_change
        feat[:, TF_DIST_KEY] = dist_key
        feat[:, TF_DIST_TONIC] = dist_tonic
        feat[:, TF_DIST_SUBDOM] = dist_subdom
        feat[:, TF_DIST_DOM] = dist_dom
        feat[:, TF_DISSONANCE] = dissonance
        feat[:, TF_MOD_PRESSURE] = mod_pressure
        feat[:, TF_TENSION_SCALAR] = tension_scalar
        feat[:, TF_TENSION_SLOPE] = tension_slope
        feat[:, TF_RESOLUTION] = resolution
        return feat

    def peek_features_batch(self, pitches: List[int], velocity: int = 64) -> Tensor:
        """[K, TENSION_FEATURE_DIM] feature vectors as if each candidate pitch
        were played right now, WITHOUT mutating the tracker. Fully vectorized —
        the integrator energy and transition prediction are computed once and
        shared across candidates, so scoring K candidates costs a couple of
        small matmuls (well under 1 ms) and ZERO extra decoder passes. Used by
        the inference-time tension-critic rerank."""
        K = len(pitches)
        if K == 0:
            return torch.zeros(0, TENSION_FEATURE_DIM, dtype=torch.float32,
                               device=self.device)
        t = self._last_time if self._last_time is not None else time.monotonic()
        pt = torch.tensor([max(0, min(127, int(p))) for p in pitches],
                          dtype=torch.long, device=self.device)
        pc = pt % 12
        rows = torch.arange(K, device=self.device)

        # hypothetical short-window sonority per candidate
        base_ch = torch.zeros(12, dtype=torch.float32, device=self.device)
        for p in self.pitch_buffer[-(self.short_window - 1):] if self.short_window > 1 else []:
            base_ch[p % 12] += 1.0
        ch = base_ch.unsqueeze(0).repeat(K, 1)
        ch[rows, pc] += 1.0
        tiv_short = chroma_to_tiv(ch)                                        # [K,12]

        # hypothetical integrator energy per candidate (base shared)
        base_e = self._integrator_chroma(t)
        reg_w = _register_weight(pt.float(), self.bass_weight) * float(velocity)
        en = base_e.unsqueeze(0).repeat(K, 1)
        en[rows, pc] += reg_w
        if self.locked_key is not None:
            posterior = self.posterior.unsqueeze(0).expand(K, NUM_KEYS)
        else:
            corr = _key_correlation(en, self.profiles)                       # [K,24]
            trans = _key_transition_matrix(self.trans_alpha).to(corr)
            pred = torch.matmul(self.posterior, trans).unsqueeze(0)          # [1,24]
            post = pred * torch.exp(self.key_alpha * corr)
            posterior = post / post.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return self._features_from_state_batch(tiv_short, posterior)

    def peek_features(self, pitch: int, velocity: int = 64) -> Tensor:
        """Single-candidate convenience wrapper around peek_features_batch."""
        if int(pitch) < 0 or int(pitch) >= 128:
            return self._feat
        return self.peek_features_batch([int(pitch)], velocity)[0]

    def _buffer_chroma(self, window: int) -> Tensor:
        """Unnormalized chroma count over the last `window` buffered notes."""
        chroma = torch.zeros(12, dtype=torch.float32, device=self.device)
        for p in self.pitch_buffer[-window:]:
            chroma[p % 12] += 1.0
        return chroma

    def current_features(self) -> Tensor:
        """Current [TENSION_FEATURE_DIM] tension feature vector for the most
        recent note (all-zero before any note has been fed)."""
        return self._feat

    def current_chroma(self) -> Tensor:
        """Current short-window chroma [12], normalized to sum 1 (for
        visualization / PC-bias). All-zero before any note has been fed."""
        if not self._has_notes:
            return torch.zeros(12, dtype=torch.float32, device=self.device)
        ch = self._buffer_chroma(self.short_window)
        return ch / ch.sum().clamp_min(1e-8)
