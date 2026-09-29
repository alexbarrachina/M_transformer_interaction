#===================================================================================================
# Monster Genie harmonic_step.py Python module
#
# Zero-training harmonic-step baseline: detect when the generated stream is
# STUCK on one implied triad, and choose a neo-Riemannian (Tonnetz) neighbor —
# L / P / R, literally "one step in the harmony", sharing 2 of 3 tones with the
# current chord — to nudge the sampler toward via a soft pitch-class logit bias.
#
# Pure Python (no torch/numpy): 24-template correlation over 12 dims per note is
# negligible, and this keeps the module trivially unit-testable. The interaction
# script converts the bias list to a tensor for AE_style.gen_pitch_token.
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
import random
from collections import deque
from typing import Deque, List, Optional, Tuple

PC_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# A triad is (root_pc, is_minor). Only maj/min triads: they are the Tonnetz
# vertices the L/P/R transformations are defined on, and template-matching a
# noisy pitch stream against anything finer than maj/min is not robust anyway.
Triad = Tuple[int, bool]


def triad_name(triad: Triad) -> str:
    root, minor = triad
    return f"{PC_NAMES[root % 12]}{'m' if minor else ''}"


def triad_pcs(triad: Triad) -> Tuple[int, int, int]:
    root, minor = triad
    third = 3 if minor else 4
    return (root % 12, (root + third) % 12, (root + 7) % 12)


def triad_chroma(triad: Triad) -> List[float]:
    """12-d binary chroma of a maj/min triad."""
    row = [0.0] * 12
    for pc in triad_pcs(triad):
        row[pc] = 1.0
    return row


def lpr_neighbors(triad: Triad) -> List[Tuple[Triad, str]]:
    """The three neo-Riemannian neighbors (each shares 2 of 3 tones):
       P (parallel)      C  <-> Cm
       R (relative)      C  <-> Am     (Cm <-> Eb)
       L (leading-tone)  C  <-> Em     (Cm <-> Ab)
    """
    root, minor = triad
    if not minor:
        return [((root % 12, True), 'P'),
                (((root + 9) % 12, True), 'R'),
                (((root + 4) % 12, True), 'L')]
    else:
        return [((root % 12, False), 'P'),
                (((root + 3) % 12, False), 'R'),
                (((root + 8) % 12, False), 'L')]


def _correlate(hist: List[float], chroma: List[float]) -> float:
    """Cosine similarity between a pc histogram and a triad chroma."""
    dot = sum(h * c for h, c in zip(hist, chroma))
    nh = math.sqrt(sum(h * h for h in hist))
    nc = math.sqrt(sum(c * c for c in chroma))
    if nh <= 1e-9 or nc <= 1e-9:
        return 0.0
    return dot / (nh * nc)


def estimate_triad(pc_hist: List[float]) -> Tuple[Optional[Triad], float, float]:
    """Best maj/min triad for a 12-d pc histogram.
    Returns (triad, score, margin) where margin = best - runner-up score
    (runner-up over triads with a DIFFERENT pitch-class set, so the relative
    maj/min ambiguity of a bare fifth does not zero the margin)."""
    scored: List[Tuple[float, Triad]] = []
    for root in range(12):
        for minor in (False, True):
            t = (root, minor)
            scored.append((_correlate(pc_hist, triad_chroma(t)), t))
    scored.sort(key=lambda s: s[0], reverse=True)
    best_score, best = scored[0]
    if best_score <= 0.0:
        return None, 0.0, 0.0
    best_set = set(triad_pcs(best))
    margin = best_score
    for score, t in scored[1:]:
        if set(triad_pcs(t)) != best_set:
            margin = best_score - score
            break
    return best, best_score, margin


def choose_step(current: Triad,
                long_hist: List[float],
                avoid: Optional[Triad] = None,
                rng: Optional[random.Random] = None) -> Tuple[Triad, str]:
    """Pick the L/P/R neighbor of `current` whose chroma best fits the long-window
    pc histogram (prefers the diatonic-ish step), skipping `avoid` (the triad we
    stepped away from last time, to prevent A->B->A ping-pong) unless that would
    leave no candidates. Ties broken randomly."""
    rng = rng or random
    cands = [(t, name) for t, name in lpr_neighbors(current)
             if avoid is None or t != avoid]
    if not cands:
        cands = lpr_neighbors(current)
    scored = [(_correlate(long_hist, triad_chroma(t)), rng.random(), t, name)
              for t, name in cands]
    scored.sort(reverse=True)
    _, _, t, name = scored[0]
    return t, name


def step_bias(target: Triad, current: Optional[Triad], num_logits: int,
              up: float = 1.0, down: float = 0.5) -> List[float]:
    """Per-MIDI-pitch additive logit bias implementing one Tonnetz step:
       +up   for pitches whose pc is a target-chord tone
       -down for pitches whose pc is a CURRENT-chord tone not in the target
             (for L/P/R that is exactly the single tone that must move).
    Scale the result by the interaction script's bias weight before use."""
    t_pcs = set(triad_pcs(target))
    c_pcs = set(triad_pcs(current)) - t_pcs if current is not None else set()
    bias = [0.0] * num_logits
    for p in range(num_logits):
        pc = p % 12
        if pc in t_pcs:
            bias[p] = up
        elif pc in c_pcs:
            bias[p] = -down
    return bias


class HarmonicStepTracker:
    """Tracks the triad implied by the generated pitch stream and detects
    harmonic stagnation ("stuck on the same chord").

    Feed every generated pitch with add_note(). The current triad is estimated
    from a recency-weighted pc histogram over the last `window` notes; `stuck`
    means the estimate has not changed for >= `stuck_after` consecutive notes
    (with enough template-match confidence). A `long_window` unweighted
    histogram is kept for choose_step()'s diatonic preference.
    """

    def __init__(self,
                 window: int = 16,
                 long_window: int = 64,
                 stuck_after: int = 12,
                 decay: float = 0.85,
                 min_score: float = 0.55,
                 min_margin: float = 0.05):
        self.window = window
        self.long_window = long_window
        self.stuck_after = stuck_after
        self.decay = decay
        self.min_score = min_score
        self.min_margin = min_margin
        self.notes: Deque[int] = deque(maxlen=window)
        self.long_notes: Deque[int] = deque(maxlen=long_window)
        self.current: Optional[Triad] = None
        self.score: float = 0.0
        self.margin: float = 0.0
        self.held_notes: int = 0          # consecutive notes with the same confident estimate

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.notes.clear()
        self.long_notes.clear()
        self.current = None
        self.score = 0.0
        self.margin = 0.0
        self.held_notes = 0

    def seed(self, pitches: List[int]) -> None:
        """Prime the tracker with existing context (e.g. the primer tail)."""
        for p in pitches:
            self.add_note(int(p))

    # ------------------------------------------------------------------ #
    def _short_hist(self) -> List[float]:
        hist = [0.0] * 12
        n = len(self.notes)
        for idx, p in enumerate(self.notes):
            w = self.decay ** (n - 1 - idx)          # most recent note -> weight 1
            hist[p % 12] += w
        return hist

    def long_hist(self) -> List[float]:
        hist = [0.0] * 12
        for p in self.long_notes:
            hist[p % 12] += 1.0
        return hist

    # ------------------------------------------------------------------ #
    def add_note(self, pitch: int) -> Optional[Triad]:
        """Register a generated pitch; returns the (possibly new) triad estimate."""
        self.notes.append(int(pitch))
        self.long_notes.append(int(pitch))

        triad, score, margin = estimate_triad(self._short_hist())
        self.score, self.margin = score, margin

        confident = (triad is not None and score >= self.min_score
                     and margin >= self.min_margin)
        if confident and triad == self.current:
            self.held_notes += 1
        elif confident:
            self.current = triad
            self.held_notes = 1
        else:
            # keep last estimate but do not accumulate stuck evidence
            self.held_notes = max(0, self.held_notes - 1)
        return self.current

    def is_stuck(self) -> bool:
        return self.current is not None and self.held_notes >= self.stuck_after
