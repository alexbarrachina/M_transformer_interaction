#===================================================================================================
# Monster Genie pitch_reins.py Python module
# Reins: a pedal-controlled pitch window for the style joker, with no retraining.
# Each button has an ideal pitch, the button-th note of the key's scale above the
# layout's tonic (buttons 0, 7 and 14 are tonics, 18 is the 5th two octaves up).
# The pedal sets how far the model may move away from it: fully pressed plays the
# ideal pitch, barely pressed lets the model choose within MAX_WIDTH semitones, in the
# direction the hand moved. Fully released (pedal <= LEARNED_PEDAL) has no window: the
# button goes to the model as in interaction_buttons_style_chords.py (the learned map,
# relative to the passage, ~2.8 semitones per button). Until 2026-10-27 the released
# pedal was the +-MAX_WIDTH window too. The layout's octave follows the register of the music.
# Pure Python (no torch), like harmonic_step.py; the interaction script converts
# the bias list to a tensor for gen_pitch_token(logit_bias=...).
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

import statistics
from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

PC_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
# Natural minor. Harmonic minor is (0, 2, 3, 5, 7, 8, 11); with the pedal not fully
# pressed the window already lets the model raise the 6th or 7th.
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)

MAX_WIDTH = 7              # window half-width in semitones with the pedal barely pressed (a 5th)
LEARNED_PEDAL = 0.0        # at or below this pedal: no window, the learned button map
PULL = 0.0                 # optional pull toward the ideal pitch: -PULL * pedal * distance^2
DIRECTION_MIN_WIDTH = 2    # narrower windows need no direction rule
CHORD_GAP_MS = 30.0        # presses closer than this are a chord: no direction rule
OUTSIDE_BIAS = -1e4        # finite, so the softmax never sees only -inf

# Register: the 19-button layout spans 31 semitones (tonic .. 5th two octaves up).
LAYOUT_SPAN = 31
REGISTER_NOTES = 24        # sounded notes in the register median
OCTAVE_HYSTERESIS = 9.0    # semitones between the median and the layout's middle before moving
FREEZE_PEDAL = 0.8         # at or above this pedal the octave stays put
LOWEST_PITCH, HIGHEST_PITCH = 21, 108  # piano range


def pitch_name(pitch: int) -> str:
    """MIDI pitch name, 57 -> 'A3'."""
    return f"{PC_NAMES[pitch % 12]}{pitch // 12 - 1}"


def key_name(tonic_pc: int, minor: bool) -> str:
    return f"{PC_NAMES[tonic_pc % 12]} {'minor' if minor else 'major'}"


def ideal_pitch(button: int, anchor: int, minor: bool) -> int:
    """Button 0 = anchor (a tonic), 7 = the tonic an octave up, 14 two octaves up."""
    scale = MINOR_SCALE if minor else MAJOR_SCALE
    return anchor + 12 * (button // 7) + scale[button % 7]


def window_width(pedal: float, max_width: int = MAX_WIDTH) -> int:
    """Half-width in semitones: max_width at pedal 0, 0 fully pressed (at LEARNED_PEDAL press() uses no window)."""
    return int(max_width * (1.0 - pedal) + 0.5)


def allowed_pitches(ideal: int, width: int, direction: int, last_pitch: Optional[int]) -> List[int]:
    """Pitches within ideal +- width; when the window is wide, only those above
    (direction 1) or below (-1) the last sounded pitch, unless none would remain."""
    window = [p for p in range(ideal - width, ideal + width + 1) if LOWEST_PITCH <= p <= HIGHEST_PITCH]
    if not window:
        window = [min(max(ideal, LOWEST_PITCH), HIGHEST_PITCH)]
    if width < DIRECTION_MIN_WIDTH or direction == 0 or last_pitch is None:
        return window
    followed = [p for p in window if (p > last_pitch if direction > 0 else p < last_pitch)]
    return followed or window


def pitch_bias(allowed: Sequence[int], ideal: int, pedal: float, vocab_size: int) -> List[float]:
    """Additive logit bias: 0 on allowed pitches (minus the optional pull), OUTSIDE_BIAS elsewhere.
    Removing the other pitches raises the allowed ones together, so the model's ranking
    among the allowed pitches is kept."""
    bias = [OUTSIDE_BIAS] * vocab_size
    for p in allowed:
        bias[p] = -PULL * pedal * (p - ideal) ** 2
    return bias


class Reins:
    """Key, layout octave, pedal and the last press, for one performer.

    Call press() on each note-on to get the ideal and allowed pitches, then
    after_note() with the sounded pitch. The pedal is a float in [0, 1]."""

    def __init__(self, tonic_pc: int, minor: bool, primer_pitches: Sequence[int]):
        self.tonic_pc: int = tonic_pc % 12
        self.minor: bool = minor
        self.pedal: float = 0.0
        self.recent: Deque[int] = deque(primer_pitches[-REGISTER_NOTES:], maxlen=REGISTER_NOTES)
        self.anchor: int = self._nearest_anchor()
        self.last_button: Optional[int] = None
        self.last_press_ms: Optional[float] = None
        self.last_pitch: Optional[int] = self.recent[-1] if self.recent else None

    def _centre(self) -> float:
        return statistics.median(self.recent) if self.recent else 60.0

    def _nearest_anchor(self) -> int:
        """The tonic whose layout middle (anchor + 15.5) is nearest the register median."""
        candidates = [t for t in range(LOWEST_PITCH, HIGHEST_PITCH - LAYOUT_SPAN + 1)
                      if t % 12 == self.tonic_pc]
        centre = self._centre()
        return min(candidates, key=lambda t: abs(t + LAYOUT_SPAN / 2 - centre))

    def set_key(self, tonic_pc: int, minor: bool) -> None:
        self.tonic_pc, self.minor = tonic_pc % 12, minor
        self.anchor = self._nearest_anchor()

    def reset(self, primer_pitches: Sequence[int]) -> None:
        self.recent = deque(primer_pitches[-REGISTER_NOTES:], maxlen=REGISTER_NOTES)
        self.anchor = self._nearest_anchor()
        self.last_button = self.last_press_ms = None
        self.last_pitch = self.recent[-1] if self.recent else None

    def update_anchor(self) -> bool:
        """Move the layout by octaves when the register drifts; frozen at strict pedal.
        Returns True when the anchor moved."""
        if self.pedal >= FREEZE_PEDAL:
            return False
        if abs(self._centre() - (self.anchor + LAYOUT_SPAN / 2)) <= OCTAVE_HYSTERESIS:
            return False
        anchor = self._nearest_anchor()
        moved = anchor != self.anchor
        self.anchor = anchor
        return moved

    def press(self, button: int, now_ms: float) -> Tuple[int, int, Optional[List[int]]]:
        """Ideal pitch, window half-width and allowed pitches for this press.
        Allowed is None with the pedal released: the button itself goes to the model."""
        ideal = ideal_pitch(button, self.anchor, self.minor)
        width = window_width(self.pedal)
        direction = 0
        is_chord = self.last_press_ms is not None and now_ms - self.last_press_ms < CHORD_GAP_MS
        if self.last_button is not None and not is_chord:
            direction = (button > self.last_button) - (button < self.last_button)
        self.last_button, self.last_press_ms = button, now_ms
        if self.pedal <= LEARNED_PEDAL:
            return ideal, width, None
        return ideal, width, allowed_pitches(ideal, width, direction, self.last_pitch)

    def after_note(self, pitch: int) -> None:
        """Every sounded note (Reins or Joker) feeds the register and the direction rule."""
        self.recent.append(pitch)
        self.last_pitch = pitch

    def status(self) -> str:
        return (f"Reins: {key_name(self.tonic_pc, self.minor)}, key 0 = {pitch_name(self.anchor)}"
                f"  |  pedal {self.pedal:.2f}, "
                + ("learned map" if self.pedal <= LEARNED_PEDAL else f"+-{window_width(self.pedal)} st"))
