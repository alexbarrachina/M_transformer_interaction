#===================================================================================================
# Monster Genie tonic_shift.py Python module
# Tonic shift: the tonic on key 7 while playing, with no retraining.
# The style model's map from buttons to pitches is left as it is and only moved along the
# buttons: key 7 sends the model the button where its map has the tonic now, and every other
# key moves by the same number of buttons (the shift). The model still chooses every note,
# so key 7 gives the tonic or a note next to it (a button covers about 3 semitones).
# Where the tonic is in the map is read from the last notes and the buttons the model received
# for them, so the model evaluates nothing more. Keys that fall beyond the model's first or
# last button send that end button.
# The shift also hides buttons at the opposite end. Pressing the end key there again (key 18 twice
# in a row) pushes the window one button further, until the map's own end. Reaching key 7 undoes the push,
# so key 7 is always the tonic.
# Pure Python (no torch), like pitch_reins.py.
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
from typing import Optional, Sequence

from pitch_reins import HIGHEST_PITCH, LOWEST_PITCH, PC_NAMES, pitch_name

TONIC_BUTTON = 7        # the key that holds the tonic, as button 7 of the Reins layout (pitch_reins.ideal_pitch)
HISTORY = 32            # last notes used to find the tonic in the model's map
DEFAULT_SLOPE = 0.36    # buttons per semitone (2.8 semitones per button), when the notes give no slope
SHIFT_HYSTERESIS = 1.0  # buttons the tonic must have moved in the map before the shift follows it; 1.5 = only larger moves
OCTAVE_MARGIN = 1.0     # buttons by which another octave of the tonic must be nearer key 7 to replace this one


def map_slope(pitches: Sequence[int], buttons: Sequence[int]) -> float:
    """Buttons per semitone of the model's map: least-squares slope of the buttons on the pitches."""
    mean_pitch, mean_button = statistics.fmean(pitches), statistics.fmean(buttons)
    spread = sum((pitch - mean_pitch) ** 2 for pitch in pitches)
    if spread == 0:
        return DEFAULT_SLOPE
    slope = sum((pitch - mean_pitch) * (button - mean_button) for pitch, button in zip(pitches, buttons)) / spread
    return slope if slope > 0 else DEFAULT_SLOPE


def map_position(pitch: int, pitches: Sequence[int], buttons: Sequence[int], slope: float) -> float:
    """Where a pitch sits in the model's map, in buttons: the button of the latest note nearest to it,
    moved along the slope by the distance between them."""
    nearest = min(reversed(range(len(pitches))), key=lambda note: abs(pitches[note] - pitch))
    return buttons[nearest] + slope * (pitch - pitches[nearest])


class TonicShift:
    """The tonic on key 7, the model button taken as its place, and the shift, for one performer.

    Call update() before each press with the notes so far and the buttons the model received for
    them; press() then gives the button to send for the pressed key."""

    def __init__(self, tonic_pc: int, num_buttons: int):
        self.tonic_pc: int = tonic_pc % 12
        self.num_buttons: int = num_buttons
        self.tonic: Optional[int] = None       # MIDI pitch on key TONIC_BUTTON; None until the first update()
        self.tonic_button: int = TONIC_BUTTON  # the model button where the tonic is taken to be
        self.push: int = 0                     # buttons the window is pushed from the shift, towards a shift of 0
        self.last_key: Optional[int] = None    # the last pressed key, Joker notes not included

    @property
    def shift(self) -> int:
        """Buttons the map is moved by: a key sends the model button key - shift + push."""
        return TONIC_BUTTON - self.tonic_button

    def set_tonic(self, tonic_pc: int) -> None:
        """A new key, or the same one after a reset: the next update() chooses the octave again."""
        self.tonic_pc, self.tonic = tonic_pc % 12, None
        self.push, self.last_key = 0, None

    def _button(self, position: float) -> int:
        return min(max(round(position), 0), self.num_buttons - 1)

    def update(self, pitches: Sequence[int], buttons: Sequence[int]) -> bool:
        """Finds the tonic in the model's map again. pitches and buttons are the notes so far and the buttons
        the model received for them, without Joker notes; the last HISTORY are used.
        Returns True when the tonic or the shift changed."""
        pitches, buttons = pitches[-HISTORY:], buttons[-HISTORY:]
        if not pitches:
            return False
        slope = map_slope(pitches, buttons)
        position = lambda pitch: map_position(pitch, pitches, buttons, slope)
        distance = lambda pitch: abs(position(pitch) - TONIC_BUTTON)  # buttons the map would have to move
        before = (self.tonic, self.tonic_button)
        # Octave: the tonic that needs the smallest shift. It stays until another octave is clearly nearer,
        # so the shift rarely exceeds 3 buttons (two tonics are about 4 buttons apart).
        nearest = min((pitch for pitch in range(LOWEST_PITCH, HIGHEST_PITCH + 1) if pitch % 12 == self.tonic_pc),
                      key=distance)
        if self.tonic is None or distance(nearest) + OCTAVE_MARGIN < distance(self.tonic):
            self.tonic = nearest
            self.tonic_button = self._button(position(nearest))
        elif abs(position(self.tonic) - self.tonic_button) >= SHIFT_HYSTERESIS:
            # Shift: it follows the tonic only after a move of a whole button, as when the tonic itself has just
            # sounded on a neighbouring button; smaller moves are ignored, so the keys seldom change meaning.
            self.tonic_button = self._button(position(self.tonic))
        return (self.tonic, self.tonic_button) != before

    def press(self, key: int) -> int:
        """The button the model receives for a pressed key, pushing the window first if it is an end key pressed again.
        Keys beyond the model's ends send its end button."""
        if (self.push > 0 and key <= TONIC_BUTTON) or (self.push < 0 and key >= TONIC_BUTTON):
            self.push = 0
        elif key == self.last_key and key in (0, self.num_buttons - 1):
            self.push += 1 if key else -1
        # A push can only reveal buttons the shift hides: it goes towards a shift of 0, and follows the shift if it shrinks.
        self.push = min(max(self.push, min(self.shift, 0)), max(self.shift, 0))
        self.last_key = key
        return min(max(key - self.shift + self.push, 0), self.num_buttons - 1)

    def key(self, button: int) -> int:
        """The key that sends a model button: the reverse of press without a push, to show the primer's buttons as keys."""
        return min(max(button + self.shift, 0), self.num_buttons - 1)

    def status(self) -> str:
        tonic = pitch_name(self.tonic) if self.tonic is not None else PC_NAMES[self.tonic_pc]
        pushed = f", window pushed {self.push:+d}" if self.push else ""
        return f"Tonic shift: key {TONIC_BUTTON} = {tonic}, map moved {self.shift:+d} buttons{pushed}"
