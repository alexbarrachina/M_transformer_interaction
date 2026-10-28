"""Select fixed chord suggestions with held keyboard/MIDI controls; no harmony analysis."""

from collections import deque
from typing import Deque, Dict, NamedTuple, Optional, Tuple

import torch
from torch import Tensor


class Chord(NamedTuple):
    name: str
    root_pc: int
    intervals: Tuple[int, ...]


# The change control asks for a new chord without naming it; the model chooses the chord.
CHANGE = Chord('change (model chooses)', 0, ())


class ChordControls:
    """The last pressed chord key wins; releasing all chord keys clears FiLM.

    The optional change key acts like a chord key, but sends chord_change instead of a chord.
    """

    def __init__(self, primer_length: int, context_length: int, chords: Dict[str, Chord],
                 midi_chords: Optional[Dict[int, str]] = None, change_key: Optional[str] = None,
                 cc_chords: Optional[Dict[int, str]] = None) -> None:
        if context_length < 1:
            raise ValueError('context_length must be positive')
        self.context_length = context_length
        self.chords = chords
        self.midi_chords = dict(midi_chords or {})
        self.cc_chords = dict(cc_chords or {})  # MIDI CC number -> chord key, toggled by each press (handle_cc)
        self.change_key = change_key
        if any(not 0 <= note <= 127 or key not in chords for note, key in self.midi_chords.items()):
            raise ValueError('MIDI chord mappings require notes 0..127 and configured chord keys')
        if any(not 0 <= cc <= 127 or key not in chords for cc, key in self.cc_chords.items()):
            raise ValueError('CC chord mappings require CC numbers 0..127 and configured chord keys')
        self.held: Dict[str, str] = {}
        self.history: Deque[Tuple[Tensor, int, bool, bool]] = deque(maxlen=context_length)
        self.chord_name = ''
        self.chord_notes: Tuple[int, ...] = ()
        self.chroma = torch.zeros(12, dtype=torch.bool)
        self.anchor_pc = 0
        self.active = False
        self.change = False
        self.reset(primer_length)

    def reset(self, primer_length: int) -> None:
        empty = (torch.zeros(12, dtype=torch.bool), 0, False, False)
        self.history = deque([empty] * min(primer_length, self.context_length), maxlen=self.context_length)
        self.release()

    def release(self) -> None:
        self.held.clear()
        self.chord_name = ''
        self.chord_notes = ()
        self.chroma.zero_()
        self.anchor_pc = 0
        self.active = False
        self.change = False
        # Zeroing only the newest gate cannot erase earlier attention effects.
        empty = (torch.zeros(12, dtype=torch.bool), 0, False, False)
        self.history = deque([empty] * len(self.history), maxlen=self.context_length)

    def handle_key(self, key: str, pressed: bool, shift: bool = False) -> bool:
        if key == self.change_key:
            self._handle_control('key:' + key, CHANGE.name, pressed)
            return True
        if key not in self.chords:
            return False
        selected = 'shift+' + key if shift and 'shift+' + key in self.chords else key
        # Remember the physical key so releasing Shift first still releases the chord.
        self._handle_control('key:' + key, selected, pressed)
        return True

    def handle_midi(self, status: int, note: int, velocity: int) -> bool:
        kind, channel = status & 0xF0, status & 0x0F
        if kind not in (0x80, 0x90) or note not in self.midi_chords:
            return False
        self._handle_control(f'midi:{channel}:{note}', self.midi_chords[note],
                             kind == 0x90 and velocity > 0)
        return True

    def handle_cc(self, status: int, controller: int, value: int) -> bool:
        """CC chords are toggles: a press (value > 0) turns its chord on, or off if it was on, and
        turns off any other CC chord. Releases (value 0) are ignored. As with keys, the last pressed control wins."""
        kind, channel = status & 0xF0, status & 0x0F
        if kind != 0xB0 or controller not in self.cc_chords:
            return False
        if value > 0:
            source = f'cc:{channel}:{controller}'
            was_on = source in self.held
            self.held = {other: chord for other, chord in self.held.items() if not other.startswith('cc:')}
            if not was_on:
                self.held[source] = self.cc_chords[controller]
            self._select_held()
        return True

    def release_keyboard(self) -> None:
        self.held = {source: chord for source, chord in self.held.items() if not source.startswith('key:')}
        self._select_held()

    def _handle_control(self, source: str, selected: str, pressed: bool) -> None:
        if pressed:
            if source in self.held:
                return
            self.held[source] = selected
        else:
            self.held.pop(source, None)
        self._select_held()

    def _select_held(self) -> None:
        if not self.held:
            self.release()
            return
        selected = list(self.held.values())[-1]
        self.change = selected == CHANGE.name
        chord = CHANGE if self.change else self.chords[selected]
        self.chord_name = chord.name
        self.chord_notes = tuple(60 + chord.root_pc + interval for interval in chord.intervals)
        self.chroma.zero_()
        self.chroma[[note % 12 for note in self.chord_notes]] = True
        self.anchor_pc = chord.root_pc
        self.active = not self.change

    def next_tokens(self, device: torch.device) -> Dict[str, Tensor]:
        states = list(self.history) + [(self.chroma, self.anchor_pc, self.active, self.change)]
        tokens = {'chord_chroma': torch.stack([state[0] for state in states]).unsqueeze(0).to(device),
                  'chord_anchor_pc': torch.tensor([[state[1] for state in states]], device=device),
                  'chord_active': torch.tensor([[state[2] for state in states]], device=device)}
        if self.change_key:
            tokens['chord_change'] = torch.tensor([[state[3] for state in states]], device=device)
        return tokens

    def commit_note(self) -> None:
        self.history.append((self.chroma.clone(), self.anchor_pc, self.active, self.change))

    def status(self) -> str:
        names = ('C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B')
        note_text = ' '.join(f'{names[note % 12]}{note // 12 - 1}' for note in self.chord_notes) or 'none'
        if self.active or self.change:
            return f'Harmony: {self.chord_name}\nChord notes: {note_text}'
        return 'Harmony: free\nChord notes: none'
