"""Momentary channel-5 performer controls; no harmony analysis."""

import json
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class ChordMapping:
    name: str
    pitch_classes: Tuple[int, ...]
    anchor_pc: int
    key: str
    midi_note: int = -1
    pedal_cc: int = -1


# A C-major example bank, configurable independently of the model and primer.
DEFAULT_CHORDS = (
    ChordMapping('C', (0, 4, 7), 0, '1', 36, 20),
    ChordMapping('F', (5, 9, 0), 5, '2', 37, 21),
    ChordMapping('G7', (7, 11, 2, 5), 7, '3', 38, 22),
    ChordMapping('Am', (9, 0, 4), 9, '4', 39, 23),
    ChordMapping('D7', (2, 6, 9, 0), 2, '5', 40, 24),
)


def read_chord_mappings(path: Optional[str] = None) -> List[ChordMapping]:
    """Optional JSON list with the same fields as ChordMapping; pitches are PCs 0..11."""
    if path is None:
        return list(DEFAULT_CHORDS)
    with open(path) as source:
        rows = json.load(source)
    return [ChordMapping(name=row['name'], pitch_classes=tuple(row['pitch_classes']),
                         anchor_pc=row['anchor_pc'], key=row.get('key', ''),
                         midi_note=row.get('midi_note', -1), pedal_cc=row.get('pedal_cc', -1))
            for row in rows]


class ChordControls:
    """Last-held control wins; all-release clears control history, never pitches."""

    def __init__(self, primer_length: int, context_length: int,
                 mappings: Optional[List[ChordMapping]] = None) -> None:
        if context_length < 1:
            raise ValueError('context_length must be positive')
        self.context_length = context_length
        self.mappings = list(DEFAULT_CHORDS) if mappings is None else mappings
        self.keys: Dict[str, int] = {}
        self.midi_notes: Dict[int, int] = {}
        self.pedals: Dict[int, int] = {}
        self.chroma = torch.zeros((len(self.mappings) + 1, 12), dtype=torch.bool)
        self.anchors = torch.zeros(len(self.mappings) + 1, dtype=torch.long)
        for index, chord in enumerate(self.mappings, 1):
            if (not chord.pitch_classes or any(pc < 0 or pc > 11 for pc in chord.pitch_classes)
                    or chord.anchor_pc not in chord.pitch_classes):
                raise ValueError('Each chord requires pitch classes 0..11 and an anchor among its tones')
            self.chroma[index, list(chord.pitch_classes)] = True
            self.anchors[index] = chord.anchor_pc
            if chord.key:
                if chord.key in self.keys:
                    raise ValueError('Chord keyboard mappings must be unique')
                self.keys[chord.key] = index
            for value, mapping in ((chord.midi_note, self.midi_notes), (chord.pedal_cc, self.pedals)):
                if value != -1:
                    if not 0 <= value <= 127 or value in mapping:
                        raise ValueError('MIDI chord mappings must be unique values in 0..127')
                    mapping[value] = index
        self.held: Dict[str, int] = {}
        self.history: Deque[int] = deque(maxlen=context_length)
        self.reset(primer_length)

    @property
    def target(self) -> int:
        return next(reversed(self.held.values()), 0)

    def reset(self, primer_length: int) -> None:
        self.held.clear()
        self.history = deque([0] * min(primer_length, self.context_length), maxlen=self.context_length)

    def press(self, source: str, chord: int) -> None:
        # Auto-repeat and repeated pedal values must not change priority.
        if source not in self.held:
            self.held[source] = chord

    def release(self, source: str) -> None:
        self.held.pop(source, None)
        if not self.held:
            self.history = deque([0] * len(self.history), maxlen=self.context_length)

    def handle_key(self, key: str, pressed: bool) -> bool:
        chord = self.keys.get(key)
        if chord is None:
            return False
        if pressed:
            self.press('key:' + key, chord)
        else:
            self.release('key:' + key)
        return True

    def handle_midi(self, status: int, number: int, value: int) -> bool:
        kind, channel = status & 0xF0, status & 0x0F
        if kind in (0x80, 0x90):
            chord = self.midi_notes.get(number)
            pressed = kind == 0x90 and value > 0
            source = f'note:{channel}:{number}'
        elif kind == 0xB0:
            chord = self.pedals.get(number)
            pressed = value >= 64
            source = f'cc:{channel}:{number}'
        else:
            return False
        if chord is None:
            return False
        if pressed:
            self.press(source, chord)
        else:
            self.release(source)
        return True

    def next_tokens(self, device: torch.device) -> Dict[str, Tensor]:
        ids = torch.tensor([list(self.history) + [self.target]], dtype=torch.long)
        return {'chord_chroma': self.chroma[ids].to(device),
                'chord_anchor_pc': self.anchors[ids].to(device),
                'chord_active': (ids != 0).to(device)}

    def commit_note(self) -> None:
        self.history.append(self.target)

    def status(self) -> str:
        return ('Harmony: free' if self.target == 0
                else f'Harmony suggestion: {self.mappings[self.target - 1].name} (held)')
