"""Shared performance I/O and latched controls for the tension joker.

MIDI timing stays outside the pitch model. Annotated corpus MIDIs contribute
only performed notes, never the analyzer's synthetic chord tones.
"""

from collections import deque
from dataclasses import dataclass
import re
from typing import Deque, Dict, List, Optional, Tuple

import mido

from params import HOME_KEY_UNKNOWN, NUM_TENSION_LEVELS, TENSION_NULL
from roman_movement import key_from_marker


def parse_home_key(text: str) -> int:
    value = text.strip()
    if re.match(r'^home\s*:', value, re.IGNORECASE):
        value = value.split(':', 1)[1].strip()
    if not value.lower().startswith('key:'):
        value = 'key: ' + value
    key = key_from_marker(value)
    if key is None:
        raise ValueError(f'Invalid home key {text!r}; expected, for example, C major or F# minor')
    return key[0] + 12 * key[1]


def home_key_name(key: int) -> str:
    if key == HOME_KEY_UNKNOWN:
        return 'unknown'
    return ('C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B')[key % 12] + (
        ' minor' if key >= 12 else ' major')


class TensionControls:
    """Control history since the last release, cropped with the pitch context."""

    def __init__(self, primer_length: int, context_length: int) -> None:
        self.context_length = context_length
        self.history: Deque[int] = deque(maxlen=context_length)
        self.target = TENSION_NULL
        self.cfg_weight = 1.0
        self.reset(primer_length)

    def reset(self, primer_length: int) -> None:
        self.history = deque([TENSION_NULL] * min(primer_length, self.context_length),
                             maxlen=self.context_length)
        self.target = TENSION_NULL

    def set_level(self, level: Optional[int]) -> None:
        if level is not None and not 0 <= level < NUM_TENSION_LEVELS:
            raise ValueError('Tension level must be 0..4 or None for free')
        self.target = TENSION_NULL if level is None else level + 1
        if level is None:
            # A gate at the newest position alone cannot erase earlier attention effects.
            self.history = deque([TENSION_NULL] * len(self.history), maxlen=self.context_length)

    def next_targets(self) -> List[int]:
        return list(self.history) + [self.target]

    def commit_note(self) -> None:
        self.history.append(self.target)

    def handle_key(self, key: str) -> bool:
        if key in ('0', '1', '2', '3', '4'):
            self.set_level(int(key))
        elif key == ' ':
            self.set_level(None)
        elif key == '[':
            self.cfg_weight = max(0.0, self.cfg_weight - 0.5)
        elif key == ']':
            self.cfg_weight += 0.5
        else:
            return False
        return True

    def status(self, home_key: int) -> str:
        level = 'free' if self.target == TENSION_NULL else str(self.target - 1)
        return f'Requested tension: {level} | Home: {home_key_name(home_key)} | Guidance: {self.cfg_weight:.1f}'


@dataclass
class PianoPerformance:
    pitches: List[int]
    starts_ms: List[float]
    durations_ms: List[float]
    velocities: List[int]
    channels: List[int]
    home_key: int = HOME_KEY_UNKNOWN

    def tokens(self) -> Dict[str, List[int]]:
        # The live visualizer and recorder use 32 ms units; offline scoring uses
        # the unquantized arrays above, including long gaps and note overlaps.
        starts = [int(time / 32) for time in self.starts_ms]
        previous = [starts[0]] + starts[:-1] if starts else []
        return {'pitch': self.pitches.copy(),
                'dtime': [min(127, max(0, time - past)) for time, past in zip(starts, previous)],
                'dur': [min(127, max(1, int(duration / 32))) for duration in self.durations_ms],
                'vel': self.velocities.copy(), 'chan': self.channels.copy()}

    def write(self, path: str, pitches: List[int]) -> None:
        if len(pitches) > len(self.pitches):
            raise ValueError('More generated pitches than source timing events')
        midi = mido.MidiFile(ticks_per_beat=1000)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo', tempo=1_000_000))  # one tick = one ms
        events = []
        for index, pitch in enumerate(pitches):
            start = round(self.starts_ms[index])
            end = max(start + 1, round(self.starts_ms[index] + self.durations_ms[index]))
            channel = self.channels[index]
            events.append((start, 1, index, mido.Message('note_on', note=pitch,
                           velocity=self.velocities[index], channel=channel)))
            events.append((end, 0, index, mido.Message('note_off', note=pitch,
                           velocity=0, channel=channel)))
        previous = 0
        for tick, _, _, message in sorted(events, key=lambda event: event[:3]):
            track.append(message.copy(time=tick - previous))
            previous = tick
        midi.save(path)


def read_tension_performance(path: str, home_override: Optional[str] = None) -> PianoPerformance:
    midi = mido.MidiFile(path)
    annotated = False
    home_key = HOME_KEY_UNKNOWN
    for track in midi.tracks:
        for message in track:
            if message.type in ('marker', 'text', 'lyrics', 'cue_marker'):
                text = message.text.strip()
                if re.match(r'^(home|tension)\s*:', text, re.IGNORECASE):
                    annotated = True
                if re.match(r'^home\s*:', text, re.IGNORECASE):
                    parsed = parse_home_key(text)
                    if home_key != HOME_KEY_UNKNOWN and home_key != parsed:
                        raise ValueError('Conflicting home-key markers in primer')
                    home_key = parsed
    if home_key == HOME_KEY_UNKNOWN and home_override:
        home_key = parse_home_key(home_override)

    starts: List[float] = []
    ends: List[float] = []
    pitches: List[int] = []
    velocities: List[int] = []
    channels: List[int] = []
    active: Dict[Tuple[int, int], Deque[int]] = {}
    tempo = 500_000
    time_ms = 0.0
    for message in mido.merge_tracks(midi.tracks):
        time_ms += mido.tick2second(message.time, midi.ticks_per_beat, tempo) * 1000.0
        if message.type == 'set_tempo':
            tempo = message.tempo
        if message.type not in ('note_on', 'note_off'):
            continue
        if message.channel == 9 or (annotated and message.channel not in (0, 1, 10, 11)):
            continue
        key = (message.channel, message.note)
        if message.type == 'note_on' and message.velocity > 0:
            active.setdefault(key, deque()).append(len(pitches))
            pitches.append(message.note)
            starts.append(time_ms)
            ends.append(time_ms + 1.0)
            velocities.append(message.velocity)
            channels.append(message.channel)
        elif active.get(key):
            index = active[key].popleft()
            ends[index] = max(starts[index] + 1.0, time_ms)
    for pending in active.values():
        for index in pending:
            ends[index] = max(starts[index] + 1.0, time_ms)
    # The pickle builder orders exact simultaneous onsets by descending pitch.
    order = sorted(range(len(pitches)), key=lambda index: (starts[index], -pitches[index]))
    return PianoPerformance([pitches[i] for i in order], [starts[i] for i in order],
                            [ends[i] - starts[i] for i in order],
                            [velocities[i] for i in order], [channels[i] for i in order], home_key)


class TensionScorer:
    """Offline corpus-scale proxy, recomputed from generated notes and fixed timing.

This shares the label builder's construct/calibration, so it is not an
independent perceptual judge. Chords use fixed 500 ms chroma-template frames;
neither source chord tones nor source local-key/tension annotations are read.
"""

    def __init__(self, analyzer_root: str, calibration_path: Optional[str] = None) -> None:
        import hashlib
        import importlib.util
        import json
        from pathlib import Path
        import sys
        import numpy as np
        from params import QUALITY_TEMPLATES

        root = Path(analyzer_root).resolve()
        path = root / 'build_key_tension_dataset.py'
        if not path.is_file():
            raise FileNotFoundError(f'Tension evaluator not found: {path}; set --analyzer-root')
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location('_joker_tension_builder', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.builder = module
        self.calibration = (json.loads(Path(calibration_path).read_text()) if calibration_path
                            else dict(module.DEFAULT_CALIBRATION))
        self.frame_ms = 500
        self.chords = [(root_pc, frozenset((root_pc + interval) % 12 for interval in intervals))
                       for root_pc in range(12) for intervals in QUALITY_TEMPLATES]
        templates = np.array([[float(pc in pcs) for pc in range(12)] for _, pcs in self.chords])
        self.templates = templates / np.linalg.norm(templates, axis=1, keepdims=True)
        self.provenance = {
            'calibration': self.calibration, 'chord_frame_ms': self.frame_ms,
            'analyzer_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ('build_key_tension_dataset.py', 'key_track.py', 'key_detect.py')},
            'measurement': 'corpus-scale proxy; generated chroma-template chords and recomputed local keys',
        }

    def score(self, performance: PianoPerformance, pitches: List[int]) -> Dict[str, object]:
        import numpy as np

        if performance.home_key == HOME_KEY_UNKNOWN:
            raise ValueError('Scoring requires a fixed home key; provide a home marker or --home-key')
        if not pitches:
            raise ValueError('Cannot score an empty performance')
        builder = self.builder
        notes = [builder.Note(round(performance.starts_ms[index]),
                              max(round(performance.starts_ms[index]) + 1,
                                  round(performance.starts_ms[index] + performance.durations_ms[index])),
                              pitch, performance.velocities[index], performance.channels[index])
                 for index, pitch in enumerate(pitches)]
        reader = builder.ChromaReader(notes)
        end = max(note.end for note in notes)
        spans = []
        for start in range(notes[0].start, end, self.frame_ms):
            stop = min(end, start + self.frame_ms)
            chroma = reader.chroma(start, stop)
            if chroma.sum() <= 0:
                continue
            root, pcs = self.chords[int(np.argmax(self.templates @ chroma))]
            spans.append(builder.ChordSpan(start, stop, pcs, root))
        home = (performance.home_key % 12, 'm' if performance.home_key >= 12 else 'M')
        keys = builder.analyze_keys(notes, spans, home)
        # The tracker can reject a supplied home when generated notes move far away.
        # Keep its generated local-key path, but always measure against the requested reference.
        keys.home = home
        for segment in keys.segments:
            if segment.collection == builder.key_collection(home):
                segment.key = home
        rows = builder.tension_indicators(reader, spans, keys)
        composite, levels = builder.tension_levels(rows, self.calibration)
        frame_indices = np.searchsorted([span.start for span in spans],
                                        performance.starts_ms[:len(pitches)], side='right') - 1
        frame_indices = np.clip(frame_indices, 0, len(spans) - 1)
        return {'levels': np.asarray(levels)[frame_indices].tolist(),
                'composite': composite[frame_indices].tolist(),
                'frame_indicators': rows}
