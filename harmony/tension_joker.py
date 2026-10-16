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
        if key in ('1', '2', '3', '4', '5'):
            self.set_level(int(key) - 1)
        elif key == ' ':
            self.set_level(None)
        elif key == '-':
            self.cfg_weight = max(0.0, self.cfg_weight - 0.5)
        elif key == '+':
            self.cfg_weight += 0.5
        else:
            return False
        return True

    def status(self, home_key: int) -> str:
        level = 'free' if self.target == TENSION_NULL else str(self.target - 1)
        return (f'Requested tension: {level} | Intensity: {self.cfg_weight:.1f}x '
                f'| Home: {home_key_name(home_key)}')


@dataclass
class PianoPerformance:
    pitches: List[int]
    starts_ms: List[float]
    durations_ms: List[float]
    velocities: List[int]
    channels: List[int]
    home_key: int = HOME_KEY_UNKNOWN
    chord_spans_ms: Optional[List[Tuple[float, float]]] = None

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
    chord_active: Dict[Tuple[int, int], Deque[int]] = {}
    chord_starts: List[float] = []
    chord_ends: List[float] = []
    tempo = 500_000
    time_ms = 0.0
    for message in mido.merge_tracks(midi.tracks):
        time_ms += mido.tick2second(message.time, midi.ticks_per_beat, tempo) * 1000.0
        if message.type == 'set_tempo':
            tempo = message.tempo
        if message.type not in ('note_on', 'note_off'):
            continue
        if annotated and message.channel == 3:
            key = (message.channel, message.note)
            if message.type == 'note_on' and message.velocity > 0:
                chord_active.setdefault(key, deque()).append(len(chord_starts))
                chord_starts.append(time_ms)
                chord_ends.append(time_ms + 1.0)
            elif chord_active.get(key):
                index = chord_active[key].popleft()
                chord_ends[index] = max(chord_starts[index] + 1.0, time_ms)
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
    for pending in chord_active.values():
        for index in pending:
            chord_ends[index] = max(chord_starts[index] + 1.0, time_ms)
    chord_ends_by_start: Dict[float, float] = {}
    for start, end in zip(chord_starts, chord_ends):
        chord_ends_by_start[start] = max(chord_ends_by_start.get(start, end), end)
    chord_spans = sorted(chord_ends_by_start.items()) or None
    # The pickle builder orders exact simultaneous onsets by descending pitch.
    order = sorted(range(len(pitches)), key=lambda index: (starts[index], -pitches[index]))
    return PianoPerformance([pitches[i] for i in order], [starts[i] for i in order],
                            [ends[i] - starts[i] for i in order],
                            [velocities[i] for i in order], [channels[i] for i in order],
                            home_key, chord_spans)


class TensionScorer:
    """Offline corpus-scale proxy, recomputed from generated notes and fixed timing.

This shares the label builder's construct/calibration, so it is not an
independent perceptual judge. Chords use fixed 500 ms chroma-template frames;
neither source chord tones nor source local-key/tension annotations are read.
"""

    def __init__(self, calibration_path: Optional[str] = None) -> None:
        import hashlib
        import json
        from pathlib import Path
        import numpy as np
        from params import QUALITY_TEMPLATES
        from . import build_key_tension_dataset as module

        root = Path(__file__).resolve().parent
        self.builder = module
        self.calibration = (json.loads(Path(calibration_path).read_text()) if calibration_path
                            else dict(module.DEFAULT_CALIBRATION))
        self.frame_ms = 500
        self.chords = [(root_pc, frozenset((root_pc + interval) % 12 for interval in intervals))
                       for root_pc in range(12) for intervals in QUALITY_TEMPLATES]
        templates = np.array([[float(pc in pcs) for pc in range(12)] for _, pcs in self.chords])
        self.templates = templates / np.linalg.norm(templates, axis=1, keepdims=True)
        self.provenance = {
            'calibration': self.calibration, 'fallback_chord_frame_ms': self.frame_ms,
            'chord_windows': 'source chord timing only; chord tones inferred from performed notes',
            'analyzer_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ('tension_joker.py', 'build_key_tension_dataset.py',
                                            'key_track.py', 'key_detect.py', 'key_benchmark.py')},
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
        windows = (performance.chord_spans_ms if performance.chord_spans_ms else
                   [(start, min(end, start + self.frame_ms))
                    for start in range(notes[0].start, end, self.frame_ms)])
        for start, stop in windows:
            start, stop = round(start), round(stop)
            if start >= end:
                break
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


def validate_tension_controls(model, performances: List[PianoPerformance], scorer: TensionScorer,
                              cfg: dict, device) -> Dict[str, object]:
    """Paired free-running targets and NULL; scoring never reads the corpus labels."""
    import torch

    primer, notes = cfg['tens_control_primer'], cfg['tens_control_notes']
    seeds, settle = cfg['tens_control_seeds'], cfg['tens_control_settle']
    if (not performances or seeds < 1 or not 1 <= primer <= model.max_seq_len
            or not 0 <= settle < notes or notes < 4
            or cfg['tens_control_temperature'] <= 0 or cfg['tens_control_guidance'] < 0):
        raise ValueError('Invalid controllability validation settings')
    levels, continuations = [], []
    model.eval()
    with torch.inference_mode():
        for index, source in enumerate(performances):
            if len(source.pitches) < primer + notes:
                raise ValueError('Controllability excerpt has insufficient timing events')
            # Batch the five levels and free, with identical sampling noise per seed.
            pitches = torch.zeros((seeds, 6, primer + notes), dtype=torch.long, device=device)
            pitches[:, :, :primer] = torch.tensor(source.pitches[:primer], device=device)
            targets = torch.zeros_like(pitches)
            targets[:, :, primer:] = torch.tensor([1, 2, 3, 4, 5, 0], device=device)[None, :, None]
            pitches, targets = pitches.flatten(0, 1), targets.flatten(0, 1)
            home = torch.full((seeds * 6,), source.home_key, dtype=torch.long, device=device)
            generator = torch.Generator(device=device).manual_seed(cfg['tens_control_seed'] + index)
            # Each generated pitch becomes the next input, so time steps are sequential.
            for position in range(primer, primer + notes):
                start = max(0, position - model.max_seq_len)
                logits = model.next_pitch_logits(
                    {'pitch': pitches[:, start:position + 1], 'home_key': home,
                     'tension_target': targets[:, start:position + 1]}, cfg['tens_control_guidance'])
                noise = torch.empty((seeds, 1, logits.size(-1)), device=device).exponential_(generator=generator)
                sampled = (logits.reshape(seeds, 6, -1) / cfg['tens_control_temperature']
                           - noise.log()).argmax(-1)
                pitches[:, position] = sampled.flatten()
            outputs = pitches.cpu().tolist()
            measured = [scorer.score(source, output)['levels'][primer:] for output in outputs]
            levels.append(torch.tensor(measured).reshape(seeds, 6, notes))
            continuations.append(pitches[:, primer:].cpu().reshape(seeds, 6, notes))

    realized = torch.cat(levels)
    generated = torch.cat(continuations)
    requested = torch.arange(NUM_TENSION_LEVELS)[None, :, None]
    held = realized[:, :5, settle:]
    means = held.float().mean(-1)
    # Average tied ranks; compute per primer/seed so different home keys cannot confound ordering.
    ranks = ((means[:, :, None] > means[:, None, :]).sum(-1)
             + ((means[:, :, None] == means[:, None, :]).sum(-1) - 1) / 2)
    centered = ranks - ranks.mean(-1, keepdim=True)
    denominator = (centered.square().sum(-1) * 10).sqrt()
    rho = ((centered * torch.arange(-2, 3)).sum(-1) / denominator.clamp_min(1e-12)).mean().item()
    within = ((held - requested).abs() <= 1).float().mean().item()
    repeats = (generated[:, :, 1:] == generated[:, :, :-1]).float().mean(-1)
    grams = generated.unfold(-1, 4, 1)
    repeated_grams = torch.tensor([1 - len(torch.unique(row, dim=0)) / row.size(0)
                                   for row in grams.flatten(0, 1)]).reshape(-1, 6)
    close = (realized[:, :5] - requested).abs() <= 1
    settled = close.unfold(-1, 3, 1).all(-1)[:, :, :settle + 1].any(-1)
    confusion = torch.bincount((requested.expand_as(held) * 5 + held).flatten(), minlength=25).reshape(5, 5)
    result = {
        'control_score': max(0.0, rho) * within,
        'control_spearman': rho,
        'control_within_one': within,
        'control_mae': (held - requested).abs().float().mean().item(),
        'control_span': (means[:, 4] - means[:, 0]).mean().item(),
        'control_settled_fraction': settled.float().mean().item(),
        'control_repeat_rate': repeats[:, :5].mean().item(),
        'control_free_repeat_rate': repeats[:, 5].mean().item(),
        'control_repeated_4gram_rate': repeated_grams[:, :5].mean().item(),
        'control_free_repeated_4gram_rate': repeated_grams[:, 5].mean().item(),
        'control_confusion_matrix': confusion.tolist(),
        'control_cases': means.size(0),
        'control_acceptance_met': rho >= 0.6 and within >= 0.7,
    }
    result.update({f'control_mean_level_{level}': value for level, value in enumerate(means.mean(0).tolist())})
    return result
