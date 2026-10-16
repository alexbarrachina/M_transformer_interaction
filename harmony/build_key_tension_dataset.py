#!/usr/bin/env python3
"""Rebuild giantMIDI/all_roman_time_fixed at real speed with key and tension markers.

Input: the `*_with_chords.mid` files of `all_roman_time_fixed` (performance notes
on channels 1/2/11, java chords on channel 4, musiclang roman + java key
markers). Output: one MIDI per input, same file name, holding

  * every note of the input - channel 4 chords included - at the performance's
    real speed, 1 tick = 1 ms (1000 ppq, 60 bpm), velocities restored from the
    source GiantMIDI file (the channel-segmentation step flattened them to 90);
  * none of the input markers (java keys and musiclang romans are not reliable);
  * a "Key and Tension" track of markers recomputed from the notes:
        home: C major                    tick 0, the piece's home key
        key: C major                     tick 0 and at every local key change
        change key: C major -> G major   at every local key change
        tension: 0..4                    at every channel-4 chord onset

Speed. all_roman_time_fixed stores 1 source beat as 1000 ticks at 1 s/beat, and
every GiantMIDI source is 384 ppq / 120 bpm, so the files play at half speed.
Each file's mapping (fixed tick -> source beat -> source tempo map -> ms) is
checked against the source notes; a file that already runs in real time keeps
an identity mapping.

Keys. `key_track` on the performance notes only (the channel-4 chord tones
would bias the chroma towards the java analysis): Viterbi over diatonic
collections, then
  * the home key is the filename key ("... in F minor") when the music
    supports it - its collection must carry >= MIN_FILENAME_SUPPORT of the
    onset-weighted path - otherwise key_track's global key. Every decision is
    written to the report CSV (verified / corrected / rejected);
  * a small emission bonus for the home collection (HOME_PRIOR), a switch
    penalty of KEY_SWITCH_PENALTY and a minimum segment of MIN_SEGMENT_MS
    keep tonicisations from reading as modulations. Tuned on 500 filename-keyed
    pieces: ~1.1 changes per real minute, median segment ~30 s, 93% of
    boundaries have the new key fitting the notes better after than before;
  * a segment on the home collection takes the home key's mode; each key
    boundary is moved to the channel-4 chord onset (inside the window overlap)
    where the old->new key evidence flips most.

Tension. One score per channel-4 chord, the mean of three standardised
indicators that fail in different ways:
  1. tps   - Lerdahl's Tonal Pitch Space hierarchical distance: region distance
             home tonic -> local tonic, plus chord distance local tonic -> the
             chord (delta = i + j + k on the basic space). Uses the chord labels
             and the key track. I=0, V=IV=5, vi=7, ii=8, V/V=9; a G-major region
             in C adds 7.
  2. tonal - 1 - Krumhansl-Kessler correlation between the chord span's
             performed chroma and the home key. Note-based, no chord labels.
  3. rough - interval-class roughness of the performed chroma in the span
             (sensory dissonance). Note-based, key-independent.
tps and tonal measure distance from home, roughness measures surface dissonance -
the two components of Lerdahl & Krumhansl's tension model. Each indicator is
robust-z standardised with corpus statistics, clipped to +-3, averaged, smoothed
with a duration-weighted centred window (SMOOTH_MS) and cut into TENSION_LEVELS
equal-population levels at corpus quantiles, with HYSTERESIS so a score hovering
on a cut point does not flicker. Statistics, cut points and both settings come
from `--calibrate` (DEFAULT_CALIBRATION: 1500 pieces, 426k chords).

Validation on pieces outside the calibration sample:
  * every indicator is lower on a piece's final chord (within-piece z
    -0.6 / -0.6 / -0.7); ~78% of pieces end on level 0-1;
  * dominant -> tonic in the local key: level falls 54%, stays 42%, rises 4%;
    tonic -> dominant: rises 45%, falls 6%. A falling level is a resolution,
    so no separate resolve marker is written;
  * levels are lower before phrase-end gaps (>= 1.5 s) than on average;
  * replacing the channel-4 chords by an independent chroma-template chord
    leaves 94% of levels unchanged and 100% within one level, so the label
    errors of the java analysis barely move the tension;
  * the weak point is the home key: where the notes and the title disagree
    (9% of titled pieces) only 34% of levels match (85% within one). Titled
    pieces use the title; untitled ones depend on key_track (~83% correct).
Cronbach's alpha of the three indicators is ~0.61 (0.64 smoothed): moderate,
as expected for indicators of two distinct facets (distance vs dissonance).

Usage:

    python3 build_key_tension_dataset.py                       # whole corpus
    python3 build_key_tension_dataset.py --limit 20 --force    # quick look
    python3 build_key_tension_dataset.py --calibrate 1500      # re-fit tension stats

Sidecar JSON per file (keys, segments, per-chord indicators) and report.csv go
to --analysis-dir, kept out of the MIDI folder so dataset loaders that walk it
only see MIDI files.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import random
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = ROOT_DIR / ".vendor"
if VENDOR_DIR.is_dir():
    # Appended, never inserted: .vendor holds a broken numpy source checkout.
    sys.path.append(str(VENDOR_DIR))

import mido
import numpy as np

from . import key_detect, key_track
from .key_benchmark import parse_ground_truth

GIANTMIDI_DIR = Path("/Volumes/DADES/DataSets/MIDI/giantMIDI")
DEFAULT_INPUT_DIR = GIANTMIDI_DIR / "all_roman_time_fixed"
DEFAULT_SOURCE_DIR = GIANTMIDI_DIR / "midis"
DEFAULT_OUTPUT_DIR = GIANTMIDI_DIR / "all_key_tension"
INPUT_SUFFIX = "_with_chords"

CHORD_CHANNEL = 3                    # MIDI channel 4
CHORD_TRACK_NAME = "Java Chords"
MARKER_TRACK_NAME = "Key and Tension"
FIXED_TICKS_PER_BEAT = 1000          # all_roman_time_fixed: 1 source beat == 1000 ticks
OUT_PPQ = 1000
OUT_TEMPO = 1_000_000                # with OUT_PPQ: 1 tick == 1 ms
DROPPED_META = {"marker", "text", "lyrics", "cue_marker"}
SPEED_MATCH_TOLERANCE_MS = 3
VELOCITY_MATCH_TOLERANCE_MS = 3

KEY_SWITCH_PENALTY = 3.0
HOME_PRIOR = 0.03                    # bonus added to the home collection's window score
MIN_SEGMENT_MS = 8000
MIN_FILENAME_SUPPORT = 0.10          # share of the path the filename key's collection must hold
BOUNDARY_EVIDENCE_MS = 6000          # chroma window either side when placing a key boundary

TENSION_LEVELS = 5
SMOOTH_MS = 1000                     # duration-weighted window centred on each chord
HYSTERESIS = 0.08                    # composite units a level must be overshot by to change
MIN_SPAN_MS = 400                    # shorter chord spans read chroma from a centred 400 ms window
INDICATORS = ("tps", "tonal", "rough")

# Measured with --calibrate 1500 on all_roman_time_fixed (see module docstring).
DEFAULT_CALIBRATION: Dict[str, object] = {
    "center": {"tps": 11.0, "tonal": 0.7358, "rough": 0.1348},
    "scale": {"tps": 8.8955, "tonal": 0.4197, "rough": 0.0552},
    "thresholds": [-0.5324, -0.1477, 0.2011, 0.6229],
    "smooth_ms": SMOOTH_MS,
    "hysteresis": HYSTERESIS,
}

PC_OF_LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
MAJOR_NAMES = key_detect.MAJOR_NAMES
MINOR_NAMES = key_detect.MINOR_NAMES
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
HARMONIC_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 11)
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# Sensory roughness per interval class (ic1..ic6): seconds and tritones rub, thirds/sixths barely.
INTERVAL_CLASS_ROUGHNESS = (0.0, 1.0, 0.45, 0.15, 0.1, 0.05, 0.7)

Key = Tuple[int, str]                # (tonic pc, 'M' | 'm')


def key_label(key: Key) -> str:
    names = MAJOR_NAMES if key[1] == "M" else MINOR_NAMES
    return f"{names[key[0]]} {'major' if key[1] == 'M' else 'minor'}"


def key_collection(key: Key) -> int:
    """Relative-major pc of the key's diatonic set (C major and A minor -> 0)."""
    return key[0] if key[1] == "M" else (key[0] + 3) % 12


def key_relation(reference: Key, other: Optional[Key]) -> str:
    if other is None:
        return "none"
    if reference == other:
        return "same"
    if key_collection(reference) == key_collection(other):
        return "relative"
    if reference[0] == other[0]:
        return "parallel"
    if reference[1] == other[1] and (other[0] - reference[0]) % 12 == 7:
        return "dominant"
    if reference[1] == other[1] and (other[0] - reference[0]) % 12 == 5:
        return "subdominant"
    return "other"


# --------------------------------------------------------------------------- #
# notes and time
# --------------------------------------------------------------------------- #

@dataclass
class Note:
    start: int
    end: int
    pitch: int
    velocity: int
    channel: int


def read_track(track: mido.MidiTrack) -> Tuple[List[Note], List[Tuple[int, mido.Message]]]:
    """Split a track into paired notes and the remaining (tick, message) events."""
    notes: List[Note] = []
    others: List[Tuple[int, mido.Message]] = []
    pending: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    tick = 0
    for msg in track:
        tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            pending[(msg.channel, msg.note)].append((tick, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            starts = pending.get((msg.channel, msg.note))
            if starts:
                start, velocity = starts.pop(0)
                notes.append(Note(start, tick, msg.note, velocity, msg.channel))
        elif msg.type != "end_of_track":
            others.append((tick, msg))
    for (channel, pitch), starts in pending.items():      # unterminated notes
        for start, velocity in starts:
            notes.append(Note(start, max(tick, start + 1), pitch, velocity, channel))
    return notes, others


class SourceClock:
    """Source beat -> real milliseconds through the source file's tempo map."""

    def __init__(self, midi: Optional[mido.MidiFile]):
        self.ppq = midi.ticks_per_beat if midi is not None else 384
        anchors: List[Tuple[int, float, int]] = [(0, 0.0, 500000)]
        if midi is not None:
            tick = 0
            for msg in mido.merge_tracks(midi.tracks):
                tick += msg.time
                if msg.type == "set_tempo":
                    last_tick, last_ms, last_tempo = anchors[-1]
                    ms = last_ms + (tick - last_tick) * last_tempo / (1000.0 * self.ppq)
                    if tick == last_tick:
                        anchors[-1] = (tick, ms, msg.tempo)
                    else:
                        anchors.append((tick, ms, msg.tempo))
        self._anchors = anchors
        self._ticks = [anchor[0] for anchor in anchors]

    def ms_at_source_tick(self, tick: float) -> float:
        index = max(0, bisect.bisect_right(self._ticks, tick) - 1)
        anchor_tick, anchor_ms, tempo = self._anchors[index]
        return anchor_ms + (tick - anchor_tick) * tempo / (1000.0 * self.ppq)

    def ms_at_fixed_tick(self, tick: int) -> float:
        return self.ms_at_source_tick(tick * self.ppq / FIXED_TICKS_PER_BEAT)


def source_index(midi: mido.MidiFile, clock: SourceClock) -> Dict[int, Tuple[List[float], List[int]]]:
    """pitch -> (sorted onset ms, velocities) of the source performance."""
    by_pitch: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for track in midi.tracks:
        notes, _ = read_track(track)
        for note in notes:
            by_pitch[note.pitch].append((clock.ms_at_source_tick(note.start), note.velocity))
    index = {}
    for pitch, items in by_pitch.items():
        items.sort()
        index[pitch] = ([item[0] for item in items], [item[1] for item in items])
    return index


def nearest_source(index, pitch: int, ms: float, tolerance: float) -> Optional[int]:
    """Position of the closest same-pitch source onset within tolerance, else None."""
    entry = index.get(pitch)
    if not entry:
        return None
    onsets = entry[0]
    position = bisect.bisect_left(onsets, ms)
    best = None
    for candidate in (position - 1, position):
        if 0 <= candidate < len(onsets) and abs(onsets[candidate] - ms) <= tolerance:
            if best is None or abs(onsets[candidate] - ms) < abs(onsets[best] - ms):
                best = candidate
    return best


def match_rate(notes: Sequence[Note], to_ms, index) -> float:
    if not notes or not index:
        return 0.0
    hits = sum(
        1 for note in notes
        if nearest_source(index, note.pitch, to_ms(note.start), SPEED_MATCH_TOLERANCE_MS) is not None
    )
    return hits / len(notes)


# --------------------------------------------------------------------------- #
# retiming
# --------------------------------------------------------------------------- #

@dataclass
class Retimed:
    midi: mido.MidiFile
    music: List[Note]                 # performance notes (not channel 4), output ms
    chords: List[Note]                # channel-4 chord tones, output ms
    mapping: str                      # 'source-beats' | 'identity' | 'assumed-120bpm'
    speed_match: float                # share of performance onsets found in the source
    velocity_restored: float          # share of performance notes given a source velocity
    pedal_events: int


def retime(
    fixed: mido.MidiFile,
    source: Optional[mido.MidiFile],
    restore_velocity: bool = True,
    copy_pedal: bool = False,
) -> Retimed:
    clock = SourceClock(source)
    index = source_index(source, clock) if source is not None else {}
    parsed = [(track, *read_track(track)) for track in fixed.tracks]
    performance = [n for _, notes, _ in parsed for n in notes if n.channel != CHORD_CHANNEL]

    # "If needed": the beat mapping is expected, but a file already in real time
    # (1 tick == 1 ms of the source) keeps an identity mapping.
    if source is not None:
        sample = performance[:: max(1, len(performance) // 400)]
        beat_rate = match_rate(sample, clock.ms_at_fixed_tick, index)
        identity_rate = match_rate(sample, lambda tick: float(tick) * fixed.ticks_per_beat / OUT_PPQ, index)
        if identity_rate > beat_rate:
            mapping, rate = "identity", identity_rate
            to_ms = lambda tick: float(tick) * OUT_PPQ / fixed.ticks_per_beat  # noqa: E731
        else:
            mapping, rate = "source-beats", beat_rate
            to_ms = clock.ms_at_fixed_tick
    else:
        mapping, rate = "assumed-120bpm", 0.0
        to_ms = clock.ms_at_fixed_tick

    out = mido.MidiFile(type=1, ticks_per_beat=OUT_PPQ)
    music: List[Note] = []
    chords: List[Note] = []
    restored = 0
    for position, (track, notes, others) in enumerate(parsed):
        events: List[Tuple[int, int, mido.Message]] = []    # (tick, order, msg)
        for tick, msg in others:
            if msg.is_meta and msg.type in DROPPED_META:
                continue
            if msg.is_meta and msg.type == "set_tempo":
                continue
            events.append((int(round(to_ms(tick))), 0, msg))
        if position == 0:
            events.append((0, -1, mido.MetaMessage("set_tempo", tempo=OUT_TEMPO, time=0)))
        for note in notes:
            start = int(round(to_ms(note.start)))
            end = max(start + 1, int(round(to_ms(note.end))))
            velocity = note.velocity
            if restore_velocity and note.channel != CHORD_CHANNEL and index:
                hit = nearest_source(index, note.pitch, to_ms(note.start), VELOCITY_MATCH_TOLERANCE_MS)
                if hit is not None:
                    velocity = index[note.pitch][1][hit]
                    restored += 1
            retimed = Note(start, end, note.pitch, velocity, note.channel)
            (chords if note.channel == CHORD_CHANNEL else music).append(retimed)
            events.append((start, 2, mido.Message("note_on", channel=note.channel, note=note.pitch, velocity=velocity)))
            events.append((end, 1, mido.Message("note_off", channel=note.channel, note=note.pitch, velocity=0)))
        out.tracks.append(build_track(events))

    pedal = 0
    if copy_pedal and source is not None:
        events = []
        for track in source.tracks:
            tick = 0
            for msg in track:
                tick += msg.time
                if msg.type == "control_change" and msg.control in (64, 66, 67):
                    events.append((int(round(clock.ms_at_source_tick(tick))), 0, msg.copy(channel=0)))
        if events:
            pedal = len(events)
            pedal_track = build_track(events)
            pedal_track.insert(0, mido.MetaMessage("track_name", name="Source Pedal", time=0))
            out.tracks.append(pedal_track)

    music.sort(key=lambda n: (n.start, n.pitch))
    chords.sort(key=lambda n: (n.start, n.pitch))
    return Retimed(
        midi=out,
        music=music,
        chords=chords,
        mapping=mapping,
        speed_match=rate,
        velocity_restored=restored / len(music) if music else 0.0,
        pedal_events=pedal,
    )


def build_track(events: List[Tuple[int, int, mido.Message]]) -> mido.MidiTrack:
    """Absolute (tick, order, msg) events -> delta-timed track. At equal ticks:
    metas/controls first, then note_offs, then note_ons (a re-struck pitch is
    released before it sounds again)."""
    track = mido.MidiTrack()
    last = 0
    for tick, _, msg in sorted(events, key=lambda item: (item[0], item[1])):
        track.append(msg.copy(time=tick - last))
        last = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


# --------------------------------------------------------------------------- #
# chord spans
# --------------------------------------------------------------------------- #

@dataclass
class ChordSpan:
    start: int
    end: int
    pcs: frozenset
    root: int


def chord_spans(chords: Sequence[Note]) -> List[ChordSpan]:
    """Channel-4 tones grouped by onset. add_chords_from_java writes each chord in
    root position (root + ascending intervals), so the lowest tone is the root."""
    groups: Dict[int, List[Note]] = defaultdict(list)
    for note in chords:
        groups[note.start].append(note)
    spans = []
    for start, group in sorted(groups.items()):
        spans.append(ChordSpan(
            start=start,
            end=max(n.end for n in group),
            pcs=frozenset(n.pitch % 12 for n in group),
            root=min(group, key=lambda n: n.pitch).pitch % 12,
        ))
    return spans


class ChromaReader:
    """Duration-weighted performed chroma over arbitrary [start, end) ranges."""

    def __init__(self, music: Sequence[Note]):
        self.notes = sorted(music, key=lambda n: n.start)
        self.starts = [n.start for n in self.notes]
        self.longest = max((n.end - n.start for n in self.notes), default=0)

    def chroma(self, start: int, end: int) -> np.ndarray:
        result = np.zeros(12)
        low = bisect.bisect_left(self.starts, start - self.longest)
        high = bisect.bisect_left(self.starts, end)
        for note in self.notes[low:high]:
            overlap = min(note.end, end) - max(note.start, start)
            if overlap > 0:
                result[note.pitch % 12] += overlap
        return result


# --------------------------------------------------------------------------- #
# key tracking
# --------------------------------------------------------------------------- #

@dataclass
class KeySegment:
    start: int
    end: int
    key: Key
    collection: int


@dataclass
class KeyAnalysis:
    home: Key
    home_source: str                  # 'filename' | 'notes'
    key_check: str                    # verified / corrected-<relation> / rejected-<relation> / no-filename-key
    filename_key: Optional[Key]
    notes_key: Optional[Key]
    filename_support: Optional[float]
    segments: List[KeySegment]


def frames_from_notes(music: Sequence[Note]) -> List[key_track.Frame]:
    """key_track frames (NOTES_PER_FRAME onsets each) built straight from notes."""
    ordered = sorted(music, key=lambda n: (n.start, n.pitch))
    frames = []
    step = key_track.NOTES_PER_FRAME
    for index in range(0, len(ordered), step):
        group = ordered[index : index + step]
        chroma = [0.0] * 12
        for note in group:
            chroma[note.pitch % 12] += note.end - note.start
        frames.append(key_track.Frame(group[0].start, max(n.end for n in group), chroma))
    return frames


def path_runs(path: Sequence[int]) -> List[Tuple[int, int]]:
    runs = []
    index = 0
    while index < len(path):
        stop = index
        while stop + 1 < len(path) and path[stop + 1] == path[index]:
            stop += 1
        runs.append((index, stop))
        index = stop + 1
    return runs


def merge_short_runs(windows, path: Sequence[int], min_ms: int) -> List[int]:
    """Absorb runs shorter than min_ms (start-to-next-start) into the neighbour
    whose collection scores best over the run's windows, shortest run first."""
    path = list(path)
    while True:
        runs = path_runs(path)
        if len(runs) <= 1:
            return path
        spans = []
        for position, (first, _) in enumerate(runs):
            begin = windows[first].start_tick if position else windows[0].start_tick
            finish = windows[runs[position + 1][0]].start_tick if position + 1 < len(runs) else windows[-1].end_tick
            spans.append(finish - begin)
        shortest = int(np.argmin(spans))
        if spans[shortest] >= min_ms:
            return path
        first, last = runs[shortest]
        neighbours = []
        if shortest > 0:
            neighbours.append(path[runs[shortest - 1][0]])
        if shortest + 1 < len(runs):
            neighbours.append(path[runs[shortest + 1][0]])
        best = max(neighbours, key=lambda c: sum(windows[w].scores[c] for w in range(first, last + 1)))
        for w in range(first, last + 1):
            path[w] = best


def ks_correlation(chroma: np.ndarray, key: Key) -> float:
    profile = np.roll(KK_MAJOR if key[1] == "M" else KK_MINOR, key[0])
    if chroma.sum() <= 0 or chroma.std() == 0:
        return 0.0
    return float(np.corrcoef(chroma, profile)[0, 1])


def place_boundary(reader: ChromaReader, low: int, high: int, old: Key, new: Key,
                   chord_onsets: Sequence[int], note_onsets: Sequence[int]) -> int:
    """Pick the chord onset in [low, high] where the old->new key evidence flips
    most; without one, the note onset nearest the middle of the range."""
    candidates = [t for t in chord_onsets if low <= t <= high]
    if not candidates:
        middle = (low + high) // 2
        position = bisect.bisect_left(note_onsets, middle)
        options = [note_onsets[p] for p in (position - 1, position) if 0 <= p < len(note_onsets)]
        return min(options, key=lambda t: abs(t - middle)) if options else middle

    def flip(t: int) -> float:
        before = reader.chroma(t - BOUNDARY_EVIDENCE_MS, t)
        after = reader.chroma(t, t + BOUNDARY_EVIDENCE_MS)
        return (ks_correlation(after, new) - ks_correlation(after, old)) - (
            ks_correlation(before, new) - ks_correlation(before, old))

    return max(candidates, key=flip)


def global_rotation(music: Sequence[Note], collection: int) -> Key:
    """Relative major/minor choice for the global collection (key_detect's vote)."""
    midi = mido.MidiFile(type=0, ticks_per_beat=OUT_PPQ)
    events = [(0, -1, mido.MetaMessage("set_tempo", tempo=OUT_TEMPO, time=0))]
    for note in music:
        events.append((note.start, 2, mido.Message("note_on", note=note.pitch, velocity=64)))
        events.append((note.end, 1, mido.Message("note_off", note=note.pitch, velocity=0)))
    midi.tracks.append(build_track(events))
    with tempfile.NamedTemporaryFile(suffix=".mid") as handle:
        midi.save(handle.name)
        features = key_detect.extract_midi_features(Path(handle.name))
    if features is None:
        return (collection, "M")
    (tonic_pc, mode), _ = key_detect.choose_rotation(features, collection)
    return (tonic_pc, mode)


def analyze_keys(music: Sequence[Note], spans: Sequence[ChordSpan], filename_key: Optional[Key]) -> KeyAnalysis:
    frames = frames_from_notes(music)
    windows = key_track.build_windows(frames)
    key_track.score_windows(windows)
    raw_scores = [list(w.scores) for w in windows]

    if not windows:
        raise ValueError("too few notes for key tracking")

    # Pass 1: plain key_track with its benchmarked settings - what the notes alone say.
    path = key_track.viterbi_path(windows, key_track.DEFAULT_SWITCH_PENALTY)
    global_collection, _ = key_track.global_collection_vote(windows, path)
    notes_key = global_rotation(music, global_collection)

    onsets = [w.onset_count for w in windows]
    support = None
    if filename_key is not None:
        target = key_collection(filename_key)
        support = sum(o for o, c in zip(onsets, path) if c == target) / (sum(onsets) or 1)
        relation = key_relation(filename_key, notes_key)
        if relation == "same":
            home, home_source, check = filename_key, "filename", "verified"
        elif support >= MIN_FILENAME_SUPPORT:
            home, home_source, check = filename_key, "filename", f"corrected-{relation}"
        else:
            home, home_source, check = notes_key, "notes", f"rejected-{relation}"
    else:
        home, home_source, check = notes_key, "notes", "no-filename-key"

    # Pass 2: home prior + minimum segment length.
    home_collection = key_collection(home)
    for window, scores in zip(windows, raw_scores):
        window.scores = list(scores)
        window.scores[home_collection] += HOME_PRIOR
    path = key_track.viterbi_path(windows, KEY_SWITCH_PENALTY)
    path = merge_short_runs(windows, path, MIN_SEGMENT_MS)

    reader = ChromaReader(music)
    chord_onsets = [span.start for span in spans]
    note_onsets = sorted({n.start for n in music})
    runs = path_runs(path)
    keys: List[Key] = []
    for first, last in runs:
        collection = path[first]
        if collection == home_collection:
            keys.append(home)
        else:
            chroma = [0.0] * 12
            for window in windows[first : last + 1]:
                for pc in range(12):
                    chroma[pc] += window.chroma[pc]
            keys.append(key_track.segment_rotation(chroma, collection))

    starts = [music[0].start if music else 0]
    for position in range(1, len(runs)):
        previous_last = runs[position - 1][1]
        first = runs[position][0]
        low = windows[first].start_tick
        high = max(low, windows[previous_last].end_tick)
        boundary = place_boundary(reader, low, high, keys[position - 1], keys[position], chord_onsets, note_onsets)
        starts.append(max(boundary, starts[-1] + 1))
    end = max((n.end for n in music), default=0)
    segments = [
        KeySegment(start=starts[p], end=starts[p + 1] if p + 1 < len(starts) else end,
                   key=keys[p], collection=path[runs[p][0]])
        for p in range(len(runs))
    ]
    # Adjacent runs can share a key label only through rotation; merge them.
    merged: List[KeySegment] = []
    for segment in segments:
        if merged and merged[-1].key == segment.key:
            merged[-1].end = segment.end
        else:
            merged.append(segment)
    return KeyAnalysis(home, home_source, check, filename_key, notes_key, support, merged)


def key_at(segments: Sequence[KeySegment], tick: int) -> Key:
    starts = [segment.start for segment in segments]
    return segments[max(0, bisect.bisect_right(starts, tick) - 1)].key


# --------------------------------------------------------------------------- #
# tension
# --------------------------------------------------------------------------- #

def scale_of(key: Key) -> List[int]:
    return [(key[0] + d) % 12 for d in (MAJOR_SCALE if key[1] == "M" else HARMONIC_MINOR_SCALE)]


def tonic_triad(key: Key) -> frozenset:
    third = 4 if key[1] == "M" else 3
    return frozenset({key[0], (key[0] + third) % 12, (key[0] + 7) % 12})


def basic_space(root: int, pcs: frozenset, key: Key) -> Tuple[set, set, set, set]:
    """Lerdahl's basic space: root / root+fifth / chord / diatonic (+ chromatic chord tones)."""
    fifth = next(((root + iv) % 12 for iv in (7, 6, 8) if (root + iv) % 12 in pcs), None)
    return ({root}, {root} | ({fifth} if fifth is not None else set()), set(pcs), set(scale_of(key)) | set(pcs))


def basic_space_distance(x, y) -> int:
    """k: pitch classes new to y's basic space, counted level by level."""
    return sum(len(level_y - level_x) for level_x, level_y in zip(x, y))


def fifths_steps(a: int, b: int) -> int:
    n = ((b - a) * 7) % 12
    return min(n, 12 - n)


def chord_root_distance(key: Key, root: int) -> int:
    """j: steps on the key's diatonic circle of fifths, chromatic circle for foreign roots."""
    scale = scale_of(key)
    if root in scale:
        position = (2 * scale.index(root)) % 7
        return min(position, 7 - position)
    return fifths_steps(key[0], root)


def region_distance(home: Key, local: Key) -> int:
    """delta(I/home -> I/local); i = collection steps + 1 for a mode change."""
    if home == local:
        return 0
    i = fifths_steps(key_collection(home), key_collection(local)) + (1 if home[1] != local[1] else 0)
    j = fifths_steps(home[0], local[0])
    k = basic_space_distance(basic_space(home[0], tonic_triad(home), home),
                             basic_space(local[0], tonic_triad(local), local))
    return i + j + k


def chord_distance(local: Key, root: int, pcs: frozenset) -> int:
    """delta(I/local -> chord/local)."""
    j = chord_root_distance(local, root)
    k = basic_space_distance(basic_space(local[0], tonic_triad(local), local), basic_space(root, pcs, local))
    return j + k


def roughness(chroma: np.ndarray) -> float:
    total = chroma.sum()
    if total <= 0:
        return 0.0
    x = chroma / total
    value = 0.0
    for a in range(12):
        for b in range(a + 1, 12):
            value += INTERVAL_CLASS_ROUGHNESS[min(b - a, 12 - b + a)] * x[a] * x[b]
    pairs = 1.0 - float((x ** 2).sum())        # normalise away how many pcs sound
    return value / pairs if pairs > 1e-9 else 0.0


def tension_indicators(reader: ChromaReader, spans: Sequence[ChordSpan], keys: KeyAnalysis) -> List[Dict[str, float]]:
    rows = []
    for span in spans:
        start, end = span.start, span.end
        if end - start < MIN_SPAN_MS:
            middle = (start + end) // 2
            start, end = middle - MIN_SPAN_MS // 2, middle + MIN_SPAN_MS // 2
        chroma = reader.chroma(start, end)
        local = key_at(keys.segments, span.start)
        region = region_distance(keys.home, local)
        local_part = chord_distance(local, span.root, span.pcs)
        rows.append({
            "start": span.start,
            "end": span.end,
            "local_key": key_label(local),
            "region": region,
            "chord": local_part,
            "tps": float(region + local_part),
            "tonal": 1.0 - ks_correlation(chroma, keys.home) if chroma.sum() > 0 else float("nan"),
            "rough": roughness(chroma) if chroma.sum() > 0 else float("nan"),
        })
    return rows


def composite_scores(rows: Sequence[Dict[str, float]], calibration: Dict[str, object]) -> np.ndarray:
    """Mean of clipped robust z-scores; an indicator missing for a chord (no notes
    sounding under it) is left out of that chord's mean."""
    center, scale = calibration["center"], calibration["scale"]
    z = np.array([
        [np.clip((row[name] - center[name]) / scale[name], -3, 3) if row[name] == row[name] else np.nan
         for name in INDICATORS]
        for row in rows
    ], dtype=float)
    if len(z) == 0:
        return np.zeros(0)
    counts = (~np.isnan(z)).sum(1)
    return np.where(counts > 0, np.nansum(z, 1) / np.maximum(counts, 1), 0.0)


def smooth(rows: Sequence[Dict[str, float]], scores: np.ndarray, window_ms: int = SMOOTH_MS) -> np.ndarray:
    """Duration-weighted mean of the chord scores overlapping a window centred on
    each chord. Removes beat-level flicker; a cadence held for a beat or more
    still pulls the level down."""
    if window_ms <= 0 or len(scores) == 0:
        return scores.copy()
    starts = np.array([row["start"] for row in rows], dtype=float)
    ends = np.array([row["end"] for row in rows], dtype=float)
    out = np.empty_like(scores)
    half = window_ms / 2.0
    for index in range(len(scores)):
        middle = (starts[index] + ends[index]) / 2.0
        overlap = np.clip(np.minimum(ends, middle + half) - np.maximum(starts, middle - half), 0, None)
        overlap[index] = max(overlap[index], 1.0)
        out[index] = float((overlap * scores).sum() / overlap.sum())
    return out


def levels_of(smoothed: np.ndarray, thresholds: Sequence[float], hysteresis: float = 0.0) -> List[int]:
    """Quantise into levels; with hysteresis a level only changes once the score
    passes a cut point by that margin, so values hovering on a cut do not flicker."""
    cuts = np.asarray(thresholds, dtype=float)
    levels: List[int] = []
    current: Optional[int] = None
    for value in smoothed:
        raw = int(np.searchsorted(cuts, value, side="right"))
        if current is None or hysteresis <= 0:
            current = raw
        elif raw > current:
            current = max(current, int(np.searchsorted(cuts + hysteresis, value, side="right")))
        elif raw < current:
            current = min(current, int(np.searchsorted(cuts - hysteresis, value, side="right")))
        levels.append(current)
    return levels


def tension_levels(rows: Sequence[Dict[str, float]], calibration: Dict[str, object]) -> Tuple[np.ndarray, List[int]]:
    smoothed = smooth(rows, composite_scores(rows, calibration), int(calibration.get("smooth_ms", SMOOTH_MS)))
    return smoothed, levels_of(smoothed, calibration["thresholds"], float(calibration.get("hysteresis", HYSTERESIS)))


# --------------------------------------------------------------------------- #
# one file
# --------------------------------------------------------------------------- #

def source_for(fixed_path: Path, source_dir: Path) -> Optional[Path]:
    stem = fixed_path.stem
    if stem.endswith(INPUT_SUFFIX):
        stem = stem[: -len(INPUT_SUFFIX)]
    for suffix in (".mid", ".midi"):
        candidate = source_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def analyze_file(path: Path, source_dir: Path, restore_velocity: bool, copy_pedal: bool):
    fixed = mido.MidiFile(str(path))
    source_path = source_for(path, source_dir)
    source = mido.MidiFile(str(source_path)) if source_path else None
    retimed = retime(fixed, source, restore_velocity=restore_velocity, copy_pedal=copy_pedal)
    if not retimed.music:
        raise ValueError("no performance notes")
    spans = chord_spans(retimed.chords)
    stem = path.stem[: -len(INPUT_SUFFIX)] if path.stem.endswith(INPUT_SUFFIX) else path.stem
    filename_key = parse_ground_truth(stem)
    keys = analyze_keys(retimed.music, spans, tuple(filename_key) if filename_key else None)
    rows = tension_indicators(ChromaReader(retimed.music), spans, keys)
    return retimed, spans, keys, rows, source_path


def marker_track(keys: KeyAnalysis, rows: Sequence[Dict[str, float]], levels: Sequence[int]) -> mido.MidiTrack:
    events: List[Tuple[int, int, mido.Message]] = [
        (0, 0, mido.MetaMessage("track_name", name=MARKER_TRACK_NAME)),
        (0, 1, mido.MetaMessage("marker", text=f"home: {key_label(keys.home)}")),
        (0, 2, mido.MetaMessage("marker", text=f"key: {key_label(keys.segments[0].key)}")),
    ]
    for previous, segment in zip(keys.segments, keys.segments[1:]):
        events.append((segment.start, 2, mido.MetaMessage("marker", text=f"key: {key_label(segment.key)}")))
        events.append((segment.start, 3, mido.MetaMessage(
            "marker", text=f"change key: {key_label(previous.key)} -> {key_label(segment.key)}")))
    for row, level in zip(rows, levels):
        events.append((int(row["start"]), 4, mido.MetaMessage("marker", text=f"tension: {level}")))
    return build_track(events)


def process_file(job: Tuple[str, str, str, str, Dict[str, object], bool, bool, bool]) -> Dict[str, object]:
    path_text, source_dir, output_dir, analysis_dir, calibration, restore_velocity, copy_pedal, force = job
    path = Path(path_text)
    output_path = Path(output_dir) / path.name
    report: Dict[str, object] = {"file": path.name, "status": "ok"}
    if output_path.exists() and not force:
        report["status"] = "skipped"
        return report
    try:
        retimed, spans, keys, rows, source_path = analyze_file(path, Path(source_dir), restore_velocity, copy_pedal)
        smoothed, levels = tension_levels(rows, calibration)
        retimed.midi.tracks.append(marker_track(keys, rows, levels))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        retimed.midi.save(str(output_path))

        sidecar = {
            "file": path.name,
            "source": str(source_path) if source_path else None,
            "time": {"mapping": retimed.mapping, "speed_match": round(retimed.speed_match, 4),
                     "velocity_restored": round(retimed.velocity_restored, 4), "ms_per_tick": 1.0},
            "keys": {
                "home": key_label(keys.home), "home_source": keys.home_source, "key_check": keys.key_check,
                "filename_key": key_label(keys.filename_key) if keys.filename_key else None,
                "notes_key": key_label(keys.notes_key) if keys.notes_key else None,
                "filename_support": None if keys.filename_support is None else round(keys.filename_support, 3),
                "segments": [{"start": s.start, "end": s.end, "key": key_label(s.key)} for s in keys.segments],
            },
            "chords": [
                {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
                 "score": round(float(score), 4), "level": level}
                for row, score, level in zip(rows, smoothed, levels)
            ],
        }
        Path(analysis_dir).mkdir(parents=True, exist_ok=True)
        (Path(analysis_dir) / f"{path.stem}.json").write_text(json.dumps(sidecar), encoding="utf-8")

        duration_min = (max(n.end for n in retimed.music) - retimed.music[0].start) / 60000.0
        histogram = Counter(levels)
        report.update({
            "mapping": retimed.mapping,
            "speed_match": round(retimed.speed_match, 3),
            "velocity_restored": round(retimed.velocity_restored, 3),
            "pedal_events": retimed.pedal_events,
            "duration_min": round(duration_min, 2),
            "filename_key": key_label(keys.filename_key) if keys.filename_key else "",
            "notes_key": key_label(keys.notes_key) if keys.notes_key else "",
            "key_check": keys.key_check,
            "filename_support": "" if keys.filename_support is None else round(keys.filename_support, 3),
            "home_key": key_label(keys.home),
            "home_source": keys.home_source,
            "key_changes": len(keys.segments) - 1,
            "chords": len(rows),
            **{f"tension_{level}": histogram.get(level, 0) for level in range(TENSION_LEVELS)},
        })
    except Exception as exc:
        report.update({"status": "error", "detail": f"{type(exc).__name__}: {exc}",
                       "trace": traceback.format_exc()[-600:]})
    return report


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #

def calibrate_one(job: Tuple[str, str]) -> Optional[List[Dict[str, float]]]:
    path_text, source_dir = job
    try:
        _, _, _, rows, _ = analyze_file(Path(path_text), Path(source_dir), True, False)
        return rows
    except Exception:
        return None


def calibrate(files: Sequence[Path], source_dir: Path, jobs: int) -> Dict[str, object]:
    """Robust centre/scale per indicator, then equal-population level cut points
    of the smoothed composite, all from the same random sample of pieces."""
    per_file: List[List[Dict[str, float]]] = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(calibrate_one, (str(p), str(source_dir))) for p in files]
        for index, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            if rows:
                per_file.append(rows)
            if index % 200 == 0:
                print(f"  calibrated {index}/{len(files)}", flush=True)
    pooled = {name: np.array([r[name] for rows in per_file for r in rows if r[name] == r[name]]) for name in INDICATORS}
    center = {name: round(float(np.median(v)), 4) for name, v in pooled.items()}
    scale = {name: round(float((np.percentile(v, 75) - np.percentile(v, 25)) / 1.349) or 1.0, 4) for name, v in pooled.items()}
    provisional = {"center": center, "scale": scale, "thresholds": []}
    smoothed = np.concatenate([smooth(rows, composite_scores(rows, provisional), SMOOTH_MS) for rows in per_file])
    quantiles = [100.0 * q / TENSION_LEVELS for q in range(1, TENSION_LEVELS)]
    thresholds = [round(float(np.percentile(smoothed, q)), 4) for q in quantiles]
    return {"center": center, "scale": scale, "thresholds": thresholds,
            "smooth_ms": SMOOTH_MS, "hysteresis": HYSTERESIS,
            "pieces": len(per_file), "chords": int(len(smoothed))}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild all_roman_time_fixed at real speed: keep every note (channel-4 chords "
            "included), drop the old markers, add key_track key / change-key markers and a "
            "tension level per channel-4 chord."
        )
    )
    parser.add_argument("input_dir", nargs="?", default=str(DEFAULT_INPUT_DIR),
                        help="Directory of *_with_chords.mid files. Default: %(default)s")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR),
                        help="Original GiantMIDI files (time map + velocities). Default: %(default)s")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Output MIDI directory. Default: %(default)s")
    parser.add_argument("--analysis-dir", default=None,
                        help="Sidecar JSON + report.csv directory. Default: <output-dir>_analysis")
    parser.add_argument("--calibration", default=None,
                        help="Calibration JSON from --calibrate. Default: the built-in DEFAULT_CALIBRATION.")
    parser.add_argument("--calibrate", type=int, default=0, metavar="N",
                        help="Fit tension statistics on N random pieces, write <analysis-dir>/calibration.json and exit.")
    parser.add_argument("--keep-velocity", action="store_true",
                        help="Keep the input velocities (all 90) instead of restoring them from the source.")
    parser.add_argument("--pedal", action="store_true",
                        help="Also copy the source sustain/sostenuto/soft pedal CCs into a 'Source Pedal' track.")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Parallel worker processes. Default: %(default)s")
    parser.add_argument("--limit", type=int, default=0, help="Process only N files (random sample, seed 0).")
    parser.add_argument("--force", action="store_true", help="Rebuild files whose output already exists.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else output_dir.parent / f"{output_dir.name}_analysis"
    files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in (".mid", ".midi"))
    if not files:
        print(f"Error: no MIDI files in {input_dir}", file=sys.stderr)
        return 1

    if args.calibrate:
        random.seed(0)
        sample = random.sample(files, min(args.calibrate, len(files)))
        print(f"Calibrating tension on {len(sample)} pieces with {args.jobs} workers...")
        result = calibrate(sample, Path(args.source_dir), args.jobs)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        target = analysis_dir / "calibration.json"
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"Wrote {target}")
        return 0

    calibration = DEFAULT_CALIBRATION
    if args.calibration:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    if args.limit:
        random.seed(0)
        files = sorted(random.sample(files, min(args.limit, len(files))))

    print(f"Building {len(files)} file(s) -> {output_dir} with {args.jobs} worker(s)...")
    jobs = [(str(p), args.source_dir, str(output_dir), str(analysis_dir), calibration,
             not args.keep_velocity, args.pedal, args.force) for p in files]
    reports: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(process_file, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            reports.append(future.result())
            if index % 250 == 0 or index == len(files):
                print(f"  {index}/{len(files)}", flush=True)

    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_path = analysis_dir / "report.csv"
    fields: List[str] = []
    for report in reports:
        fields.extend(k for k in report if k not in fields)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in sorted(reports, key=lambda r: r["file"]):
            writer.writerow(report)

    status = Counter(r["status"] for r in reports)
    print("\nStatus: " + ", ".join(f"{k} {v}" for k, v in status.most_common()))
    done = [r for r in reports if r["status"] == "ok"]
    if done:
        print("Time mapping: " + ", ".join(f"{k} {v}" for k, v in Counter(r["mapping"] for r in done).most_common()))
        print(f"Velocity restored: mean {np.mean([r['velocity_restored'] for r in done]):.3f}")
        print("Key check: " + ", ".join(f"{k} {v}" for k, v in Counter(r["key_check"] for r in done).most_common()))
        minutes = sum(r["duration_min"] for r in done)
        print(f"Key changes per minute: {sum(r['key_changes'] for r in done) / max(minutes, 1e-9):.2f}")
        totals = [sum(r[f"tension_{lv}"] for r in done) for lv in range(TENSION_LEVELS)]
        print("Tension levels: " + ", ".join(f"{lv}: {t / max(1, sum(totals)):.1%}" for lv, t in enumerate(totals)))
    errors = [r for r in reports if r["status"] == "error"]
    for report in errors[:5]:
        print(f"[error] {report['file']}: {report['detail']}")
    print(f"Report: {report_path}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
