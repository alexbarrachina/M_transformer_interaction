#!/usr/bin/env python3
"""Windowed diatonic-collection tracking for performance MIDI.

Motivation (measured in key_benchmark.py / the midi_keys.csv gold set): global
key errors are dominated by picking the wrong diatonic *collection* — one
whole-piece chroma histogram smears modulations together — while choosing the
tonic inside a known collection is ~96% solved. This module therefore estimates
the collection *locally* and makes the sequence of collections the primary
output; the global key is just a readout of that path.

Pipeline:
  1. Chroma frames: duration-weighted pitch-class mass per second of wall-clock
     time (mido's tempo-mapped timeline; music21 is never involved).
  2. Windows: overlapping sums of frames (window/hop in seconds).
  3. Emissions: per window, each of the 12 collections is scored by the
     profile-averaged best correlation achievable inside it (same scorer as
     key_detect.score_collections).
  4. Viterbi over the 12 collection states. Switching cost grows with distance
     on the circle of fifths, so the path is a modulation map, not flicker.
     Mode is intentionally NOT a state: relative majors/minors and modal
     recolorings share a collection, so they cannot create false modulations.
  5. Segments: maximal runs of one collection, each carrying its summed chroma
     and a local relative-pair rotation guess (enough to label the segment's
     key; finer modal readout can be layered on later).
  6. Global key: duration vote over the path with the opening/closing windows
     up-weighted (a piece that modulates still starts and ends at home), then
     the rotation decided by key_detect.choose_rotation on whole-piece features.

The dataclasses are JSON-friendly so downstream annotation can consume the
segment map directly.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = ROOT_DIR / ".vendor"
if VENDOR_DIR.is_dir():
    # Appended, never inserted: .vendor holds a broken numpy source checkout.
    sys.path.append(str(VENDOR_DIR))

import mido

from .key_detect import (
    MAJOR_NAMES,
    MINOR_NAMES,
    KeyResult,
    choose_modal_rotation,
    choose_rotation,
    collection_candidates,
    correlate,
    extract_midi_features,
    mode_label,
    rotated_profile,
    score_collections,
)

# Framing is measured in NOTE ONSETS, not seconds. The two corpora encode time
# incompatibly - all_chan_segmented runs 1 tick = 1 ms at 1000 ppq / 60 bpm while
# sel_chan_segmented uses 480 ppq / 120 bpm, so the same piece is 4x longer in
# wall-clock in one than the other. Seconds-based windows therefore cover
# different amounts of music per corpus and no tuned parameter transfers.
# Onset counts are also the right statistical unit: the reliability of a chroma
# estimate depends on how many notes fed it, not on elapsed time.
NOTES_PER_FRAME = 8

# Tuned on 400 filename-labeled all_chan_segmented pieces disjoint from the
# midi_keys.csv gold set (tune_track.py in the session scratchpad).
DEFAULT_WINDOW_FRAMES = 8         # 64 note onsets
DEFAULT_HOP_FRAMES = 4            # 50% overlap
DEFAULT_SWITCH_PENALTY = 0.6      # base cost of leaving the current collection
DEFAULT_DISTANCE_PENALTY = 0.25   # extra cost per circle-of-fifths step beyond 1
DEFAULT_EMISSION_SCALE = 8.0      # correlation -> pseudo-log-likelihood scale
EDGE_FRACTION = 0.10              # head/tail share of the piece that votes extra
EDGE_WEIGHT = 3.0

FIFTHS_POSITION = [(pc * 7) % 12 for pc in range(12)]  # pc -> steps along the circle


def fifths_distance(collection_a: int, collection_b: int) -> int:
    """Steps between two collections on the circle of fifths (0..6)."""
    delta = (FIFTHS_POSITION[collection_b] - FIFTHS_POSITION[collection_a]) % 12
    return min(delta, 12 - delta)


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #

@dataclass
class Frame:
    """One group of NOTES_PER_FRAME consecutive onsets."""
    start_tick: int
    end_tick: int
    chroma: List[float]               # duration-weighted, in ticks


class TickClock:
    """Tick -> seconds using the file's own set_tempo events."""

    def __init__(self, midi: mido.MidiFile):
        self.ppq = midi.ticks_per_beat or 480
        anchors: List[Tuple[int, float, int]] = [(0, 0.0, 500000)]
        tick = 0
        for msg in mido.merge_tracks(midi.tracks):
            tick += msg.time
            if msg.type == "set_tempo":
                last_tick, last_seconds, last_tempo = anchors[-1]
                seconds = last_seconds + (tick - last_tick) * last_tempo / (1e6 * self.ppq)
                if tick == last_tick:
                    anchors[-1] = (tick, seconds, msg.tempo)
                else:
                    anchors.append((tick, seconds, msg.tempo))
        self._anchors = anchors

    def seconds(self, tick: int) -> float:
        index = 0
        for position, anchor in enumerate(self._anchors):
            if anchor[0] <= tick:
                index = position
            else:
                break
        anchor_tick, anchor_seconds, tempo = self._anchors[index]
        return anchor_seconds + (tick - anchor_tick) * tempo / (1e6 * self.ppq)


def extract_chroma_frames(path: Path) -> Optional[Tuple[List[Frame], TickClock]]:
    """Group notes into fixed-onset-count frames on the file's raw tick grid.

    Durations are accumulated in ticks, so nothing depends on the file's tempo
    map being musically sane - only on it being internally consistent.
    """
    try:
        midi = mido.MidiFile(str(path))
    except Exception:
        return None

    notes: List[Tuple[int, int, int]] = []           # (start_tick, end_tick, pitch)
    pending: Dict[Tuple[int, int], List[int]] = {}
    tick = 0
    for msg in mido.merge_tracks(midi.tracks):
        tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            pending.setdefault((msg.channel, msg.note), []).append(tick)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            starts = pending.get((msg.channel, msg.note))
            if starts:
                start = starts.pop(0)
                if tick > start:
                    notes.append((start, tick, msg.note))
    for (_, pitch), starts in pending.items():        # unterminated notes
        for start in starts:
            if tick > start:
                notes.append((start, tick, pitch))
    if not notes:
        return None

    notes.sort()
    frames: List[Frame] = []
    for index in range(0, len(notes), NOTES_PER_FRAME):
        group = notes[index : index + NOTES_PER_FRAME]
        chroma = [0.0] * 12
        for start, stop, pitch in group:
            chroma[pitch % 12] += stop - start
        frames.append(
            Frame(
                start_tick=group[0][0],
                end_tick=max(note[1] for note in group),
                chroma=chroma,
            )
        )
    return frames, TickClock(midi)


# --------------------------------------------------------------------------- #
# windows and emissions
# --------------------------------------------------------------------------- #

@dataclass
class Window:
    start_tick: int
    end_tick: int
    onset_count: int                     # notes feeding this window - its vote weight
    chroma: List[float]                  # unnormalized mass, in ticks
    scores: List[float] = field(default_factory=list)   # 12 collection scores


def build_windows(
    frames: Sequence[Frame],
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    hop_frames: int = DEFAULT_HOP_FRAMES,
) -> List[Window]:
    windows: List[Window] = []
    start = 0
    while start < len(frames):
        stop = min(len(frames), start + window_frames)
        chroma = [0.0] * 12
        for frame in frames[start:stop]:
            for pc in range(12):
                chroma[pc] += frame.chroma[pc]
        if sum(chroma) > 0:
            windows.append(
                Window(
                    start_tick=frames[start].start_tick,
                    end_tick=frames[stop - 1].end_tick,
                    onset_count=(stop - start) * NOTES_PER_FRAME,
                    chroma=chroma,
                )
            )
        if stop == len(frames):
            break
        start += hop_frames
    return windows


def score_windows(windows: Sequence[Window]) -> None:
    for window in windows:
        window.scores = score_collections(window.chroma)


# --------------------------------------------------------------------------- #
# Viterbi over collections
# --------------------------------------------------------------------------- #

def viterbi_path(
    windows: Sequence[Window],
    switch_penalty: float = DEFAULT_SWITCH_PENALTY,
    distance_penalty: float = DEFAULT_DISTANCE_PENALTY,
    emission_scale: float = DEFAULT_EMISSION_SCALE,
) -> List[int]:
    if not windows:
        return []
    transition = [
        [
            0.0 if a == b
            else -(switch_penalty + distance_penalty * (fifths_distance(a, b) - 1))
            for b in range(12)
        ]
        for a in range(12)
    ]
    scores = [emission_scale * value for value in windows[0].scores]
    back: List[List[int]] = []
    for window in windows[1:]:
        emissions = [emission_scale * value for value in window.scores]
        pointers = [0] * 12
        next_scores = [0.0] * 12
        for state in range(12):
            best_prev, best_value = 0, -math.inf
            for prev in range(12):
                value = scores[prev] + transition[prev][state]
                if value > best_value:
                    best_prev, best_value = prev, value
            pointers[state] = best_prev
            next_scores[state] = best_value + emissions[state]
        back.append(pointers)
        scores = next_scores
    state = max(range(12), key=lambda s: scores[s])
    path = [state]
    for pointers in reversed(back):
        state = pointers[state]
        path.append(state)
    path.reverse()
    return path


# --------------------------------------------------------------------------- #
# segments and global readout
# --------------------------------------------------------------------------- #

@dataclass
class Segment:
    start_tick: int
    end_tick: int
    start_seconds: float
    end_seconds: float
    onset_count: int
    collection: int                     # relative-major pc of the diatonic set
    tonic_pc: int                       # relative-pair (major/minor) rotation
    mode: str                           # 'M' or 'm'
    chroma: List[float]
    modal_tonic_pc: int = 0             # best of all seven degrees
    modal_mode: str = "ionian"          # ionian/dorian/.../locrian
    modal_margin: float = 0.0           # lead over the runner-up degree

    @property
    def label(self) -> str:
        names = MAJOR_NAMES if self.mode == "M" else MINOR_NAMES
        return f"{names[self.tonic_pc]} {'major' if self.mode == 'M' else 'minor'}"

    @property
    def modal_label(self) -> str:
        """'D dorian' - the seven-degree reading, which can name any mode."""
        return mode_label(self.modal_tonic_pc, self.modal_mode)

    @property
    def is_modal_candidate(self) -> bool:
        """True when the seven-degree reading escapes the major/minor reading.

        The default weights carry a prior favouring ionian and aeolian, fitted
        on labels that were never anything else, so a segment that lands on
        another degree *anyway* did so against that pull. Those are the segments
        worth a human ear - and the ones that would grow the modal ground truth
        the prior currently cannot be fitted against.
        """
        return self.modal_mode not in ("ionian", "aeolian")


@dataclass
class KeyTrack:
    windows: List[Window]
    path: List[int]                     # collection per window
    segments: List[Segment]
    global_key: Optional[KeyResult]
    path_stability: float               # fraction of windows on the modal collection

    def to_json(self) -> Dict[str, object]:
        return {
            "segments": [
                {
                    "start_tick": segment.start_tick,
                    "end_tick": segment.end_tick,
                    "start_seconds": round(segment.start_seconds, 2),
                    "end_seconds": round(segment.end_seconds, 2),
                    "onset_count": segment.onset_count,
                    "collection": segment.collection,
                    "label": segment.label,
                    "modal_label": segment.modal_label,
                    "modal_margin": round(segment.modal_margin, 4),
                }
                for segment in self.segments
            ],
            "global_key": self.global_key.label if self.global_key else None,
            "global_confidence": self.global_key.confidence if self.global_key else None,
            "path_stability": self.path_stability,
        }


def segment_rotation(chroma: Sequence[float], collection: int) -> Tuple[int, str]:
    """Relative-pair choice for one segment from its own chroma (profile vote)."""
    candidates = collection_candidates(collection)
    votes = {candidate: 0 for candidate in candidates}
    for name in ("ks", "aarden", "simple"):
        winner = max(
            candidates,
            key=lambda pm: correlate(chroma, rotated_profile(name, pm[1], pm[0])),
        )
        votes[winner] += 1
    return max(candidates, key=lambda candidate: votes[candidate])


def build_segments(
    windows: Sequence[Window],
    path: Sequence[int],
    clock: Optional[TickClock] = None,
) -> List[Segment]:
    segments: List[Segment] = []
    index = 0
    while index < len(path):
        stop = index
        while stop + 1 < len(path) and path[stop + 1] == path[index]:
            stop += 1
        chroma = [0.0] * 12
        onsets = 0
        for window in windows[index : stop + 1]:
            onsets += window.onset_count
            for pc in range(12):
                chroma[pc] += window.chroma[pc]

        # The closing chroma must come from the END of the segment. Passing the
        # segment's own full chroma here would make the terminal term a second
        # copy of the tonic term, destroying the one signal that separates a
        # tonic from a dominant - and a piece leaning on its dominant then reads
        # as mixolydian (major) or phrygian (minor) on that degree.
        tail_start = max(index, stop + 1 - max(1, (stop + 1 - index) // 4))
        tail_chroma = [0.0] * 12
        for window in windows[tail_start : stop + 1]:
            for pc in range(12):
                tail_chroma[pc] += window.chroma[pc]

        tonic_pc, mode = segment_rotation(chroma, path[index])
        modal_tonic_pc, modal_mode, modal_margin = choose_modal_rotation(
            chroma, path[index], tail_chroma=tail_chroma
        )
        start_tick = windows[index].start_tick
        end_tick = windows[stop].end_tick
        segments.append(
            Segment(
                start_tick=start_tick,
                end_tick=end_tick,
                start_seconds=clock.seconds(start_tick) if clock else 0.0,
                end_seconds=clock.seconds(end_tick) if clock else 0.0,
                onset_count=onsets,
                collection=path[index],
                tonic_pc=tonic_pc,
                mode=mode,
                chroma=chroma,
                modal_tonic_pc=modal_tonic_pc,
                modal_mode=modal_mode,
                modal_margin=modal_margin,
            )
        )
        index = stop + 1
    return segments


def global_collection_vote(windows: Sequence[Window], path: Sequence[int]) -> Tuple[int, float]:
    """Onset-weighted vote over the path, with the head and tail up-weighted.

    Weighting by onsets rather than elapsed ticks keeps the vote independent of
    the file's tempo encoding, and stops a slow sustained passage from
    outvoting a dense one. A piece that modulates still starts and ends home,
    hence EDGE_WEIGHT on the outer EDGE_FRACTION of the piece.
    """
    if not path:
        return 0, 0.0
    positions = [0]
    for window in windows:
        positions.append(positions[-1] + window.onset_count)
    total_onsets = positions[-1] or 1
    votes = [0.0] * 12
    for index, (window, collection) in enumerate(zip(windows, path)):
        weight = float(window.onset_count)
        center = (positions[index] + positions[index + 1]) / 2.0 / total_onsets
        if center < EDGE_FRACTION or center > 1.0 - EDGE_FRACTION:
            weight *= EDGE_WEIGHT
        votes[collection] += weight
    ranked = sorted(range(12), key=lambda c: votes[c], reverse=True)
    margin = (votes[ranked[0]] - votes[ranked[1]]) / (sum(votes) or 1.0)
    return ranked[0], margin


def track_key(
    midi_path: Path,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    hop_frames: int = DEFAULT_HOP_FRAMES,
    switch_penalty: float = DEFAULT_SWITCH_PENALTY,
    distance_penalty: float = DEFAULT_DISTANCE_PENALTY,
) -> Optional[KeyTrack]:
    extracted = extract_chroma_frames(midi_path)
    if extracted is None:
        return None
    frames, clock = extracted
    windows = build_windows(frames, window_frames, hop_frames)
    if not windows:
        return None
    score_windows(windows)
    path = viterbi_path(windows, switch_penalty, distance_penalty)
    segments = build_segments(windows, path, clock)
    collection, margin = global_collection_vote(windows, path)
    stability = sum(1 for state in path if state == collection) / len(path)

    global_key: Optional[KeyResult] = None
    features = extract_midi_features(midi_path)
    if features is not None:
        (tonic_pc, mode), votes = choose_rotation(features, collection)
        global_key = KeyResult(
            tonic_pc=tonic_pc,
            mode=mode,
            collection=collection,
            confidence=margin,
            rotation_votes=votes,
        )
    return KeyTrack(
        windows=windows,
        path=path,
        segments=segments,
        global_key=global_key,
        path_stability=stability,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Track local diatonic collections in MIDI files.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--segments", action="store_true", help="print the modulation map per file")
    parser.add_argument("--modal", action="store_true", help="label segments with all seven modes")
    parser.add_argument(
        "--switch",
        type=float,
        default=DEFAULT_SWITCH_PENALTY,
        help="collection switch penalty; raise it for coarser, more readable maps",
    )
    args = parser.parse_args(argv)

    targets: List[Path] = []
    for path in args.paths:
        if path.is_dir():
            targets.extend(sorted(list(path.glob("*.mid")) + list(path.glob("*.midi"))))
        else:
            targets.append(path)

    for path in targets:
        result = track_key(path, switch_penalty=args.switch)
        if result is None or result.global_key is None:
            print(f"{path.name}\t(no notes)")
            continue
        key = result.global_key
        print(
            f"{path.name}\t{key.label}\tconf={key.confidence:.3f}\t"
            f"stability={result.path_stability:.2f}\tsegments={len(result.segments)}"
        )
        if args.segments:
            for segment in result.segments:
                label = segment.modal_label if args.modal else segment.label
                print(
                    f"    {segment.start_seconds:7.1f}-{segment.end_seconds:7.1f}s  "
                    f"{label}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
