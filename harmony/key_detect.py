#!/usr/bin/env python3
"""Global key/mode detection from performance MIDI, on the file's own tick grid.

Two-stage detector, built from the diagnostic in `key_benchmark.py`:

  1. Collection - which diatonic 7-pitch-class set. Scored as the mean, over four
     key-weight profiles (Aarden-Essen, Bellman-Budge, Temperley-Kostka-Payne,
     Simple), of the best correlation achievable inside each collection. Averaging
     profiles is worth ~+1 point over any single one.
  2. Rotation - which degree of that collection is the tonic, i.e. the choice
     between the relative major and minor. Decided by a 5-way vote: the
     Krumhansl-Schmuckler, Aarden-Essen and Simple profiles, the final bass pitch
     class, and the chroma of the closing 5% of the piece.

Measured on 1523 name-keyed all_chan_segmented pieces (`python -m harmony.key_benchmark`):

    exact key   78.5%   (java analyzer key column: 60.3%,  musiclang: ~59%)
    collection  81.6%
    rotation    93.4%   given the true collection

The rotation stage is close to saturated; the collection stage carries almost all
the remaining error, and a sizeable part of that is ground-truth noise (15.6% of
the pieces where all five profiles agree, they agree *against* the filename).

Chroma is accumulated from raw MIDI ticks via mido, never through music21, whose
import resamples unquantised performance timelines by up to ~25%.

    from harmony.key_detect import detect_key
    result = detect_key(Path("piece.mid"))
    print(result.label, result.confidence)      # 'Eb major', 0.31
"""
from __future__ import annotations

import bisect
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from cadence import CadenceFeatures

ROOT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = ROOT_DIR / ".vendor"
if VENDOR_DIR.is_dir():
    # Appended, never inserted: .vendor holds a broken numpy source checkout.
    sys.path.append(str(VENDOR_DIR))

import mido

PC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Conventional tonic spellings, which differ by mode: Db major but C# minor,
# Eb minor but D#/Eb ambiguity resolved flat-wards, and so on.
MAJOR_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]

# music21.analysis.discrete key-weight profiles (music21 8.1), inlined so the
# correlation runs on our own tick-accurate chroma without a music21 import.
PROFILES: Dict[str, Dict[str, List[float]]] = {
    "ks": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    },
    "aarden": {
        "major": [17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587,
                  0.291248, 22.062, 0.145624, 8.15494, 0.232998, 4.95122],
        "minor": [18.2648, 0.737619, 14.0499, 16.8599, 0.702494, 14.4362,
                  0.702494, 18.6161, 4.56621, 1.93186, 7.37619, 1.75623],
    },
    "bellman": {
        "major": [16.80, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 20.28, 1.80, 8.04, 0.62, 10.57],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 21.07, 7.49, 1.53, 0.92, 10.21],
    },
    "tkp": {
        "major": [0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400],
        "minor": [0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330],
    },
    "simple": {
        "major": [2, 0, 1, 0, 1, 1, 0, 2, 0, 1, 0, 1],
        "minor": [2, 0, 1, 1, 0, 1, 0, 2, 1, 0, 0.5, 0.5],
    },
}

COLLECTION_PROFILES = ("aarden", "bellman", "tkp", "simple")
ROTATION_PROFILES = ("ks", "aarden", "simple")
TAIL_FRACTION = 0.05

# The seven rotations of a diatonic collection, as (semitones above the
# collection's major tonic, mode name). Choosing among all seven - rather than
# only ionian and aeolian - is what lets a segment be labelled 'D dorian'
# instead of being forced into the nearest major or minor.
MODE_DEGREES: List[Tuple[int, str]] = [
    (0, "ionian"),
    (2, "dorian"),
    (4, "phrygian"),
    (5, "lydian"),
    (7, "mixolydian"),
    (9, "aeolian"),
    (11, "locrian"),
]

MODE_TO_MAJMIN = {
    "ionian": "M", "lydian": "M", "mixolydian": "M",
    "dorian": "m", "phrygian": "m", "aeolian": "m", "locrian": "m",
}

# Weights for the mode-agnostic tonic-strength score, fitted by grid search on
# 400 filename-labelled all_chan_segmented pieces disjoint from the gold set
# (tune_modal.py in the session scratchpad). The labels there are only ever
# major or minor, so the fit encodes a real-world prior favouring ionian and
# aeolian; the scorer's *form* stays mode-agnostic, so strong evidence can still
# push a segment onto another degree.
# Note final_bass fitted to 0.0: a flat bonus for "the last bass note is this
# pitch class" is too blunt and outvotes everything else. The closing-chroma
# term carries the same terminal evidence more smoothly.
MODAL_WEIGHTS: Dict[str, float] = {
    "tonic": 1.0,
    "fifth": 1.0,
    "third": 0.3,
    "final_bass": 0.0,
    "tail": 0.4,
    "ionian_prior": 0.15,
    "aeolian_prior": 0.12,
    # Cadential motion terms (cadence.py). Fitted separately - see tune_cadence.py.
    "leading_tone": 0.0,
    "bass_fifth": 0.0,
    "phrase_end": 0.0,
}


@dataclass
class MidiFeatures:
    """Pitch-class evidence extracted in a single pass over the tick timeline."""
    chroma: List[float]         # duration-weighted, normalized
    chroma_onset: List[float]   # onset-count weighted, normalized
    bass_chroma: List[float]    # duration of the lowest sounding voice, normalized
    tail_chroma: List[float]    # duration-weighted over the closing TAIL_FRACTION
    final_bass_pc: Optional[int]
    final_top_pc: Optional[int]
    first_bass_pc: Optional[int]
    lowest_pc: Optional[int]
    note_count: int


@dataclass
class KeyResult:
    tonic_pc: int
    mode: str                   # 'M' or 'm'
    collection: int             # relative major pc of the diatonic set
    confidence: float           # margin between the best and runner-up collection
    rotation_votes: int         # out of 5, for the winning rotation

    @property
    def tonic_name(self) -> str:
        return (MAJOR_NAMES if self.mode == "M" else MINOR_NAMES)[self.tonic_pc]

    @property
    def label(self) -> str:
        return f"{self.tonic_name} {'major' if self.mode == 'M' else 'minor'}"

    @property
    def romantext(self) -> str:
        """'Eb' for major, 'c' for minor - musiclang/romantext spelling."""
        name = self.tonic_name
        return name if self.mode == "M" else name[0].lower() + name[1:]


def normalize(vector: Sequence[float]) -> List[float]:
    total = float(sum(vector))
    if total <= 0:
        return [0.0] * 12
    return [value / total for value in vector]


def extract_midi_features(path: Path) -> Optional[MidiFeatures]:
    try:
        midi = mido.MidiFile(str(path))
    except Exception:
        return None

    # (start_tick, end_tick, pitch) for every note, from the merged tick timeline.
    notes: List[Tuple[int, int, int]] = []
    pending: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    tick = 0
    for msg in mido.merge_tracks(midi.tracks):
        tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            pending[(msg.channel, msg.note)].append(tick)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            starts = pending.get((msg.channel, msg.note))
            if starts:
                start = starts.pop(0)
                if tick > start:
                    notes.append((start, tick, msg.note))
    for (_, pitch), starts in pending.items():          # unterminated notes
        for start in starts:
            if tick > start:
                notes.append((start, tick, pitch))
    if not notes:
        return None

    notes.sort()
    end_tick = max(note[1] for note in notes)
    tail_start = end_tick - max(1, int(TAIL_FRACTION * end_tick))

    chroma = [0.0] * 12
    chroma_onset = [0.0] * 12
    tail_chroma = [0.0] * 12
    for start, stop, pitch in notes:
        pc = pitch % 12
        chroma[pc] += stop - start
        chroma_onset[pc] += 1.0
        overlap = min(stop, end_tick) - max(start, tail_start)
        if overlap > 0:
            tail_chroma[pc] += overlap

    # Sweep line over the sounding set: between consecutive events the lowest
    # sounding pitch is the bass, and the set at the last onset is the final chord.
    events: List[Tuple[int, int, int]] = []             # (tick, +1/-1, pitch)
    for start, stop, pitch in notes:
        events.append((start, 1, pitch))
        events.append((stop, -1, pitch))
    events.sort(key=lambda item: (item[0], item[1]))

    bass_chroma = [0.0] * 12
    sounding: List[int] = []
    previous_tick = events[0][0]
    last_sounding: List[int] = []
    last_onset_tick = -1
    for event_tick, delta, pitch in events:
        if event_tick > previous_tick:
            if sounding:
                bass_chroma[sounding[0] % 12] += event_tick - previous_tick
            previous_tick = event_tick
        if delta > 0:
            bisect.insort(sounding, pitch)
            if event_tick > last_onset_tick:
                last_onset_tick = event_tick
                last_sounding = list(sounding)
        else:
            index = bisect.bisect_left(sounding, pitch)
            if index < len(sounding) and sounding[index] == pitch:
                sounding.pop(index)

    first_tick = notes[0][0]
    return MidiFeatures(
        chroma=normalize(chroma),
        chroma_onset=normalize(chroma_onset),
        bass_chroma=normalize(bass_chroma),
        tail_chroma=normalize(tail_chroma),
        final_bass_pc=min(last_sounding) % 12 if last_sounding else None,
        final_top_pc=max(last_sounding) % 12 if last_sounding else None,
        first_bass_pc=min(pitch for start, _, pitch in notes if start == first_tick) % 12,
        lowest_pc=min(pitch for _, _, pitch in notes) % 12,
        note_count=len(notes),
    )


def correlate(vector: Sequence[float], profile: Sequence[float]) -> float:
    mean_v = sum(vector) / 12.0
    mean_p = sum(profile) / 12.0
    dv = [value - mean_v for value in vector]
    dp = [value - mean_p for value in profile]
    denominator = math.sqrt(sum(a * a for a in dv) * sum(b * b for b in dp))
    if not denominator:
        return 0.0
    return sum(a * b for a, b in zip(dv, dp)) / denominator


def rotated_profile(profile_name: str, mode: str, tonic_pc: int) -> List[float]:
    base = PROFILES[profile_name]["major" if mode == "M" else "minor"]
    return [base[(index - tonic_pc) % 12] for index in range(12)]


def collection_candidates(collection: int) -> List[Tuple[int, str]]:
    """The relative pair sharing a diatonic collection: (C, 'M') and (A, 'm')."""
    return [(collection, "M"), ((collection + 9) % 12, "m")]


def score_collections(chroma: Sequence[float]) -> List[float]:
    """Per collection, the profile-averaged best correlation inside it."""
    scores: List[float] = []
    for collection in range(12):
        per_profile = [
            max(
                correlate(chroma, rotated_profile(name, mode, pc))
                for pc, mode in collection_candidates(collection)
            )
            for name in COLLECTION_PROFILES
        ]
        scores.append(sum(per_profile) / len(per_profile))
    return scores


def choose_rotation(features: MidiFeatures, collection: int) -> Tuple[Tuple[int, str], int]:
    """Vote between the relative major and minor of a fixed collection."""
    candidates = collection_candidates(collection)
    votes: Counter = Counter()

    for name in ROTATION_PROFILES:
        winner = max(
            candidates,
            key=lambda pm: correlate(features.chroma, rotated_profile(name, pm[1], pm[0])),
        )
        votes[winner] += 1

    # Terminal evidence: the closing bass note is the tonic far more often than not.
    if features.final_bass_pc == candidates[1][0]:
        votes[candidates[1]] += 1
    elif features.final_bass_pc == candidates[0][0]:
        votes[candidates[0]] += 1

    # Closing chroma: the last 5% of the piece is usually unambiguously tonic.
    tail = features.tail_chroma
    votes[candidates[0] if tail[candidates[0][0]] >= tail[candidates[1][0]] else candidates[1]] += 1

    winner, count = votes.most_common(1)[0]
    return winner, count


def modal_candidates(collection: int) -> List[Tuple[int, str]]:
    """All seven rotations of a collection: [(tonic pc, mode name), ...]."""
    return [((collection + offset) % 12, name) for offset, name in MODE_DEGREES]


def collection_pitch_classes(collection: int) -> List[int]:
    return [(collection + offset) % 12 for offset in (0, 2, 4, 5, 7, 9, 11)]


def score_modal_tonic(
    chroma: Sequence[float],
    collection: int,
    final_bass_pc: Optional[int] = None,
    tail_chroma: Optional[Sequence[float]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[float, int, str]]:
    """Score all seven degrees of a collection as candidate tonics.

    Deliberately template-free. Key-weight profiles only exist for major and
    minor, so scoring seven modes with them would mean inventing five profiles
    with no data behind them. Instead this scores the evidence that makes any
    pitch class function as a tonic, whatever the mode:

      * its own weight in the chroma,
      * the weight of its perfect fifth (a tonic needs one - this is what makes
        locrian, whose fifth is diminished, correctly almost unreachable),
      * the weight of its third,
      * whether the passage ends on it, in the bass and in the closing chroma.

    Returns the candidates sorted best-first.
    """
    weights = weights or MODAL_WEIGHTS
    total = float(sum(chroma)) or 1.0
    normalized = [value / total for value in chroma]
    members = set(collection_pitch_classes(collection))

    tail_normalized: Optional[List[float]] = None
    if tail_chroma is not None and sum(tail_chroma) > 0:
        tail_total = float(sum(tail_chroma))
        tail_normalized = [value / tail_total for value in tail_chroma]

    scored: List[Tuple[float, int, str]] = []
    for tonic_pc, mode_name in modal_candidates(collection):
        fifth_pc = (tonic_pc + 7) % 12
        third_pc = (tonic_pc + 4) % 12
        if third_pc not in members:
            third_pc = (tonic_pc + 3) % 12

        score = weights["tonic"] * normalized[tonic_pc]
        if fifth_pc in members:
            score += weights["fifth"] * normalized[fifth_pc]
        if third_pc in members:
            score += weights["third"] * normalized[third_pc]
        if final_bass_pc is not None and final_bass_pc == tonic_pc:
            score += weights["final_bass"]
        if tail_normalized is not None:
            score += weights["tail"] * tail_normalized[tonic_pc]
        if mode_name == "ionian":
            score += weights["ionian_prior"]
        elif mode_name == "aeolian":
            score += weights["aeolian_prior"]
        scored.append((score, tonic_pc, mode_name))

    scored.sort(reverse=True)
    return scored


def score_modal_tonic_cadential(
    chroma: Sequence[float],
    collection: int,
    cadence: "CadenceFeatures",
    tail_chroma: Optional[Sequence[float]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[float, int, str]]:
    """score_modal_tonic plus cadential motion evidence.

    The chroma terms cannot separate a tonic from its dominant - both readings
    contain the same pitch classes. These three terms can, because they measure
    motion rather than content: what resolves onto a pitch class, what falls a
    fifth onto it, and where phrases come to rest.
    """
    weights = weights or MODAL_WEIGHTS
    base = score_modal_tonic(chroma, collection, None, tail_chroma, weights)
    rescored: List[Tuple[float, int, str]] = []
    for score, tonic_pc, mode_name in base:
        score += weights.get("leading_tone", 0.0) * cadence.leading_tone[tonic_pc]
        score += weights.get("bass_fifth", 0.0) * cadence.bass_fifth[tonic_pc]
        score += weights.get("phrase_end", 0.0) * cadence.phrase_end[tonic_pc]
        rescored.append((score, tonic_pc, mode_name))
    rescored.sort(reverse=True)
    return rescored


def choose_modal_rotation(
    chroma: Sequence[float],
    collection: int,
    final_bass_pc: Optional[int] = None,
    tail_chroma: Optional[Sequence[float]] = None,
    cadence: Optional["CadenceFeatures"] = None,
) -> Tuple[int, str, float]:
    """Best of the seven degrees: (tonic pc, mode name, margin over runner-up)."""
    if cadence is not None:
        scored = score_modal_tonic_cadential(chroma, collection, cadence, tail_chroma)
    else:
        scored = score_modal_tonic(chroma, collection, final_bass_pc, tail_chroma)
    margin = scored[0][0] - scored[1][0] if len(scored) > 1 else 0.0
    return scored[0][1], scored[0][2], margin


def mode_label(tonic_pc: int, mode_name: str) -> str:
    """'D dorian', 'C major' - ionian and aeolian keep their common names."""
    names = MAJOR_NAMES if MODE_TO_MAJMIN[mode_name] == "M" else MINOR_NAMES
    if mode_name == "ionian":
        return f"{names[tonic_pc]} major"
    if mode_name == "aeolian":
        return f"{names[tonic_pc]} minor"
    return f"{names[tonic_pc]} {mode_name}"


def detect_key_from_features(features: MidiFeatures) -> Optional[KeyResult]:
    if features.note_count == 0 or sum(features.chroma) <= 0:
        return None
    scores = score_collections(features.chroma)
    ranked = sorted(range(12), key=lambda c: scores[c], reverse=True)
    collection = ranked[0]
    margin = scores[ranked[0]] - scores[ranked[1]]
    (tonic_pc, mode), votes = choose_rotation(features, collection)
    return KeyResult(
        tonic_pc=tonic_pc,
        mode=mode,
        collection=collection,
        confidence=margin,
        rotation_votes=votes,
    )


def detect_key(midi_path: Path) -> Optional[KeyResult]:
    features = extract_midi_features(Path(midi_path))
    if features is None:
        return None
    return detect_key_from_features(features)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Detect the global key of MIDI files.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)

    targets: List[Path] = []
    for path in args.paths:
        targets.extend(sorted(path.glob("*.mid")) if path.is_dir() else [path])

    for path in targets:
        result = detect_key(path)
        if result is None:
            print(f"{path.name}\t(no notes)")
            continue
        print(
            f"{path.name}\t{result.label}\t"
            f"conf={result.confidence:.3f}\tvotes={result.rotation_votes}/5"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
