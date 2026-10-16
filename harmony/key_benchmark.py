#!/usr/bin/env python3
"""Instrumented key/mode benchmark: split *collection* accuracy from *tonic* accuracy.

The pipeline's key sources (PM2S, musiclang, music21's Krumhansl-Schmuckler) all
score a duration-weighted pitch-class histogram, which identifies the diatonic
collection but is nearly blind to which degree of it is the tonic: relatives and
modes share the same histogram. This script measures the two halves separately so
the remaining error budget is attributable:

    exact          = tonic pc and mode both right
    collection     = right 7-pitch-class set (relative major pc)
    tonic|coll     = exact, restricted to pieces whose collection was right

plus an error taxonomy (relative / parallel / dominant / subdominant / other).

Ground truth comes from "... in <note> <major|minor> ..." filenames in the corpus.
It is noisy (segmented multi-piece videos), so absolute numbers are pessimistic;
the paired comparison between predictors is fair because they face the same labels.

Predictors compared:
  java     - the java_harmony_analyzer key column, row-majority, with the
             leading-default-block skip rule from add_chords_from_java.py
  ks/aarden/bellman/tkp/simple
           - the five music21 key-weight profiles, correlated against the
             duration-weighted chroma computed here from raw ticks (not via
             music21, which resamples performance-MIDI timelines)

Feature extraction is cached to JSONL and deliberately extracts more than this
report needs (bass chroma, terminal sonorities, chord-root stream) so the later
tonic-scorer work can reuse the same expensive MIDI pass.

Usage:
    python -m harmony.key_benchmark                       # extract (cached) + report
    python -m harmony.key_benchmark --limit 100           # quick smoke run
    python -m harmony.key_benchmark --refresh             # ignore the cache
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .key_detect import (
    PROFILES,
    MidiFeatures,
    correlate,
    detect_key_from_features,
    extract_midi_features,
    rotated_profile,
)

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT_DIR / "all_chan_segmented"
DEFAULT_CACHE = ROOT_DIR / "analysis" / "key_features.jsonl"

PC_OF_LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# "Prelude in D-flat major", "Waltz in E minor", "Etude in F sharp minor"
GT_RE = re.compile(
    r"\bin\s+([A-G])\b(?:[-\s](flat|sharp))?\s+(major|minor)\b",
    re.IGNORECASE,
)

# Namer.java seeds its scale analysis with C DUR / A MOLL; a leading block in one
# of those is a warm-up artifact far more often than a detection.
JAVA_DEFAULT_KEYS = {(0, "M"), (9, "m")}


# --------------------------------------------------------------------------- #
# ground truth
# --------------------------------------------------------------------------- #

def parse_ground_truth(stem: str) -> Optional[Tuple[int, str]]:
    """'Etude in G major, atoyZOOjndw' -> (7, 'M'); None when unlabelled."""
    matches = GT_RE.findall(stem)
    if len(matches) != 1:
        # No label, or a multi-key title ("Preludes in C major and A minor").
        return None
    letter, accidental, mode_text = matches[0]
    pc = PC_OF_LETTER[letter.upper()]
    accidental = (accidental or "").lower()
    if accidental == "flat":
        pc = (pc - 1) % 12
    elif accidental == "sharp":
        pc = (pc + 1) % 12
    return pc, "M" if mode_text.lower() == "major" else "m"


# --------------------------------------------------------------------------- #
# java analyzer txt
# --------------------------------------------------------------------------- #

def normalize_root(text: str) -> str:
    """Namer.java's only non-English note name is 'H#' (B-sharp)."""
    return "B" + text[1:] if text.startswith("H") else text


def root_to_pc(text: str) -> Optional[int]:
    text = normalize_root(text.strip())
    if not text or text[0].upper() not in PC_OF_LETTER:
        return None
    pc = PC_OF_LETTER[text[0].upper()]
    for char in text[1:]:
        if char == "#":
            pc += 1
        elif char in ("b", "-"):
            pc -= 1
        else:
            return None
    return pc % 12


def parse_java_key(key_text: str) -> Optional[Tuple[int, str]]:
    """'A MOLL' -> (9, 'm'); 'Eb DUR' -> (3, 'M')."""
    key_text = (key_text or "").strip()
    if not key_text or key_text == "null":
        return None
    parts = key_text.split("/")[0].split()
    if len(parts) != 2:
        return None
    pc = root_to_pc(parts[0])
    if pc is None:
        return None
    mode_text = parts[1].upper()
    if mode_text.startswith("DUR"):
        return pc, "M"
    if mode_text.startswith("MOLL"):
        return pc, "m"
    return None


@dataclass
class JavaAnalysis:
    keys: List[Tuple[int, str]]          # per row, in order, nulls dropped
    chords: List[Tuple[int, str]]        # (root pc, chord type) per row with a chord


def read_java_txt(path: Path) -> JavaAnalysis:
    keys: List[Tuple[int, str]] = []
    chords: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(":", 9)
            if len(parts) not in {8, 9, 10}:
                continue
            try:
                int(parts[0].strip())
            except ValueError:
                continue
            parsed_key = parse_java_key(parts[4])
            if parsed_key is not None:
                keys.append(parsed_key)
            chord_text = parts[2].strip()
            if chord_text and chord_text != "null":
                pieces = chord_text.split()
                if len(pieces) == 2:
                    pc = root_to_pc(pieces[0])
                    if pc is not None:
                        chords.append((pc, pieces[1]))
    return JavaAnalysis(keys=keys, chords=chords)


def java_majority_key(keys: Sequence[Tuple[int, str]]) -> Optional[Tuple[int, str]]:
    """Row-majority java key, skipping a leading C DUR / A MOLL warm-up block."""
    if not keys:
        return None
    voting = list(keys)
    if voting[0] in JAVA_DEFAULT_KEYS:
        first = voting[0]
        leading_end = next((i for i, k in enumerate(voting) if k != first), len(voting))
        rest = voting[leading_end:]
        if rest and Counter(rest).most_common(1)[0][0] != first:
            voting = rest
    return Counter(voting).most_common(1)[0][0]


# --------------------------------------------------------------------------- #
# profile correlation (single-profile baselines)
# --------------------------------------------------------------------------- #

def profile_key(chroma: Sequence[float], profile_name: str) -> Optional[Tuple[int, str]]:
    """Plain 24-way argmax for one key-weight profile - the classic algorithm."""
    if sum(chroma) <= 0:
        return None
    best: Optional[Tuple[float, int, str]] = None
    for mode in ("M", "m"):
        for tonic in range(12):
            score = correlate(chroma, rotated_profile(profile_name, mode, tonic))
            if best is None or score > best[0]:
                best = (score, tonic, mode)
    return (best[1], best[2]) if best else None


def detect_key_prediction(record: Dict[str, object]) -> Optional[Tuple[int, str]]:
    """The two-stage detector in key_detect.py, run from cached features."""
    features = MidiFeatures(
        chroma=record["chroma"],
        chroma_onset=record["chroma_onset"],
        bass_chroma=record["bass_chroma"],
        tail_chroma=record["tail_chroma"],
        final_bass_pc=record.get("final_bass_pc"),
        final_top_pc=record.get("final_top_pc"),
        first_bass_pc=record.get("first_bass_pc"),
        lowest_pc=record.get("lowest_pc"),
        note_count=record["note_count"],
    )
    result = detect_key_from_features(features)
    return (result.tonic_pc, result.mode) if result else None


# --------------------------------------------------------------------------- #
# extraction driver
# --------------------------------------------------------------------------- #

def iter_pieces(corpus: Path) -> List[Tuple[str, Path, Optional[Path]]]:
    pieces: List[Tuple[str, Path, Optional[Path]]] = []
    for midi_path in sorted(corpus.glob("*.mid")):
        stem = midi_path.stem
        if parse_ground_truth(stem) is None:
            continue
        txt_path = midi_path.with_suffix(".txt")
        pieces.append((stem, midi_path, txt_path if txt_path.is_file() else None))
    return pieces


def extract_one(job: Tuple[str, str, Optional[str]]) -> Optional[Dict[str, object]]:
    stem, midi_path, txt_path = job
    ground_truth = parse_ground_truth(stem)
    if ground_truth is None:
        return None
    features = extract_midi_features(Path(midi_path))
    if features is None:
        return None
    record: Dict[str, object] = {
        "name": stem,
        "gt_pc": ground_truth[0],
        "gt_mode": ground_truth[1],
        **asdict(features),
    }
    if txt_path:
        java = read_java_txt(Path(txt_path))
        majority = java_majority_key(java.keys)
        record["java_pc"] = majority[0] if majority else None
        record["java_mode"] = majority[1] if majority else None
        record["java_first"] = list(java.keys[0]) if java.keys else None
        record["java_key_rows"] = len(java.keys)
        record["chords"] = [[pc, quality] for pc, quality in java.chords]
    return record


def build_cache(corpus: Path, cache: Path, jobs: int, limit: Optional[int]) -> None:
    pieces = iter_pieces(corpus)
    if limit:
        pieces = pieces[:limit]
    print(f"extracting features for {len(pieces)} name-keyed pieces ...", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    payload = [(stem, str(midi), str(txt) if txt else None) for stem, midi, txt in pieces]
    written = 0
    with cache.open("w", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for index, record in enumerate(pool.map(extract_one, payload, chunksize=8), 1):
                if record is not None:
                    handle.write(json.dumps(record) + "\n")
                    written += 1
                if index % 200 == 0:
                    print(f"  {index}/{len(pieces)}", flush=True)
    print(f"wrote {written} records to {cache}", flush=True)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def collection_of(pc: int, mode: str) -> int:
    """Diatonic collection id = the relative major's tonic pc."""
    return pc if mode == "M" else (pc + 3) % 12


def classify_error(
    truth: Tuple[int, str],
    prediction: Optional[Tuple[int, str]],
) -> str:
    if prediction is None:
        return "missing"
    gt_pc, gt_mode = truth
    pc, mode = prediction
    if (pc, mode) == truth:
        return "exact"
    if collection_of(pc, mode) == collection_of(gt_pc, gt_mode):
        return "relative"
    if pc == gt_pc:
        return "parallel"
    if pc == (gt_pc + 7) % 12:
        return "dominant"
    if pc == (gt_pc + 5) % 12:
        return "subdominant"
    return "other"


@dataclass
class Tally:
    total: int = 0
    exact: int = 0
    mode_ok: int = 0
    collection_ok: int = 0
    tonic_given_collection: int = 0
    errors: Counter = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = Counter()

    def add(self, truth: Tuple[int, str], prediction: Optional[Tuple[int, str]]) -> None:
        self.total += 1
        outcome = classify_error(truth, prediction)
        self.errors[outcome] += 1
        if prediction is None:
            return
        if outcome == "exact":
            self.exact += 1
        if prediction[1] == truth[1]:
            self.mode_ok += 1
        if collection_of(*prediction) == collection_of(*truth):
            self.collection_ok += 1
            if outcome == "exact":
                self.tonic_given_collection += 1


def percent(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:5.1f}%" if denominator else "    - "


def report(cache: Path) -> None:
    records = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        print("no records in cache")
        return

    predictors = ["java"] + list(PROFILES) + ["detect"]
    subsets = {
        "all": lambda record: True,
        "major": lambda record: record["gt_mode"] == "M",
        "minor": lambda record: record["gt_mode"] == "m",
    }
    tallies: Dict[str, Dict[str, Tally]] = {
        subset: {name: Tally() for name in predictors} for subset in subsets
    }
    # Additional chroma variants for the best-performing profile family.
    variant_tallies = {name: Tally() for name in ("aarden@bass", "aarden@onset", "aarden@tail")}
    agreement = Counter()

    for record in records:
        truth = (record["gt_pc"], record["gt_mode"])
        predictions: Dict[str, Optional[Tuple[int, str]]] = {}
        java_pc = record.get("java_pc")
        predictions["java"] = (java_pc, record["java_mode"]) if java_pc is not None else None
        for name in PROFILES:
            predictions[name] = profile_key(record["chroma"], name)
        predictions["detect"] = detect_key_prediction(record)
        for subset, keep in subsets.items():
            if not keep(record):
                continue
            for name in predictors:
                tallies[subset][name].add(truth, predictions[name])
        for variant, field in (
            ("aarden@bass", "bass_chroma"),
            ("aarden@onset", "chroma_onset"),
            ("aarden@tail", "tail_chroma"),
        ):
            variant_tallies[variant].add(truth, profile_key(record[field], "aarden"))

        if predictions["java"] and predictions["aarden"]:
            agree = predictions["java"] == predictions["aarden"]
            correct = predictions["java"] == truth
            agreement[("agree" if agree else "disagree", "correct" if correct else "wrong")] += 1

    def print_block(title: str, rows: Dict[str, Tally]) -> None:
        print(f"\n{title}")
        print(f"  {'predictor':<14} {'n':>5} {'exact':>7} {'mode':>7} {'collection':>11} {'tonic|coll':>11}")
        for name, tally in rows.items():
            print(
                f"  {name:<14} {tally.total:>5} "
                f"{percent(tally.exact, tally.total):>7} "
                f"{percent(tally.mode_ok, tally.total):>7} "
                f"{percent(tally.collection_ok, tally.total):>11} "
                f"{percent(tally.tonic_given_collection, tally.collection_ok):>11}"
            )

    print(f"\n{'='*78}\nkey benchmark - {len(records)} name-keyed pieces\n{'='*78}")
    for subset in subsets:
        print_block(f"[{subset}]", tallies[subset])

    print_block("[all] aarden profile on alternative chroma weightings", variant_tallies)

    print("\nerror taxonomy (all pieces, % of pieces)")
    categories = ["exact", "relative", "parallel", "dominant", "subdominant", "other", "missing"]
    print(f"  {'predictor':<14}" + "".join(f"{category:>13}" for category in categories))
    for name in predictors:
        tally = tallies["all"][name]
        print(
            f"  {name:<14}"
            + "".join(f"{percent(tally.errors[category], tally.total):>13}" for category in categories)
        )

    agree_total = sum(count for (state, _), count in agreement.items() if state == "agree")
    disagree_total = sum(count for (state, _), count in agreement.items() if state == "disagree")
    print("\njava vs aarden agreement as a confidence signal")
    print(f"  agree    n={agree_total:<5} java correct {percent(agreement[('agree', 'correct')], agree_total)}")
    print(f"  disagree n={disagree_total:<5} java correct {percent(agreement[('disagree', 'correct')], disagree_total)}")

    oracle_report(records)
    offset_report(records)
    confidence_report(records)
    label_noise_report(records)


def confidence_report(records: Sequence[Dict[str, object]]) -> None:
    """Is the detector's collection margin usable as a flag-for-review gate?"""
    buckets = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.00)]
    rows = {bucket: [0, 0] for bucket in buckets}
    vote_rows = {votes: [0, 0] for votes in (3, 4, 5)}
    for record in records:
        features = MidiFeatures(
            chroma=record["chroma"], chroma_onset=record["chroma_onset"],
            bass_chroma=record["bass_chroma"], tail_chroma=record["tail_chroma"],
            final_bass_pc=record.get("final_bass_pc"), final_top_pc=record.get("final_top_pc"),
            first_bass_pc=record.get("first_bass_pc"), lowest_pc=record.get("lowest_pc"),
            note_count=record["note_count"],
        )
        result = detect_key_from_features(features)
        if result is None:
            continue
        correct = (result.tonic_pc, result.mode) == (record["gt_pc"], record["gt_mode"])
        for low, high in buckets:
            if low <= result.confidence < high:
                rows[(low, high)][1] += 1
                rows[(low, high)][0] += int(correct)
                break
        if result.rotation_votes in vote_rows:
            vote_rows[result.rotation_votes][1] += 1
            vote_rows[result.rotation_votes][0] += int(correct)

    print("\ndetector confidence calibration (collection margin)")
    for (low, high), (correct, total) in rows.items():
        print(f"  margin {low:.2f}-{high:.2f}  n={total:<5} exact {percent(correct, total)}")
    print("\ndetector confidence calibration (rotation votes)")
    for votes, (correct, total) in vote_rows.items():
        print(f"  {votes}/5 votes      n={total:<5} exact {percent(correct, total)}")


# --------------------------------------------------------------------------- #
# oracle diagnostics: is the loss in the collection or in the rotation?
# --------------------------------------------------------------------------- #

def tonic_from_collection(record: Dict[str, object], collection: int, method: str) -> Tuple[int, str]:
    """Pick between the two candidates of a diatonic collection (relative pair)."""
    candidates = [(collection, "M"), ((collection + 9) % 12, "m")]
    if method.startswith("profile:"):
        profile_name = method.split(":", 1)[1]
        weights = PROFILES[profile_name]
        best = None
        for pc, mode in candidates:
            base = weights["major" if mode == "M" else "minor"]
            rotated = [base[(index - pc) % 12] for index in range(12)]
            score = correlate(record["chroma"], rotated)
            if best is None or score > best[0]:
                best = (score, (pc, mode))
        return best[1]
    if method == "final_bass":
        pc = record.get("final_bass_pc")
        return candidates[1] if pc == candidates[1][0] else candidates[0]
    if method == "lowest":
        pc = record.get("lowest_pc")
        return candidates[1] if pc == candidates[1][0] else candidates[0]
    if method == "tail_chroma":
        tail = record["tail_chroma"]
        return candidates[0] if tail[candidates[0][0]] >= tail[candidates[1][0]] else candidates[1]
    if method == "java_mode":
        mode = record.get("java_mode")
        return candidates[1] if mode == "m" else candidates[0]
    raise ValueError(method)


def oracle_report(records: Sequence[Dict[str, object]]) -> None:
    """Force the true collection, then measure pure tonic (relative-pair) choice.

    Unlike `tonic|coll` in the main table this is not conditioned on the
    predictor having found the collection, so it is an unbiased estimate of how
    hard the rotation problem actually is.
    """
    methods = [
        "profile:aarden", "profile:ks", "profile:bellman", "profile:tkp",
        "profile:simple", "final_bass", "lowest", "tail_chroma", "java_mode",
    ]
    print("\noracle-collection tonic choice (true collection given, pick the relative pair)")
    print(f"  {'method':<18} {'all':>8} {'major':>8} {'minor':>8}")
    for method in methods:
        counts = {"all": [0, 0], "major": [0, 0], "minor": [0, 0]}
        for record in records:
            truth = (record["gt_pc"], record["gt_mode"])
            prediction = tonic_from_collection(record, collection_of(*truth), method)
            subset = "major" if truth[1] == "M" else "minor"
            for bucket in ("all", subset):
                counts[bucket][1] += 1
                if prediction == truth:
                    counts[bucket][0] += 1
        print(
            f"  {method:<18} "
            + "".join(f"{percent(counts[b][0], counts[b][1]):>8}" for b in ("all", "major", "minor"))
        )

    # And the converse: how often is the collection itself right?
    print("\ncollection accuracy alone (mode ignored)")
    for name in PROFILES:
        correct = sum(
            1 for record in records
            if (lambda p: p is not None and collection_of(*p) == collection_of(record["gt_pc"], record["gt_mode"]))(
                profile_key(record["chroma"], name)
            )
        )
        print(f"  {name:<10} {percent(correct, len(records))}")


def offset_report(records: Sequence[Dict[str, object]]) -> None:
    """Where do the wrong answers land, in fifths, relative to the truth?"""
    print("\naarden error structure: distance on the circle of fifths (signed, collection level)")
    histogram = Counter()
    for record in records:
        prediction = profile_key(record["chroma"], "aarden")
        if prediction is None:
            continue
        truth_collection = collection_of(record["gt_pc"], record["gt_mode"])
        predicted_collection = collection_of(*prediction)
        # semitone difference -> position on the circle of fifths
        semitones = (predicted_collection - truth_collection) % 12
        fifths = (semitones * 7) % 12          # inverse of *7 mod 12
        fifths = fifths - 12 if fifths > 6 else fifths
        histogram[fifths] += 1
    total = sum(histogram.values())
    for fifths in sorted(histogram):
        label = {0: "same collection", 1: "+1 sharp (dominant)", -1: "-1 flat (subdominant)"}.get(
            fifths, f"{fifths:+d} fifths"
        )
        print(f"  {label:<24} {histogram[fifths]:>5}  {percent(histogram[fifths], total)}")


def label_noise_report(records: Sequence[Dict[str, object]]) -> None:
    """Unanimous-predictors-vs-label disagreement bounds the filename label noise."""
    unanimous_wrong = 0
    unanimous = 0
    for record in records:
        predictions = {profile_key(record["chroma"], name) for name in PROFILES}
        if len(predictions) != 1:
            continue
        unanimous += 1
        if predictions.pop() != (record["gt_pc"], record["gt_mode"]):
            unanimous_wrong += 1
    print("\nlabel-noise probe (all five profiles unanimous)")
    print(f"  unanimous on {unanimous}/{len(records)} pieces ({percent(unanimous, len(records))})")
    print(f"  unanimous but disagreeing with the filename: {unanimous_wrong} ({percent(unanimous_wrong, unanimous)})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh", action="store_true", help="rebuild the feature cache")
    args = parser.parse_args(argv)

    if args.refresh or not args.cache.is_file():
        build_cache(args.corpus, args.cache, args.jobs, args.limit)
    report(args.cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
