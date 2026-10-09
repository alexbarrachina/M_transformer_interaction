#!/usr/bin/env python3
"""Score every MIDI file in metrics/samples against the giantMIDI reference distribution.

Reuses the reference-stats machinery already in FMD.py (same clamp2 model, same
saved giantMIDI mean/covariance under .reference_stats/) so this stays consistent
with `python FMD.py --build-reference-stats` / `--sample-file` / `--per-file`.

For each sample file this reports:
  - a per-file score: squared distance between that file's clamp2 embedding and
    the giantMIDI mean embedding (same convention as FMD.py's --per-file option,
    where a single MIDI has no covariance of its own so the reference covariance
    is reused for both sides of the Frechet distance -- this collapses the
    covariance terms and leaves just ||mu_file - mu_reference||^2).
  - an aggregate FMD score: the real Frechet distance between the *whole*
    samples/ set treated as one distribution and the giantMIDI distribution.
    This is the number to quote for "how close is this batch of generations to
    giantMIDI overall".

Usage:
    conda activate tgenie
    python test/run_fmd_samples.py
"""

import csv
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.FMD import (
    collect_midi_files,
    compute_fmd,
    get_default_reference_dir,
    get_default_reference_stats_path,
    get_project_root,
    get_reference_stats,
    prepare_runtime_home,
)

MODEL = "clamp2"
ESTIMATOR_NAME = "mle"


def score_samples(
    samples_dir: Path,
    reference_dir: Path,
    cache_home: Path,
    model: str = MODEL,
    estimator_name: str = ESTIMATOR_NAME,
) -> Tuple[List[Tuple[str, float]], float]:
    prepare_runtime_home(cache_home)

    from frechet_music_distance.gaussian_estimators.utils import get_estimator_by_name
    from frechet_music_distance.models.utils import get_feature_extractor_by_name

    reference_files = collect_midi_files(reference_dir)
    if not reference_files:
        raise FileNotFoundError(f"No MIDI files were found under the reference directory: {reference_dir}")

    sample_files = collect_midi_files(samples_dir)
    if not sample_files:
        raise FileNotFoundError(f"No MIDI files were found under the samples directory: {samples_dir}")

    print(f"Reference dataset: {reference_dir} ({len(reference_files)} files)")
    print(f"Samples dataset:   {samples_dir} ({len(sample_files)} files)")

    project_root = get_project_root()
    extractor = get_feature_extractor_by_name(model, verbose=True)
    estimator = get_estimator_by_name(estimator_name)

    stats_path = get_default_reference_stats_path(project_root, reference_dir, model, estimator_name)
    mean_reference, covariance_reference = get_reference_stats(
        reference_dir=reference_dir,
        stats_path=stats_path,
        reference_files=reference_files,
        extractor=extractor,
        estimator=estimator,
        model=model,
        estimator_name=estimator_name,
        force_rebuild=False,
    )
    print(f"Reference stats file: {stats_path}\n")

    per_file_scores: List[Tuple[str, float]] = []
    features = []
    for midi_file in sample_files:
        try:
            feature = extractor.extract_feature(midi_file).flatten()
        except Exception as exc:  # noqa: BLE001 - report and keep scoring the rest
            print(f"  [skip] {midi_file.name}: {exc}", file=sys.stderr)
            continue

        features.append(feature)
        score = compute_fmd(mean_reference, feature, covariance_reference, covariance_reference)
        per_file_scores.append((midi_file.name, score))
        print(f"{midi_file.name}: {score:.4f}")

    if not features:
        raise RuntimeError("Feature extraction failed for every sample file; nothing to score.")

    test_features = np.stack(features)
    mean_test, covariance_test = estimator.estimate_parameters(test_features)
    aggregate_score = compute_fmd(mean_reference, mean_test, covariance_reference, covariance_test)

    per_file_scores.sort(key=lambda item: item[1])
    return per_file_scores, aggregate_score


def write_csv(csv_path: Path, per_file_scores: List[Tuple[str, float]], aggregate_score: float) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "fmd_score_vs_giantMIDI"])
        writer.writerows([(name, f"{score:.6f}") for name, score in per_file_scores])
        writer.writerow(["AGGREGATE (samples/ as one distribution)", f"{aggregate_score:.6f}"])


def main() -> int:
    project_root = get_project_root()
    samples_dir = project_root / "samples"
    reference_dir = get_default_reference_dir(project_root)
    cache_home = project_root / ".runtime_home"
    csv_path = project_root / "out" / "fmd_samples_scores.csv"

    per_file_scores, aggregate_score = score_samples(samples_dir, reference_dir, cache_home)

    print("\nPer-file FMD scores (closest to giantMIDI first):")
    for name, score in per_file_scores:
        print(f"  {score:10.4f}  {name}")

    print(f"\nAggregate FMD score (samples/ as one distribution vs giantMIDI): {aggregate_score:.4f}")

    write_csv(csv_path, per_file_scores, aggregate_score)
    print(f"\nSaved results to {csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
