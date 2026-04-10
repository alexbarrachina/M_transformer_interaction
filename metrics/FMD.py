#!/usr/bin/env python3
"""CLI wrapper for Frechet Music Distance with reusable reference stats."""

import argparse
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np


MIDI_SUFFIXES = {".mid", ".midi"}


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def candidate_reference_dirs(project_root: Path) -> List[Path]:
    return [
        (project_root / "../../../../DataSets/MIDI/giantMIDI/midis").resolve(),
        (project_root / "../../../../../DataSets/MIDI/giantMIDI/midis").resolve(),
        Path("/Volumes/DADES/DataSets/MIDI/giantMIDI/midis"),
    ]


def get_default_reference_dir(project_root: Path) -> Path:
    for candidate in candidate_reference_dirs(project_root):
        if candidate.exists():
            return candidate
    return candidate_reference_dirs(project_root)[0]


def parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Frechet Music Distance for local MIDI files and folders.",
    )
    parser.add_argument(
        "--reference-dir",
        default=str(get_default_reference_dir(project_root)),
        help="Reference dataset directory. Only .mid/.midi files are used.",
    )
    parser.add_argument(
        "--samples-dir",
        default=str(project_root / "samples"),
        help="Directory with MIDI files to evaluate.",
    )
    parser.add_argument(
        "--sample-file",
        help="Single MIDI file to score against the saved reference distribution.",
    )
    parser.add_argument(
        "--cache-home",
        default=str(project_root / ".runtime_home"),
        help="Project-local runtime home used for FMD, Hugging Face, and Torch caches.",
    )
    parser.add_argument(
        "--reference-stats",
        help="Optional .npz path for the saved reference mean and covariance.",
    )
    parser.add_argument(
        "--build-reference-stats",
        action="store_true",
        help="Build and save the reference distribution, then exit.",
    )
    parser.add_argument(
        "--force-rebuild-reference-stats",
        action="store_true",
        help="Recompute the saved reference distribution even if it already exists.",
    )
    parser.add_argument(
        "--model",
        choices=["clamp2", "clamp"],
        default="clamp2",
        help="Embedding model to use. MIDI requires clamp2.",
    )
    parser.add_argument(
        "--estimator",
        choices=["mle", "bootstrap", "oas", "shrinkage", "leodit_wolf"],
        default="mle",
        help="Gaussian estimator used by frechet_music_distance.",
    )
    parser.add_argument(
        "--inf",
        action="store_true",
        help="Compute FMD-Inf instead of the standard FMD score.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of FMD-Inf regression points.",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=500,
        help="Minimum sample size used for FMD-Inf.",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Also report per-song scores for each MIDI file in the samples directory.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the frechet_music_distance joblib cache before computing.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress logging from the FMD package.",
    )
    return parser.parse_args()


def resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def iter_midi_files(directory: Path) -> Iterable[Path]:
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in MIDI_SUFFIXES:
            yield path


def collect_midi_files(directory: Path) -> List[Path]:
    return sorted(iter_midi_files(directory))


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "reference"


def get_default_reference_stats_path(project_root: Path, reference_dir: Path, model: str, estimator: str) -> Path:
    label = slugify(f"{reference_dir.parent.name}_{reference_dir.name}")
    digest = hashlib.sha1(str(reference_dir).encode("utf-8")).hexdigest()[:12]
    return project_root / ".reference_stats" / f"{label}_{digest}_{model}_{estimator}.npz"


def prepare_runtime_home(cache_home: Path) -> None:
    cache_root = cache_home / ".cache"
    for directory in (
        cache_home,
        cache_root,
        cache_root / "frechet_music_distance" / "precomputed",
        cache_root / "frechet_music_distance" / "checkpoints",
        cache_root / "huggingface" / "hub",
        cache_root / "torch",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["HOME"] = str(cache_home)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "huggingface" / "hub")
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def validate_inputs(
    args: argparse.Namespace,
    reference_dir: Path,
    samples_dir: Path,
    sample_file: Optional[Path],
) -> None:
    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference directory does not exist: {reference_dir}")
    if args.model != "clamp2":
        raise ValueError("MIDI evaluation requires --model clamp2 because the clamp model supports ABC only.")
    if args.sample_file and args.per_file:
        raise ValueError("Use either --sample-file or --per-file, not both.")
    if args.sample_file and args.inf:
        raise ValueError("--inf requires a dataset directory, not --sample-file.")
    if sample_file is not None:
        if not sample_file.exists():
            raise FileNotFoundError(f"Sample file does not exist: {sample_file}")
        if not sample_file.is_file():
            raise ValueError(f"Sample path is not a file: {sample_file}")
        if sample_file.suffix.lower() not in MIDI_SUFFIXES:
            raise ValueError(f"Sample file must be one of {sorted(MIDI_SUFFIXES)}: {sample_file}")
    elif not args.build_reference_stats:
        if not samples_dir.exists():
            raise FileNotFoundError(f"Samples directory does not exist: {samples_dir}")
        if not samples_dir.is_dir():
            raise ValueError(f"Samples path is not a directory: {samples_dir}")


def print_reference_summary(reference_dir: Path, reference_files: List[Path]) -> None:
    print(f"Reference dataset: {reference_dir}")
    print(f"Reference MIDI files found: {len(reference_files)}")
    print("Included extensions: .mid, .midi")


def print_dataset_summary(reference_dir: Path, samples_dir: Path, reference_files: List[Path], sample_files: List[Path]) -> None:
    print(f"Reference dataset: {reference_dir}")
    print(f"Samples dataset:   {samples_dir}")
    print(f"Reference MIDI files found: {len(reference_files)}")
    print(f"Sample MIDI files found:    {len(sample_files)}")
    print("Included extensions: .mid, .midi")


def print_single_file_summary(reference_dir: Path, sample_file: Path, reference_files: List[Path]) -> None:
    print(f"Reference dataset: {reference_dir}")
    print(f"Sample file:       {sample_file}")
    print(f"Reference MIDI files found: {len(reference_files)}")
    print("Included extensions: .mid, .midi")


def compute_fmd(
    mean_reference: np.ndarray,
    mean_test: np.ndarray,
    covariance_reference: np.ndarray,
    covariance_test: np.ndarray,
    eps: float = 1e-6,
) -> float:
    import scipy.linalg

    mu_test = np.atleast_1d(mean_test)
    mu_ref = np.atleast_1d(mean_reference)
    sigma_test = np.atleast_2d(covariance_test)
    sigma_ref = np.atleast_2d(covariance_reference)

    if mu_test.shape != mu_ref.shape:
        raise ValueError(f"Reference and test mean vectors differ: {mu_ref.shape} vs {mu_test.shape}")
    if sigma_test.shape != sigma_ref.shape:
        raise ValueError(f"Reference and test covariances differ: {sigma_ref.shape} vs {sigma_test.shape}")

    diff = mu_test - mu_ref
    covmean, _ = scipy.linalg.sqrtm(sigma_test.dot(sigma_ref), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_test.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma_test + offset).dot(sigma_ref + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma_test) + np.trace(sigma_ref) - 2 * tr_covmean).item()


def compute_fmd_inf(
    mean_reference: np.ndarray,
    covariance_reference: np.ndarray,
    test_features: np.ndarray,
    max_likelihood_estimator: Any,
    steps: int,
    min_n: int,
) -> Tuple[float, float, float]:
    max_n = len(test_features)
    if min_n >= max_n:
        raise ValueError(f"--min-n ({min_n}) must be smaller than the number of test items ({max_n}).")

    ns = [int(n) for n in np.linspace(min_n, max_n, steps)]
    rng = np.random.default_rng()
    results = []

    for n in ns:
        indices = rng.choice(test_features.shape[0], size=n, replace=True)
        sample_test_features = test_features[indices]
        mean_test, cov_test = max_likelihood_estimator.estimate_parameters(sample_test_features)
        results.append([n, compute_fmd(mean_reference, mean_test, covariance_reference, cov_test)])

    ys = np.array(results)
    xs = 1 / np.array(ns)
    slope, intercept = np.polyfit(xs, ys[:, 1], 1)
    residual = np.sum((ys[:, 1] - (slope * xs + intercept)) ** 2)
    total = np.sum((ys[:, 1] - np.mean(ys[:, 1])) ** 2)
    r2 = 1 - residual / total
    return intercept.item(), slope.item(), r2.item()


def save_reference_stats(
    stats_path: Path,
    reference_dir: Path,
    model: str,
    estimator: str,
    reference_file_count: int,
    mean_reference: np.ndarray,
    covariance_reference: np.ndarray,
) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stats_path,
        reference_dir=str(reference_dir),
        model=model,
        estimator=estimator,
        reference_file_count=reference_file_count,
        mean=mean_reference,
        covariance=covariance_reference,
    )


def load_reference_stats(
    stats_path: Path,
    reference_dir: Path,
    model: str,
    estimator: str,
) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(stats_path, allow_pickle=False) as data:
        saved_reference_dir = str(data["reference_dir"].item())
        saved_model = str(data["model"].item())
        saved_estimator = str(data["estimator"].item())
        if saved_reference_dir != str(reference_dir):
            raise ValueError(
                f"Reference stats were built for {saved_reference_dir}, not the requested reference dir {reference_dir}."
            )
        if saved_model != model or saved_estimator != estimator:
            raise ValueError(
                f"Reference stats were built for model={saved_model}, estimator={saved_estimator}, "
                f"not model={model}, estimator={estimator}."
            )
        return data["mean"], data["covariance"]


def get_reference_stats(
    reference_dir: Path,
    stats_path: Path,
    reference_files: List[Path],
    extractor: Any,
    estimator: Any,
    model: str,
    estimator_name: str,
    force_rebuild: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if stats_path.exists() and not force_rebuild:
        print(f"Using saved reference stats: {stats_path}")
        return load_reference_stats(stats_path, reference_dir, model, estimator_name)

    print(f"Building reference stats: {stats_path}")
    reference_features = extractor.extract_features(reference_dir)
    mean_reference, covariance_reference = estimator.estimate_parameters(reference_features)
    save_reference_stats(
        stats_path=stats_path,
        reference_dir=reference_dir,
        model=model,
        estimator=estimator_name,
        reference_file_count=len(reference_files),
        mean_reference=mean_reference,
        covariance_reference=covariance_reference,
    )
    return mean_reference, covariance_reference


def main() -> int:
    
    score = FMD_score("./samples/test1.midi")
    print(f"Single-file FMD score: {score}")
    '''    
    project_root = get_project_root()
    args = parse_args(project_root)

    reference_dir = resolve_path(args.reference_dir, project_root)
    samples_dir = resolve_path(args.samples_dir, project_root)
    cache_home = resolve_path(args.cache_home, project_root)
    sample_file = resolve_path(args.sample_file, project_root) if args.sample_file else None

    validate_inputs(args, reference_dir, samples_dir, sample_file)

    reference_files = collect_midi_files(reference_dir)
    if not reference_files:
        raise FileNotFoundError(f"No MIDI files were found under the reference directory: {reference_dir}")

    sample_files: List[Path] = []
    if sample_file is not None:
        sample_files = [sample_file]
    elif not args.build_reference_stats:
        sample_files = collect_midi_files(samples_dir)
        if not sample_files:
            raise FileNotFoundError(f"No MIDI files were found under the samples directory: {samples_dir}")

    prepare_runtime_home(cache_home)

    from frechet_music_distance.utils import clear_cache
    from frechet_music_distance.gaussian_estimators.max_likelihood_estimator import MaxLikelihoodEstimator
    from frechet_music_distance.gaussian_estimators.utils import get_estimator_by_name
    from frechet_music_distance.models.utils import get_feature_extractor_by_name

    if args.clear_cache:
        shutil.rmtree(cache_home / ".cache" / "frechet_music_distance", ignore_errors=True)
        clear_cache()

    if args.build_reference_stats:
        print_reference_summary(reference_dir, reference_files)
    elif sample_file is not None:
        print_single_file_summary(reference_dir, sample_file, reference_files)
    else:
        print_dataset_summary(reference_dir, samples_dir, reference_files, sample_files)
    print(f"Runtime cache home: {cache_home}")

    extractor = get_feature_extractor_by_name(args.model, verbose=(not args.quiet))
    estimator = get_estimator_by_name(args.estimator)
    max_likelihood_estimator = MaxLikelihoodEstimator()

    stats_path = (
        resolve_path(args.reference_stats, project_root)
        if args.reference_stats
        else get_default_reference_stats_path(project_root, reference_dir, args.model, args.estimator)
    )

    mean_reference, covariance_reference = get_reference_stats(
        reference_dir=reference_dir,
        stats_path=stats_path,
        reference_files=reference_files,
        extractor=extractor,
        estimator=estimator,
        model=args.model,
        estimator_name=args.estimator,
        force_rebuild=args.force_rebuild_reference_stats,
    )
    print(f"Reference stats file: {stats_path}")

    if args.build_reference_stats:
        return 0

    if sample_file is not None:
        single_feature = extractor.extract_feature(sample_file)
        score = compute_fmd(
            mean_reference,
            single_feature.flatten(),
            covariance_reference,
            covariance_reference,
        )
        print("")
        print(f"Single-file FMD score: {score}")
        return 0

    test_features = extractor.extract_features(samples_dir)

    if args.inf:
        score, slope, r2 = compute_fmd_inf(
            mean_reference=mean_reference,
            covariance_reference=covariance_reference,
            test_features=test_features,
            max_likelihood_estimator=max_likelihood_estimator,
            steps=args.steps,
            min_n=args.min_n,
        )
        print("")
        print(f"FMD-Inf score: {score}")
        print(f"Regression slope: {slope}")
        print(f"Regression R^2:   {r2}")
    else:
        mean_test, covariance_test = estimator.estimate_parameters(test_features)
        score = compute_fmd(mean_reference, mean_test, covariance_reference, covariance_test)
        print("")
        print(f"FMD score: {score}")

    if args.per_file:
        print("")
        print("Per-file scores:")
        for midi_file in sample_files:
            feature = extractor.extract_feature(midi_file)
            song_score = compute_fmd(
                mean_reference,
                feature.flatten(),
                covariance_reference,
                covariance_reference,
            )
            print(f"{midi_file.name}: {song_score}")

    return 0
'''

def FMD_score(sample_file: Optional[Union[str, Path]]) -> int:
    if sample_file is not None:
        sample_file = Path(sample_file)
    project_root = get_project_root()
    args = parse_args(project_root)

    reference_dir = resolve_path(args.reference_dir, project_root)
    samples_dir = resolve_path(args.samples_dir, project_root)
    cache_home = resolve_path(args.cache_home, project_root)
    #sample_file = resolve_path(args.sample_file, project_root) if args.sample_file else None

    validate_inputs(args, reference_dir, samples_dir, sample_file)

    reference_files = collect_midi_files(reference_dir)
    if not reference_files:
        raise FileNotFoundError(f"No MIDI files were found under the reference directory: {reference_dir}")

    sample_files: List[Path] = []
    if sample_file is not None:
        sample_files = [sample_file]
    elif not args.build_reference_stats:
        sample_files = collect_midi_files(samples_dir)
        if not sample_files:
            raise FileNotFoundError(f"No MIDI files were found under the samples directory: {samples_dir}")

    prepare_runtime_home(cache_home)

    from frechet_music_distance.utils import clear_cache
    from frechet_music_distance.gaussian_estimators.max_likelihood_estimator import MaxLikelihoodEstimator
    from frechet_music_distance.gaussian_estimators.utils import get_estimator_by_name
    from frechet_music_distance.models.utils import get_feature_extractor_by_name

    if args.clear_cache:
        shutil.rmtree(cache_home / ".cache" / "frechet_music_distance", ignore_errors=True)
        clear_cache()

    if args.build_reference_stats:
        print_reference_summary(reference_dir, reference_files)
    elif sample_file is not None:
        print_single_file_summary(reference_dir, sample_file, reference_files)
    else:
        print_dataset_summary(reference_dir, samples_dir, reference_files, sample_files)
    print(f"Runtime cache home: {cache_home}")

    extractor = get_feature_extractor_by_name(args.model, verbose=(not args.quiet))
    estimator = get_estimator_by_name(args.estimator)
    max_likelihood_estimator = MaxLikelihoodEstimator()

    stats_path = (
        resolve_path(args.reference_stats, project_root)
        if args.reference_stats
        else get_default_reference_stats_path(project_root, reference_dir, args.model, args.estimator)
    )

    mean_reference, covariance_reference = get_reference_stats(
        reference_dir=reference_dir,
        stats_path=stats_path,
        reference_files=reference_files,
        extractor=extractor,
        estimator=estimator,
        model=args.model,
        estimator_name=args.estimator,
        force_rebuild=args.force_rebuild_reference_stats,
    )
    print(f"Reference stats file: {stats_path}")

    if args.build_reference_stats:
        return 0

    if sample_file is not None:
        single_feature = extractor.extract_feature(sample_file)
        score = compute_fmd(
            mean_reference,
            single_feature.flatten(),
            covariance_reference,
            covariance_reference,
        )
        return score

    test_features = extractor.extract_features(samples_dir)

    if args.inf:
        score, slope, r2 = compute_fmd_inf(
            mean_reference=mean_reference,
            covariance_reference=covariance_reference,
            test_features=test_features,
            max_likelihood_estimator=max_likelihood_estimator,
            steps=args.steps,
            min_n=args.min_n,
        )
        print("")
        print(f"FMD-Inf score: {score}")
        print(f"Regression slope: {slope}")
        print(f"Regression R^2:   {r2}")
    else:
        mean_test, covariance_test = estimator.estimate_parameters(test_features)
        score = compute_fmd(mean_reference, mean_test, covariance_reference, covariance_test)
        print("")
        print(f"FMD score: {score}")

    if args.per_file:
        print("")
        print("Per-file scores:")
        for midi_file in sample_files:
            feature = extractor.extract_feature(midi_file)
            song_score = compute_fmd(
                mean_reference,
                feature.flatten(),
                covariance_reference,
                covariance_reference,
            )
            print(f"{midi_file.name}: {song_score}")

    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
