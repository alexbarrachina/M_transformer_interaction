#===================================================================================================
# Monster Genie oracle_control_stream evaluation
# Generates 4 oracle-button continuations, computes per-test metrics, and builds an HTML report.
#
# Copyright 2025 Alex Barrachina
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===================================================================================================

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from FMD import FMD_score
from params import *  # noqa: F401,F403 - project modules expect this import side-effect.
from model_loader import load_model
from models import get_model_hparams
from midiUtils import dict_to_song, midi_to_dict, ms_SONG_to_MIDI_Converter, to_device
from metrics_visualizer import EvaluationCase, build_evaluation_report


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = PROJECT_ROOT.parent / "samples"
OUTPUT_DIR = PROJECT_ROOT / "out" / "oracle_control_stream"
REPORT_DIR = OUTPUT_DIR / "report"

MODEL_NAME = "AE_no_dtime_saturation_v1"
TEST_IDS = [1, 2, 3, 4]
DEFAULT_CTX_LEN = 64
MIN_CONTINUATION_NOTES = 16
TEMPERATURE = 0.0001
TIMINGS_MULTIPLIER = 2


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clone_token_dict(token_dict: Dict[str, Sequence[int]]) -> Dict[str, List[int]]:
    return {key: [int(value) for value in values] for key, values in token_dict.items()}


def as_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def select_context_len(num_notes: int, default_ctx_len: int, min_continuation_notes: int) -> int:
    max_context_len = num_notes - min_continuation_notes
    if max_context_len < 2:
        raise ValueError(
            f"Sequence is too short for oracle evaluation: num_notes={num_notes}, "
            f"requires at least {min_continuation_notes + 2} notes."
        )
    return min(default_ctx_len, max_context_len)


def build_full_context(token_dict: Dict[str, List[int]], device: torch.device) -> Dict[str, torch.Tensor]:
    context = {
        "dtime": torch.tensor(token_dict["dtime"], dtype=torch.long).unsqueeze(0),
        "pitch": torch.tensor(token_dict["pitch"], dtype=torch.long).unsqueeze(0),
        "dur": torch.tensor(token_dict["dur"], dtype=torch.long).unsqueeze(0),
    }
    return to_device(context, device)


def extract_oracle_buttons(model, token_dict: Dict[str, List[int]], device: torch.device) -> List[int]:
    context = build_full_context(token_dict, device)
    with torch.inference_mode():
        encoded = model.encoder(context)
        buttons = model.real_to_discrete(encoded).squeeze(0).detach().cpu().tolist()
    return [int(button) for button in buttons]


def generate_oracle_continuation(
    model,
    ground_truth_tokens: Dict[str, List[int]],
    oracle_buttons: Sequence[int],
    ctx_len: int,
    device: torch.device,
    temperature: float,
) -> Dict[str, List[int]]:
    generated_tokens = clone_token_dict(ground_truth_tokens)
    generated_pitches = list(generated_tokens["pitch"])
    num_notes = len(generated_pitches)

    for start_idx in range(0, num_notes - ctx_len):
        target_idx = start_idx + ctx_len
        context = {
            "dtime": torch.tensor(ground_truth_tokens["dtime"][start_idx:target_idx + 1], dtype=torch.long).unsqueeze(0),
            "pitch": torch.tensor(generated_pitches[start_idx:target_idx + 1], dtype=torch.long).unsqueeze(0),
            "dur": torch.tensor(ground_truth_tokens["dur"][start_idx:target_idx + 1], dtype=torch.long).unsqueeze(0),
            "button": torch.tensor(oracle_buttons[start_idx:target_idx + 1], dtype=torch.long).unsqueeze(0),
        }
        context = to_device(context, device)

        with torch.inference_mode():
            new_pitch_token = model.gen_pitch_token(context, temperature=temperature)

        generated_pitch = as_int(new_pitch_token)
        generated_pitches[target_idx] = generated_pitch
        generated_tokens["pitch"][target_idx] = generated_pitch

    return generated_tokens


def save_tokens_to_midi(token_dict: Dict[str, List[int]], output_stem: Path) -> Path:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    song_dict = {
        "dtime": token_dict["dtime"],
        "pitch": token_dict["pitch"],
        "dur": token_dict["dur"],
    }
    song_data = dict_to_song(song_dict)
    ms_SONG_to_MIDI_Converter(
        song_data,
        output_file_name=str(output_stem),
        timings_multiplier=TIMINGS_MULTIPLIER,
    )
    return output_stem.with_suffix(".mid")


def save_button_roll_midi(
    oracle_buttons: Sequence[int],
    reference_tokens: Dict[str, List[int]],
    output_stem: Path,
) -> Path:
    button_roll_tokens = {
        "dtime": list(reference_tokens["dtime"]),
        "dur": list(reference_tokens["dur"]),
        "pitch": [int(button) + 60 for button in oracle_buttons[: len(reference_tokens["pitch"])]],
    }
    return save_tokens_to_midi(button_roll_tokens, output_stem)


def compute_pitch_similarity(reference_pitches: Sequence[int], generated_pitches: Sequence[int]) -> Tuple[float, int]:
    reference = np.asarray(reference_pitches, dtype=np.int64)
    generated = np.asarray(generated_pitches, dtype=np.int64)
    if reference.shape != generated.shape:
        raise ValueError(f"Pitch sequences must have the same shape, got {reference.shape} vs {generated.shape}")
    matches = int(np.sum(reference == generated))
    similarity = matches / len(reference) if len(reference) else 0.0
    return float(similarity), matches


def write_metrics_csv(rows: Iterable[Dict[str, object]], csv_path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    torch.manual_seed(0)
    device = get_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_NAME}")
    cfg = get_model_hparams(MODEL_NAME)
    model = load_model(model_name=MODEL_NAME, cfg=cfg)
    model.to(device)
    model.eval()

    report_cases: List[EvaluationCase] = []
    metrics_rows: List[Dict[str, object]] = []

    for test_id in TEST_IDS:
        sample_path = SAMPLES_DIR / f"test{test_id}.midi"
        if not sample_path.exists():
            raise FileNotFoundError(f"Missing test file: {sample_path}")

        print("")
        print(f"Running oracle evaluation for {sample_path.name}")
        ground_truth_tokens, num_notes = midi_to_dict(str(sample_path))
        ground_truth_tokens = clone_token_dict(ground_truth_tokens)
        ctx_len = select_context_len(num_notes, DEFAULT_CTX_LEN, MIN_CONTINUATION_NOTES)
        continuation_len = num_notes - ctx_len

        oracle_buttons = extract_oracle_buttons(model, ground_truth_tokens, device)
        generated_tokens = generate_oracle_continuation(
            model=model,
            ground_truth_tokens=ground_truth_tokens,
            oracle_buttons=oracle_buttons,
            ctx_len=ctx_len,
            device=device,
            temperature=TEMPERATURE,
        )

        generated_full_midi = save_tokens_to_midi(
            generated_tokens,
            OUTPUT_DIR / f"oracle_test{test_id}_generated_full",
        )
        save_button_roll_midi(
            oracle_buttons,
            ground_truth_tokens,
            OUTPUT_DIR / f"oracle_test{test_id}_oracle_buttons",
        )

        pitch_similarity, matching_notes = compute_pitch_similarity(
            ground_truth_tokens["pitch"][ctx_len:],
            generated_tokens["pitch"][ctx_len:],
        )
        ground_truth_fmd = float(FMD_score(str(sample_path)))
        generated_fmd = float(FMD_score(str(generated_full_midi)))

        metrics = {
            "pitch_similarity": pitch_similarity,
            "ground_truth_fmd": ground_truth_fmd,
            "generated_fmd": generated_fmd,
            "matching_notes": float(matching_notes),
            "continuation_notes": float(continuation_len),
        }

        report_cases.append(
            EvaluationCase(
                name=f"Test {test_id}",
                ground_truth_tokens=ground_truth_tokens,
                generated_tokens=generated_tokens,
                button_values=oracle_buttons,
                metrics=metrics,
                generated_midi_path=generated_full_midi,
                ground_truth_midi_path=sample_path,
                context_len=ctx_len,
                metadata={
                    "Sample": sample_path.name,
                    "Total notes": num_notes,
                    "Context notes": ctx_len,
                    "Continuation notes": continuation_len,
                },
            )
        )

        metrics_rows.append(
            {
                "test_id": test_id,
                "sample_file": sample_path.name,
                "num_notes": num_notes,
                "context_notes": ctx_len,
                "continuation_notes": continuation_len,
                "matching_notes": matching_notes,
                "pitch_similarity": pitch_similarity,
                "ground_truth_fmd": ground_truth_fmd,
                "generated_fmd": generated_fmd,
                "generated_full_midi": str(generated_full_midi),
            }
        )

        print(
            f"  context={ctx_len} | continuation={continuation_len} | "
            f"pitch_similarity={pitch_similarity:.2%} | "
            f"ground_truth_fmd={ground_truth_fmd:.4f} | generated_fmd={generated_fmd:.4f}"
        )

    metrics_csv_path = OUTPUT_DIR / "oracle_control_stream_metrics.csv"
    write_metrics_csv(metrics_rows, metrics_csv_path)

    report_path = build_evaluation_report(
        report_cases,
        REPORT_DIR,
        title="Oracle Control Stream Evaluation",
        subtitle=(
            "Pitch similarity is computed on the continuation region only. "
            "FMD scores are computed independently for the input MIDI and the generated MIDI against the saved giantMIDI reference distribution. "
            "The piano rolls show the full sequence with a dashed line at the seed/generation boundary."
        ),
    )

    print("")
    print(f"Saved metrics CSV: {metrics_csv_path}")
    print(f"Saved HTML report: {report_path}")


if __name__ == "__main__":
    main()
