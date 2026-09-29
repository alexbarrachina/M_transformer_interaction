"""Reusable evaluation report builder for piano-roll comparisons."""

from __future__ import annotations

import html
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np
from scipy.io import wavfile


_RUNTIME_HOME = Path(__file__).resolve().parent / ".runtime_home"
(_RUNTIME_HOME / ".cache").mkdir(parents=True, exist_ok=True)
(_RUNTIME_HOME / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_RUNTIME_HOME / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_RUNTIME_HOME / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import pretty_midi
except ImportError:  # pragma: no cover - handled at runtime if the dependency is missing.
    pretty_midi = None


ROLL_COLORS = {
    "ground_truth": "#4477DD",
    "generated": "#F56B45",
    "buttons": "#1A9A6A",
}


@dataclass
class EvaluationCase:
    """All assets and metadata required to render one evaluation panel."""

    name: str
    ground_truth_tokens: Mapping[str, Sequence[int]]
    generated_tokens: Mapping[str, Sequence[int]]
    button_values: Sequence[int]
    metrics: Mapping[str, float]
    generated_midi_path: Path
    ground_truth_midi_path: Optional[Path] = None
    context_len: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)


def _slugify(text: str) -> str:
    filtered = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            filtered.append(char.lower())
        else:
            filtered.append("_")
    slug = "".join(filtered).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "case"


def _to_list(values: Sequence[int]) -> list[int]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


def _token_dict_to_lists(token_dict: Mapping[str, Sequence[int]]) -> Dict[str, list[int]]:
    return {key: _to_list(value) for key, value in token_dict.items()}


def _compute_note_starts(dtimes: Sequence[int]) -> np.ndarray:
    dtimes_array = np.asarray(_to_list(dtimes), dtype=np.float32)
    if dtimes_array.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.cumsum(dtimes_array)


def _format_metric_label(metric_name: str, value: float) -> str:
    if metric_name == "pitch_similarity":
        return f"Pitch similarity: {value:.2%}"
    if metric_name == "ground_truth_fmd":
        return f"Input FMD vs giantMIDI: {value:.4f}"
    if metric_name == "generated_fmd":
        return f"Generated FMD vs giantMIDI: {value:.4f}"
    if metric_name == "matching_notes":
        return f"Matching notes: {int(round(value))}"
    if metric_name == "continuation_notes":
        return f"Continuation notes: {int(round(value))}"
    pretty_name = metric_name.replace("_", " ").strip().title()
    return f"{pretty_name}: {value:.4f}"


def _format_metric_value(metric_name: str, value: float) -> str:
    if metric_name == "pitch_similarity":
        return f"{value:.2%}"
    if metric_name in {"matching_notes", "continuation_notes"}:
        return str(int(round(value)))
    return f"{value:.4f}"


def _render_roll(ax: plt.Axes, starts: np.ndarray, durations: Sequence[int], values: Sequence[int], color: str, title: str, y_label: str) -> None:
    durations_list = _to_list(durations)
    values_list = _to_list(values)

    for start, duration, value in zip(starts, durations_list, values_list):
        width = max(float(duration), 0.8)
        ax.add_patch(
            Rectangle(
                (float(start), float(value) - 0.42),
                width,
                0.84,
                facecolor=color,
                edgecolor="none",
                alpha=0.9,
            )
        )

    if values_list:
        ax.set_ylim(min(values_list) - 2, max(values_list) + 2)
    ax.set_title(title, loc="left", fontsize=11, pad=8)
    ax.set_ylabel(y_label)
    ax.grid(True, color="#DFE4EA", linewidth=0.6)
    ax.set_axisbelow(True)


def _render_case_figure(case: EvaluationCase, figure_path: Path) -> Path:
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    ground_truth = _token_dict_to_lists(case.ground_truth_tokens)
    generated = _token_dict_to_lists(case.generated_tokens)
    button_values = _to_list(case.button_values)

    ground_truth_starts = _compute_note_starts(ground_truth["dtime"])
    generated_starts = _compute_note_starts(generated["dtime"])

    max_end = 1.0
    if ground_truth_starts.size and ground_truth["dur"]:
        max_end = max(max_end, float(np.max(ground_truth_starts + np.asarray(_to_list(ground_truth["dur"]), dtype=np.float32))))
    if generated_starts.size and generated["dur"]:
        max_end = max(max_end, float(np.max(generated_starts + np.asarray(_to_list(generated["dur"]), dtype=np.float32))))

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 2.4, 1.3]},
    )

    fig.patch.set_facecolor("white")
    metric_lines = [_format_metric_label(name, float(value)) for name, value in case.metrics.items()]
    fig.suptitle(case.name, x=0.07, ha="left", fontsize=15, fontweight="bold")
    fig.text(
        0.07,
        0.94,
        "\n".join(metric_lines),
        ha="left",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#F6F8FA",
            "edgecolor": "#D0D7DE",
            "alpha": 0.95,
        },
    )

    _render_roll(
        axes[0],
        ground_truth_starts,
        ground_truth["dur"],
        ground_truth["pitch"],
        ROLL_COLORS["ground_truth"],
        "Ground Truth",
        "Pitch",
    )
    _render_roll(
        axes[1],
        generated_starts,
        generated["dur"],
        generated["pitch"],
        ROLL_COLORS["generated"],
        "Generated Output",
        "Pitch",
    )
    _render_roll(
        axes[2],
        ground_truth_starts[: len(button_values)],
        ground_truth["dur"][: len(button_values)],
        button_values,
        ROLL_COLORS["buttons"],
        "Oracle Buttons",
        "Button",
    )

    boundary_x = None
    if 0 < case.context_len < len(ground_truth_starts):
        boundary_x = float(ground_truth_starts[case.context_len])
    elif case.context_len == len(ground_truth_starts) and ground_truth_starts.size:
        boundary_x = float(ground_truth_starts[-1])

    for ax in axes:
        ax.set_xlim(0, max_end)
        if boundary_x is not None:
            ax.axvline(boundary_x, color="#5E6C84", linestyle="--", linewidth=1.2)
            ax.text(
                boundary_x,
                ax.get_ylim()[1],
                "  generation starts",
                ha="left",
                va="top",
                fontsize=9,
                color="#344054",
                backgroundcolor="white",
            )

    axes[-1].set_xlabel("Quantized Time")
    fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.9))
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def synthesize_midi_to_wav(midi_path: Path, wav_path: Path, sample_rate: int = 16000) -> Path:
    """Render MIDI to a simple audio preview using pretty_midi's sine synthesizer."""

    if pretty_midi is None:
        raise RuntimeError(
            "pretty_midi is required to render audio previews. "
            "Install it in the evaluation environment before running the visualizer."
        )

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    audio = midi.synthesize(fs=sample_rate)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    wavfile.write(wav_path, sample_rate, np.int16(np.clip(audio, -1.0, 1.0) * 32767))
    return wav_path


def _copy_metric_values(metrics: Mapping[str, float]) -> Dict[str, float]:
    return {name: float(value) for name, value in metrics.items()}


def _summarize_metrics(cases: Sequence[EvaluationCase]) -> Dict[str, float]:
    aggregates: MutableMapping[str, list[float]] = {}
    for case in cases:
        for name, value in _copy_metric_values(case.metrics).items():
            aggregates.setdefault(name, []).append(value)
    return {name: float(np.mean(values)) for name, values in aggregates.items() if values}


def _metadata_to_html(metadata: Mapping[str, object]) -> str:
    if not metadata:
        return ""
    pieces = []
    for key, value in metadata.items():
        pieces.append(f"<span><strong>{html.escape(str(key))}:</strong> {html.escape(str(value))}</span>")
    return "<div class='metadata'>" + "".join(pieces) + "</div>"


def _metric_grid_to_html(metrics: Mapping[str, float]) -> str:
    cells = []
    for name, value in metrics.items():
        cells.append(
            "<div class='metric-card'>"
            f"<div class='metric-name'>{html.escape(name.replace('_', ' ').title())}</div>"
            f"<div class='metric-value'>{html.escape(_format_metric_value(name, float(value)))}</div>"
            "</div>"
        )
    return "<div class='metric-grid'>" + "".join(cells) + "</div>"


def build_evaluation_report(
    cases: Sequence[EvaluationCase],
    output_dir: Path,
    title: str = "Evaluation Report",
    subtitle: Optional[str] = None,
) -> Path:
    """Render piano-roll figures, audio previews, and an HTML report."""

    if not cases:
        raise ValueError("At least one evaluation case is required.")

    output_dir = Path(output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    summary_metrics = _summarize_metrics(cases)
    case_sections = []

    for case in cases:
        slug = _slugify(case.name)
        figure_path = _render_case_figure(case, assets_dir / f"{slug}.png")
        generated_audio_path = synthesize_midi_to_wav(case.generated_midi_path, assets_dir / f"{slug}_generated.wav")
        ground_truth_audio_html = ""

        if case.ground_truth_midi_path is not None:
            gt_audio_path = synthesize_midi_to_wav(case.ground_truth_midi_path, assets_dir / f"{slug}_ground_truth.wav")
            ground_truth_audio_html = (
                "<div class='player-block'>"
                "<div class='player-label'>Ground truth</div>"
                f"<audio controls preload='none' src='{html.escape(gt_audio_path.relative_to(output_dir).as_posix())}'></audio>"
                "</div>"
            )

        case_sections.append(
            "<section class='case-card'>"
            f"<h2>{html.escape(case.name)}</h2>"
            f"{_metadata_to_html(case.metadata)}"
            f"{_metric_grid_to_html(case.metrics)}"
            f"<img class='figure' src='{html.escape(figure_path.relative_to(output_dir).as_posix())}' alt='{html.escape(case.name)} piano roll' />"
            "<div class='players'>"
            f"{ground_truth_audio_html}"
            "<div class='player-block'>"
            "<div class='player-label'>Generated</div>"
            f"<audio controls preload='none' src='{html.escape(generated_audio_path.relative_to(output_dir).as_posix())}'></audio>"
            "</div>"
            "</div>"
            "</section>"
        )

    summary_html = _metric_grid_to_html(summary_metrics)
    subtitle_html = f"<p class='subtitle'>{html.escape(subtitle)}</p>" if subtitle else ""

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif;
      background: #F4F6F8;
      color: #182230;
    }}
    main {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 32px 24px 56px;
    }}
    h1 {{
      margin-bottom: 6px;
      font-size: 32px;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 24px;
    }}
    .subtitle {{
      margin-top: 0;
      color: #475467;
      max-width: 960px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .metric-card {{
      padding: 14px 16px;
      border-radius: 14px;
      background: white;
      border: 1px solid #D0D7DE;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }}
    .metric-name {{
      color: #475467;
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .metric-value {{
      font-size: 24px;
      font-weight: 700;
    }}
    .case-card {{
      margin-top: 26px;
      padding: 24px;
      background: white;
      border-radius: 20px;
      border: 1px solid #D0D7DE;
      box-shadow: 0 12px 40px rgba(15, 23, 42, 0.06);
    }}
    .metadata {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      color: #475467;
      font-size: 14px;
      margin-bottom: 8px;
    }}
    .figure {{
      width: 100%;
      margin: 12px 0 20px;
      border-radius: 16px;
      border: 1px solid #D0D7DE;
      background: white;
    }}
    .players {{
      display: flex;
      flex-wrap: wrap;
      gap: 20px;
    }}
    .player-block {{
      flex: 1 1 320px;
      min-width: 280px;
    }}
    .player-label {{
      margin-bottom: 8px;
      font-weight: 600;
    }}
    audio {{
      width: 100%;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(title)}</h1>
    {subtitle_html}
    <section class='case-card'>
      <h2>Average Metrics</h2>
      {summary_html}
    </section>
    {''.join(case_sections)}
  </main>
</body>
</html>
"""

    report_path = output_dir / "report.html"
    report_path.write_text(report_html, encoding="utf-8")
    return report_path
