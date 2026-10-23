"""Test whether the AE_style_jokerParam_change adapter (Space) makes the style joker change chord:
test/eval_chord_change.py for the style joker.

Same press points (structural changes and holds inside a chord), free and Space runs, metrics,
paired statistics and verdicts as eval_chord_change.py. Buttons and style come from the automatic
player of test/eval_style_chord_strength.py, in two button modes:
  guided - the button the frozen encoder gives the pianist's real note at every generated note. At
           hold points the pianist stays in the previous chord, so its contour pulls toward that chord
           and Space has to work against it.
  joker  - the joker is held from the press on, together with Space.
Results and summary.json are written per button mode, in --output/guided and --output/joker.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_chord_change import KINDS, press_points, print_summary, summarize
from eval_chord_strength import read_piece, require
from eval_style_chord_strength import MODES, load_style_checkpoint, perform
from x_transformer import AE_style_jokerParam_chords


def generate(model: AE_style_jokerParam_chords, piece: Dict[str, np.ndarray], press: int, noise: torch.Tensor,
             context: int, temperature: float, mode: str, device: torch.device) -> Dict[str, list]:
    """Rows = seeds x (free, change); Space is held from the first generated note on."""
    seeds, notes = noise.shape[:2]
    rows = [(seed, name) for seed in range(seeds) for name in ('free', 'change')]
    length = context + notes
    change = torch.zeros(len(rows), length, dtype=torch.bool, device=device)
    change[[row for row, (_, name) in enumerate(rows) if name == 'change'], context:] = True  # Space held
    controls = {'chord_chroma': torch.zeros(len(rows), length, 12, dtype=torch.bool, device=device),
                'chord_anchor_pc': torch.zeros(len(rows), length, dtype=torch.long, device=device),
                'chord_active': torch.zeros(len(rows), length, dtype=torch.bool, device=device),
                'chord_change': change}
    return perform(model, piece, press, rows, controls, noise, context, temperature, mode, device)


def run_file(path: Path, file_index: int, model: AE_style_jokerParam_chords, args: argparse.Namespace,
             device: torch.device) -> Dict[str, dict]:
    """One record per button mode, in eval_chord_change.py's format."""
    piece = read_piece(path)
    points = press_points(piece, args)
    records = {mode: {'midi': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'performed_notes': len(piece['pitch']),
                      'usable': {kind: len(value) for kind, value in points.items()}, 'presses': []}
               for mode in MODES}
    for kind_index, kind in enumerate(KINDS):
        count = min(args.presses, len(points[kind]))
        chosen = points[kind][np.linspace(0, len(points[kind]) - 1, count).round().astype(int)]  # spread over the piece
        for press_index, press in enumerate(chosen.tolist()):
            rng = torch.Generator().manual_seed(args.seed + 1000 * file_index + 100 * kind_index + press_index)
            noise = torch.empty(args.seeds, args.notes, 128).exponential_(generator=rng)
            for mode in MODES:
                records[mode]['presses'].append({
                    'kind': kind, 'press': press,
                    'previous': np.flatnonzero(piece['chroma'][press - 1]).tolist(),
                    'new': np.flatnonzero(piece['chroma'][press]).tolist(),
                    'real_continuation': piece['pitch'][press:press + args.notes].tolist(),
                    **generate(model, piece, press, noise, args.context, args.temperature, mode, device)})
        print(f'{path.name}: {count} {kind} presses done', flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'save_models/AE_style_jokerParam_change_best.pth')
    parser.add_argument('--midi-dir', type=Path, default=ROOT / 'samples/not_seen')
    parser.add_argument('--midi', nargs='*', help='Only these file names from --midi-dir (default: all)')
    parser.add_argument('--output', type=Path, default=ROOT / 'out/style_chord_change')
    parser.add_argument('--presses', type=int, default=20, help='Press points per kind and file, spread over the piece')
    parser.add_argument('--seeds', type=int, default=2, help='Sampling seeds per press point')
    parser.add_argument('--notes', type=int, default=16, help='Generated notes per run')
    parser.add_argument('--horizons', type=int, nargs='+', default=[4, 10], help='N for the metrics @N')
    parser.add_argument('--min-notes', type=int, default=4, help='Training: chord_change_min_notes')
    parser.add_argument('--min-new-tones', type=int, default=2, help='Training: chord_change_min_new_tones')
    parser.add_argument('--context', type=int, default=300, help='Primer / sliding context, as in performance')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    require(max(args.horizons) <= args.notes and args.context + args.notes <= 512, 'Invalid horizons or context')
    require(1 <= args.presses <= 100, '--presses must be in 1..100 (noise seeds)')
    torch.set_num_threads(args.threads)
    if not args.summarize_only:
        device = torch.device(args.device)
        model = load_style_checkpoint(args.checkpoint, device)
        require(model.decoder.change_film is not None, f'No change adapter in {args.checkpoint}')
        files = sorted(path for path in args.midi_dir.iterdir() if path.suffix.lower() in ('.mid', '.midi'))
        # The index in the folder seeds the noise, so one-file runs reproduce a full run.
        for file_index, path in enumerate(files):
            if not args.midi or path.name in args.midi:
                for mode, record in run_file(path, file_index, model, args, device).items():
                    (args.output / mode).mkdir(parents=True, exist_ok=True)
                    (args.output / mode / (path.stem + '.json')).write_text(json.dumps(record))
    for mode in MODES:
        print(f'Button mode: {mode}')
        print_summary(summarize(args.output / mode, args))


if __name__ == '__main__':
    main()
