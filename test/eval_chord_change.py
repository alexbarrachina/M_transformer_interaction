"""Test whether the D_Chord_Joker change adapter (Space) makes the generation change chord.

The change control never names the chord, so the evidence is measured against the chord that
was sounding before the press (the previous chord, from the MIDI's channel-5 labels).

Protocol (paired: free and change runs share the primer and the sampling noise):
  1. Each test MIDI is read exactly as in test/eval_chord_strength.py (= midis2pickles.py +
     ChordPitchSamplerDataset). The real performance up to the press is the primer (--context notes).
  2. Two kinds of press point, spread over each piece:
       change - a structural chord change of the piece, as in training: the previous chord lasts
                at least --min-notes notes, the new one too, and it adds --min-new-tones pitch classes.
       hold   - inside a chord that the pianist plays for --min-notes notes before the press and
                keeps for all --notes notes after it: a change here can only come from the control,
                as when a performer presses Space at any moment.
     At each point the model generates --notes notes free (no control) and with Space held from
     the first generated note on.
  3. Per run, over the first N generated notes (model probabilities, not samples):
       previous-chord mass  - probability that a note is a tone of the previous chord (should drop)
       new minus previous   - change points only: mass on tones only in the pianist's new chord minus
                              mass on tones only in the previous chord (should rise)
       outlines a new chord - 1 when some major/minor/diminished triad with a tone outside the previous
                              chord gets more mass than every triad inside it (should rise)
       triad fit            - mass of the best triad: the notes still form a chord, they are not noise
       base NLL             - frozen-base negative log-likelihood of the generated notes (plausibility)
     Change minus free per press point, with a paired bootstrap CI and a sign-flip permutation test.
     The pianist's real continuation is scored the same way (one-hot notes) as a reference.

Generation results are saved per MIDI file in --output; --summarize-only rewrites summary.json.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_chord_strength import bootstrap_ci, load_checkpoint, permutation_p, read_piece, require
from x_transformer import D_Chord_Joker

KINDS = ('change', 'hold')
METRICS = ('previous_mass', 'new_minus_previous', 'new_chord', 'triad_fit', 'base_nll')
# Major, minor and diminished triads on every root: [36, 12] pitch-class masks.
TRIADS = np.zeros((36, 12), dtype=bool)
for _root in range(12):
    for _quality, _intervals in enumerate(((0, 4, 7), (0, 3, 7), (0, 3, 6))):
        TRIADS[12 * _quality + _root, [(_root + interval) % 12 for interval in _intervals]] = True


def press_points(piece: Dict[str, np.ndarray], args: argparse.Namespace) -> Dict[str, np.ndarray]:
    """Notes where Space is pressed: real structural changes, and notes inside a held chord."""
    chroma = piece['chroma']
    index = np.arange(len(chroma))
    start = np.r_[True, (chroma[1:] != chroma[:-1]).any(axis=1)]  # first note of a chord run
    run = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    last = np.r_[first[1:], len(chroma)] - 1
    before = index - first[run] + 1  # notes of this chord up to and including this note
    after = last[run] - index + 1    # notes of this chord from this note on
    previous = np.maximum(index - 1, 0)
    labelled = chroma.any(axis=1)
    usable = labelled & (index >= max(args.context, 1)) & (index + args.notes <= len(chroma))
    established = labelled[previous] & (before[previous] >= args.min_notes)
    new_tones = (chroma & ~chroma[previous]).sum(axis=1)
    change = usable & start & established & (after >= args.min_notes) & (new_tones >= args.min_new_tones)
    hold = usable & ~start & established & (after >= args.notes)
    return {'change': index[change], 'hold': index[hold]}


@torch.inference_mode()
def generate(model: D_Chord_Joker, piece: Dict[str, np.ndarray], press: int, noise: torch.Tensor,
             context: int, temperature: float, device: torch.device) -> Dict[str, list]:
    """Rows = seeds x (free, change); all rows share the real primer and the noise of their seed."""
    seeds, notes = noise.shape[:2]
    rows = [(seed, name) for seed in range(seeds) for name in ('free', 'change')]
    length = context + notes
    pitch = torch.zeros(len(rows), length, dtype=torch.long, device=device)
    pitch[:, :context] = torch.as_tensor(piece['pitch'][press - context:press], device=device)
    change = torch.zeros(len(rows), length, dtype=torch.bool, device=device)
    change[[row for row, (_, name) in enumerate(rows) if name == 'change'], context:] = True  # Space held
    controls = {'chord_chroma': torch.zeros(len(rows), length, 12, dtype=torch.bool, device=device),
                'chord_anchor_pc': torch.zeros(len(rows), length, dtype=torch.long, device=device),
                'chord_active': torch.zeros(len(rows), length, dtype=torch.bool, device=device),
                'chord_change': change}
    to_pc = F.one_hot(torch.arange(128, device=device) % 12, 12).float()
    seed_of_row = torch.tensor([seed for seed, _ in rows], device=device)
    noise = noise.to(device)
    pc_mass = []
    for step in range(notes):
        position = context + step
        window = slice(position - context, position + 1)  # same sliding context as performance
        logits = model.next_pitch_logits({'pitch': pitch[:, window],
                                          **{key: value[:, window] for key, value in controls.items()}})
        pc_mass.append(F.softmax(logits / temperature, dim=-1) @ to_pc)
        # Gumbel-max with noise shared by the free and change runs (common random numbers).
        pitch[:, position] = (logits / temperature - noise[seed_of_row, step].log()).argmax(-1)
    # Plausibility: log-likelihood of each note under the frozen base (decoder alone = no control).
    real = torch.as_tensor(piece['pitch'][press - context:press + notes], device=device).unsqueeze(0)
    scored = torch.cat((pitch, real))
    base = F.log_softmax(model.decoder({'pitch': scored[:, :-1]}).float(), dim=-1)
    base_logp = base[:, context - 1:].gather(-1, scored[:, context:].unsqueeze(-1)).squeeze(-1).cpu()
    return {'rows': rows, 'pitches': pitch[:, context:].tolist(),
            'pc_mass': torch.stack(pc_mass, dim=1).cpu().round(decimals=6).tolist(),  # round is not on MPS
            'base_logp': base_logp[:-1].round(decimals=5).tolist(),
            'real_base_logp': base_logp[-1].round(decimals=5).tolist()}


def run_file(path: Path, file_index: int, model: D_Chord_Joker, args: argparse.Namespace,
             device: torch.device) -> dict:
    piece = read_piece(path)
    points = press_points(piece, args)
    record = {'midi': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'performed_notes': len(piece['pitch']),
              'usable': {kind: len(value) for kind, value in points.items()}, 'presses': []}
    for kind_index, kind in enumerate(KINDS):
        count = min(args.presses, len(points[kind]))
        chosen = points[kind][np.linspace(0, len(points[kind]) - 1, count).round().astype(int)]  # spread over the piece
        for press_index, press in enumerate(chosen.tolist()):
            rng = torch.Generator().manual_seed(args.seed + 1000 * file_index + 100 * kind_index + press_index)
            noise = torch.empty(args.seeds, args.notes, 128).exponential_(generator=rng)
            record['presses'].append({
                'kind': kind, 'press': press,
                'previous': np.flatnonzero(piece['chroma'][press - 1]).tolist(),
                'new': np.flatnonzero(piece['chroma'][press]).tolist(),
                'real_continuation': piece['pitch'][press:press + args.notes].tolist(),
                **generate(model, piece, press, noise, args.context, args.temperature, device)})
        print(f'{path.name}: {count} {kind} presses done', flush=True)
    return record


#==========================================================================

def run_metrics(mass: np.ndarray, logp: np.ndarray, previous: np.ndarray, new: np.ndarray,
                horizon: int) -> Dict[str, float]:
    """One run over its first N notes. mass: [notes, 12] pitch-class probabilities (one-hot for
    the pianist); previous/new: [12] chord masks."""
    profile = mass[:horizon].mean(axis=0)
    triads = TRIADS @ profile
    inside = ~(TRIADS & ~previous).any(axis=1)  # triads made of previous-chord tones only
    return {'previous_mass': float(profile[previous].sum()),
            'new_minus_previous': float(profile[new & ~previous].sum() - profile[previous & ~new].sum()),
            # Undefined when the previous chord holds no such triad (about 1 % of the labels).
            'new_chord': float(triads[~inside].max() > triads[inside].max()) if inside.any() else np.nan,
            'triad_fit': float(triads.max()),
            'base_nll': float(-logp[:horizon].mean())}


def press_metrics(entry: dict, horizon: int) -> Dict[str, Dict[str, float]]:
    """free / change (mean over seeds) and pianist values for one press point."""
    previous, new = np.zeros(12, dtype=bool), np.zeros(12, dtype=bool)
    previous[entry['previous']], new[entry['new']] = True, True
    runs = {'free': [], 'change': []}
    for row, (_, name) in enumerate(entry['rows']):
        runs[name].append(run_metrics(np.asarray(entry['pc_mass'][row]), np.asarray(entry['base_logp'][row]),
                                      previous, new, horizon))
    result = {name: {key: float(np.mean([run[key] for run in values])) for key in METRICS}
              for name, values in runs.items()}
    real = np.eye(12)[np.asarray(entry['real_continuation']) % 12]
    result['pianist'] = run_metrics(real, np.asarray(entry['real_base_logp']), previous, new, horizon)
    return result


def summarize(output: Path, args: argparse.Namespace) -> dict:
    records = [json.loads(path.read_text()) for path in sorted(output.glob('*.json')) if path.name != 'summary.json']
    require(bool(records), f'No generation results in {output}')
    rng = np.random.default_rng(args.seed)
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    summary = {'protocol': protocol,
               'files': {record['midi']: {'performed_notes': record['performed_notes'], 'usable': record['usable'],
                                          'tested': {kind: sum(entry['kind'] == kind for entry in record['presses'])
                                                     for kind in KINDS}} for record in records},
               'results': {}, 'per_note': {}, 'verdict': {}}
    for kind in KINDS:
        entries = [entry for record in records for entry in record['presses'] if entry['kind'] == kind]
        if not entries:
            continue
        for horizon in args.horizons:
            rows = [press_metrics(entry, horizon) for entry in entries]
            block = {}
            # At hold points the previous chord continues, so there is no new chord to aim at.
            for metric in METRICS if kind == 'change' else [m for m in METRICS if m != 'new_minus_previous']:
                values = {name: np.asarray([row[name][metric] for row in rows]) for name in ('free', 'change', 'pianist')}
                paired = ~np.isnan(values['free']) & ~np.isnan(values['change'])
                difference = values['change'][paired] - values['free'][paired]
                block[metric] = {'free': float(values['free'][paired].mean()),
                                 'change': float(values['change'][paired].mean()),
                                 'pianist': float(np.nanmean(values['pianist'])),
                                 'change_minus_free': float(difference.mean()),
                                 'ci95': bootstrap_ci(difference, rng), 'p_permutation': permutation_p(difference, rng),
                                 'change_higher_fraction': float((difference > 0).mean()),
                                 'change_lower_fraction': float((difference < 0).mean()), 'presses': int(paired.sum())}
            summary['results'][f'{kind}@{horizon}'] = block
        # Per generated note k: probability that note k is a tone of the previous chord.
        curves = {'free': [], 'change': []}
        for entry in entries:
            for name in curves:
                curves[name].append(np.mean([np.asarray(entry['pc_mass'][row])[:, entry['previous']].sum(-1)
                                             for row, (_, run) in enumerate(entry['rows']) if run == name], axis=0))
        difference = np.asarray(curves['change']) - np.asarray(curves['free'])
        summary['per_note'][kind] = {
            'free': np.mean(curves['free'], axis=0).tolist(), 'change': np.mean(curves['change'], axis=0).tolist(),
            'change_minus_free_ci95': [bootstrap_ci(difference[:, k], rng) for k in range(difference.shape[1])],
            'pianist': np.mean([np.isin(np.asarray(entry['real_continuation']) % 12, entry['previous'])
                                for entry in entries], axis=0).tolist()}
        # Verdict at the first horizon: a significant paired effect in the expected direction.
        first = summary['results'][f'{kind}@{args.horizons[0]}']
        lower = lambda metric: first[metric]['ci95'][1] < 0 and first[metric]['p_permutation'] < 0.01
        higher = lambda metric: first[metric]['ci95'][0] > 0 and first[metric]['p_permutation'] < 0.01
        summary['verdict'][kind] = {'leaves_previous_chord': lower('previous_mass'),
                                    'outlines_a_new_chord': higher('new_chord')}
        if kind == 'change':
            summary['verdict'][kind]['moves_toward_pianist_new_chord'] = higher('new_minus_previous')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def print_summary(summary: dict) -> None:
    names = {'previous_mass': 'previous-chord mass', 'new_minus_previous': 'new minus previous',
             'new_chord': 'outlines a new chord', 'triad_fit': 'triad fit', 'base_nll': 'base NLL'}
    print('=' * 112)
    print(f'{"press@N":<10} {"metric":<21} {"free":>7} {"change":>7} {"pianist":>8}   '
          f'{"change - free [95% CI]":<28} {"p":>8}  change higher/lower')
    for key, block in summary['results'].items():
        for metric, value in block.items():
            print(f'{key:<10} {names[metric]:<21} {value["free"]:7.3f} {value["change"]:7.3f} {value["pianist"]:8.3f}   '
                  f'{value["change_minus_free"]:+.3f} [{value["ci95"][0]:+.3f}, {value["ci95"][1]:+.3f}]{"":<6} '
                  f'{value["p_permutation"]:8.1e}  {100 * value["change_higher_fraction"]:3.0f} % / '
                  f'{100 * value["change_lower_fraction"]:3.0f} % of {value["presses"]}')
        print('-' * 112)
    print('Per generated note k: probability that note k is a tone of the previous chord')
    for kind, block in summary['per_note'].items():
        print(f'{kind}: {"k":>3} {"free":>7} {"change":>7} {"pianist":>8}   change - free [95% CI]')
        for k in range(len(block['free'])):
            ci = block['change_minus_free_ci95'][k]
            print(f'{"":<{len(kind) + 1}} {k + 1:>4} {block["free"][k]:7.3f} {block["change"][k]:7.3f} '
                  f'{block["pianist"][k]:8.2f}   {block["change"][k] - block["free"][k]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]')
    print('-' * 112)
    for kind, checks in summary['verdict'].items():
        print(f'Verdict at {kind} presses: ' + ', '.join(f'{check.replace("_", " ")}: '
                                                        f'{"CONFIRMED" if passed else "NOT SHOWN"}'
                                                        for check, passed in checks.items()))
    print('=' * 112)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'save_models/D_Chord_Joker_change_latest_best.pth')
    parser.add_argument('--midi-dir', type=Path, default=ROOT / 'samples/not_seen')
    parser.add_argument('--midi', nargs='*', help='Only these file names from --midi-dir (default: all)')
    parser.add_argument('--output', type=Path, default=ROOT / 'out/chord_change')
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
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        device = torch.device(args.device)
        model = load_checkpoint(args.checkpoint, device)
        require(isinstance(model, D_Chord_Joker) and model.decoder.change_film is not None,
                f'Not a D_Chord_Joker checkpoint with a change adapter: {args.checkpoint}')
        files = sorted(path for path in args.midi_dir.iterdir() if path.suffix.lower() in ('.mid', '.midi'))
        # The index in the folder seeds the noise, so one-file runs reproduce a full run.
        for file_index, path in enumerate(files):
            if not args.midi or path.name in args.midi:
                record = run_file(path, file_index, model, args, device)
                (args.output / (path.stem + '.json')).write_text(json.dumps(record))
    print_summary(summarize(args.output, args))


if __name__ == '__main__':
    main()
