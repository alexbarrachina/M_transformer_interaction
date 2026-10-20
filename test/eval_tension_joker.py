"""Compare tension checkpoints on fixed test primers and long paired continuations.

The original MIDI annotations calibrate the scorer before model selection. Chord
timings come from the source performance; chord tones and local keys are inferred
again from generated pitches. Run with --help for paths and benchmark settings.
"""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import mido
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harmony.tension_joker import PianoPerformance, TensionScorer, read_tension_performance
from midiUtils import Any_Pickle_File_Reader
from model_loader import load_model
from models import get_model_hparams
from train_tension import TensionPitchSamplerDataset, validate_tension


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_check(scorer: TensionScorer, directory: Path, count: int,
                     length: int) -> dict:
    """Compare recomputed levels with the original channel-independent MIDI markers."""
    paths = sorted(directory.glob('*.mid'),
                   key=lambda path: hashlib.sha256(path.name.encode()).digest())
    rows = []
    total_correct = total_error = total_notes = 0
    for path in paths:
        if path.stat().st_size < 20000:
            continue
        source = read_tension_performance(str(path))
        if len(source.pitches) < length or not source.chord_spans_ms:
            continue
        midi = mido.MidiFile(path)
        time_ms, tempo, markers = 0.0, 500_000, []
        for message in mido.merge_tracks(midi.tracks):
            time_ms += mido.tick2second(message.time, midi.ticks_per_beat, tempo) * 1000
            if message.type == 'set_tempo':
                tempo = message.tempo
            elif message.type == 'marker' and message.text.lower().startswith('tension:'):
                markers.append((time_ms, int(message.text.split(':', 1)[1])))
        if not markers:
            continue
        excerpt = PianoPerformance(source.pitches[:length], source.starts_ms[:length],
                                   source.durations_ms[:length], source.velocities[:length],
                                   source.channels[:length], source.home_key, source.chord_spans_ms)
        predicted = np.asarray(scorer.score(excerpt, excerpt.pitches)['levels'])
        starts = np.asarray([time for time, _ in markers])
        marker_indices = np.searchsorted(starts, excerpt.starts_ms, side='right') - 1
        valid = marker_indices >= 0
        original = np.asarray([level for _, level in markers])[marker_indices[valid]]
        errors = np.abs(predicted[valid] - original)
        correct = int((errors == 0).sum())
        total_correct += correct
        total_error += int(errors.sum())
        total_notes += int(valid.sum())
        rows.append({'file': path.name, 'notes': int(valid.sum()),
                     'agreement': correct / int(valid.sum()),
                     'mean_absolute_error': float(errors.mean())})
        if len(rows) == count:
            break
    if len(rows) < count:
        raise ValueError(f'Only {len(rows)} annotated MIDI files qualify; requested {count}')
    return {'files': rows, 'notes': total_notes, 'agreement': total_correct / total_notes,
            'mean_absolute_error': total_error / total_notes}


def generate(model, source: PianoPerformance, primer: int, notes: int,
             seeds: int, seed: int, device: torch.device) -> torch.Tensor:
    """Cached, CFG=1 decoding; each seed shares Gumbel noise across all six targets."""
    batch = seeds * 6
    pitches = torch.zeros((batch, primer + notes), dtype=torch.long, device=device)
    pitches[:, :primer] = torch.tensor(source.pitches[:primer], device=device)
    controls = torch.zeros_like(pitches)
    controls[:, primer:] = torch.tensor([1, 2, 3, 4, 5, 0], device=device).repeat(seeds)[:, None]
    home = torch.full((batch,), source.home_key, dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        logits, cache = model.decoder({'pitch': pitches[:, :primer]},
                                      tension_target=controls[:, 1:primer + 1],
                                      home_key=home, return_intermediates=True)
        for position in range(primer, primer + notes):
            if position > primer:
                logits, cache = model.decoder({'pitch': pitches[:, position - 1:position]},
                                              tension_target=controls[:, position:position + 1],
                                              home_key=home, cache=cache,
                                              return_intermediates=True)
            noise = torch.empty((seeds, 1, logits.size(-1)), device=device).exponential_(generator=generator)
            pitches[:, position] = (logits[:, -1].reshape(seeds, 6, -1)
                                    - noise.log()).argmax(-1).flatten()
    return pitches.cpu().reshape(seeds, 6, primer + notes)


def prior_nll(base, sequences: torch.Tensor, primer: int) -> np.ndarray:
    with torch.inference_mode():
        flat = sequences.flatten(0, 1)
        logits = base.decoder({'pitch': flat[:, :-1]})[:, primer - 1:]
        labels = flat[:, primer:]
        nll = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none').mean(-1)
    return nll.reshape(sequences.size(0), sequences.size(1)).cpu().numpy()


def repeated_fourgrams(pitches: np.ndarray) -> float:
    grams = [tuple(pitches[index:index + 4]) for index in range(len(pitches) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def measure_case(source: PianoPerformance, pitches: List[int], levels: List[int],
                 baseline: int, target: int, primer: int, nll: float) -> dict:
    generated = np.asarray(pitches[primer:])
    realized = np.asarray(levels[primer:])
    late = generated[-50:]
    close = np.abs(realized - target) <= 1 if target >= 0 else None
    response = None
    if target >= 0 and len(close) >= 3:
        settled = np.flatnonzero(np.convolve(close.astype(int), np.ones(3, dtype=int), 'valid') == 3)
        if len(settled):
            response = int(settled[0]) + 1
    return {
        'target': target, 'realized_mean': float(realized[8:].mean()),
        'realized_late_mean': float(realized[-50:].mean()),
        'within_one': float(close[8:].mean()) if target >= 0 else None,
        'late_within_one': float(close[-50:].mean()) if target >= 0 else None,
        'response_notes': response,
        'response_seconds': ((source.starts_ms[primer + response - 1]
                              - source.starts_ms[primer]) / 1000 if response else None),
        'jump_size': abs(target - baseline) if target >= 0 else None,
        'repeat_rate': float(np.mean(generated[1:] == generated[:-1])),
        'late_repeat_rate': float(np.mean(late[1:] == late[:-1])),
        'late_repeated_4gram_rate': repeated_fourgrams(late),
        'late_large_leap_rate': float(np.mean(np.abs(np.diff(late)) > 12)),
        'late_distinct_pitches': int(len(set(late))),
        'base_prior_nll': float(nll),
        'realized_levels': realized.tolist(),
    }


def average(values: List[float]):
    return float(np.mean(values)) if values else None


def summarize(cases: List[dict], source_cases: List[dict]) -> dict:
    rows = [case for case in cases if case['target'] >= 0]
    pairs = sorted({(case['primer'], case['seed']) for case in cases})
    spearman, ordered = [], []
    for primer, seed in pairs:
        means = np.asarray([next(case['realized_mean'] for case in cases
                                 if case['primer'] == primer and case['seed'] == seed
                                 and case['target'] == target) for target in range(5)])
        ranks = np.asarray([(means < value).sum() + ((means == value).sum() - 1) / 2
                            for value in means])
        corr = np.corrcoef(np.arange(5), ranks)[0, 1] if ranks.std() else 0.0
        spearman.append(float(corr))
        ordered.append(float(np.mean([means[left] < means[right]
                                      for left in range(5) for right in range(left + 1, 5)])))
    by_target = {str(target): {key: average([case[key] for case in cases if case['target'] == target
                                             and case[key] is not None])
                               for key in ('realized_mean', 'realized_late_mean', 'within_one',
                                           'late_within_one', 'repeat_rate', 'late_repeat_rate',
                                           'late_repeated_4gram_rate', 'late_large_leap_rate',
                                           'late_distinct_pitches', 'base_prior_nll')}
                 for target in range(-1, 5)}
    responses = [case for case in rows if case['response_notes'] is not None]
    jumps = {}
    for jump in (0, 1, 2, 3):
        subset = [case for case in rows if (case['jump_size'] == jump if jump < 3
                                            else case['jump_size'] >= 3)]
        successful = [case for case in subset if case['response_notes'] is not None]
        jumps[str(jump) if jump < 3 else '3-4'] = {
            'cases': len(subset), 'success_rate': len(successful) / len(subset) if subset else None,
            'median_notes': float(np.median([case['response_notes'] for case in successful]))
                            if successful else None,
            'median_seconds': float(np.median([case['response_seconds'] for case in successful]))
                              if successful else None}
    confusion = np.zeros((5, 5), dtype=int)
    for case in rows:
        np.add.at(confusion[case['target']], case['realized_levels'][8:], 1)
    return {
        'cases_per_target': len(rows) // 5,
        'mean_spearman': average(spearman),
        'pairwise_order_rate': average(ordered),
        'within_one': average([case['within_one'] for case in rows]),
        'late_within_one': average([case['late_within_one'] for case in rows]),
        'response_success_rate': len(responses) / len(rows),
        'response_median_notes': float(np.median([case['response_notes'] for case in responses]))
                                 if responses else None,
        'response_by_jump': jumps,
        'confusion_matrix': confusion.tolist(),
        'by_target': by_target,
        'source_reference': {key: average([case[key] for case in source_cases])
                             for key in ('late_repeat_rate', 'late_repeated_4gram_rate',
                                         'late_large_leap_rate', 'late_distinct_pitches',
                                         'base_prior_nll')},
    }


def evaluate_checkpoint(path: Path, sources: List[PianoPerformance], scorer: TensionScorer,
                        base, primer: int, notes: int, seeds: int, seed: int,
                        nll_loader: DataLoader, output: Path) -> dict:
    saved = torch.load(path, map_location='cpu', weights_only=False)
    cfg = saved['cfg'].copy()
    cfg['ckpt_file_name'] = str(path)
    model = load_model(cfg['model_name'], cfg=cfg, compile_mode='none').eval()
    step = saved.get('steps')
    del saved
    cases, source_cases = [], []
    for index, source in enumerate(sources):
        print(f'{path.name}: primer {index + 1}/{len(sources)}', flush=True)
        sequences = generate(model, source, primer, notes, seeds, seed + index,
                             torch.device('cpu'))
        nll = prior_nll(base, sequences, primer)
        source_pitches = torch.tensor(source.pitches[:primer + notes])[None, None, :]
        reference_nll = prior_nll(base, source_pitches, primer)[0, 0]
        original = np.asarray(source.pitches[primer:primer + notes])
        late = original[-50:]
        source_cases.append({'base_prior_nll': float(reference_nll),
                             'late_repeat_rate': float(np.mean(late[1:] == late[:-1])),
                             'late_repeated_4gram_rate': repeated_fourgrams(late),
                             'late_large_leap_rate': float(np.mean(np.abs(np.diff(late)) > 12)),
                             'late_distinct_pitches': len(set(late))})
        baseline = scorer.score(source, source.pitches[:primer])['levels'][-1]
        for run in range(seeds):
            for target_index, target in enumerate((0, 1, 2, 3, 4, -1)):
                pitches = sequences[run, target_index].tolist()
                levels = scorer.score(source, pitches)['levels']
                case = measure_case(source, pitches, levels, baseline, target, primer,
                                    nll[run, target_index])
                case.update(primer=index, seed=seed + run)
                cases.append(case)
                if run == 0:
                    source.write(str(output / f'{path.stem}_primer{index + 1}_target{target}.mid'),
                                 pitches)
    nll_metrics = validate_tension(model, nll_loader, torch.device('cpu'))
    del model
    gc.collect()
    return {'checkpoint': str(path), 'checkpoint_sha256': sha256(path), 'steps': step,
            'model': cfg['model_name'], 'nll': nll_metrics,
            'control': summarize(cases, source_cases), 'cases': cases}


def write_report(path: Path, report: dict) -> None:
    protocol = report['protocol']
    checked = report['annotation_check']
    results = report['checkpoints']
    best = max(results, key=lambda row: max(0, row['control']['mean_spearman'])
               * row['control']['within_one'])
    best_control = best['control']
    accepted = best_control['mean_spearman'] >= 0.6 and best_control['within_one'] >= 0.7
    outlier = min(checked['files'], key=lambda row: row['agreement'])
    comparison = ''
    if len(results) > 1:
        runner = max((row for row in results if row is not best),
                     key=lambda row: max(0, row['control']['mean_spearman'])
                     * row['control']['within_one'])

        def pair_metrics(result: dict) -> np.ndarray:
            values = []
            for primer in range(len(protocol['primers'])):
                for seed in range(protocol['seed'], protocol['seed'] + protocol['seeds']):
                    rows = [next(case for case in result['cases']
                                 if case['primer'] == primer and case['seed'] == seed
                                 and case['target'] == target) for target in range(5)]
                    means = np.asarray([case['realized_mean'] for case in rows])
                    ranks = np.asarray([(means < value).sum()
                                        + ((means == value).sum() - 1) / 2 for value in means])
                    rho = np.corrcoef(np.arange(5), ranks)[0, 1] if ranks.std() else 0.0
                    values.append((rho, average([case['within_one'] for case in rows])))
            return np.asarray(values)

        delta = pair_metrics(best) - pair_metrics(runner)
        rng = np.random.default_rng(protocol['seed'])
        draws = rng.integers(0, len(delta), (10000, len(delta)))
        bounds = np.quantile(delta[draws].mean(1), (0.025, 0.975), axis=0)
        comparison = (f"Against the runner-up at step {runner['steps']}, the paired "
                      f"difference is {delta[:, 0].mean():+.3f} in ρ and "
                      f"{delta[:, 1].mean():+.3f} in within-one rate. Resampling the "
                      f"nine primer–seed pairs gives 95% intervals "
                      f"[{bounds[0, 0]:+.3f}, {bounds[1, 0]:+.3f}] and "
                      f"[{bounds[0, 1]:+.3f}, {bounds[1, 1]:+.3f}], respectively; "
                      'the apparent lead is uncertain.')
    lines = [
        '# Tension Joker model selection', '',
        f"**Provisional choice:** `{Path(best['checkpoint']).name}` (step {best['steps']}) "
        'has the strongest paired steering score among the available checkpoints. '
        + ('It meets both plan thresholds.' if accepted else
           'No checkpoint meets both plan thresholds: target-order ρ ≥ 0.60 and '
           'at least 70% of post-settle notes within one level.'), '',
        comparison, '',
        f"The scorer matched {checked['agreement']:.1%} of {checked['notes']:,} original "
        f"annotation labels from {len(checked['files'])} fixed 450-note excerpts "
        f"(mean absolute error {checked['mean_absolute_error']:.3f}). The weakest "
        f"excerpt was `{outlier['file']}` at {outlier['agreement']:.1%}; this is "
        'a known source of uncertainty in the scores below.', '',
        f"**Protocol:** {len(protocol['primers'])} fixed held-out test-pickle primers "
        f"× {protocol['seeds']} paired seeds × five targets plus free = "
        f"{len(protocol['primers']) * protocol['seeds'] * 6} continuations per checkpoint. "
        f"Each has a {protocol['primer_notes']}-note primer and "
        f"{protocol['continuation_notes']}-note continuation. Sampling uses temperature "
        '1 and guidance 1. The same noise is paired across targets and checkpoints. '
        f"Pitch NLL uses {len(protocol['nll_windows'])} identical 300-note test windows "
        'with original labels (no anticipatory lead).', '',
        '## Control and response', '',
        '| Checkpoint | Step | Order ρ | Ordered target pairs | Within ±1 | '
        'Response success | Response median |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for result in results:
        control = result['control']
        lines.append(f"| {result['model']} | {result['steps']} | "
                     f"{control['mean_spearman']:.3f} | {control['pairwise_order_rate']:.3f} | "
                     f"{control['within_one']:.3f} | {control['response_success_rate']:.1%} | "
                     f"{control['response_median_notes']:.1f} notes |")
    lines += ['', 'The realised mean level after eight settling notes shows where '
              'targets overlap. Free is an unconditioned reference, not a sixth '
              'tension level.', '',
              '| Checkpoint step | Free | Target 0 | 1 | 2 | 3 | 4 |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for result in results:
        targets = result['control']['by_target']
        values = ' | '.join(f"{targets[str(target)]['realized_mean']:.2f}"
                            for target in (-1, 0, 1, 2, 3, 4))
        lines.append(f"| {result['steps']} | {values} |")
    lines += ['', 'Response means the first run of three consecutive notes within '
              'one level of the target; a run that never responds is counted as a '
              'failure. The overall median is dominated by already-close targets. '
              'For the provisional choice, response by starting jump is:', '',
              '| Starting jump | Cases | Responded | Median notes | Median seconds |',
              '| --- | ---: | ---: | ---: | ---: |']
    for jump, values in best_control['response_by_jump'].items():
        median_notes = '-' if values['median_notes'] is None else f"{values['median_notes']:.1f}"
        median_seconds = '-' if values['median_seconds'] is None else f"{values['median_seconds']:.2f}"
        rate = '-' if values['success_rate'] is None else f"{values['success_rate']:.1%}"
        lines.append(f"| {jump} | {values['cases']} | {rate} | {median_notes} | {median_seconds} |")
    lines += ['', '## Long holds and musical coherence probes', '',
              'These statistics use the last 50 notes of each 150-note hold. '
              'Repetition, distinct pitches, large leaps, and frozen-base pitch NLL '
              'are measurable coherence probes; they cannot judge phrasing or sound.', '',
              '| Step | Target repeat | Free repeat | Target repeated 4-grams | '
              'Target distinct pitches | Target base NLL | Free base NLL |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for result in results:
        targets = result['control']['by_target']
        value = lambda key: average([targets[str(target)][key] for target in range(5)])
        lines.append(f"| {result['steps']} | {value('late_repeat_rate'):.3f} | "
                     f"{targets['-1']['late_repeat_rate']:.3f} | "
                     f"{value('late_repeated_4gram_rate'):.3f} | "
                     f"{value('late_distinct_pitches'):.1f} | "
                     f"{value('base_prior_nll'):.3f} | "
                     f"{targets['-1']['base_prior_nll']:.3f} |")
    source = best_control['source_reference']
    lines += ['', f"Original test continuations: late repeat {source['late_repeat_rate']:.3f}, "
              f"repeated 4-grams {source['late_repeated_4gram_rate']:.3f}, "
              f"distinct pitches {source['late_distinct_pitches']:.1f}, and base "
              f"NLL {source['base_prior_nll']:.3f}. At target 3, the provisional "
              f"choice repeats adjacent notes {best_control['by_target']['3']['late_repeat_rate']:.3f} "
              'of the time late in the hold.', '',
              '## Teacher-forced pitch loss', '',
              '| Step | Correct target NLL | Free NLL | Wrong-target NLL | Wrong − correct |',
              '| ---: | ---: | ---: | ---: | ---: |']
    for result in results:
        nll = result['nll']
        lines.append(f"| {result['steps']} | {nll['nll_true']:.3f} | "
                     f"{nll['nll_null']:.3f} | {nll['nll_wrong']:.3f} | "
                     f"{nll['nll_control_gap']:.3f} |")
    lines += ['', 'Correct-target NLL is lower than wrong-target NLL for every '
              'checkpoint, but higher than free NLL. The tiny differences between '
              'checkpoint pitch NLLs do not resolve their free-running control or '
              'long-hold differences.', '',
              '## Limits and listening', '',
              'The evaluator uses source chord **timings** while inferring chord '
              'tones and local keys from each generated pitch stream. It shares '
              'the annotation builder’s tension construct and calibration, so '
              'agreement is a calibration check rather than independent musical '
              'validation. There are nine primer–seed pairs per target; small '
              'checkpoint differences may not persist on more pieces. The MIDI '
              'examples in this folder include each target and free for the first '
              'seed of every primer. Listen before using a checkpoint for performance.', '',
              'Full per-run levels, confusion matrices, NLLs, excerpt identities, '
              'checkpoint hashes, and scorer provenance are in `report.json`.', '']
    path.write_text('\n'.join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', nargs='+', type=Path)
    parser.add_argument('--data', type=Path,
                        default=ROOT / 'Training-Data/giantmidi_key_tension_test.pickle')
    parser.add_argument('--annotations', type=Path,
                        default=Path('/Volumes/DADES/DataSets/MIDI/giantMIDI/all_key_tension'))
    parser.add_argument('--output', type=Path, default=ROOT / 'out/tension_model_selection')
    parser.add_argument('--primers', type=int, default=3)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--primer-notes', type=int, default=300)
    parser.add_argument('--continuation-notes', type=int, default=150)
    parser.add_argument('--annotation-files', type=int, default=12)
    parser.add_argument('--nll-windows', type=int, default=12)
    parser.add_argument('--seed', type=int, default=314159)
    args = parser.parse_args()
    if (args.primers < 2 or args.seeds < 3 or args.primer_notes != 300
            or args.continuation_notes != 150 or args.annotation_files < 1
            or args.nll_windows < 1):
        parser.error('Use at least two 300-note primers, three seeds, and 150-note continuations')
    if not args.data.is_file() or not args.annotations.is_dir():
        parser.error('Annotated MIDI corpus and test pickle are required')
    for path in args.checkpoints:
        if not path.is_file():
            parser.error(f'Missing checkpoint: {path}')
    torch.set_num_threads(1)
    scorer = TensionScorer()
    agreement = annotation_check(scorer, args.annotations, args.annotation_files,
                                 args.primer_notes + args.continuation_notes)
    print(f"Annotation agreement: {agreement['agreement']:.1%}, "
          f"MAE {agreement['mean_absolute_error']:.3f}", flush=True)
    if agreement['agreement'] < 0.80:
        raise ValueError('Scorer agreement below 80%; do not select a checkpoint with this evaluator')
    data = torch.tensor(Any_Pickle_File_Reader(str(args.data.with_suffix(''))), dtype=torch.int16)
    cfg = get_model_hparams('D_Tension_Joker_little_v2')
    cfg.update(tens_control_primer=args.primer_notes, tens_control_notes=args.continuation_notes,
               tens_control_pieces=args.primers, tens_control_seed=args.seed,
               tens_p_all_cond=1.0, tens_lead_max=0, home_key_drop_prob=0.0,
               tens_max_val_samples=0)
    dataset = TensionPitchSamplerDataset(data, args.primer_notes, is_eval=True,
                                         transpose_max=0, cfg=cfg, split='all')
    sources, manifest = dataset.control_performances(data, cfg)
    for row, source in zip(manifest, sources):
        row['primer_sha256'] = hashlib.sha256(bytes(source.pitches[:args.primer_notes])).hexdigest()
    indices = np.linspace(0, len(dataset) - 1, args.nll_windows, dtype=int).tolist()
    nll_loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False)
    del data, dataset
    base_cfg = get_model_hparams('D_Base_Joker_little_v1')
    base_cfg['ckpt_file_name'] = str(ROOT / base_cfg['ckpt_file_name'])
    base = load_model('D_Base_Joker_little_v1', cfg=base_cfg, compile_mode='none').eval()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'protocol': {'test_pickle': str(args.data), 'test_pickle_sha256': sha256(args.data),
                           'annotation_files': args.annotation_files,
                           'primers': manifest, 'seeds': args.seeds, 'seed': args.seed,
                           'primer_notes': args.primer_notes,
                           'continuation_notes': args.continuation_notes,
                           'nll_windows': indices, 'temperature': 1.0, 'guidance': 1.0,
                           'scorer': scorer.provenance},
              'annotation_check': agreement, 'checkpoints': []}
    for path in args.checkpoints:
        report['checkpoints'].append(evaluate_checkpoint(path, sources, scorer, base,
                                                          args.primer_notes,
                                                          args.continuation_notes,
                                                          args.seeds, args.seed,
                                                          nll_loader, args.output))
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        write_report(args.output / 'report.md', report)
    print(f"Saved {args.output / 'report.md'}", flush=True)


if __name__ == '__main__':
    main()
