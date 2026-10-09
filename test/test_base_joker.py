"""Generate five MIDI continuations by replaying source timing as joker presses.

The primer is retained in each MIDI. Subsequent source dtimes, durations and
velocities control playback only; the decoder receives generated pitch history.
The base runner uses the existing 32 ms MIDI tokenization. Tension continuations
retain source timing in milliseconds and support paired targets, control
schedules, corpus-proxy scoring, and blinded filenames. No MIDI hardware is
needed. Run --check-tension for CPU checks, or --help for generation options.
"""

import argparse
import math
import json
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from midiUtils import dict_to_song, midi_to_dict, ms_SONG_to_MIDI_Converter
from model_loader import load_model, warm_start_tension
from models import get_model_hparams
from x_transformer import D_base, D_Tension_Joker
from tension_joker import (TensionControls, TensionScorer, PianoPerformance,
                          read_tension_performance, HOME_KEY_UNKNOWN)


@torch.inference_mode()
def generate_pitches(model: D_base,
                     primer: List[int], notes: int, context_length: int,
                     temperature: float, seed: int, device: torch.device) -> List[int]:
    torch.manual_seed(seed)
    pitches = torch.zeros((1, len(primer) + notes), dtype=torch.long, device=device)
    pitches[0, :len(primer)] = torch.tensor(primer, dtype=torch.long, device=device)
    # Each press depends on earlier generated pitches, so steps are sequential.
    for target in range(len(primer), pitches.size(1)):
        start = max(0, target - context_length)
        # The existing generation API drops the final, unknown-pitch placeholder.
        context = {'pitch': pitches[:, start:target + 1]}
        pitches[0, target] = model.gen_pitch_token(context, temperature=temperature)
    return pitches[0].cpu().tolist()


@torch.inference_mode()
def generate_tension_pitches(model: D_Tension_Joker, source: PianoPerformance,
                             primer_length: int, notes: int, context_length: int,
                             schedule: Dict[int, Optional[int]], temperature: float,
                             guidance: float, seed: int, device: torch.device
                             ) -> Tuple[List[int], List[int], List[float]]:
    torch.manual_seed(seed)
    pitches = torch.zeros((1, primer_length + notes), dtype=torch.long, device=device)
    pitches[0, :primer_length] = torch.tensor(source.pitches[:primer_length], device=device)
    controls = TensionControls(primer_length, context_length)
    controls.cfg_weight = guidance
    requests: List[int] = []
    latency_ms: List[float] = []
    home = torch.tensor([source.home_key], device=device)
    # Generation is sequential because each pitch depends on previous sampled pitches.
    for step in range(notes):
        if step in schedule:
            controls.set_level(schedule[step])
        position = primer_length + step
        start = max(0, position - context_length)
        context = {'pitch': pitches[:, start:position + 1], 'home_key': home,
                   'tension_target': torch.tensor([controls.next_targets()], device=device)}
        before = time.perf_counter()
        pitches[0, position] = model.gen_pitch_token(context, temperature, guidance)
        latency_ms.append((time.perf_counter() - before) * 1000)
        requests.append(controls.target)
        controls.commit_note()
    return pitches[0].cpu().tolist(), requests, latency_ms


def parse_schedule(text: str) -> Dict[int, Optional[int]]:
    schedule: Dict[int, Optional[int]] = {}
    for entry in text.split(','):
        index, level = entry.strip().split(':', 1)
        step = int(index)
        target = None if level.strip().lower() == 'free' else int(level)
        if step < 0 or step in schedule or (target is not None and not 0 <= target <= 4):
            raise ValueError('Schedules require unique non-negative positions and levels 0..4/free')
        schedule[step] = target
    if 0 not in schedule:
        raise ValueError('A schedule must specify its initial state at note 0')
    return dict(sorted(schedule.items()))


def continuation_metrics(source: PianoPerformance, pitches: List[int], requests: List[int],
                         primer_length: int, measured: List[int]) -> Dict[str, object]:
    import numpy as np

    generated = np.asarray(pitches[primer_length:])
    realized = np.asarray(measured[primer_length:])
    targets = np.asarray(requests) - 1
    confusion = np.zeros((5, 5), dtype=int)
    active = targets >= 0
    np.add.at(confusion, (targets[active], realized[active]), 1)
    cuts = np.r_[0, np.flatnonzero(targets[1:] != targets[:-1]) + 1, len(targets)]
    transitions = []
    post_latency = np.zeros(len(targets), dtype=bool)
    for start, end in zip(cuts[:-1], cuts[1:]):
        if targets[start] < 0:
            continue
        close = np.abs(realized[start:end] - targets[start]) <= 1
        # Three successive notes avoid counting a single chance hit as settling.
        settled = np.flatnonzero(np.convolve(close.astype(int), np.ones(3, dtype=int), 'valid') == 3) if len(close) >= 3 else []
        delay = int(settled[0]) if len(settled) else None
        transitions.append({'at_note': int(start), 'target': int(targets[start]),
                            'jump_size': int(abs(targets[start] - (realized[start-1] if start else measured[primer_length-1]))),
                            'response_notes': None if delay is None else delay + 1,
                            'response_seconds': None if delay is None else (
                                source.starts_ms[primer_length + start + delay] - source.starts_ms[primer_length + start]) / 1000})
        post_latency[start + 8:end] = True
    grams = [tuple(generated[i:i + 4]) for i in range(max(0, len(generated) - 3))]
    return {'mean_realized_level': float(realized.mean()),
            'mean_first_40': float(realized[:40].mean()),
            'mean_after_40': float(realized[40:].mean()) if len(realized) > 40 else None,
            'within_one_after_8_notes': float((np.abs(realized[post_latency] - targets[post_latency]) <= 1).mean()) if post_latency.any() else None,
            'confusion_matrix': confusion.tolist(), 'transitions': transitions,
            'repeated_note_rate': float((generated[1:] == generated[:-1]).mean()) if len(generated) > 1 else 0.0,
            'repeated_4gram_rate': 1.0 - len(set(grams)) / len(grams) if grams else 0.0}


def summarize_tension(runs: List[dict]) -> Dict[str, object]:
    import numpy as np

    scored = [run for run in runs if 'metrics' in run]
    if not scored:
        return {}
    def target_summary(records: List[dict]) -> Dict[str, object]:
        means = []
        for level in range(5):
            values = [run['metrics']['mean_realized_level'] for run in records
                      if run['schedule'] == {0: level}]
            means.append(float(np.mean(values)) if values else None)
        correlation = None
        if all(value is not None for value in means) and len(set(means)) > 1:
            values = np.asarray(means)
            ranks = (values[:, None] > values).sum(1) + ((values[:, None] == values).sum(1) - 1) / 2
            correlation = float(np.corrcoef(np.arange(5), ranks)[0, 1])
        return {'mean_realized_by_target': means, 'spearman_of_target_means': correlation}

    return {**target_summary(scored),
            'per_primer': {sample: target_summary([run for run in scored if run['sample'] == sample])
                           for sample in sorted({run['sample'] for run in scored})},
            'confusion_matrix': np.sum([run['metrics']['confusion_matrix'] for run in scored], axis=0).tolist(),
            'note': 'Related corpus-scale proxy; coherence and audible steering also require listening.'}


def check_tension() -> None:
    """Focused CPU checks, including optimization, boundaries, release and checkpoint loading."""
    import mido
    from train import TensionPitchSamplerDataset, load_checkpoint

    torch.manual_seed(7)
    cfg = get_model_hparams('D_Base_Joker_little_v1')
    cfg.update(seq_len=8, emb_dim=32, num_layers=2, heads=2)
    base = load_model(cfg=cfg, set_only=True).eval()
    tension_cfg = {**cfg, 'model_type': 'D_Tension_Joker', 'tens_cond_dim': 16, 'loss_film_reg': 0.01}
    model = load_model(cfg=tension_cfg, set_only=True)
    batch = {'pitch': torch.tensor([[60, 62, 64, 65, 67, 69, 71, 72, 60]]),
             'tension_target': torch.tensor([[0, 0, 0, 1, 2, 3, 4, 5, 1]]),
             'home_key': torch.tensor([0])}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'base.pth'
        torch.save(base.state_dict(), path)
        warm_start_tension(model, str(path))
        original = base.decoder({'pitch': batch['pitch'][:, :-1]})
        model.eval()
        torch.testing.assert_close(model.pitch_logits(batch), original, rtol=0, atol=0)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
        frozen = {name: p.clone() for name, p in model.named_parameters() if not p.requires_grad}
        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            loss, _ = model(batch)
            loss['loss_total'].backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
        assert all(torch.equal(frozen[name], p) for name, p in model.named_parameters() if not p.requires_grad)
        assert model.decoder.tension_conditioner.target_emb.weight.grad.abs().sum() > 0
        model.eval()
        null = {**batch, 'tension_target': torch.zeros_like(batch['tension_target'])}
        torch.testing.assert_close(model.pitch_logits(null), original, rtol=0, atol=0)
        assert model(null)[0]['loss_film_reg'].item() == 0
        future = {**batch, 'tension_target': batch['tension_target'].clone()}
        future['tension_target'][:, 6:] = 5
        torch.testing.assert_close(model.pitch_logits(batch)[:, :5], model.pitch_logits(future)[:, :5])
        placeholder = {**batch, 'pitch': batch['pitch'].clone()}
        placeholder['pitch'][:, -1] = 1
        torch.testing.assert_close(model.pitch_logits(batch), model.pitch_logits(placeholder))
        expected = original[:, -1] + 1.5 * (model.pitch_logits(batch)[:, -1] - original[:, -1])
        torch.testing.assert_close(model.next_pitch_logits(batch, 1.5), expected, rtol=1e-5, atol=1e-6)

        trained = Path(directory) / 'tension.pth'
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': 2, 'steps': 3}, trained)
        restored = load_model(cfg={**tension_cfg, 'ckpt_file_name': str(trained)}, compile_mode='none')
        restored_optimizer = torch.optim.Adam([p for p in restored.parameters() if p.requires_grad])
        assert load_checkpoint(restored, restored_optimizer, str(trained), torch.device('cpu')) == (2, 3)
        torch.testing.assert_close(restored.eval().pitch_logits(batch), model.pitch_logits(batch))
        incomplete = model.state_dict()
        del incomplete['decoder.harm_film.proj.weight']
        torch.save(incomplete, trained)
        try:
            load_model(cfg={**tension_cfg, 'ckpt_file_name': str(trained)}, compile_mode='none')
        except RuntimeError:
            pass
        else:
            raise AssertionError('A trained tension checkpoint must load strictly')
        try:
            warm_start_tension(model, str(trained))
        except ValueError:
            pass
        else:
            raise AssertionError('Warm-start accepted an adapter checkpoint as a base')

        midi = mido.MidiFile(ticks_per_beat=1000)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.extend([mido.MetaMessage('set_tempo', tempo=1_000_000),
                      mido.MetaMessage('marker', text='home: F# minor'),
                      mido.Message('note_on', channel=0, note=60, velocity=80),
                      mido.Message('note_on', channel=3, note=90, velocity=80),
                      mido.Message('note_on', channel=10, note=72, velocity=70),
                      mido.Message('note_on', channel=1, note=64, velocity=90, time=100),
                      mido.Message('note_off', channel=0, note=60, time=100),
                      mido.Message('note_off', channel=10, note=72),
                      mido.Message('note_off', channel=1, note=64, time=300)])
        source_path = Path(directory) / 'source.mid'
        midi.save(source_path)
        source = read_tension_performance(str(source_path), 'C major')
        assert source.pitches == [72, 60, 64] and source.home_key == 18
        assert source.starts_ms == [0, 0, 100] and source.durations_ms == [200, 200, 400]
        source.write(str(Path(directory) / 'roundtrip.mid'), [73, 61, 65])
        roundtrip = read_tension_performance(str(Path(directory) / 'roundtrip.mid'))
        assert roundtrip.starts_ms == source.starts_ms and roundtrip.durations_ms == source.durations_ms
        assert roundtrip.home_key == HOME_KEY_UNKNOWN
        assert read_tension_performance(str(Path(directory) / 'roundtrip.mid'), 'Eb major').home_key == 3

    rows = []
    for piece in range(40):
        rows.extend(([126, 126, 0, 0, 0], [0, piece % 12, piece % 2, 0, 124]))
        for note in range(24):
            if note == 2:
                rows.append([0, piece % 5, 0, 0, 123])
            if note == 15:
                rows.append([0, (piece + 1) % 5, 0, 0, 123])
            rows.append([1, 2, 30 + piece + note % 12, 90, (0, 1, 10)[note % 3]])
    data = torch.tensor(rows).flatten()
    dataset = TensionPitchSamplerDataset(data, 8, cfg={'home_key_drop_prob': 0.0,
                                                     'tens_validation_fraction': 0.3}, split='all')
    assert (dataset.levels.reshape(40, 24)[:, :2] == 0).all()
    assert torch.equal(dataset.home_keys[1:], torch.arange(40) % 12 + 12 * (torch.arange(40) % 2))
    for seed in range(20):
        torch.manual_seed(seed)
        piece, start, pitches, shift = dataset._sample_window(0)
        torch.manual_seed(seed)
        item = dataset[0]
        torch.testing.assert_close(item['pitch'], pitches)
        home = int(dataset.home_keys[piece])
        assert int(item['home_key']) == (home % 12 + shift) % 12 + 12 * (home // 12)
        active = item['tension_target'] > 0
        torch.testing.assert_close(item['tension_target'][active], dataset.levels[start:start + 9][active])
        assert active[1:].any()
    dataset.select_split('train')
    validation = dataset.validation_dataset()
    assert not set(dataset.piece_ids.tolist()) & set(validation.piece_ids.tolist())
    assert all(torch.equal(validation[0][key], validation[0][key]) for key in validation[0])
    dataset.home_drop = 1.0
    assert dataset[0]['home_key'] == HOME_KEY_UNKNOWN
    controls = TensionControls(3, 4)
    controls.set_level(4)
    assert controls.next_targets() == [0, 0, 0, 5]
    for _ in range(6):
        controls.commit_note()
    controls.set_level(0)
    assert controls.next_targets() == [5, 5, 5, 5, 1]
    controls.handle_key(' ')
    assert controls.next_targets() == [0, 0, 0, 0, 0]
    controls.set_level(2)
    assert controls.next_targets()[-1] == 3
    controls.reset(3)
    assert controls.next_targets() == [0, 0, 0, 0]
    print('Tension checks passed: alignment, transposition, split, causality, initialization, '
          'frozen weights, gradients, release, guidance, MIDI filtering and checkpoint resume.')


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='D_Base_Joker_little_v1')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--sample', type=Path, nargs='+',
                        default=[ROOT / 'samples/Bach_Prelude_and_Fugue_in_C_major.mid'])
    parser.add_argument('--output', type=Path, default=ROOT / 'out')
    parser.add_argument('--primer-length', type=int, default=300)
    parser.add_argument('--context-length', type=int)
    parser.add_argument('--notes', type=int, help='Maximum generated notes; default: 150 for tension, all remaining for base')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1, help='First consecutive random seed')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--targets', default='free,0,1,2,3,4', help='Tension model only')
    parser.add_argument('--schedule', action='append', default=[], help='Additional control sequence, e.g. 0:0,40:4,80:free,100:2')
    parser.add_argument('--home-key', help='Fallback when a MIDI has no home marker, e.g. C major')
    parser.add_argument('--guidance', type=float, default=1.0)
    parser.add_argument('--score-tension', action='store_true')
    parser.add_argument('--analyzer-root', type=Path, default=ROOT.parent / 'PROCESS/harmony_detector')
    parser.add_argument('--calibration', type=Path, help='Frozen corpus calibration JSON; otherwise use builder defaults')
    parser.add_argument('--blind', action='store_true', help='Randomized MIDI names; identities remain in report.json')
    parser.add_argument('--check-tension', action='store_true', help='Run focused CPU checks without a trained checkpoint')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    args = parser.parse_args(argv)
    if args.check_tension:
        check_tension()
        return
    cfg = get_model_hparams(args.model)
    if cfg.get('model_type') not in ('D_Base_Joker', 'D_Tension_Joker'):
        parser.error('Choose D_Base_Joker_little_v1 or a D_Tension_Joker configuration')
    tension = cfg['model_type'] == 'D_Tension_Joker'
    args.checkpoint = args.checkpoint or ROOT / cfg['ckpt_file_name']
    if not args.checkpoint.is_file():
        parser.error(f'No trained checkpoint at {args.checkpoint}; train first or provide --checkpoint')
    saved = torch.load(args.checkpoint, map_location='cpu')
    if 'cfg' in saved:
        if saved['cfg']['model_type'] != cfg['model_type']:
            parser.error('Checkpoint model type does not match --model')
        cfg.update(saved['cfg'])
    del saved
    args.context_length = args.context_length or cfg['seq_len']
    if not 1 <= args.primer_length <= args.context_length <= cfg['seq_len']:
        parser.error('Require 1 <= primer-length <= context-length <= model seq_len')
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('Temperature must be finite and positive')
    if args.notes is not None and args.notes < 1:
        parser.error('Notes must be positive')
    if args.seeds < 1 or not math.isfinite(args.guidance) or args.guidance < 0:
        parser.error('Require positive seeds and finite non-negative guidance')
    if len({sample.stem for sample in args.sample}) != len(args.sample):
        parser.error('Sample filenames must have distinct stems to avoid overwriting continuations')
    if (args.score_tension or args.schedule) and not tension:
        parser.error('Tension scoring and schedules require a D_Tension_Joker model')
    device_name = args.device
    if device_name == 'auto':
        device_name = ('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(device_name)
    cfg['ckpt_file_name'] = str(args.checkpoint)
    model = load_model(args.model, cfg=cfg, compile_mode='none').to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    scorer = TensionScorer(str(args.analyzer_root), str(args.calibration) if args.calibration else None) if args.score_tension else None
    try:
        schedules = ([parse_schedule('0:' + level) for level in args.targets.split(',')]
                     + [parse_schedule(text) for text in args.schedule]) if tension else [{0: None}]
    except ValueError as error:
        parser.error(str(error))
    import random
    order = list(range(1, len(args.sample) * args.seeds * len(schedules) + 1))
    random.Random(args.seed).shuffle(order)
    runs = []
    for sample_index, sample in enumerate(args.sample):
        if not sample.is_file():
            parser.error(f'Source MIDI not found: {sample}')
        performance = read_tension_performance(str(sample), args.home_key) if tension else None
        source, count = (performance.tokens(), len(performance.pitches)) if tension else midi_to_dict(str(sample))
        remaining = count - args.primer_length
        if remaining < 1:
            parser.error(f'The source MIDI must contain notes after the primer: {sample}')
        notes = min(args.notes or (150 if tension else remaining), remaining)
        if any(max(schedule) >= notes for schedule in schedules):
            parser.error(f'A schedule extends beyond the {notes} available generated notes in {sample}')
        if scorer and performance.home_key == HOME_KEY_UNKNOWN:
            parser.error(f'Scoring requires a home marker or --home-key: {sample}')
        for schedule_index, schedule in enumerate(schedules):
            for run in range(args.seeds):
                seed = args.seed + run
                print(f'{sample.name}: schedule {schedule_index + 1}/{len(schedules)}, seed {seed}', flush=True)
                if tension:
                    pitches, requests, latencies = generate_tension_pitches(
                        model, performance, args.primer_length, notes, args.context_length,
                        schedule, args.temperature, args.guidance, seed, device)
                else:
                    pitches = generate_pitches(model, source['pitch'][:args.primer_length], notes,
                                               args.context_length, args.temperature, seed, device)
                    requests, latencies = [], []
                name = (f'listening_{order[len(runs)]:04d}' if args.blind else
                        f'{sample.stem}_tension_{schedule_index + 1}_seed_{seed}' if tension else
                        f'testbase_joker_{sample_index + 1}_{run + 1}' if len(args.sample) > 1 else f'testbase_joker_{run + 1}')
                path = args.output / (name + '.mid')
                if tension:
                    performance.write(str(path), pitches)
                else:
                    output = {key: values[:len(pitches)] for key, values in source.items()}
                    output['pitch'] = pitches
                    ms_SONG_to_MIDI_Converter(dict_to_song(output, force_vel=False), output_file_name=str(path),
                                              timings_multiplier=1, add_extension=False)
                record = {'sample': str(sample), 'output': str(path), 'seed': seed, 'schedule': schedule,
                          'temperature': args.temperature, 'primer_length': args.primer_length,
                          'context_length': args.context_length, 'guidance': args.guidance,
                          'requested_tokens': requests, 'latency_ms': latencies}
                if scorer:
                    measured = scorer.score(performance, pitches)
                    record['home_key'] = performance.home_key
                    record['realized_levels'] = measured['levels'][args.primer_length:]
                    record['composite_scores'] = measured['composite'][args.primer_length:]
                    record['frame_indicators'] = measured['frame_indicators']
                    record['metrics'] = continuation_metrics(performance, pitches, requests,
                                                              args.primer_length, measured['levels'])
                runs.append(record)
                report = {'checkpoint': str(args.checkpoint), 'model': args.model,
                          'base_training_overlap': cfg.get('base_training_overlap', 'unknown'),
                          'evaluator': scorer.provenance if scorer else None,
                          'runs': runs, 'summary': summarize_tension(runs)}
                (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
                print(f'Saved {path}', flush=True)
    if tension:
        print(json.dumps(summarize_tension(runs), indent=2))


if __name__ == '__main__':
    main()
