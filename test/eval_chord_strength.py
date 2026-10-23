"""Compare the conditioning strength of D_Chord_Joker checkpoints on unseen MIDI files.

A D_Base_Joker checkpoint (no conditioning: it ignores the chord) can be compared the same
way, e.g. --checkpoint <base joker> <chord joker>: whatever the base gets for a chord comes
from the context alone, so the difference measures what FiLM adds.

Protocol (paired: every checkpoint and condition shares the same history and sampling noise):
  1. Each test MIDI is read exactly as midis2pickles.py + ChordPitchSamplerDataset read it.
  2. At chord changes of the piece, the real performance up to the change is the primer
     (--context notes), and the model generates --notes notes with the chord held from
     the first generated note on. Requested chords:
       real        - the channel-5 chord of the piece at that change
       dominant    - major triad on the fifth above the real chord's root (its V)
       subdominant - major triad on the fourth above the real chord's root (its IV)
       distant     - major triad a tritone above the real root (no common tone with a triad)
     plus a free run (no chord) as the baseline.
  3. Chord-tone mass (CTM@N): mean over the first N generated notes of the probability the
     model gives to the requested chord's pitch classes (any octave). Free CTM@N is the same
     quantity for the free run measured on the same chord.
     Conditioning strength = CTM_cond@N - CTM_free@N, compared between checkpoints per
     change with a paired bootstrap CI and a paired sign-flip permutation test.
  4. Per generated note k (summary 'per_note'): the same probability for note k alone, to see
     how many notes the model needs to reach the requested harmony.

Generation results are saved per MIDI file in --output, so several processes can share the
work (one file each); --summarize-only then writes summary.json from all saved files.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from filter_unseen_pickles import pitch_ngrams
from midiUtils import Any_Pickle_File_Reader, midi2ms_score
from model_loader import load_model
from params import ACCOMP_CHANNEL, CHORDS_CHANNEL, EXTRA_ACCOMP_CHANNEL, MELODY_CHANNEL, MELODY_CHORDS_CHANNEL
from x_transformer import D_Base, D_Chord_Joker

NOTE_CHANNELS = (MELODY_CHANNEL, MELODY_CHORDS_CHANNEL, ACCOMP_CHANNEL, EXTRA_ACCOMP_CHANNEL)
CONDITIONS = ('real', 'dominant', 'subdominant', 'distant')
ADAPTER = ('decoder.chord_conditioner.', 'decoder.harm_film.', 'decoder.history_mask_embedding')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_piece(path: Path) -> Dict[str, np.ndarray]:
    """Performed pitches and the chord of each note, as midis2pickles.py + the sampler see them.

    Same channels (MIDI 1, 2, 11, 12 notes; MIDI 5 chord tones), same order (time, then pitch
    descending), same 10 ms onsets and chord expiry (onset + duration of its longest tone).
    """
    score = midi2ms_score(path.read_bytes())
    events = [list(event) for track in score[1:] for event in track
              if event[0] == 'note' and event[3] in NOTE_CHANNELS + (CHORDS_CHANNEL,)]
    events.sort(key=lambda event: event[4], reverse=True)
    events.sort(key=lambda event: event[1])
    pitch, group, chords = [], [], []  # group: index in chords of each note's chord, -1 = none
    current, chord_time, chord_end = -1, None, -1
    for _, start, duration, channel, note, _ in events:
        time, note = int(start / 10), max(1, min(126, note))
        if channel == CHORDS_CHANNEL:
            if time != chord_time:  # first tone of a new chord
                chord_time, current = time, len(chords)
                chords.append([])
            chords[current].append(note)
            chord_end = max(chord_end, time + 2 * int(duration / 20))
            continue
        if chord_end >= 0 and time >= chord_end:  # the chord has expired: unlabelled notes
            current, chord_end = -1, -1
        pitch.append(note)
        group.append(current)
    # Like the sampler, a note gets ALL tones of its chord, also those listed after it.
    chroma = np.zeros((len(pitch), 12), dtype=bool)
    anchor = np.zeros(len(pitch), dtype=np.int64)
    for index, chord in enumerate(group):
        if chord >= 0:
            chroma[index, np.asarray(chords[chord]) % 12] = True
            anchor[index] = min(chords[chord]) % 12
    return {'pitch': np.asarray(pitch), 'chroma': chroma, 'anchor': anchor}


def chord_changes(piece: Dict[str, np.ndarray], context: int, notes: int) -> np.ndarray:
    """Notes whose labelled chord differs from the previous note's chord (training definition)."""
    chroma = piece['chroma']
    change = chroma.any(axis=1) & np.r_[True, (chroma[1:] != chroma[:-1]).any(axis=1)]
    index = np.arange(len(chroma))
    return index[change & (index >= context) & (index + notes <= len(chroma))]


def requested_chords(piece: Dict[str, np.ndarray], onset: int) -> Dict[str, dict]:
    root = int(piece['anchor'][onset])
    triad = lambda r: {'pcs': sorted({r % 12, (r + 4) % 12, (r + 7) % 12}), 'anchor': r % 12}
    return {'real': {'pcs': np.flatnonzero(piece['chroma'][onset]).tolist(), 'anchor': root},
            'dominant': triad(root + 7), 'subdominant': triad(root + 5), 'distant': triad(root + 6)}


def load_checkpoint(path: Path, device: torch.device) -> D_Base:
    saved = torch.load(path, map_location='cpu', weights_only=False)
    cfg = saved['cfg']
    require(cfg.get('model_type') in ('D_Chord_Joker', 'D_Base_Joker'),
            f'Expected a D_Chord_Joker or D_Base_Joker checkpoint: {path}')
    model = load_model(cfg['model_name'], cfg=cfg, set_only=True)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def generate(model: D_Base, piece: Dict[str, np.ndarray], onset: int, chords: Dict[str, dict],
             noise: torch.Tensor, context: int, temperature: float, device: torch.device) -> Dict[str, list]:
    """Rows = seeds x (free + CONDITIONS); all rows share the real primer."""
    seeds, notes = noise.shape[:2]
    rows = [(seed, name) for seed in range(seeds) for name in ('free',) + CONDITIONS]
    length = context + notes
    pitch = torch.zeros(len(rows), length, dtype=torch.long, device=device)
    pitch[:, :context] = torch.as_tensor(piece['pitch'][onset - context:onset], device=device)
    chroma = torch.zeros(len(rows), length, 12, dtype=torch.bool, device=device)
    anchor = torch.zeros(len(rows), length, dtype=torch.long, device=device)
    active = torch.zeros(len(rows), length, dtype=torch.bool, device=device)
    for row, (_, name) in enumerate(rows):
        if name != 'free':  # chord held from the first generated note on
            chroma[row, context:, chords[name]['pcs']] = True
            anchor[row, context:] = chords[name]['anchor']
            active[row, context:] = True
    to_pc = F.one_hot(torch.arange(128, device=device) % 12, 12).float()
    seed_of_row = torch.tensor([seed for seed, _ in rows], device=device)
    noise = noise.to(device)
    pc_mass = []
    for step in range(notes):
        position = context + step
        window = slice(position - context, position + 1)  # same sliding context as performance
        if isinstance(model, D_Chord_Joker):
            logits = model.next_pitch_logits({'pitch': pitch[:, window], 'chord_chroma': chroma[:, window],
                                              'chord_anchor_pc': anchor[:, window], 'chord_active': active[:, window]})
        else:  # D_Base_Joker has no chord input: every row sees the context only
            logits = model.decoder({'pitch': pitch[:, window][:, :-1]})[:, -1]
        pc_mass.append(F.softmax(logits / temperature, dim=-1) @ to_pc)
        # Gumbel-max with noise shared by every condition and checkpoint (common random numbers).
        pitch[:, position] = (logits / temperature - noise[seed_of_row, step].log()).argmax(-1)
    # Coherence: log-likelihood of each generated note under the frozen base (no chord),
    # given the primer and the notes generated before it (one causal pass).
    base = F.log_softmax(model.decoder({'pitch': pitch[:, :-1]}).float(), dim=-1)  # decoder alone = no chord
    base_logp = base[:, context - 1:].gather(-1, pitch[:, context:].unsqueeze(-1)).squeeze(-1)
    return {'rows': rows, 'pitches': pitch[:, context:].tolist(),
            'pc_mass': torch.stack(pc_mass, dim=1).round(decimals=6).tolist(),
            'base_logp': base_logp.round(decimals=5).tolist()}


def run_file(path: Path, file_index: int, models: Dict[str, D_Chord_Joker], args: argparse.Namespace,
             device: torch.device) -> dict:
    piece = read_piece(path)
    changes = chord_changes(piece, args.context, args.notes)
    require(len(changes) >= args.onsets, f'{path.name}: only {len(changes)} usable chord changes')
    onsets = changes[np.linspace(0, len(changes) - 1, args.onsets).round().astype(int)]  # spread over the piece
    record = {'midi': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'performed_notes': len(piece['pitch']), 'usable_changes': len(changes), 'onsets': []}
    for onset_index, onset in enumerate(onsets.tolist()):
        chords = requested_chords(piece, onset)
        rng = torch.Generator().manual_seed(args.seed + 1000 * file_index + onset_index)
        noise = torch.empty(args.seeds, args.notes, 128).exponential_(generator=rng)
        entry = {'onset': onset, 'chords': chords, 'last_primer_pitch': int(piece['pitch'][onset - 1]),
                 'real_continuation': piece['pitch'][onset:onset + args.notes].tolist(), 'models': {}}
        for name, model in models.items():
            entry['models'][name] = generate(model, piece, onset, chords, noise, args.context, args.temperature, device)
        record['onsets'].append(entry)
        print(f'{path.name}: change {onset_index + 1}/{args.onsets} done', flush=True)
    return record


#==========================================================================

def per_change_metrics(entry: dict, model: str, condition: str, horizon: int) -> Dict[str, float]:
    """Averages over seeds for one chord change, checkpoint, condition and horizon N."""
    run = entry['models'][model]
    pcs = entry['chords'][condition]['pcs']
    values = {key: [] for key in ('ctm_cond', 'ctm_free', 'ctr_cond', 'ctr_free', 'coverage_cond',
                                  'coverage_free', 'nll_cond', 'nll_free', 'leaps_cond', 'leaps_free')}
    for seed in {seed for seed, _ in run['rows']}:
        for kind, name in (('cond', condition), ('free', 'free')):
            row = run['rows'].index([seed, name])
            pitches = np.asarray(run['pitches'][row][:horizon])
            values['ctm_' + kind].append(np.asarray(run['pc_mass'][row])[:horizon][:, pcs].sum(-1).mean())
            values['ctr_' + kind].append(np.isin(pitches % 12, pcs).mean())
            values['coverage_' + kind].append(len(set((pitches % 12).tolist()) & set(pcs)) / len(pcs))
            values['nll_' + kind].append(-np.mean(run['base_logp'][row][:horizon]))
            jumps = np.abs(np.diff(np.r_[entry['last_primer_pitch'], pitches]))
            values['leaps_' + kind].append((jumps > 12).mean())
    result = {key: float(np.mean(value)) for key, value in values.items()}
    result['strength'] = result['ctm_cond'] - result['ctm_free']
    return result


def per_note_mass(entry: dict, model: str, condition: str) -> np.ndarray:
    """[notes, 2]: probability that generated note k is a tone of the requested chord,
    conditioned and free (same chord), averaged over seeds."""
    run = entry['models'][model]
    pcs = entry['chords'][condition]['pcs']
    seeds = sorted({seed for seed, _ in run['rows']})
    return np.mean([np.stack([np.asarray(run['pc_mass'][run['rows'].index([seed, name])])[:, pcs].sum(-1)
                              for name in (condition, 'free')], axis=-1) for seed in seeds], axis=0)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, resamples: int = 10000) -> List[float]:
    means = values[rng.integers(len(values), size=(resamples, len(values)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def permutation_p(differences: np.ndarray, rng: np.random.Generator, resamples: int = 20000) -> float:
    """Two-sided paired sign-flip test of mean difference = 0."""
    signs = rng.choice((-1.0, 1.0), size=(resamples, len(differences)))
    null = np.abs((signs * differences).mean(axis=1))
    return float((1 + (null >= abs(differences.mean())).sum()) / (resamples + 1))


@torch.inference_mode()
def pianist_reference(entries: list, args: argparse.Namespace) -> dict:
    """The pianist's real next notes at each tested change, scored like the generated ones."""
    model = load_checkpoint(args.checkpoint[0], torch.device('cpu'))  # no chord = the frozen base
    pieces, rates, nll = {}, {h: [] for h in args.horizons}, {h: [] for h in args.horizons}
    for midi, entry in entries:
        piece = pieces.setdefault(midi, read_piece(args.midi_dir / midi))
        onset = entry['onset']
        pitch = torch.as_tensor(piece['pitch'][onset - args.context:onset + args.notes]).unsqueeze(0)
        real = pitch[0, args.context:]
        logp = F.log_softmax(model.decoder({'pitch': pitch[:, :-1]}).float(), dim=-1)[0, args.context - 1:]
        logp = logp.gather(-1, real.unsqueeze(-1)).squeeze(-1).numpy()
        for h in args.horizons:
            rates[h].append(np.isin(real[:h].numpy() % 12, entry['chords']['real']['pcs']).mean())
            nll[h].append(-logp[:h].mean())
    return {f'@{h}': {'real_chord_tone_rate': float(np.mean(rates[h])), 'base_nll': float(np.mean(nll[h]))}
            for h in args.horizons}


def unseen_check(midis: List[Path], checkpoints: List[Path], base_train: Path) -> dict:
    """Share of each test file's 8-note pitch sequences found in each training/validation pickle."""
    pickles = {'base_train': base_train}
    for path in checkpoints:
        cfg = torch.load(path, map_location='cpu', weights_only=False)['cfg']
        pickles[path.stem + ':train'] = ROOT / cfg['dataset_train_path']
        pickles[path.stem + ':validation'] = ROOT / cfg['dataset_val_path']
    sequences = {}
    for midi in midis:
        pitch = read_piece(midi)['pitch']
        events = np.zeros((len(pitch), 5), dtype=np.int16)  # notes only, one piece
        events[:, 2], events[:, 3] = pitch, 1
        sequences[midi.name] = pitch_ngrams(events)[0]
    report = {}
    for name, path in pickles.items():
        data = np.asarray(Any_Pickle_File_Reader(str(path), verbose=False), dtype=np.int16).reshape(-1, 5)
        known = np.unique(pitch_ngrams(data)[0])
        report[name] = {'path': str(path), **{midi: float(np.isin(hashes, known).mean())
                                              for midi, hashes in sequences.items()}}
    return report


def summarize(output: Path, args: argparse.Namespace, reference=pianist_reference) -> dict:
    """reference(entries, args) scores the pianist's real notes; test/eval_style_chord_strength.py
    passes its own, since its frozen base also needs buttons and style."""
    records = [json.loads(path.read_text()) for path in sorted(output.glob('*.json')) if path.name != 'summary.json']
    require(bool(records), f'No generation results in {output}')
    models = list(records[0]['onsets'][0]['models'])
    entries = [(record['midi'], entry) for record in records for entry in record['onsets']]
    # Both checkpoints share the frozen base, so their free runs must be identical.
    free_identical = all(run['pitches'][run['rows'].index([seed, 'free'])] ==
                         entry['models'][models[0]]['pitches'][run['rows'].index([seed, 'free'])]
                         for _, entry in entries for run in entry['models'].values()
                         for seed, name in run['rows'] if name == 'free')
    rng = np.random.default_rng(args.seed)
    protocol = {key: [str(item) for item in value] if key == 'checkpoint' else
                str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    summary = {'protocol': protocol,
               'files': {record['midi']: {'performed_notes': record['performed_notes'],
                                          'usable_changes': record['usable_changes'],
                                          'changes_tested': len(record['onsets'])} for record in records},
               'free_runs_identical_across_checkpoints': free_identical, 'results': {}}
    # Results saved before a chord was added to CONDITIONS are summarized with their own chords.
    conditions = [condition for condition in CONDITIONS if condition in records[0]['onsets'][0]['chords']]
    for condition in conditions:
        for horizon in args.horizons:
            metrics = {model: [per_change_metrics(entry, model, condition, horizon) for _, entry in entries]
                       for model in models}
            block = {}
            for model in models:
                table = {key: np.asarray([row[key] for row in metrics[model]]) for key in metrics[model][0]}
                block[model] = {key: {'mean': float(value.mean()), 'ci95': bootstrap_ci(value, rng)}
                                for key, value in table.items()}
                # Share of the probability that was NOT on chord tones that the chord moved onto them.
                free, cond = table['ctm_free'].mean(), table['ctm_cond'].mean()
                block[model]['normalized_strength'] = float((cond - free) / (1 - free))
            if len(models) == 2:
                old, new = models
                difference = (np.asarray([row['strength'] for row in metrics[new]])
                              - np.asarray([row['strength'] for row in metrics[old]]))
                by_file = {midi: float(difference[[name == midi for name, _ in entries]].mean())
                           for midi in summary['files']}
                block['comparison'] = {'new_minus_old_strength': float(difference.mean()),
                                       'ci95': bootstrap_ci(difference, rng),
                                       'p_permutation': permutation_p(difference, rng),
                                       'new_stronger_fraction': float((difference > 0).mean()),
                                       'by_file': by_file, 'changes': len(difference)}
            summary['results'][f'{condition}@{horizon}'] = block
    # Per generated note k: how many notes the model needs to reach the requested harmony.
    summary['per_note'] = {}
    for condition in conditions:
        curves = {model: np.asarray([per_note_mass(entry, model, condition) for _, entry in entries])
                  for model in models}  # [changes, notes, cond/free]
        block = {'free': curves[models[0]][:, :, 1].mean(axis=0).tolist()}
        for model in models:
            strength = curves[model][:, :, 0] - curves[model][:, :, 1]
            mean = strength.mean(axis=0)
            plateau = mean[-4:].mean()  # strength over the last 4 generated notes
            block[model] = {'cond': curves[model][:, :, 0].mean(axis=0).tolist(), 'strength': mean.tolist(),
                            'strength_ci95': [bootstrap_ci(strength[:, k], rng) for k in range(strength.shape[1])],
                            # Undefined (None) without any final strength, e.g. for the base joker.
                            'notes_to_half_plateau': int(np.argmax(mean >= 0.5 * plateau)) + 1 if plateau > 0 else None}
        if condition == 'real':  # the pianist's real notes: tone of the real chord or not
            block['pianist'] = np.mean([np.isin(np.asarray(entry['real_continuation']) % 12,
                                                entry['chords']['real']['pcs']) for _, entry in entries], axis=0).tolist()
        if len(models) == 2:
            old, new = models
            difference = ((curves[new][:, :, 0] - curves[new][:, :, 1])
                          - (curves[old][:, :, 0] - curves[old][:, :, 1]))
            block['comparison'] = [{'new_minus_old_strength': float(difference[:, k].mean()),
                                    'ci95': bootstrap_ci(difference[:, k], rng),
                                    'p_permutation': permutation_p(difference[:, k], rng)}
                                   for k in range(difference.shape[1])]
        summary['per_note'][condition] = block
    # Reference: how often the pianist's real next notes are tones of the real chord, and how
    # likely the frozen base finds them (generated notes are scored the same way).
    summary['pianist_reference'] = reference(entries, args)
    if not args.skip_unseen_check:
        summary['training_overlap'] = unseen_check([args.midi_dir / name for name in summary['files']],
                                               args.checkpoint, args.base_train)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def print_summary(summary: dict) -> None:
    models = [key for key in next(iter(summary['results'].values())) if key != 'comparison']
    print('=' * 100)
    print('Strength = CTM_cond - CTM_free (chord-tone probability mass, mean over the first N notes)')
    for key, block in summary['results'].items():
        line = f'{key:<12}'
        for model in models:
            line += (f' | {model[:28]:<28} free {block[model]["ctm_free"]["mean"]:.3f} cond {block[model]["ctm_cond"]["mean"]:.3f}'
                     f' strength {block[model]["strength"]["mean"]:+.3f}')
        print(line)
        if 'comparison' in block:
            comparison = block['comparison']
            # 'new' = second checkpoint, 'old' = first (e.g. chord joker vs base joker).
            print(f'{"":<12} | second - first {comparison["new_minus_old_strength"]:+.3f} '
                  f'[{comparison["ci95"][0]:+.3f}, {comparison["ci95"][1]:+.3f}]  p = {comparison["p_permutation"]:.1e}'
                  f'  second stronger at {100 * comparison["new_stronger_fraction"]:.0f} % of {comparison["changes"]} changes')
    print('Pianist real continuation (chord-tone rate of the real chord, frozen-base NLL):', summary['pianist_reference'])
    print('-' * 100)
    print('Per generated note k: probability that note k is a tone of the requested chord')
    for condition, block in summary['per_note'].items():
        print(f'{condition}: notes to half of the final strength: '
              + ', '.join(f'{model[:28]} {block[model]["notes_to_half_plateau"]}' for model in models))
        print(f'{"k":>3} {"free":>6} ' + ' '.join(f'{model[:20]:>20}' for model in models)
              + ('  new-old strength [95% CI]' if 'comparison' in block else '') + ('  pianist' if 'pianist' in block else ''))
        for k in range(len(block['free'])):
            line = f'{k + 1:>3} {block["free"][k]:6.3f} ' + ' '.join(
                f'{block[model]["cond"][k]:8.3f} ({block[model]["strength"][k]:+.3f})' for model in models)
            if 'comparison' in block:
                comparison = block['comparison'][k]
                line += (f'  {comparison["new_minus_old_strength"]:+.3f} '
                         f'[{comparison["ci95"][0]:+.3f}, {comparison["ci95"][1]:+.3f}]')
            if 'pianist' in block:
                line += f'  {block["pianist"][k]:.2f}'
            print(line)
    print('Free runs identical across checkpoints:', summary['free_runs_identical_across_checkpoints'])
    print('=' * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, nargs='+', default=[
        ROOT / 'save_models/D_Chord_Joker_mixed_history_latest_best.pth',
        ROOT / 'save_models/D_Chord_Joker_unseen_onset_latest_best.pth'], help='Old first, new second')
    parser.add_argument('--midi-dir', type=Path, default=ROOT / 'samples/not_seen')
    parser.add_argument('--midi', nargs='*', help='Only these file names from --midi-dir (default: all)')
    parser.add_argument('--output', type=Path, default=ROOT / 'out/chord_strength')
    parser.add_argument('--base-train', type=Path, default=ROOT / 'Training-Data/giantmidi_full_train.pickle',
                        help='Pickle the frozen base joker was trained on (for the unseen check)')
    parser.add_argument('--onsets', type=int, default=20, help='Chord changes per file, spread over the piece')
    parser.add_argument('--seeds', type=int, default=2, help='Sampling seeds per chord change')
    parser.add_argument('--notes', type=int, default=16, help='Generated notes per run')
    parser.add_argument('--horizons', type=int, nargs='+', default=[4, 10], help='N for CTM@N')
    parser.add_argument('--context', type=int, default=300, help='Primer / sliding context, as in performance')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--skip-unseen-check', action='store_true')
    args = parser.parse_args()
    require(max(args.horizons) <= args.notes and args.context + args.notes <= 512, 'Invalid horizons or context')
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        device = torch.device(args.device)
        models = {path.stem: load_checkpoint(path, device) for path in args.checkpoint}
        frozen = [{key: value for key, value in model.state_dict().items() if not key.startswith(ADAPTER)}
                  for model in models.values()]
        require(all(state.keys() == frozen[0].keys() and all(torch.equal(value, frozen[0][key]) for key, value in state.items())
                    for state in frozen[1:]), 'Checkpoints do not share the same frozen base')
        files = sorted(path for path in args.midi_dir.iterdir() if path.suffix.lower() in ('.mid', '.midi'))
        # The index in the folder seeds the noise, so one-file runs reproduce a full run.
        for file_index, path in enumerate(files):
            if not args.midi or path.name in args.midi:
                record = run_file(path, file_index, models, args, device)
                (args.output / (path.stem + '.json')).write_text(json.dumps(record))
    print_summary(summarize(args.output, args))


if __name__ == '__main__':
    main()
