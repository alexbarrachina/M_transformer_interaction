"""Verify the style joker's chord controls and export paired listening comparisons:
test/eval_chord_joker.py for AE_style_jokerParam_chords / AE_style_jokerParam_change checkpoints.

Use --check-only for these checks without generation:
  - keyboard/MIDI chord controls (eval_chord_joker.check_controls);
  - training style windows: next to the unchanged press window, in the same piece, with the same
    transposition and at least half full (StyleChordSamplerDataset on eval_chord_joker's fixture);
  - decoder inputs: chord controls, buttons and joker positions describe the predicted note, Space
    forgets the chord held before it, a released row is the free style joker, and gen_pitch_token
    samples next_pitch_logits.
A normal run also checks each checkpoint against the original style joker (equal frozen weights,
and bit-identical logits for every row without a held control at the probed generation steps),
then exports MIDI, audio (when FluidSynth is installed), report.json and eval_chord_joker's blind
listening page.

Every schedule of eval_chord_joker (free, hold/release of each chord key, switch, overlap restore,
plus Space schedules for a change checkpoint) is played in two button modes: guided by the automatic
player (the button the frozen encoder gives each real note of the sample's continuation) and with
the joker held. The style is the sample's first style_seq_len notes, as interaction_buttons_style_chords.py
encodes its style prompts (so it includes the primer; the samples are too short for a separate window).
"""

import argparse
import gc
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_chord_joker import (check_controls, fixture, quality_metrics, render_audio, schedules, sha256,
                              write_midi, write_report)
from eval_style_chord_strength import MODES, free_logits, mode_buttons, require, style_inputs
from harmony.chord_joker import Chord, ChordControls
from interaction_1_button_chords import FIXED_CHORDS, read_primer, sample_midi_paths
from model_loader import load_model
from models import get_model_hparams
from train_chords import ChordPitchSamplerDataset
from train_style_chords import StyleChordSamplerDataset
from x_transformer import AE_style_jokerParam, AE_style_jokerParam_chords

CHORD_INPUTS = ('chord_chroma', 'chord_anchor_pc', 'chord_active')


def check_style_windows(samples: int = 64) -> dict:
    """Training samples keep their press window; the style window sits next to it in the same piece."""
    dataset = StyleChordSamplerDataset(fixture(), 64, 64)
    sides = {'before': 0, 'after': 0}
    for index in range(samples):
        torch.manual_seed(1729 + index)
        source = ChordPitchSamplerDataset.__getitem__(dataset, index)  # the same draws, with source keys
        torch.manual_seed(1729 + index)
        sample = dataset[index]
        require(all(torch.equal(sample[key], source[key]) for key in ('pitch',) + CHORD_INPUTS),
                'The style window changed the press window')
        start, stop, shift = (int(source[key]) for key in ('source_start', 'source_stop', 'transpose'))
        piece = int(dataset.note_piece[start])
        lower, upper = int(dataset.piece_edges[piece]), int(dataset.piece_edges[piece + 1])
        count = int(sample['style_mask'].sum())
        require(count >= 32 and bool(sample['style_mask'][:count].all()), 'Style window is not a prefix at least half full')
        style = sample['style_pitch'][:count].numpy() - shift
        if stop + count <= upper and np.array_equal(style, dataset.pitches[stop:stop + count]):
            sides['after'] += 1
        else:
            require(start - count >= lower and np.array_equal(style, dataset.pitches[start - count:start]),
                    'Style window is not next to the press window in the same piece')
            sides['before'] += 1
    return {'samples': samples, 'style_window_side': sides}


@torch.inference_mode()
def check_inputs() -> dict:
    """Decoder inputs of a small random change model (both adapters nonzero)."""
    cfg = get_model_hparams('AE_style_jokerParam_change')
    cfg.update(seq_len=64, emb_dim=32, num_layers=2, heads=2, chord_cond_dim=16, style_seq_len=16, style_encoder_depth=1)
    model = load_model(cfg['model_name'], cfg=cfg, set_only=True).eval()
    torch.manual_seed(123)
    for film in (model.decoder.harm_film, model.decoder.change_film):
        film.proj.weight.normal_(0, 0.05)
    style_mask = torch.ones(2, 16, dtype=torch.bool)
    style = model.encode_style(torch.randint(40, 80, (2, 16)), style_mask)
    tokens = {'pitch': torch.randint(40, 80, (2, 65)), 'button': torch.randint(0, cfg['num_buttons'], (2, 65)),
              'chord_chroma': torch.zeros(2, 65, 12, dtype=torch.bool), 'chord_anchor_pc': torch.full((2, 65), 7),
              'chord_active': torch.zeros(2, 65, dtype=torch.bool), 'chord_change': torch.zeros(2, 65, dtype=torch.bool)}
    tokens['button'][:, 50:] = model.joker_button_idx
    tokens['chord_chroma'][:, 20:, [2, 7, 11]] = True
    tokens['chord_active'][:, 20:] = True
    captured: Dict[str, Tensor] = {}

    def capture_inputs(module: torch.nn.Module, inputs: tuple, kwargs: dict) -> None:
        captured.clear()
        captured.update(inputs[0])
        captured.update(kwargs)

    hook = model.decoder.register_forward_pre_hook(capture_inputs, with_kwargs=True)
    model.next_pitch_logits(tokens, style, style_mask)
    require(torch.equal(captured['pitch'], tokens['pitch'][:, :-1]), 'Decoder received a target pitch early')
    for key in CHORD_INPUTS:
        require(torch.equal(captured[key], tokens[key][:, 1:]), 'Chord control is not aligned with the predicted note')
    joker = tokens['button'][:, 1:] == model.joker_button_idx
    require(torch.equal(captured['joker_mask'], joker), 'Joker positions are not aligned with the predicted note')
    require(torch.equal(captured['button'], torch.where(joker, 0.0, model.quantizer.discrete_to_real(tokens['button'][:, 1:]))),
            'Buttons are not aligned with the predicted note')
    switched = {key: value.clone() for key, value in tokens.items()}
    switched['chord_active'][:, 40:] = False
    switched['chord_change'][:, 40:] = True  # Space pressed while the chord was held
    model.next_pitch_logits(switched, style, style_mask)
    require(not captured['chord_active'].any() and torch.equal(captured['chord_change'], switched['chord_change'][:, 1:]),
            'The chord held before Space was not forgotten')
    hook.remove()
    released = {key: value.clone() for key, value in tokens.items()}
    released['chord_active'][1, 40:] = False  # row 1 released the chord; row 0 still holds it
    logits = model.next_pitch_logits(released, style, style_mask)
    free = free_logits(model, released['pitch'], released['button'], style, style_mask)[:, -1]
    require(torch.equal(logits[1], free[1]), 'A released row differs from the free style joker')
    require(not torch.equal(logits[0], free[0]), 'A held chord did not change the logits')
    for row in (0, 1):  # held chord, then released (the cached free path)
        single = {key: value[row:row + 1] for key, value in released.items()}
        torch.manual_seed(91)
        sampled = model.gen_pitch_token(single, style[row:row + 1], style_mask[row:row + 1])[0]
        torch.manual_seed(91)
        expected = torch.multinomial(F.softmax(model.next_pitch_logits(single, style[row:row + 1], style_mask[row:row + 1]), -1), 1)
        require(sampled == int(expected.item()), 'gen_pitch_token does not sample next_pitch_logits')
    return {'target_alignment': True, 'button_and_joker_alignment': True, 'space_forgets_held_chord': True,
            'release_is_free_style_joker': True, 'gen_pitch_token_samples_next_pitch_logits': True}


def load_pair(checkpoint: Path, base_checkpoint: Path, device: torch.device
              ) -> Tuple[AE_style_jokerParam_chords, AE_style_jokerParam, dict]:
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = saved['cfg']
    require(cfg.get('model_type') == 'AE_style_jokerParam_chords', 'Expected an AE_style_jokerParam_chords checkpoint')
    model = load_model(cfg['model_name'], cfg=cfg, set_only=True)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    # The original style joker: the same configuration without the chord adapters.
    base = load_model(cfg['model_name'], cfg={**cfg, 'model_type': 'AE_style_jokerParam'}, set_only=True)
    original = torch.load(base_checkpoint, map_location='cpu', weights_only=False)
    base.load_state_dict(original.get('model_state_dict', original), strict=True)
    require(all(torch.equal(value, model.state_dict()[key]) for key, value in base.state_dict().items()),
            'Checkpoint style joker differs from its original')
    require(all(not model.get_parameter(key).requires_grad for key, _ in base.named_parameters()),
            'Original style joker parameters are not frozen')
    metadata = {key: saved.get(key) for key in ('epoch', 'steps', 'validation')}
    return model.to(device).eval(), base.to(device).eval(), metadata


def space_schedules(first: str, notes: int) -> List[dict]:
    """Change control (Space) schedules, for a checkpoint with a change adapter."""
    return [{'name': 'hold_space', 'events': [(0, 'space', True)]},
            {'name': 'release_space', 'events': [(0, 'space', True), (notes // 2, 'space', False)]},
            {'name': 'chord_then_space', 'events': [(0, first, True), (notes // 2, 'space', True), (notes // 2, first, False)]}]


@torch.inference_mode()
def generate(model: AE_style_jokerParam_chords, base: AE_style_jokerParam, source: Dict[str, List[int]],
             primer_length: int, style_pitch: np.ndarray, bank: Dict[str, Chord], cases: List[dict], notes: int,
             context_length: int, seed: int, temperature: float, change_key, device: torch.device
             ) -> Tuple[List[Tuple[str, str]], Tensor, List[List[dict]], List[List[dict]]]:
    """Rows = button modes x schedules, with identical per-step Gumbel noise; no attention cache."""
    rows = [(mode, case) for mode in MODES for case in cases]
    names = [(mode, case['name']) for mode, case in rows]
    length = primer_length + notes
    real = torch.tensor(source['pitch'][:length], device=device).unsqueeze(0)
    # An empty window at 0 makes style_inputs take the sample's first style_seq_len notes.
    style, style_mask = style_inputs(model, style_pitch, 0, 0, device)
    base_style, _ = style_inputs(base, style_pitch, 0, 0, device)
    played = {mode: mode_buttons(model, real, primer_length, mode) for mode in MODES}
    buttons = torch.cat([played[mode] for mode, _ in rows])
    pitches = torch.zeros(len(rows), length, dtype=torch.long, device=device)
    pitches[:, :primer_length] = real[:, :primer_length]
    controls = [ChordControls(primer_length, context_length, bank, change_key=change_key) for _ in rows]
    requests: List[List[dict]] = [[] for _ in rows]
    probes: List[List[dict]] = [[] for _ in rows]
    rng = torch.Generator().manual_seed(seed)
    probe_steps = {0, notes - 1, notes // 2}
    probe_steps.update(max(0, step - 1) for case in cases for step, _, _ in case['events'])
    probe_steps.update(step for case in cases for step, _, _ in case['events'])
    # Each generated pitch becomes the next history input, so time steps are sequential.
    for step in range(notes):
        for control, (_, case) in zip(controls, rows):
            for at, key, pressed in case['events']:
                if at == step:
                    control.handle_key(key.removeprefix('shift+'), pressed, shift=key.startswith('shift+'))
        states = [control.next_tokens(torch.device('cpu')) for control in controls]
        tokens = {key: torch.cat([state[key] for state in states]).to(device) for key in states[0]}
        position = primer_length + step
        window = slice(max(0, position - context_length), position + 1)
        tokens['pitch'], tokens['button'] = pitches[:, window], buttons[:, window]
        logits = model.next_pitch_logits(tokens, style.expand(len(rows), -1, -1), style_mask.expand(len(rows), -1))
        if step in probe_steps:
            original = free_logits(base, tokens['pitch'], tokens['button'], base_style, style_mask)[:, -1]
            held = tokens['chord_active'][:, -1]
            if 'chord_change' in tokens:
                held = held | tokens['chord_change'][:, -1]
            # Identical actual pitches, buttons and batch shape, including after a guided passage.
            require(torch.equal(logits[~held], original[~held]), f'Free rows differ from the original at generated note {step}')
            tv = 0.5 * (logits.softmax(-1) - original.softmax(-1)).abs().sum(-1)
            for index, control in enumerate(controls):
                probes[index].append({'note': step, 'active': control.active or control.change,
                                      'tv_vs_base_same_history': float(tv[index]),
                                      'max_logit_difference': float((logits[index] - original[index]).abs().max())})
        # CPU generator gives the same noise across devices and checkpoints, too.
        noise = torch.empty(128).exponential_(generator=rng).to(device)
        pitches[:, position] = (logits / temperature - noise.log()).argmax(-1)
        for index, control in enumerate(controls):
            requests[index].append({'name': control.chord_name or 'free', 'active': control.active or control.change,
                                    'anchor_pc': control.anchor_pc if control.active else None,
                                    'pitch_classes': control.chroma.nonzero().flatten().tolist()})
            control.commit_note()
    for index, (mode, name) in enumerate(names):
        if name.startswith('release_'):
            held_row = names.index((mode, name.replace('release_', 'hold_', 1)))
            require(torch.equal(pitches[index, :primer_length + notes // 2], pitches[held_row, :primer_length + notes // 2]),
                    'A future release affected the continuation before release')
    return names, pitches.cpu(), requests, probes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, nargs='+', default=[ROOT / 'save_models/AE_style_jokerParam_chords_best.pth'])
    parser.add_argument('--base-checkpoint', type=Path,
                        default=ROOT / get_model_hparams('AE_style_jokerParam_chords')['init_from_ckpt'],
                        help='The original AE_style_jokerParam checkpoint of both adapters')
    parser.add_argument('--sample', type=Path, nargs='+', default=[ROOT / sample_midi_paths[index] for index in (0, 4)])
    parser.add_argument('--chord-keys', default='1,4,5,shift+2', help='Keys in each primer\'s existing fixed chord palette')
    parser.add_argument('--primer-length', type=int, default=300,
                        help='Maximum primer notes; shortened if necessary to leave source timing for the continuation')
    parser.add_argument('--context-length', type=int, default=300)
    parser.add_argument('--notes', type=int, default=64)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--output', type=Path, default=ROOT / 'out/style_chord_verification')
    parser.add_argument('--soundfont', type=Path, default=ROOT / 'piano.sf2')
    parser.add_argument('--no-audio', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if (args.notes < 12 or args.seeds < 1 or args.threads < 1
            or not 1 <= args.primer_length <= args.context_length
            or not np.isfinite(args.temperature) or args.temperature <= 0):
        parser.error('Require notes >= 12, positive seeds/threads/temperature, and primer <= context')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a new/empty output directory to preserve earlier comparisons')
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    print('Checking controls, style windows and decoder inputs...', flush=True)
    checks = {'controls': check_controls(), 'style_windows': check_style_windows(), 'inputs': check_inputs()}
    report = {'checks': checks, 'checkpoints': [], 'completed': False, 'listening_assessment': 'pending',
              'protocol': {key: str(value) if isinstance(value, Path) else
                           [str(item) for item in value] if isinstance(value, list) else value
                           for key, value in vars(args).items()}}
    if args.check_only:
        report['completed'] = True
        write_report(args.output, report)
        print('All implementation checks passed:', args.output / 'report.json')
        return
    keys = args.chord_keys.split(',')
    if len(keys) < 2 or len(set(keys)) != len(keys):
        parser.error('Choose at least two distinct chord keys')
    indices = {(ROOT / path).resolve(): index for index, path in enumerate(sample_midi_paths)}
    sources = []
    for path in args.sample:
        if path.resolve() not in indices:
            parser.error('Use a configured performer sample so the explicit chord palette is known')
        bank = FIXED_CHORDS[indices[path.resolve()]]
        if any(key not in bank for key in keys):
            parser.error(f'Unknown chord key for {path.name}')
        source = read_primer(str(path), args.primer_length + args.notes)
        primer_length = min(args.primer_length, len(source['pitch']) - args.notes)
        if primer_length < 1:
            parser.error(f'Not enough source timing events in {path}')
        style_pitch = np.asarray(read_primer(str(path), 10 ** 9)['pitch'])  # the whole sample; the style takes its start
        sources.append((path, source, bank, primer_length, style_pitch))
    audio_available = bool(shutil.which('fluidsynth') and args.soundfont.is_file() and not args.no_audio)
    report['audio_rendered'] = audio_available
    # Enough shuffled clip numbers for every run, with the Space schedules of a change checkpoint.
    runs_per_seed = len(MODES) * (len(schedules(keys, args.notes)) + len(space_schedules(keys[0], args.notes)))
    identifiers = list(range(len(args.checkpoint) * len(sources) * args.seeds * runs_per_seed))
    random.Random(args.seed).shuffle(identifiers)
    next_id = 0
    for checkpoint in args.checkpoint:
        model, base, metadata = load_pair(checkpoint, args.base_checkpoint, torch.device(args.device))
        if args.context_length > model.max_seq_len:
            parser.error('context-length exceeds checkpoint sequence length')
        change_key = 'space' if model.decoder.change_film is not None else None
        cases = schedules(keys, args.notes) + (space_schedules(keys[0], args.notes) if change_key else [])
        result = {'path': str(checkpoint.resolve()), 'sha256': sha256(checkpoint), 'metadata': metadata,
                  'base_path': str(args.base_checkpoint.resolve()), 'base_sha256': sha256(args.base_checkpoint), 'runs': []}
        report['checkpoints'].append(result)
        for sample_index, (path, source, bank, primer_length, style_pitch) in enumerate(sources):
            for offset in range(args.seeds):
                seed = args.seed + offset
                print(f'{checkpoint.name}: {path.name}, seed {seed}', flush=True)
                names, pitches, requests, probes = generate(model, base, source, primer_length, style_pitch, bank, cases,
                                                            args.notes, args.context_length, seed, args.temperature,
                                                            change_key, torch.device(args.device))
                for index, (mode, name) in enumerate(names):
                    clip_name = f'clip_{identifiers[next_id]:04d}'
                    next_id += 1
                    midi, clip, audio_path = (args.output / (clip_name + suffix) for suffix in ('.mid', '_listen.mid', '.wav'))
                    sequence = pitches[index].tolist()
                    write_midi(midi, source, sequence, primer_length, requests[index])
                    write_midi(clip, source, sequence, primer_length, requests[index], lead_in=24)
                    if audio_available:
                        render_audio(clip, args.soundfont, audio_path)
                    events = next(case['events'] for case in cases if case['name'] == name)
                    transitions = [step for step, _, _ in events]
                    free = names.index((mode, 'free'))
                    result['runs'].append({
                        'id': clip_name, 'pair': f'{sample_index + 1}-{offset + 1}', 'sample': str(path), 'seed': seed,
                        'primer_length': primer_length, 'mode': mode, 'condition': f'{mode}_{name}', 'events': events,
                        'requests': requests[index], 'pitches': sequence,
                        'quality': quality_metrics(sequence, primer_length, transitions),
                        'source_quality': quality_metrics(source['pitch'], primer_length, transitions),
                        'probes': probes[index],
                        'disagreement_with_paired_free': float((pitches[index, primer_length:] != pitches[free, primer_length:]).float().mean()),
                        'midi': midi.name, 'clip_midi': clip.name, 'audio': audio_path.name if audio_available else None})
                write_report(args.output, report)
        del model, base
        gc.collect()
    report['completed'] = True
    write_report(args.output, report)
    print('Saved comparison:', args.output / 'listening.html', flush=True)


if __name__ == '__main__':
    main()
