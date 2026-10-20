"""Verify chord conditioning and export paired performance/listening comparisons.

Use --check-only for data, control and masking checks without generation. A normal
run also checks each checkpoint against its original base and exports MIDI, audio
(when FluidSynth is installed), report.json and a blind listening page. Chord
blocks are read exactly as exported; this evaluator performs no harmony analysis.
"""

import argparse
import gc
import hashlib
import html
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mido
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harmony.chord_joker import Chord, ChordControls
from interaction_1_button_chords import FIXED_CHORDS, read_primer, sample_midi_paths
from midiUtils import Any_Pickle_File_Reader
from model_loader import load_model
from models import get_model_hparams
from train_chords import ChordPitchSamplerDataset, collate_chord_windows
from x_transformer import D_Base, D_Chord_Joker, Decoder_no_conditioning

FEATURES = ('pitch', 'chord_chroma', 'chord_anchor_pc', 'chord_active')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def reference_blocks(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent event-by-event reference, including interleaved exported tones."""
    pitches, pieces, groups, blocks = [], [], [], []
    piece, block = 0, -1
    for dt, dur, pitch, velocity, channel in data.reshape(-1, 5).tolist():
        if [dt, dur, pitch, velocity, channel] == [126, 126, 0, 0, 0]:
            piece += 1
            block = -1
        elif channel == 120:
            block = len(blocks)
            blocks.append([])
        elif channel == 4:
            if velocity == 0:
                block = -1
            else:
                require(block >= 0, 'Exported tones have no chord-start marker')
                blocks[block].append(pitch)
        elif 0 <= channel < 16 and channel not in (3, 9) and velocity > 0:
            pitches.append(pitch)
            pieces.append(piece)
            groups.append(block)
    chroma = np.zeros((len(pitches), 12), dtype=bool)
    anchors = np.zeros(len(pitches), dtype=np.int64)
    for index, group in enumerate(groups):
        if group >= 0 and blocks[group]:
            chroma[index, np.asarray(blocks[group]) % 12] = True
            anchors[index] = min(blocks[group]) % 12
        else:
            groups[index] = -1
    return np.asarray(pitches), np.asarray(pieces), np.asarray(groups), chroma, anchors


def data_fidelity(data: np.ndarray, seq_len: int, samples: int) -> dict:
    pitch, piece, groups, chroma, anchors = reference_blocks(data)
    dataset = ChordPitchSamplerDataset(data, seq_len, include_source=True)
    require(np.array_equal(dataset.pitches, pitch), 'Annotation notes leaked into musical pitches')
    require(np.array_equal(dataset.note_piece, piece), 'Piece boundaries differ from the export')
    require(np.array_equal(dataset.chroma[dataset.groups], chroma), 'Assembled chord tones differ from export')
    require(np.array_equal(dataset.anchors[dataset.groups], anchors), 'Assembled anchors differ from lowest exported note')
    # Distinct exported blocks must remain distinct even when chroma is identical.
    valid = groups >= 0
    pairs = np.unique(np.stack((groups[valid], dataset.groups[valid]), axis=-1), axis=0)
    require(len(pairs) == len(np.unique(groups[valid])) == len(np.unique(pairs[:, 1])),
            'Different chord blocks were merged or split')
    counts = [0, 0, 0]
    shifts = set()
    for mode in ('full', 'mixed'):
        dataset.history_mode = mode
        for index in range(samples):
            torch.manual_seed(1729 + index)
            sample = dataset[index % len(dataset)]
            lo, hi, shift = (int(sample[key]) for key in ('source_start', 'source_stop', 'transpose'))
            require(len(np.unique(piece[lo:hi])) == 1, 'Sample crossed a piece boundary')
            require(np.array_equal(sample['pitch'].numpy(), pitch[lo:hi] + shift), 'Targets changed during augmentation')
            active = sample['chord_active'].numpy()
            require(active[1:].any(), 'Sample lost every supervised target')
            require(len(np.unique(groups[lo:hi][active])) == 1 and (groups[lo:hi][active] >= 0).all(),
                    'Simulated hold crossed an exported chord boundary')
            expected = np.roll(chroma[lo:hi], shift, axis=-1) & active[:, None]
            require(np.array_equal(sample['chord_chroma'].numpy(), expected), 'Sampled chroma/transposition mismatch')
            expected_anchor = np.where(active, (anchors[lo:hi] + shift) % 12, 0)
            require(np.array_equal(sample['chord_anchor_pc'].numpy(), expected_anchor), 'Sampled anchor/transposition mismatch')
            shifts.add(shift)
            if mode == 'mixed':
                kind = int(sample['history_kind'])
                counts[kind] += 1
                masked = sample['history_mask'].nonzero().flatten()
                if kind == 1:
                    require(1 <= hi - lo - 1 <= 256, 'Shortened suffix exceeds configured history')
                elif kind == 2:
                    require(8 <= len(masked) <= 32 and int(masked[-1] - masked[0]) + 1 == len(masked),
                            'History mask is not one 8..32-note span')
                    last = int(sample['chord_active'].nonzero()[-1])
                    require(int(masked[-1]) < last and bool((~sample['history_mask'][:last]).any()),
                            'Mask does not leave visible context before a supervised target')
    evaluation = ChordPitchSamplerDataset(data, seq_len, is_eval=True, history_mode='mixed')
    require(evaluation.history_mode == 'full' and 'history_mask' not in evaluation[0], 'Validation used reduced history')
    require(any(shift != 0 for shift in shifts), 'No nonzero transposition exercised')
    require(all(counts), 'Not every training history mode was exercised')
    return {'performed_notes': len(pitch), 'exported_blocks': len(pairs), 'sample_checks': samples * 2,
            'transpositions': sorted(shifts), 'mixed_history_counts': counts}


def fixture() -> np.ndarray:
    events = []
    for transpose in (0, 12):
        events.append([126, 126, 0, 0, 0])
        events += [[1, 1, 45 + transpose, 80, 0]] * 8
        for chord in ((55, 60, 64), (55, 60, 64), (50, 54, 57)):
            events.append([0, 0, 0, 0, 120])
            for index, tone in enumerate(chord):
                # Performed notes between exported tones must see the complete block.
                events += [[0, 30, tone + transpose, 80, 4], [1, 1, 48 + index + transpose, 90, 0]]
            events += [[1, 1, 48 + (index * 7) % 24 + transpose, 90, 0] for index in range(80)]
        events += [[0, 0, 0, 0, 4], [1, 1, 44 + transpose, 80, 0], [0, 1, 110, 90, 3]]
        # Leave an active block at the piece end to exercise boundary clearing.
        events += [[0, 0, 0, 0, 120], [0, 30, 48 + transpose, 80, 4], [1, 1, 50 + transpose, 80, 0]]
    return np.asarray(events, dtype=np.int16).reshape(-1)


def check_controls() -> dict:
    bank = {'1': Chord('C', 0, (0, 4, 7)), '5': Chord('G7', 7, (0, 4, 7, 10)),
            'shift+1': Chord('C minor', 0, (0, 3, 7))}
    keyboard = ChordControls(3, 8, bank)
    midi = ChordControls(3, 8, bank, {60: '1', 67: '5'})
    actions = [('1', True), ('1', True), ('5', True), ('1', True), ('5', False), ('1', False),
               ('5', True), ('1', True), ('5', False), ('1', False)]
    for key, pressed in actions:
        before = keyboard.next_tokens(torch.device('cpu'))
        keyboard.handle_key(key, pressed)
        midi.handle_midi(0x90, 60 if key == '1' else 67, 80 if pressed else 0)
        left, right = keyboard.next_tokens(torch.device('cpu')), midi.next_tokens(torch.device('cpu'))
        require(all(torch.equal(left[k], right[k]) for k in left), 'Keyboard and MIDI control states differ')
        if keyboard.active:
            require(torch.equal(left['chord_chroma'][:, :-1], before['chord_chroma'][:, :-1]),
                    'Chord switching rewrote past conditioning')
        else:
            require(not left['chord_active'].any() and not left['chord_chroma'].any(), 'Release left control history')
        keyboard.commit_note()
        midi.commit_note()
    controls = ChordControls(2, 8, bank, {60: '1', 67: '5'})
    controls.handle_key('1', True, shift=True)
    controls.handle_key('1', False)  # Shift released before the physical key.
    require(not controls.active, 'Shift release left a stuck chord')
    controls.handle_key('1', True)
    controls.handle_midi(0x91, 67, 80)
    controls.handle_midi(0x90, 67, 80)
    controls.handle_midi(0x80, 67, 0)
    require(controls.anchor_pc == 7, 'Same MIDI note on independent channels was released early')
    controls.release_keyboard()
    require(controls.active, 'Focus loss released a still-held MIDI chord')
    controls.handle_midi(0x91, 67, 0)
    require(not controls.active, 'Velocity-zero note-on did not release the chord')
    require(not controls.handle_midi(0x90, 61, 80), 'Unmapped MIDI trigger was consumed as a chord')
    require(not controls.handle_midi(0xB0, 60, 127), 'CC was misread as a note')
    controls.reset(1)
    require(controls.next_tokens(torch.device('cpu'))['chord_active'].shape == (1, 2), 'Reset/history length mismatch')
    return {'keyboard_midi_equivalent': True, 'switch_history_preserved': True,
            'overlap_restore': True, 'velocity_zero_off': True, 'channel_independence': True}


def check_masking() -> dict:
    cfg = get_model_hparams('D_Chord_Joker_mixed_history_v1')
    cfg.update(seq_len=64, emb_dim=32, num_layers=2, heads=2, chord_cond_dim=16)
    model = load_model(cfg['model_name'], cfg=cfg, set_only=True).train()
    torch.manual_seed(123)
    tokens = {'pitch': torch.randint(48, 84, (2, 65)),
              'chord_chroma': torch.zeros(2, 65, 12, dtype=torch.bool),
              'chord_anchor_pc': torch.zeros(2, 65, dtype=torch.long),
              'chord_active': torch.ones(2, 65, dtype=torch.bool),
              'history_mask': torch.zeros(2, 65, dtype=torch.bool)}
    tokens['chord_chroma'][..., [0, 4, 7]] = True
    tokens['history_mask'][:, 40:64] = True  # Includes the most recent input pitch.
    captured: Dict[str, Tensor] = {}

    def capture_inputs(module: torch.nn.Module, inputs: tuple, kwargs: dict) -> None:
        captured.update(kwargs)
        captured['pitch'] = inputs[0]['pitch']

    aligned = {key: value.clone() for key, value in tokens.items()}
    aligned['chord_anchor_pc'] = torch.arange(65).remainder(12).expand(2, -1)
    aligned['chord_chroma'] = F.one_hot(aligned['chord_anchor_pc'], 12).bool()
    aligned['chord_active'][:, :16] = False
    hook = model.decoder.register_forward_pre_hook(capture_inputs, with_kwargs=True)
    with torch.no_grad():
        model.pitch_logits(aligned)
    hook.remove()
    require(torch.equal(captured['pitch'], aligned['pitch'][:, :-1]), 'Decoder received a target pitch early')
    for key in FEATURES[1:]:
        require(torch.equal(captured[key], aligned[key][:, 1:]), 'Chord condition is not aligned to the predicted note')
    changed = {key: value.clone() for key, value in tokens.items()}
    changed['pitch'][tokens['history_mask']] += 1
    with torch.no_grad():
        model.decoder.harm_film.proj.weight.normal_(0, 0.01)
        require(torch.equal(model.pitch_logits(tokens), model.pitch_logits(changed)), 'Hidden pitches leaked into decoder states')
    original_targets = tokens['pitch'][:, 1:].clone()
    logits = model.pitch_logits(tokens)
    losses, _ = model(tokens)
    require(torch.equal(tokens['pitch'][:, 1:], original_targets), 'Masking mutated training targets')
    require(torch.allclose(losses['loss_recons'], F.cross_entropy(logits.transpose(1, 2), original_targets)),
            'Masked inputs replaced the original prediction targets')
    losses['loss_total'].backward()
    require(bool(model.decoder.history_mask_embedding.grad.abs().sum() > 0), 'Mask embedding has no gradient')
    require(all(p.grad is None for p in model.parameters() if not p.requires_grad), 'Original decoder received gradients')
    clean = {key: value for key, value in tokens.items() if key != 'history_mask'}
    for training in (True, False):
        model.train(training)
        require(torch.equal(model.next_pitch_logits(tokens), model.next_pitch_logits(clean)),
                'Generation used masked history')
        torch.manual_seed(91)
        first = model.gen_pitch_token({key: value[:1] for key, value in tokens.items()})
        torch.manual_seed(91)
        second = model.gen_pitch_token({key: value[:1] for key, value in clean.items()})
        require(first == second and 0 <= first < 128, 'Generated pitch depended on mask metadata')
    require(model.decoder.to_logits.out_features == 128, 'Mask embedding entered the output vocabulary')
    # A short example has no prefix in the decoder; future padding cannot restore it.
    samples = [{key: value[index, :length].clone() for key, value in tokens.items()}
               for index, length in ((0, 65), (1, 17))]
    padded = collate_chord_windows(samples)
    with torch.inference_mode():
        single = model.pitch_logits({key: value.unsqueeze(0) for key, value in samples[1].items()})
        torch.testing.assert_close(model.pitch_logits(padded)[1:2, :16], single, atol=2e-6, rtol=2e-6)
    return {'hidden_pitch_invariance': True, 'targets_unchanged': True, 'input_only_mask': True,
            'target_condition_alignment': True, 'generation_ignores_masks_in_train_and_eval': True,
            'shortened_padding_invariance': True}


def load_pair(checkpoint: Path, base_override: Optional[Path], device: torch.device) -> Tuple[D_Chord_Joker, D_Base, dict, Path]:
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = saved['cfg']
    require(cfg.get('model_type') == 'D_Chord_Joker', 'Expected a D_Chord_Joker checkpoint')
    model = load_model(cfg['model_name'], cfg=cfg, set_only=True)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    base_path = base_override or Path(cfg['init_from_ckpt'])
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    base = D_Base(Decoder_no_conditioning(max_seq_len=cfg['seq_len'], dim=cfg['emb_dim'],
                                         depth=cfg['num_layers'], heads=cfg['heads']), cfg)
    original = torch.load(base_path, map_location='cpu', weights_only=False)
    base.load_state_dict(original.get('model_state_dict', original), strict=True)
    require(all(torch.equal(value, model.state_dict()[key]) for key, value in base.state_dict().items()),
            'Checkpoint pitch prior differs from its original base')
    require(all(not model.get_parameter(key).requires_grad for key, _ in base.named_parameters()),
            'Original decoder parameters are not frozen')
    metadata = {key: saved.get(key) for key in ('epoch', 'steps', 'validation')}
    metadata['training_seed'] = cfg.get('chord_training_seed')
    metadata['history_mode'] = cfg.get('chord_history_mode', 'full')
    return model.to(device).eval(), base.to(device).eval(), metadata, base_path


def schedules(keys: List[str], notes: int) -> List[dict]:
    cases = [{'name': 'free', 'events': []}]
    for key in keys:
        cases += [{'name': 'hold_' + key, 'events': [(0, key, True)]},
                  {'name': 'release_' + key, 'events': [(0, key, True), (notes // 2, key, False)]}]
    first, second = keys[:2]
    cases += [{'name': 'switch', 'events': [(0, first, True), (notes // 2, second, True), (notes // 2, first, False)]},
              {'name': 'overlap_restore', 'events': [(0, first, True), (notes // 3, second, True),
                                                    (2 * notes // 3, second, False), (notes - 4, first, False)]}]
    return cases


@torch.inference_mode()
def generate(model: D_Chord_Joker, base: D_Base, prefix: List[int], bank: Dict[str, Chord],
             cases: List[dict], notes: int, context_length: int, seed: int,
             temperature: float, device: torch.device) -> Tuple[Tensor, List[List[dict]], List[List[dict]]]:
    """Batch variants with identical per-step Gumbel noise; no attention cache."""
    pitches = torch.zeros(len(cases), len(prefix) + notes, dtype=torch.long, device=device)
    pitches[:, :len(prefix)] = torch.tensor(prefix, device=device)
    controls = [ChordControls(len(prefix), context_length, bank) for _ in cases]
    requests: List[List[dict]] = [[] for _ in cases]
    probes: List[List[dict]] = [[] for _ in cases]
    rng = torch.Generator().manual_seed(seed)
    probe_steps = {0, notes - 1, notes // 2}
    probe_steps.update(max(0, step - 1) for case in cases for step, _, _ in case['events'])
    probe_steps.update(step for case in cases for step, _, _ in case['events'])
    # Each generated pitch becomes the next history input, so time steps are sequential.
    for step in range(notes):
        for control, case in zip(controls, cases):
            for at, key, pressed in case['events']:
                if at == step:
                    physical = key.removeprefix('shift+')
                    control.handle_key(physical, pressed, shift=key.startswith('shift+'))
        states = [control.next_tokens(torch.device('cpu')) for control in controls]
        tokens = {key: torch.cat([state[key] for state in states]).to(device) for key in states[0]}
        position = len(prefix) + step
        tokens['pitch'] = pitches[:, max(0, position - context_length):position + 1]
        logits = model.next_pitch_logits(tokens)
        if step in probe_steps:
            original = base.decoder({'pitch': tokens['pitch'][:, :-1]})[:, -1]
            free = ~tokens['chord_active'][:, -1]
            # Identical actual pitches and batch shape, including after a guided passage.
            require(torch.equal(logits[free], original[free]), f'Bypass differs from base at generated note {step}')
            tv = 0.5 * (logits.softmax(-1) - original.softmax(-1)).abs().sum(-1)
            pairwise = (0.5 * (logits.softmax(-1)[:, None] - logits.softmax(-1)[None, :]).abs().sum(-1)).cpu().tolist() if step == 0 else []
            for index, control in enumerate(controls):
                probes[index].append({'note': step, 'active': control.active,
                                      'tv_vs_base_same_history': float(tv[index]),
                                      'max_logit_difference': float((logits[index] - original[index]).abs().max())})
                if step == 0:
                    probes[index][-1]['tv_to_first_step_variants'] = {case['name']: value for case, value in zip(cases, pairwise[index])}
        # CPU generator gives the same noise across devices and checkpoints, too.
        noise = torch.empty(128).exponential_(generator=rng).to(device)
        pitches[:, position] = (logits / temperature - noise.log()).argmax(-1)
        for index, control in enumerate(controls):
            requests[index].append({'name': control.chord_name or 'free', 'active': control.active,
                                    'anchor_pc': control.anchor_pc if control.active else None,
                                    'pitch_classes': control.chroma.nonzero().flatten().tolist()})
            control.commit_note()
    for index, case in enumerate(cases):
        if case['name'].startswith('release_'):
            held = next(i for i, other in enumerate(cases) if other['name'] == case['name'].replace('release_', 'hold_', 1))
            require(torch.equal(pitches[index, :len(prefix) + notes // 2], pitches[held, :len(prefix) + notes // 2]),
                    'A future release affected the continuation before release')
    return pitches.cpu(), requests, probes


def quality_metrics(pitches: List[int], primer_length: int, transitions: List[int]) -> dict:
    generated = np.asarray(pitches[primer_length:])
    previous = np.asarray(pitches[primer_length - 1:])
    jumps = np.abs(np.diff(previous))
    grams = [tuple(generated[index:index + 4]) for index in range(max(0, len(generated) - 3))]
    edges = np.r_[0, np.flatnonzero(np.diff(generated)) + 1, len(generated)]
    return {'repeat_fraction': float(np.mean(np.diff(generated) == 0)),
            'repeated_4gram_fraction': 1 - len(set(grams)) / len(grams) if grams else 0.0,
            'longest_same_pitch_run': int(np.diff(edges).max()), 'distinct_pitches': len(set(generated.tolist())),
            'pitch_min': int(generated.min()), 'pitch_max': int(generated.max()),
            'pitch_range': int(np.ptp(generated)), 'mean_absolute_jump': float(jumps.mean()),
            'large_leap_fraction': float((jumps > 12).mean()),
            'transition_jumps': {str(step): int(jumps[step]) for step in sorted(set([0] + transitions))}}


def write_midi(path: Path, source: Dict[str, List[int]], pitches: List[int], primer_length: int,
               requests: List[dict], lead_in: Optional[int] = None) -> None:
    first = max(0, primer_length - lead_in) if lead_in is not None else 0
    starts = np.cumsum(source['dtime'][:len(pitches)]) * 32
    starts -= starts[first]
    events = []
    for index in range(first, len(pitches)):
        at = int(starts[index])
        events.append((at, 2, mido.Message('note_on', note=pitches[index], velocity=source['vel'][index])))
        events.append((at + max(1, source['dur'][index] * 32), 0,
                       mido.Message('note_off', note=pitches[index], velocity=0)))
        if index >= primer_length:
            step = index - primer_length
            if step == 0 or requests[step]['name'] != requests[step - 1]['name']:
                events.append((at, 1, mido.MetaMessage('marker', text='Guidance: ' + requests[step]['name'])))
    track = mido.MidiTrack([mido.MetaMessage('set_tempo', tempo=1_000_000)])
    midi = mido.MidiFile(ticks_per_beat=1000)
    midi.tracks.append(track)
    previous = 0
    for at, _, message in sorted(events, key=lambda item: item[:2]):
        track.append(message.copy(time=at - previous))
        previous = at
    midi.save(path)


def render_audio(midi: Path, soundfont: Path, output: Path) -> None:
    subprocess.run(['fluidsynth', '-ni', '-g', '0.5', '-r', '22050', '-F', str(output),
                    str(soundfont), str(midi)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def write_report(output: Path, report: dict) -> None:
    for checkpoint in report['checkpoints']:
        free_probes = [probe for run in checkpoint['runs'] for probe in run['probes'] if not probe['active']]
        after_release = [probe for run in checkpoint['runs'] for probe in run['probes']
                         if not probe['active'] and any(request['active'] for request in run['requests'][:probe['note']])]
        checkpoint['bypass_verification'] = {
            'probes': len(free_probes), 'after_guided_passage_probes': len(after_release),
            'exact_equality': bool(free_probes) and all(probe['max_logit_difference'] == 0 for probe in free_probes)}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    rows = []
    for checkpoint in report['checkpoints']:
        for condition in sorted({run['condition'] for run in checkpoint['runs']}):
            runs = [run for run in checkpoint['runs'] if run['condition'] == condition]
            means = [np.mean([run['quality'][key] for run in runs]) for key in
                     ('repeat_fraction', 'repeated_4gram_fraction', 'pitch_range', 'mean_absolute_jump', 'large_leap_fraction')]
            response = np.mean([run['probes'][0]['tv_vs_base_same_history'] for run in runs]) * 100
            rows.append('<tr><td>' + html.escape(checkpoint['path']) + '</td><td>' + html.escape(condition)
                        + f'</td><td>{response:.2f}%</td>' + ''.join(f'<td>{value:.3f}</td>' for value in means) + '</tr>')
    clips = []
    for checkpoint in report['checkpoints']:
        for run in checkpoint['runs']:
            identity = f"{Path(checkpoint['path']).name} · {run['sample']} · seed {run['seed']} · {run['condition']}"
            player = f'<audio controls preload="none" src="{run["audio"]}"></audio>' if run['audio'] else '<p>Open the MIDI in your piano player.</p>'
            clips.append((run['id'], f'<section data-pair="{run["pair"]}"><h3>{run["id"]} · pair {run["pair"]}</h3>{player}'
                          f'<p><a href="{run["midi"]}">Full MIDI</a> · <a href="{run["clip_midi"]}">Listening MIDI</a></p>'
                          f'<label>Listening notes: <textarea data-id="{run["id"]}" placeholder="Continuity, coherence, repetition, register; then reveal the request to judge steering"></textarea></label>'
                          f'<details><summary>Reveal checkpoint and requested chords</summary><p>{html.escape(identity)}</p>'
                          f'<p>{html.escape(json.dumps(run["events"]))}</p>'
                          f'<pre>{html.escape(json.dumps(run["quality"], indent=2))}</pre></details></section>'))
    checks = html.escape(json.dumps(report['checks'], indent=2))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Chord guidance comparison</title>
<style>body{max-width:1150px;margin:36px auto;padding:0 24px;font:16px/1.5 system-ui;color:#223;background:#fafafa}section{padding:18px;margin:18px 0;background:white;border:1px solid #ddd;border-radius:10px}audio{width:100%}textarea{display:block;width:95%;min-height:65px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:7px}pre{white-space:pre-wrap}button{padding:10px}</style>
<h1>Chord guidance: paired listening</h1>
<p>Each pair shares a primer, source timing, temperature and random seed. Only pitch is generated. Clips include the last 24 primer notes; full MIDI files include the complete primer. Listen to the numbered clips before revealing their conditions.</p>
<p>Judge continuity, musical coherence, repetition and register first. Then reveal the request and judge whether it steers the continuation. After release, the model uses its original base on the pitches actually generated; it need not return to the trajectory that was free from the beginning.</p>
<p>Interval jumps include both hands and serialized chord notes. These measurements are descriptive, not musical-quality scores. No chord-tone percentage, inferred harmony label or loss is used as a success criterion. Listening assessment is pending.</p>
<p>Checkpoint comparisons are observational unless initialization and training budget were matched. Checkpoint epochs, updates and available training seeds are recorded in report.json.</p>
<p>Initial response measures how much probability mass moves from the free prediction for the identical prefix. It measures reaction to guidance, not whether that reaction is musically appropriate.</p>
<label>Compare one prefix and seed: <select id="pair" onchange="filterPairs()">PAIRS</select></label>
<button onclick="downloadRatings()">Download listening notes</button>
<details><summary>Implementation verification</summary><pre>CHECKS</pre></details>
<details><summary>Reveal aggregate measurements</summary><table><tr><th>Checkpoint</th><th>Condition</th><th>Initial response</th><th>Repeated notes</th><th>Repeated 4-grams</th><th>Range</th><th>Mean jump</th><th>Jumps &gt; octave</th></tr>ROWS</table></details>
CLIPS
<script>function filterPairs(){let pair=document.getElementById('pair').value;document.querySelectorAll('section').forEach(x=>x.hidden=pair!=='all'&&x.dataset.pair!==pair)}
function downloadRatings(){let notes=Array.from(document.querySelectorAll('textarea')).map(x=>({id:x.dataset.id,notes:x.value}));let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(notes,null,2)],{type:'application/json'}));a.download='listening_notes.json';a.click();URL.revokeObjectURL(a.href)}filterPairs()</script></html>'''
    pairs = {run['pair']: f'{Path(run["sample"]).stem} · seed {run["seed"]}'
             for checkpoint in report['checkpoints'] for run in checkpoint['runs']}
    options = ''.join(f'<option value="{key}">{html.escape(value)}</option>' for key, value in sorted(pairs.items()))
    page = page.replace('PAIRS', options + '<option value="all">All examples</option>')
    page = page.replace('CHECKS', checks).replace('ROWS', ''.join(rows)).replace('CLIPS', ''.join(clip for _, clip in sorted(clips)))
    (output / 'listening.html').write_text(page)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, nargs='+', default=[ROOT / 'save_models/D_Chord_Joker_mixed_history_latest_best.pth'])
    parser.add_argument('--base-checkpoint', type=Path, help='Override relocated original base checkpoint')
    parser.add_argument('--data', type=Path, default=ROOT / 'Training-Data/giantmidi_full_w_chords_test.pickle')
    parser.add_argument('--data-pieces', type=int, default=8, help='Complete pieces spread across the pickle; 0 checks all')
    parser.add_argument('--sample', type=Path, nargs='+', default=[ROOT / sample_midi_paths[index] for index in (0, 4)])
    parser.add_argument('--chord-keys', default='1,4,5,shift+2', help='Keys in each primer\'s existing fixed chord palette')
    parser.add_argument('--primer-length', type=int, default=300,
                        help='Maximum primer notes; shortened if necessary to leave source timing for the continuation')
    parser.add_argument('--context-length', type=int, default=300)
    parser.add_argument('--notes', type=int, default=64)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--output', type=Path, default=ROOT / 'out/chord_verification')
    parser.add_argument('--soundfont', type=Path, default=ROOT / 'piano.sf2')
    parser.add_argument('--no-audio', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if (args.notes < 12 or args.seeds < 1 or args.data_pieces < 0 or args.threads < 1
            or not 1 <= args.primer_length <= args.context_length
            or not np.isfinite(args.temperature) or args.temperature <= 0):
        parser.error('Require notes >= 12, positive seeds/threads/temperature, data-pieces >= 0, and primer <= context')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a new/empty output directory to preserve earlier comparisons')
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    print('Checking exported blocks, controls and training masks...', flush=True)
    checks = {'controlled_data': data_fidelity(fixture(), 64, 96),
              'controls': check_controls(), 'masking': check_masking()}
    data = np.asarray(Any_Pickle_File_Reader(str(args.data)), dtype=np.int16)
    events = data.reshape(-1, 5)
    boundaries = np.flatnonzero((events == np.array([126, 126, 0, 0, 0])).all(axis=-1))
    edges = np.unique(np.r_[0, boundaries, len(events)])
    eligible = np.flatnonzero(np.diff(edges) > 512)
    if args.data_pieces:
        chosen = eligible[np.unique(np.linspace(0, len(eligible) - 1, min(args.data_pieces, len(eligible)), dtype=int))]
        data = np.concatenate([events[edges[index]:edges[index + 1]] for index in chosen]).reshape(-1)
    else:
        chosen = eligible
    checks['exported_data'] = data_fidelity(data, 512, 128)
    checks['exported_data']['piece_indices'] = chosen.tolist()
    checks['exported_data']['pickle'] = str(args.data.resolve())
    checks['exported_data']['sha256'] = sha256(args.data)
    del data, events
    gc.collect()
    report = {'checks': checks, 'protocol': vars(args).copy(), 'checkpoints': [],
              'completed': False, 'listening_assessment': 'pending'}
    report['protocol'] = {key: str(value) if isinstance(value, Path) else
                          [str(item) for item in value] if isinstance(value, list) else value
                          for key, value in report['protocol'].items()}
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
        sources.append((path, source, bank, primer_length))
    audio_available = bool(shutil.which('fluidsynth') and args.soundfont.is_file() and not args.no_audio)
    report['audio_rendered'] = audio_available
    cases = schedules(keys, args.notes)
    identifiers = list(range(len(args.checkpoint) * len(sources) * args.seeds * len(cases)))
    random.Random(args.seed).shuffle(identifiers)
    next_id = 0
    for checkpoint in args.checkpoint:
        model, base, metadata, base_path = load_pair(checkpoint, args.base_checkpoint, torch.device(args.device))
        if args.context_length > model.max_seq_len:
            parser.error('context-length exceeds checkpoint sequence length')
        result = {'path': str(checkpoint.resolve()), 'sha256': sha256(checkpoint), 'metadata': metadata,
                  'base_path': str(base_path.resolve()), 'base_sha256': sha256(base_path), 'runs': []}
        report['checkpoints'].append(result)
        for sample_index, (path, source, bank, primer_length) in enumerate(sources):
            for offset in range(args.seeds):
                seed = args.seed + offset
                print(f'{checkpoint.name}: {path.name}, seed {seed}', flush=True)
                pitches, requests, probes = generate(model, base, source['pitch'][:primer_length], bank,
                                                     cases, args.notes, args.context_length, seed,
                                                     args.temperature, torch.device(args.device))
                for index, case in enumerate(cases):
                    name = f'clip_{identifiers[next_id]:04d}'
                    next_id += 1
                    midi, clip, audio_path = (args.output / (name + suffix) for suffix in ('.mid', '_listen.mid', '.wav'))
                    sequence = pitches[index].tolist()
                    write_midi(midi, source, sequence, primer_length, requests[index])
                    write_midi(clip, source, sequence, primer_length, requests[index], lead_in=24)
                    if audio_available:
                        render_audio(clip, args.soundfont, audio_path)
                    transitions = [step for step, _, _ in case['events']]
                    run = {'id': name, 'pair': f'{sample_index + 1}-{offset + 1}', 'sample': str(path), 'seed': seed,
                           'primer_length': primer_length,
                           'condition': case['name'], 'events': case['events'], 'requests': requests[index],
                           'pitches': sequence, 'quality': quality_metrics(sequence, primer_length, transitions),
                           'source_quality': quality_metrics(source['pitch'], primer_length, transitions),
                           'probes': probes[index],
                           'disagreement_with_paired_free': float((pitches[index, primer_length:] != pitches[0, primer_length:]).float().mean()),
                           'midi': midi.name, 'clip_midi': clip.name, 'audio': audio_path.name if audio_available else None}
                    result['runs'].append(run)
                write_report(args.output, report)
        del model, base
        gc.collect()
    report['completed'] = True
    write_report(args.output, report)
    print('Saved comparison:', args.output / 'listening.html', flush=True)


if __name__ == '__main__':
    main()
