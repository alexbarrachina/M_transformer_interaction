"""Offline Satie timing diagnostic. Run `python test/evaluate_timing.py --help`.

Uses the existing decoder and shared timing features; no training takes place.
The CLI, MIDI I/O and plotting run on the host. Tensor scoring helpers are
scripted; this does not change how the existing model is loaded or executed.
"""

import argparse
import base64
import hashlib
import html
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from midiUtils import midi_to_dict, score2midi
from model_loader import load_model
from models import get_model_hparams
from timing import TIME_UNIT, timing_features
from x_transformer import AE_style_jokerParam


CHECKPOINT = ROOT / 'save_models/AE_style_jokerParam_dtime_tester_v1_437_eps_1384719_steps_1.1275_loss_0.996_val_loss_0.9017_acc.pth'
SAMPLE = ROOT / 'samples/Satie_Gymnopedie_No1_no_melody.midi'


class Pattern(NamedTuple):
    pitch: Tensor
    onset: Tensor
    duration: Tensor
    group: Tensor
    role: Tensor  # 0: isolated note; 1: first chord note; 2: later chord note
    primer: int


@torch.jit.script
def isolated_probability(probs: Tensor, split: float, high: bool) -> Tensor:
    pitches = torch.arange(probs.size(-1), device=probs.device)
    selected = pitches > split if high else pitches < split
    return (probs * selected.to(probs.dtype)).sum(-1)


@torch.jit.script
def role_scores(isolated_prob: Tensor, roles: Tensor) -> Dict[str, float]:
    isolated = isolated_prob[roles == 0].mean()
    chord = (1.0 - isolated_prob[roles != 0]).mean()
    return {
        'balanced': float((isolated + chord) / 2.0),
        'isolated': float(isolated),
        'chord_start': float((1.0 - isolated_prob[roles == 1]).mean()),
        'chord_inner': float((1.0 - isolated_prob[roles == 2]).mean()),
    }


@torch.jit.script
def sample_with_uniform(probs: Tensor, uniforms: Tensor) -> Tensor:
    # Use identical random numbers for each on/off pair. This removes an
    # avoidable source of differences while keeping ordinary pitch sampling.
    return (probs.cumsum(-1) < uniforms.unsqueeze(-1)).sum(-1).clamp_max(probs.size(-1) - 1)


@torch.jit.script
def cycle_separation(pitches: Tensor, groups: Tensor, roles: Tensor, high: bool) -> Dict[str, float]:
    # A second score tolerates global register drift: is the isolated note
    # below every following chord note (or above them in the reversed test)?
    isolated = torch.where(roles == 0)[0]
    chord = groups.unsqueeze(0) == (groups[isolated] + 1).unsqueeze(1)
    gaps = (pitches.unsqueeze(0) - pitches[isolated].unsqueeze(1)).float()
    if high:
        gaps = -gaps
    smallest_gap = gaps.masked_fill(~chord, float('inf')).min(-1).values
    mean_gap = (gaps * chord).sum(-1) / chord.sum(-1)
    return {'ordered_cycles': float((smallest_gap > 0).float().mean()),
            'mean_register_gap_semitones': float(mean_gap.mean())}


def read_pattern(path: Path, primer_cycles: int, cycles: int) -> Pattern:
    notes, _ = midi_to_dict(str(path))
    pitch = torch.tensor(notes['pitch'], dtype=torch.long)
    intervals = torch.tensor(notes['dtime'], dtype=torch.float32) * TIME_UNIT
    duration = torch.tensor(notes['dur'], dtype=torch.float32) * TIME_UNIT
    starts = torch.cat([torch.tensor([0]), torch.where(intervals > 2 * TIME_UNIT + 1e-6)[0]])
    ends = torch.cat([starts[1:], torch.tensor([pitch.numel()])])
    sizes = ends - starts
    # Stop at the first passage that is no longer singleton + chord. Later
    # cadential chords include bass pitches and do not share this binary task.
    available = 0
    for i in range(0, len(sizes) - 1, 2):
        if sizes[i] != 1 or sizes[i + 1] < 2:
            break
        available += 1
    requested = primer_cycles + cycles
    if requested > available:
        raise ValueError(f'Requested {requested} cycles, but the opening contains {available} alternating cycles')
    end = int(ends[2 * requested - 1])
    group = torch.repeat_interleave(torch.arange(2 * requested), sizes[:2 * requested])
    positions = torch.arange(end) - starts[group]
    roles = torch.where(group % 2 == 0, 0, torch.where(positions == 0, 1, 2))
    primer = int(starts[2 * primer_cycles])
    print(f'Source: {len(pitch)} notes, {available} alternating cycles; using {primer} primer + {end - primer} target notes', flush=True)
    return Pattern(pitch[:end], intervals.cumsum(0)[:end], duration[:end], group, roles, primer)


def spread_chords(pattern: Pattern, spacing: float) -> Pattern:
    starts = torch.cat([torch.tensor([0]), torch.where(torch.diff(pattern.group) != 0)[0] + 1])
    ends = torch.cat([starts[1:], torch.tensor([len(pattern.pitch)])])
    group_times = pattern.onset[starts]
    next_times = torch.cat([group_times[1:], group_times[-1:] + 1.0])
    # Preserve every group's first onset. Cap the spread to leave a clear gap
    # before the next group; labels remain those of the original grouping.
    step = torch.minimum(torch.full_like(group_times, spacing),
                         0.6 * (next_times - group_times) / (ends - starts - 1).clamp_min(1))
    positions = torch.arange(len(pattern.pitch)) - starts[pattern.group]
    onsets = group_times[pattern.group] + positions * step[pattern.group]
    durations = (pattern.onset + pattern.duration - onsets).clamp_min(0.064)
    return pattern._replace(onset=onsets, duration=durations)


def variable_chords(pattern: Pattern, seed: int) -> Pattern:
    rng = torch.Generator().manual_seed(seed)
    rows: List[Tensor] = []
    onsets: List[Tensor] = []
    durations: List[Tensor] = []
    groups: List[Tensor] = []
    roles: List[Tensor] = []
    primer_group = int(pattern.group[pattern.primer])
    primer = 0
    # This small host-side loop constructs ragged MIDI groups; model batches
    # and probability calculations remain tensor operations.
    for group in range(int(pattern.group[-1]) + 1):
        indices = torch.where(pattern.group == group)[0]
        pitches = pattern.pitch[indices]
        count = 1 if group % 2 == 0 else int(torch.randint(2, 5, (1,), generator=rng))
        if count <= len(pitches):
            selected = torch.randperm(len(pitches), generator=rng)[:count].sort().values
            pitches = pitches[selected]
        else:
            # Add octave doublings, keeping pitch classes and existing order.
            extra = pitches[:count - len(pitches)] + 12
            pitches = torch.cat([pitches, extra])
        rows.append(pitches)
        onsets.append(pattern.onset[indices[0]] + torch.arange(count) * TIME_UNIT)
        durations.append(pattern.duration[indices[0]].repeat(count))
        groups.append(torch.full((count,), group, dtype=torch.long))
        role = torch.full((count,), 2, dtype=torch.long)
        role[0] = 0 if group % 2 == 0 else 1
        roles.append(role)
        if group < primer_group:
            primer += count
    return Pattern(torch.cat(rows), torch.cat(onsets), torch.cat(durations),
                   torch.cat(groups), torch.cat(roles), primer)


def make_patterns(base: Pattern) -> Dict[str, Pattern]:
    reversed_pitch = base.pitch + torch.where(base.role == 0, 36, -12)
    reversed_pattern = base._replace(pitch=reversed_pitch)
    factors = torch.linspace(0.6, 1.6, int(base.group[-1]) + 1)[base.group]
    intervals = torch.cat([base.onset[:1], torch.diff(base.onset)])
    return {
        'original': base,
        'reversed_registers': reversed_pattern,
        'spread_80ms': spread_chords(base, 0.08),
        'spread_160ms': spread_chords(base, 0.16),
        'tempo_2x': base._replace(onset=base.onset * 0.5, duration=base.duration * 0.5),
        'tempo_half': base._replace(onset=base.onset * 2, duration=base.duration * 2),
        'tempo_ramp': base._replace(onset=(intervals * factors).cumsum(0), duration=base.duration * factors),
        'variable_chord_sizes': variable_chords(base, 7919),
        'reversed_spread': spread_chords(reversed_pattern, 0.16),
    }


def encode_timing(pattern: Pattern, shuffled: bool = False) -> Tuple[Tensor, Tensor]:
    intervals = torch.cat([pattern.onset[:1], torch.diff(pattern.onset)])
    if shuffled:
        # Destroy the rhythm/pitch relation while preserving the interval set.
        rng = torch.Generator().manual_seed(7919)
        indices = torch.arange(1, len(intervals))
        indices = indices[indices != pattern.primer]
        intervals[indices] = intervals[indices[torch.randperm(len(indices), generator=rng)]]
    values = intervals.tolist()
    values[0] = None
    # Match live entry: no clock connects the recorded primer to the first
    # performed note. Exclude that first target from all reported scores.
    values[pattern.primer] = None
    return timing_features(values)


def decoder_inputs(pitches: Tensor, buttons: Tensor, joker: Tensor, features: Tensor,
                   valid: Tensor, start: int, end: int) -> Dict[str, Tensor]:
    # Position t pairs previous pitch[t-1] with current button/time[t].
    return {'pitch': pitches[:, start:end], 'button': buttons[:, start + 1:end + 1],
            'joker_mask': joker[:, start + 1:end + 1],
            'timing_features': features[:, start + 1:end + 1],
            'timing_mask': valid[:, start + 1:end + 1]}


def known_history_probs(model: AE_style_jokerParam, pitches: Tensor, buttons: Tensor,
                        joker: Tensor, features: Tensor, valid: Tensor, style: Tensor,
                        style_mask: Tensor, primer: int, context_length: int) -> Tensor:
    end = min(pitches.size(1) - 1, context_length)
    logits = model.decoder(decoder_inputs(pitches, buttons, joker, features, valid, 0, end),
                           style_context=style, style_context_mask=style_mask)
    outputs = [logits[:, primer - 1:]]
    # Rebuild only when cropping is needed, exactly as in uncached live use.
    for target in range(end + 1, pitches.size(1)):
        inputs = decoder_inputs(pitches, buttons, joker, features, valid, target - context_length, target)
        outputs.append(model.decoder(inputs, style_context=style, style_context_mask=style_mask)[:, -1:])
    return torch.cat(outputs, dim=1).softmax(-1)


def generate_pairs(model: AE_style_jokerParam, pattern: Pattern, buttons: Tensor,
                   joker: Tensor, features: Tensor, valid: Tensor, style: Tensor,
                   style_mask: Tensor, seeds: List[int], context_length: int,
                   temperature: float) -> Tensor:
    count = len(seeds)
    device = features.device
    pitches = pattern.pitch.to(device).unsqueeze(0).repeat(2 * count, 1)
    pitches[:, pattern.primer:] = 0  # Future reference pitches never enter generation.
    buttons = buttons.expand(2 * count, -1)
    joker = joker.expand(2 * count, -1)
    features = features.expand(2 * count, -1, -1)
    valid = valid.expand(2 * count, -1).clone()
    valid[count:] = False
    style = style.expand(2 * count, -1, -1)
    style_mask = style_mask.expand(2 * count, -1)
    draws = torch.stack([torch.rand(len(pattern.pitch) - pattern.primer,
                                   generator=torch.Generator().manual_seed(seed)) for seed in seeds]).to(device)
    for step, target in enumerate(range(pattern.primer, len(pattern.pitch))):
        start = max(0, target - context_length)
        inputs = decoder_inputs(pitches, buttons, joker, features, valid, start, target)
        logits = model.decoder(inputs, style_context=style, style_context_mask=style_mask)[:, -1]
        pitches[:, target] = sample_with_uniform((logits / temperature).softmax(-1), draws[:, step].repeat(2))
    return pitches.cpu()


def write_midi(path: Path, pitch: Tensor, pattern: Pattern) -> None:
    events = [['set_tempo', 0, 500000]]
    for note, onset, duration in zip(pitch.tolist(), pattern.onset.tolist(), pattern.duration.tolist()):
        # 120 BPM and 480 ticks/beat means 960 ticks/second.
        events.append(['note', round(onset * 960), max(1, round(duration * 960)), 0, note, 85])
    path.write_bytes(score2midi([480, events]))


def plot_pattern(path: Path, name: str, pattern: Pattern, generated: Tensor,
                 isolated_probs: Tensor, split: float, seeds: List[int]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(13, 8), constrained_layout=True, sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1, 0.85]})
    notes = [pattern.pitch, generated[0], generated[len(seeds)]]
    plotted_pitches = torch.stack(notes)
    pitch_limits = (int(plotted_pitches.min()) - 4, int(plotted_pitches.max()) + 4)
    labels = ['Reference', f'Timing ON · seed {seeds[0]}', f'Timing OFF · seed {seeds[0]}']
    colors = ['#c6552d' if role == 0 else '#247c9b' for role in pattern.role.tolist()]
    for ax, pitches, label in zip(axes[:3], notes, labels):
        ax.hlines(pitches.tolist(), pattern.onset.tolist(),
                  (pattern.onset + pattern.duration.clamp_max(0.45)).tolist(), colors=colors, linewidth=2)
        ax.axhline(split, color='#999999', linestyle=':', linewidth=0.8)
        ax.set_ylabel('MIDI pitch')
        ax.set_ylim(*pitch_limits)
        ax.set_title(label, loc='left', fontsize=10)
    target_times = pattern.onset[pattern.primer:]
    axes[3].plot(target_times, isolated_probs[0], color='#247c9b', label='Timing ON')
    axes[3].plot(target_times, isolated_probs[1], color='#888888', label='Timing OFF', alpha=0.8)
    axes[3].scatter(target_times, (pattern.role[pattern.primer:] == 0).float(), s=9,
                    color='#c6552d', label='Expected isolated role')
    axes[3].set_ylabel('P(isolated register)')
    axes[3].set_ylim(-0.08, 1.08)
    axes[3].legend(loc='upper right', ncol=3, fontsize=8)
    axes[3].set_xlabel('Seconds · same timing and durations for ON and OFF')
    for ax in axes:
        ax.axvline(float(pattern.onset[pattern.primer]), color='#333333', linestyle='--', linewidth=0.8)
        ax.grid(axis='y', alpha=0.15)
    fig.suptitle(f'{name.replace("_", " ")} · orange = isolated event, blue = chord event', fontsize=13)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(output: Path, results: Dict, metadata: Dict) -> None:
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    y = torch.arange(len(names)).numpy()
    for index, (mode, color) in enumerate([('on', '#247c9b'), ('off', '#999999'), ('shuffled', '#c6552d')]):
        values = [100 * results[name]['known_history'][mode]['balanced'] for name in names]
        axes[0].barh(y + (index - 1) * 0.24, values, height=0.22, color=color, label=mode.upper())
    for index, (mode, color) in enumerate([('on', '#247c9b'), ('off', '#999999')]):
        runs = torch.tensor([[row['balanced'] for row in results[name]['generated'][mode]] for name in names])
        axes[1].barh(y + (index - 0.5) * 0.32, runs.mean(1).numpy() * 100,
                     xerr=runs.std(1, unbiased=False).numpy() * 100, height=0.29, color=color, label=mode.upper())
    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels([name.replace('_', ' ') for name in names])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.axvline(50, color='#333333', linestyle=':', linewidth=1)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=3, fontsize=8)
        ax.grid(axis='x', alpha=0.15)
    axes[0].set_title('Known pitch history: probability of correct register', pad=30)
    axes[1].set_title('Generated pitches: correct register · mean ± seed SD', pad=30)
    fig.savefig(output / 'summary.png', dpi=150)
    plt.close(fig)
    rows = []
    details = []
    for name, result in results.items():
        known = result['known_history']
        free = result['generated']
        on = sum(row['balanced'] for row in free['on']) / len(free['on'])
        off = sum(row['balanced'] for row in free['off']) / len(free['off'])
        ordered_on = sum(row['ordered_cycles'] for row in free['on']) / len(free['on'])
        ordered_off = sum(row['ordered_cycles'] for row in free['off']) / len(free['off'])
        rows.append(f'<tr><td>{name}</td><td>{known["on"]["balanced"]:.1%}</td>'
                    f'<td>{known["off"]["balanced"]:.1%}</td><td>{known["shuffled"]["balanced"]:.1%}</td>'
                    f'<td>{on:.1%}</td><td>{off:.1%}</td><td>{ordered_on:.1%}</td><td>{ordered_off:.1%}</td></tr>')
        data = base64.b64encode((output / f'{name}.png').read_bytes()).decode('ascii')
        seed = metadata['seeds'][0]
        links = ' · '.join(f'<a href="midi/{name}_{suffix}.mid">{label}</a>' for suffix, label in
                           [('reference', 'Reference MIDI'), (f'on_seed{seed}', 'Timing ON MIDI'), (f'off_seed{seed}', 'Timing OFF MIDI')])
        details.append(f'<h2>{name.replace("_", " ")}</h2><p>{links}</p><img src="data:image/png;base64,{data}">')
    summary = base64.b64encode((output / 'summary.png').read_bytes()).decode('ascii')
    finding = ''
    if 'original' in results:
        original = results['original']
        known = original['known_history']
        means = {mode: sum(row['balanced'] for row in original['generated'][mode]) / len(original['generated'][mode])
                 for mode in ('on', 'off')}
        finding = (f'<p><strong>Original pattern:</strong> with correct preceding pitches, the register score is '
                   f'{known["on"]["balanced"]:.1%} with timing and {known["off"]["balanced"]:.1%} without. '
                   f'When the model uses its own generated pitches, the scores are {means["on"]:.1%} and {means["off"]:.1%}. '
                   'The second test includes errors accumulating through the generated history.</p>')
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Satie timing diagnostic</title>
<style>body{{font:16px/1.5 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#203039}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}code{{word-break:break-all}}a{{color:#247c9b}}</style>
<h1>Satie: does timing identify isolated notes and chords?</h1>
<p>Checkpoint: <code>{html.escape(metadata['checkpoint'])}</code><br>Primer: {metadata['primer_cycles']} cycles; continuation: {metadata['target_cycles']} cycles. Seeds: {metadata['seeds']}; temperature: {metadata['temperature']}; device: {metadata['device']}.</p>
{finding}
<p>Register scores give equal weight to isolated notes and chord notes. 50% is the score of always choosing one register. Known-history scores measure probability mass at temperature 1; generation scores measure sampled pitches. Error bars show variation across seeds, not a confidence interval across musical pieces.</p>
<p>The additional ordered-cycles score measures whether each isolated note is below every note of its following chord, or above every note in the reversed test. It tolerates a shift of the whole performance to another register. The 50% reference line does not apply to this stricter cycle score.</p>
<p>Only the primer supplies style and guided buttons. All subsequent notes use Joker; future reference pitches are available only in the known-history test. Current timing is paired with the pitch being predicted. The first performed interval is unknown, matching live entry, and its note is excluded from scoring. Timing OFF masks the same trained checkpoint, not the pitch-only base model.</p>
<p>Reversed registers transpose isolated notes +36 semitones and chords −12, preserving pitch classes. Spread variants keep group starts and use 80/160 ms spacing, capped at 60% of the next group gap. Tempo changes transform primer and continuation together; the ramp scales intervals gradually from 0.6× to 1.6×. Variable chords contain 2–4 notes. Shuffled timing is a diagnostic negative control, not a valid alternative reference performance.</p>
<p>The source is quantized at 32 ms using the application's MIDI reader. Groups are separated by gaps above 64 ms, before any transformations. The score uses a pitch split separating reference registers; that split and the role labels are never model inputs. The opening alternating section is selected before the cadential passage containing consecutive chords. These results describe this one musical example, not general model quality.</p>
<table><tr><th>Variation</th><th>Known ON</th><th>Known OFF</th><th>Known shuffled</th><th>Generated ON</th><th>Generated OFF</th><th>Ordered ON</th><th>Ordered OFF</th></tr>{''.join(rows)}</table>
<img src="data:image/png;base64,{summary}">
<p>In the plots below, the dashed line ends the primer. Orange marks intended isolated events, even if their generated pitch lands in the wrong register. The bottom panel uses known preceding pitches. Each ON/OFF MIDI pair has identical timing and note durations. All seed outputs are in the midi folder; detailed probabilities and per-role scores are in results.json.</p>
{''.join(details)}</html>'''
    (output / 'report.html').write_text(report)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--sample', type=Path, default=SAMPLE)
    parser.add_argument('--output', type=Path, default=ROOT / 'out/timing_satie_1384719')
    parser.add_argument('--primer-cycles', type=int, default=17)
    parser.add_argument('--cycles', type=int, default=10, help='Bass + chord cycles after the primer')
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    parser.add_argument('--context-length', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--variants', nargs='+', help='Optional subset of the nine named variations')
    args = parser.parse_args()
    if min(args.primer_cycles, args.cycles, args.context_length, args.threads) < 1 or args.temperature <= 0:
        parser.error('Cycles, context length, threads and temperature must be positive')
    if args.cycles < 2:
        parser.error('At least two target cycles are needed after excluding the first target note')
    torch.set_num_threads(args.threads)
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    base = read_pattern(args.sample, args.primer_cycles, args.cycles)
    patterns = make_patterns(base)
    if args.variants:
        if any(name not in patterns for name in args.variants):
            parser.error(f'Available variations: {", ".join(patterns)}')
        patterns = {name: patterns[name] for name in args.variants}
    cfg = get_model_hparams('AE_style_jokerParam_dtime_tester_v1')
    cfg['ckpt_file_name'] = str(args.checkpoint)
    model = load_model(cfg=cfg, compile_mode='none').to(device).eval()
    if args.context_length > model.decoder.max_seq_len or base.primer > args.context_length:
        parser.error('Context must fit the model and include the primer')
    output = args.output
    (output / 'midi').mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with args.checkpoint.open('rb') as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b''):
            digest.update(chunk)
    metadata = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest.hexdigest(),
                    sample=str(args.sample.resolve()), sample_sha256=hashlib.sha256(args.sample.read_bytes()).hexdigest(),
                    seeds=args.seeds, temperature=args.temperature, device=device, torch_version=torch.__version__,
                    primer_cycles=args.primer_cycles, target_cycles=args.cycles, context_length=args.context_length,
                    cache=False, source_group_gap_seconds=0.064, excluded_first_target=True)
    results: Dict = {}
    masked_references: Dict[int, Tensor] = {}
    for name, pattern in patterns.items():
        started = time.monotonic()
        print(f'{name}: known-history comparison and {len(args.seeds)} paired generations ({len(pattern.pitch) - pattern.primer} notes each)', flush=True)
        if pattern.primer > args.context_length:
            raise ValueError('Variant primer exceeds context length')
        pitch = pattern.pitch.to(device).unsqueeze(0)
        primer = pitch[:, :pattern.primer]
        style_mask = torch.ones_like(primer, dtype=torch.bool)
        style = model.encode_style(primer, style_mask)
        discrete = model.real_to_discrete(model.encoder({'pitch': primer}))
        all_buttons = torch.full_like(pitch, model.joker_button_idx)
        all_buttons[:, :pattern.primer] = discrete
        buttons, joker = model._buttons_discrete_to_inputs(all_buttons)
        features, valid = encode_timing(pattern)
        shuffled_features, shuffled_valid = encode_timing(pattern, shuffled=True)
        feature_batch = torch.stack([features, features, shuffled_features]).to(device)
        valid_batch = torch.stack([valid, torch.zeros_like(valid), shuffled_valid]).to(device)
        probabilities = known_history_probs(model, pitch.expand(3, -1), buttons.expand(3, -1),
                                             joker.expand(3, -1), feature_batch, valid_batch,
                                             style.expand(3, -1, -1), style_mask.expand(3, -1),
                                             pattern.primer, args.context_length).cpu()
        if not results:
            # Removing all future notes must leave the chosen prediction intact.
            target_index = min(pattern.primer + 3, args.context_length)
            inputs = decoder_inputs(pitch, buttons, joker, feature_batch[:1], valid_batch[:1], 0, target_index)
            causal_probs = model.decoder(inputs, style_context=style, style_context_mask=style_mask)[:, -1].softmax(-1).cpu()
            torch.testing.assert_close(causal_probs[0], probabilities[0, target_index - pattern.primer], rtol=1e-4, atol=1e-5)
        high = 'reversed' in name
        isolated_notes = pattern.pitch[pattern.role == 0]
        chord_notes = pattern.pitch[pattern.role != 0]
        low_edge = int(chord_notes.max()) if high else int(isolated_notes.max())
        high_edge = int(isolated_notes.min()) if high else int(chord_notes.min())
        if low_edge >= high_edge:
            raise ValueError(f'{name}: reference registers overlap; a binary register score would be misleading')
        split = (low_edge + high_edge) / 2.0
        iso_probs = isolated_probability(probabilities, split, high)
        roles = pattern.role[pattern.primer + 1:]
        target = pattern.pitch[pattern.primer + 1:]
        known: Dict = {}
        for index, mode in enumerate(['on', 'off', 'shuffled']):
            scores = role_scores(iso_probs[index, 1:], roles)
            true_prob = probabilities[index, 1:].gather(-1, target.unsqueeze(-1)).squeeze(-1)
            scores['pitch_nll'] = float(-true_prob.clamp_min(1e-12).log().mean())
            scores['pitch_accuracy'] = float((probabilities[index, 1:].argmax(-1) == target).float().mean())
            known[mode] = scores
        generated = generate_pairs(model, pattern, buttons, joker, feature_batch[:1], valid_batch[:1],
                                   style, style_mask, args.seeds, args.context_length, args.temperature)
        free: Dict = {'on': [], 'off': []}
        for index, mode in enumerate(['on', 'off']):
            for run, seed in enumerate(args.seeds):
                notes = generated[index * len(args.seeds) + run]
                is_isolated_register = notes[pattern.primer + 1:] > split if high else notes[pattern.primer + 1:] < split
                scores = role_scores(is_isolated_register.float(), roles)
                scores.update(cycle_separation(notes[pattern.primer + 1:], pattern.group[pattern.primer + 1:], roles, high))
                scores['seed'] = seed
                free[mode].append(scores)
                write_midi(output / 'midi' / f'{name}_{mode}_seed{seed}.mid', notes, pattern)
        # With the same pitches, buttons and random draws, masked timing must
        # make tempo/spread variants identical. This catches alignment mistakes.
        same_pitch = torch.equal(pattern.pitch, base.pitch) and pattern.primer == base.primer
        if same_pitch:
            off = generated[len(args.seeds):]
            if 0 in masked_references:
                torch.testing.assert_close(off, masked_references[0], rtol=0, atol=0)
            else:
                masked_references[0] = off
        write_midi(output / 'midi' / f'{name}_reference.mid', pattern.pitch, pattern)
        plot_pattern(output / f'{name}.png', name, pattern, generated, iso_probs, split, args.seeds)
        results[name] = dict(known_history=known, generated=free, split_pitch=split, isolated_register_high=high,
                             primer_notes=pattern.primer, scored_notes=len(roles),
                             mean_total_variation=float(0.5 * (probabilities[0, 1:] - probabilities[1, 1:]).abs().sum(-1).mean()),
                             reference=dict(pitch=pattern.pitch.tolist(), onset=pattern.onset.tolist(), duration=pattern.duration.tolist(),
                                            role=pattern.role.tolist(), group=pattern.group.tolist()),
                             isolated_probabilities={mode: iso_probs[i].tolist() for i, mode in enumerate(['on', 'off', 'shuffled'])},
                             generated_pitches=generated.tolist(), seconds=time.monotonic() - started)
        (output / 'results.json').write_text(json.dumps(dict(metadata=metadata, variations=results), indent=2, allow_nan=False))
        print(f'  known ON/OFF: {known["on"]["balanced"]:.1%} / {known["off"]["balanced"]:.1%}; finished in {time.monotonic() - started:.1f}s', flush=True)
    write_report(output, results, metadata)
    shutil.make_archive(str(output), 'zip', root_dir=str(output))
    print(f'Report: {output / "report.html"}', flush=True)


if __name__ == '__main__':
    main()
