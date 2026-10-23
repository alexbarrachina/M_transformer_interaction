"""Compare the chord conditioning strength of AE_style_jokerParam_chords checkpoints on unseen MIDI
files: test/eval_chord_strength.py for the style joker.

The style joker also needs buttons and a style. At every tested chord change an automatic player
supplies them, in two button modes:
  guided - for every generated note the player presses the button that the frozen button encoder
           gives the pianist's real note there (an oracle: the contour of the real continuation).
           A requested chord other than the real one then competes with a contour that the pianist
           played over the real chord, e.g. a distant chord. Primer and continuation are encoded
           together, as in training (buttons are relative to the encoded passage).
  joker  - the joker is held from the first generated note on (the model chooses every pitch). The
           primer is encoded alone, as the interaction encodes its seed.
The style is the style_seq_len notes next to primer + continuation in the same piece, on the longer
side, as in training (StyleChordSamplerDataset).

Everything else follows eval_chord_strength.py: real / dominant / subdominant / distant chords held
from the first generated note on, a free run, chord-tone mass CTM@N, strength = CTM_cond - CTM_free,
per-note curves and paired comparisons between checkpoints that share the frozen base. Results and
summary.json are written per button mode, in --output/guided and --output/joker. The guided summary
also reports button fidelity: how often the button read back from a generated note (frozen encoder
over primer + generated notes) equals the player's button, and how much each chord lowers it.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_chord_strength import (CONDITIONS, bootstrap_ci, chord_changes, permutation_p, print_summary,
                                 read_piece, require, requested_chords, summarize)
from model_loader import load_model
from x_transformer import AE_style_jokerParam_chords

MODES = ('guided', 'joker')
ADAPTERS = ('decoder.chord_conditioner.', 'decoder.harm_film.', 'decoder.change_conditioner.', 'decoder.change_film.')


def load_style_checkpoint(path: Path, device: torch.device) -> AE_style_jokerParam_chords:
    saved = torch.load(path, map_location='cpu', weights_only=False)
    cfg = saved['cfg']
    require(cfg.get('model_type') == 'AE_style_jokerParam_chords',
            f'Expected an AE_style_jokerParam_chords checkpoint: {path}')
    model = load_model(cfg['model_name'], cfg=cfg, set_only=True)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def style_inputs(model, pitch: np.ndarray, first: int, last: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    """Style context [1, S, dim] and mask [1, S] from the S = style_seq_len notes next to the notes
    first..last-1 of the piece, on their longer side (as in training); padded and masked if short."""
    length = model.style_encoder.max_seq_len
    notes = pitch[last:last + length] if len(pitch) - last >= first else pitch[max(0, first - length):first]
    require(len(notes) > 0, 'No notes left in the piece for a style window')
    style = torch.zeros(1, length, dtype=torch.long, device=device)
    style[0, :len(notes)] = torch.as_tensor(np.asarray(notes), device=device)
    mask = (torch.arange(length, device=device) < len(notes)).unsqueeze(0)
    return model.encode_style(style, mask), mask


@torch.inference_mode()
def player_buttons(model, real: Tensor) -> Tensor:
    """Buttons [1, T] of the notes real [1, T], as training computes them: the frozen encoder over the
    whole passage without its first note (never a target; its button 0 is unused).

    The encoder is bidirectional and its buttons are relative to the encoded passage, so a guided
    primer and its continuation must be encoded together: with the primer encoded alone, the
    pianist's own continuation cost 1.1-2.9 nats instead of 0.2-0.9 under the frozen base.
    """
    return F.pad(model.real_to_discrete(model.encoder({'pitch': real[:, 1:]})), (1, 0))


def mode_buttons(model, real: Tensor, context: int, mode: str) -> Tensor:
    """Primer + continuation buttons [1, T] of the automatic player in this button mode."""
    if mode == 'guided':
        # At every note, the button the encoder gives the pianist's real note there.
        return player_buttons(model, real)
    # Joker: the primer is encoded alone, as the interaction encodes its seed, so its buttons
    # carry nothing of the real continuation; the joker is held from the first generated note on.
    buttons = F.pad(player_buttons(model, real[:, :context]), (0, real.size(1) - context))
    buttons[:, context:] = model.joker_button_idx
    return buttons


@torch.inference_mode()
def free_logits(model, pitch: Tensor, button: Tensor, style: Tensor, style_mask: Tensor) -> Tensor:
    """Logits [B, T, V] for pitch[:, 1:] from the frozen style joker alone (no chord input)."""
    button_real, joker = model._buttons_discrete_to_inputs(button)
    return model.decoder({'pitch': pitch[:, :-1], 'button': button_real[:, 1:], 'joker_mask': joker[:, 1:]},
                         style_context=style.expand(len(pitch), -1, -1),
                         style_context_mask=style_mask.expand(len(pitch), -1))


@torch.inference_mode()
def perform(model: AE_style_jokerParam_chords, piece: Dict[str, np.ndarray], onset: int, rows: list,
            controls: Dict[str, Tensor], noise: Tensor, context: int, temperature: float, mode: str,
            device: torch.device) -> Dict[str, list]:
    """Continue the real primer before onset once per row, paired: all rows share the primer, the style,
    the player's buttons (guided) or the held joker, and the Gumbel noise of their seed.

    rows: (seed, name) per row; controls: chord inputs aligned with pitch, [rows, context + notes(, 12)].
    """
    notes = noise.shape[1]
    real = torch.as_tensor(piece['pitch'][onset - context:onset + notes], device=device).unsqueeze(0)
    style, style_mask = style_inputs(model, piece['pitch'], onset - context, onset + notes, device)
    played = player_buttons(model, real)  # the player's buttons, kept for the guided button fidelity
    button = mode_buttons(model, real, context, mode).expand(len(rows), -1)
    pitch = torch.zeros(len(rows), context + notes, dtype=torch.long, device=device)
    pitch[:, :context] = real[:, :context]
    rows_style, rows_mask = style.expand(len(rows), -1, -1), style_mask.expand(len(rows), -1)
    to_pc = F.one_hot(torch.arange(128, device=device) % 12, 12).float()
    seed_of_row = torch.tensor([seed for seed, _ in rows], device=device)
    noise = noise.to(device)
    pc_mass = []
    for step in range(notes):
        position = context + step
        window = slice(position - context, position + 1)  # same sliding context as performance
        logits = model.next_pitch_logits({'pitch': pitch[:, window], 'button': button[:, window],
                                          **{key: value[:, window] for key, value in controls.items()}},
                                         rows_style, rows_mask)
        pc_mass.append(F.softmax(logits / temperature, dim=-1) @ to_pc)
        # Gumbel-max with noise shared by every request, button mode and checkpoint (common random numbers).
        pitch[:, position] = (logits / temperature - noise[seed_of_row, step].log()).argmax(-1)
    # Plausibility: log-likelihood of each note under the frozen style joker (no chord) with the same
    # buttons, given everything before it (one causal pass); the last row is the pianist's continuation.
    scored, scored_button = torch.cat((pitch, real)), torch.cat((button, button[:1]))
    logp = F.log_softmax(free_logits(model, scored, scored_button, style, style_mask).float(), dim=-1)
    logp = logp[:, context - 1:].gather(-1, scored[:, context:].unsqueeze(-1)).squeeze(-1).cpu()
    # Buttons the frozen encoder reads back from primer + generated notes, for the button fidelity.
    read_back = player_buttons(model, pitch)[:, context:]
    return {'rows': rows, 'pitches': pitch[:, context:].tolist(),
            'pc_mass': torch.stack(pc_mass, dim=1).cpu().round(decimals=6).tolist(),  # round is not on MPS
            'base_logp': logp[:-1].round(decimals=5).tolist(), 'real_base_logp': logp[-1].round(decimals=5).tolist(),
            'player_buttons': played[0, context:].tolist(), 'read_back_buttons': read_back.tolist()}


def generate(model: AE_style_jokerParam_chords, piece: Dict[str, np.ndarray], onset: int, chords: Dict[str, dict],
             noise: Tensor, context: int, temperature: float, mode: str, device: torch.device) -> Dict[str, list]:
    """Rows = seeds x (free + CONDITIONS); a requested chord is held from the first generated note on."""
    seeds, notes = noise.shape[:2]
    rows = [(seed, name) for seed in range(seeds) for name in ('free',) + CONDITIONS]
    length = context + notes
    controls = {'chord_chroma': torch.zeros(len(rows), length, 12, dtype=torch.bool, device=device),
                'chord_anchor_pc': torch.zeros(len(rows), length, dtype=torch.long, device=device),
                'chord_active': torch.zeros(len(rows), length, dtype=torch.bool, device=device)}
    for row, (_, name) in enumerate(rows):
        if name != 'free':
            controls['chord_chroma'][row, context:, chords[name]['pcs']] = True
            controls['chord_anchor_pc'][row, context:] = chords[name]['anchor']
            controls['chord_active'][row, context:] = True
    return perform(model, piece, onset, rows, controls, noise, context, temperature, mode, device)


def run_file(path: Path, file_index: int, models: Dict[str, AE_style_jokerParam_chords],
             args: argparse.Namespace, device: torch.device) -> Dict[str, dict]:
    """One record per button mode, in eval_chord_strength.py's format."""
    piece = read_piece(path)
    changes = chord_changes(piece, args.context, args.notes)
    require(len(changes) >= args.onsets, f'{path.name}: only {len(changes)} usable chord changes')
    onsets = changes[np.linspace(0, len(changes) - 1, args.onsets).round().astype(int)]  # spread over the piece
    records = {mode: {'midi': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'performed_notes': len(piece['pitch']), 'usable_changes': len(changes), 'onsets': []}
               for mode in MODES}
    for onset_index, onset in enumerate(onsets.tolist()):
        chords = requested_chords(piece, onset)
        rng = torch.Generator().manual_seed(args.seed + 1000 * file_index + onset_index)
        noise = torch.empty(args.seeds, args.notes, 128).exponential_(generator=rng)
        for mode in MODES:
            entry = {'onset': onset, 'chords': chords, 'last_primer_pitch': int(piece['pitch'][onset - 1]),
                     'real_continuation': piece['pitch'][onset:onset + args.notes].tolist(), 'models': {}}
            for name, model in models.items():
                entry['models'][name] = generate(model, piece, onset, chords, noise, args.context,
                                                 args.temperature, mode, device)
            records[mode]['onsets'].append(entry)
        print(f'{path.name}: change {onset_index + 1}/{args.onsets} done', flush=True)
    return records


#==========================================================================

def pianist_reference(entries: list, args: argparse.Namespace) -> dict:
    """The pianist's real next notes: chord-tone rate of the real chord, and frozen-base NLL with this
    mode's buttons and style (generated notes are scored the same way)."""
    reference = {}
    for h in args.horizons:
        rates, nll = [], []
        for _, entry in entries:
            run = next(iter(entry['models'].values()))  # the frozen base is the same in every checkpoint
            rates.append(np.isin(np.asarray(entry['real_continuation'][:h]) % 12, entry['chords']['real']['pcs']).mean())
            nll.append(-np.mean(run['real_base_logp'][:h]))
        reference[f'@{h}'] = {'real_chord_tone_rate': float(np.mean(rates)), 'base_nll': float(np.mean(nll))}
    return reference


def button_fidelity(output: Path, args: argparse.Namespace) -> dict:
    """Guided mode: share of generated notes whose read-back button equals the player's button, and mean
    absolute button difference, per checkpoint, request and horizon; chord minus free is paired per change."""
    records = [json.loads(path.read_text()) for path in sorted(output.glob('*.json')) if path.name != 'summary.json']
    entries = [entry for record in records for entry in record['onsets']]
    rng = np.random.default_rng(args.seed)
    result = {}
    for model in entries[0]['models']:
        for horizon in args.horizons:
            values = {}
            for name in ('free',) + CONDITIONS:
                match, distance = [], []
                for entry in entries:
                    run = entry['models'][model]
                    player = np.asarray(run['player_buttons'][:horizon])
                    read = np.asarray([run['read_back_buttons'][row][:horizon]
                                       for row, (_, request) in enumerate(run['rows']) if request == name])
                    match.append((read == player).mean())
                    distance.append(np.abs(read - player).mean())
                values[name] = np.asarray(match), np.asarray(distance)
            block = {}
            for name, (match, distance) in values.items():
                block[name] = {'button_match': float(match.mean()), 'button_distance': float(distance.mean())}
                if name != 'free':
                    difference = match - values['free'][0]
                    block[name].update(match_minus_free=float(difference.mean()), ci95=bootstrap_ci(difference, rng),
                                       p_permutation=permutation_p(difference, rng))
            result[f'{model}@{horizon}'] = block
    return result


def print_fidelity(fidelity: dict) -> None:
    print('Guided button fidelity: read-back button = player button (match), mean |button difference|, '
          'chord - free match [95% CI], p')
    for key, block in fidelity.items():
        for name, value in block.items():
            line = f'{key[-40:]:<40} {name:<12} match {value["button_match"]:.3f} distance {value["button_distance"]:.2f}'
            if 'match_minus_free' in value:
                line += (f'  {value["match_minus_free"]:+.3f} [{value["ci95"][0]:+.3f}, {value["ci95"][1]:+.3f}]'
                         f'  p = {value["p_permutation"]:.1e}')
            print(line)
    print('=' * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, nargs='+',
                        default=[ROOT / 'save_models/AE_style_jokerParam_chords_best.pth'],
                        help='Checkpoints of the same frozen base; with two, the second is compared to the first')
    parser.add_argument('--midi-dir', type=Path, default=ROOT / 'samples/not_seen')
    parser.add_argument('--midi', nargs='*', help='Only these file names from --midi-dir (default: all)')
    parser.add_argument('--output', type=Path, default=ROOT / 'out/style_chord_strength')
    parser.add_argument('--base-train', type=Path, default=ROOT / 'Training-Data/giantmidi_full_train.pickle',
                        help='Pickle the frozen style joker was trained on (for the unseen check)')
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
    if not args.summarize_only:
        device = torch.device(args.device)
        models = {path.stem: load_style_checkpoint(path, device) for path in args.checkpoint}
        frozen = [{key: value for key, value in model.state_dict().items() if not key.startswith(ADAPTERS)}
                  for model in models.values()]
        require(all(state.keys() == frozen[0].keys() and all(torch.equal(value, frozen[0][key]) for key, value in state.items())
                    for state in frozen[1:]), 'Checkpoints do not share the same frozen base')
        files = sorted(path for path in args.midi_dir.iterdir() if path.suffix.lower() in ('.mid', '.midi'))
        # The index in the folder seeds the noise, so one-file runs reproduce a full run.
        for file_index, path in enumerate(files):
            if not args.midi or path.name in args.midi:
                for mode, record in run_file(path, file_index, models, args, device).items():
                    (args.output / mode).mkdir(parents=True, exist_ok=True)
                    (args.output / mode / (path.stem + '.json')).write_text(json.dumps(record))
    for mode in MODES:
        print(f'Button mode: {mode}')
        summary = summarize(args.output / mode, args, reference=pianist_reference)
        if mode == 'guided':
            summary['button_fidelity'] = button_fidelity(args.output / mode, args)
            (args.output / mode / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        print_summary(summary)
        if mode == 'guided':
            print_fidelity(summary['button_fidelity'])
        args.skip_unseen_check = True  # The training overlap does not depend on the button mode: check it once.


if __name__ == '__main__':
    main()
