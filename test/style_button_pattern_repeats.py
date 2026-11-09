"""
Question: Does playing the button pattern of the recording in test/style_chord_recording_buttons.py make the style joker repeat the previous note on a changed button, and does holding a chord make it more frequent?
In that recording the pattern 6 8 9 10 9 8 was played during two chord holds, and button 9 after button 8 gave the previous note again 4 times out of 4.
Claim 1 (main): with no chord, playing the pattern makes notes on a changed button repeat the previous note more often than a natural player does.
Supported if: the 95 % range of pattern minus natural player lies above zero. Refuted if it reaches zero or lies below it.
Claim 2: with the pattern, holding the piece's real chord makes these repeats more frequent than no chord.
Supported if: the 95 % range of chord held minus no chord lies above zero. Refuted if it reaches zero or lies below it.
Model: save_models/AE_style_jokerParam_chords_clear_best.pth; without a chord it is the frozen style joker.
Data: 60 pieces of Training-Data/giantmidi_full_w_chords_unseen_test, chosen as in test/chord_strength_many_chords.py but with at least 24 notes after the chord change, so a few moments differ. The style joker never saw this split; it chose the clear adapter's best checkpoint but never trained it.
Method: as in the app, the primer is the pianist's 128 notes before the chord change, encoded alone, and the model generates 24 notes over a sliding 128-note window, sampled at temperature 1, with the piece's notes next to the passage as style. Three players: the pattern 6 8 9 10 9 8 played 4 times with no chord, the same with the real chord held from the first generated note, and a natural player whose buttons are those the encoder gives the pianist's real 24 notes, encoded with the primer as in training, with no chord. All three share the primer, the style and the sampling noise.
Measure: share of generated notes played on a button other than the previous note's that repeat the previous note's pitch, one value per piece. Higher means more repeats on changed buttons.
Exploratory: the same share at the pattern's 8 -> 9 steps, all repeats, and the share of generated notes that are chord notes.
Size and time: 60 pieces, 24 notes, 3 players, 88 s on the M1.
Run: python test/style_button_pattern_repeats.py
Result (2026-11-08): claim 1 supported, claim 2 refuted. The pattern doubles the repeats on a changed button (9.2 % of notes against 4.2 % for the natural player); holding the chord adds none (8.8 %), although it raises chord notes from 38 to 66 %. No piece repeated at 3 or 4 of the 4 steps 8 -> 9, against 4 of 4 in the recording. See test/results/style_button_pattern_repeats.txt.
Caveat (2026-11-09): claim 1 compares players whose primers were also encoded differently (alone for the pattern, with the continuation for the natural player), and test/style_button_pattern_chopin.py shows that the encoding alone can change these repeats a lot; part of the difference may come from it.
"""

import datetime
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Moments chosen as in the many-chords test (its 200-note context guarantees the 128-note primer here).
from chord_strength_many_chords import PIECES, SEED, TEST_PICKLE, choose_moments, chords_at, overlap_per_piece, read_events
from chord_strength_many_chords import CONTEXT as MOMENT_CONTEXT, NOTES as MOMENT_NOTES
from eval_chord_strength import bootstrap_ci, permutation_p, require
from eval_style_chord_strength import load_style_checkpoint, player_buttons, style_inputs
from train_chords import ChordPitchSamplerDataset

MODEL = ROOT / 'save_models/AE_style_jokerParam_chords_clear_best.pth'
PATTERN = (6, 8, 9, 10, 9, 8)  # buttons of the recording's chord holds
GENERATED = 24  # the pattern 4 times
CONTEXT = 128  # the app's CTX_LEN on the M1: primer length and sliding window
DEVICE = 'mps'
ROWS_PER_PASS = 45  # rows per forward pass that fit in the M1's memory
RESULTS = ROOT / 'test/results/style_button_pattern_repeats.txt'
PLAYERS = ('Natural player, no chord', 'Pattern, no chord', 'Pattern, real chord held')


@torch.inference_mode()
def player_inputs(model, data: ChordPitchSamplerDataset, onset: int, device: torch.device) -> dict:
    """Rows of the three players at one chord change: pitch and button [players, CONTEXT + GENERATED], chord inputs,
    style [players, S, dim] and mask [players, S]. Pitches after the primer are filled in by generation."""
    piece = data.note_piece[onset]
    start, end = data.piece_edges[piece], data.piece_edges[piece + 1]
    require(onset + GENERATED <= end, 'Piece too short for the natural player')
    real = torch.as_tensor(data.pitches[onset - CONTEXT:onset + GENERATED].astype(np.int64), device=device).unsqueeze(0)
    natural = player_buttons(model, real)[0]  # primer and real continuation encoded together, as in training
    pattern = torch.cat((player_buttons(model, real[:, :CONTEXT])[0],  # the app encodes its primer alone
                         torch.tensor(PATTERN, device=device).repeat(GENERATED // len(PATTERN))))
    length = CONTEXT + GENERATED
    pitch = real.expand(len(PLAYERS), -1).clone()
    pitch[:, CONTEXT:] = 0
    chord = chords_at(data, onset)['real']
    held = torch.zeros(len(PLAYERS), length, dtype=torch.bool, device=device)
    held[2, CONTEXT:] = True  # only the third player holds the chord, from the first generated note on
    chroma = torch.zeros(12, dtype=torch.bool, device=device)
    chroma[chord['pcs']] = True
    style, style_mask = style_inputs(model, data.pitches[start:end], onset - CONTEXT - start, onset + GENERATED - start, device)
    member = np.zeros(12, dtype=bool)
    member[chord['pcs']] = True
    return {'pitch': pitch, 'button': torch.stack((natural, pattern, pattern)), 'chord_active': held,
            'chord_chroma': held.unsqueeze(-1) & chroma, 'chord_anchor_pc': torch.full_like(pitch, chord['anchor']),
            'style': style.expand(len(PLAYERS), -1, -1), 'style_mask': style_mask.expand(len(PLAYERS), -1), 'member': member}


@torch.inference_mode()
def generate(model, rows: dict, noise: torch.Tensor) -> torch.Tensor:
    """Fill rows['pitch'][:, CONTEXT:] note by note; noise [rows, GENERATED, 128] (Gumbel-max sampling, shared per piece)."""
    controls = ('pitch', 'button', 'chord_active', 'chord_chroma', 'chord_anchor_pc')
    for step in range(GENERATED):
        window = slice(step, step + CONTEXT + 1)  # the app's sliding window: CONTEXT notes + the target
        logits = torch.cat([model.next_pitch_logits({key: rows[key][part, window] for key in controls},
                                                    rows['style'][part], rows['style_mask'][part])
                            for part in torch.arange(len(rows['pitch']), device=noise.device).split(ROWS_PER_PASS)])
        rows['pitch'][:, CONTEXT + step] = (logits.float() - noise[:, step].log()).argmax(-1)
    return rows['pitch']


def main() -> None:
    started = time.time()
    device = torch.device(DEVICE)
    events = read_events(TEST_PICKLE)
    data = ChordPitchSamplerDataset(events, MOMENT_CONTEXT + MOMENT_NOTES, is_eval=True)
    pieces_in_pickle = len(data.piece_edges) - 1
    moments, usable_pieces = choose_moments(data, overlap_per_piece(events.reshape(-1, 5), pieces_in_pickle), notes=GENERATED)
    model = load_style_checkpoint(MODEL, device)
    pieces = [player_inputs(model, data, onset, device) for onset in moments.tolist()]
    rows = {key: torch.cat([piece[key] for piece in pieces]) for key in pieces[0] if key != 'member'}  # [pieces * players, ...]
    member = np.stack([piece['member'] for piece in pieces])  # [pieces, 12]
    noise = torch.empty(PIECES, GENERATED, 128).exponential_(generator=torch.Generator().manual_seed(SEED))
    pitch = generate(model, rows, noise.repeat_interleave(len(PLAYERS), 0).to(device))
    print(f'generated, {time.time() - started:.0f} s', flush=True)

    # [pieces, players, CONTEXT + GENERATED]; note t of the generation is compared with note t - 1 (the primer's last note for t = 0).
    pitch = pitch.view(PIECES, len(PLAYERS), -1)[..., CONTEXT - 1:].cpu().numpy()
    button = rows['button'].view(PIECES, len(PLAYERS), -1)[..., CONTEXT - 1:].cpu().numpy()
    repeat = pitch[..., 1:] == pitch[..., :-1]  # [pieces, players, GENERATED]
    changed = button[..., 1:] != button[..., :-1]
    share = 100 * (repeat & changed).sum(-1) / np.maximum(changed.sum(-1), 1)  # [pieces, players], %
    eight_nine = (button[..., :-1] == 8) & (button[..., 1:] == 9)
    share_eight_nine = 100 * (repeat & eight_nine).sum(-1) / np.maximum(eight_nine.sum(-1), 1)
    chord_notes = 100 * member[np.arange(PIECES)[:, None, None], pitch[..., 1:] % 12]  # [pieces, players, GENERATED]

    rng = np.random.default_rng(SEED)
    claims = [('Claim 1, no chord: pattern minus natural player', share[:, 1] - share[:, 0]),
              ('Claim 2, pattern: chord held minus no chord', share[:, 2] - share[:, 1])]
    verdicts, claim_lines = [], []
    for label, difference in claims:
        low, high = bootstrap_ci(difference, rng)
        verdicts.append('supported' if low > 0 else 'refuted')
        claim_lines.append(f'{label}: {difference.mean():+.1f} points, 95 % range {low:+.1f} to {high:+.1f}, '
                           f'p = {permutation_p(difference, rng):.1e}, higher in {(difference > 0).sum()} of {len(difference)} pieces, '
                           f'lower in {(difference < 0).sum()}.')

    git = lambda *args: subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    lines = ['#' * 64,
             f'Date: {datetime.datetime.now():%Y-%m-%d %H:%M}  Commit: {git("rev-parse", "--short", "HEAD")}'
             f'  Uncommitted files: {len(git("status", "--short").splitlines())}',
             f'Model: {MODEL.name}  Device: {DEVICE}',
             f'Data: {TEST_PICKLE.name}, {PIECES} of the {usable_pieces} usable pieces ({pieces_in_pickle} in the file), one chord change each. '
             f'Primer and window {CONTEXT} notes, {GENERATED} generated notes, pattern {PATTERN}, temperature 1, seed {SEED}.',
             f'Running time: {time.time() - started:.0f} s', '',
             f'{"Player":<26} {"Repeats on a changed button %":>30} {"Repeats at 8 -> 9 %":>20} {"All repeats %":>14} {"Chord notes %":>14}']
    for index, name in enumerate(PLAYERS):
        eight_nine_value = f'{share_eight_nine[:, index].mean():.1f}' if eight_nine[:, index].any() else '-'
        lines.append(f'{name:<26} {share[:, index].mean():>30.1f} {eight_nine_value:>20} {100 * repeat[:, index].mean():>14.1f}'
                     f' {chord_notes[:, index].mean():>14.1f}')
    lines += ['Chord notes: share of generated notes in the piece\'s real chord, held or not.', '', *claim_lines,
              f'Result: claim 1 {verdicts[0]}, claim 2 {verdicts[1]}.', '']
    report = '\n'.join(lines)
    print(report)
    with open(RESULTS, 'a', encoding='utf-8') as file:
        file.write(report + '\n')


if __name__ == '__main__':
    main()
