"""
Question: Does the frozen style joker give the previous note to a neighbouring button because the buttons it was trained with did?
It learned from the frozen encoder's buttons of real passages. If, in those buttons, a repeated note often gets a button one step away from the previous note's, the decoder has learned that a one-step move can mean the same note.
Claim 1 (main): in the encoder's buttons, a one-step button move (up or down) comes with a repeat of the previous note more often than consecutive notes repeat in general.
Supported if: the 95 % range of the one-step repeat rate minus the overall repeat rate lies above zero. Refuted if it reaches zero or lies below it.
Claim 2: over the same real histories, the decoder gives the previous note to a one-step move about as often as its training buttons do.
Supported if: its probability is between half and twice the one-step repeat rate of claim 1, for moves up and for moves down. Refuted otherwise; above twice means it exaggerates what it was taught.
Model: the frozen style joker shared by the chord adapters, from save_models/AE_style_jokerParam_chords_clear_best.pth with no chord pressed; its frozen encoder gives the buttons.
Data: the same 60 pieces and chord changes of Training-Data/giantmidi_full_w_chords_unseen_test as test/chord_strength_many_chords.py. The style joker never saw this split.
Method: each passage is the pianist's 200 notes before a chord change and 8 after it, with the buttons the encoder gives the whole passage, as in training, and the piece's notes next to it as style. Claim 1 counts the consecutive note pairs of the passage by how far the button moves. For claim 2, at each of the 8 notes after the change, the button is set to the previous note's button moved by -2 to +2 steps, with the real notes and buttons before it, and the decoder's probability of the previous note is read. No sampling.
Measures: share of note pairs with a given button move in which the note repeats; the decoder's probability of the previous note for that move. One value per piece.
Size and time: 60 pieces, about 200 note pairs and 8 x 5 decoder rows each, 77 s on the M1.
Run: python test/style_button_repeat_cause.py
Result (2026-11-08): both claims supported, but the effect is small. Notes repeat in 1.6 % of consecutive pairs; a one-step button move comes with a repeat in 2.0 % (up) and 3.5 % (down) of cases, and the decoder gives the previous note to a one-step move with 4.0 % and 4.4 %. Training explains a few repeats per 100 one-step moves, not the 92-100 % of the recording in test/style_chord_recording_buttons.py. See test/results/style_button_repeat_cause.txt.
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

# The same moments as the many-chords test: its constants (200-note context, 8 notes, 60 pieces, seed) apply here too.
from chord_strength_many_chords import CONTEXT, NOTES, PIECES, SEED, TEST_PICKLE, choose_moments, overlap_per_piece, read_events
from eval_chord_strength import bootstrap_ci, permutation_p
from eval_style_chord_strength import load_style_checkpoint, player_buttons, style_inputs
from train_chords import ChordPitchSamplerDataset

MODEL = ROOT / 'save_models/AE_style_jokerParam_chords_clear_best.pth'  # used without chord: the frozen style joker
MOVES = (-2, -1, 0, 1, 2)  # button moves from the previous note's button tried on the decoder
DEVICE = 'mps'
RESULTS = ROOT / 'test/results/style_button_repeat_cause.txt'


@torch.inference_mode()
def piece_values(model, data: ChordPitchSamplerDataset, onset: int, device: torch.device) -> tuple:
    """Button moves [pairs] and repeats [pairs] of the encoder's buttons over the passage, and the decoder's
    probability of the previous note [NOTES, moves] (NaN where the moved button leaves 0..18)."""
    piece = data.note_piece[onset]
    start, end = data.piece_edges[piece], data.piece_edges[piece + 1]
    passage = torch.as_tensor(data.pitches[onset - CONTEXT:onset + NOTES].astype(np.int64), device=device)
    buttons = player_buttons(model, passage.unsqueeze(0))[0]  # as training; button 0 is unused
    move = (buttons[2:] - buttons[1:-1]).cpu().numpy()
    repeat = (passage[2:] == passage[1:-1]).cpu().numpy()
    # One window of CONTEXT notes + the target per note after the change, one row per move: [NOTES * moves, CONTEXT + 1].
    windows = lambda values: values.unfold(0, CONTEXT + 1, 1).repeat_interleave(len(MOVES), 0)
    pitch, button = windows(passage), windows(buttons).clone()
    moved = button[:, -2] + torch.tensor(MOVES, device=device).repeat(NOTES)
    valid = (moved >= 0) & (moved < model.joker_button_idx)
    button[:, -1] = moved.clamp(0, model.joker_button_idx - 1)
    # 8 notes x 5 moves = 40 rows: one forward pass per piece fits in the M1's memory.
    style, style_mask = style_inputs(model, data.pitches[start:end], onset - CONTEXT - start, onset + NOTES - start, device)
    logits = model.next_pitch_logits({'pitch': pitch, 'button': button}, style.expand(len(pitch), -1, -1),
                                     style_mask.expand(len(pitch), -1))
    probability = F.softmax(logits.float(), dim=-1).gather(-1, pitch[:, -2:-1]).squeeze(-1)
    probability = torch.where(valid, probability, torch.nan)
    return move, repeat, probability.view(NOTES, len(MOVES)).cpu().numpy()


def main() -> None:
    started = time.time()
    device = torch.device(DEVICE)
    events = read_events(TEST_PICKLE)
    data = ChordPitchSamplerDataset(events, CONTEXT + NOTES, is_eval=True)
    pieces_in_pickle = len(data.piece_edges) - 1
    moments, usable_pieces = choose_moments(data, overlap_per_piece(events.reshape(-1, 5), pieces_in_pickle))
    model = load_style_checkpoint(MODEL, device)
    rate = {key: [] for key in ('all', 'one step', *MOVES)}  # per piece: share of pairs with that move that repeat
    repeats_by_move = np.zeros(4)  # repeats whose button stays, moves one step, two steps, more
    decoder = []  # per piece [moves]: mean probability of the previous note over the notes after the change
    for index, onset in enumerate(moments.tolist()):
        move, repeat, probability = piece_values(model, data, onset, device)
        rate['all'].append(repeat.mean())
        rate['one step'].append(repeat[np.abs(move) == 1].mean())
        for value in MOVES:
            rate[value].append(repeat[move == value].mean() if (move == value).any() else np.nan)
        repeats_by_move += np.bincount(np.minimum(np.abs(move[repeat]), 3), minlength=4)
        decoder.append(np.nanmean(probability, axis=0))
        print(f'moment {index + 1}/{PIECES}, {time.time() - started:.0f} s', flush=True)
    rate = {key: 100 * np.array(value) for key, value in rate.items()}
    decoder = 100 * np.array(decoder)  # [pieces, moves], %

    rng = np.random.default_rng(SEED)
    difference = rate['one step'] - rate['all']
    low, high = bootstrap_ci(difference, rng)
    claim1 = 'supported' if low > 0 else 'refuted'
    ratios = {value: np.nanmean(decoder[:, MOVES.index(value)]) / np.nanmean(rate[value]) for value in (-1, 1)}
    claim2 = 'supported' if all(0.5 <= ratio <= 2 for ratio in ratios.values()) else 'refuted'

    git = lambda *args: subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    lines = ['#' * 64,
             f'Date: {datetime.datetime.now():%Y-%m-%d %H:%M}  Commit: {git("rev-parse", "--short", "HEAD")}'
             f'  Uncommitted files: {len(git("status", "--short").splitlines())}',
             f'Model: {MODEL.name}, no chord (the frozen style joker)  Device: {DEVICE}',
             f'Data: {TEST_PICKLE.name}, {PIECES} of the {usable_pieces} usable pieces ({pieces_in_pickle} in the file), one chord change each. '
             f'Passages of {CONTEXT} + {NOTES} notes, buttons from the encoder over the whole passage, seed {SEED}.',
             f'Running time: {time.time() - started:.0f} s', '',
             f'Overall: {rate["all"].mean():.1f} % of consecutive note pairs repeat the note. Repeats by button move: '
             + ', '.join(f'{name} {100 * count / repeats_by_move.sum():.0f} %' for name, count in
                         zip(('same button', 'one step', 'two steps', 'more'), repeats_by_move)) + f' (of {repeats_by_move.sum():.0f} repeats).', '',
             'By button move from the previous note: share of the encoder\'s note pairs that repeat the note (training buttons),',
             'and the decoder\'s probability of the previous note when given that move at the 8 notes after the change. Means over pieces.',
             f'{"Button move":>11} {"Training buttons: repeats %":>28} {"Decoder: previous note %":>25} {"Pieces":>6}']
    for value in MOVES:
        column = MOVES.index(value)
        lines.append(f'{value:>+11d} {np.nanmean(rate[value]):>28.1f} {np.nanmean(decoder[:, column]):>25.1f}'
                     f' {np.isfinite(decoder[:, column]).sum():>6}')
    lines += ['', f'Claim 1, one-step repeat rate minus overall rate: {difference.mean():+.1f} points, 95 % range {low:+.1f} to {high:+.1f}, '
                  f'p = {permutation_p(difference, rng):.1e}, higher in {(difference > 0).sum()} of {len(difference)} pieces.',
              'Claim 2, decoder probability / training repeat rate: '
              + ', '.join(f'move {value:+d} {ratio:.2f}' for value, ratio in ratios.items()) + '.',
              f'Result: claim 1 {claim1}, claim 2 {claim2}.', '']
    report = '\n'.join(lines)
    print(report)
    with open(RESULTS, 'a', encoding='utf-8') as file:
        file.write(report + '\n')


if __name__ == '__main__':
    main()
