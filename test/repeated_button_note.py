"""
Question: With the repaired repeated-note rule, does pressing the previous note's button again give the previous note more often?
Claim: when the next press is on the previous note's button, the new style joker gives the previous note more probability than the present one.
Supported if: the 95 % range of new minus present lies above zero. Refuted if it reaches zero or lies below it.
Reference: the same probability when the next press is one button higher than the previous note's. It should not rise; if it
rises as much, the gain is not specific to pressing the same button again.
Models: present AE_style_jokerParam_tester_v2 and new AE_style_jokerParam_tester_v3 (repaired button losses,
docs/Button_loss_repair_plan.md), as in test/button_control_strength.py. The new preset is the first argument; giving the
present one checks the test, since every difference must then be exactly 0.
Data: the same 60 pieces and moments of Training-Data/giantmidi_full_w_chords_unseen_test as test/style_button_repeat_cause.py,
which leaves out pieces with over 10 % of their 8-note sequences in giantmidi_full_train (both models' training data) or in the
chord adapters' training data. Neither style joker trained on this split.
Method: each passage is the pianist's 200 notes before a chord change and the 8 after it, with the buttons that model's encoder
gives the whole passage, as in training, and the piece's notes next to it as style. At each of the 8 notes, with the real notes
and buttons before it, the button of that note is set to the previous note's button, then to one button higher, and the
probability of the previous note's pitch is read. Each model uses its own encoder. No sampling.
Measure: that probability with the previous note's button, averaged over the 8 notes of each passage, new minus present, in
percentage points.
Exploratory: the share of repeated notes in the 60 passages that keep their button in each model's encoder buttons.
Size and time: 60 passages, 8 notes, 2 buttons per model, 66 s on the M1 (12 s to choose the moments).
Run: python test/repeated_button_note.py [new preset, default AE_style_jokerParam_tester_v3] [present preset, default AE_style_jokerParam_tester_v2]
Result (2026-11-11): check run only, with the present model in both roles: every difference is exactly 0. The present model gives
the previous note 8.2 % on its button and 4.0 % one button higher, and its encoder keeps the button on 34 of 202 repeated notes
(17 %). Not run yet with the new model. See test/results/repeated_button_note.txt.
Result (2026-11-12, base models AE_style_tester_v3 epoch 48 against AE_style_tester_v2 epoch 150): supported. The previous note's
button gives the previous note 48.8 % against 10.4 % (+38.4 points, 95 % range +32.9 to +43.8, higher in 57 of 60 passages); one
button higher rises only 1.7 points. The new encoder keeps the button on 199 of 202 repeated notes, the old one on 48.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from button_control_strength import (DEVICE, NEW, PRESENT, git_line, last_logits, load_style_model, passage_inputs,
                                     read_passages)
from chord_strength_many_chords import CONTEXT, NOTES, PIECES, SEED, TEST_PICKLE
from eval_chord_strength import bootstrap_ci, permutation_p
from eval_style_chord_strength import player_buttons
from models import get_model_hparams

MOVES = (0, 1)  # the next press: the previous note's button, then one button higher
RESULTS = ROOT / 'test/results/repeated_button_note.txt'


@torch.inference_mode()
def model_values(model, data, moments: np.ndarray, device: torch.device) -> dict:
    """Per passage [P] probabilities of the previous note for each move, and pooled counts of repeated notes."""
    pitch, style, style_mask = passage_inputs(model, data, moments, device)
    buttons = player_buttons(model, pitch)  # [P, CONTEXT + NOTES], as training; button 0 is unused
    repeat = pitch[:, 2:] == pitch[:, 1:-1]
    kept = repeat & (buttons[:, 2:] == buttons[:, 1:-1])

    # One window of CONTEXT notes + the target per note after the change, one row per move: [P * NOTES * moves, CONTEXT + 1].
    windows = lambda values: values.unfold(1, CONTEXT + 1, 1).flatten(0, 1).repeat_interleave(len(MOVES), 0)
    window_pitch, window_button = windows(pitch), windows(buttons).clone()
    moved = window_button[:, -2] + torch.tensor(MOVES, device=device).repeat(len(pitch) * NOTES)
    count = int(model.cfg['num_buttons'])  # buttons 0..count-1
    valid = moved < count  # one button higher than button 18 does not exist
    window_button[:, -1] = moved.clamp(max=count - 1)
    passage = torch.arange(len(pitch), device=device).repeat_interleave(NOTES * len(MOVES))
    logits = last_logits(model, window_pitch, window_button, style, style_mask, passage, 1)[:, 0]
    probability = F.softmax(logits, -1).gather(-1, window_pitch[:, -2:-1])[:, 0]  # of the previous note's pitch
    probability = torch.where(valid, probability, torch.nan).view(len(pitch), NOTES, len(MOVES)).cpu().numpy()
    with np.errstate(all='ignore'):  # a passage whose 8 previous notes are all on button 18 has no higher button
        mean = 100 * np.nanmean(probability, axis=1)  # [P, moves]
    return {'same': mean[:, 0], 'higher': mean[:, 1],
            'repeats': int(repeat.sum()), 'kept': int(kept.sum())}


def main() -> None:
    started = time.time()
    device = torch.device(DEVICE)
    data, moments, usable_pieces, pieces_in_pickle = read_passages()
    print(f'Moments chosen, {time.time() - started:.0f} s', flush=True)
    values = {}
    for role, name in (('present', PRESENT), ('new', NEW)):
        values[role] = model_values(load_style_model(name, device), data, moments, device)
        print(f'{role} model done, {time.time() - started:.0f} s', flush=True)
    present, new = values['present'], values['new']

    rng = np.random.default_rng(SEED)
    difference = new['same'] - present['same']
    low, high = bootstrap_ci(difference, rng)
    verdict = 'supported' if low > 0 else 'refuted'
    higher = new['higher'] - present['higher']
    higher = higher[np.isfinite(higher)]
    higher_low, higher_high = bootstrap_ci(higher, rng)

    same_label = "On the previous note's button"
    report =['#' * 64, git_line(),
              f'Present: {PRESENT} ({Path(get_model_hparams(PRESENT)["ckpt_file_name"]).name})',
              f'New: {NEW} ({Path(get_model_hparams(NEW)["ckpt_file_name"]).name})  Device: {DEVICE}',
              f'Data: {TEST_PICKLE.name}, {PIECES} of the {usable_pieces} usable pieces ({pieces_in_pickle} in the file), one chord change each. '
              f'Passages of {CONTEXT} + {NOTES} notes, seed {SEED}.',
              f'Running time: {time.time() - started:.0f} s', '',
              'Probability of the previous note at the 8 notes after the change, means over passages.',
              f'{"Next press":<36} {"Present %":>10} {"New %":>8} {"New minus present":>18} {"Passages":>9}',
              f'{same_label:<36} {present["same"].mean():>10.2f} {new["same"].mean():>8.2f}'
              f' {difference.mean():>+18.2f} {len(difference):>9}',
              f'{"One button higher":<36} {np.nanmean(present["higher"]):>10.2f} {np.nanmean(new["higher"]):>8.2f}'
              f' {higher.mean():>+18.2f} {len(higher):>9}', '',
              f'Main, previous note\'s button, new minus present: {difference.mean():+.2f} points, 95 % range {low:+.2f} to {high:+.2f}, '
              f'p = {permutation_p(difference, rng):.1e}, higher in {(difference > 0).sum()} of {len(difference)} passages, '
              f'lower in {(difference < 0).sum()}.',
              f'Reference, one button higher, new minus present: {higher.mean():+.2f} points, 95 % range {higher_low:+.2f} to {higher_high:+.2f}.',
              'Exploratory, repeated notes in the passages that keep their button: '
              + ', '.join(f'{role} {value["kept"]} of {value["repeats"]} ({100 * value["kept"] / max(value["repeats"], 1):.0f} %)'
                          for role, value in values.items()) + '.']
    if NEW == PRESENT:
        largest = max(np.nanmax(np.abs(new[key] - present[key])) for key in ('same', 'higher'))
        largest = max(largest, abs(new['kept'] - present['kept']))
        report.append(f'Check run, the same preset in both roles: largest difference over all measures and passages = {largest:g} (must be 0).')
    report += [f'Result: {verdict} (supported if the range lies above zero).', '']
    report = '\n'.join(report)
    print(report)
    with open(RESULTS, 'a', encoding='utf-8') as file:
        file.write(report + '\n')


if __name__ == '__main__':
    main()
