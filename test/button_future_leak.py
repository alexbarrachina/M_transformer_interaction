"""
Question: Do a model's buttons carry the notes that come after them, so that its decoder follows the buttons well in training
but worse in play, where a press can only know the notes already played?
In training the bidirectional encoder computes every button from the whole passage, so a button can depend on later notes
(AE_style_tester_frame_v1's loss even asks for it: each note's target is its place among the 16 notes before and the 16 after).
The decoder sees only the earlier notes, but it sees the buttons of the earlier notes too, and these may already tell where the
melody goes next. A player's presses cannot carry that.
Claim 1 (main): the new model's decoder relies more than the present one's on buttons that know the coming notes: the probability
it gives the real note drops more when the buttons can only know the notes up to them.
Supported if: the 95 % range of new minus present lies above zero. Refuted if it lies below zero. Undecided if it contains zero.
Claim 2: with buttons that know only the notes up to them, the new model gives the real note less probability than the present one.
Supported if: the 95 % range of new minus present lies below zero. Refuted if it lies above zero. Undecided if it contains zero.
Both claims come from an exploratory run on 58 of these passages (2026-11-18, frame_v1 against v2), so this test repeats that
finding on the same pieces; it is not an independent confirmation.
Models: new AE_style_tester_frame_v1 and present AE_style_tester_v2 by default, presets of models.py with their checkpoints.
Base style models (AE_style_*) and style jokers (AE_style_jokerParam_*) work; each model uses its own encoder and decoder.
Data: the 60 pieces and moments of Training-Data/giantmidi_full_w_chords_unseen_test shared by the other tests (neither default
model trained on this split), without the moments whose piece has fewer than 206 notes before the moment or 23 after it, which
the windows below need. The passages, the style and the 8 notes are those of test/button_control_strength.py.
Method: each passage is the pianist's 200 notes before a chord change and the 8 after it. Only the buttons change between three
conditions; the notes, the style and the decoder stay the same. No sampling.
  whole passage: the encoder reads the 207 notes together, as in training. This is test/button_control_strength.py's main
    measure, so both tests give the same value for each passage they share;
  primer, then later notes knowing 16 notes ahead: the 200 notes are encoded alone, as the app encodes its primer, and each of
    the 8 notes gets the button the encoder gives it among 207 notes that end 16 notes after it;
  primer, then later notes knowing only the past: as above, but among the 207 notes that end with it, as a player knows them.
Measure: probability given to the pianist's real note at the 8 notes, averaged per passage, in %. Claim 1: whole passage minus
"knowing only the past", new minus present, in percentage points. Claim 2: "knowing only the past", new minus present.
Exploratory: the middle condition; how many of the 8 notes change button when their 16 later notes are hidden; how many of the
primer's last 16 buttons change when the primer is encoded with the 8 notes after it instead of alone.
Size and time: 60 passages, 3 button conditions, 2 models, about 60 s on the M1 (12 s of it to choose the moments).
Run: python test/button_future_leak.py [new preset] [present preset]
Results: test/results/button_future_leak__<new preset>.txt, one file per evaluated model.
Result (2026-11-18, check run, AE_style_tester_v2 in both roles): every difference is exactly 0. 59 of the 60 moments have the
notes the windows need. v2 gives the real note 66.7 % with the whole passage encoded together and 64.5 % with buttons knowing only
the past, a drop of 2.2 points. See test/results/button_future_leak__AE_style_tester_v2.txt.
Result (2026-11-18, AE_style_tester_frame_v1 best epoch 205 against AE_style_tester_v2 epoch 150): both claims supported.
frame_v1 gives the real note 76.8 % against 66.7 % with the whole passage encoded together, but 53.6 % against 64.5 % with
buttons knowing only the past. Its drop is 21.0 points larger (95 % range +14.9 to +27.4, larger in 46 of 59 passages); claim 2
-10.9 points (range -16.8 to -5.3, lower in 38 of 59). Its button moves 2 or more buttons when the 16 later notes are hidden for
19 % of the notes against 2 %, and 39 % of its last 16 primer buttons change when the 8 notes are added against 17 %. See
test/results/button_future_leak__AE_style_tester_frame_v1.txt.
Result (2026-11-22, AE_style_tester_frame_v2 best epoch 308 against AE_style_tester_v2 epoch 150): claim 1 undecided, claim 2 refuted:
the past-only target removed frame_v1's leak. frame_v2 gives the real note 75.3 % with the whole passage encoded together and
74.8 % with buttons knowing only the past, a drop of 0.4 points against v2's 2.2 (-1.7 points, 95 % range -3.8 to +0.4); with
buttons knowing only the past it gives 10.3 points more than v2 (range +6.8 to +13.9, higher in 46 of 59). Hiding the 16 later
notes changes 5.5 % of its buttons, none by 2 or more (v2 27 % and 2 %, frame_v1 53 % and 19 %), and adding the 8 notes changes
3 % of the last 16 primer buttons (v2 17 %, frame_v1 39 %). See test/results/button_future_leak__AE_style_tester_frame_v2.txt.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from button_control_strength import git_line, last_logits, load_style_model, passage_inputs, read_passages
from chord_strength_many_chords import CONTEXT, NOTES, PIECES, SEED, TEST_PICKLE
from chord_strength_guidance import bootstrap_ci, permutation_p
from chord_strength_style import player_buttons
from models import get_model_hparams

NEW = sys.argv[1] if len(sys.argv) > 1 else 'AE_style_tester_frame_v1'
PRESENT = sys.argv[2] if len(sys.argv) > 2 else 'AE_style_tester_v2'
LENGTH = CONTEXT + NOTES - 1  # notes the encoder reads in training (the passage without its first note)
AHEAD = 16  # later notes a button knows in the middle condition: the future half of frame_v1's 33-note window
PRIMER_END = 16  # last primer buttons compared between the primer encoded alone and with the 8 notes after it
ENCODER_ROWS = 60  # encoder rows per forward pass; 60 rows of 207 notes take about 4 GB on the M1, 240 rows over 16 GB
DEVICE = 'mps'
RESULTS = ROOT / f'test/results/button_future_leak__{NEW}.txt'  # one file per evaluated model


@torch.inference_mode()
def note_buttons(model, windows: np.ndarray, index: int, device: torch.device) -> torch.Tensor:
    """Button [P, NOTES] the encoder gives the note at position `index` of each window [P, NOTES, LENGTH] of notes."""
    rows = torch.as_tensor(windows.reshape(-1, LENGTH), device=device)
    position = torch.cat([model.encoder({'pitch': chunk}) for chunk in rows.split(ENCODER_ROWS)])  # [P * NOTES, LENGTH]
    return model.real_to_discrete(position)[:, index].view(len(windows), NOTES)


@torch.inference_mode()
def model_values(model, data, moments: np.ndarray, windows: dict, device: torch.device) -> dict:
    """Per passage [P] values of one model."""
    pitch, style, style_mask = passage_inputs(model, data, moments, device)  # pitch [P, CONTEXT + NOTES]
    target = pitch[:, CONTEXT:]  # [P, NOTES], the pianist's real notes
    primer = player_buttons(model, pitch[:, :CONTEXT])  # [P, CONTEXT], the primer encoded alone
    later = {name: note_buttons(model, notes, index, device) for name, (notes, index) in windows.items()}  # [P, NOTES] each
    buttons = {'whole': player_buttons(model, pitch),  # [P, CONTEXT + NOTES], as in training
               'ahead': torch.cat([primer, later['ahead']], 1),
               'past': torch.cat([primer, later['past']], 1)}
    values = {}
    for name, button in buttons.items():
        logits = last_logits(model, pitch, button, style, style_mask, torch.arange(len(pitch), device=device), NOTES)
        values[name] = 100 * F.softmax(logits, -1).gather(-1, target[..., None])[..., 0].mean(-1).cpu().numpy()
    values['drop'] = values['whole'] - values['past']
    moved = (later['ahead'] - later['past']).abs()
    values['moved'] = 100 * (moved > 0).float().mean(-1).cpu().numpy()
    values['moved_2'] = 100 * (moved >= 2).float().mean(-1).cpu().numpy()
    values['primer_changed'] = 100 * (buttons['whole'][:, CONTEXT - PRIMER_END:CONTEXT]
                                      != primer[:, -PRIMER_END:]).float().mean(-1).cpu().numpy()
    return values


def main() -> None:
    started = time.time()
    device = torch.device(DEVICE)
    data, moments, usable_pieces, pieces_in_pickle = read_passages()
    piece = data.note_piece[moments]
    start, end = data.piece_edges[piece], data.piece_edges[piece + 1]
    keep = (moments - (LENGTH - 1) >= start) & (moments + NOTES - 1 + AHEAD < end)
    moments = moments[keep]
    note = moments[:, None] + np.arange(NOTES)  # [P, NOTES], indices of the 8 notes in the data
    pitches = data.pitches.astype(np.int64)
    # Encoder windows of LENGTH notes for each of the 8 notes, and the position of that note in them.
    windows = {'ahead': (pitches[note[..., None] - (LENGTH - 1 - AHEAD) + np.arange(LENGTH)], LENGTH - 1 - AHEAD),
               'past': (pitches[note[..., None] - (LENGTH - 1) + np.arange(LENGTH)], LENGTH - 1)}
    print(f'Moments chosen, {keep.sum()} of {len(keep)} kept, {time.time() - started:.0f} s', flush=True)

    values = {}
    for role, name in (('present', PRESENT), ('new', NEW)):
        model = load_style_model(name, device)
        values[role] = model_values(model, data, moments, windows, device)
        del model
        torch.mps.empty_cache()
        print(f'{role} model done, {time.time() - started:.0f} s', flush=True)
    present, new = values['present'], values['new']

    rng = np.random.default_rng(SEED)
    claim_lines, verdicts = [], []
    for label, key, side in (('Claim 1, drop from the whole passage to buttons knowing only the past', 'drop', 1),
                             ('Claim 2, buttons knowing only the past', 'past', -1)):
        difference = new[key] - present[key]
        low, high = bootstrap_ci(difference, rng)
        bounds = side * np.array([low, high])  # positive = on the side the claim expects
        verdicts.append('supported' if bounds.min() > 0 else 'refuted' if bounds.max() < 0 else 'undecided')
        claim_lines.append(f'{label}, new minus present: {difference.mean():+.1f} points, 95 % range {low:+.1f} to {high:+.1f}, '
                           f'p = {permutation_p(difference, rng):.1e}, higher in {(difference > 0).sum()} of {len(difference)} '
                           f'passages, lower in {(difference < 0).sum()}.')

    rows = (('whole', 'Real note, whole passage encoded together, %'),
            ('ahead', 'Real note, primer alone, later notes knowing 16 ahead, %'),
            ('past', 'Real note, primer alone, later notes knowing only the past, %'),
            ('drop', 'Drop from the whole passage to knowing only the past, points'),
            ('moved', '8 notes whose button changes when 16 later notes are hidden, %'),
            ('moved_2', '  by 2 buttons or more, %'),
            ('primer_changed', f'Last {PRIMER_END} primer buttons that change with the 8 notes added, %'))
    report = ['#' * 64, git_line(),
              f'Present: {PRESENT} ({Path(get_model_hparams(PRESENT)["ckpt_file_name"]).name})',
              f'New: {NEW} ({Path(get_model_hparams(NEW)["ckpt_file_name"]).name})  Device: {DEVICE}',
              f'Data: {TEST_PICKLE.name}, {len(moments)} of the {PIECES} shared moments ({usable_pieces} usable pieces, '
              f'{pieces_in_pickle} in the file), one chord change each, seed {SEED}; the others lack the {LENGTH - 1} notes before '
              f'or the {NOTES - 1 + AHEAD} after that the windows need. Passages of {CONTEXT} + {NOTES} notes; encoder windows of '
              f'{LENGTH} notes, {AHEAD} notes ahead in the middle condition.',
              f'Running time: {time.time() - started:.0f} s', '',
              f'Means over passages; probabilities of the real note averaged over the {NOTES} notes of each passage.',
              f'{"Measure":<66} {"Present":>8} {"New":>8} {"New minus present":>18}']
    for key, label in rows:
        report.append(f'{label:<66} {present[key].mean():>8.2f} {new[key].mean():>8.2f} {(new[key] - present[key]).mean():>+18.2f}')
    report += ['', *claim_lines]
    if NEW == PRESENT:
        largest = max(np.abs(new[key] - present[key]).max() for key, _ in rows)
        report.append(f'Check run, the same preset in both roles: largest difference over all measures and passages = {largest:g} (must be 0).')
    report += [f'Result: claim 1 {verdicts[0]}, claim 2 {verdicts[1]}.', '']
    report = '\n'.join(report)
    print(report)
    with open(RESULTS, 'a', encoding='utf-8') as file:
        file.write(report + '\n')


if __name__ == '__main__':
    main()
