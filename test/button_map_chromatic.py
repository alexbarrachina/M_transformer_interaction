"""
Question: Is the new button map more chromatic, and does its encoder crowd the notes onto the end buttons 0 and 18?
Playing AE_style_tester_v3, the map sounded less musical and more chromatic than AE_style_tester_v2's, and its 19 buttons span
32 semitones instead of 47 (test/button_control_strength.py).
Claim 1 (main): the 19 buttons of the new model give more notes outside the passage's scale than the present model's.
Supported if: the 95 % range of new minus present lies above zero. Refuted if it reaches zero or lies below it.
Claim 2: the new model's encoder puts more of a real passage's notes on buttons 0 and 18 than the present model's.
Supported if: the 95 % range of new minus present lies above zero. Refuted if it reaches zero or lies below it.
Models: new AE_style_tester_v3 and present AE_style_tester_v2 by default, presets of models.py with their checkpoints.
Data: the same 60 pieces and moments of Training-Data/giantmidi_full_w_chords_unseen_test as test/button_control_strength.py.
Neither model trained on this split.
Scale: the 7 most frequent pitch classes among the pianist's 200 notes before the moment (a key proxy without labels; it follows
modulations within a piece). A map that spreads its notes evenly over the 12 pitch classes would give about 5/12 = 42 % outside.
Method: at the moment (the first note after a chord change), with the pianist's 200 real notes and the buttons the model's encoder
gives the passage before it, each of the 19 buttons is pressed in turn and the model's most likely note is read (the sweep of
test/button_control_strength.py). The encoder's buttons of the whole 208-note passage, as in training, give claim 2. No sampling.
Measures, one value per passage:
  claim 1: share of the 19 buttons whose most likely note is outside the scale, %;
  claim 2: share of the passage's notes that the encoder puts on button 0 or 18, %.
Exploratory: neighbouring buttons whose most likely notes are a semitone apart; the pianist's own next 8 notes outside the scale
(real music as reference); the encoder's distinct buttons per passage, its semitones per button (least-squares slope of pitch on
button) and, of consecutive real notes one button apart, the share a semitone apart; the spread of the encoder's positions;
the share of the notes on each of the 19 buttons, and the share of notes on the button their pitch gets most often in the
passage (near 100 % means the button depends on the pitch alone, not on the melody around it); both added 2026-11-12, after
the first run.
Size and time: 60 passages, 19 buttons, 2 models; 83 s on the M1.
Run: python test/button_map_chromatic.py [new preset] [present preset]
Results: test/results/button_map_chromatic__<new preset>.txt, one file per evaluated model.
Result (2026-11-12, AE_style_tester_v3 epoch 48 against AE_style_tester_v2 epoch 150): both claims supported. The new map gives
23.0 % of its buttons' notes outside the scale against 15.5 % (the pianist's own next notes: 15.4 %), and 39.8 % of neighbouring
buttons are a semitone apart against 10.2 %. Its encoder puts 38.3 % of the notes on button 0 or 18 against 7.0 %, in all 60
passages, and squeezes the rest to 1.7 semitones per button against 2.6. See test/results/button_map_chromatic__AE_style_tester_v3.txt.
Result (2026-11-12, AE_style_tester_v2_repeat_fix epoch 78 against AE_style_tester_v2): both claims supported, about half of v3's
change. Buttons outside the scale 21.1 % against 15.5 %, neighbouring buttons a semitone apart 32.4 % against 10.2 %, notes on
button 0 or 18 22.4 % against 7.0 % (18.1 % on button 0 alone). The buttons do not depend more on the pitch alone (70 % against
73 % of notes on their pitch's usual button; v3: 84 %).
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from button_control_strength import git_line, load_style_model, model_values, passage_inputs, read_passages
from chord_strength_many_chords import CONTEXT, NOTES, PIECES, SEED, TEST_PICKLE
from chord_strength_guidance import bootstrap_ci, permutation_p
from models import get_model_hparams

NEW = sys.argv[1] if len(sys.argv) > 1 else 'AE_style_tester_v3'
PRESENT = sys.argv[2] if len(sys.argv) > 2 else 'AE_style_tester_v2'
SCALE_SIZE = 7  # pitch classes counted as the passage's scale
DEVICE = 'mps'
RESULTS = ROOT / f'test/results/button_map_chromatic__{NEW}.txt'  # one file per evaluated model


def outside_scale(pitch: np.ndarray, notes: np.ndarray) -> np.ndarray:
    """Per passage [P] share (%) of notes [P, N] whose pitch class is not among the SCALE_SIZE most frequent of pitch [P, T]."""
    counts = np.stack([np.bincount(row % 12, minlength=12) for row in pitch])  # [P, 12]
    scale = np.argsort(-counts, axis=1, kind='stable')[:, :SCALE_SIZE]  # [P, SCALE_SIZE]
    inside = (notes[..., None] % 12 == scale[:, None, :]).any(-1)  # [P, N]
    return 100 * (~inside).mean(-1)


@torch.inference_mode()
def map_values(model, data, moments: np.ndarray, device: torch.device) -> dict:
    """Per passage [P] values of one model's button map and encoder."""
    top = model_values(model, data, moments, device)['top']  # [P, 19], most likely note of each button
    pitch = passage_inputs(model, data, moments, device)[0]  # [P, CONTEXT + NOTES]
    position = model.encoder({'pitch': pitch[:, 1:]})  # [P, T - 1], encoder positions of notes 1.. (note 0 is never a target)
    button = model.real_to_discrete(position).cpu().numpy().astype(float)
    position, pitch = position.float().cpu().numpy(), pitch.cpu().numpy()
    last = int(model.cfg['num_buttons']) - 1
    played = pitch[:, 1:].astype(float)
    centred = button - button.mean(1, keepdims=True)
    slope = (centred * (played - played.mean(1, keepdims=True))).sum(1) / np.maximum((centred ** 2).sum(1), 1e-9)
    # Notes on the button their pitch gets most often in the passage: near 100 % = the button depends on the pitch alone.
    usual = [sum(np.bincount(row_button[row_pitch == value].astype(int)).max() for value in np.unique(row_pitch))
             for row_pitch, row_button in zip(played, button)]
    one_step = np.abs(np.diff(button, axis=1)) == 1
    semitone = np.abs(np.diff(played, axis=1)) == 1
    return {'map_outside': outside_scale(pitch[:, :CONTEXT], top),
            'map_semitone': 100 * (np.abs(np.diff(top, axis=1)) == 1).mean(1),
            'real_outside': outside_scale(pitch[:, :CONTEXT], pitch[:, CONTEXT:]),
            'extremes': 100 * ((button == 0) | (button == last)).mean(1),
            'used': np.array([len(np.unique(row)) for row in button], dtype=float),
            'slope': slope,
            'step_semitone': 100 * (one_step & semitone).sum(1) / np.maximum(one_step.sum(1), 1),
            'spread': position.std(1),
            'usual_button': 100 * np.array(usual, dtype=float) / button.shape[1],
            # [P, buttons], share of the passage's notes on each button, %
            'histogram': 100 * np.stack([np.bincount(row.astype(int), minlength=last + 1) for row in button]) / button.shape[1]}


def main() -> None:
    started = time.time()
    device = torch.device(DEVICE)
    data, moments, usable_pieces, pieces_in_pickle = read_passages()
    values = {}
    for role, name in (('present', PRESENT), ('new', NEW)):
        values[role] = map_values(load_style_model(name, device), data, moments, device)
        print(f'{role} model done, {time.time() - started:.0f} s', flush=True)
    present, new = values['present'], values['new']

    rng = np.random.default_rng(SEED)
    claim_lines, verdicts = [], []
    for label, key in (('Claim 1, buttons giving a note outside the scale', 'map_outside'),
                       ('Claim 2, encoder notes on button 0 or 18', 'extremes')):
        difference = new[key] - present[key]
        low, high = bootstrap_ci(difference, rng)
        verdicts.append('supported' if low > 0 else 'refuted')
        claim_lines.append(f'{label}, new minus present: {difference.mean():+.1f} points, 95 % range {low:+.1f} to {high:+.1f}, '
                           f'p = {permutation_p(difference, rng):.1e}, higher in {(difference > 0).sum()} of {len(difference)} '
                           f'passages, lower in {(difference < 0).sum()}.')

    rows = (('map_outside', 'Buttons whose most likely note is outside the scale, %'),
            ('map_semitone', 'Neighbouring buttons a semitone apart, %'),
            ('real_outside', "Pianist's next 8 notes outside the scale, % (reference)"),
            ('extremes', 'Encoder: notes on button 0 or 18, %'),
            ('used', 'Encoder: different buttons used in 207 notes'),
            ('slope', 'Encoder: semitones per button'),
            ('step_semitone', 'Encoder: one-button moves that are a semitone, %'),
            ('spread', 'Encoder: standard deviation of positions (range -1 to 1)'),
            ('usual_button', "Encoder: notes on their pitch's most frequent button, %"))
    report = ['#' * 64, git_line(),
              f'Present: {PRESENT} ({Path(get_model_hparams(PRESENT)["ckpt_file_name"]).name})',
              f'New: {NEW} ({Path(get_model_hparams(NEW)["ckpt_file_name"]).name})  Device: {DEVICE}',
              f'Data: {TEST_PICKLE.name}, {PIECES} of the {usable_pieces} usable pieces ({pieces_in_pickle} in the file), one chord change each. '
              f'Passages of {CONTEXT} + {NOTES} notes, seed {SEED}. Scale: the {SCALE_SIZE} most frequent pitch classes of the {CONTEXT} notes '
              f'before the moment; an even spread over the 12 pitch classes would give {100 * (12 - SCALE_SIZE) / 12:.0f} % outside.',
              f'Running time: {time.time() - started:.0f} s', '',
              f'{"Measure, mean over passages":<58} {"Present":>8} {"New":>8} {"New minus present":>18}']
    for key, label in rows:
        report.append(f'{label:<58} {present[key].mean():>8.2f} {new[key].mean():>8.2f} {(new[key] - present[key]).mean():>+18.2f}')
    report += ['', 'Exploratory: share of the passages\' real notes that the encoder puts on each button, mean over passages, %.',
               f'{"Button":>6} {"Present":>8} {"New":>8}']
    for index in range(present['histogram'].shape[1]):
        report.append(f'{index:>6} {present["histogram"][:, index].mean():>8.1f} {new["histogram"][:, index].mean():>8.1f}')
    report += ['', *claim_lines, f'Result: claim 1 {verdicts[0]}, claim 2 {verdicts[1]}.', '']
    report = '\n'.join(report)
    print(report)
    with open(RESULTS, 'a', encoding='utf-8') as file:
        file.write(report + '\n')


if __name__ == '__main__':
    main()
