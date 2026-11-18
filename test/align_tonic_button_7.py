"""
Question: After the stage 2 fine-tune, does pressing button 7 give the tonic of the passage's key, in the octave of the
Reins layout?
Claim: with button 7 pressed, the new style model gives that reference pitch more probability than the present one.
Supported if: the 95 % range of new minus present lies above zero. Refuted if it reaches zero or lies below it.
Reference: the same probability when the note keeps its own button, the one that model's encoder gives the real note. If it
rises as much, the new model only likes the tonic more, whatever the button.
Models: present AE_style_tester_v2 and new AE_style_tester_tonic_v1 (v2 fine-tuned with the tonic on button 7 and v2's
own button losses, docs/2026-11-11_Button_loss_repair_plan.md, stage 2). The new preset is the first argument; giving the present one checks
the test, since every difference must then be exactly 0. The cost in predicting the pianist's notes is measured by
test/button_control_strength.py with the same two presets.
Data: the same 60 pieces and moments of Training-Data/giantmidi_full_w_chords_unseen_test as test/button_control_strength.py.
Neither model trained on these pieces, and the new one validated on the other unseen split. The key is the RNBert local key that
covers most of the passage's notes, as in training.
Method: each passage is the pianist's 200 notes before a chord change and the 8 after it, with the buttons that model's
encoder gives the whole passage, and the piece's notes next to it as style. The reference pitch at every note follows
pitch_reins.reference_pitches, the policy used in training: button 7 of the Reins layout for that key, the first 24 notes
acting as primer. At each of the 8 notes, with the real notes and buttons before it, the button of that note is set to 7,
then left as the encoder's, and the probability of the reference pitch is read. No sampling.
Measure: that probability with button 7, averaged over the 8 notes of each passage, new minus present, in percentage points.
Exploratory: how often the most likely note of button 7 is the reference pitch, and the share of the passages' reference-pitch
notes that each encoder puts on button 7.
Size and time: 60 passages, 8 notes, 2 buttons per model, 63 s on the M1 (12 s to choose the moments).
Run: python test/align_tonic_button_7.py [new preset, default AE_style_tester_tonic_v1] [present preset, default AE_style_tester_v2]
Results: test/results/align_tonic_button_7__<new preset>.txt, one file per evaluated model.
Result (2026-11-12): check run only, with AE_style_tester_v3 in both roles: every difference is exactly 0. v3 gives the
reference pitch 7.8 % with button 7 and 5.9 % with the note's own button, and its encoder puts 36 of the 789 reference-pitch
notes on button 7 (5 %). Not run yet with the new model. See test/results/align_tonic_button_7__AE_style_tester_v3.txt.
Result (2026-11-12, after stage 1 was reverted): check run only, with AE_style_tester_v2 in both roles: every difference is
exactly 0. v2 gives the reference pitch 11.5 % with button 7 and 5.7 % with the note's own button; button 7's most likely note
is the reference pitch at 13.5 % of the notes, and its encoder puts 136 of the 789 reference-pitch notes on button 7 (17 %).
Not run yet with the new model.
Result (2026-11-13, AE_style_tester_tonic_v1 epoch 129 against AE_style_tester_v2 epoch 150): refuted. The encoder learned the
rule (it puts 53 % of the tonic notes on button 7 against 17 %), but pressing button 7 gives the tonic 13.5 % against 11.5 %
(+2.0 points, 95 % range -4.1 to +8.2, higher in 36 of 60 passages). See test/results/align_tonic_button_7__AE_style_tester_tonic_v1.txt.
Result (2026-11-16, AE_style_tester_tonic_v2 epoch 342 against AE_style_tester_v2 epoch 150): supported. Pressing button 7 gives
the tonic 19.4 % against 11.5 % (+7.9 points, 95 % range +1.5 to +14.3, higher in 43 of 60 passages); with the note's own button
6.2 % against 5.7 %, so the gain belongs to button 7. The encoder puts 69 % of the tonic notes on button 7 against 17 %. Epoch 337
(preset AE_style_tester_tonic_v2_epoch337): supported, 21.5 % (+10.0 points, range +3.1 to +16.9, higher in 43 of 60). Button 7
still holds about 20 % of all notes, of which only about a fifth are the tonic (exploratory). See
test/results/align_tonic_button_7__AE_style_tester_tonic_v2.txt and align_tonic_button_7__AE_style_tester_tonic_v2_epoch337.txt.
Result (2026-11-18, AE_style_tester_frame_v1 best epoch 205 against AE_style_tester_v2 epoch 150): refuted, no measurable
difference, as expected from a loss that knows no key. Pressing button 7 gives the tonic 11.9 % against 11.5 % (+0.4 points,
95 % range -6.9 to +8.7, higher in 20 of 60 passages); the encoder puts 14 % of the tonic notes on button 7 against 17 %. See
test/results/align_tonic_button_7__AE_style_tester_frame_v1.txt.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from button_control_strength import DEVICE, git_line, last_logits, load_style_model, passage_inputs, read_passages
from chord_strength_many_chords import CONTEXT, NOTES, PIECES, SEED, TEST_PICKLE, read_events
from chord_strength_guidance import bootstrap_ci, permutation_p
from chord_strength_style import player_buttons
from models import get_model_hparams
from pitch_reins import reference_pitches
from train_style import local_keys, performed_notes

NEW = sys.argv[1] if len(sys.argv) > 1 else 'AE_style_tester_tonic_v1'
PRESENT = sys.argv[2] if len(sys.argv) > 2 else 'AE_style_tester_v2'
TONIC_BUTTON = 7
RESULTS = ROOT / f'test/results/align_tonic_button_7__{NEW}.txt'  # one file per evaluated model


def passage_references(data, moments: np.ndarray) -> torch.Tensor:
    """Reference pitch [P, CONTEXT + NOTES] at every note of each passage, from the key covering most of its notes."""
    events = torch.from_numpy(read_events(TEST_PICKLE).reshape(-1, 5).astype(np.int64))
    notes = performed_notes(events)
    keys = local_keys(events)[notes].numpy()
    if not np.array_equal(events[notes, 2].numpy(), data.pitches):
        raise SystemExit('The notes of performed_notes differ from those of the chord dataset')
    references = []
    for onset in moments.tolist():
        passage = data.pitches[onset - CONTEXT:onset + NOTES].tolist()
        passage_keys = keys[onset - CONTEXT:onset + NOTES]
        key = int(np.bincount(passage_keys[passage_keys >= 0], minlength=24).argmax())
        references.append(reference_pitches(passage, key % 12, key >= 12, TONIC_BUTTON))
    return torch.tensor(references)


@torch.inference_mode()
def model_values(model, data, moments: np.ndarray, reference: torch.Tensor, device: torch.device) -> dict:
    """Per passage [P] values of one model, and pooled counts of reference-pitch notes."""
    pitch, style, style_mask = passage_inputs(model, data, moments, device)
    reference = reference.to(device)
    buttons = player_buttons(model, pitch)  # [P, CONTEXT + NOTES], as training; button 0 is unused
    is_reference = pitch[:, 1:] == reference[:, 1:]
    on_button = is_reference & (buttons[:, 1:] == TONIC_BUTTON)

    # One window of CONTEXT notes + the target per note after the change: [P * NOTES, CONTEXT + 1].
    windows = lambda values: values.unfold(1, CONTEXT + 1, 1).flatten(0, 1)
    window_pitch, own_button = windows(pitch), windows(buttons)
    target_reference = reference[:, CONTEXT:].reshape(-1, 1)  # [P * NOTES, 1]
    passage = torch.arange(len(pitch), device=device).repeat_interleave(NOTES)
    values = {}
    for name, last_button in (('button 7', TONIC_BUTTON), ('own button', None)):
        window_button = own_button.clone()
        if last_button is not None:
            window_button[:, -1] = last_button
        logits = last_logits(model, window_pitch, window_button, style, style_mask, passage, 1)[:, 0]
        probability = F.softmax(logits, -1).gather(-1, target_reference)[:, 0].view(len(pitch), NOTES)
        values[name] = 100 * probability.mean(-1).cpu().numpy()
        if last_button is not None:
            top = logits.argmax(-1, keepdim=True) == target_reference
            values['top'] = 100 * top.view(len(pitch), NOTES).float().mean(-1).cpu().numpy()
    values.update(references=int(is_reference.sum()), on_button=int(on_button.sum()))
    return values


def main() -> None:
    started = time.time()
    device = torch.device(DEVICE)
    data, moments, usable_pieces, pieces_in_pickle = read_passages()
    reference = passage_references(data, moments)
    print(f'Moments chosen, {time.time() - started:.0f} s', flush=True)
    values = {}
    for role, name in (('present', PRESENT), ('new', NEW)):
        values[role] = model_values(load_style_model(name, device), data, moments, reference, device)
        print(f'{role} model done, {time.time() - started:.0f} s', flush=True)
    present, new = values['present'], values['new']

    rng = np.random.default_rng(SEED)
    difference = new['button 7'] - present['button 7']
    low, high = bootstrap_ci(difference, rng)
    verdict = 'supported' if low > 0 else 'refuted'
    own = new['own button'] - present['own button']
    own_low, own_high = bootstrap_ci(own, rng)

    rows = (('Button 7: probability of the reference pitch, %', 'button 7'),
            ("Note's own button: probability of the reference pitch, %", 'own button'),
            ('Button 7: most likely note is the reference pitch, % of notes', 'top'))
    report = ['#' * 64, git_line(),
              f'Present: {PRESENT} ({Path(get_model_hparams(PRESENT)["ckpt_file_name"]).name})',
              f'New: {NEW} ({Path(get_model_hparams(NEW)["ckpt_file_name"]).name})  Device: {DEVICE}',
              f'Data: {TEST_PICKLE.name}, {PIECES} of the {usable_pieces} usable pieces ({pieces_in_pickle} in the file), one chord change each. '
              f'Passages of {CONTEXT} + {NOTES} notes, seed {SEED}. Reference pitch on button {TONIC_BUTTON}.',
              f'Running time: {time.time() - started:.0f} s', '',
              'At the 8 notes after the change, means over passages.',
              f'{"Measure":<64} {"Present":>8} {"New":>8} {"New minus present":>18}']
    report += [f'{label:<64} {present[key].mean():>8.2f} {new[key].mean():>8.2f} {new[key].mean() - present[key].mean():>+18.2f}'
               for label, key in rows]
    report += ['',
               f'Main, button 7, new minus present: {difference.mean():+.2f} points, 95 % range {low:+.2f} to {high:+.2f}, '
               f'p = {permutation_p(difference, rng):.1e}, higher in {(difference > 0).sum()} of {len(difference)} passages, '
               f'lower in {(difference < 0).sum()}.',
               f"Reference, the note's own button, new minus present: {own.mean():+.2f} points, 95 % range {own_low:+.2f} to {own_high:+.2f}.",
               'Exploratory, reference-pitch notes of the passages that the encoder puts on button 7: '
               + ', '.join(f'{role} {value["on_button"]} of {value["references"]} ({100 * value["on_button"] / max(value["references"], 1):.0f} %)'
                           for role, value in values.items()) + '.']
    if NEW == PRESENT:
        largest = max(np.abs(new[key] - present[key]).max() for _, key in rows)
        largest = max(largest, abs(new['on_button'] - present['on_button']))
        report.append(f'Check run, the same preset in both roles: largest difference over all measures and passages = {largest:g} (must be 0).')
    report += [f'Result: {verdict} (supported if the range lies above zero).', '']
    report = '\n'.join(report)
    print(report)
    with open(RESULTS, 'a', encoding='utf-8') as file:
        file.write(report + '\n')


if __name__ == '__main__':
    main()
