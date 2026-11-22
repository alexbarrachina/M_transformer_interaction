"""
Question: How faithfully does a model's encoder compress a passage's pitches into buttons?
The encoder gives the buttons the decoder learns to follow, so its map is what a player has to learn: whether a higher button
means a higher note, whether the buttons keep the notes' relative heights, and whether the end buttons hold many different notes.
Claim 1 (main): the new encoder keeps the melodic contour better than the present one: more consecutive note pairs move the same
way in buttons as in pitch (both up, both down, or both staying).
Claim 2: the new encoder keeps the notes' relative heights better: within windows of 32 notes, a note's place between the
window's lowest and highest button is closer to its place between the window's lowest and highest pitch.
Claim 3: the new encoder saturates less: it puts fewer of the passage's notes on the first or the last button.
Each claim is supported if the 95 % range of new minus present lies on the better side of zero, refuted if it lies on the worse
side, and undecided if it contains zero.
Models: new AE_style_tester_v2_repeat_fix (the app's default model) and present AE_style_tester_v2 by default, presets of
models.py with their checkpoints. Any style model with a button encoder works (AE_style_*, AE_style_jokerParam_*); only the
encoder is used.
Data: the same 60 pieces and moments of Training-Data/giantmidi_full_w_chords_unseen_test as test/button_map_chromatic.py.
Neither model trained on this split.
Reference: a linear map, which spreads each passage's pitch range evenly over the buttons and rounds (the lowest pitch on the
first button, the highest on the last). It keeps the order of the pitches and uses the whole button range, so it shows what the
rounding to this many buttons costs by itself.
Method: the encoder reads the passage's 207 notes (the 208-note passage without its first note, as in training) together, and
its buttons are compared with their pitches. No decoder, no sampling.
Measures, one value per passage:
  claim 1: share of the 206 consecutive note pairs whose button moves the same way as the pitch, %; higher is better;
  claim 2: distance between a note's height in pitch and its height in buttons, each rescaled from 0 (the window's lowest) to 1
           (its highest), in % of that range, averaged over the notes of 6 windows of 32 notes; lower is better. A window
           whose notes all have one button gets height 0.5 for all of them;
  claim 3: share of the passage's notes on the first or the last button, %; lower is better. This is claim 2 of
           test/button_map_chromatic.py on the same passages, so the two tests must give the same numbers.
Exploratory: claim 1 split into pitch moves the buttons keep, lose (same button) or reverse, and pitch repeats the buttons
keep; claim 2 over the whole passage instead of windows, and for the two distributions alone, ignoring which note is which
(distance between the sorted heights, the 1-D earth mover's distance); the mean button height of the notes in each tenth of the
pitch height (the linear map gives a straight line, saturation an S shape); encoder outputs outside -1 to 1, which the rounding
clamps onto the end buttons; and the number of different pitches on the end buttons.
Figures: for every MIDI file in test/samples, its pitches above the new model's buttons, the first 512 notes (the encoder's
length) encoded together as the app encodes its primer, in test/results/button_map_compression__<new preset>/<file>.jpg.
Where these files come from, and whether the models trained on them, is unknown: the figures illustrate the map, they are not
evidence.
Size and time: 60 passages, 2 models, encoder only, 10 figures; 17 s on the M1 (12 s of it to choose the moments).
Run: python test/button_map_compression.py [new preset] [present preset]
Results: test/results/button_map_compression__<new preset>.txt, one file per evaluated model.
Result (2026-11-13, check run, AE_style_tester_v2 in both roles): every difference is exactly 0. The v2 encoder moves the
button the same way as the pitch in 94.4 % of consecutive note pairs, as often as the linear map (94.2 %), but keeps the button
on a repeated pitch only 26 % of the time; heights are off by 9.3 % of the window range (linear map 2.2 %); 7.0 % of the notes
are on an end button (linear map 2.7 %). See test/results/button_map_compression__AE_style_tester_v2.txt.
Result (2026-11-13, AE_style_tester_v2_repeat_fix epoch 78 against AE_style_tester_v2 epoch 150): claim 1 undecided, claims 2
and 3 refuted. Same direction in 94.5 % against 94.4 % of note pairs (95 % range of the difference -0.9 to +1.0 points): it
keeps the button on 98 % of repeated pitches instead of 26 %, but leaves 5.5 % of pitch moves on one button instead of 4.0 %.
Heights off by 15.2 % against 9.3 % of the window range, worse in 59 of 60 passages: the lower half of the pitches is squeezed
onto the lowest buttons. 22.4 % against 7.0 % of the notes on an end button, worse in all 60 passages.
Result (2026-11-13, AE_style_tester_v3 epoch 48 against AE_style_tester_v2): all three claims refuted. Same direction in 84.9 %
against 94.4 % of note pairs, mostly because 15.0 % of pitch moves stay on one button; heights off by 15.1 % against 9.3 %;
38.3 % against 7.0 % of the notes on an end button, worse in all 60 passages; both ends of the pitch range are squeezed.
Result (2026-11-13, AE_style_tester_tonic_v1 epoch 129 against AE_style_tester_v2 epoch 150): claims 1 and 2 refuted, claim 3
undecided. Same direction in 91.0 % of note pairs against 94.4 % (-3.4 points, 95 % range -4.4 to -2.5, worse in 53 of 60
passages), because 7.8 % of pitch moves keep the button against 4.0 %; heights off by 11.6 % against 9.3 %; end buttons 7.2 %
against 7.0 %. The figures show notes near button 7 squeezed onto a few buttons. See
test/results/button_map_compression__AE_style_tester_tonic_v1.txt.
Result (2026-11-16, AE_style_tester_tonic_v2 epoch 342 against AE_style_tester_v2 epoch 150): all three claims refuted, a little
worse than tonic_v1. Same direction in 90.3 % of note pairs against 94.4 % (-4.1 points, 95 % range -5.5 to -2.9, worse in 49 of
60 passages), because 8.9 % of pitch moves keep the button against 4.0 %; heights off by 11.9 % against 9.3 %; 9.3 % against
7.0 % of the notes on an end button (+2.4, range +1.5 to +3.2). In test10 (a C major scale) C, D, E and F share button 7. The
epoch 337 checkpoint of the same run (preset AE_style_tester_tonic_v2_epoch337) gives the same verdicts with differences 0.2 to
0.6 points smaller. See test/results/button_map_compression__AE_style_tester_tonic_v2.txt and
test/results/button_map_compression__AE_style_tester_tonic_v2_epoch337.txt.
Result (2026-11-18, AE_style_tester_frame_v1 best epoch 205 against AE_style_tester_v2 epoch 150): claims 1 and 2 supported,
claim 3 refuted. Same direction in 96.1 % of note pairs against 94.4 % (+1.7 points, 95 % range +0.9 to +2.4, better in 47 of 60
passages), better than the linear map (94.2 %) because it keeps the button on 84 % of repeated pitches against 26 %; heights off
by 5.4 % against 9.3 % of the window range (better in 55 of 60); but 10.1 % against 7.0 % of the notes on an end button (+3.1,
range +2.2 to +4.2, worse in 45 of 60). Pitch moves reversed in buttons 0.8 % against 0.4 %. See
test/results/button_map_compression__AE_style_tester_frame_v1.txt.
Result (2026-11-22, AE_style_tester_frame_v2 best epoch 308 against AE_style_tester_v2 epoch 150): claims 1 and 2 supported,
claim 3 refuted, as frame_v1. Contour 95.4 % against 94.4 % (+1.0 points, 95 % range +0.3 to +1.7, better in 39 of 60
passages); height distance 6.3 % against 9.3 % (-3.0 points, range -3.9 to -2.2, better in 50 of 60); notes on the first or
last button 8.7 % against 7.0 % (+1.7 points, range +1.1 to +2.5); frame_v1 had 96.1, 5.4 and 10.1 %. Repeated pitches keep
their button 88 % of the time against 26 %. In the figures the button peaks one note before the pitch at the top of the runs
of test1 and test10 (test10: notes 81, 83, 84 on buttons 15, 13, 14, where the past-only target gives 12, 14, 15), also when
each note is encoded with only the notes before it (exploratory). See test/results/button_map_compression__AE_style_tester_frame_v2.txt
and its folder of figures.
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # files only, no window
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from button_control_strength import git_line, load_style_model, read_passages
from chord_strength_many_chords import CONTEXT, NOTES, PIECES, SEED, TEST_PICKLE
from chord_strength_guidance import bootstrap_ci, permutation_p
from midiUtils import midi_to_dict
from models import get_model_hparams

NEW = sys.argv[1] if len(sys.argv) > 1 else 'AE_style_tester_v2_repeat_fix'
PRESENT = sys.argv[2] if len(sys.argv) > 2 else 'AE_style_tester_v2'
WINDOW = 32  # notes per window for claim 2
DEVICE = 'mps'
SAMPLES = ROOT / 'test/samples'
RESULTS = ROOT / f'test/results/button_map_compression__{NEW}.txt'  # one file per evaluated model
FIGURES = ROOT / f'test/results/button_map_compression__{NEW}'  # one .jpg per MIDI file of SAMPLES
MARK, BAND, INK, MUTED = '#2a78d6', '#ecebe8', '#0b0b0b', '#52514e'  # figure colours: notes, end buttons, text, axes


def heights(values: np.ndarray) -> np.ndarray:
    """values [..., N] rescaled along the last axis to 0 (lowest) .. 1 (highest); 0.5 where all N values are equal."""
    values = values.astype(float)
    low, high = values.min(-1, keepdims=True), values.max(-1, keepdims=True)
    return np.where(high > low, (values - low) / np.maximum(high - low, 1e-9), 0.5)


def map_measures(pitch: np.ndarray, button: np.ndarray, position: np.ndarray, last: int) -> dict:
    """Per passage [P] measures of the buttons [P, N] given to the pitches [P, N]; position [P, N] is the encoder's output
    before rounding (-1 = first button, 1 = last); last is the last button."""
    pitch_step, button_step = np.sign(np.diff(pitch, axis=1)), np.sign(np.diff(button, axis=1))  # [P, N - 1]: -1, 0 or 1
    moves = pitch_step != 0
    # % of the pairs in `of` that are in `hit`; nan for a passage without such pairs (some have no repeated pitch)
    share = lambda hit, of: np.where(of.any(1), 100 * (hit & of).sum(1) / np.maximum(of.sum(1), 1), np.nan)
    windows = lambda values: values[:, :values.shape[1] // WINDOW * WINDOW].reshape(len(values), -1, WINDOW)  # [P, W, WINDOW]
    pitch_height, button_height = heights(windows(pitch)), heights(windows(button))  # [P, W, WINDOW]
    # Mean button height of the notes in each tenth of the pitch height, [P, 10]; nan where a passage has no note in a tenth.
    tenth = np.minimum((pitch_height * 10).astype(int), 9).reshape(len(pitch), -1, 1) == np.arange(10)  # [P, W * WINDOW, 10]
    with np.errstate(invalid='ignore'):
        curve = (tenth * button_height.reshape(len(pitch), -1, 1)).sum(1) / tenth.sum(1)
    return {'contour': 100 * (pitch_step == button_step).mean(1),
            'moves_kept': share(button_step == pitch_step, moves),
            'moves_lost': share(button_step == 0, moves),
            'moves_reversed': share(button_step == -pitch_step, moves),
            'repeats_kept': share(button_step == 0, ~moves),
            'height': 100 * np.abs(pitch_height - button_height).mean((1, 2)),
            'height_whole': 100 * np.abs(heights(pitch) - heights(button)).mean(1),
            'distribution': 100 * np.abs(np.sort(pitch_height, -1) - np.sort(button_height, -1)).mean((1, 2)),
            'ends': 100 * ((button == 0) | (button == last)).mean(1),
            'clamped': 100 * (np.abs(position) > 1).mean(1),
            'end_pitches': np.array([len(np.unique(row_pitch[row_button == 0])) + len(np.unique(row_pitch[row_button == last]))
                                     for row_pitch, row_button in zip(pitch, button)], dtype=float),
            'curve': curve}


@torch.inference_mode()
def encode(model, pitch: np.ndarray, device: torch.device) -> tuple:
    """Buttons and encoder outputs [P, N] of the pitches [P, N], each row encoded together."""
    position = model.encoder({'pitch': torch.as_tensor(pitch.astype(np.int64), device=device)})
    return model.real_to_discrete(position).cpu().numpy(), position.float().cpu().numpy()


def plot_sample(pitch: np.ndarray, button: np.ndarray, last: int, title: str, path: Path) -> None:
    """The pitches [N] of a file above their buttons [N], note by note."""
    notes = np.arange(len(pitch))
    figure, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(max(10.0, len(pitch) / 30), 6), constrained_layout=True)
    bottom.axhspan(-0.5, 0.5, color=BAND, linewidth=0)
    bottom.axhspan(last - 0.5, last + 0.5, color=BAND, linewidth=0)
    for axis, values, label in ((top, pitch, 'Pitch (MIDI note number)'), (bottom, button, 'Button (end buttons shaded)')):
        axis.plot(notes, values, color=MARK, linewidth=1, marker='o', markersize=3)
        axis.set_ylabel(label, color=MUTED)
        axis.grid(axis='y', color=BAND, linewidth=0.8)
        axis.spines[['top', 'right']].set_visible(False)
        axis.spines[['left', 'bottom']].set_color(MUTED)
        axis.tick_params(colors=MUTED)
    bottom.set_ylim(-0.6, last + 0.6)
    bottom.set_yticks(range(0, last + 1, 2))
    bottom.set_xlabel('Note, in the order of the file', color=MUTED)
    top.set_title(title, loc='left', color=INK, fontsize=10)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> None:
    started = time.time()
    device = torch.device(DEVICE)
    data, moments, usable_pieces, pieces_in_pickle = read_passages()
    # [P, CONTEXT + NOTES - 1]: each passage without its first note, which the encoder never sees in training.
    pitch = np.stack([data.pitches[onset - CONTEXT + 1:onset + NOTES] for onset in moments.tolist()]).astype(np.int64)
    values = {}
    for role, name in (('present', PRESENT), ('new', NEW)):
        model = load_style_model(name, device)  # the new model stays loaded for the figures
        last = int(model.cfg['num_buttons']) - 1
        values[role] = map_measures(pitch, *encode(model, pitch, device), last)
        print(f'{role} model done, {time.time() - started:.0f} s', flush=True)
    linear = np.round(heights(pitch) * last)  # the reference, with the new model's number of buttons
    values['linear'] = map_measures(pitch, linear, linear / last * 2 - 1, last)
    present, new = values['present'], values['new']

    FIGURES.mkdir(parents=True, exist_ok=True)
    samples = sorted(SAMPLES.glob('*.mid*'))
    for path in samples:
        sample_pitch = np.array(midi_to_dict(str(path))[0]['pitch'][:int(model.cfg['seq_len'])])
        sample_button, sample_position = encode(model, sample_pitch[None], device)
        sample = {key: value[0] for key, value in map_measures(sample_pitch[None], sample_button, sample_position, last).items()}
        title = (f'{path.name}, {NEW}: {len(sample_pitch)} notes\nContour kept in {sample["contour"]:.0f} % of note pairs; '
                 f'height off by {sample["height"]:.0f} % of the window range; '
                 f'{sample["ends"]:.0f} % of notes on an end button')
        plot_sample(sample_pitch, sample_button[0], last, title, FIGURES / f'{path.stem}.jpg')
    print(f'{len(samples)} figures written to {FIGURES.relative_to(ROOT)}, {time.time() - started:.0f} s', flush=True)

    rng = np.random.default_rng(SEED)
    claim_lines, verdicts = [], []
    for label, key, better in (('Claim 1, note pairs moving the same way in buttons as in pitch', 'contour', 1),
                               ('Claim 2, distance between pitch height and button height', 'height', -1),
                               ('Claim 3, notes on the first or last button', 'ends', -1)):
        difference = new[key] - present[key]
        low, high = bootstrap_ci(difference, rng)
        side = better * np.array([low, high])  # positive = the new encoder is more faithful
        verdicts.append('supported' if side.min() > 0 else 'refuted' if side.max() < 0 else 'undecided')
        claim_lines.append(f'{label}, new minus present: {difference.mean():+.1f} points, 95 % range {low:+.1f} to {high:+.1f}, '
                           f'p = {permutation_p(difference, rng):.1e}, better in {(better * difference > 0).sum()} of '
                           f'{len(difference)} passages, worse in {(better * difference < 0).sum()}.')

    rows = (('contour', 'Claim 1: note pairs moving the same way, %'),
            ('moves_kept', '  of pitch moves: button moves the same way, %'),
            ('moves_lost', '  of pitch moves: button stays, %'),
            ('moves_reversed', '  of pitch moves: button moves the other way, %'),
            ('repeats_kept', '  of repeated pitches: button stays, %'),
            ('height', f'Claim 2: height distance, {WINDOW}-note windows, %'),
            ('height_whole', '  over the whole passage, %'),
            ('distribution', '  distributions alone (sorted heights), %'),
            ('ends', 'Claim 3: notes on the first or last button, %'),
            ('clamped', '  encoder outputs outside -1 to 1, %'),
            ('end_pitches', '  different pitches on the end buttons'))
    report = ['#' * 64, git_line(),
              f'Present: {PRESENT} ({Path(get_model_hparams(PRESENT)["ckpt_file_name"]).name})',
              f'New: {NEW} ({Path(get_model_hparams(NEW)["ckpt_file_name"]).name})  Device: {DEVICE}',
              f'Data: {TEST_PICKLE.name}, {PIECES} of the {usable_pieces} usable pieces ({pieces_in_pickle} in the file), one chord change each. '
              f'The encoder reads the {pitch.shape[1]} notes of each passage (seed {SEED}); windows of {WINDOW} notes for claim 2; '
              f'{last + 1} buttons. Linear map: each passage\'s pitch range spread evenly over the buttons and rounded.',
              f'Figures: {len(samples)} files of {SAMPLES.relative_to(ROOT)} in {FIGURES.relative_to(ROOT)}',
              f'Running time: {time.time() - started:.0f} s', '',
              f'{"Measure, mean over passages":<50} {"Linear map":>10} {"Present":>8} {"New":>8} {"New minus present":>18}']
    for key, label in rows:
        report.append(f'{label:<50} {np.nanmean(values["linear"][key]):>10.2f} {np.nanmean(present[key]):>8.2f} '
                      f'{np.nanmean(new[key]):>8.2f} {np.nanmean(new[key] - present[key]):>+18.2f}')
    report += ['', f'Exploratory: mean button height (0 = the window\'s lowest button, 1 = its highest) of the notes in each tenth of '
                   f'the pitch height, {WINDOW}-note windows, mean over passages.',
               f'{"Pitch height":<14} {"Linear map":>10} {"Present":>8} {"New":>8}']
    for index in range(10):
        report.append(f'{f"{index / 10:.1f} to {(index + 1) / 10:.1f}":<14} '
                      + ' '.join(f'{np.nanmean(values[role]["curve"][:, index]):>{width}.2f}'
                                 for role, width in (('linear', 10), ('present', 8), ('new', 8))))
    report += ['', *claim_lines]
    if NEW == PRESENT:
        largest = max(np.nanmax(np.abs(new[key] - present[key])) for key, _ in rows)
        report.append(f'Check run, the same preset in both roles: largest difference over all measures and passages = {largest:g} (must be 0).')
    report += [f'Result: claim 1 {verdicts[0]}, claim 2 {verdicts[1]}, claim 3 {verdicts[2]}.', '']
    report = '\n'.join(report)
    print(report)
    with open(RESULTS, 'a', encoding='utf-8') as file:
        file.write(report + '\n')


if __name__ == '__main__':
    main()
