"""
Question: When a player's rising runs never reach the top button, does the pitch climb from run to run?
In AE_style_tester_frame_v1's training, every 33-note frame (the 16 notes before a note, the note and the 16 after) has its
highest note on the top button and its lowest on button 0, and the buttons of earlier notes know the 16 notes after them.
Runs that stop below the top button may then tell its decoder that the frame's highest note is still to come and higher than
anything played: the decoder plays a higher note on a later press, and the next run, again stopping below the top, asks for
more. The user heard the pitch run up to the top of the keyboard in the app.
Claim 1 (main): with rising runs that stop at button 11, the new model's highest notes climb more from run to run than the
present model's.
Supported if: the 95 % range of new minus present lies above zero. Refuted if it lies below zero. Undecided if it contains zero.
Claim 2 (the cause): the new model climbs less when the same runs are stretched to reach both end buttons.
Supported if: the 95 % range of the new model's climb stopping at 11 minus its climb stretched lies above zero. Refuted if it
lies below zero. Undecided if it contains zero.
Both claims come from an exploratory run with the app's Chopin primer and from the app's recordings in out/ (2026-11-19);
this test plays other pieces.
Models: new AE_style_tester_frame_v1 and present AE_style_tester_v2 by default, presets of models.py with their checkpoints.
Base style models (AE_style_*) and style jokers (AE_style_jokerParam_*) work; each model uses its own encoder for the primer.
Data: the first 20 of the 60 moments of Training-Data/giantmidi_full_w_chords_unseen_test shared by the other tests, one per
piece; neither default model trained on this split. Only 20, because generating costs about 15 ms per passage and note.
Method: as in the app and test/button_moves_generation.py, the primer is the pianist's 128 notes before the moment, encoded
alone, and the model generates over a sliding 128-note window, sampled at temperature 1, with the piece's other notes as
style. An automatic player presses 6 runs of 7 buttons:
  stopping at 11: 1 3 5 7 9 10 11, the runs of out/2_interactive_buttons_style;
  stretched: 0 3 6 9 12 15 18, the same rising shape reaching both end buttons.
These buttons are for 19 buttons and are scaled to the model's own number. Both models play both conditions with the same
primer, style and sampling noise; only the model or the buttons change.
Measure: the climb, the mean highest note of the last 2 runs minus that of the first 2 runs, in semitones, one value per
passage. Claim 1: new minus present, stopping at 11. Claim 2: the new model, stopping at 11 minus stretched.
Exploratory: the share of notes at pitch 100 (E7) or higher, the top octave of the piano; the median highest note of each run.
Size and time: 20 passages, 42 notes, 2 conditions, 2 models, 149 s on the M1 (12 s of it to choose the moments); that day one
generation step of the 20 passages took 0.3 to 1.1 s, so the time varies with the machine's state.
Run: python test/button_run_drift.py [new preset] [present preset]
Results: test/results/button_run_drift__<new preset>.txt, one file per evaluated model.
Result (2026-11-20, check run, AE_style_tester_v2 in both roles): every difference is exactly 0. v2 does not climb: with runs
stopping at 11 the median highest note per run stays at 72-74 (climb +0.2 semitones), stretched at 86-89 (+0.9); its own
climb stopping at 11 minus stretched is -0.7 semitones (95 % range -2.2 to +1.0). Not run yet with AE_style_tester_frame_v1.
See test/results/button_run_drift__AE_style_tester_v2.txt.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from button_control_strength import git_line, load_style_model, read_passages
from button_moves_generation import CONTEXT, generate, model_style
from chord_strength_many_chords import PIECES, SEED, TEST_PICKLE
from chord_strength_guidance import bootstrap_ci, permutation_p
from models import get_model_hparams

NEW = sys.argv[1] if len(sys.argv) > 1 else 'AE_style_tester_frame_v1'
PRESENT = sys.argv[2] if len(sys.argv) > 2 else 'AE_style_tester_v2'
PASSAGES = 20  # the first of the 60 shared moments
RUNS = 6
COMPARED = 2  # first and last runs whose highest notes give the climb
STOPPING, STRETCHED = 'stopping at 11', 'stretched'
CONDITIONS = {STOPPING: (1, 3, 5, 7, 9, 10, 11),  # the rising runs of out/2_interactive_buttons_style
              STRETCHED: (0, 3, 6, 9, 12, 15, 18)}  # the same shape reaching both end buttons
RUN_BUTTONS = 19  # the buttons of CONDITIONS are for 19 buttons
HIGH = 100  # E7, the top octave of the piano
DEVICE = 'mps'
RESULTS = ROOT / f'test/results/button_run_drift__{NEW}.txt'  # one file per evaluated model


def run_values(pitch: np.ndarray) -> dict:
    """Per passage values of the generated notes pitch [P, RUNS * run length]."""
    tops = pitch.reshape(len(pitch), RUNS, -1).max(-1)  # [P, RUNS], the highest note of each run
    return {'climb': tops[:, -COMPARED:].mean(-1) - tops[:, :COMPARED].mean(-1),
            'high': 100 * (pitch >= HIGH).mean(-1),
            'tops': tops}


@torch.inference_mode()
def model_values(name: str, data, moments: np.ndarray, noise: torch.Tensor, device: torch.device) -> dict:
    """run_values of one model in each condition."""
    model = load_style_model(name, device)
    top = get_model_hparams(name)['num_buttons'] - 1
    primer = torch.as_tensor(np.stack([data.pitches[onset - CONTEXT:onset] for onset in moments.tolist()]).astype(np.int64),
                             device=device)  # [P, CONTEXT], the pianist's notes before each moment
    style, style_mask = model_style(model, data, moments, device)
    values = {}
    for condition, run in CONDITIONS.items():
        pattern = tuple(round(button * top / (RUN_BUTTONS - 1)) for button in run) * RUNS
        values[condition] = run_values(generate(model, primer, style, style_mask, noise, pattern)[:, CONTEXT:].cpu().numpy())
    return values


def main() -> None:
    started = time.time()
    device = torch.device(DEVICE)
    data, moments, usable_pieces, pieces_in_pickle = read_passages()
    moments = moments[:PASSAGES]
    notes = RUNS * len(CONDITIONS[STOPPING])
    # Gumbel-max noise, the same for both models and both conditions
    noise = torch.empty(PASSAGES, notes, 128).exponential_(generator=torch.Generator().manual_seed(SEED)).to(device)
    print(f'Moments chosen, {time.time() - started:.0f} s', flush=True)

    values = {}
    for role, name in (('present', PRESENT), ('new', NEW)):
        values[role] = model_values(name, data, moments, noise, device)
        torch.mps.empty_cache()
        print(f'{role} model done, {time.time() - started:.0f} s', flush=True)
    present, new = values['present'], values['new']

    rng = np.random.default_rng(SEED)
    claim_lines, verdicts = [], []
    for label, difference in (('Claim 1, climb stopping at 11, new minus present',
                               new[STOPPING]['climb'] - present[STOPPING]['climb']),
                              ('Claim 2, the new model\'s climb, stopping at 11 minus stretched',
                               new[STOPPING]['climb'] - new[STRETCHED]['climb'])):
        low, high = bootstrap_ci(difference, rng)
        verdicts.append('supported' if low > 0 else 'refuted' if high < 0 else 'undecided')
        claim_lines.append(f'{label}: {difference.mean():+.1f} semitones, 95 % range {low:+.1f} to {high:+.1f}, '
                           f'p = {permutation_p(difference, rng):.1e}, higher in {(difference > 0).sum()} of {len(difference)} '
                           f'passages, lower in {(difference < 0).sum()}.')

    report = ['#' * 64, git_line(),
              f'Present: {PRESENT} ({Path(get_model_hparams(PRESENT)["ckpt_file_name"]).name})',
              f'New: {NEW} ({Path(get_model_hparams(NEW)["ckpt_file_name"]).name})  Device: {DEVICE}',
              f'Data: {TEST_PICKLE.name}, the first {PASSAGES} of the {PIECES} shared moments ({usable_pieces} usable pieces, '
              f'{pieces_in_pickle} in the file), seed {SEED}. Primer and window {CONTEXT} notes, {RUNS} runs of '
              f'{len(CONDITIONS[STOPPING])} presses, temperature 1. Runs for {RUN_BUTTONS} buttons: '
              + '; '.join(f'{condition} {" ".join(map(str, run))}' for condition, run in CONDITIONS.items()) + '.',
              f'Running time: {time.time() - started:.0f} s', '',
              f'Means over passages. Climb: mean highest note of the last {COMPARED} runs minus the first {COMPARED}.',
              f'{"Measure":<52} {"Present":>8} {"New":>8} {"New minus present":>18}']
    for condition in CONDITIONS:
        for key, label in (('climb', 'climb, semitones'), ('high', f'notes at pitch {HIGH} or higher, %')):
            report.append(f'{condition + ", " + label:<52} {present[condition][key].mean():>8.2f} '
                          f'{new[condition][key].mean():>8.2f} {(new[condition][key] - present[condition][key]).mean():>+18.2f}')
    report += ['', 'Exploratory, median over passages of the highest note of each run:']
    for condition in CONDITIONS:
        for role, value in values.items():
            report.append(f'  {condition:<16} {role:<8} ' + ' '.join(f'{v:>4.0f}' for v in np.median(value[condition]['tops'], 0)))
    report += ['', *claim_lines]
    if NEW == PRESENT:
        largest = max(np.abs(new[condition][key] - present[condition][key]).max()
                      for condition in CONDITIONS for key in ('climb', 'high', 'tops'))
        report.append(f'Check run, the same preset in both roles: largest difference over all measures and passages = {largest:g} (must be 0).')
    report += [f'Result: claim 1 {verdicts[0]}, claim 2 {verdicts[1]}.', '']
    report = '\n'.join(report)
    print(report)
    with open(RESULTS, 'a', encoding='utf-8') as file:
        file.write(report + '\n')


if __name__ == '__main__':
    main()
