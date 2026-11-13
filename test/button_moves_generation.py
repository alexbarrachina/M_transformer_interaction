"""
Question: While playing, how does the note follow the button move with the repaired button losses: does the same button
repeat the note, do neighbouring buttons wrongly repeat it, and does the note move the way the button moves?
Claim 1 (gain): pressing the same button again repeats the previous note more often with the new model than with the present one.
Supported if: the 95 % range of new minus present lies above zero. Refuted if it reaches zero or lies below it.
Claim 2 (risk): on a button one step from the previous one, the new model repeats the previous note at most 5 percentage points
more often than the present one (5 points = one extra repeat in 20 such moves).
Supported if: the 95 % range of new minus present lies below +5. Refuted if it lies above +5. Undecided if it contains +5.
Claim 3 (risk): on a move of 1 to 4 buttons, the note moves in the button's direction (higher for a higher button) at most
5 points less often with the new model.
Supported if: the 95 % range of new minus present lies above -5. Refuted if it lies below -5. Undecided if it contains -5.
Models: new AE_style_tester_v3 and present AE_style_tester_v2 by default, presets of models.py with their checkpoints (base
style models: buttons and style, no joker; style jokers work too). Each model uses its own encoder for the primer.
Data: 60 pieces of Training-Data/giantmidi_full_w_chords_unseen_test, chosen as in test/chord_strength_many_chords.py but with
at least 33 notes after the chord change, so a few moments differ from test/button_control_strength.py. Neither model trained
on this split (both trained on giantmidi_full_train, and pieces with over 10 % of their 8-note sequences there are left out).
Method: as in the app, the primer is the pianist's 128 notes before the chord change, encoded alone, and the model generates
33 notes over a sliding 128-note window, sampled at temperature 1, with the piece's notes next to the passage as style. An
automatic player presses the same 33 buttons in every piece (PATTERN). Its 32 moves are 11 times the same button, 11 one-step
moves (5 up, 6 down), 6 two-step moves and 4 four-step moves, half up and half down; the first note, whose move from the primer is arbitrary,
is not counted. Both models play the same pattern with the same primer, style and sampling noise; only the model changes.
Measures, one value per piece, in % of that piece's moves of that kind:
  claim 1: same button, the note equals the previous note;
  claim 2: one-step move, the note equals the previous note;
  claim 3: move of 1 to 4 buttons, the note is higher for a higher button and lower for a lower one.
Exploratory: the same measures by move size; on one-step moves, a different note with the previous note's pitch class (the
octave lock heard before); the mean interval in semitones per move size; and the plausibility of the notes, the probability
that a model without buttons (the joker of AE_style_jokerParam_tester_v2, fine-tuned from the present model, so it may favour
its habits) gives each generated note, with the pianist's real 33 notes as reference.
Size and time: 60 pieces, 33 notes, 2 models; 93 s on the M1.
Run: python test/button_moves_generation.py [new preset] [present preset]
Results: test/results/button_moves_generation__<new preset>.txt, one file per evaluated model.
Result (2026-11-12, AE_style_tester_v3 epoch 48 against AE_style_tester_v2 epoch 150): all three claims supported. The same
button repeats the note 80.6 % of the time instead of 6.1 %; one-step moves repeat it as rarely as before (3.8 % against 4.7 %)
and two- and four-step moves never; the note follows the button's direction as often (95.6 % against 93.8 %). The new buttons
take smaller steps: 1.4 instead of 2.2 semitones per one-step move, 4.9 instead of 9.9 per four-step move. The judge finds the
new notes slightly more plausible. See test/results/button_moves_generation__AE_style_tester_v3.txt.
Result (2026-11-12, AE_style_tester_v2_repeat_fix epoch 78 against AE_style_tester_v2): claims 1 and 3 supported, claim 2
undecided. Same button repeats 74.1 % against 6.1 %; one-step moves 6.2 % against 4.7 % (+1.5 points, 95 % range -2.1 to +5.2);
direction 96.7 % against 93.8 %; 1.5 instead of 2.2 semitones per one-step move, 6.2 instead of 9.9 per four-step move.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from button_control_strength import git_line, last_logits, load_style_model
from chord_strength_many_chords import CONTEXT as MOMENT_CONTEXT, NOTES as MOMENT_NOTES
from chord_strength_many_chords import PIECES, SEED, TEST_PICKLE, choose_moments, overlap_per_piece, read_events
from chord_strength_guidance import bootstrap_ci, permutation_p
from chord_strength_style import mode_buttons, player_buttons, style_inputs
from models import get_model_hparams
from train_chords import ChordPitchSamplerDataset

NEW = sys.argv[1] if len(sys.argv) > 1 else 'AE_style_tester_v3'
PRESENT = sys.argv[2] if len(sys.argv) > 2 else 'AE_style_tester_v2'
JUDGE = 'AE_style_jokerParam_tester_v2'  # its joker scores the notes without buttons
# Moves after the first press: 0 +1 +1 0 -1 -1 0 +2 +2 0 -2 -2 0 +4 0 -4 0 -1 -1 0 +1 +1 +2 0 -2 -4 0 +4 +1 -1 0 -1
PATTERN = (9, 9, 10, 11, 11, 10, 9, 9, 11, 13, 13, 11, 9, 9, 13, 13, 9, 9, 8, 7, 7, 8, 9, 11, 11, 9, 5, 5, 9, 10, 9, 9, 8)
GENERATED = len(PATTERN)
CONTEXT = 128  # the app's CTX_LEN on the M1: primer length and sliding window
DEVICE = 'mps'
RESULTS = ROOT / f'test/results/button_moves_generation__{NEW}.txt'  # one file per evaluated model


def read_passages() -> tuple:
    """Real notes [P, CONTEXT + GENERATED] around each chord change, the data, the moments and the piece counts."""
    events = read_events(TEST_PICKLE)
    data = ChordPitchSamplerDataset(events, MOMENT_CONTEXT + MOMENT_NOTES, is_eval=True)
    pieces_in_pickle = len(data.piece_edges) - 1
    moments, usable_pieces = choose_moments(data, overlap_per_piece(events.reshape(-1, 5), pieces_in_pickle), notes=GENERATED)
    real = np.stack([data.pitches[onset - CONTEXT:onset + GENERATED].astype(np.int64) for onset in moments.tolist()])
    return real, data, moments, usable_pieces, pieces_in_pickle


@torch.inference_mode()
def model_style(model, data: ChordPitchSamplerDataset, moments: np.ndarray, device: torch.device) -> tuple:
    """This model's style context [P, S, dim] and mask [P, S]: the piece's notes next to each passage."""
    styles = []
    for onset in moments.tolist():
        piece = data.note_piece[onset]
        start, end = data.piece_edges[piece], data.piece_edges[piece + 1]
        styles.append(style_inputs(model, data.pitches[start:end], onset - CONTEXT - start, onset + GENERATED - start, device))
    return torch.cat([style for style, _ in styles]), torch.cat([mask for _, mask in styles])


@torch.inference_mode()
def generate(model, real: torch.Tensor, style: torch.Tensor, style_mask: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Pitches [P, CONTEXT + GENERATED]: the real primer, then GENERATED notes sampled on the pattern's buttons.
    noise [P, GENERATED, 128] gives Gumbel-max sampling at temperature 1, the same for both models."""
    pitch = real.clone()
    pitch[:, CONTEXT:] = 0
    button = torch.cat((player_buttons(model, real[:, :CONTEXT]),  # the primer encoded alone, as the app's seed
                        torch.tensor(PATTERN, device=real.device).expand(len(real), -1)), 1)
    passage = torch.arange(len(real), device=real.device)
    for step in range(GENERATED):
        window = slice(step, step + CONTEXT + 1)  # the app's sliding window: CONTEXT notes + the target
        logits = last_logits(model, pitch[:, window], button[:, window], style, style_mask, passage, 1)[:, 0]
        pitch[:, CONTEXT + step] = (logits - noise[:, step].log()).argmax(-1)
    return pitch


@torch.inference_mode()
def judge_log_probability(judge, pitch: torch.Tensor, style: torch.Tensor, style_mask: torch.Tensor) -> np.ndarray:
    """Mean log-probability per piece [P] that the judge, holding the joker, gives the last GENERATED notes of pitch."""
    button = mode_buttons(judge, pitch, CONTEXT, 'joker')  # primer encoded alone, joker on the generated notes
    passage = torch.arange(len(pitch), device=pitch.device)
    logits = last_logits(judge, pitch, button, style, style_mask, passage, GENERATED)
    return F.log_softmax(logits, -1).gather(-1, pitch[:, CONTEXT:, None])[..., 0].mean(-1).cpu().numpy()


def move_values(pitch: np.ndarray) -> dict:
    """Per piece [P] shares (%) and mean intervals of the generated notes, by the pattern's move; the first note is skipped."""
    move = np.diff(PATTERN)  # [GENERATED - 1]
    played = pitch[:, CONTEXT:]
    interval = played[:, 1:] - played[:, :-1]  # [P, GENERATED - 1], semitones
    share = lambda hit, where: 100 * hit[:, where].mean(-1)
    follows = np.sign(interval) == np.sign(move)
    values = {'same_repeat': share(interval == 0, move == 0),
              'step_repeat': share(interval == 0, np.abs(move) == 1),
              'direction': share(follows, move != 0),
              'step_octave': share((interval != 0) & (interval % 12 == 0), np.abs(move) == 1)}
    for size in (1, 2, 4):
        values[f'repeat_{size}'] = share(interval == 0, np.abs(move) == size)
        values[f'direction_{size}'] = share(follows, np.abs(move) == size)
        values[f'interval_{size}'] = np.abs(interval[:, np.abs(move) == size]).mean(-1).astype(float)
    return values


def main() -> None:
    started = time.time()
    device = torch.device(DEVICE)
    real, data, moments, usable_pieces, pieces_in_pickle = read_passages()
    real = torch.as_tensor(real, device=device)
    noise = torch.empty(PIECES, GENERATED, 128).exponential_(generator=torch.Generator().manual_seed(SEED)).to(device)
    pitches = {}
    for role, name in (('present', PRESENT), ('new', NEW)):
        model = load_style_model(name, device)
        pitches[role] = generate(model, real, *model_style(model, data, moments, device), noise)
        print(f'{role} model generated, {time.time() - started:.0f} s', flush=True)
        del model
    judge = load_style_model(JUDGE, device)
    judge_style = model_style(judge, data, moments, device)
    log_probability = {role: judge_log_probability(judge, pitch, *judge_style)
                       for role, pitch in (('real', real), *pitches.items())}
    values = {role: move_values(pitch.cpu().numpy()) for role, pitch in pitches.items()}
    present, new = values['present'], values['new']

    rng = np.random.default_rng(SEED)
    claims = (('Claim 1, same button: repeats, new minus present', 'same_repeat', 0, 'above'),
              ('Claim 2, one-step move: repeats, new minus present', 'step_repeat', 5, 'below'),
              ('Claim 3, move of 1-4 buttons: note in the button\'s direction, new minus present', 'direction', -5, 'above'))
    claim_lines, verdicts = [], []
    for label, key, limit, side in claims:
        difference = new[key] - present[key]
        low, high = bootstrap_ci(difference, rng)
        if side == 'above':
            verdict = 'supported' if low > limit else 'refuted' if high < limit or limit == 0 else 'undecided'
        else:
            verdict = 'supported' if high < limit else 'refuted' if low > limit else 'undecided'
        verdicts.append(verdict)
        claim_lines.append(f'{label}: {difference.mean():+.1f} points, 95 % range {low:+.1f} to {high:+.1f}, '
                           f'p = {permutation_p(difference, rng):.1e}, higher in {(difference > 0).sum()} of {len(difference)} '
                           f'pieces, lower in {(difference < 0).sum()}. {verdict.capitalize()} (limit {limit:+d}, range must lie {side}).')

    rows = (('same_repeat', 'Same button: previous note repeated, %'),
            ('repeat_1', 'One-step move: previous note repeated, %'),
            ('repeat_2', 'Two-step move: previous note repeated, %'),
            ('repeat_4', 'Four-step move: previous note repeated, %'),
            ('step_octave', 'One-step move: same pitch class, other octave, %'),
            ('direction', 'Move of 1-4 buttons: note in its direction, %'),
            ('direction_1', 'One-step move: note in its direction, %'),
            ('direction_2', 'Two-step move: note in its direction, %'),
            ('direction_4', 'Four-step move: note in its direction, %'),
            ('interval_1', 'One-step move: mean interval, semitones'),
            ('interval_2', 'Two-step move: mean interval, semitones'),
            ('interval_4', 'Four-step move: mean interval, semitones'))
    judge_probability = {role: 100 * np.exp(value.mean()) for role, value in log_probability.items()}
    judge_difference = log_probability['new'] - log_probability['present']
    judge_low, judge_high = bootstrap_ci(judge_difference, rng)
    report = ['#' * 64, git_line(),
              f'Present: {PRESENT} ({Path(get_model_hparams(PRESENT)["ckpt_file_name"]).name})',
              f'New: {NEW} ({Path(get_model_hparams(NEW)["ckpt_file_name"]).name})',
              f'Judge without buttons: {JUDGE} ({Path(get_model_hparams(JUDGE)["ckpt_file_name"]).name}), joker held  Device: {DEVICE}',
              f'Data: {TEST_PICKLE.name}, {PIECES} of the {usable_pieces} usable pieces ({pieces_in_pickle} in the file), one chord change each. '
              f'Primer and window {CONTEXT} notes, {GENERATED} generated notes on the pattern {PATTERN}, temperature 1, seed {SEED}.',
              f'Running time: {time.time() - started:.0f} s', '',
              'Means over pieces of the per-piece values; moves counted from the second generated note on.',
              f'{"Measure":<52} {"Present":>8} {"New":>8} {"New minus present":>18}']
    for key, label in rows:
        report.append(f'{label:<52} {present[key].mean():>8.1f} {new[key].mean():>8.1f} {(new[key] - present[key]).mean():>+18.1f}')
    report += ['',
               'Plausibility (exploratory): typical probability the judge gives a note, e^(mean log-probability), higher is more plausible:',
               f'  pianist\'s real notes {judge_probability["real"]:.1f} %, present model {judge_probability["present"]:.1f} %, '
               f'new model {judge_probability["new"]:.1f} %. New minus present: {judge_difference.mean():+.3f} nats per note, '
               f'95 % range {judge_low:+.3f} to {judge_high:+.3f}.', '',
               *claim_lines,
               f'Result: claim 1 {verdicts[0]}, claim 2 {verdicts[1]}, claim 3 {verdicts[2]}.', '']
    report = '\n'.join(report)
    print(report)
    with open(RESULTS, 'a', encoding='utf-8') as file:
        file.write(report + '\n')


if __name__ == '__main__':
    main()
