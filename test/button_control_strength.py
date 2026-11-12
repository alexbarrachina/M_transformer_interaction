"""
Question: Without the rule that pushed buttons to move with the pitch, does a button still tell the style joker which note to play?
Claim: the new style joker predicts the pianist's real next note from its button no worse than the present one.
Supported if: the 95 % range of new minus present lies above -5 percentage points. Refuted if it lies below -5. Undecided if it contains -5.
Models: present AE_style_jokerParam_tester_v2 (trained with the old button losses) and new AE_style_jokerParam_tester_v3
(repaired button losses, docs/Button_loss_repair_plan.md), presets of models.py with their checkpoints. The new preset is the
first argument; giving the present one checks the test, since every difference must then be exactly 0.
Data: the same 60 pieces and moments of Training-Data/giantmidi_full_w_chords_unseen_test as test/style_button_repeat_cause.py,
which leaves out pieces with over 10 % of their 8-note sequences in giantmidi_full_train (both models' training data) or in the
chord adapters' training data. The chord adapters used this split for validation; neither style joker trained on it.
Method: each passage is the pianist's 200 notes before a chord change and the 8 after it, with the piece's notes next to it as
style. Button known: every note has the button that model's encoder gives the whole passage, as in training. No button: the
200 notes are encoded alone and the 8 notes get the joker, as in the app. Each model uses its own encoder. No sampling.
Measure: probability given to the pianist's real note at the 8 notes, averaged per passage, new minus present, in percentage points.
Reference: the same probability with no button, for both models.
Exploratory: at the first of the 8 notes, with the real notes and buttons before it, each of the 19 buttons in turn: how many
different most likely notes they give, how often two neighbouring buttons give the same most likely note or a lower one, and
the distance in semitones from button 0's most likely note to button 18's.
Size and time: 60 passages, 2 button conditions and 19 buttons per model, 83 s on the M1 (12 s to choose the moments, most of
the rest for the 19-button sweep).
Base style models (AE_style, no joker) can be compared too; they have no "no button" reference.
Run: python test/button_control_strength.py [new preset, default AE_style_jokerParam_tester_v3] [present preset, default AE_style_jokerParam_tester_v2]
Result (2026-11-11): check run only, with the present model in both roles: every difference is exactly 0. The present model gives
the real note 64.7 % with its button and 22.9 % without one; its 19 buttons give 15.0 different notes. Not run yet with the new
model. See test/results/button_control_strength.txt.
Result (2026-11-12, base models AE_style_tester_v3 epoch 48 against AE_style_tester_v2 epoch 150): undecided, no measurable loss.
The real note gets 66.3 % against 66.9 % (-0.5 points, 95 % range -5.2 to +4.2); the 19 buttons give the same number of different
notes, but span 32 semitones instead of 47, and 2.6 % instead of 0.6 % of neighbouring buttons come in the wrong order.
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
from eval_style_chord_strength import free_logits, mode_buttons, style_inputs
from model_loader import load_model
from models import get_model_hparams
from train_chords import ChordPitchSamplerDataset

NEW = sys.argv[1] if len(sys.argv) > 1 else 'AE_style_jokerParam_tester_v3'
PRESENT = sys.argv[2] if len(sys.argv) > 2 else 'AE_style_jokerParam_tester_v2'
DEVICE = 'mps'
ROWS_PER_PASS = 60  # decoder rows per forward pass, to fit in the M1's memory
RESULTS = ROOT / 'test/results/button_control_strength.txt'


def load_style_model(name: str, device: torch.device):
    """A style joker preset of models.py with its checkpoint (ckpt_file_name)."""
    return load_model(name, cfg=get_model_hparams(name), compile_mode='none').to(device).eval()


def read_passages() -> tuple:
    """The data, the 60 moments (first note after each chord change), the usable pieces and the pieces in the file."""
    events = read_events(TEST_PICKLE)
    data = ChordPitchSamplerDataset(events, CONTEXT + NOTES, is_eval=True)
    pieces_in_pickle = len(data.piece_edges) - 1
    moments, usable_pieces = choose_moments(data, overlap_per_piece(events.reshape(-1, 5), pieces_in_pickle))
    return data, moments, usable_pieces, pieces_in_pickle


@torch.inference_mode()
def passage_inputs(model, data: ChordPitchSamplerDataset, moments: np.ndarray, device: torch.device) -> tuple:
    """Real notes [P, CONTEXT + NOTES], this model's style context [P, S, dim] and style mask [P, S] of each passage."""
    pitch = torch.stack([torch.as_tensor(data.pitches[onset - CONTEXT:onset + NOTES].astype(np.int64))
                         for onset in moments.tolist()]).to(device)
    styles = []
    for onset in moments.tolist():
        piece = data.note_piece[onset]
        start, end = data.piece_edges[piece], data.piece_edges[piece + 1]
        styles.append(style_inputs(model, data.pitches[start:end], onset - CONTEXT - start, onset + NOTES - start, device))
    return pitch, torch.cat([style for style, _ in styles]), torch.cat([mask for _, mask in styles])


@torch.inference_mode()
def last_logits(model, pitch: torch.Tensor, button: torch.Tensor, style: torch.Tensor, style_mask: torch.Tensor,
                passage: torch.Tensor, last: int) -> torch.Tensor:
    """Next-note logits [rows, last, 128] for the last `last` notes of pitch [rows, T], in passes of ROWS_PER_PASS rows.
    passage [rows] is the row of style [P, S, dim] and style_mask [P, S] that each pitch row uses."""
    return torch.cat([decode_logits(model, pitch[rows], button[rows], style[passage[rows]], style_mask[passage[rows]])[:, -last:].float()
                      for rows in torch.arange(len(pitch), device=pitch.device).split(ROWS_PER_PASS)])


@torch.inference_mode()
def decode_logits(model, pitch: torch.Tensor, button: torch.Tensor, style: torch.Tensor, style_mask: torch.Tensor) -> torch.Tensor:
    """free_logits for a style joker. A base style model (AE_style) has no joker, so its buttons go to the decoder
    as in its gen_pitch_token."""
    if hasattr(model, 'joker_button_idx'):
        return free_logits(model, pitch, button, style, style_mask)
    return model.decoder({'pitch': pitch[:, :-1], 'button': model.quantizer.discrete_to_real(button)[:, 1:]},
                         style_context=style, style_context_mask=style_mask)


@torch.inference_mode()
def model_values(model, data: ChordPitchSamplerDataset, moments: np.ndarray, device: torch.device) -> dict:
    """Per passage [P] values of one model."""
    pitch, style, style_mask = passage_inputs(model, data, moments, device)
    target = pitch[:, CONTEXT:]  # [P, NOTES], the pianist's real notes
    passage = torch.arange(len(pitch), device=device)
    modes = ('guided', 'joker') if hasattr(model, 'joker_button_idx') else ('guided',)  # a base model has no joker
    buttons = {mode: mode_buttons(model, pitch, CONTEXT, mode) for mode in modes}  # [P, CONTEXT + NOTES]
    values = {'joker': np.full(len(pitch), np.nan)}
    for mode, button in buttons.items():
        logits = last_logits(model, pitch, button, style, style_mask, passage, NOTES)
        values[mode] = 100 * F.softmax(logits, -1).gather(-1, target[..., None])[..., 0].mean(-1).cpu().numpy()

    # Sweep at the first of the 8 notes: one row per passage and button, the real notes and guided buttons before it.
    count = int(model.cfg['num_buttons'])  # buttons 0..count-1; in a style joker, count itself is the joker
    window = lambda values: values[:, :CONTEXT + 1].repeat_interleave(count, 0)  # [P * count, CONTEXT + 1]
    button = window(buttons['guided']).clone()
    button[:, -1] = torch.arange(count, device=device).repeat(len(pitch))
    logits = last_logits(model, window(pitch), button, style, style_mask, passage.repeat_interleave(count), 1)
    top = logits[:, 0].argmax(-1).view(len(pitch), count).cpu().numpy()  # [P, buttons], most likely note of each button
    values['different'] = np.array([len(np.unique(row)) for row in top])
    values['neighbours_same'] = 100 * (top[:, 1:] == top[:, :-1]).mean(-1)
    values['neighbours_lower'] = 100 * (top[:, 1:] < top[:, :-1]).mean(-1)
    values['span'] = (top[:, -1] - top[:, 0]).astype(float)
    return values


def git_line() -> str:
    git = lambda *args: subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    return (f'Date: {datetime.datetime.now():%Y-%m-%d %H:%M}  Commit: {git("rev-parse", "--short", "HEAD")}'
            f'  Uncommitted files: {len(git("status", "--short").splitlines())}')


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
    difference = new['guided'] - present['guided']
    low, high = bootstrap_ci(difference, rng)
    verdict = 'supported' if low > -5 else 'refuted' if high < -5 else 'undecided'

    rows = (('guided', 'Button known: probability of the real note, %'),
            ('joker', 'No button: probability of the real note, %'),
            ('different', 'Different most likely notes of the 19 buttons'),
            ('neighbours_same', 'Neighbouring buttons with the same most likely note, %'),
            ('neighbours_lower', 'Neighbouring buttons with a lower most likely note, %'),
            ('span', 'Most likely note of button 18 minus button 0, semitones'))
    report = ['#' * 64, git_line(),
              f'Present: {PRESENT} ({Path(get_model_hparams(PRESENT)["ckpt_file_name"]).name})',
              f'New: {NEW} ({Path(get_model_hparams(NEW)["ckpt_file_name"]).name})  Device: {DEVICE}',
              f'Data: {TEST_PICKLE.name}, {PIECES} of the {usable_pieces} usable pieces ({pieces_in_pickle} in the file), one chord change each. '
              f'Passages of {CONTEXT} + {NOTES} notes, seed {SEED}. The 19-button sweep is at the first of the {NOTES} notes.',
              f'Running time: {time.time() - started:.0f} s', '',
              'Means over passages. The first two rows are averaged over the 8 notes of each passage.',
              f'{"Measure":<56} {"Present":>8} {"New":>8} {"New minus present":>18}']
    for key, label in rows:
        if np.isnan(present[key]).any() or np.isnan(new[key]).any():
            continue  # no joker in a base style model
        report.append(f'{label:<56} {present[key].mean():>8.2f} {new[key].mean():>8.2f} {(new[key] - present[key]).mean():>+18.2f}')
    joker_difference = new['joker'] - present['joker']
    report += ['',
               f'Main, button known, new minus present: {difference.mean():+.2f} points, 95 % range {low:+.2f} to {high:+.2f}, '
               f'p = {permutation_p(difference, rng):.1e}, higher in {(difference > 0).sum()} of {len(difference)} passages, '
               f'lower in {(difference < 0).sum()}.']
    if np.isfinite(joker_difference).all():
        joker_low, joker_high = bootstrap_ci(joker_difference, rng)
        report.append(f'Reference, no button, new minus present: {joker_difference.mean():+.2f} points, 95 % range {joker_low:+.2f} to {joker_high:+.2f}.')
    if NEW == PRESENT:
        largest = max(np.abs(new[key] - present[key]).max() for key, _ in rows)
        report.append(f'Check run, the same preset in both roles: largest difference over all measures and passages = {largest:g} (must be 0).')
    report += [f'Result: {verdict} (supported if the range lies above -5 points).', '']
    report = '\n'.join(report)
    print(report)
    with open(RESULTS, 'a', encoding='utf-8') as file:
        file.write(report + '\n')


if __name__ == '__main__':
    main()
