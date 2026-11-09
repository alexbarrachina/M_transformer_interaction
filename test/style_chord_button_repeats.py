"""
Question: When a chord is held, does the clear-passage style chord adapter make different buttons give the same note?
Claim 1 (main): with the piece's real chord held, the 19 buttons give fewer different notes than with no chord pressed.
Supported if: the 95 % range of chord held minus no chord lies below zero. Refuted if it reaches zero or lies above it.
Claim 2: with the same chord held, the clear adapter gives fewer different notes than the previous adapter.
Supported if: the 95 % range of clear minus previous lies below zero. Refuted if it reaches zero or lies above it.
Models: save_models/AE_style_jokerParam_chords_clear_best.pth (clear, trained only on chords that at least 60 % of their notes follow) and save_models/AE_style_jokerParam_chords_best_best.pth (previous, trained on all labelled chords of the same data). Same frozen style joker.
Data: Training-Data/giantmidi_full_w_chords_unseen_test, the adapters' validation split: it chose their best checkpoints but never trained them, and the style joker never saw it. The same 60 pieces and chord changes as test/chord_strength_many_chords.py, which leaves out pieces with over 10 % of their 8-note sequences in training data.
Method: at each chord change, the history is the pianist's 200 notes before it with the buttons the frozen encoder gives them, as in training, and the style is the piece's notes next to the passage. The real chord is pressed at the change. At notes 1, 4 and 8 of the hold (1 = the note where the chord is pressed), each of the 19 buttons is tried in turn, with the pianist's real notes and buttons before it. Pressing the chord or not is the only difference, and both adapters see the same inputs. No sampling.
Measure: how many different notes the 19 buttons give as their most likely note, averaged over the 3 hold notes; one value per piece. Lower means more buttons repeat a note that another button gives.
Exploratory: share of buttons whose most likely note is a chord note or has the previous note's pitch class, the chance that two neighbouring buttons play the same note when sampled at temperature 1 (as the app does), the probability given to the chord's notes, and the measure at each of the 3 hold notes.
Size and time: 60 pieces, 3 hold notes, 19 buttons, 3 conditions, 274 s on the M1 (all 8 hold notes would take about 11 minutes).
Run: python test/style_chord_button_repeats.py
Result (2026-11-08): claim 1 refuted, claim 2 supported but tiny. Over the pianist's real notes, holding the real chord barely changes how many different notes the buttons give: 14.6 of 19 with the clear adapter against 14.8 with no chord (range -0.50 to +0.06) and 14.8 with the previous adapter. The chord moves the buttons onto chord notes (72.5 % against 50.5 % of buttons) without making them repeat. See test/results/style_chord_button_repeats.txt.
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
from chord_strength_many_chords import (CONTEXT, NOTES, PIECES, SEED, TEST_PICKLE, choose_moments, chords_at,
                                        overlap_per_piece, read_events)
from eval_chord_strength import bootstrap_ci, permutation_p, require
from eval_style_chord_strength import ADAPTERS, load_style_checkpoint, player_buttons, style_inputs
from train_chords import ChordPitchSamplerDataset

CLEAR = ROOT / 'save_models/AE_style_jokerParam_chords_clear_best.pth'
PREVIOUS = ROOT / 'save_models/AE_style_jokerParam_chords_best_best.pth'
DEVICE = 'mps'
HOLD_NOTES = (1, 4, 8)  # notes of the hold where every button is tried; 1 = the note where the chord is pressed
ROWS_PER_PASS = 38  # 19 buttons at two hold notes per forward pass
RESULTS = ROOT / 'test/results/style_chord_button_repeats.txt'
CONDITIONS = ('No chord pressed', 'Previous adapter, real chord held', 'Clear adapter, real chord held')


@torch.inference_mode()
def button_sweep(model, data: ChordPitchSamplerDataset, onset: int, chord: dict, hold: bool, device: torch.device) -> torch.Tensor:
    """Next-note logits [hold notes, buttons, 128]: at each of the HOLD_NOTES from onset, each button in turn,
    with the pianist's real notes and buttons before it. hold presses the chord from onset on."""
    piece = data.note_piece[onset]
    start, end = data.piece_edges[piece], data.piece_edges[piece + 1]
    passage = torch.as_tensor(data.pitches[onset - CONTEXT:onset + NOTES].astype(np.int64), device=device)
    style, style_mask = style_inputs(model, data.pitches[start:end], onset - CONTEXT - start, onset + NOTES - start, device)
    buttons = player_buttons(model, passage.unsqueeze(0))[0]  # as training: the encoder over the whole passage
    count = model.joker_button_idx  # buttons 0..count-1; count itself is the joker
    # One window of CONTEXT notes + the target per hold note: [hold notes, CONTEXT + 1], then one row per button.
    hold_index = torch.tensor(HOLD_NOTES, device=device) - 1
    windows = lambda values: values.unfold(0, CONTEXT + 1, 1)[hold_index].unsqueeze(1).expand(-1, count, -1).flatten(0, 1)
    pitch, button = windows(passage), windows(buttons).clone()
    button[:, -1] = torch.arange(count, device=device).repeat(len(HOLD_NOTES))
    inputs = {'pitch': pitch, 'button': button}
    if hold:
        active = windows(torch.arange(CONTEXT + NOTES, device=device) >= CONTEXT)
        chroma = torch.zeros(12, dtype=torch.bool, device=device)
        chroma[chord['pcs']] = True
        inputs.update(chord_active=active, chord_chroma=active.unsqueeze(-1) & chroma,
                      chord_anchor_pc=torch.full_like(pitch, chord['anchor']))
    # The hold notes * buttons rows do not fit in the M1's memory in one pass.
    logits = torch.cat([model.next_pitch_logits({key: value[rows] for key, value in inputs.items()},
                                                style.expand(len(rows), -1, -1), style_mask.expand(len(rows), -1))
                        for rows in torch.arange(len(pitch), device=device).split(ROWS_PER_PASS)])
    return logits.float().view(len(HOLD_NOTES), count, -1)


def measure(logits: torch.Tensor, member: np.ndarray, previous_pc: np.ndarray) -> dict:
    """Per hold note [hold notes] values of one sweep. member: [12] chord pitch classes; previous_pc: [hold notes]."""
    top = logits.argmax(-1).cpu().numpy()  # [hold notes, buttons], most likely note of each button
    probs = F.softmax(logits, dim=-1)
    pc_mass = (probs @ F.one_hot(torch.arange(128, device=probs.device) % 12, 12).float()).cpu().numpy()
    return {'different': np.array([len(np.unique(row)) for row in top]),
            'chord_note': member[top % 12].mean(-1),
            'previous_pc': (top % 12 == previous_pc[:, None]).mean(-1),
            'neighbours_same': (probs[:, 1:] * probs[:, :-1]).sum(-1).mean(-1).cpu().numpy(),
            'chord_probability': pc_mass[..., member].sum(-1).mean(-1)}


def main() -> None:
    started = time.time()
    device = torch.device(DEVICE)
    events = read_events(TEST_PICKLE)
    data = ChordPitchSamplerDataset(events, CONTEXT + NOTES, is_eval=True)
    pieces_in_pickle = len(data.piece_edges) - 1
    moments, usable_pieces = choose_moments(data, overlap_per_piece(events.reshape(-1, 5), pieces_in_pickle))
    previous, clear = load_style_checkpoint(PREVIOUS, device), load_style_checkpoint(CLEAR, device)
    frozen = [{key: value for key, value in model.state_dict().items() if not key.startswith(ADAPTERS)} for model in (previous, clear)]
    require(frozen[0].keys() == frozen[1].keys() and all(torch.equal(value, frozen[1][key]) for key, value in frozen[0].items()),
            'The checkpoints do not share the same frozen style joker')
    values = {condition: {} for condition in CONDITIONS}  # condition -> measure -> [pieces, hold notes]
    for index, onset in enumerate(moments.tolist()):
        chord = chords_at(data, onset)['real']
        member = np.zeros(12, dtype=bool)
        member[chord['pcs']] = True
        previous_pc = data.pitches[onset + np.array(HOLD_NOTES) - 2].astype(np.int64) % 12  # the note before each tried one
        sweeps = (button_sweep(clear, data, onset, chord, False, device),  # no chord: the shared frozen style joker
                  button_sweep(previous, data, onset, chord, True, device),
                  button_sweep(clear, data, onset, chord, True, device))
        for condition, logits in zip(CONDITIONS, sweeps):
            for key, value in measure(logits, member, previous_pc).items():
                values[condition].setdefault(key, []).append(value)
        print(f'moment {index + 1}/{PIECES}, {time.time() - started:.0f} s', flush=True)
    values = {condition: {key: np.stack(value) for key, value in measures.items()} for condition, measures in values.items()}
    free, held_previous, held_clear = (values[condition] for condition in CONDITIONS)

    rng = np.random.default_rng(SEED)
    per_piece = lambda store, key: store[key].mean(-1)  # mean over the hold notes: [pieces]
    claims = [('Claim 1, clear adapter: chord held minus no chord', per_piece(held_clear, 'different') - per_piece(free, 'different')),
              ('Claim 2, chord held: clear minus previous adapter', per_piece(held_clear, 'different') - per_piece(held_previous, 'different')),
              ('Previous adapter: chord held minus no chord', per_piece(held_previous, 'different') - per_piece(free, 'different'))]
    lines = []
    verdicts = []
    for label, difference in claims:
        low, high = bootstrap_ci(difference, rng)
        verdicts.append('supported' if high < 0 else 'refuted')
        lines.append(f'{label}: {difference.mean():+.2f} different notes, 95 % range {low:+.2f} to {high:+.2f}, '
                     f'p = {permutation_p(difference, rng):.1e}, fewer in {(difference < 0).sum()} of {len(difference)} pieces, '
                     f'more in {(difference > 0).sum()}.')

    git = lambda *args: subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    head = (f'{"Condition":<36} {"Different notes of 19":>21} {"Chord note %":>12} {"Previous pc %":>13}'
            f' {"Neighbours same %":>17} {"Chord probability %":>19}')
    report = ['#' * 64,
              f'Date: {datetime.datetime.now():%Y-%m-%d %H:%M}  Commit: {git("rev-parse", "--short", "HEAD")}'
              f'  Uncommitted files: {len(git("status", "--short").splitlines())}',
              f'Previous: {PREVIOUS.name}  Clear: {CLEAR.name}  Device: {DEVICE}',
              f'Data: {TEST_PICKLE.name}, {PIECES} of the {usable_pieces} usable pieces ({pieces_in_pickle} in the file), one chord change each. '
              f'Context {CONTEXT} real notes, notes {HOLD_NOTES} of the hold, all 19 buttons, seed {SEED}.',
              f'Running time: {time.time() - started:.0f} s', '',
              'Columns: different most likely notes among the 19 buttons; buttons whose most likely note is a chord note, and has the',
              "previous note's pitch class; chance two neighbouring buttons play the same note when sampled; probability on the chord's notes.",
              'Means over pieces and hold notes.', head]
    for condition, store in values.items():
        report.append(f'{condition:<36} {store["different"].mean():>21.2f} {100 * store["chord_note"].mean():>12.1f}'
                      f' {100 * store["previous_pc"].mean():>13.1f} {100 * store["neighbours_same"].mean():>17.1f}'
                      f' {100 * store["chord_probability"].mean():>19.1f}')
    report += ['', f'Different notes of 19 at hold notes {HOLD_NOTES}, 1 = the note where the chord is pressed (exploratory):']
    report += [f'  {condition:<36} ' + ' '.join(f'{value:5.2f}' for value in store['different'].mean(0)) for condition, store in values.items()]
    report += ['', *lines, f'Result: claim 1 {verdicts[0]}, claim 2 {verdicts[1]}.', '']
    report = '\n'.join(report)
    print(report)
    with open(RESULTS, 'a', encoding='utf-8') as file:
        file.write(report + '\n')


if __name__ == '__main__':
    main()
