"""
Question: Does the app's Chopin primer explain the lock heard in the recording of test/style_chord_recording_buttons.py,
where button 9 right after button 8 gave the previous note again 4 times out of 4?
Over 60 validation pieces (test/style_button_pattern_repeats.py) the same pattern repeated at only 7.1 % of these steps with no
chord, and no piece did so at 3 or 4 of its 4 steps.
Claim 1 (main): with the app's Chopin primer and no chord, playing the pattern makes button 9 after button 8 repeat the previous
note at more than half of these steps, as in the recording.
Supported if: the 95 % range over sampling seeds lies above 50 %. Refuted if it lies below 50 %; undecided if it contains 50 %.
Claim 2: the app's way of encoding the primer causes it. The app encodes the whole 834-note Chopin file and keeps the buttons
of its first 128 notes; the encoder is bidirectional, so these buttons differ from those of the 128 notes encoded alone.
Supported if: the 95 % range of app encoding minus primer encoded alone lies above zero. Refuted if it reaches zero or lies below it.
Model: save_models/AE_style_jokerParam_chords_clear_best.pth, the recording's model; without a chord it is the frozen style joker.
Data: one primer, the app's: the first 128 notes of samples/Chopin_Nocturnes_Op9No1_In_B_Flat_Minor.mid, with the first 512
notes of the file as style (STYLE_IDX_INIT = 3). The result is about this primer only.
Method: as in the app, the model generates 24 notes over a sliding 128-note window, sampled at temperature 1, while the pattern
6 8 9 10 9 8 is played 4 times from the first generated note (in the recording it began after 12 other notes). Three players,
each sampled with the same 40 seeds: the app's primer buttons with no chord; the primer encoded alone with no chord; and
(exploratory) the app's primer buttons with the recording's chords, Ebm (iv) for the first 12 notes and F7 (V7) for the last 12,
pressed through the app's ChordControls. Paired by seed: the players share the sampling noise.
Measure: share of the 4 steps 8 -> 9 of each generation that repeat the previous note's pitch, one value per seed.
Exploratory: repeats on any changed button, repeats by position in the pattern, and the share of generated notes in the held chord.
Size and time: 1 primer, 40 seeds, 24 notes, 3 players; 52 s on the M1.
Run: python test/style_button_pattern_chopin.py
Result (2026-11-09): both claims refuted. With the app's Chopin primer and no chord, the pattern repeats at 8.1 % of the 8 -> 9 steps (range 3.8 to 12.5 %), like the 60 pieces' 7.1 %, and none of the 40 seeds repeated at 3 or 4 of its 4 steps; the recorded chords change little (9.4 %). Encoding the primer alone doubles the repeats (16.2 %) instead of lowering them, so the app's encoding is not the cause either. The recording's lock must come from the notes played before the pattern or from the notes it happened to sample. See test/results/style_button_pattern_chopin.txt.
"""

import datetime
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_chord_strength import bootstrap_ci, permutation_p, require
from eval_style_chord_strength import load_style_checkpoint
from midiUtils import midi_to_dict
# The app's inputs, as rebuilt and checked against the recording there.
from style_chord_recording_buttons import CONTEXT, PRIMER, app_inputs

MODEL = ROOT / 'save_models/AE_style_jokerParam_chords_clear_best.pth'
PATTERN = (6, 8, 9, 10, 9, 8)  # buttons of the recording's chord holds
GENERATED = 24  # the pattern 4 times
CHORD_EVENTS = [(0, 'Ebm (iv)'), (12, 'free'), (12, 'F7 (V7)')]  # (generated notes before it, chord), the recording's two chords
SEEDS = 40
SEED = 1729
DEVICE = 'mps'
ROWS_PER_PASS = 45  # rows per forward pass that fit in the M1's memory
PIECES_EIGHT_NINE = 7.1  # % of 8 -> 9 steps that repeated over the 60 validation pieces, no chord (style_button_pattern_repeats.txt, 2026-11-08)
RESULTS = ROOT / 'test/results/style_button_pattern_chopin.txt'
PLAYERS = ('App primer, no chord', 'Primer encoded alone, no chord', 'App primer, recorded chords')


@torch.inference_mode()
def primer_buttons(model, device: torch.device) -> tuple:
    """Primer pitches [CONTEXT] and buttons [CONTEXT]: as the app (whole file encoded) and with the primer encoded alone."""
    primer = midi_to_dict(str(PRIMER))[0]
    tokens = {key: torch.tensor(primer[key], dtype=torch.long, device=device).unsqueeze(0) for key in ('dtime', 'pitch', 'dur')}
    whole = model.real_to_discrete(model.encoder(tokens))[0, :CONTEXT]
    alone = model.real_to_discrete(model.encoder({key: value[:, :CONTEXT] for key, value in tokens.items()}))[0]
    return tokens['pitch'][0, :CONTEXT], whole, alone


@torch.inference_mode()
def generate(model, pitch, button, chord_rows: dict, style, style_mask, noise) -> torch.Tensor:
    """Fill pitch[:, CONTEXT:] note by note. pitch, button [rows, CONTEXT + GENERATED]; chord_rows: the chord window of each
    generated note [GENERATED, rows, CONTEXT + 1(, 12)]; noise [rows, GENERATED, 128] (Gumbel-max sampling, shared per seed)."""
    for step in range(GENERATED):
        window = slice(step, step + CONTEXT + 1)  # the app's sliding window: CONTEXT notes + the target
        inputs = {'pitch': pitch[:, window], 'button': button[:, window], **{key: value[step] for key, value in chord_rows.items()}}
        logits = torch.cat([model.next_pitch_logits({key: value[part] for key, value in inputs.items()},
                                                    style.expand(len(part), -1, -1), style_mask.expand(len(part), -1))
                            for part in torch.arange(len(pitch), device=pitch.device).split(ROWS_PER_PASS)])
        pitch[:, CONTEXT + step] = (logits.float() - noise[:, step].log()).argmax(-1)
    return pitch


def main() -> None:
    started = time.time()
    device = torch.device(DEVICE)
    model = load_style_checkpoint(MODEL, device)
    primer_pitch, whole, alone = primer_buttons(model, device)
    pattern = np.tile(PATTERN, GENERATED // len(PATTERN))
    # The pitches passed here are placeholders: only the chord windows, the buttons and the style are used.
    _, window_button, chord, _, style, style_mask = app_inputs(model, np.zeros(GENERATED, dtype=np.int64), pattern, CHORD_EVENTS, device)
    require(torch.equal(window_button[0, :CONTEXT], whole), 'The primer buttons differ from the app replay')

    # Rows [seeds * players], seed-major: row = seed * players + player.
    players = len(PLAYERS)
    pattern_tensor = torch.as_tensor(pattern, device=device)
    button = torch.stack((torch.cat((whole, pattern_tensor)), torch.cat((alone, pattern_tensor)), torch.cat((whole, pattern_tensor))))
    button = button.repeat(SEEDS, 1)
    pitch = torch.cat((primer_pitch, torch.zeros(GENERATED, dtype=torch.long, device=device))).repeat(SEEDS * players, 1)
    held = (torch.arange(SEEDS * players, device=device) % players) == 2  # only the third player presses the chords
    chord_rows = {key: value.unsqueeze(1) * held.view(1, -1, *[1] * (value.dim() - 1)) for key, value in chord.items()}
    noise = torch.empty(SEEDS, GENERATED, 128).exponential_(generator=torch.Generator().manual_seed(SEED))
    pitch = generate(model, pitch, button, chord_rows, style, style_mask, noise.repeat_interleave(players, 0).to(device))
    print(f'generated, {time.time() - started:.0f} s', flush=True)

    # [seeds, players, 1 + GENERATED]; note t of the generation is compared with note t - 1 (the primer's last note for t = 0).
    pitch = pitch.view(SEEDS, players, -1)[..., CONTEXT - 1:].cpu().numpy()
    button = button.view(SEEDS, players, -1)[..., CONTEXT - 1:].cpu().numpy()
    repeat = pitch[..., 1:] == pitch[..., :-1]  # [seeds, players, GENERATED]
    changed = button[..., 1:] != button[..., :-1]
    eight_nine = (button[..., :-1] == 8) & (button[..., 1:] == 9)
    share_eight_nine = 100 * (repeat & eight_nine).sum(-1) / eight_nine.sum(-1)  # [seeds, players], %
    share_changed = 100 * (repeat & changed).sum(-1) / changed.sum(-1)
    member = chord['chord_chroma'][:, -1].cpu().numpy()  # [GENERATED, 12], the held chord at each generated note
    in_chord = 100 * member[np.arange(GENERATED), pitch[..., 1:] % 12]  # [seeds, players, GENERATED]
    by_position = repeat.reshape(SEEDS, players, -1, len(PATTERN)).sum((0, 2))  # [players, pattern positions], counts

    rng = np.random.default_rng(SEED)
    low, high = bootstrap_ci(share_eight_nine[:, 0], rng)
    verdict1 = 'supported' if low > 50 else 'refuted' if high < 50 else 'undecided'
    difference = share_eight_nine[:, 0] - share_eight_nine[:, 1]
    low2, high2 = bootstrap_ci(difference, rng)
    verdict2 = 'supported' if low2 > 0 else 'refuted'
    differing = (whole != alone).sum().item()

    git = lambda *args: subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    steps = ' '.join(f'{a}->{b}' for a, b in zip((pattern[-1],) + PATTERN[:-1], PATTERN))
    lines = ['#' * 64,
             f'Date: {datetime.datetime.now():%Y-%m-%d %H:%M}  Commit: {git("rev-parse", "--short", "HEAD")}'
             f'  Uncommitted files: {len(git("status", "--short").splitlines())}',
             f'Model: {MODEL.name}  Device: {DEVICE}',
             f'Primer: first {CONTEXT} notes of {PRIMER.name}, style its first {style.shape[1]} notes. {GENERATED} generated notes, '
             f'pattern {PATTERN}, temperature 1, {SEEDS} seeds from {SEED}. Chords of the third player: {CHORD_EVENTS}.',
             f'Primer buttons that differ between the app encoding and the primer encoded alone: {differing} of {CONTEXT}.',
             f'Running time: {time.time() - started:.0f} s', '',
             f'{"Player":<32} {"Repeats at 8 -> 9 %":>20} {"Repeats on a changed button %":>30} {"Notes in the held chord %":>26}']
    for index, name in enumerate(PLAYERS):
        lines.append(f'{name:<32} {share_eight_nine[:, index].mean():>20.1f} {share_changed[:, index].mean():>30.1f}'
                     f' {in_chord[:, index].mean():>26.1f}')
    lines += ['Means over seeds. Notes in the held chord: share of generated notes in Ebm for the first 12 and F7 for the last 12, held or not.',
              f'Reference, 60 validation pieces with no chord: {PIECES_EIGHT_NINE} % repeats at 8 -> 9.', '',
              f'Seeds by repeats at the 4 steps 8 -> 9 (0, 1, 2, 3, 4 of 4):']
    lines += [f'  {name:<32} ' + ' '.join(f'{count:3d}' for count in np.bincount((share_eight_nine[:, index] / 25).round().astype(int), minlength=5))
              for index, name in enumerate(PLAYERS)]
    lines += [f'Repeats by step of the pattern, counts over {SEEDS * GENERATED // len(PATTERN)} steps each ({steps}):']
    lines += [f'  {name:<32} ' + ' '.join(f'{count:5d}' for count in by_position[index]) for index, name in enumerate(PLAYERS)]
    lines += ['',
              f'Claim 1, app primer with no chord: {share_eight_nine[:, 0].mean():.1f} % of 8 -> 9 steps repeat, 95 % range {low:.1f} to {high:.1f}.',
              f'Claim 2, app encoding minus primer encoded alone: {difference.mean():+.1f} points, 95 % range {low2:+.1f} to {high2:+.1f}, '
              f'p = {permutation_p(difference, rng):.1e}, higher in {(difference > 0).sum()} of {SEEDS} seeds, lower in {(difference < 0).sum()}.',
              f'Result: claim 1 {verdict1}, claim 2 {verdict2}.', '']
    report = '\n'.join(lines)
    print(report)
    with open(RESULTS, 'a', encoding='utf-8') as file:
        file.write(report + '\n')


if __name__ == '__main__':
    main()
