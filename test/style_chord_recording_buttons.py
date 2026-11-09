"""
Question: In a recording made with the clear-passage style chord adapter, does the held chord make different buttons give the same note?
In the recording, button 9 right after button 8 gave the previous note again 4 times out of 4, all with a chord held.
Claim 1 (main): at the notes played with a chord held on a button other than the previous note's, the held chord raises the probability that the pressed button repeats the previous note, compared with no chord over the same history.
Supported if: the probability is higher with the chord on average and at more than half of these notes. Refuted otherwise.
Claim 2: at the notes played with a chord held, the 19 buttons give fewer different most likely notes with the chord than without it.
Supported if: fewer on average and at more than half of these notes. Refuted otherwise.
Models: save_models/AE_style_jokerParam_chords_clear_best.pth (clear, the adapter of the recording), and save_models/AE_style_jokerParam_chords_best_best.pth (previous, trained on all labelled chords) for comparison. Same frozen style joker.
Data: one recording of interaction_buttons_style_chords.py, out/interactive_buttons_style.mid with its chord log and buttons in out/interactive_buttons_style_chords.txt. 33 notes; Ebm (iv) held for notes 13-22 and F7 (V7) for notes 23-33. The app saves every recording to the same files, so the SHA-256 of both is written with the result.
Method: every window the app gave the model is rebuilt: the first 128 notes of the Chopin primer with the buttons the encoder gives the whole primer, then the recorded notes and buttons, the Chopin style, and the chord inputs of the app's ChordControls replaying the chord log. At each recorded note all 19 buttons are tried in turn, with no chord, with the recorded chord on the clear adapter and on the previous adapter. The history is the recorded one in every condition. No sampling.
Measures: probability that the pressed button gives the previous note again; number of different most likely notes among the 19 buttons. Higher repeat probability and fewer different notes mean more buttons repeat a note.
Check: the probability the replay gives to the recorded notes, which the app sampled from these distributions if the rebuild is right.
Limits: one short recording, two chord presses and 21 held notes, whose neighbouring notes are not independent. A case study, not a general result.
Size and time: 33 notes, 19 buttons, 3 conditions, 32 s on the M1.
Run: python test/style_chord_recording_buttons.py
Result (2026-11-08): both claims formally supported, but the effect is tiny and does not explain the heard repeats. The 4 recorded repeats are as likely with no chord over the same history (96.6-100 % against 94.5-100 % with the chord); the chord raises the repeat probability by 1.7 points on average (23.1 to 24.9 %), mostly at 2 notes where no repeat happened, and lowers the different notes from 12.8 to 12.4 of 19. The criteria (any increase, at over half the notes) were too weak for one recording. The repeats come from the frozen style joker's button map over this history. See test/results/style_chord_recording_buttons.txt.
"""

import datetime
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_chord_strength import require
from eval_style_chord_strength import ADAPTERS, load_style_checkpoint
from harmony.chord_joker import ChordControls
from interaction_1_button_chords import FIXED_CHORDS
from midiUtils import midi_to_dict

RECORDING = ROOT / 'out/interactive_buttons_style'  # .mid and _chords.txt, saved by interaction_buttons_style_chords.py
CLEAR = ROOT / 'save_models/AE_style_jokerParam_chords_clear_best.pth'
PREVIOUS = ROOT / 'save_models/AE_style_jokerParam_chords_best_best.pth'
PRIMER = ROOT / 'samples/Chopin_Nocturnes_Op9No1_In_B_Flat_Minor.mid'  # the app's primer and initial style (STYLE_IDX_INIT = 3)
CHORD_BANK = FIXED_CHORDS[2]  # the app's chords for this primer
CONTEXT = 128  # the app's CTX_LEN on the M1
DEVICE = 'mps'
ROWS_PER_PASS = 38  # 19 buttons at two notes per forward pass
RESULTS = ROOT / 'test/results/style_chord_recording_buttons.txt'
CONDITIONS = ('No chord pressed', 'Clear adapter, recorded chord', 'Previous adapter, recorded chord')


def read_recording() -> tuple:
    """Recorded pitches and buttons [notes] and the chord log as (notes played before it, chord name)."""
    pitch = midi_to_dict(str(RECORDING) + '.mid')[0]['pitch']
    lines = Path(str(RECORDING) + '_chords.txt').read_text().splitlines()
    require(lines[-1].startswith('buttons\t'), 'The chord log has no buttons line (recordings before 2026-11-04)')
    buttons = [int(value) for value in lines[-1].split('\t')[1].split()]
    require(len(buttons) == len(pitch), 'The buttons do not match the recorded notes')
    events = [(int(notes_before), name) for _, notes_before, name in (line.split('\t') for line in lines[1:-1])]
    return np.array(pitch), np.array(buttons), events


@torch.inference_mode()
def app_inputs(model, pitch: np.ndarray, buttons: np.ndarray, events: list, device: torch.device) -> tuple:
    """The inputs of interaction_buttons_style_chords.manageNote at every recorded note: pitch and button windows
    [notes, CONTEXT + 1], chord inputs [notes, CONTEXT + 1(, 12)] and the held chord's name per note, style context and mask."""
    primer = midi_to_dict(str(PRIMER))[0]
    primer_tokens = {key: torch.tensor(primer[key], dtype=torch.long, device=device).unsqueeze(0) for key in ('dtime', 'pitch', 'dur')}
    primer_buttons = model.real_to_discrete(model.encoder(primer_tokens))[0, :CONTEXT]  # the app encodes the whole primer
    # The last window's target is the last recorded note: the decoder never reads a target's pitch.
    windows = lambda values: values.unfold(0, CONTEXT + 1, 1)
    window_pitch = windows(torch.tensor(primer['pitch'][:CONTEXT] + pitch.tolist(), device=device))
    window_button = windows(torch.cat((primer_buttons, torch.as_tensor(buttons, device=device))))
    controls, chord, chord_names = ChordControls(CONTEXT, CONTEXT, CHORD_BANK), [], []
    key_of = {value.name: key for key, value in CHORD_BANK.items()}
    for note in range(len(pitch)):
        for notes_before, name in events:
            if notes_before == note and name == 'free':
                controls.release()
            elif notes_before == note:
                controls.handle_key(key_of[name][-1], True, shift=key_of[name].startswith('shift+'))
        chord.append(controls.next_tokens(device))
        chord_names.append(controls.chord_name.split(' ')[0] or 'none')
        controls.commit_note()
    chord = {key: torch.cat([tokens[key] for tokens in chord]) for key in chord[0]}
    style_length = model.style_encoder.max_seq_len
    require(len(primer['pitch']) >= style_length, 'Primer shorter than the style window')
    style_pitch = torch.tensor(primer['pitch'][:style_length], device=device).unsqueeze(0)
    style_mask = torch.ones_like(style_pitch, dtype=torch.bool)
    return window_pitch, window_button, chord, chord_names, model.encode_style(style_pitch, style_mask), style_mask


@torch.inference_mode()
def button_sweep(model, window_pitch, window_button, chord, style, style_mask) -> torch.Tensor:
    """Next-note logits [notes, buttons, 128]: at every recorded note each button in turn; chord=None for no chord."""
    count = model.joker_button_idx  # buttons 0..count-1; count itself is the joker
    notes = len(window_pitch)
    rows = lambda values: values.unsqueeze(1).expand(-1, count, *values.shape[1:]).flatten(0, 1)
    inputs = {'pitch': rows(window_pitch), 'button': rows(window_button).clone()}
    inputs['button'][:, -1] = torch.arange(count, device=window_pitch.device).repeat(notes)
    if chord is not None:
        inputs.update({key: rows(value) for key, value in chord.items()})
    # All rows do not fit in the M1's memory in one pass.
    logits = torch.cat([model.next_pitch_logits({key: value[part] for key, value in inputs.items()},
                                                style.expand(len(part), -1, -1), style_mask.expand(len(part), -1))
                        for part in torch.arange(notes * count, device=window_pitch.device).split(ROWS_PER_PASS)])
    return logits.float().view(notes, count, -1)


def main() -> None:
    started = time.time()
    device = torch.device(DEVICE)
    pitch, buttons, events = read_recording()
    clear, previous = load_style_checkpoint(CLEAR, device), load_style_checkpoint(PREVIOUS, device)
    frozen = [{key: value for key, value in model.state_dict().items() if not key.startswith(ADAPTERS)} for model in (clear, previous)]
    require(frozen[0].keys() == frozen[1].keys() and all(torch.equal(value, frozen[1][key]) for key, value in frozen[0].items()),
            'The checkpoints do not share the same frozen style joker')
    window_pitch, window_button, chord, chord_names, style, style_mask = app_inputs(clear, pitch, buttons, events, device)
    sweeps = [button_sweep(clear, window_pitch, window_button, None, style, style_mask),
              button_sweep(clear, window_pitch, window_button, chord, style, style_mask),
              button_sweep(previous, window_pitch, window_button, chord, style, style_mask)]
    probs = [F.softmax(logits, dim=-1).cpu().numpy() for logits in sweeps]  # [notes, buttons, 128] per condition
    notes = np.arange(len(pitch))
    held = chord['chord_active'][:, -1].cpu().numpy()  # [notes], a chord was held for this note
    other_button = held & (np.r_[-1, buttons[:-1]] != buttons)
    other_button[0] = False  # the first note has no previous note
    previous_pitch = np.r_[0, pitch[:-1]]
    repeat = [p[notes, buttons, previous_pitch] for p in probs]  # [notes] the pressed button gives the previous note
    different = [np.array([len(np.unique(row)) for row in p.argmax(-1)]) for p in probs]  # [notes]
    recorded = [p[notes, buttons, pitch] for p in probs]  # [notes] probability of the recorded note

    git = lambda *args: subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    sha = lambda suffix: hashlib.sha256(Path(str(RECORDING) + suffix).read_bytes()).hexdigest()[:16]
    lines = ['#' * 64,
             f'Date: {datetime.datetime.now():%Y-%m-%d %H:%M}  Commit: {git("rev-parse", "--short", "HEAD")}'
             f'  Uncommitted files: {len(git("status", "--short").splitlines())}',
             f'Clear: {CLEAR.name}  Previous: {PREVIOUS.name}  Device: {DEVICE}',
             f'Recording: {RECORDING.name}.mid (SHA-256 {sha(".mid")}...) and _chords.txt ({sha("_chords.txt")}...), {len(pitch)} notes, '
             f'{held.sum()} with a chord held, {other_button.sum()} of them on a button other than the previous note\'s. '
             f'Chord log: {events}. Primer {PRIMER.name}, context {CONTEXT}.',
             f'Running time: {time.time() - started:.0f} s', '',
             'Check: probability of the recorded note on its button, mean over notes, and share of notes that were the most likely one:']
    for condition, value, p in zip(CONDITIONS, recorded, probs):
        top = p[notes, buttons].argmax(-1) == pitch
        lines.append(f'  {condition:<34} free notes {100 * value[~held].mean():5.1f} %, held notes {100 * value[held].mean():5.1f} %, '
                     f'most likely {top[held].mean():.0%} of held notes')
    lines += ['', 'Per held note: probability that the pressed button repeats the previous note, and different most likely notes',
              'among the 19 buttons, per condition. * = the recorded note repeated the previous one on another button.',
              f'{"Note":>4} {"Chord":<9} {"Button":>6} {"Played":>6}'
              + ''.join(f' {name + " repeat %":>18}' for name in ('No chord', 'Clear', 'Previous'))
              + ''.join(f' {name + " different":>18}' for name in ('No chord', 'Clear', 'Previous'))]
    for note in notes[held]:
        mark = '*' if other_button[note] and pitch[note] == previous_pitch[note] else ' '
        lines.append(f'{note + 1:>4} {chord_names[note]:<9} {buttons[note]:>6} {pitch[note]:>5}{mark}'
                     + ''.join(f' {100 * value[note]:>18.1f}' for value in repeat)
                     + ''.join(f' {value[note]:>18d}' for value in different))
    lines += ['', f'No chord held in the recording (notes 1-{np.flatnonzero(held)[0]}): {different[0][~held].mean():.2f} different notes of 19.']
    verdicts = []
    # direction +1: the claim expects the chord to raise the value; -1: to lower it.
    claims = (('Claim 1, probability that the pressed button repeats the previous note, %', [100 * value for value in repeat], other_button, 1),
              ('Claim 2, different most likely notes of 19', different, held, -1))
    for label, values, mask, direction in claims:
        no_chord, with_clear, with_previous = (value[mask] for value in values)
        moved = direction * (with_clear - no_chord) > 0
        verdicts.append('supported' if direction * (with_clear.mean() - no_chord.mean()) > 0 and moved.mean() > 0.5 else 'refuted')
        lines.append(f'{label}, over {mask.sum()} notes: no chord {no_chord.mean():.2f}, clear {with_clear.mean():.2f}, '
                     f'previous {with_previous.mean():.2f}. Clear moved as claimed at {moved.sum()} notes, '
                     f'the other way at {(direction * (with_clear - no_chord) < 0).sum()}.')
    lines += [f'Result: claim 1 {verdicts[0]}, claim 2 {verdicts[1]}.', '']
    report = '\n'.join(lines)
    print(report)
    with open(RESULTS, 'a', encoding='utf-8') as file:
        file.write(report + '\n')


if __name__ == '__main__':
    main()
