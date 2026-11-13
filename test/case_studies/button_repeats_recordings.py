"""
Question: Do the repeated notes on neighbouring buttons heard while playing come from the training of the button + style models?
In six recordings, notes played on a button one step from the previous note's often repeated the previous note.
In training, the frozen encoder gives the buttons. Its deviate_loss (loss_funcs.py) should keep the button when a note repeats,
but it checks one note too early (pitches[:, 1:-1] == pitches[:, :-2] instead of pitches[:, 2:] == pitches[:, 1:-1], since e
encodes pitch[:, 1:]), and no other encoder loss constrains a repeated note.
Claim 1 (main): in the buttons both models were trained with, most repeated notes get a different button from the previous note's.
Supported if: for both models, the 95 % range of the share of repeated notes that keep their button lies below 50 %. Refuted otherwise.
Claim 2: in the recordings, the model gives the previous note more probability on a neighbouring button than over real music
with its training buttons.
Supported if: in all 6 recordings, the mean is above the upper end of the 95 % range over real music for that model. Refuted otherwise.
Models: the style joker AE_style_jokerParam_tester_v2, frozen inside save_models/AE_style_jokerParam_chords_clear_best.pth and used
without a chord; and AE_style_tester (save_models/AE_style_tester_30_eps_434_steps_0.5793_loss_0.8586_acc.pth), an older model
trained separately with the same encoder losses.
Recordings: out/buttons_style_chords_clear/1-5_interactive_buttons_style.mid (interaction_buttons_style_chords.py with the clear
model, no chord pressed, Chopin primer and style) and out/interactive_buttons_style.mid (interaction_buttons_style.py with
AE_style_tester, Chopin primer, Satie style), each with its _chords.txt button log. out/ is not in git and the app overwrites
the last one, so the SHA-256 of every file is written with the result.
Real music: the same 60 passages of Training-Data/giantmidi_full_w_chords_unseen_test as test/button_repeat_neighbour_labels.py
(200 + 8 notes, buttons from each model's encoder over the passage, style from the piece's notes next to it). The pieces may
be in both models' training data, which is what matters here: claim 1 is about the buttons they were trained with.
Method: every window the app gave the model is rebuilt: the first 128 primer notes with the buttons the encoder gives the whole
primer file, then the recorded notes and buttons, and the app's style. One forward pass per recorded note, no sampling.
Over real music, one forward pass per passage reads the decoder's prediction at every note.
Measures: claim 1, share of the repeated notes of a passage whose button equals the previous note's, one value per passage.
Claim 2, the probability the decoder gives the previous note at notes played on a button one step from the previous note's,
one mean per recording or per passage.
Check: the probability the replay gives to the recorded notes, which the app sampled from these distributions if the rebuild is right.
Size and time: 6 recordings (910 notes), 60 passages, 2 models; 37 s on the M1.
Run: python test/case_studies/button_repeats_recordings.py
Result (2026-11-09): claim 1 supported, claim 2 refuted by one recording. In their training buttons, only 16.5 % (style joker) and 22.3 % (AE_style_tester) of repeated notes keep their button, so the decoders learned that a neighbouring button can repeat the note: over real music they give the previous note about 5 % at one-step moves. In 5 of the 6 recordings this rose to 12-27 %; in recording 5 it stayed at 5.0 %, inside the real-music range (3.6-6.0 %). See test/results/button_repeats_recordings__AE_style_jokerParam_chords_clear_best_and_AE_style_tester.txt.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'test'))  # the behaviour tests in test/ whose helpers this script reuses

# The same passages as test/button_repeat_neighbour_labels.py: 200-note context, 8 notes, 60 pieces, seed.
from chord_strength_many_chords import CONTEXT as PASSAGE_CONTEXT, NOTES, PIECES, SEED, TEST_PICKLE, choose_moments, overlap_per_piece, read_events
from chord_strength_guidance import bootstrap_ci, require
from chord_strength_style import load_style_checkpoint, player_buttons, style_inputs
from midiUtils import midi_to_dict
from model_loader import load_model
from train_chords import ChordPitchSamplerDataset

CLEAR = ROOT / 'save_models/AE_style_jokerParam_chords_clear_best.pth'  # without a chord: the frozen style joker
TESTER = 'AE_style_tester'  # preset in models.py
PRIMER = ROOT / 'samples/Chopin_Nocturnes_Op9No1_In_B_Flat_Minor.mid'  # both apps' primer
RECORDINGS = [(ROOT / f'out/buttons_style_chords_clear/{index}_interactive_buttons_style', 'clear', ROOT / 'samples/Chopin_Nocturnes_Op9No1_In_B_Flat_Minor.mid')
              for index in range(1, 6)]  # (recording, model, style file): the chords app's STYLE_IDX_INIT = 3
RECORDINGS.append((ROOT / 'out/interactive_buttons_style', 'tester', ROOT / 'samples/Satie_Gymnopedie_No1_no_melody.midi'))  # STYLE_IDX_INIT = 5
CONTEXT = 128  # the apps' CTX_LEN on the M1
DEVICE = 'mps'
ROWS_PER_PASS = 45  # windows per forward pass that fit in the M1's memory
RESULTS = ROOT / f'test/results/button_repeats_recordings__{CLEAR.stem}_and_{TESTER}.txt'


@torch.inference_mode()
def decoder_logits(model, pitch, button, style, style_mask):
    """Next-note logits [B, T, 128] for the targets pitch[:, 1:], with pitch and discrete button [B, T + 1], as the
    uncached branch of each model's gen_pitch_token. Without chord inputs the clear checkpoint is the free style joker."""
    if hasattr(model, '_buttons_discrete_to_inputs'):  # style joker: buttons with a joker mask
        button_real, joker_mask = model._buttons_discrete_to_inputs(button)
        context = {'pitch': pitch[:, :-1], 'button': button_real[:, 1:], 'joker_mask': joker_mask[:, 1:]}
    else:  # AE_style
        context = {'pitch': pitch[:, :-1], 'button': model.quantizer.discrete_to_real(button)[:, 1:]}
    return model.decoder(context, style_context=style.expand(len(pitch), -1, -1), style_context_mask=style_mask.expand(len(pitch), -1))


@torch.inference_mode()
def replay(model, recording: Path, style_file: Path, device: torch.device) -> tuple:
    """Recorded pitch and buttons [notes] and the replayed next-note probabilities [notes, 128]."""
    pitch = np.array(midi_to_dict(str(recording) + '.mid')[0]['pitch'])
    log = Path(str(recording) + '_chords.txt').read_text().splitlines()
    require(log[-1].startswith('buttons\t') and len(log) == 2, 'Expected a button log with no chord presses')
    buttons = np.array([int(value) for value in log[-1].split('\t')[1].split()])
    require(len(buttons) == len(pitch), 'The buttons do not match the recorded notes')
    primer = midi_to_dict(str(PRIMER))[0]
    tokens = {key: torch.tensor(primer[key], dtype=torch.long, device=device).unsqueeze(0) for key in ('dtime', 'pitch', 'dur')}
    primer_buttons = model.real_to_discrete(model.encoder(tokens))[0, :CONTEXT]  # the apps encode the whole primer file
    windows = lambda values: values.unfold(0, CONTEXT + 1, 1)  # [notes, CONTEXT + 1], the apps' sliding window
    window_pitch = windows(torch.tensor(primer['pitch'][:CONTEXT] + pitch.tolist(), device=device))
    window_button = windows(torch.cat((primer_buttons, torch.as_tensor(buttons, device=device))))
    # The apps' style: the first style_seq_len notes of the file, padded and masked if shorter.
    length = model.style_encoder.max_seq_len
    notes = midi_to_dict(str(style_file))[0]['pitch'][:length]
    style_pitch = torch.full((1, length), model.cfg.get('pad_idx', 128), dtype=torch.long, device=device)
    style_pitch[0, :len(notes)] = torch.tensor(notes, device=device)
    style_mask = (torch.arange(length, device=device) < len(notes)).unsqueeze(0)
    style = model.encode_style(style_pitch, style_mask)
    logits = torch.cat([decoder_logits(model, window_pitch[part], window_button[part], style, style_mask)[:, -1]
                        for part in torch.arange(len(pitch), device=device).split(ROWS_PER_PASS)])
    return pitch, buttons, F.softmax(logits.float(), dim=-1).cpu().numpy()


@torch.inference_mode()
def real_music(model, data: ChordPitchSamplerDataset, moments: np.ndarray, device: torch.device) -> tuple:
    """Per passage: share of repeated notes that keep their button (NaN without repeats), the decoder's mean probability
    of the previous note at one-step button moves, and the share of one-step moves whose note repeats."""
    keep, probability, label_repeat = [], [], []
    for onset in moments.tolist():
        piece = data.note_piece[onset]
        start, end = data.piece_edges[piece], data.piece_edges[piece + 1]
        passage = torch.as_tensor(data.pitches[onset - PASSAGE_CONTEXT:onset + NOTES].astype(np.int64), device=device).unsqueeze(0)
        buttons = player_buttons(model, passage)  # as training; button 0 is unused
        style, style_mask = style_inputs(model, data.pitches[start:end], onset - PASSAGE_CONTEXT - start, onset + NOTES - start, device)
        probs = F.softmax(decoder_logits(model, passage, buttons, style, style_mask).float(), dim=-1)[0]  # [T, 128] for notes 1..T
        # Note pairs (t - 1, t) from t = 2, the first note with a used button before it.
        pitch, button = passage[0].cpu().numpy(), buttons[0].cpu().numpy()
        repeat = pitch[2:] == pitch[1:-1]
        one_step = np.abs(button[2:] - button[1:-1]) == 1
        previous = probs[1:].gather(-1, passage[0, 1:-1, None]).squeeze(-1).cpu().numpy()  # [T - 1], probability of note t - 1 at note t
        keep.append((button[2:] == button[1:-1])[repeat].mean() if repeat.any() else np.nan)
        probability.append(previous[one_step].mean())
        label_repeat.append(repeat[one_step].mean())
    return 100 * np.array(keep), 100 * np.array(probability), 100 * np.array(label_repeat)


def main() -> None:
    started = time.time()
    device = torch.device(DEVICE)
    models = {'clear': load_style_checkpoint(CLEAR, device), 'tester': load_model(TESTER, compile_mode='none').to(device).eval()}
    names = {'clear': 'Style joker (clear checkpoint, no chord)', 'tester': 'AE_style_tester'}
    events = read_events(TEST_PICKLE)
    data = ChordPitchSamplerDataset(events, PASSAGE_CONTEXT + NOTES, is_eval=True)
    moments, _ = choose_moments(data, overlap_per_piece(events.reshape(-1, 5), len(data.piece_edges) - 1))
    reference = {key: real_music(model, data, moments, device) for key, model in models.items()}
    print(f'real music done, {time.time() - started:.0f} s', flush=True)

    rng = np.random.default_rng(SEED)
    # 95 % range over passages of the decoder's probability of the previous note at one-step moves, per model.
    reference_range = {key: bootstrap_ci(values[1], rng) for key, values in reference.items()}
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    recording_lines, verdict2 = [], True
    for recording, key, style_file in RECORDINGS:
        pitch, buttons, probs = replay(models[key], recording, style_file, device)
        notes = np.arange(1, len(pitch))  # the first note's previous note is the primer's
        one_step = notes[np.abs(buttons[1:] - buttons[:-1]) == 1]
        previous = probs[one_step, pitch[one_step - 1]]  # probability of the previous note at one-step moves
        repeated = (pitch[one_step] == pitch[one_step - 1]).sum()
        recorded = probs[np.arange(len(pitch)), pitch]
        verdict2 &= 100 * previous.mean() > reference_range[key][1]
        recording_lines.append(f'{recording.parent.name + "/" + recording.name:<56} {names[key]:<42} {len(pitch):>5} {len(one_step):>9}'
                               f' {repeated:>9} {100 * previous.mean():>14.1f} {100 * np.exp(np.log(recorded).mean()):>14.1f}'
                               f'   {sha(Path(str(recording) + ".mid"))} {sha(Path(str(recording) + "_chords.txt"))}')

    git = lambda *args: subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    lines = ['#' * 64,
             f'Date: {datetime.datetime.now():%Y-%m-%d %H:%M}  Commit: {git("rev-parse", "--short", "HEAD")}'
             f'  Uncommitted files: {len(git("status", "--short").splitlines())}',
             f'Models: {CLEAR.name} without chord; {TESTER} ({models["tester"].cfg["ckpt_file_name"]})  Device: {DEVICE}',
             f'Real music: {TEST_PICKLE.name}, {PIECES} passages of {PASSAGE_CONTEXT} + {NOTES} notes, seed {SEED}. '
             f'Recordings replayed with the {CONTEXT}-note window and primer {PRIMER.name}.',
             f'Running time: {time.time() - started:.0f} s', '',
             'Real music, buttons from each model\'s encoder (its training buttons); means over passages, 95 % range over passages:']
    verdict1 = True
    for key, (keep, probability, label_repeat) in reference.items():
        low, high = bootstrap_ci(keep[np.isfinite(keep)], rng)
        verdict1 &= high < 50
        lines.append(f'  {names[key]:<42} repeated notes that keep their button {np.nanmean(keep):5.1f} % (range {low:.1f} to {high:.1f}, '
                     f'{np.isfinite(keep).sum()} passages with repeats); one-step moves whose note repeats {label_repeat.mean():4.1f} %; '
                     f'decoder probability of the previous note at one-step moves {probability.mean():4.1f} % '
                     f'(range {reference_range[key][0]:.1f} to {reference_range[key][1]:.1f})')
    lines += ['', 'Recordings: notes; notes on a button one step from the previous note\'s; how many of them repeated the previous note;',
              'mean probability the replay gives the previous note there; typical probability of the recorded notes (check); SHA-256 of .mid and _chords.txt.',
              f'{"Recording":<56} {"Model":<42} {"Notes":>5} {"One-step":>9} {"Repeated":>9} {"Previous note %":>14} {"Recorded note %":>14}']
    lines += recording_lines
    lines += ['', f'Claim 1 (training buttons keep the button on under half of the repeated notes, both models): {"supported" if verdict1 else "refuted"}.',
              f'Claim 2 (every recording above the real-music range of its model): {"supported" if verdict2 else "refuted"}.',
              f'Result: claim 1 {"supported" if verdict1 else "refuted"}, claim 2 {"supported" if verdict2 else "refuted"}.', '']
    report = '\n'.join(lines)
    print(report)
    with open(RESULTS, 'a', encoding='utf-8') as file:
        file.write(report + '\n')


if __name__ == '__main__':
    main()
