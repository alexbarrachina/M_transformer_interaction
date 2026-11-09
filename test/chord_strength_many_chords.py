"""
Question: Over many chords and pieces, does a new chord adapter give the held chord's notes more probability than the adapter the app uses now?
The new adapter is the first argument, by default the clear-passage one.
Supported if: new minus current has a 95 % range above zero both for the piece's real chord and for the average of all 24 major and minor triads.
Refuted if: either range reaches zero or is below it.
Models: save_models/D_Chord_Joker_change_latest_best.pth (current, the app's adapter, trained on all labelled notes) and a new adapter: save_models/D_Chord_Joker_clear_latest_best.pth (clear, trained only on clear passages) or save_models/D_Chord_Joker_transposed_latest_best.pth (transposed, half of its windows move the notes after the press and their chord by up to a tritone). Same frozen base.
For the transposed adapter, claim 2 of test/results/eval_chord_strength_training_data.txt, written before it was trained, also applies: supported if the major chord a tritone above the real root gains with a 95 % range above zero and the real chord loses less than 5 percentage points on average.
Data: Training-Data/giantmidi_full_w_chords_unseen_test, the adapters' validation split. Validation chose each adapter's best checkpoint but never trained it. Pieces with over 10 % of their 8-note sequences in the base's or the adapters' training data are left out.
Method: one chord change at random in each piece. From the pianist's 200 notes before it, each model generates 8 notes with a chord held, once for the piece's real chord and once for each of the 24 triads, plus once with no chord pressed. All runs at a moment share their sampling noise. No guidance.
Measure: probability given to the held chord's notes, averaged over the 8 generated notes, new minus current, in percentage points. One value per piece. Higher means a stronger effect.
Cost: base NLL, how unlikely the frozen base without chord input finds the generated notes (lower is more likely), compared with the current adapter and with the pianist's real notes at the same moments.
Exploratory: results by how many of the triad's notes are in the real chord, the chords of test/eval_chord_strength.py, and notes by position.
Size and time: 60 pieces, 25 chords each, 303 s on the M1.
Run: python test/chord_strength_many_chords.py [checkpoint of the new adapter]
Result for clear (2026-11-07): supported in these 60 pieces. The real chord's notes get 78.7 % against 67.9 %, and all 24 triads on average 71.0 % against 47.0 %; the generated notes stay as likely as the pianist's with the real chord but become less likely with triads outside it. See test/results/chord_strength_many_chords.txt.
Result for transposed (2026-11-08): refuted as a general strengthening, because the real chord gains only +2.0 points (range -1.1 to +5.2); its own claim 2 is supported, with the tritone chord +31.4 points (range +26.1 to +36.9) and no loss on the real chord. Its notes are less likely under the base than with clear: base NLL 3.01 against 2.70 with the real chord, 3.68 against 3.20 over all triads.
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

from eval_chord_strength import bootstrap_ci, generate, load_checkpoint, permutation_p, require
from filter_unseen_pickles import pitch_ngrams
from midiUtils import Any_Pickle_File_Reader
from train_chords import ChordPitchSamplerDataset
from x_transformer import D_Chord_Joker

CURRENT = ROOT / 'save_models/D_Chord_Joker_change_latest_best.pth'
# The adapter compared with CURRENT: the first argument, by default the clear-passage one.
NEW = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'save_models/D_Chord_Joker_clear_latest_best.pth'
TEST_PICKLE = ROOT / 'Training-Data/giantmidi_full_w_chords_unseen_test.pickle'
TRAINING_PICKLES = (ROOT / 'Training-Data/giantmidi_full_train.pickle',  # the base's training data
                    ROOT / 'Training-Data/giantmidi_full_w_chords_unseen_train.pickle')  # the adapters'
MAX_OVERLAP = 0.1  # share of a piece's 8-note sequences allowed in those training data
PIECES = 60
CONTEXT = 200
NOTES = 8
SEED = 1729
DEVICE = 'mps'
RESULTS = ROOT / 'test/results/chord_strength_many_chords.txt'
RAW_VALUES = ROOT / f'out/chord_strength_many_chords_{NEW.stem}.npz'  # to look at the results another way without rerunning


def read_events(path: Path) -> np.ndarray:
    return np.asarray(Any_Pickle_File_Reader(str(path), verbose=False), dtype=np.int16)


def overlap_per_piece(events: np.ndarray, pieces: int) -> np.ndarray:
    """Largest share of each piece's 8-note sequences found in any of the training data."""
    hashes, piece = pitch_ngrams(events)
    total = np.maximum(np.bincount(piece, minlength=pieces), 1)
    overlap = np.zeros(pieces)
    for path in TRAINING_PICKLES:
        known = np.unique(pitch_ngrams(read_events(path).reshape(-1, 5))[0])
        overlap = np.maximum(overlap, np.bincount(piece[np.isin(hashes, known)], minlength=pieces) / total)
    return overlap


def choose_moments(data: ChordPitchSamplerDataset, overlap: np.ndarray, notes: int = NOTES) -> tuple:
    """One chord change at random per piece, with at least notes after it, from pieces spread over the pickle;
    also how many pieces could be chosen."""
    piece = data.note_piece[data.onsets]
    usable = ((data.onsets - data.piece_edges[piece] >= CONTEXT) & (data.onsets + notes <= data.piece_edges[piece + 1])
              & (overlap[piece] <= MAX_OVERLAP))
    pieces = np.unique(piece[usable])
    require(len(pieces) >= PIECES, f'Only {len(pieces)} usable pieces')
    chosen = pieces[np.linspace(0, len(pieces) - 1, PIECES).round().astype(int)]
    rng = np.random.default_rng(SEED)
    return np.array([rng.choice(data.onsets[usable & (piece == index)]) for index in chosen]), len(pieces)


def chords_at(data: ChordPitchSamplerDataset, onset: int) -> dict:
    """The piece's real chord, then the 12 major and the 12 minor triads."""
    group = data.groups[onset]
    chords = {'real': {'pcs': np.flatnonzero(data.chroma[group]).tolist(), 'anchor': int(data.anchors[group])}}
    for third, quality in ((4, 'major'), (3, 'minor')):
        for root in range(12):
            chords[f'{quality} {root}'] = {'pcs': sorted({root, (root + third) % 12, (root + 7) % 12}), 'anchor': root}
    return chords


def measure_run(run: dict, member: np.ndarray, last_pitch: int) -> dict:
    """member: [chords, 12]. Row 0 of the run has no chord pressed; the others follow member."""
    pc_mass = np.asarray(run['pc_mass'])  # [rows, notes, 12]
    pitches = np.asarray(run['pitches'])  # [rows, notes]
    jumps = np.abs(np.diff(np.c_[np.full(len(pitches), last_pitch), pitches], axis=1))
    return {'mass': (pc_mass[1:] * member[:, None]).sum(-1),  # [chords, notes]
            'free_mass': (pc_mass[0] * member[:, None]).sum(-1),  # [chords, notes]
            'nll': -np.asarray(run['base_logp'])[1:].mean(-1),  # [chords]
            'free_nll': np.full(len(member), -np.mean(run['base_logp'][0])),  # [chords]
            'far': (jumps[1:] > D_Chord_Joker.guidance_range).mean(-1),  # [chords]
            'repeated': (jumps[1:] == 0).mean(-1),  # [chords]
            'chord_notes': member[np.arange(len(member))[:, None], pitches[1:] % 12].mean(-1)}  # [chords]


@torch.inference_mode()
def pianist_notes(model, data: ChordPitchSamplerDataset, onset: int, member_real: np.ndarray, device: torch.device) -> dict:
    """The pianist's real next notes, scored like the generated ones."""
    pitch = torch.as_tensor(data.pitches[onset - CONTEXT:onset + NOTES].astype(np.int64), device=device).unsqueeze(0)
    logp = F.log_softmax(model.decoder({'pitch': pitch[:, :-1]}).float(), dim=-1)[0, CONTEXT - 1:]
    real = data.pitches[onset:onset + NOTES].astype(np.int64)
    jumps = np.abs(np.diff(np.r_[data.pitches[onset - 1], real]))
    return {'nll': -logp.gather(-1, pitch[0, CONTEXT:].unsqueeze(-1)).mean().item(),
            'far': (jumps > D_Chord_Joker.guidance_range).mean(), 'repeated': (jumps == 0).mean(),
            'chord_notes': member_real[real % 12].mean()}


def per_piece(values: np.ndarray, include: np.ndarray) -> np.ndarray:
    """Mean of [pieces, chords] values over the included chords; pieces with none of them drop out."""
    count = include.sum(1)
    return (values * include).sum(1)[count > 0] / count[count > 0]


def table_row(label: str, include: np.ndarray, current: dict, new: dict, rng: np.random.Generator) -> str:
    free, before, after = [100 * per_piece(value, include) for value in
                           (current['free_mass'].mean(-1), current['mass'].mean(-1), new['mass'].mean(-1))]
    low, high = bootstrap_ci(after - before, rng)
    nll_before, nll_after = per_piece(current['nll'], include), per_piece(new['nll'], include)
    nll_low, nll_high = bootstrap_ci(nll_after - nll_before, rng)
    return (f'{label:<44} {len(free):>6} {free.mean():>8.1f} {before.mean():>9.1f} {after.mean():>7.1f}'
            f' {(after - before).mean():>+8.1f} [{low:+6.1f},{high:+6.1f}] {(after > before).sum():>5}'
            f' {nll_before.mean():>8.2f} {nll_after.mean():>7.2f} {(nll_after - nll_before).mean():>+7.2f} [{nll_low:+.2f},{nll_high:+.2f}]')


def main() -> None:
    started = time.time()
    device = torch.device(DEVICE)
    events = read_events(TEST_PICKLE)
    data = ChordPitchSamplerDataset(events, CONTEXT + NOTES, is_eval=True)
    pieces_in_pickle = len(data.piece_edges) - 1
    moments, usable_pieces = choose_moments(data, overlap_per_piece(events.reshape(-1, 5), pieces_in_pickle))
    models = [load_checkpoint(CURRENT, device), load_checkpoint(NEW, device)]
    saved = [{}, {}]
    pianist = []
    for index, onset in enumerate(moments.tolist()):
        chords = chords_at(data, onset)
        member = np.zeros((len(chords), 12), dtype=bool)
        for row, chord in enumerate(chords.values()):
            member[row, chord['pcs']] = True
        noise = torch.empty(1, NOTES, 128).exponential_(generator=torch.Generator().manual_seed(SEED + index))
        runs = [generate(model, {'pitch': data.pitches}, onset, chords, noise, CONTEXT, 1.0, device) for model in models]
        require(runs[0]['pitches'][0] == runs[1]['pitches'][0], 'Runs with no chord differ: the frozen bases differ')
        for store, run in zip(saved, runs):
            for key, value in measure_run(run, member, int(data.pitches[onset - 1])).items():
                store.setdefault(key, []).append(value)
        saved[0].setdefault('member', []).append(member)
        saved[0].setdefault('root', []).append(chords['real']['anchor'])
        pianist.append(pianist_notes(models[0], data, onset, member[0], device))
        print(f'moment {index + 1}/{PIECES}, {time.time() - started:.0f} s', flush=True)
    member, real_root = np.stack(saved[0].pop('member')), np.asarray(saved[0].pop('root'))  # [pieces, chords, 12], [pieces]
    current, new = [{key: np.stack(value) for key, value in store.items()} for store in saved]
    np.savez(RAW_VALUES, moments=moments, member=member, real_root=real_root,
             **{f'current_{key}': value for key, value in current.items()},
             **{f'new_{key}': value for key, value in new.items()},
             **{f'pianist_{key}': np.array([row[key] for row in pianist]) for key in pianist[0]})

    # Which chords each table row averages, as [pieces, chords] masks. Chord 0 is the real one.
    real = np.zeros(member.shape[:2], dtype=bool)
    real[:, 0] = True
    triads = ~real
    shared = (member & member[:, :1]).sum(-1)  # how many of each chord's notes are in the real chord
    roots = np.tile(np.r_[0, np.arange(12), np.arange(12)], (len(moments), 1))
    is_major = np.tile(np.r_[False, np.ones(12, dtype=bool), np.zeros(12, dtype=bool)], (len(moments), 1))
    major_above = lambda interval: triads & is_major & (roots == (real_root[:, None] + interval) % 12)

    rng = np.random.default_rng(SEED)
    gain = 100 * (new['mass'].mean(-1) - current['mass'].mean(-1))  # [pieces, chords], percentage points
    gain_real, gain_triads = per_piece(gain, real), per_piece(gain, triads)
    range_real, range_triads = bootstrap_ci(gain_real, rng), bootstrap_ci(gain_triads, rng)
    verdict = 'supported' if min(range_real[0], range_triads[0]) > 0 else 'refuted'

    head = (f'{"Chord held":<44} {"Pieces":>6} {"NoChord%":>8} {"Current%":>9} {"New%":>7}'
            f' {"New-current, points [95 % range]":>24} {"Higher":>5}'
            f' {"NLLcurr":>8} {"NLLnew":>7} {"NLL new-current [95 % range]"}')
    groups = [("The piece's real chord", real), ('All 24 triads', triads)]
    groups += [(f'Triads with {count} of 3 notes in the real chord', triads & (shared == count)) for count in (3, 2, 1, 0)]
    groups += [(f'Major chord {name} the real root', major_above(interval))
               for name, interval in (('a fifth above', 7), ('a fourth above', 5), ('a tritone above', 6))]
    per_note = lambda store, mask: 100 * (store['mass'] * mask[..., None]).sum((0, 1)) / mask.sum()
    git = lambda *args: subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    lines = ['#' * 64,
             f'Date: {datetime.datetime.now():%Y-%m-%d %H:%M}  Commit: {git("rev-parse", "--short", "HEAD")}'
             f'  Uncommitted files: {len(git("status", "--short").splitlines())}',
             f'Current: {CURRENT.name}  New: {NEW.name}  Device: {DEVICE}',
             f'Data: {TEST_PICKLE.name}, {PIECES} of the {usable_pieces} usable pieces ({pieces_in_pickle} in the file), one chord change each. Context {CONTEXT} notes, '
             f'{NOTES} generated notes, seed {SEED}, no guidance, at most {MAX_OVERLAP:.0%} of a piece in training data.',
             f'Running time: {time.time() - started:.0f} s', '', head]
    lines += [table_row(label, mask, current, new, rng) for label, mask in groups]
    lines += ['', 'Real chord, chord notes among the generated notes and other costs at the same moments:',
              *[f'  {name:<8} chord notes {100 * chord_notes:5.1f} %  base NLL {nll:.2f}  over 8 semitones {100 * far:5.1f} %  repeated {100 * repeated:4.1f} %'
                for name, chord_notes, nll, far, repeated in
                (('Current', current['chord_notes'][:, 0].mean(), current['nll'][:, 0].mean(), current['far'][:, 0].mean(), current['repeated'][:, 0].mean()),
                 ('New', new['chord_notes'][:, 0].mean(), new['nll'][:, 0].mean(), new['far'][:, 0].mean(), new['repeated'][:, 0].mean()),
                 ('Pianist', *[np.mean([row[key] for row in pianist]) for key in ('chord_notes', 'nll', 'far', 'repeated')]))],
              f'  No chord pressed: base NLL {current["free_nll"][:, 0].mean():.2f}',
              '', 'Probability on the chord notes by generated note, % (exploratory):']
    for label, mask in groups[:2]:
        lines += [f'  {label:<24} current ' + ' '.join(f'{value:5.1f}' for value in per_note(current, mask)),
                  f'  {"":<24} new     ' + ' '.join(f'{value:5.1f}' for value in per_note(new, mask))]
    lines += ['', f'Real chord: {gain_real.mean():+.1f} points, 95 % range {range_real[0]:+.1f} to {range_real[1]:+.1f}, p = {permutation_p(gain_real, rng):.1e}, higher in {(gain_real > 0).sum()} of {len(gain_real)} pieces.',
              f'All 24 triads: {gain_triads.mean():+.1f} points, 95 % range {range_triads[0]:+.1f} to {range_triads[1]:+.1f}, p = {permutation_p(gain_triads, rng):.1e}, higher in {(gain_triads > 0).sum()} of {len(gain_triads)} pieces.',
              f'Result: {verdict}.', '']
    report = '\n'.join(lines)
    print(report)
    with open(RESULTS, 'a', encoding='utf-8') as file:
        file.write(report + '\n')


if __name__ == '__main__':
    main()
