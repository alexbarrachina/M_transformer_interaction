#===================================================================================================
# Monster Genie filter_unseen_pickles.py Python module
# Keeps only the chord-pickle pieces that the frozen base joker never saw in training
#
# The base joker (D_Base_Joker_little_v1) was trained on giantmidi_full_train. The chord pickles
# were split again at random, so ~83 % of their pieces (train AND test) are pieces the base has
# memorised. There the base already knows the next note, so the chord adds nothing to learn
# (docs/D_Chord_Joker_steering_plan.md, step 2). The pickles store no file names, so a piece is
# recognised by its notes: it counts as seen when most of its 8-note pitch sequences also appear
# in the base training pickle (seen pieces have >= 90 % of them there, unseen pieces ~0 %).
#
# Output: giantmidi_full_w_chords_unseen_{train,test}.pickle, same format, whole pieces only.
# Rerun it whenever the chord pickles are rebuilt (midis2pickles.py splits at random).
#
# Copyright 2025 Alex Barrachina
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===================================================================================================


import numpy as np

from midiUtils import Any_Pickle_File_Reader, Any_Pickle_File_Writer
from params import CHORDS_CHANNEL, HARMONY_CHANNEL

BASE_TRAIN_PATH = './Training-Data/giantmidi_full_train'  # what the frozen base was trained on
CHORD_PATHS = {'train': './Training-Data/giantmidi_full_w_chords_train',
               'test': './Training-Data/giantmidi_full_w_chords_test'}
OUTPUT_NAME = './Training-Data/giantmidi_full_w_chords_unseen'
NGRAM = 8             # length of the pitch sequences that recognise a piece
SEEN_FRACTION = 0.5   # seen = more than half of the piece's sequences are in the base data
BOUNDARY = np.array([126, 126, 0, 0, 0])  # first event of every piece

#==========================================================================

def pitch_ngrams(events: np.ndarray):
    """Hash of every run of NGRAM performed pitches inside one piece, and that piece's index."""
    boundary = (events == BOUNDARY).all(axis=1)
    channel = events[:, 4]
    # Same performed-note filter as ChordPitchSamplerDataset: annotations are not music.
    notes = (~boundary & (channel >= 0) & (channel < 16) & (channel != 9)
             & (channel != HARMONY_CHANNEL) & (channel != CHORDS_CHANNEL) & (events[:, 3] > 0))
    pitch = events[notes, 2].astype(np.uint64)
    piece = np.cumsum(boundary)[notes]
    count = max(len(pitch) - NGRAM + 1, 0)
    hashes = np.zeros(count, dtype=np.uint64)
    for offset in range(NGRAM):
        hashes = hashes * np.uint64(131) + pitch[offset:offset + count]
    inside = piece[:count] == piece[NGRAM - 1:]  # drop sequences that cross a piece boundary
    return hashes[inside], piece[:count][inside]


def main() -> None:
    base = np.asarray(Any_Pickle_File_Reader(BASE_TRAIN_PATH), dtype=np.int16).reshape(-1, 5)
    base_ngrams = np.unique(pitch_ngrams(base)[0])
    del base

    for split, path in CHORD_PATHS.items():
        events = np.asarray(Any_Pickle_File_Reader(path), dtype=np.int16).reshape(-1, 5)
        event_piece = np.cumsum((events == BOUNDARY).all(axis=1))
        pieces = int(event_piece[-1]) + 1
        hashes, piece = pitch_ngrams(events)
        # Share of each piece's pitch sequences that the base was trained on.
        total = np.bincount(piece, minlength=pieces)
        in_base = np.bincount(piece, weights=np.isin(hashes, base_ngrams), minlength=pieces)
        unseen = (total > 0) & (in_base <= SEEN_FRACTION * total)
        print('=' * 70)
        print(f'{split}: {unseen.sum()} unseen of {(total > 0).sum()} pieces')
        # Whole pieces (boundary, notes, chords, keys) are copied unchanged.
        Any_Pickle_File_Writer(events[unseen[event_piece]].reshape(-1).tolist(), f'{OUTPUT_NAME}_{split}')
        print('=' * 70)


if __name__ == '__main__':
    main()
