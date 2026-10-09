#===================================================================================================
# Monster Genie midis2pickles.py Python module
# Converts MIDI files into a pickle file
#
# Copyright 2025 Alex Barrachina
#
# Based on Project Los Angeles / Tegridy Code 2025
# https://github.com/asigalov61/monsterpianotransformer
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
# limitations under the License.'''
#===================================================================================================


import random
import os
import re
from collections import Counter
from tqdm import tqdm

from midiUtils import midi2ms_score, Any_Pickle_File_Writer, parse_harmony_label, identify_chord
from roman_movement import roman_from_label, parse_marker, key_from_marker, RomanMovementReducer
from params import *

#===================================================================================================
# OVERVIEW
#===================================================================================================
# Reads a folder of MIDI files and writes two pickles (train / test): one flat list of integers,
# 5 tokens per event, with a [126, 126, 0, 0, 0] boundary event in front of every piece.
#
# Event layout (NO OFFSETS: every token is stored raw, 0-127; the channel can also be a pseudo
# channel > 15 for the harmony events below):
#
#   [dtime, dur, pitch, vel, chan]
#
# Time is quantized by time2quant() / dur2quant() below: dtime in 10 ms units, dur in 20 ms units,
# both clipped to 126 (max dtime 1.26 s, max dur 2.52 s). The MIDI is read with midi2ms_score(),
# i.e. 1 tick = 1 ms, so for all_key_tension (1 tick = 1 ms of REAL performance time) the units
# above are real time. Older folders (all_roman, all_roman_time_fixed) played at 2x / 4x slow.
#
# --- Playable notes: the only events that advance time ----------------------------------------
#   chan 0   MELODY_CHANNEL         MIDI channel 1   [dtime, dur, pitch, vel, 0]
#   chan 1   MELODY_CHORDS_CHANNEL  MIDI channel 2   [dtime, dur, pitch, vel, 1]
#   chan 10  ACCOMP_CHANNEL         MIDI channel 11  [dtime, dur, pitch, vel, 10]
#   chan 11  EXTRA_ACCOMP_CHANNEL   MIDI channel 12  [dtime, dur, pitch, vel, 11]
#   dtime is measured from the previous PLAYABLE note (variable `pe` below), whatever its channel.
#   Every event listed from here on has dtime = 0 and never updates `pe`, so the playable-note
#   timing is identical whether or not the harmony events are present.
#   Which MIDI channels are kept is chosen by `useful_channels` (see its comment). The default
#   keeps all four note channels; the chord tones / tension / keys were computed from all of them.
#
# --- Harmony events: the conditioning data (all with dtime = 0) --------------------------------
#   chan 4   CHORDS_CHANNEL        [0, dur, pitch, vel, 4]       one event per chord tone
#                                  [0, 0, 0, 0, 4]               chord-off marker (all chord tones expired)
#            The chord tones are the CHROMA source: the loaders turn them into a 12-bin
#            pitch-class vector and a bass pitch class at load time (nothing is stored here).
#   chan 120 CHORD_LABEL_CHANNEL   [0, root_pc, quality_id, function_id, 120]
#            One per chord, written just before its first chord tone.
#            root_pc 0-11 (C=0), PC_UNKNOWN=12; quality_id = QUALITY_* in params.py;
#            function_id = FUNC_*, always FUNC_UNKNOWN for all_key_tension (no functional labels).
#   chan 121 KEY_CHANNEL           [0, key_pc, mode, is_change, 121]
#            Local key; mode 0 = major, 1 = minor. is_change = 1 when a 'change key:' marker
#            starts at the same time, 0 for the first key of the piece.
#   chan 124 HOME_KEY_CHANNEL      [0, home_pc, mode, 0, 124]    once per piece, the global key
#   chan 123 TENSION_CHANNEL       [0, tension_level, 0, 0, 123] at every chord onset;
#            0 (relaxed) .. 4 (tense). A falling level is a resolution - no resolve event exists.
#   Order inside one quantized time step: home key, local key, tension, then the chord label
#   and its chord tones (they come from the note stream, right after the pseudo-events).
#
# --- Legacy harmony events (NOT written while KEY_TENSION_ONLY is True) -------------------------
#   chan 3   HARMONY_CHANNEL       [0, 0, movement_type, 0, 3]   old MIDI channel 4 movement notes
#            (sta/prp/ten/sol/col/cho/evd/mod = pitch 60..67), the OLD "tension" information.
#   chan 122 ROMAN_MOVE_CHANNEL    [0, roman_move_id, 0, 0, 122] 5-class movement from musiclang romans
#   In the old pipeline the chord notes (old chroma) lived on MIDI channel 5; in all_key_tension
#   the java chords live on MIDI channel 4 instead. See "DATASET FORMATS".
#
#===================================================================================================
# DATASET FORMATS
#===================================================================================================
# 1) all_key_tension  (current, KEY_TENSION_ONLY = True)   built by build_key_tension_dataset.py
#      MIDI channels : 1 / 2 / 11 performance notes, 4 = java chord tones in root position
#                      (root = lowest tone), velocity 60
#      markers       : 'home: C major'  'key: C major'  'change key: C major -> G major'
#                      'tension: 0..4'  (the old java / musiclang markers were removed)
#      A file is recognised as this format by having a 'home:' or 'tension:' marker.
#
# 2) legacy (all_roman, harmony_*): chord notes on MIDI channel 5, movement notes on MIDI channel
#    4, German chord labels or musiclang romans as markers. Only processed with
#    KEY_TENSION_ONLY = False. MIDI channel 4 means something DIFFERENT in the two formats
#    (movement notes vs. chord tones), so mixing them would silently corrupt the chords.
#    That is why KEY_TENSION_ONLY skips every file that is not format 1, and why the pickle is
#    checked for legacy channels before it is written (see "Legacy-content guard" below).
#===================================================================================================


#===================================================================================================
# SETTINGS
#===================================================================================================
melody_only = False # if True, only process melody notes (channel 0)  (not used below, kept for reference)
sorted_or_random_file_loading_order = False # Sorted order is NOT usually recommended
dataset_ratio = 1 # Change this if you need more or less % of the dataset

#train_and_test_ratio = 1. # 100% for training
train_and_test_ratio = 0.8 # 80% for training, 20% for testing

# Which notes go into the pickle. Values refer to the pickle channel numbers (= MIDI channel - 1).
#   MELODY_ONLY / ACCOMP_ONLY / MELODY_AND_ACCOMP : keep channel 0 and/or 10 plus the harmony
#       channels. Everything else is dropped - in particular MIDI channels 2 and 12 (pickle 1
#       and 11); this is what the earlier pickles used.
#   MELODY_AND_ACCOMP_NOT_HARMONY : melody + accompaniment only, no chords at all.
#   ALL_CHANNELS : keep every channel, whatever it is (no filtering at all).
#   ALL_NOTE_CHANNELS : keep exactly the four note channels MIDI 1, 2, 11 and 12 (pickle 0, 1,
#       10, 11) plus the harmony channels, and nothing else. A stray channel in a MIDI file is
#       still dropped, which ALL_CHANNELS would not do.
MELODY_ONLY = 0 # melody only (harmony not used for dtime)
ACCOMP_ONLY = 1 # accompaniment only (harmony not used for dtime)
MELODY_AND_ACCOMP = 2 # melody and accompaniment (harmony not used for dtime)
MELODY_AND_ACCOMP_NOT_HARMONY = 3 # melody, accompaniment. Not harmony no chords
ALL_CHANNELS = 4 # we will use all channels in the pickle (harmony not used for dtime)
ALL_NOTE_CHANNELS = 5 # MIDI channels 1, 2, 11, 12 + harmony (harmony not used for dtime)

useful_channels = ALL_NOTE_CHANNELS

# True  -> build the key+tension pickle: files that are not in the all_key_tension format are
#          skipped, old-format content (movement notes, channel-5 chord notes, musiclang romans,
#          German chord labels) is never copied, and the finished pickle is checked for legacy
#          channels before it is saved.
# False -> the previous behaviour for the legacy folders.
KEY_TENSION_ONLY = True

#===================================================================================================
# COUNTERS (printed at the end)
#===================================================================================================
files_count = 0
skipped_legacy_files = 0 # files not in the all_key_tension format, skipped by KEY_TENSION_ONLY

gfiles = []

train_data1 = []
test_data1 = []

# Channel statistics
total_notes = 0
channel_0_notes = 0      # MIDI channel 1
channel_1_notes = 0      # MIDI channel 2
channel_10_notes = 0     # MIDI channel 11
channel_11_notes = 0     # MIDI channel 12
channel_4_movements = 0  # legacy chan 3 events: must stay 0 for the key+tension pickle
channel_5_chords = 0     # chord-tone events written on CHORDS_CHANNEL (pickle chan 4)
chord_label_events = 0   # one packed (root, quality, function) event per chord
key_events = 0           # one (key_pc, mode) event per key change
chords_with_label = 0    # chords whose factors came from a parsed text label
chords_derived = 0       # chords whose factors were derived from the notes (no label)
roman_move_events = 0    # legacy chan 122 events: must stay 0 for the key+tension pickle
key_tension_files = 0    # files in the all_key_tension format
home_key_events = 0      # one HOME_KEY_CHANNEL event per key_tension file
key_change_events = 0    # KEY_CHANNEL events flagged is_change=1
tension_events = 0       # one TENSION_CHANNEL event per chord onset
tension_histogram = [0] * NUM_TENSION_LEVELS

#===================================================================================================
# INPUT / OUTPUT
#===================================================================================================
# dataset_addr = "./Samples"  # when testing
# dataset_addr = "../../../DataSets/MIDI/giantMIDI/all_roman"   # legacy: misaligned, 2x/4x slow
dataset_addr = "../../../DataSets/MIDI/giantMIDI/all_key_tension"
# Output file names (written to ./Training-Data/<output_name>_train.pickle / _test.pickle)
output_name = 'giantmidi_key_tension'

# all_key_tension: java chord tones live on MIDI channel 4 (mido 3), not on channel 5. They are
# renamed to CHORDS_CHANNEL below, because pickle channel 3 means "legacy movement note".
KEY_TENSION_CHORD_TONES_CHANNEL = 3

# Marker grammar of all_key_tension (case/space tolerant):
#   'tension: 3'   'home: F# minor'   'change key: C major -> G major'
# The plain 'key: C major' marker is parsed by roman_movement.key_from_marker via parse_marker.
_TENSION_MARKER = re.compile(r'^\s*tension\s*:\s*(\d+)\s*$', re.IGNORECASE)
_HOME_MARKER = re.compile(r'^\s*home\s*:\s*(.+?)\s*$', re.IGNORECASE)
_CHANGE_KEY_MARKER = re.compile(r'^\s*change\s+key\s*:\s*(.+?)\s*->\s*(.+?)\s*$', re.IGNORECASE)


# Java chord shapes missing from QUALITY_TEMPLATES; both act as dominant sevenths.
_ROOT_POSITION_EXTRA_QUALITIES = {
    frozenset({0, 4, 10}): QUALITY_DOMINANT_SEVENTH,      # dominant 7th without fifth
    frozenset({0, 4, 8, 10}): QUALITY_DOMINANT_SEVENTH,   # augmented 7th (V7#5)
}


def root_position_chord(pcs, bass_pitch):
    """all_key_tension chord tones are written in root position (root + ascending
    intervals), so the lowest tone IS the java root - no guessing, which also
    settles symmetric chords (dim7, aug) that identify_chord roots arbitrarily.
    Returns (root_pc, quality_id) for the CHORD_LABEL_CHANNEL event."""
    if bass_pitch is None or not pcs:
        return identify_chord(pcs)
    root = bass_pitch % 12
    intervals = frozenset((p - root) % 12 for p in pcs)
    quality = QUALITY_TEMPLATES.get(intervals, _ROOT_POSITION_EXTRA_QUALITIES.get(intervals, QUALITY_UNKNOWN))
    return root, quality


def parse_key_tension_marker(text):
    """all_key_tension markers -> ('tension', level) | ('home', pc, mode) |
    ('change', pc, mode) | None. Must run before parse_harmony_label, which
    misreads 'change key: C major -> G major' as a German chord label.
    'key: X' markers are left to parse_marker."""
    m = _TENSION_MARKER.match(text)
    if m:
        return ('tension', min(int(m.group(1)), NUM_TENSION_LEVELS - 1))
    m = _HOME_MARKER.match(text)
    if m:
        k = key_from_marker('key: ' + m.group(1))
        return ('home', k[0], k[1]) if k else None
    m = _CHANGE_KEY_MARKER.match(text)
    if m:
        k = key_from_marker('key: ' + m.group(2))
        return ('change', k[0], k[1]) if k else None
    return None

# Reducer: roman-numeral labels -> 5 harmony-movement classes (+ NULL), computed
# per chord transition (see roman_movement.py / the reduction-map spec).
# Legacy only: it never sees data while KEY_TENSION_ONLY is True.
roman_reducer = RomanMovementReducer()

# Process MIDIs

# first quantize the time and duration, then calculate the time difference and maximum duration
# (time in 10 ms steps, duration in 20 ms steps - note the different units, see chord expiry below)
def time2quant(time):
    return int(time/10)

def dur2quant(dur):
    return int(dur/20)


#===================================================================================================
# FILE LIST
#===================================================================================================
filez = list()
for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
    filez += [os.path.join(dirpath, file) for file in filenames]
print('=' * 70)

if filez == []:
    print('Could not find any MIDI files. Please check Dataset dir...')
    print('=' * 70)

if sorted_or_random_file_loading_order:
    print('Sorting files...')
    filez.sort()
    print('Done!')
    print('=' * 70)
else:
    print('Randomizing file list...')
    random.shuffle(filez)


#===================================================================================================
# MAIN LOOP: one MIDI file -> one piece in the pickle
#===================================================================================================
print('Processing MIDI files. Please wait...')
for f in tqdm(filez[:int(len(filez) * dataset_ratio)]):
    try:
        fn = os.path.basename(f)
        fn1 = fn.split('.')[0]

        #print('Loading MIDI file...')
        score = midi2ms_score(open(f, 'rb').read())

        events_matrix = []    # note events: ['note', start_ms, dur_ms, channel, pitch, velocity]
        label_events_ms = []  # (time_ms, parsed_label, roman_str) from German-label meta-events  [legacy]
        roman_events_ms = []  # (time_ms, roman_str) from 'all_roman' bare-roman markers          [legacy]
        key_events_ms = []    # (time_ms, key_pc, mode) from 'key: X minor' markers
        home_events_ms = []     # (time_ms, key_pc, mode) from 'home: X major' markers
        change_key_ms = set()   # times of 'change key: A -> B' markers
        tension_events_ms = []  # (time_ms, level) from 'tension: N' markers

        # --- 1) Read every track: collect the notes and sort the markers by dialect ----------
        itrack = 1

        while itrack < len(score):
            for event in score[itrack]:
                if event[0] == 'note' and event[3] != 9: # skip percussion notes
                    events_matrix.append(event)
                elif event[0] in ('marker', 'text_event', 'lyric', 'cue_point'):
                    # Chord-name + harmonic-function labels are carried as text meta-events.
                    raw = event[2]
                    try:
                        s = raw.decode('latin-1', 'ignore') if isinstance(raw, (bytes, bytearray)) else str(raw)
                    except Exception:
                        s = ''
                    # (0) all_key_tension dialect: home / change key / tension markers.
                    kt = parse_key_tension_marker(s)
                    if kt is not None:
                        if kt[0] == 'tension':
                            tension_events_ms.append((event[1], kt[1]))
                        elif kt[0] == 'home':
                            home_events_ms.append((event[1], kt[1], kt[2]))
                        else:
                            change_key_ms.add(event[1])
                        continue
                    # (a) German functional dialect: '<root> <QUAL>:...:<FUNC> (<roman>)'   [legacy]
                    parsed = parse_harmony_label(s)
                    if parsed is not None:
                        label_events_ms.append((event[1], parsed, roman_from_label(s)))
                    else:
                        # (b) 'all_roman' dialect: bare roman markers ('IV65/III',
                        #     'ii%', 'i') + key markers ('key: A minor').
                        #     The 'key:' markers are shared with all_key_tension.
                        kind = parse_marker(s)
                        if kind is not None and kind[0] == 'key':
                            key_events_ms.append((event[1], kind[1], kind[2]))
                        elif kind is not None:
                            roman_events_ms.append((event[1], kind[1]))
            itrack += 1

        # --- 2) Decide the dialect and keep old-format content out -----------------------------
        # A file is all_key_tension if it carries 'home:' / 'tension:' markers.
        is_key_tension = bool(tension_events_ms or home_events_ms)

        if KEY_TENSION_ONLY and not is_key_tension:
            # Legacy file: its MIDI channel 4 holds movement notes (old "tension" info) and its
            # channel 5 holds the old chord notes (old chroma). Neither may enter this pickle.
            skipped_legacy_files += 1
            continue

        if is_key_tension:
            # Defensive: a key+tension file must not carry any legacy harmony content, so
            # ignore legacy-format markers and legacy MIDI channel-5 chord notes (pickle chan 4).
            label_events_ms = []
            roman_events_ms = []
            events_matrix = [e for e in events_matrix if e[3] != CHORDS_CHANNEL]
            # The java chord tones are on MIDI channel 4 (pickle chan 3 = HARMONY_CHANNEL in the
            # legacy layout, i.e. movement notes) -> rename them to the chord channel so they
            # go through the chord path below instead of being read as movement events.
            for event in events_matrix:
                if event[3] == KEY_TENSION_CHORD_TONES_CHANNEL:
                    event[3] = CHORDS_CHANNEL

        if len(events_matrix) > 0:

          # Sorting by pitch (descending) then by time (ascending). when the second sort reorders by time,
          # notes with the same start time (i.e., chords) retain their relative order from the first sort (by pitch descending).
          events_matrix.sort(key=lambda x: x[4], reverse=True) # pitch
          events_matrix.sort(key=lambda x: x[1]) # time

          # --- 3) Channel selection (see `useful_channels`). HARMONY_CHANNEL / CHORDS_CHANNEL are
          # always kept except in the NOT_HARMONY mode; for key+tension files HARMONY_CHANNEL is empty.
          if useful_channels == MELODY_ONLY:
            filtered_events_matrix = [e for e in events_matrix if e[3] in (MELODY_CHANNEL, HARMONY_CHANNEL, CHORDS_CHANNEL)]
          elif useful_channels == ACCOMP_ONLY:
            filtered_events_matrix = [e for e in events_matrix if e[3] in (ACCOMP_CHANNEL, HARMONY_CHANNEL, CHORDS_CHANNEL)]
          elif useful_channels == MELODY_AND_ACCOMP:
            filtered_events_matrix = [e for e in events_matrix if e[3] in (MELODY_CHANNEL, ACCOMP_CHANNEL, HARMONY_CHANNEL, CHORDS_CHANNEL)]
          elif useful_channels == MELODY_AND_ACCOMP_NOT_HARMONY:
            filtered_events_matrix = [e for e in events_matrix if e[3] in (MELODY_CHANNEL, ACCOMP_CHANNEL)]
          elif useful_channels == ALL_NOTE_CHANNELS:
            filtered_events_matrix = [e for e in events_matrix if e[3] in (
                MELODY_CHANNEL, MELODY_CHORDS_CHANNEL, ACCOMP_CHANNEL, EXTRA_ACCOMP_CHANNEL,
                HARMONY_CHANNEL, CHORDS_CHANNEL)]
          else:
            filtered_events_matrix = events_matrix

          # Skip files with no playable notes
          if not any(e[3] not in (HARMONY_CHANNEL, CHORDS_CHANNEL) for e in filtered_events_matrix):
              continue

          # Quantize timings for all events: e[1] = start (10 ms units), e[2] = duration (20 ms units)
          for e in filtered_events_matrix:
              e[1] = time2quant(e[1])
              e[2] = dur2quant(e[2])

          # Determine if this file goes to train or test
          is_train = random.random() < train_and_test_ratio
          target_data = train_data1 if is_train else test_data1

          # Intro/Zero seq (5 tokens) - no offsets, raw values. Marks the start of a piece.
          target_data.extend([126, 126, 0, 0, 0])  # dtime, dur, pitch, vel, chan

          # pe tracks the last PLAYABLE note for dtime calculation
          # Harmony events (channel 3) and chord events (channel 4) do NOT update pe
          pe = next(e for e in filtered_events_matrix if e[3] not in (HARMONY_CHANNEL, CHORDS_CHANNEL))

          # Track when the current chord expires (abs time, quantized)
          chord_end_abs_time = -1

          # --- 4) Harmony-label lookups (quantized-time indexed) ---
          # Legacy German dialect: quantized onset time -> (root_pc, quality_id, key_pc, mode, function_id).
          # Empty for key+tension files, whose chord labels are derived from the chord tones instead.
          label_by_qtime = {}
          roman_by_qtime = {}
          for t_ms, parsed, roman_str in label_events_ms:
              q = time2quant(t_ms)
              label_by_qtime[q] = parsed
              roman_by_qtime[q] = roman_str
          # Roman-derived 5-class harmony movement, computed per chord TRANSITION
          # over the ordered chord-label sequence (key context = parsed key_pc/mode),
          # then keyed by quantized onset for lookup in the emit loop below.
          # Legacy only: empty for key+tension files (their roman markers were removed).
          movement_by_qtime = {}
          if label_by_qtime:   # German dialect: key is inside the parsed tuple
              ordered_q = sorted(label_by_qtime.keys())
              move_seq = [(roman_by_qtime.get(q), label_by_qtime[q][2], label_by_qtime[q][3])
                          for q in ordered_q]   # (roman_str, key_pc, mode)
              for q, mv in zip(ordered_q, roman_reducer.reduce(move_seq)):
                  movement_by_qtime[q] = mv
          if roman_events_ms:  # 'all_roman' dialect: romans + separate key markers
              key_tl = sorted(key_events_ms)
              move_seq = []; onset_qs = []
              ki = 0; kpc, kmode = PC_UNKNOWN, MODE_UNKNOWN
              for t_ms, roman_str in sorted(roman_events_ms):
                  while ki < len(key_tl) and key_tl[ki][0] <= t_ms:
                      kpc, kmode = key_tl[ki][1], key_tl[ki][2]; ki += 1
                  move_seq.append((roman_str, kpc, kmode))
                  onset_qs.append(time2quant(t_ms))
              for q, mv in zip(onset_qs, roman_reducer.reduce(move_seq)):
                  movement_by_qtime[q] = mv
          # Chord-tone pitch classes and lowest pitch per onset, for the chord label
          # (all_key_tension chords have no text label: root = lowest tone, quality from the intervals)
          chord_group_pcs = {}
          chord_group_bass = {}   # lowest chord-tone pitch per onset (= java root in all_key_tension)
          for e in filtered_events_matrix:
              if e[3] == CHORDS_CHANNEL:
                  chord_group_pcs.setdefault(e[1], set()).add(e[4] % 12)
                  chord_group_bass[e[1]] = min(chord_group_bass.get(e[1], 127), e[4])

          current_key = (PC_UNKNOWN, MODE_UNKNOWN)
          last_chord_label_qtime = None

          # --- 5) Pseudo-events from markers: (quantized time, order, 5 tokens) ---
          # They are injected in quantized-time order, just before the first event they precede.
          # Within one quantized time: 0 home key, 1 local key, 2 tension, 3 roman movement (legacy).
          pseudo_events = [(q, 3, [0, mv, 0, 0, ROMAN_MOVE_CHANNEL]) for q, mv in movement_by_qtime.items()]
          if is_key_tension:
              change_qs = {time2quant(t_ms) for t_ms in change_key_ms}
              # Global key, once per piece (at the time of the 'home:' marker, tick 0)
              for t_ms, kpc, kmode in home_events_ms:
                  pseudo_events.append((time2quant(t_ms), 0, [0, kpc, kmode, 0, HOME_KEY_CHANNEL]))
              # Local key: every 'key:' marker; is_change = 1 if a 'change key:' marker is at the same time
              for t_ms, kpc, kmode in key_events_ms:
                  q = time2quant(t_ms)
                  pseudo_events.append((q, 1, [0, kpc, kmode, 1 if q in change_qs else 0, KEY_CHANNEL]))
              # Tension level, one per chord onset
              for t_ms, level in tension_events_ms:
                  pseudo_events.append((time2quant(t_ms), 2, [0, level, 0, 0, TENSION_CHANNEL]))
          pseudo_events.sort(key=lambda p: (p[0], p[1]))
          ps_ptr = 0

          # --- 6) Emit the piece: walk the notes in time order, injecting pseudo-events first ---
          for e in filtered_events_matrix:

              # Flush every pseudo-event whose time has been reached
              while ps_ptr < len(pseudo_events) and pseudo_events[ps_ptr][0] <= e[1]:
                  tokens = pseudo_events[ps_ptr][2]
                  target_data.extend(tokens)
                  if tokens[4] == ROMAN_MOVE_CHANNEL:
                      roman_move_events += 1
                  elif tokens[4] == HOME_KEY_CHANNEL:
                      home_key_events += 1
                  elif tokens[4] == KEY_CHANNEL:
                      key_events += 1
                      key_change_events += tokens[3]
                  elif tokens[4] == TENSION_CHANNEL:
                      tension_events += 1
                      tension_histogram[tokens[1]] += 1
                  ps_ptr += 1

              if e[3] == HARMONY_CHANNEL:
                  # LEGACY harmony movement note: [0, 0, movement_type, 0, chan=3], pe NOT updated.
                  # Never reached for key+tension files (their chord tones were renamed above).
                  movement_type = max(0, min(126, e[4]))
                  target_data.extend([0, 0, movement_type, 0, HARMONY_CHANNEL])
                  channel_4_movements += 1
              elif e[3] == CHORDS_CHANNEL:
                  # On the FIRST tone of each chord group, emit compact harmony
                  # conditioning: a key event (only on change) + a chord-label event.
                  onset_q = e[1]
                  if onset_q != last_chord_label_qtime:
                      last_chord_label_qtime = onset_q
                      lbl = label_by_qtime.get(onset_q)
                      if lbl is None:
                          lbl = label_by_qtime.get(onset_q - 1) or label_by_qtime.get(onset_q + 1)
                      if lbl is not None:
                          root_pc, quality_id, key_pc, mode, function_id = lbl
                          chords_with_label += 1
                      else:
                          if is_key_tension:
                              root_pc, quality_id = root_position_chord(
                                  chord_group_pcs.get(onset_q, set()), chord_group_bass.get(onset_q))
                          else:
                              root_pc, quality_id = identify_chord(chord_group_pcs.get(onset_q, set()))
                          key_pc, mode, function_id = PC_UNKNOWN, MODE_UNKNOWN, FUNC_UNKNOWN
                          chords_derived += 1
                      # Key event only when the key actually changes (very sparse). Only the legacy
                      # German labels carry a key here: key_pc is PC_UNKNOWN for key+tension files,
                      # whose keys come from the KEY_CHANNEL pseudo-events above.
                      if key_pc != PC_UNKNOWN and (key_pc, mode) != current_key:
                          target_data.extend([0, key_pc, mode, 0, KEY_CHANNEL])
                          current_key = (key_pc, mode)
                          key_events += 1
                      # Chord-label event: [0, root_pc, quality_id, function_id, 120], pe NOT updated
                      target_data.extend([0, root_pc, quality_id, function_id, CHORD_LABEL_CHANNEL])
                      chord_label_events += 1

                  # Chord reference: [0, dur, pitch, vel, chan=4], pe NOT updated
                  dur = max(1, min(126, e[2]))
                  ptc = max(1, min(126, e[4]))
                  vel = max(1, min(126, e[5]))
                  target_data.extend([0, dur, ptc, vel, CHORDS_CHANNEL])
                  channel_5_chords += 1
                  # Track chord expiry: onset + duration (whichever note lasts longest).
                  # Onsets are in time2quant units (10 ms), durations in dur2quant units
                  # (20 ms): convert, or every chord switches off at half its length.
                  this_end = e[1] + 2 * e[2]
                  if this_end > chord_end_abs_time:
                      chord_end_abs_time = this_end
              else:
                  # Before writing the playable note, check if the chord has expired
                  if chord_end_abs_time >= 0 and e[1] >= chord_end_abs_time:
                      # Insert chord-off marker: [0, 0, 0, 0, CHORDS_CHANNEL]
                      # vel=0 distinguishes it from real chord notes in the dataset
                      target_data.extend([0, 0, 0, 0, CHORDS_CHANNEL])
                      chord_end_abs_time = -1

                  # Playable note: compute dtime from last playable note
                  time = max(0, min(126, e[1]-pe[1]))
                  dur = max(1, min(126, e[2]))
                  chan = max(0, min(15, e[3]))
                  ptc = max(1, min(126, e[4]))
                  vel = max(1, min(126, e[5]))

                  target_data.extend([time, dur, ptc, vel, chan])

                  total_notes += 1
                  if e[3] == MELODY_CHANNEL:
                      channel_0_notes += 1
                  elif e[3] == MELODY_CHORDS_CHANNEL:
                      channel_1_notes += 1
                  elif e[3] == ACCOMP_CHANNEL:
                      channel_10_notes += 1
                  elif e[3] == EXTRA_ACCOMP_CHANNEL:
                      channel_11_notes += 1

                  pe = e

          files_count += 1
          key_tension_files += is_key_tension

    except KeyboardInterrupt:
        print('Quitting...')
        break

    except Exception as ex:
        print(f'Bad MIDI: {f} - {ex}')
        continue


#===================================================================================================
# LEGACY-CONTENT GUARD
#===================================================================================================
def channel_census(data):
    """Events per pickle channel: every 5th token (index 4 of each event) is the channel."""
    return Counter(data[4::5])

census = {'train': channel_census(train_data1), 'test': channel_census(test_data1)}
print('=' * 70)
print('Pickle channel census (events per channel):')
for split, counts in census.items():
    print(f'  {split}: ' + ', '.join(f'chan {c}: {n}' for c, n in sorted(counts.items())))

if KEY_TENSION_ONLY:
    # The old tension (movement notes on chan 3, roman movement on chan 122) must be absent.
    # Chan 4 is the chord-tone (chroma) channel and is expected - it is the new aligned java chords.
    legacy_found = {split: {c: counts[c] for c in (HARMONY_CHANNEL, ROMAN_MOVE_CHANNEL) if counts[c]}
                    for split, counts in census.items()}
    if any(legacy_found.values()) or channel_4_movements or roman_move_events:
        raise SystemExit(f'ABORTED, nothing saved: legacy harmony events found in the pickle: {legacy_found}')
    print('Legacy-content guard OK: no chan 3 (movement) and no chan 122 (roman movement) events.')
print('=' * 70)

#===================================================================================================
# SAVE
#===================================================================================================
# Save training data
output_path = './Training-Data/' + output_name + '_train'
print('Saving training data...')
Any_Pickle_File_Writer(train_data1, output_path)
print(f'Training data saved to {output_path}.pickle')
print(f'{len(train_data1)} tokens ({len(train_data1)//5} notes)')
print('=' * 70)

# Save test data
if train_and_test_ratio < 1.:
    print('Saving test data...')
    output_path = './Training-Data/' + output_name + '_test'
    Any_Pickle_File_Writer(test_data1, output_path)
    print(f'Test data saved to {output_path}.pickle')
    print(f'{len(test_data1)} tokens ({len(test_data1)//5} notes)')
    print('=' * 70)

#===================================================================================================
# STATISTICS
#===================================================================================================
# Display channel statistics
print('Channel Statistics:')
print(f'Files written: {files_count} (key+tension format: {key_tension_files}); '
      f'skipped as non-key+tension: {skipped_legacy_files}')
print(f'Total notes processed: {total_notes}')
if total_notes > 0:
    for label, count in (('Channel 0 (MIDI 1, melody)', channel_0_notes),
                         ('Channel 1 (MIDI 2)', channel_1_notes),
                         ('Channel 10 (MIDI 11, accompaniment)', channel_10_notes),
                         ('Channel 11 (MIDI 12)', channel_11_notes)):
        print(f'{label} notes: {count} ({count / total_notes * 100:.2f}%)')
print(f'Legacy channel 3 harmony movement events inserted: {channel_4_movements}')
print(f'Chord-tone (chan 4) events inserted: {channel_5_chords}')
print(f'Chord-label events inserted: {chord_label_events} '
      f'(from text label: {chords_with_label}, notes-derived: {chords_derived})')
print(f'Key events inserted: {key_events}')
print(f'Legacy roman harmony-movement events inserted: {roman_move_events}')
print(f'Home-key events inserted: {home_key_events}; key changes flagged: {key_change_events}')
print(f'Tension events inserted: {tension_events} '
      f'(levels 0..{NUM_TENSION_LEVELS - 1}: {tension_histogram})')
if total_notes > 0:
    print(f'Movement events per note ratio: {channel_4_movements / total_notes:.4f}')
    print(f'Chord events per note ratio: {channel_5_chords / total_notes:.4f}')
    print(f'Chord-label events per note ratio: {chord_label_events / total_notes:.4f}')
print('=' * 70)

print('Done!')
