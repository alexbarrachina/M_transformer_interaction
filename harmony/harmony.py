#===================================================================================================
# Monster Genie harmony.py Python module
# Real-time key and tension tracking: a live reference for the performer.
#
# Copyright 2026 Alex Barrachina
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
"""Real-time (causal) key and tension tracking for piano performance.

The tension labels in giantmidi_key_tension_{train,test}.pickle are computed offline, on whole
recorded pieces, by harmony_detector/build_key_tension_dataset.py. This module computes the same
quantities while the music is being played, from the notes heard so far, so that a performer
sees the current tension and key, on the same 0..4 scale the Monster Genie models are trained on.

Contents of this header
    1. Pipeline at a glance
    2. Input representation: onset-framed, duration-weighted chroma
    3. Key tracking: key profiles, diatonic collections, HMM / Viterbi, home key, stable key
    4. Chord estimation: pitch-class templates
    5. Tension: three indicators and their combination
    6. Offline (dataset) vs online (this module)
    7. Evaluation protocol and results
    8. Limitations and threats to validity
    9. Programming notes (API contract, units, invariants, how to re-tune)
   10. References


1. PIPELINE AT A GLANCE
=======================

    note_on / note_off (sounding pitch, time in ms)
        |
        +--> onset list --(every 32 onsets)--> 64-onset key window --> chroma (12)
        |        key profiles -> 12 collection scores -> + home prior -> Viterbi forward step
        |        -> local key (fast)  -> dwell filter (8 s) -> stable key (shown)
        |        -> home key (given, or estimated from the opening and locked after 60 s)
        |
        +--> every update (<= 100 ms): chroma of the last 500 ms
                 -> best chord template (root, quality)
                 -> tps   = Lerdahl distance: region(home -> local key) + chord(local tonic -> chord)
                 -> tonal = 1 - corr(chroma, Krumhansl-Kessler profile of the home key)
                 -> rough = interval-class roughness of the chroma
                 -> robust z-scores, clipped to +-3, averaged          = raw composite
                 -> trailing 1 s mean                                  = score
                 -> 4 cut points with hysteresis 0.15                  = level 0..4


2. INPUT REPRESENTATION
=======================
* A note is (onset_ms, offset_ms or still sounding, pitch, velocity). Only pitch class and
  sounding duration are used; velocity and register are ignored (tension here is harmonic,
  not dynamic). A note still held counts up to "now".
* Chroma (pitch-class profile, Fujishima 1999): a 12-vector where bin pc accumulates the time,
  in ms, that notes of pitch class pc sound inside an analysis span (duration overlap, so long
  notes weigh more than short ones).
* Two framings, on purpose:
  - KEY: framed by NOTE ONSETS, not seconds - frames of 8 onsets, windows of 8 frames (64
    onsets), hop 4 frames (32 onsets). The reliability of a chroma estimate depends on how many
    notes fed it, not on elapsed time, and onset framing is tempo-invariant (a window of N
    seconds holds far more notes in a fast passage than in a slow one; seconds-based windows
    tuned on one corpus did not transfer to another). This is the framing of the offline
    tracker (key_track.py), kept identical.
  - CHORD / TENSION: framed by TIME - the last WINDOW_MS = 500 ms of sound, recomputed every
    UPDATE_MS = 100 ms, because a chord is a short-time event.


3. KEY TRACKING
===============
3.1 Key profiles (key-finding by profile correlation, Krumhansl-Schmuckler; Krumhansl 1990).
    Each key is a 12-weight profile (the tonal hierarchy of its pitch classes) rotated to its
    tonic; a chroma "fits" a key by the Pearson correlation between chroma and profile.
    Five published profiles are used, with the values distributed by Humdrum's keycor (Sapp)
    and music21's key-weight classes (Cuthbert & Ariza 2010): Krumhansl-Kessler 1982 ("ks"),
    Aarden-Essen (Aarden 2003, "aarden"), Bellman-Budge (Bellman 2005, from Budge's 1943 chord
    statistics, "bellman"), Temperley-Kostka-Payne (Temperley 2007, "tkp") and Craig Sapp's
    simple weights ("simple"). Averaging several profiles reduces each one's known bias
    (e.g. tkp favours the relative major, aarden the subdominant). music21 notes that the
    Aarden-Essen minor weights are of uncertain origin.

3.2 State = diatonic COLLECTION, not key. The hidden state is one of 12 diatonic collections
    (the 7-note sets; C major and A minor share collection 0). Score of collection c for a window:
        score(c) = mean over profiles {aarden, bellman, tkp, simple} of
                   max( corr(chroma, major profile on c), corr(chroma, minor profile on c+9) )
    Reason (measured on 1523 filename-labelled GiantMIDI pieces, harmony_detector
    KEY_DETECTION_PLAN.md): almost all key-finding errors are picking the wrong 7-note set
    (collection right 81.6%); once the set is known, choosing major vs relative minor is right
    93-95% of the time. Keeping mode out of the state also means relative major/minor shifts
    can never produce a spurious "modulation".

3.3 Temporal model: a hidden Markov model over the 12 collections, decoded with Viterbi
    (Rabiner 1989; HMM key-finding as in Noland & Sandler 2006, Temperley 2007).
        emission(c)  = EMISSION_SCALE * (score(c) + HOME_PRIOR * [c == home collection])
                       (8 * correlation used as a pseudo log-likelihood; home prior 0.03)
        transition   = 0 to stay; -(3.0 + 0.25 * (d - 1)) to move d steps on the circle of fifths
        delta_t(c)   = emission_t(c) + max_c' [ delta_{t-1}(c') + transition(c', c) ]
    OFFLINE (the dataset) backtracks the full path (hindsight), merges segments shorter than 8 s
    and places each boundary on the chord onset where the key evidence flips. ONLINE (here) only
    the forward recursion exists: the current collection is argmax_c delta_t(c) (max-product
    filtering, no lookahead). The cost of a switch (3.0) is what prevents flicker: a new
    collection must out-score the old one over several windows before it wins.

3.4 Major vs relative minor of the current collection: majority vote of three profiles (ks,
    aarden, simple) on the chroma heard since that collection took over; the home collection
    always takes the home key itself.

3.5 Home key (the reference of every tension value):
    - GIVEN (recommended): the primer's 'home:' marker or the user's choice.
    - ESTIMATED: onset-weighted vote over the forward decisions with the first
      HOME_OPENING_WINDOWS windows weighted x3 (offline: the opening and closing 10% of the piece
      weigh x3 - online the end is unknown), then major/minor by the 3-profile vote plus one vote
      for the opening bass note (offline: the final bass note). Before the first full window a
      provisional estimate uses all notes heard (>= 16). The estimate locks after HOME_LOCK_MS.

3.6 Stable key: the real-time stand-in for the offline 8 s minimum segment. stable_key adopts
    the fast local key only after it has held for KEY_DWELL_MS = 8 s; a move to another
    collection is reported as a key change (key_changed / last_key_change), a relabel inside
    the same collection (relative major/minor) is not - as in the dataset's 'change key' markers.


4. CHORD ESTIMATION
===================
Template matching on the chroma (Fujishima 1999). Vocabulary: 12 roots x 8 qualities
(maj, min, dim, 7, maj7, m7, m7b5, dim7) as binary pitch-class templates. With x = chroma
normalised to sum 1, the chosen chord maximises
        sum of x over the chord's pitch classes  -  0.06 * (number of chord tones)
i.e. the share of the sound the chord explains, with a size penalty so a 4-note chord must
explain more than a triad. Ties (e.g. a bare dyad) go to the lowest root. The dataset used the
chords of an external offline analyzer instead; replacing those by templates of this kind left
94% of the dataset's tension levels unchanged, so the tension is robust to chord errors.


5. TENSION
==========
Theory. Lerdahl & Krumhansl (2007) model tonal tension as (a) hierarchical distance from the
tonic in Lerdahl's Tonal Pitch Space (Lerdahl 2001) plus (b) surface dissonance; listeners'
tension ratings follow this model closely. Here (a) is measured twice - once from the chord
and key labels, once directly from the notes - and (b) by a sensory-roughness index. Because
each indicator fails in a different way (labels vs notes; key-dependent vs key-independent),
their average is more robust than any one of them.

5.1 tps - Tonal Pitch Space distance (label-based; Lerdahl 2001).
    Basic space of a chord in a key, as 4 nested pitch-class sets:
        a = {root},  b = {root, fifth},  c = chord tones,  d = scale of the key + chord tones
    (scale: major, or harmonic minor; the "fifth" is the perfect, else diminished, else
    augmented fifth present). Distance from x to y:
        delta(x -> y) = i + j + k
        i = region distance: steps between the keys' collections on the circle of fifths,
            +1 if the mode differs (0 when the key is the same)
        j = chord distance: steps between the roots on the key's diatonic circle of fifths
            (I-V-ii-vi-iii-vii-IV), on the chromatic circle of fifths for roots outside the scale
        k = basic-space difference: pitch classes of y's space missing from x's, summed over levels
    Hierarchical (two-step) form used:
        tps = delta(I/home -> I/local key)  +  delta(I/local key -> chord/local key)
    Values in a major key: I 0, I(maj7) 1, IV 5, V 5, V7 6, vi 7, iii 7, ii 8, vii-dim 8,
    V/V 9, bII 13; region C -> G major adds 7, C -> A minor 9, C -> Db major 21.

5.2 tonal - distance of the sound from the home key (note-based):
        tonal = 1 - corr(chroma, Krumhansl-Kessler profile rotated to the home key)
    0 = the sound is the tonal hierarchy of home; ~1 = unrelated; up to 2 = anti-correlated.
    It needs no chord or local-key label (cf. "distance from key" in Herremans & Chew 2016).

5.3 rough - sensory roughness of the pitch-class content (note-based, key-independent):
        x = chroma / sum(chroma)
        rough = sum over pc pairs a<b of w(ic(a,b)) * x_a * x_b  /  (1 - sum x_pc^2)
    with interval-class weights w(ic1..ic6) = 1.0, 0.45, 0.15, 0.10, 0.05, 0.70 (minor
    seconds/major sevenths and tritones rough, thirds/sixths smooth). The denominator removes
    the effect of how many pitch classes sound. This is an octave-equivalent simplification of
    the roughness/consonance models of Plomp & Levelt (1965) and Hutchinson & Knopoff (1978);
    the weights are heuristic.

5.4 Combination.
        z_i   = clip( (indicator_i - median_i) / (IQR_i / 1.349), -3, +3 )   (robust z-score)
        raw   = mean(z_tps, z_tonal, z_rough)                                (equal weights)
        score = mean of raw over the last SMOOTH_MS = 1 s of updates with sound
        level = number of cut points below score, with hysteresis: a level only changes when
                score passes a cut point by HYSTERESIS = 0.15 (Schmitt-trigger quantisation)
    The cut points are the quintiles of the score over chords, so each level holds ~20% of the
    corpus: level 0 = most relaxed fifth of the music, 4 = most tense fifth. During silence the
    last level is held. A falling level is a resolution (the dataset has no "resolve" label).
    Offline validation of the composite (dataset builder, GiantMIDI pieces outside its
    calibration sample, 400-800 pieces per test): every indicator is lower on a piece's final
    chord (within-piece z -0.6..-0.7); dominant -> tonic in the
    local key: level falls 54%, stays 42%, rises 4%; tonic -> dominant: rises 45%, falls 6%;
    lower before phrase-end pauses; Cronbach's alpha 0.61 (0.64 smoothed) - moderate, as
    expected for indicators of two facets (distance vs dissonance: tps/tonal correlate 0.60,
    roughness with them ~0.2).


6. OFFLINE (DATASET) vs ONLINE (THIS MODULE)
============================================
    aspect             offline: build_key_tension_dataset.py      online: harmony.py
    -----------------  -----------------------------------------  -------------------------------------
    key decoding       Viterbi with backtracking (hindsight)      Viterbi forward pass, argmax delta_t
    min key segment    segments < 8 s merged into neighbours      stable_key: 8 s dwell before showing
    key boundary       moved to the best chord onset              when the forward pass switches
    home key           filename key if the music supports it,    given by the app, or estimated from
                       else whole-piece vote incl. the ending     the opening and locked after 60 s
    chords             external offline analyzer (java)           templates on the last 500 ms
    analysis span      each chord's span (>= 400 ms)              trailing 500 ms, every 100 ms
    smoothing          centred 1 s, duration-weighted             trailing 1 s mean
    hysteresis         0.08                                       0.15
    calibration        DATASET_CALIBRATION                        REALTIME_CALIBRATION (re-fitted)
    identical          onset framing, profiles, collection states, transition costs, home prior,
                       relative major/minor vote, the three indicators, robust z, 5 levels


7. EVALUATION PROTOCOL AND RESULTS
==================================
Protocol (re-runnable with replay_midi()):
* 500 random pieces of giantMIDI/all_key_tension (GiantMIDI-Piano, Kong et al. 2022, rebuilt at
  real speed with the offline labels as MIDI markers). Each piece is replayed note by note
  exactly as a live performance: note_on at each onset, note_off at each offset, update() every
  100 ms, using only the performance channels (never the offline chord tones).
* Split by file-name hash: 244 pieces to fit REALTIME_CALIBRATION and choose settings
  (window 500/1000/1500 ms, smoothing 0/0.5/1/2 s, hysteresis 0-0.25, key confirmation 0/1
  windows, dwell 0/4/8 s, home given/estimated); 256 held-out pieces for the numbers below.
* Tension: the live level read d ms after each offline chord onset vs the offline level of that
  chord - exact agreement, agreement within one level, Spearman rho between the live score and
  the offline level. Key: live key vs the offline key in force, every 100 ms; a modulation counts
  as detected when the live collection reaches the new offline collection within [-4 s, +60 s]
  (or before the next offline change).
Results on the held-out pieces (final module, home key given):
* Tension read 1 s after the chord onset: 62% exact, 94.5% within one level, rho 0.89
  (median within-piece rho 0.82). At the onset itself: 43% / 82% - the live tracker lags by
  about one analysis window plus half the smoothing (~1 s). ~30 level changes per minute;
  levels 19-21% each.
* Key: live key = offline key 68% of the time; 93% of offline modulations detected, median
  9 s after the offline boundary (p25 6 s, p75 13 s); 2.7 switches/min against 1.1 offline
  (hindsight removes short tonicisations). stable_key: 1.2 switches/min, ~16 s delay.
* Home key estimated instead of given: right 69% (right collection 80%); tension rho falls to
  0.69 - the home key is the single most important input.
* Cost: ~0.2 ms per update() (Apple M1, numpy); the 256 pieces (28 hours of music) replay in
  ~30 s on 9 processes.


8. LIMITATIONS AND THREATS TO VALIDITY
======================================
* Not a perceptual ground truth: the reference is the offline algorithm, itself a model
  (Lerdahl & Krumhansl's construct, not listener ratings of these performances). Agreement
  with it measures fidelity to the dataset labels, not musical truth.
* Latency is structural: a chord cannot be measured before it sounds (~1 s for tension,
  ~9 s for a modulation, ~16 s for the stable key).
* Relative to the home key: the same passage gets different levels under a different home
  key (e.g. Satie's Gymnopedie No. 1 read in G vs D major). The display must use the same home
  key the model is conditioned on.
* Tonal assumptions: major/minor diatonic collections only. Modal, whole-tone, octatonic or
  atonal music is forced onto the nearest collection; roughness still reacts sensibly.
* Pitch-class level only: no register, voicing, inversion, dynamics, rhythm or melodic
  expectancy (other tension components, e.g. Farbood 2012). Sustain pedal is not modelled:
  only key-down durations count.
* Short-window chord ambiguity (dyads, arpeggiated textures); the 0.06 size penalty and the
  roughness weights are heuristic.
* Input quality: GiantMIDI is automatically transcribed; live input is the model's own output.


9. PROGRAMMING NOTES (for people and LLMs extending this module)
================================================================
* Units: all times are milliseconds on ONE monotonic clock (e.g. time.perf_counter() * 1000).
  Pitches are MIDI numbers; only pitch % 12 matters.
* Keys: Key = (tonic_pc 0..11, 'M' | 'm'). Integer encoding pc + 12 * minor (0..23) with
  params.HOME_KEY_UNKNOWN = 24 for unknown, as in tension_joker.py and the pickle's chan 124;
  parse_key() also reads 'C major', 'F# minor', 'home: Bb major'. key_to_int() converts back.
* Feeding: call note_on/note_off with the SOUNDING pitch (the generated pitch, not the key the
  performer pressed); note_off ends the oldest open note of that pitch; note_on with velocity 0
  is a note_off. add_note() takes a note whose end is already known; prime() warms up with a
  primer and can shift it to end at "now" (end_at_ms) to keep one timeline.
* Reading: update(now_ms) is cheap and rate-limited (recomputes at most every UPDATE_MS); call
  it from the UI loop. It returns a HarmonyState: level (None until ready), score, raw_score,
  indicators {tps, tonal, rough}, chord (root_pc, quality), local_key (fast), stable_key
  (shown), home_key + home_source ('given' | 'estimated' | 'locked' | 'unknown'), key_changed
  (True once per stable-key change) and last_key_change (time_ms, from, to).
* Threads: every public method takes one re-entrant lock, so the MIDI callback thread and the
  UI thread can call concurrently. Do not hold the lock while doing slow work elsewhere.
* Do NOT feed the dataset's analyzer chord tones (MIDI channel 4 of all_key_tension files):
  only performed notes. read_midi_notes() already drops them for annotated files.
* Do NOT use the live level as the model's conditioning target: it lags ~1 s and is a
  measurement of the past. It is a reference display for the performer.
* Keep in sync with harmony_detector (copied constants): key_track.py (NOTES_PER_FRAME,
  WINDOW_FRAMES, HOP_FRAMES, EMISSION_SCALE, DISTANCE_PENALTY), key_detect.py (PROFILES,
  COLLECTION_PROFILES, ROTATION_PROFILES), build_key_tension_dataset.py (KEY_SWITCH_PENALTY,
  HOME_PRIOR, the TPS / roughness / correlation functions, DATASET_CALIBRATION).
* Re-tuning: changing WINDOW_MS or SMOOTH_MS changes the indicator distributions, so refit
  REALTIME_CALIBRATION the same way: replay held-in dataset pieces with smoothing and
  hysteresis off, take each indicator's median and IQR / 1.349 over updates with sound, then
  the 20/40/60/80% quantiles of the smoothed composite read 500 ms after each dataset chord
  onset. Then check agreement on held-out pieces (section 7).
* Complexity per update: O(notes in the last 500 ms) for the chroma, one 96 x 12 product for
  the chord templates, one profile correlation (tonal) and 66 pitch-class pairs (roughness);
  plus, every 32 onsets, one key window: 4 profiles x 24 key correlations and a 12 x 12
  max-plus Viterbi step. Finished notes are forgotten after HISTORY_MS.


10. REFERENCES
==============
Aarden, B. (2003). Dynamic melodic expectancy. PhD dissertation, Ohio State University.
Bellman, H. (2005). About the determination of key of a musical excerpt. In Computer Music
    Modeling and Retrieval (CMMR 2005), LNCS 3902, Springer.
Budge, H. (1943). A study of chord frequencies based on the music of representative composers
    of the eighteenth and nineteenth centuries. Teachers College, Columbia University.
Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests. Psychometrika, 16(3).
Cuthbert, M. S., & Ariza, C. (2010). music21: A toolkit for computer-aided musicology and
    symbolic music data. Proc. ISMIR 2010.
Farbood, M. M. (2012). A parametric, temporal model of musical tension. Music Perception, 29(4).
Fujishima, T. (1999). Realtime chord recognition of musical sound: A system using Common Lisp
    Music. Proc. ICMC 1999.
Herremans, D., & Chew, E. (2016). Tension ribbons: Quantifying and visualising tonal tension.
    Proc. TENOR 2016.
Hutchinson, W., & Knopoff, L. (1978). The acoustic component of Western consonance.
    Interface, 7(1).
Kong, Q., Li, B., Chen, J., & Wang, Y. (2022). GiantMIDI-Piano: A large-scale MIDI dataset for
    classical piano music. Transactions of the International Society for Music Information
    Retrieval, 5(1).
Krumhansl, C. L. (1990). Cognitive Foundations of Musical Pitch. Oxford University Press.
Krumhansl, C. L., & Kessler, E. J. (1982). Tracing the dynamic changes in perceived tonal
    organization in a spatial representation of musical keys. Psychological Review, 89(4).
Lerdahl, F. (2001). Tonal Pitch Space. Oxford University Press.
Lerdahl, F., & Krumhansl, C. L. (2007). Modeling tonal tension. Music Perception, 24(4).
Noland, K., & Sandler, M. (2006). Key estimation using a hidden Markov model. Proc. ISMIR 2006.
Plomp, R., & Levelt, W. J. M. (1965). Tonal consonance and critical bandwidth. Journal of the
    Acoustical Society of America, 38(4).
Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech
    recognition. Proceedings of the IEEE, 77(2).
Temperley, D. (2007). Music and Probability. MIT Press.
Sapp, C. S. "simple" key weights, as distributed with Humdrum's keycor tool
    (http://extras.humdrum.org/man/keycor/) and music21 (analysis.discrete.SimpleWeights).

Usage from an interaction_*.py application
------------------------------------------
    from harmony.harmony import HarmonyTracker
    harmony = HarmonyTracker(home_key=home_key)          # int 0..23 (pc + 12*minor), 'C major', or None
    harmony.prime(performance.pitches, performance.starts_ms, performance.durations_ms,
                  end_at_ms=time.perf_counter() * 1000)  # optional warm-up with the primer
    # MIDI callback thread, with the SOUNDING pitch (the generated one, not the key pressed):
    harmony.note_on(pitch, time.perf_counter() * 1000, velocity)
    harmony.note_off(pitch, time.perf_counter() * 1000)
    # main / visualizer loop (cheap: recomputes at most every UPDATE_MS):
    state = harmony.update(time.perf_counter() * 1000)
    visualizer.status_text = state.describe()   # 'Tension 3 [####.] | Key: G major (home C major) | Chord: D7'

`python -m harmony.harmony some.mid [--home 'C major' | --home none]` replays a MIDI file in simulated
real time and, for all_key_tension files, reports agreement with their offline labels.
"""

from __future__ import annotations

import bisect

import re
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from params import HOME_KEY_UNKNOWN, NUM_TENSION_LEVELS

Key = Tuple[int, str]                      # (tonic pitch class 0..11, 'M' | 'm')
KeyLike = Union[None, int, str, Key]

#===================================================================================================
# SETTINGS
#===================================================================================================
# --- copied from harmony_detector/key_track.py (keep in sync) -------------------------------------
NOTES_PER_FRAME = 8              # onsets per chroma frame
WINDOW_FRAMES = 8                # 64 onsets per key window
HOP_FRAMES = 4                   # a new key window every 32 onsets
EMISSION_SCALE = 8.0             # correlation -> pseudo log-likelihood
DISTANCE_PENALTY = 0.25          # extra cost per circle-of-fifths step beyond 1
# --- copied from harmony_detector/build_key_tension_dataset.py (keep in sync) ---------------------
KEY_SWITCH_PENALTY = 3.0         # base cost of leaving the current collection
HOME_PRIOR = 0.03                # bonus added to the home collection's window score
DATASET_HYSTERESIS = 0.08        # the dataset's level hysteresis (offline, centred smoothing)
# --- real-time only (chosen in the replay study, see the module docstring) -----------------------
WINDOW_MS = 500                  # sound analysed for the current chord / tension (1000 ms: same
                                 # agreement, 0.5 s more delay)
SMOOTH_MS = 1000                 # trailing mean of the composite (the dataset used a centred 1 s)
HYSTERESIS = 0.15                # wider than the dataset's 0.08: ~30 instead of ~38 level changes/min
UPDATE_MS = 100                  # recompute at most this often
CONFIRM_WINDOWS = 0              # extra key windows before the tracked key follows a new collection
                                 # (1: fewer switches but less agreement with the dataset)
KEY_DWELL_MS = 8_000             # stable_key moves only after the new key held this long
                                 # (the dataset's minimum key segment is 8 s)
HOME_OPENING_WINDOWS = 3         # opening windows weighted x3 in the home-key vote (offline: first 10%)
HOME_LOCK_MS = 60_000            # an estimated home key stops changing after this much music
HISTORY_MS = 10_000              # notes older than this (and finished) are forgotten

# The dataset's calibration (build_key_tension_dataset.DEFAULT_CALIBRATION), for reference.
DATASET_CALIBRATION: Dict[str, object] = {
    "center": {"tps": 11.0, "tonal": 0.7358, "rough": 0.1348},
    "scale": {"tps": 8.8955, "tonal": 0.4197, "rough": 0.0552},
    "thresholds": [-0.5324, -0.1477, 0.2011, 0.6229],
}
# Re-fitted for WINDOW_MS = 500 / SMOOTH_MS = 1000 on 244 all_key_tension pieces replayed through
# this tracker: indicator median / (IQR / 1.349), and cut points at the quintiles of the smoothed
# composite read 500 ms after each dataset chord onset (so each level holds ~20% of the chords).
REALTIME_CALIBRATION: Dict[str, object] = {
    "center": {"tps": 10.0, "tonal": 0.706, "rough": 0.1281},
    "scale": {"tps": 8.8955, "tonal": 0.4208, "rough": 0.0686},
    "thresholds": [-0.3588, -0.0340, 0.2789, 0.6856],
}

INDICATORS = ("tps", "tonal", "rough")

#===================================================================================================
# MUSIC CONSTANTS (copied from key_detect.py / build_key_tension_dataset.py)
#===================================================================================================
MAJOR_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
CHORD_ROOT_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]   # key-independent spelling
_PC_OF_LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# music21 key-weight profiles, as in key_detect.PROFILES
PROFILES: Dict[str, Dict[str, List[float]]] = {
    "ks": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    },
    "aarden": {
        "major": [17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587,
                  0.291248, 22.062, 0.145624, 8.15494, 0.232998, 4.95122],
        "minor": [18.2648, 0.737619, 14.0499, 16.8599, 0.702494, 14.4362,
                  0.702494, 18.6161, 4.56621, 1.93186, 7.37619, 1.75623],
    },
    "bellman": {
        "major": [16.80, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 20.28, 1.80, 8.04, 0.62, 10.57],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 21.07, 7.49, 1.53, 0.92, 10.21],
    },
    "tkp": {
        "major": [0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400],
        "minor": [0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330],
    },
    "simple": {
        "major": [2, 0, 1, 0, 1, 1, 0, 2, 0, 1, 0, 1],
        "minor": [2, 0, 1, 1, 0, 1, 0, 2, 1, 0, 0.5, 0.5],
    },
}
COLLECTION_PROFILES = ("aarden", "bellman", "tkp", "simple")   # score the 12 collections
ROTATION_PROFILES = ("ks", "aarden", "simple")                 # relative major vs minor vote

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
HARMONIC_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 11)
# Sensory roughness per interval class (ic1..ic6): seconds and tritones rub, thirds/sixths barely.
INTERVAL_CLASS_ROUGHNESS = (0.0, 1.0, 0.45, 0.15, 0.1, 0.05, 0.7)

# Chord templates for the live chord: (quality name, intervals above the root).
CHORD_QUALITIES: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("", (0, 4, 7)), ("m", (0, 3, 7)), ("dim", (0, 3, 6)), ("7", (0, 4, 7, 10)),
    ("maj7", (0, 4, 7, 11)), ("m7", (0, 3, 7, 10)), ("m7b5", (0, 3, 6, 10)), ("dim7", (0, 3, 6, 9)),
)
CHORD_SIZE_PENALTY = 0.06        # per chord tone: a larger chord must explain more of the sound


#===================================================================================================
# KEY HELPERS
#===================================================================================================
def key_label(key: Optional[Key]) -> str:
    if key is None:
        return "unknown"
    names = MAJOR_NAMES if key[1] == "M" else MINOR_NAMES
    return f"{names[key[0]]} {'major' if key[1] == 'M' else 'minor'}"


def key_to_int(key: Optional[Key]) -> int:
    """(pc, mode) -> pc + 12 * minor, the home-key encoding of params / tension_joker."""
    return HOME_KEY_UNKNOWN if key is None else key[0] + (12 if key[1] == "m" else 0)


def parse_key(value: KeyLike) -> Optional[Key]:
    """Accepts None, an int 0..23 (pc + 12*minor; HOME_KEY_UNKNOWN = unknown), a (pc, 'M'|'m')
    tuple, or text such as 'C major', 'F# minor', 'home: Bb major', 'key: c# minor'."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        if int(value) == HOME_KEY_UNKNOWN:
            return None
        if not 0 <= int(value) < 24:
            raise ValueError(f"Key index must be 0..23, got {value}")
        return int(value) % 12, ("m" if int(value) >= 12 else "M")
    if isinstance(value, tuple):
        return int(value[0]) % 12, ("m" if str(value[1]) in ("m", "min", "minor", "Minor") else "M")
    text = re.sub(r"^\s*(home|key)\s*:\s*", "", str(value), flags=re.IGNORECASE).strip()
    match = re.match(r"^([A-Ga-g])([#b\-]*)\s*((?i:major|minor|maj|min)|M|m)?$", text)
    if not match:
        raise ValueError(f"Cannot read key {value!r}; expected e.g. 'C major' or 'F# minor'")
    pc = _PC_OF_LETTER[match.group(1).upper()]
    for accidental in match.group(2):
        pc += 1 if accidental == "#" else -1
    word = match.group(3)
    if word is None:
        minor = match.group(1).islower()            # romantext style: 'c#' = C# minor
    else:
        minor = word == "m" or word.lower() in ("min", "minor")
    return pc % 12, ("m" if minor else "M")


def key_collection(key: Key) -> int:
    """Relative-major pc of the key's diatonic set (C major and A minor -> 0)."""
    return key[0] if key[1] == "M" else (key[0] + 3) % 12


def collection_candidates(collection: int) -> List[Key]:
    """The relative pair sharing a diatonic collection: (C, 'M') and (A, 'm')."""
    return [(collection, "M"), ((collection + 9) % 12, "m")]


def fifths_steps(a: int, b: int) -> int:
    """Steps between two pitch classes (or collections) on the circle of fifths, 0..6."""
    n = ((b - a) * 7) % 12
    return min(n, 12 - n)


def _profile_matrix(name: str) -> np.ndarray:
    """24 x 12: rows 0..11 = major keys on each tonic, 12..23 = minor keys; each row centred
    and unit-norm, so a dot product with a centred chroma gives the Pearson correlation."""
    rows = []
    for mode in ("major", "minor"):
        base = PROFILES[name][mode]
        for tonic in range(12):
            rows.append([base[(index - tonic) % 12] for index in range(12)])
    matrix = np.array(rows, dtype=float)
    matrix -= matrix.mean(axis=1, keepdims=True)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


_PROFILE_ROWS = {name: _profile_matrix(name) for name in PROFILES}
_MAJOR_ROW = np.arange(12)                       # collection c -> its major key row
_MINOR_ROW = 12 + (np.arange(12) + 9) % 12       # collection c -> its relative minor row
_TRANSITIONS = np.array([[0.0 if a == b else -(KEY_SWITCH_PENALTY + DISTANCE_PENALTY * (fifths_steps(a, b) - 1))
                          for b in range(12)] for a in range(12)])


def _correlations(chroma: np.ndarray, profile: str) -> np.ndarray:
    """Pearson correlation of a chroma with the 24 keys of one profile (0 for a flat chroma)."""
    centred = chroma - chroma.mean()
    norm = float(np.linalg.norm(centred))
    if norm == 0.0:
        return np.zeros(24)
    return _PROFILE_ROWS[profile] @ centred / norm


def score_collections(chroma: np.ndarray) -> np.ndarray:
    """key_detect.score_collections: per collection, the profile-averaged best correlation of
    its two keys (relative major and minor)."""
    total = np.zeros(12)
    for profile in COLLECTION_PROFILES:
        corr = _correlations(chroma, profile)
        total += np.maximum(corr[_MAJOR_ROW], corr[_MINOR_ROW])
    return total / len(COLLECTION_PROFILES)


def rotation_vote(chroma: np.ndarray, collection: int, extra_vote: Optional[int] = None) -> Key:
    """key_track.segment_rotation: relative major vs minor by a 3-profile vote (plus an optional
    extra vote for the candidate whose tonic is `extra_vote`). Ties go to the major key."""
    candidates = collection_candidates(collection)
    votes = [0, 0]
    for profile in ROTATION_PROFILES:
        corr = _correlations(chroma, profile)
        votes[0 if corr[_MAJOR_ROW[collection]] >= corr[_MINOR_ROW[collection]] else 1] += 1
    if extra_vote is not None:
        for index, candidate in enumerate(candidates):
            if candidate[0] == extra_vote:
                votes[index] += 1
    return candidates[0] if votes[0] >= votes[1] else candidates[1]


#===================================================================================================
# TENSION INDICATORS (copied from build_key_tension_dataset.py)
#===================================================================================================
KK_MAJOR = np.array(PROFILES["ks"]["major"])
KK_MINOR = np.array(PROFILES["ks"]["minor"])


def scale_of(key: Key) -> List[int]:
    return [(key[0] + d) % 12 for d in (MAJOR_SCALE if key[1] == "M" else HARMONIC_MINOR_SCALE)]


def tonic_triad(key: Key) -> frozenset:
    third = 4 if key[1] == "M" else 3
    return frozenset({key[0], (key[0] + third) % 12, (key[0] + 7) % 12})


def basic_space(root: int, pcs: frozenset, key: Key) -> Tuple[set, set, set, set]:
    """Lerdahl's basic space: root / root+fifth / chord / diatonic (+ chromatic chord tones)."""
    fifth = next(((root + iv) % 12 for iv in (7, 6, 8) if (root + iv) % 12 in pcs), None)
    return ({root}, {root} | ({fifth} if fifth is not None else set()), set(pcs), set(scale_of(key)) | set(pcs))


def basic_space_distance(x, y) -> int:
    """k: pitch classes new to y's basic space, counted level by level."""
    return sum(len(level_y - level_x) for level_x, level_y in zip(x, y))


def chord_root_distance(key: Key, root: int) -> int:
    """j: steps on the key's diatonic circle of fifths, chromatic circle for foreign roots."""
    scale = scale_of(key)
    if root in scale:
        position = (2 * scale.index(root)) % 7
        return min(position, 7 - position)
    return fifths_steps(key[0], root)


def region_distance(home: Key, local: Key) -> int:
    """delta(I/home -> I/local); i = collection steps + 1 for a mode change."""
    if home == local:
        return 0
    i = fifths_steps(key_collection(home), key_collection(local)) + (1 if home[1] != local[1] else 0)
    j = fifths_steps(home[0], local[0])
    k = basic_space_distance(basic_space(home[0], tonic_triad(home), home),
                             basic_space(local[0], tonic_triad(local), local))
    return i + j + k


def chord_distance(local: Key, root: int, pcs: frozenset) -> int:
    """delta(I/local -> chord/local)."""
    j = chord_root_distance(local, root)
    k = basic_space_distance(basic_space(local[0], tonic_triad(local), local), basic_space(root, pcs, local))
    return j + k


def ks_correlation(chroma: np.ndarray, key: Key) -> float:
    profile = np.roll(KK_MAJOR if key[1] == "M" else KK_MINOR, key[0])
    if chroma.sum() <= 0 or chroma.std() == 0:
        return 0.0
    return float(np.corrcoef(chroma, profile)[0, 1])


def roughness(chroma: np.ndarray) -> float:
    total = chroma.sum()
    if total <= 0:
        return 0.0
    x = chroma / total
    value = 0.0
    for a in range(12):
        for b in range(a + 1, 12):
            value += INTERVAL_CLASS_ROUGHNESS[min(b - a, 12 - b + a)] * x[a] * x[b]
    pairs = 1.0 - float((x ** 2).sum())        # normalise away how many pcs sound
    return value / pairs if pairs > 1e-9 else 0.0


_CHORDS = [(root, name, frozenset((root + iv) % 12 for iv in intervals))
           for root in range(12) for name, intervals in CHORD_QUALITIES]
_CHORD_MATRIX = np.array([[1.0 if pc in pcs else 0.0 for pc in range(12)] for _, _, pcs in _CHORDS])
_CHORD_PENALTY = CHORD_SIZE_PENALTY * _CHORD_MATRIX.sum(axis=1)


def best_chord(chroma: np.ndarray) -> Optional[Tuple[int, str, frozenset]]:
    """Template chord explaining most of the sounding pitch-class mass (minus a size penalty)."""
    total = chroma.sum()
    if total <= 0:
        return None
    index = int(np.argmax(_CHORD_MATRIX @ (chroma / total) - _CHORD_PENALTY))
    return _CHORDS[index]


def composite_score(indicators: Dict[str, float], calibration: Dict[str, object]) -> float:
    """Mean of the indicators' robust z-scores, each clipped to +-3 (as in the dataset)."""
    center, scale = calibration["center"], calibration["scale"]
    return float(np.mean([np.clip((indicators[name] - center[name]) / scale[name], -3, 3) for name in INDICATORS]))


#===================================================================================================
# STATE
#===================================================================================================
@dataclass
class HarmonyState:
    time_ms: float = 0.0
    ready: bool = False                    # a home key is known and something has sounded
    sounding: bool = False                 # notes sounded inside the analysis window
    level: Optional[int] = None            # tension 0 (relaxed) .. 4 (tense), dataset scale
    score: float = 0.0                     # smoothed composite (the value cut into levels)
    raw_score: float = 0.0                 # composite of the current window, before smoothing
    indicators: Dict[str, float] = field(default_factory=dict)
    chord: Optional[Tuple[int, str]] = None          # (root pc, quality name)
    local_key: Optional[Key] = None        # fast key, used for the tension (follows the dataset best)
    stable_key: Optional[Key] = None       # key to show: moves after KEY_DWELL_MS (fewer, later changes)
    home_key: Optional[Key] = None
    home_source: str = "unknown"           # 'given' | 'estimated' | 'locked' | 'unknown'
    key_changed: bool = False              # stable_key moved to another scale since the previous update()
    last_key_change: Optional[Tuple[float, Key, Key]] = None   # (time_ms, from, to) of stable_key

    @property
    def chord_label(self) -> str:
        if self.chord is None:
            return "-"
        return CHORD_ROOT_NAMES[self.chord[0]] + self.chord[1]

    def describe(self) -> str:
        """One-line status for a visualizer."""
        if not self.ready or self.level is None:
            return f"Tension: -  | Home: {key_label(self.home_key)}"
        bar = "#" * (self.level + 1) + "." * (NUM_TENSION_LEVELS - 1 - self.level)
        return (f"Tension {self.level} [{bar}] | Key: {key_label(self.stable_key)} "
                f"(home {key_label(self.home_key)}) | Chord: {self.chord_label}")


#===================================================================================================
# ONLINE KEY TRACKER
#===================================================================================================
class _OnlineKeyTracker:
    """key_track.py run forwards: windows of 64 onsets (hop 32), Viterbi forward scores, the
    current collection = best forward score (no hindsight)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.onsets: List[list] = []          # note records in onset order (shared with the tracker)
        self.next_window_end = WINDOW_FRAMES * NOTES_PER_FRAME
        self.delta: Optional[np.ndarray] = None      # Viterbi forward scores
        self.window_collections: List[int] = []      # forward decision per window
        self.window_onsets: List[int] = []
        self.candidate: Optional[int] = None
        self.candidate_count = 0
        self.collection: Optional[int] = None        # the collection currently reported
        self.segment_chroma = np.zeros(12)           # chroma since the current collection started
        self.total_chroma = np.zeros(12)
        self.first_bass: Optional[int] = None

    def add_onset(self, note: list) -> None:
        if not self.onsets:
            self.first_bass = note[2] % 12
        elif note[0] == self.onsets[0][0]:
            self.first_bass = min(self.first_bass, note[2] % 12) if self.first_bass is not None else note[2] % 12
        self.onsets.append(note)

    def windows_due(self) -> bool:
        return len(self.onsets) >= self.next_window_end

    def window_chroma(self, now: float) -> np.ndarray:
        """Duration-weighted chroma of the next window (held notes count up to now)."""
        start = self.next_window_end - WINDOW_FRAMES * NOTES_PER_FRAME
        chroma = np.zeros(12)
        for onset, end, pitch, _velocity in self.onsets[start:self.next_window_end]:
            chroma[pitch % 12] += max(0.0, (end if end is not None else now) - onset)
        return chroma

    def step(self, chroma: np.ndarray, home: Optional[Key]) -> None:
        """Score one window and advance the forward pass."""
        scores = score_collections(chroma)
        if home is not None:
            scores = scores.copy()
            scores[key_collection(home)] += HOME_PRIOR
        emissions = EMISSION_SCALE * scores
        if self.delta is None:
            self.delta = emissions
        else:
            self.delta = np.max(self.delta[:, None] + _TRANSITIONS, axis=0) + emissions
            self.delta -= self.delta.max()                 # keep the numbers small
        best = int(np.argmax(self.delta))
        self.window_collections.append(best)
        self.window_onsets.append(WINDOW_FRAMES * NOTES_PER_FRAME)
        self.total_chroma += chroma
        # Report a new collection only after it wins CONFIRM_WINDOWS more windows.
        if self.collection is None:
            self.collection = best
            self.segment_chroma = chroma.copy()
        elif best != self.collection:
            if best == self.candidate:
                self.candidate_count += 1
            else:
                self.candidate, self.candidate_count = best, 1
            if self.candidate_count > CONFIRM_WINDOWS:
                self.collection = best
                self.segment_chroma = chroma.copy()
                self.candidate, self.candidate_count = None, 0
        else:
            self.candidate, self.candidate_count = None, 0
            self.segment_chroma += chroma
        self.next_window_end += HOP_FRAMES * NOTES_PER_FRAME

    def estimate_home(self) -> Optional[Key]:
        """Home key from what has been heard: onset-weighted vote over the forward decisions with
        the opening up-weighted (offline: opening and closing 10%), then the relative major/minor
        by the 3-profile vote plus the opening bass note (offline: the final bass note)."""
        if self.window_collections:
            votes = np.zeros(12)
            for index, collection in enumerate(self.window_collections):
                votes[collection] += self.window_onsets[index] * (3.0 if index < HOME_OPENING_WINDOWS else 1.0)
            collection = int(np.argmax(votes))
            chroma = self.total_chroma
        elif len(self.onsets) >= NOTES_PER_FRAME * 2:
            # Before the first full window: score what has been heard so far.
            chroma = self.window_chroma_partial()
            collection = int(np.argmax(score_collections(chroma)))
        else:
            return None
        return rotation_vote(chroma, collection, extra_vote=self.first_bass)

    def window_chroma_partial(self) -> np.ndarray:
        chroma = np.zeros(12)
        last = self.onsets[-1][0]
        for onset, end, pitch, _velocity in self.onsets:
            chroma[pitch % 12] += max(1.0, (end if end is not None else last) - onset)
        return chroma


#===================================================================================================
# THE TRACKER
#===================================================================================================
class HarmonyTracker:
    """Streaming key + tension tracker. Feed note_on/note_off (or add_note), call update()."""

    def __init__(self, home_key: KeyLike = None, calibration: Optional[Dict[str, object]] = None,
                 window_ms: float = WINDOW_MS, smooth_ms: float = SMOOTH_MS, update_ms: float = UPDATE_MS,
                 hysteresis: float = HYSTERESIS) -> None:
        self.calibration = calibration if calibration is not None else REALTIME_CALIBRATION
        self.window_ms = window_ms
        self.smooth_ms = smooth_ms
        self.update_ms = update_ms
        self.hysteresis = hysteresis
        self._given_home = parse_key(home_key)
        self._lock = threading.RLock()
        self.reset()

    # ------------------------------------------------------------------ input
    def reset(self) -> None:
        """Forget everything heard (keeps the given home key)."""
        with self._lock:
            self._notes: List[list] = []                    # [onset, end|None, pitch, velocity]
            self._starts: List[float] = []
            self._open: Dict[int, Deque[list]] = {}
            self._keys = _OnlineKeyTracker()
            self._home: Optional[Key] = self._given_home
            self._home_source = "given" if self._given_home is not None else "unknown"
            self._first_onset: Optional[float] = None
            self._local: Optional[Key] = self._home          # before the first key window: home
            self._stable: Optional[Key] = self._home         # displayed key (dwell-filtered _local)
            self._candidate: Optional[Key] = self._home
            self._candidate_since = 0.0
            self._scores: Deque[Tuple[float, float]] = deque()
            self._level: Optional[int] = None
            self._last_update: Optional[float] = None
            self._last_change: Optional[Tuple[float, Key, Key]] = None
            self._changed_since_update = False
            self._state = HarmonyState(home_key=self._home, home_source=self._home_source)

    def set_home_key(self, home_key: KeyLike) -> None:
        """Fix the reference key (None: estimate it from the music). Takes effect immediately."""
        with self._lock:
            self._given_home = parse_key(home_key)
            self._home = self._given_home
            self._home_source = "given" if self._given_home is not None else "unknown"
            self._refresh_local_key()
            self._stable = self._candidate = self._local

    def note_on(self, pitch: int, time_ms: float, velocity: int = 64) -> None:
        if velocity <= 0:                                   # MIDI running-status note-off
            self.note_off(pitch, time_ms)
            return
        with self._lock:
            note = [float(time_ms), None, int(pitch), int(velocity)]
            self._open.setdefault(int(pitch), deque()).append(note)
            self._insert(note)

    def note_off(self, pitch: int, time_ms: float) -> None:
        """Ends the oldest sounding note with this pitch (unmatched note-offs are ignored)."""
        with self._lock:
            queue = self._open.get(int(pitch))
            if queue:
                note = queue.popleft()
                note[1] = max(note[0] + 1.0, float(time_ms))

    def add_note(self, pitch: int, start_ms: float, end_ms: float, velocity: int = 64) -> None:
        """A note whose end is already known (e.g. from a primer or a scheduled generated note)."""
        with self._lock:
            self._insert([float(start_ms), max(float(start_ms) + 1.0, float(end_ms)), int(pitch), int(velocity)])

    def prime(self, pitches: Sequence[int], starts_ms: Sequence[float], durations_ms: Sequence[float],
              velocities: Optional[Sequence[int]] = None, end_at_ms: Optional[float] = None) -> HarmonyState:
        """Warm up with a primer. With end_at_ms the primer is shifted so that it ends at that
        time (e.g. time.perf_counter()*1000 just before the live part), keeping one timeline."""
        if not len(pitches):
            return self.state
        shift = 0.0
        if end_at_ms is not None:
            shift = float(end_at_ms) - max(float(s) + float(d) for s, d in zip(starts_ms, durations_ms))
        order = sorted(range(len(pitches)), key=lambda index: starts_ms[index])
        with self._lock:
            for index in order:
                self.add_note(pitches[index], starts_ms[index] + shift, starts_ms[index] + durations_ms[index] + shift,
                              velocities[index] if velocities is not None else 64)
            end = max(float(starts_ms[i]) + float(durations_ms[i]) for i in order) + shift
            # Run the trailing smoother over the primer's last seconds so the level is settled.
            t = max(float(starts_ms[order[0]]) + shift, end - self.smooth_ms - self.window_ms)
            while t < end:
                self._compute(t)
                t += self.update_ms
            return self._compute(end)

    def _insert(self, note: list) -> None:
        if self._first_onset is None:
            self._first_onset = note[0]
        position = bisect.bisect_right(self._starts, note[0])
        self._starts.insert(position, note[0])
        self._notes.insert(position, note)
        self._keys.add_onset(note)

    # ------------------------------------------------------------------ output
    @property
    def state(self) -> HarmonyState:
        with self._lock:
            return self._state

    def update(self, time_ms: float) -> HarmonyState:
        """Returns the current state; recomputes at most every update_ms (cheap to call per frame)."""
        with self._lock:
            if self._last_update is not None and time_ms - self._last_update < self.update_ms:
                return self._state
            return self._compute(float(time_ms))

    def _compute(self, now: float) -> HarmonyState:
        self._last_update = now
        # 1) key windows that became complete
        while self._keys.windows_due():
            self._keys.step(self._keys.window_chroma(now), self._home)
            self._refresh_home(now)
            self._refresh_local_key()
        if self._home is None:
            self._refresh_home(now)
            self._refresh_local_key()
        self._refresh_stable_key(now)
        # 2) sound of the analysis window
        chroma = self._window_chroma(now - self.window_ms, now)
        sounding = chroma.sum() > 0
        state = HarmonyState(time_ms=now, sounding=sounding, home_key=self._home, home_source=self._home_source,
                             local_key=self._local, stable_key=self._stable, level=self._level,
                             last_key_change=self._last_change, key_changed=self._changed_since_update)
        self._changed_since_update = False
        if self._home is None or not sounding:
            # Silence or unknown home: keep the last level (nothing new to measure).
            state.ready = self._home is not None and self._level is not None
            state.score = self._state.score
            self._state = state
            return state
        local = self._local or self._home
        chord = best_chord(chroma)
        root, quality, pcs = chord
        indicators = {
            "tps": float(region_distance(self._home, local) + chord_distance(local, root, pcs)),
            "tonal": 1.0 - ks_correlation(chroma, self._home),
            "rough": roughness(chroma),
        }
        raw = composite_score(indicators, self.calibration)
        # 3) trailing smoothing + hysteresis quantisation
        self._scores.append((now, raw))
        while self._scores and self._scores[0][0] <= now - max(self.smooth_ms, 1.0):
            self._scores.popleft()
        smoothed = float(np.mean([value for _, value in self._scores]))
        self._level = self._quantise(smoothed)
        state.level, state.score, state.raw_score = self._level, smoothed, raw
        state.indicators, state.chord, state.local_key = indicators, (root, quality), local
        state.ready = True
        self._state = state
        self._forget(now)
        return state

    def _quantise(self, value: float) -> int:
        cuts = np.asarray(self.calibration["thresholds"], dtype=float)
        raw = int(np.searchsorted(cuts, value, side="right"))
        current = self._level
        if current is None or self.hysteresis <= 0:
            return raw
        if raw > current:
            return max(current, int(np.searchsorted(cuts + self.hysteresis, value, side="right")))
        if raw < current:
            return min(current, int(np.searchsorted(cuts - self.hysteresis, value, side="right")))
        return current

    def _refresh_home(self, now: float) -> None:
        if self._given_home is not None or self._home_source == "locked":
            return
        estimate = self._keys.estimate_home()
        if estimate is not None:
            self._home = estimate
            self._home_source = "estimated"
            if self._first_onset is not None and now - self._first_onset >= HOME_LOCK_MS and self._keys.window_collections:
                self._home_source = "locked"

    def _refresh_local_key(self) -> None:
        collection = self._keys.collection
        if collection is None or self._home is None or collection == key_collection(self._home):
            new_local = self._home                          # home collection -> the home key itself
        else:
            # Relative major/minor from everything heard since this collection took over.
            new_local = rotation_vote(self._keys.segment_chroma, collection)
        self._local = new_local

    def _refresh_stable_key(self, now: float) -> None:
        """stable_key = local key once it has held for KEY_DWELL_MS (the real-time stand-in for the
        dataset's 8 s minimum key segment). Moving to another scale is reported as a key change;
        a relative major/minor relabel of the same scale is not (as in the dataset)."""
        if self._local != self._candidate:
            self._candidate, self._candidate_since = self._local, now
        if self._candidate is None or self._candidate == self._stable:
            return
        if self._stable is None or key_collection(self._candidate) == key_collection(self._stable):
            self._stable = self._candidate                   # first key, or relabel within the scale
        elif now - self._candidate_since >= KEY_DWELL_MS:
            self._last_change = (now, self._stable, self._candidate)
            self._changed_since_update = True
            self._stable = self._candidate

    def _window_chroma(self, start: float, end: float) -> np.ndarray:
        chroma = np.zeros(12)
        low = bisect.bisect_left(self._starts, start - HISTORY_MS)
        high = bisect.bisect_right(self._starts, end)
        for onset, note_end, pitch, _velocity in self._notes[low:high]:
            stop = end if note_end is None else min(note_end, end)
            overlap = stop - max(onset, start)
            if overlap > 0:
                chroma[pitch % 12] += overlap
        return chroma

    def _forget(self, now: float) -> None:
        """Drop finished notes older than HISTORY_MS (the key tracker keeps its own list)."""
        cutoff = now - HISTORY_MS
        drop = 0
        while drop < len(self._notes) and self._notes[drop][0] < cutoff and self._notes[drop][1] is not None \
                and self._notes[drop][1] < cutoff:
            drop += 1
        if drop > 256:
            del self._notes[:drop]
            del self._starts[:drop]


#===================================================================================================
# REPLAY (testing / demo): play a MIDI file through the tracker in simulated real time
#===================================================================================================
def read_midi_notes(path: str) -> Tuple[List[Tuple[float, float, int, int]], Dict[str, list]]:
    """Notes (start_ms, end_ms, pitch, velocity) of a MIDI file, tempo-mapped to ms, and its
    key/tension markers. Files with 'home:'/'tension:' markers (giantMIDI/all_key_tension) keep only
    the performance channels 1, 2, 11, 12, never the analyzer's chord tones on channel 4."""
    import mido
    midi = mido.MidiFile(path)
    annotated = any(msg.type == "marker" and re.match(r"^(home|tension)\s*:", msg.text.strip(), re.IGNORECASE)
                    for track in midi.tracks for msg in track)
    notes: List[Tuple[float, float, int, int]] = []
    markers: Dict[str, list] = {"home": [], "key": [], "tension": [], "change": []}
    pending: Dict[Tuple[int, int], List[Tuple[float, int]]] = {}
    now_ms, tempo = 0.0, 500_000
    for msg in mido.merge_tracks(midi.tracks):
        now_ms += mido.tick2second(msg.time, midi.ticks_per_beat, tempo) * 1000.0
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "marker":
            text = msg.text.strip()
            for name, prefix in (("home", "home:"), ("key", "key:"), ("change", "change key:"), ("tension", "tension:")):
                if text.lower().startswith(prefix):
                    value = text[len(prefix):].strip()
                    if name == "tension":
                        markers[name].append((now_ms, int(value)))
                    elif name == "change":
                        markers[name].append((now_ms, value))
                    else:
                        markers[name].append((now_ms, parse_key(value)))
        elif msg.type in ("note_on", "note_off"):
            if msg.channel == 9 or (annotated and msg.channel not in (0, 1, 10, 11)):
                continue
            key = (msg.channel, msg.note)
            if msg.type == "note_on" and msg.velocity > 0:
                pending.setdefault(key, []).append((now_ms, msg.velocity))
            elif pending.get(key):
                start, velocity = pending[key].pop(0)
                notes.append((start, max(start + 1.0, now_ms), msg.note, velocity))
    for (channel, pitch), starts in pending.items():
        for start, velocity in starts:
            notes.append((start, max(start + 1.0, now_ms), pitch, velocity))
    notes.sort()
    return notes, markers


def replay_midi(path: str, tracker: Optional[HarmonyTracker] = None, home_key: KeyLike = "file",
                tick_ms: float = UPDATE_MS) -> Tuple[List[HarmonyState], Dict[str, list]]:
    """Feeds the notes as note_on/note_off in time order and calls update() every tick_ms, exactly
    as a live application would. home_key='file' uses the file's 'home:' marker if it has one."""
    notes, markers = read_midi_notes(path)
    if tracker is None:
        if home_key == "file":
            home_key = markers["home"][0][1] if markers["home"] else None
        tracker = HarmonyTracker(home_key=home_key, update_ms=tick_ms)
    events = sorted([(start, 1, pitch, velocity) for start, _, pitch, velocity in notes] +
                    [(end, 0, pitch, 0) for _, end, pitch, _ in notes], key=lambda event: (event[0], event[1]))
    states: List[HarmonyState] = []
    if not events:
        return states, markers
    position, now = 0, events[0][0]
    while now <= events[-1][0] + tick_ms:
        while position < len(events) and events[position][0] <= now:
            time_ms, is_on, pitch, velocity = events[position]
            if is_on:
                tracker.note_on(pitch, time_ms, velocity)
            else:
                tracker.note_off(pitch, time_ms)
            position += 1
        states.append(tracker.update(now))
        now += tick_ms
    return states, markers


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Replay a MIDI file through the real-time key/tension tracker.")
    parser.add_argument("midi", help="MIDI file (an all_key_tension file also reports agreement with its labels)")
    parser.add_argument("--home", default="file",
                        help="Home key ('C major', 'F# minor'), 'file' = the file's home: marker, 'none' = estimate")
    parser.add_argument("--every", type=float, default=5.0, help="Print one line every N seconds. Default: %(default)s")
    args = parser.parse_args(argv)
    home: KeyLike = None if args.home.lower() == "none" else args.home
    states, markers = replay_midi(args.midi, home_key=home)
    if not states:
        print("No notes.")
        return 1
    next_print = states[0].time_ms
    for state in states:
        if state.time_ms >= next_print:
            print(f"{(state.time_ms - states[0].time_ms) / 1000:7.1f}s  {state.describe()}")
            next_print += args.every * 1000
    changes = [s.last_key_change for s in states if s.key_changed]
    print(f"\nKey changes reported live: {len(changes)}; home key: {key_label(states[-1].home_key)} ({states[-1].home_source})")
    if markers["tension"]:
        times = np.array([s.time_ms for s in states])
        live = np.array([-1 if s.level is None else s.level for s in states])
        for delay in (0, 500, 1000):
            index = np.clip(np.searchsorted(times, [t + delay for t, _ in markers["tension"]], side="right") - 1, 0, len(states) - 1)
            got, want = live[index], np.array([level for _, level in markers["tension"]])
            valid = got >= 0
            print(f"Agreement with the file's offline tension labels, read {delay} ms after each chord: "
                  f"exact {np.mean(got[valid] == want[valid]):.0%}, within one level {np.mean(np.abs(got[valid] - want[valid]) <= 1):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
