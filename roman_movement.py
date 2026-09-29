#===================================================================================================
# Monster Genie roman_movement.py
# Roman-numeral -> 5 harmony-movement reduction (for AE_style_roman).
#
# Implements the transition-based reduction specified in
#   docs/roman_numeral/"Functional-Harmony Reduction Map - Roman Numerals.md"
# and consumed by the FiLM conditioning in
#   docs/roman_numeral/film_conditioning_architecture.md
#
# Core principle: movement is a property of the TRANSITION (previous chord ->
# current chord, key context), not of a single chord. A static chord->class table
# only exists for the chord *function* layer (local_function); the final movement
# label is computed per chord-change event (movement / RomanMovementReducer).
#
# The >1000-token roman vocabulary collapses to 5 classes + NULL:
#   STABLE, TENSION, RESOLVE, MODULATE, COLOR  (+ NULL default).
# Priority when several rules fire: MODULATE > COLOR > RESOLVE > TENSION > STABLE.
#
# Copyright 2025 Alex Barrachina. Licensed under the Apache License, Version 2.0.
#===================================================================================================

import re
from typing import Dict, List, Optional, Tuple

from params import (
    RMOVE_STABLE, RMOVE_TENSION, RMOVE_RESOLVE, RMOVE_MODULATE, RMOVE_COLOR,
    RMOVE_NULL, PC_UNKNOWN, MODE_MAJOR, MODE_MINOR, note_name_to_pc,
)

# --------------------------------------------------------------------------- #
#  Token grammar (from the reduction map, section 2)
#      [b|#] NUMERAL [o|%|+] [(sus2)] [7|65|43|6|64|2] [/ target]
# --------------------------------------------------------------------------- #
_NUM = r'VII|VI|IV|V|III|II|I|vii|vi|iv|v|iii|ii|i'

PAT = re.compile(
    r'^(?P<acc>[b#]?)(?P<num>' + _NUM + r')'
    r'(?P<qual>[o%+]?)(?P<sus>\(sus2\))?'
    r'(?P<fig>65|43|64|7|6|2)?'
    r'(?:/(?P<ton>[b#]?(?:' + _NUM + r')))?$'
)

# Same token, un-anchored, to pull a bare roman out of an analyzer FUNC field.
_ROMAN_TOKEN = re.compile(
    r'([b#]?(?:' + _NUM + r')[o%+]?(?:\(sus2\))?(?:65|43|64|7|6|2)?'
    r'(?:/[b#]?(?:' + _NUM + r'))?)'
)

_NAME_TO_ID = {
    'STABLE': RMOVE_STABLE,
    'TENSION': RMOVE_TENSION,
    'RESOLVE': RMOVE_RESOLVE,
    'MODULATE': RMOVE_MODULATE,
    'COLOR': RMOVE_COLOR,
}


def roman_from_label(text: str) -> Optional[str]:
    """Extract the roman-numeral token from an analyzer harmony-label string.

    The German functional labels carried as MIDI text meta-events look like
        'E DOMINANT_SEVENTH_INCOMPLETE:QUARTSEXT(4,6):A DUR:D (V)'   -> 'V'
        'A MAJOR_TRIAD:ROOT(5):A DUR:T (I)'                          -> 'I'
        'G# DIMINISHED_TRIAD:ROOT(5):A DUR:VII'                      -> 'VII'
    Layout: '<root> <QUALITY>:<figbass>:<keyroot> <KEYMODE>:<FUNC> (<roman>)'.
    The roman appears either inside the FUNC field's parens (after a T/S/D
    symbol) or as a bare numeral. Returns None when there is no usable roman.
    """
    if not text or ':' not in text:
        return None
    parts = text.split(':')
    if len(parts) < 4:
        return None
    ff = parts[3].strip()
    if not ff or ff.lower() == 'null':
        return None
    # roman inside parens: 'D (V)' -> 'V', 'T (I)' -> 'I'
    if '(' in ff and ')' in ff:
        inside = ff[ff.index('(') + 1: ff.index(')')].strip()
        if inside:
            return inside
    # bare roman: 'VII', 'III', 'II'
    m = _ROMAN_TOKEN.match(ff)
    if m:
        return m.group(1)
    return None


def parse_roman(token: Optional[str]) -> Optional[Dict[str, Optional[str]]]:
    """Parse a roman-numeral token into its grammar fields, or None if it does
    not match. Fields: acc, num, qual, sus, fig, ton (missing groups are None/'')."""
    if not token:
        return None
    m = PAT.match(token.strip())
    if not m:
        return None
    return {
        'acc': m.group('acc') or '',
        'num': m.group('num'),
        'qual': m.group('qual') or '',
        'sus': m.group('sus'),          # '(sus2)' or None
        'fig': m.group('fig') or '',
        'ton': m.group('ton'),          # tonicization target or None
    }


_KEY_MARKER = re.compile(
    r'^\s*key\s*[:=]\s*(?P<root>[A-Ga-g][#b!\-]?)\s+(?P<mode>major|minor|maj|min|dur|moll)\b',
    re.IGNORECASE,
)


def key_from_marker(text: Optional[str]):
    """Parse a 'key: A minor' style marker -> (key_pc, mode) or None."""
    if not text:
        return None
    m = _KEY_MARKER.match(text)
    if not m:
        return None
    root = m.group('root').replace('!', 'b').replace('-', 'b')
    pc = note_name_to_pc(root)
    if pc == PC_UNKNOWN:
        return None
    md = m.group('mode').lower()
    mode = MODE_MINOR if md in ('minor', 'min', 'moll') else MODE_MAJOR
    return pc, mode


def parse_marker(text: Optional[str]):
    """Classify an 'all_roman' text marker into
        ('key', key_pc, mode) | ('roman', token) | None.
    Key markers take precedence; anything else matching the roman grammar is a
    roman chord label. Section names / junk markers return None (ignored)."""
    if not text:
        return None
    t = text.strip()
    k = key_from_marker(t)
    if k is not None:
        return ('key', k[0], k[1])
    if parse_roman(t) is not None:
        return ('roman', t)
    return None


def local_function(m: Dict[str, Optional[str]]) -> str:
    """Layer 1 - chord -> local function: 'T' | 'PD' | 'D' | 'COLOR'.

    Computed from the part BEFORE the slash (the local function); the slash
    target + key label are used separately to detect MODULATE boundaries.
    """
    # Inherent COLOR: chromatic root, augmented (+), or (sus2) — regardless of transition.
    if m['qual'] == '+' or m['sus'] or m['acc']:
        return 'COLOR'
    d = m['num'].upper()
    if d == 'I':
        return 'T'
    if d in ('III', 'VI'):
        return 'T'          # tonic substitutes
    if d in ('II', 'IV'):
        return 'PD'         # includes ii%
    if d in ('V', 'VII'):
        return 'D'          # includes viio (all figures)
    return 'STABLE'         # unreachable for grammar tokens; safe default


def movement(prev: Dict, cur: Dict, prev_key: Tuple[int, int], cur_key: Tuple[int, int],
             pf: Optional[str] = None, cf: Optional[str] = None) -> str:
    """Layer 2 - transition -> movement, per the reduction map (priority
    MODULATE > COLOR > RESOLVE > TENSION > STABLE).

    pf/cf are the effective local functions of prev/cur; pass them in to honour
    the cadential-6/4 override (I64 -> 'D'), otherwise they are computed here.
    """
    if pf is None:
        pf = local_function(prev)
    if cf is None:
        cf = local_function(cur)
    # 1. MODULATE: key change (both keys known), or a tonicization region
    #    starts/changes. Requiring both keys known avoids a spurious MODULATE when
    #    the key is first established mid-stream (unknown -> known is not a change).
    if (prev_key[0] != PC_UNKNOWN and cur_key[0] != PC_UNKNOWN
            and cur_key != prev_key):
        return 'MODULATE'
    if cur['ton'] and cur['ton'] != prev['ton']:
        return 'MODULATE'
    # 2. COLOR: inherently chromatic chord, chromatic tonicization target, deceptive resolution
    if cf == 'COLOR':
        return 'COLOR'
    if cur['ton'] and cur['ton'][0] in 'b#':
        return 'COLOR'
    if pf == 'D' and cur['num'].upper() == 'VI':
        return 'COLOR'      # deceptive V -> vi
    # 3. RESOLVE: tension -> tonic function, or exiting a tonicization onto home tonic
    if pf in ('D', 'PD') and cf == 'T':
        return 'RESOLVE'
    if prev['ton'] and not cur['ton'] and cf == 'T':
        return 'RESOLVE'
    # 4. TENSION: toward or prolonging the dominant / predominant motion
    if cf == 'D':
        return 'TENSION'
    if pf == 'T' and cf == 'PD':
        return 'TENSION'
    if pf == 'PD' and cf == 'PD':
        return 'TENSION'
    # 5. default
    return 'STABLE'


def _start_class(cf: str) -> str:
    """Sequence start (no previous chord): use the chord's inherent class."""
    if cf == 'COLOR':
        return 'COLOR'
    if cf == 'D':
        return 'TENSION'
    return 'STABLE'


class RomanMovementReducer:
    """Reduce an ordered chord sequence to per-chord 5-class movement ids.

    Stateless across calls; `reduce` takes the whole piece's ordered chord list
    so it can apply the one-chord cadential-6/4 lookahead. Input rows are
    (roman_str, key_pc, mode); output is a list of RMOVE_* ids, aligned 1:1.
    Unparseable / missing romans yield RMOVE_NULL and break the transition chain.
    """

    def reduce(self, chords: List[Tuple[Optional[str], int, int]]) -> List[int]:
        n = len(chords)
        parsed = [parse_roman(r) for (r, _k, _m) in chords]

        # Effective local functions, with the cadential-6/4 override: an I64/i64
        # immediately followed by a V-family (D-function) chord acts as function D.
        lf: List[Optional[str]] = [local_function(p) if p else None for p in parsed]
        for i in range(n):
            p = parsed[i]
            if p and p['num'].upper() == 'I' and p['fig'] == '64':
                nxt = parsed[i + 1] if i + 1 < n else None
                if nxt and local_function(nxt) == 'D':
                    lf[i] = 'D'

        out: List[int] = []
        prev_p: Optional[Dict] = None
        prev_pf: Optional[str] = None
        prev_roman: Optional[str] = None
        last_known_key: Tuple[int, int] = (PC_UNKNOWN, 0)
        prev_key: Optional[Tuple[int, int]] = None

        for i, (roman_str, key_pc, mode) in enumerate(chords):
            cur_p = parsed[i]
            if cur_p is None:
                out.append(RMOVE_NULL)
                prev_p = None
                prev_pf = None
                prev_roman = None
                prev_key = None
                continue

            # Carry the key forward across missing (unknown) annotations so a gap
            # in the key label does not fire a spurious MODULATE.
            cur_key = (key_pc, mode)
            if key_pc == PC_UNKNOWN:
                cur_key = last_known_key
            else:
                last_known_key = cur_key

            cf = lf[i]
            if prev_p is None or prev_key is None:
                name = _start_class(cf)
            elif roman_str == prev_roman and cur_key == prev_key:
                # Repeated identical label: prolongation -> STABLE, except D repeats.
                name = 'TENSION' if cf == 'D' else 'STABLE'
            else:
                name = movement(prev_p, cur_p, prev_key, cur_key, pf=prev_pf, cf=cf)

            out.append(_NAME_TO_ID[name])
            prev_p = cur_p
            prev_pf = cf
            prev_roman = roman_str
            prev_key = cur_key

        return out


def reduce_sequence(chords: List[Tuple[Optional[str], int, int]]) -> List[int]:
    """Convenience wrapper around RomanMovementReducer.reduce."""
    return RomanMovementReducer().reduce(chords)


# --------------------------------------------------------------------------- #
#  Validator (spec section 6 sanity gate): run over the sample label files and
#  print the movement-class histogram. Expect STABLE >> TENSION > RESOLVE >
#  MODULATE ~= COLOR.
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    import glob
    import os
    from collections import Counter

    from midiUtils import parse_harmony_label

    _ID_TO_NAME = {v: k for k, v in _NAME_TO_ID.items()}
    _ID_TO_NAME[RMOVE_NULL] = 'NULL'

    # 1) A couple of hand-checked micro-cases (roman_str, key_pc, mode).
    A = 9  # pitch class of A
    cases = {
        'V -> I  (authentic cadence)': ([('V', A, 0), ('I', A, 0)], RMOVE_RESOLVE),
        'I -> IV (predominant)':       ([('I', A, 0), ('IV', A, 0)], RMOVE_TENSION),
        'I -> V  (dominant)':          ([('I', A, 0), ('V', A, 0)], RMOVE_TENSION),
        'V -> VI (deceptive/color)':   ([('V', A, 0), ('VI', A, 0)], RMOVE_COLOR),
        'I -> vi (tonic subst.)':      ([('I', A, 0), ('VI', A, 0)], RMOVE_STABLE),
        'key change (DUR->MOLL)':      ([('I', A, 0), ('I', A, 1)], RMOVE_MODULATE),
    }
    print('=' * 70)
    print('Micro-cases (last-chord label vs expected):')
    ok = True
    for name, (seq, expected) in cases.items():
        got = reduce_sequence(seq)[-1]
        flag = 'ok ' if got == expected else 'FAIL'
        if got != expected:
            ok = False
        print(f'  [{flag}] {name:32s} -> {_ID_TO_NAME[got]:9s} (expected {_ID_TO_NAME[expected]})')
    print(f'micro-cases: {"ALL OK" if ok else "SOME FAILED"}')

    # 1b) Marker parsing ('all_roman' dialect: bare romans + 'key: X mode').
    assert parse_marker('key: A minor') == ('key', 9, 1), parse_marker('key: A minor')
    assert parse_marker('IV65/III')[0] == 'roman'
    assert parse_marker('ii%') == ('roman', 'ii%')
    assert parse_marker('V7/vi') == ('roman', 'V7/vi')
    assert parse_marker('Intro') is None
    # A short marker sequence: i (start) -> key set -> iv -> V7 -> i  in A minor.
    mk = [('i', PC_UNKNOWN, 2), ('iv', 9, 1), ('V7', 9, 1), ('i', 9, 1)]
    got = [_ID_TO_NAME[x] for x in reduce_sequence(mk)]
    print(f'marker parsing: OK   sample seq i/iv/V7/i -> {got}')

    # 2) Histogram over the repo sample label files (harm/*.txt), if present.
    hist = Counter()
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'harm', '*.txt')))
    for path in files:
        seq = []
        with open(path, 'r', encoding='latin-1', errors='ignore') as fh:
            for line in fh:
                # analyzer .txt rows: '<m>:<beat>: <harmony-label>...:<filename>'.
                # Strip the leading 'm:beat: ' and the trailing ':<filename>' so
                # what remains matches the MIDI text meta-event layout.
                body = line.strip()
                if ': ' not in body:
                    continue
                body = body.split(': ', 1)[1]        # drop 'm:beat: '
                parsed = parse_harmony_label(body)
                if parsed is None:
                    continue
                roman = roman_from_label(body)
                key_pc, mode = parsed[2], parsed[3]
                seq.append((roman, key_pc, mode))
        for mid in reduce_sequence(seq):
            hist[mid] += 1
    if files:
        total = sum(hist.values()) or 1
        print('=' * 70)
        print(f'Movement histogram over {len(files)} sample file(s) ({total} chords):')
        for mid in (RMOVE_STABLE, RMOVE_TENSION, RMOVE_RESOLVE, RMOVE_MODULATE, RMOVE_COLOR, RMOVE_NULL):
            c = hist.get(mid, 0)
            print(f'  {_ID_TO_NAME[mid]:9s} {c:6d}  ({100.0 * c / total:5.1f}%)')
    else:
        print('(no harm/*.txt sample files found for histogram)')
    print('=' * 70)
