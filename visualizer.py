#===================================================================================================
# Monster Genie visualizer.py Python module
# Real-time visualization of piano rolls
# 
# Copyright 2025 Alex Barrachina
#
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


import pygame
import threading
import time
import random
import math
from collections import deque
from typing import List, Tuple, Optional, Dict, Any, Union, Callable, Deque

# Tonnetz geometry mirrors tonnetz.py, kept local here to avoid importing the
# matplotlib animation module in every pygame visualizer user.
_SQRT3 = math.sqrt(3.0)
PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
TONNETZ_COORDS = [
    (0, 0), (-1, 2), (2, 0), (1, -1), (0, 1), (-1, 0),
    (2, -2), (1, 0), (0, -1), (-1, 1), (-2, 0), (1, 1),
]

# Detailed tonal-tension components (tension_extractor.py TF_* fields) shown as
# mini text meters in the harmony panel when provided via set_harmony_state's
# tension_detail dict. (key, label, min, max) — min/max only set the meter's
# fill range, raw values are still printed in full.
_TENSION_DETAIL_SPECS: List[Tuple[str, str, float, float]] = [
    ('key_conf', 'KeyConf', 0.0, 1.0),
    ('key_entropy', 'KeyEntr', 0.0, 1.0),
    ('harmonic_change', 'HarmChg', 0.0, 2.0),
    ('dissonance', 'Disson', 0.0, 1.0),
    ('dist_key', 'DistKey', 0.0, math.pi),
    ('dist_tonic', 'DistTon', 0.0, math.pi),
    ('dist_subdom', 'DistSub', 0.0, math.pi),
    ('dist_dom', 'DistDom', 0.0, math.pi),
    ('mod_pressure', 'ModPres', 0.0, 2.0),
    ('tension_slope', 'TensSlp', -2.0, 2.0),
    ('resolution', 'Resolv', -2.0, 2.0),
    # Directional key distances (tension_extractor neighbor_distances /
    # fifths_signature): posterior pull toward each ring-1 neighbor of the
    # current key, plus the profile-free signature-of-fifths axes
    # (FifthSh > 0 = drifting dominant-ward, ModeAx > 0 = leaning minor).
    ('p_relative', 'K>Rel', 0.0, 1.0),
    ('p_parallel', 'K>Par', 0.0, 1.0),
    ('p_dominant', 'K>Dom', 0.0, 1.0),
    ('p_subdominant', 'K>Sub', 0.0, 1.0),
    ('fifths_shift', 'FifthSh', -3.0, 3.0),
    ('mode_axis', 'ModeAx', -3.0, 3.0),
]


def lattice_to_display(x: float, y: float) -> Tuple[float, float]:
    return (x + 0.5 * y, -(_SQRT3 / 2.0) * y)


def get_pitch_at_tonnetz_coord(x: int, y: int, key_root: int) -> str:
    semitone = (7 * x + 4 * y) % 12
    return PITCH_NAMES[(key_root + semitone) % 12]

# Note: For strict TorchScript compatibility, avoid dynamic typing and use explicit types.

class Visualizer:

    def __init__(
        self,
        width: int = 1200,
        height: int = 940,
        roll_height: int = 180,
        fps: int = 30,
        button_slots: int = 12,
        harmony_panel_width: int = 0,
        harmony_grid_range: Tuple[int, int] = (-3, 3),
        harmony_trail_length: int = 50,
    ) -> None:
        pygame.init()
        self.width = width
        self.roll_height = roll_height
        self.ref_height = roll_height
        self.button_height = roll_height
        self.pitch_height = roll_height * 2  # Double height for pitches
        self.joker_height = 30  # Thin strip between pitches and buttons
        self.height = self.ref_height + 10 + self.pitch_height + 10 + self.joker_height + 10 + self.button_height + 20
        self.fps = fps
        self.button_slots = button_slots
        self.harmony_panel_width = max(0, min(int(harmony_panel_width), self.width - 220))
        self.harmony_panel_gap = 10 if self.harmony_panel_width > 0 else 0
        self.harmony_grid_min, self.harmony_grid_max = harmony_grid_range
        self.harmony_trail: Deque[Tuple[float, float]] = deque(maxlen=max(1, int(harmony_trail_length)))
        self.harmony_key_root: int = 0
        self.harmony_key_mode: str = 'major'
        self.harmony_centroid: Tuple[float, float] = (0.0, 0.0)
        self.harmony_magnitude: float = 0.0
        self.harmony_active_pcs: List[int] = []
        self.harmony_target_pcs: List[int] = []
        self.harmony_movement_label: str = ''
        self.harmony_chord_label: str = ''
        self.harmony_tension_detail: Dict[str, float] = {}
        self.harmony_panel_active: bool = False
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Real-Time Piano Roll Visualizer')
        self.clock = pygame.time.Clock()
        self.running = True
        self.notes: List[Dict[str, Any]] = []
        self.buttons: List[Dict[str, Any]] = []
        self.jokers: List[Dict[str, Any]] = []
        self.chord_pcs: List[int] = []        # pitch classes of the current predicted chord (white overlay)
        self.chord_active: bool = False       # draw the chord overlay only while a movement is active
        self.primer_pitches: List[int] = []
        self.primer_dtimes: List[int] = []
        self.primer_durs: Optional[List[int]] = None
        self.primer_buttons: Optional[List[int]] = None
        self.primer_colors: List[Tuple[int, int, int]] = []
        self.primer_playback_elapsed: Optional[float] = None  # dtime units during primer playback
        self.primer_playback_total: float = 1.0
        self.primer_playback_ms_per_unit: float = 32.0
        self.primer_playback_speed: float = 1.0  # >1 shortens wall-clock waits during play_primer
        self._primer_font = pygame.font.Font(None, 22)
        self._harmony_font_small = pygame.font.SysFont('monospace', 14)
        self._harmony_font = pygame.font.SysFont('monospace', 16)
        self._harmony_font_title = pygame.font.SysFont('monospace', 20, bold=True)
        self.time_offset: float = 0.0
        self.scroll_speed: float = 100.0  # pixels per second
        self.last_draw_time: float = time.time()
        self.start_time: float = time.time()
        self.velocity_colors = [
            (60, 60, 60),    # 0-19: dark gray
            (60, 60, 200),   # 20-39: blue
            (60, 200, 60),   # 40-59: green
            (200, 200, 60),  # 60-79: yellow
            (200, 120, 60),  # 80-99: orange
            (200, 60, 60),   # 100-119: red
            (255, 0, 255),   # 120-127: magenta
        ]

    def primer(self, pitches: Union[List[int], Any], dtimes: Union[List[int], Any],
               buttons: Optional[Union[List[int], Any]] = None,
               durs: Optional[Union[List[int], Any]] = None) -> None:
        # Accepts lists or tensors for pitches, dtimes, and optionally buttons/durs
        if hasattr(pitches, 'tolist'):
            self.primer_pitches = pitches.tolist()
        else:
            self.primer_pitches = list(pitches)
        if hasattr(dtimes, 'tolist'):
            self.primer_dtimes = dtimes.tolist()
        else:
            self.primer_dtimes = list(dtimes)
        if buttons is not None:
            if hasattr(buttons, 'tolist'):
                self.primer_buttons = buttons.tolist()
            else:
                self.primer_buttons = list(buttons)
        else:
            self.primer_buttons = None
        if durs is not None:
            if hasattr(durs, 'tolist'):
                self.primer_durs = durs.tolist()
            else:
                self.primer_durs = list(durs)
        else:
            self.primer_durs = None
        # Assign a random color for each note in the primer
        self.primer_colors = [self._random_color() for _ in self.primer_pitches]

    def _primer_timeline_total(self, dtimes: Optional[List[int]] = None) -> float:
        """Total length in dtime units for a segment (matches primer roll time axis)."""
        dt = dtimes if dtimes is not None else self.primer_dtimes
        if not dt:
            return 1.0
        times: List[float] = [0.0]
        for d in dt[:-1]:
            times.append(times[-1] + float(d))
        return times[-1] + float(dt[-1])

    def play_primer(self, play_fn: Callable[[int, int], None], dtime_to_ms: float = 32.0,
                    default_velocity: int = 80, last_n: Optional[int] = 50,
                    playback_speed: float = 1.0) -> None:
        """Play part of the primer sequence audibly while keeping the display alive.

        Call primer() first with the full context (e.g. CTX_LEN notes) for the
        static rolls and the model. This method only affects audio: by default
        it plays the last ``last_n`` notes of that stored primer so the performer
        hears a short tail preview without sitting through the whole context.

        Args:
            play_fn: callback(pitch: int, velocity: int) — velocity>0 for
                     note-on, velocity==0 for note-off.
            dtime_to_ms: conversion factor from dtime token units to real
                         milliseconds (default 32.0 matches timings_divider=32).
            default_velocity: velocity used for every note-on.
            last_n: play only the last N notes of the stored primer (default 30).
                    Use None to play every stored note (full primer listen).
            playback_speed: values >1 shorten inter-note and note-off waits in
                    wall-clock time (musical dtime/dur tokens unchanged).
        """
        if not self.primer_pitches or not self.primer_dtimes:
            return

        plen: int = len(self.primer_pitches)
        k: int = plen if last_n is None else min(int(last_n), plen)
        pitches = self.primer_pitches[-k:]
        dtimes = self.primer_dtimes[-k:]
        if self.primer_durs is not None:
            durs = self.primer_durs[-k:]
        else:
            durs = list(dtimes)

        self.primer_playback_total = self._primer_timeline_total(dtimes)
        self.primer_playback_ms_per_unit = dtime_to_ms
        self.primer_playback_speed = max(float(playback_speed), 1e-6)
        self.primer_playback_elapsed = 0.0

        pending_offs: List[Tuple[float, int]] = []  # (off_time_s, pitch)
        sp: float = self.primer_playback_speed

        for idx in range(len(pitches)):
            # Wait for dtime (time since previous note)
            wait_s: float = dtimes[idx] * dtime_to_ms / 1000.0 / sp
            end_wait: float = time.time() + wait_s
            t0: float = float(sum(dtimes[:idx]))
            t1: float = t0 + float(dtimes[idx])
            wait_start: float = time.time()

            while time.time() < end_wait:
                # Release notes whose duration has elapsed
                now: float = time.time()
                still_pending: List[Tuple[float, int]] = []
                for off_time, off_pitch in pending_offs:
                    if now >= off_time:
                        play_fn(off_pitch, 0)
                    else:
                        still_pending.append((off_time, off_pitch))
                pending_offs = still_pending
                frac: float = min(1.0, max(0.0, (now - wait_start) / max(wait_s, 1e-9)))
                self.primer_playback_elapsed = t0 + float(dtimes[idx]) * frac
                self.draw()

            self.primer_playback_elapsed = t1

            # Play note-on
            pitch: int = pitches[idx]
            play_fn(pitch, default_velocity)

            # Schedule note-off
            dur_s: float = durs[idx] * dtime_to_ms / 1000.0 / sp
            pending_offs.append((time.time() + dur_s, pitch))

        # Drain remaining note-offs (cursor at end of primer timeline)
        self.primer_playback_elapsed = self.primer_playback_total
        while pending_offs:
            now = time.time()
            still_pending = []
            for off_time, off_pitch in pending_offs:
                if now >= off_time:
                    play_fn(off_pitch, 0)
                else:
                    still_pending.append((off_time, off_pitch))
            pending_offs = still_pending
            self.draw()

        self.primer_playback_elapsed = None
        self.primer_playback_speed = 1.0

        # Reset timeline so live notes start fresh after the primer
        self.start_time = time.time()
        self.last_draw_time = self.start_time

    def _random_color(self) -> Tuple[int, int, int]:
        return (random.randint(60, 255), random.randint(60, 255), random.randint(60, 255))

    def get_note(self, pitch: int, velocity: int) -> None:
        now = time.time() - self.start_time
        if velocity > 0:
            self.notes.append({'start_time': now, 'pitch': pitch, 'duration': 0.0, 'velocity': velocity, 'active': True})
        else:
            for n in reversed(self.notes):
                if n['pitch'] == pitch and n['active']:
                    n['duration'] = now - n['start_time']
                    n['active'] = False
                    break

    def get_button(self, button: int, velocity: int) -> None:
        now = time.time() - self.start_time
        if velocity > 0:
            self.buttons.append({'start_time': now, 'pitch': button, 'duration': 0.0, 'velocity': velocity, 'active': True})
        else:
            for b in reversed(self.buttons):
                if b['pitch'] == button and b['active']:
                    b['duration'] = now - b['start_time']
                    b['active'] = False
                    break

    def get_joker(self, velocity: int) -> None:
        now = time.time() - self.start_time
        if velocity > 0:
            self.jokers.append({'start_time': now, 'duration': 0.0, 'velocity': velocity, 'active': True})
        else:
            for j in reversed(self.jokers):
                if j['active']:
                    j['duration'] = now - j['start_time']
                    j['active'] = False
                    break

    def set_chord_chroma(self, chroma: Optional[List[float]], active: bool = True) -> None:
        """Set the predicted-chord chroma to overlay as white rows on the pitch
        roll, so one can see whether the generated pitches land on chord tones.
        Pass active=False (or an empty/None chroma) to hide the overlay."""
        if chroma is None:
            self.chord_pcs = []
        else:
            self.chord_pcs = [pc for pc in range(12) if pc < len(chroma) and chroma[pc] > 0.0]
        self.chord_active = bool(active) and len(self.chord_pcs) > 0

    def set_harmony_state(
        self,
        centroid_x: float,
        centroid_y: float,
        magnitude: float,
        key_root: int = 0,
        key_mode: str = 'major',
        active_pcs: Optional[List[int]] = None,
        target_chroma: Optional[List[float]] = None,
        movement_label: str = '',
        chord_label: str = '',
        tension_detail: Optional[Dict[str, float]] = None,
        active: bool = True,
    ) -> None:
        """Update the embedded Tonnetz panel with real-time harmony analysis.

        tension_detail: optional dict of tension_extractor component values
        (see _TENSION_DETAIL_SPECS for the recognised keys, e.g. 'key_conf',
        'dissonance', 'dist_tonic', ...) rendered as extra text meters below
        the main Key/Centroid/Tension summary. Unknown keys are ignored;
        missing keys are simply not drawn. Pass None to hide the section.
        """
        self.harmony_key_root = int(key_root) % 12
        self.harmony_key_mode = str(key_mode)
        self.harmony_centroid = (float(centroid_x), float(centroid_y))
        self.harmony_magnitude = max(0.0, float(magnitude))
        self.harmony_active_pcs = sorted({int(pc) % 12 for pc in (active_pcs or [])})
        if target_chroma is None:
            self.harmony_target_pcs = []
        else:
            self.harmony_target_pcs = [
                pc for pc in range(12)
                if pc < len(target_chroma) and float(target_chroma[pc]) > 0.0
            ]
        self.harmony_movement_label = movement_label
        self.harmony_chord_label = chord_label
        self.harmony_tension_detail = dict(tension_detail) if tension_detail else {}
        self.harmony_panel_active = bool(active)
        if self.harmony_panel_active:
            self.harmony_trail.append(self.harmony_centroid)

    def clear_harmony_history(self) -> None:
        self.harmony_trail.clear()
        self.harmony_active_pcs = []
        self.harmony_target_pcs = []
        self.harmony_centroid = (0.0, 0.0)
        self.harmony_magnitude = 0.0
        self.harmony_movement_label = ''
        self.harmony_chord_label = ''
        self.harmony_tension_detail = {}
        self.harmony_panel_active = False

    def update(self) -> None:
        now = time.time()
        dt = now - self.last_draw_time
        self.last_draw_time = now
        self.time_offset += self.scroll_speed * dt

    def _timeline_rect(self) -> Tuple[int, int]:
        x = self.harmony_panel_width + self.harmony_panel_gap
        return x, max(1, self.width - x)

    def stop(self) -> None:
        self.running = False
        pygame.quit()

    '''def _mainloop(self) -> None:
        last_time = time.time()
        while self.running:
            now = time.time()
            dt = now - last_time
            last_time = now
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            self.update(dt)
            self._draw()
            self.clock.tick(self.fps)'''

    def draw(self, handle_events: bool = True) -> None:
        self.update()
        if handle_events:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    exit(0)
        self.screen.fill((30, 30, 30))
        timeline_x, timeline_width = self._timeline_rect()
        if self.harmony_panel_width > 0:
            self._draw_harmony_panel(0, 0, self.harmony_panel_width, self.height)
            self._draw_vertical_timeline_layout(timeline_x, 0, timeline_width, self.height)
            pygame.display.flip()
            self.clock.tick(self.fps)
            return
        # Draw static primer rolls at the top (pitches left, buttons right)
        ref_y = 0
        half_width = timeline_width // 2
        right_width = timeline_width - half_width
        # Draw frames
        pygame.draw.rect(self.screen, (200, 200, 200), pygame.Rect(timeline_x, ref_y, half_width, self.ref_height), 2)  # pitches
        pygame.draw.rect(self.screen, (200, 200, 200), pygame.Rect(timeline_x + half_width, ref_y, right_width, self.ref_height), 2)  # buttons
        self._draw_primer_pitch_roll(self.primer_pitches, self.primer_dtimes, ref_y, self.ref_height, half_width, self.primer_colors, timeline_x)
        if self.primer_buttons is not None:
            self._draw_primer_button_roll(self.primer_buttons, self.primer_dtimes, ref_y, self.ref_height, right_width, self.primer_colors, timeline_x + half_width)
        self._draw_primer_playback_cursor(ref_y, self.ref_height, timeline_x, half_width, right_width)
        # Draw frames for piano rolls
        pitch_y = ref_y + self.ref_height + 10
        joker_y = pitch_y + self.pitch_height + 10
        button_y = joker_y + self.joker_height + 10
        pygame.draw.rect(self.screen, (200, 200, 200), pygame.Rect(timeline_x, pitch_y, timeline_width, self.pitch_height), 2)  # pitches frame
        pygame.draw.rect(self.screen, (180, 0, 180), pygame.Rect(timeline_x, joker_y, timeline_width, self.joker_height), 2)  # joker frame (magenta)
        pygame.draw.rect(self.screen, (200, 200, 200), pygame.Rect(timeline_x, button_y, timeline_width, self.button_height), 2)  # buttons frame
        all_pitches = [n['pitch'] for n in self.notes] if self.notes else [60]
        min_pitch = min(all_pitches)
        max_pitch = max(all_pitches)
        # Predicted-chord rows (white) drawn under the notes so generated pitches
        # landing on chord tones are visible on top of the white bands.
        self._draw_chord_overlay(pitch_y, self.pitch_height, min_pitch, max_pitch, timeline_x, timeline_width)
        # Pitch roll: variable height per pitch, double height
        self._draw_roll(self.notes, pitch_y, self.pitch_height, min_pitch, max_pitch, is_button=False, x_offset=timeline_x, width=timeline_width)
        # Joker strip: bright magenta bars when joker is active
        self._draw_joker_panel(joker_y, self.joker_height, timeline_x, timeline_width)
        # Button roll: configurable slots, fixed height
        self._draw_roll(self.buttons, button_y, self.button_height, 0, self.button_slots - 1, is_button=True, x_offset=timeline_x, width=timeline_width)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _draw_primer_playback_cursor(
        self,
        ref_y: int,
        ref_height: int,
        x_offset: int,
        left_width: int,
        right_width: int,
    ) -> None:
        """Vertical cursor + remaining-time label during primer playback."""
        if self.primer_playback_elapsed is None:
            return
        total: float = max(self.primer_playback_total, 1e-6)
        frac: float = max(0.0, min(1.0, float(self.primer_playback_elapsed) / total))
        x_left: int = x_offset + int(frac * float(left_width))
        x_right: int = x_offset + left_width + int(frac * float(right_width))
        cursor_color: Tuple[int, int, int] = (255, 255, 60)
        pygame.draw.line(self.screen, cursor_color, (x_left, ref_y), (x_left, ref_y + ref_height), 3)
        pygame.draw.line(self.screen, cursor_color, (x_right, ref_y), (x_right, ref_y + ref_height), 3)
        remaining_du: float = max(0.0, total - float(self.primer_playback_elapsed))
        sp: float = max(self.primer_playback_speed, 1e-6)
        remaining_s: float = remaining_du * self.primer_playback_ms_per_unit / 1000.0 / sp
        label: str = f"Primer: {remaining_s:.1f}s left"
        surf = self._primer_font.render(label, True, (255, 255, 120))
        self.screen.blit(surf, (x_offset + 8, ref_y + 4))

    def _draw_primer_pitch_roll(
        self,
        pitches: List[int],
        dtimes: List[int],
        y_offset: int,
        height: int,
        width: int,
        colors: List[Tuple[int, int, int]],
        x_offset: int = 0,
    ) -> None:
        if not pitches or not dtimes or len(pitches) != len(dtimes):
            return
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        y_scale = height / float(max_pitch - min_pitch + 1) if max_pitch != min_pitch else height
        # Calculate cumulative time for each note
        times = [0]
        for d in dtimes[:-1]:
            times.append(times[-1] + d)
        # Scale time axis to fit half width
        if times:
            total_time = times[-1] + (dtimes[-1] if dtimes else 0)
        else:
            total_time = 1
        time_scale = width / float(max(total_time, 1))
        for i, pitch in enumerate(pitches):
            x = int(x_offset + times[i] * time_scale)
            length = int(dtimes[i] * time_scale)
            color = colors[i] if i < len(colors) else (200, 200, 200)
            y = int(y_offset + height - (pitch - min_pitch + 1) * y_scale)
            rect = pygame.Rect(x, y, max(length, 2), int(max(y_scale, 2)))
            pygame.draw.rect(self.screen, color, rect)

    def _draw_primer_button_roll(
        self,
        buttons: List[int],
        dtimes: List[int],
        y_offset: int,
        height: int,
        width: int,
        colors: List[Tuple[int, int, int]],
        x_offset: int = 0,
    ) -> None:
        if not buttons or not dtimes or len(buttons) != len(dtimes):
            return
        slot_count = self.button_slots
        y_scale = height / float(slot_count)
        # Calculate cumulative time for each note
        times = [0]
        for d in dtimes[:-1]:
            times.append(times[-1] + d)
        # Scale time axis to fit half width
        if times:
            total_time = times[-1] + (dtimes[-1] if dtimes else 0)
        else:
            total_time = 1
        time_scale = width / float(max(total_time, 1))
        for i, button in enumerate(buttons):
            x = int(x_offset + times[i] * time_scale)
            length = int(dtimes[i] * time_scale)
            color = colors[i] if i < len(colors) else (200, 200, 200)
            slot = int(button) % self.button_slots
            y = int(y_offset + height - (slot + 1) * y_scale)
            rect = pygame.Rect(x, y, max(length, 2), int(max(y_scale, 2)))
            pygame.draw.rect(self.screen, color, rect)

    def _draw_chord_overlay(
        self,
        y_offset: int,
        height: int,
        min_pitch: int,
        max_pitch: int,
        x_offset: int = 0,
        width: Optional[int] = None,
    ) -> None:
        """Paint translucent white bands at every MIDI pitch whose pitch-class is a
        tone of the current predicted chord, across the live pitch-roll range, so
        generated note bars can be compared against the chord (p % 12)."""
        if not self.chord_active or not self.chord_pcs:
            return
        if max_pitch == min_pitch:  # match _draw_roll's degenerate-range handling
            min_pitch -= 1
            max_pitch += 1
        y_scale = height / float(max_pitch - min_pitch + 1)
        band_h = int(max(y_scale, 2))
        draw_width = self.width if width is None else width
        band = pygame.Surface((draw_width, band_h), pygame.SRCALPHA)
        band.fill((255, 255, 255, 80))  # translucent white
        for pitch in range(min_pitch, max_pitch + 1):
            if (pitch % 12) in self.chord_pcs:
                y = int(y_offset + height - (pitch - min_pitch + 1) * y_scale)
                self.screen.blit(band, (x_offset, y))

    def _draw_joker_panel(self, y_offset: int, height: int, x_offset: int = 0, width: Optional[int] = None) -> None:
        draw_width = self.width if width is None else width
        for item in self.jokers:
            start_x = int(x_offset + draw_width - ((self.current_time() - item['start_time']) * self.scroll_speed))
            if item['duration'] > 0.0:
                length = int(item['duration'] * self.scroll_speed)
            else:
                length = int((self.current_time() - item['start_time']) * self.scroll_speed)
            rect = pygame.Rect(start_x, y_offset + 2, max(length, 2), height - 4)
            pygame.draw.rect(self.screen, (255, 0, 255), rect)  # bright magenta

    def _velocity_color(self, velocity: int, is_button: bool = False) -> Tuple[int, int, int]:
        v = max(0, min(int(velocity), 127))
        idx = min(v // 20, 6)
        color = self.velocity_colors[idx]
        return color

    def _draw_roll(
        self,
        items: List[Dict[str, Any]],
        y_offset: int,
        height: int,
        min_val: int,
        max_val: int,
        is_button: bool = False,
        x_offset: int = 0,
        width: Optional[int] = None,
    ) -> None:
        if max_val == min_val:
            min_val -= 1
            max_val += 1
        if is_button:
            slot_count = self.button_slots
            y_scale = height / float(slot_count)
        else:
            y_scale = height / float(max_val - min_val + 1)
        draw_width = self.width if width is None else width
        for item in items:
            start_x = int(x_offset + draw_width - ((self.current_time() - item['start_time']) * self.scroll_speed))
            if item['duration'] > 0.0:
                length = int(item['duration'] * self.scroll_speed)
            else:
                length = int((self.current_time() - item['start_time']) * self.scroll_speed)
            color = self._velocity_color(item['velocity'], is_button=is_button)
            if is_button:
                slot = int(item['pitch']) % self.button_slots
                y = int(y_offset + height - (slot + 1) * y_scale)
                rect = pygame.Rect(start_x, y, max(length, 2), int(max(y_scale, 2)))
            else:
                y = int(y_offset + height - (item['pitch'] - min_val + 1) * y_scale)
                rect = pygame.Rect(start_x, y, max(length, 2), int(max(y_scale, 2)))
            pygame.draw.rect(self.screen, color, rect)

    def _draw_panel_label(self, rect: pygame.Rect, label: str, color: Tuple[int, int, int]) -> None:
        surf = self._harmony_font_small.render(label, True, color)
        self.screen.blit(surf, (rect.x + 6, rect.y + 4))

    def _draw_vertical_timeline_layout(self, x: int, y: int, width: int, height: int) -> None:
        pad = 8
        gap = 8
        primer_h = max(150, min(220, height // 4))
        live_y = y + primer_h + gap
        live_h = max(1, height - primer_h - gap)

        inner_x = x
        inner_w = max(1, width)
        joker_w = 28
        available = max(1, inner_w - (2 * gap) - joker_w)
        button_ratio = 0.24
        button_w = max(74, int(available * button_ratio))
        pitch_w = max(140, available - button_w)
        if pitch_w + button_w + joker_w + 2 * gap > inner_w:
            button_w = max(1, int(available * button_ratio))
            pitch_w = max(1, available - button_w)

        primer_pitch_rect = pygame.Rect(inner_x, y, pitch_w, primer_h)
        primer_button_rect = pygame.Rect(inner_x + pitch_w + gap + joker_w + gap, y, button_w, primer_h)
        pitch_rect = pygame.Rect(inner_x, live_y, pitch_w, live_h)
        joker_rect = pygame.Rect(inner_x + pitch_w + gap, live_y, joker_w, live_h)
        button_rect = pygame.Rect(joker_rect.right + gap, live_y, button_w, live_h)

        for rect, color, label in (
            (primer_pitch_rect, (200, 200, 200), "Primer pitches"),
            (primer_button_rect, (200, 200, 200), "Primer buttons"),
            (pitch_rect, (200, 200, 200), "Pitches"),
            (joker_rect, (180, 0, 180), "J"),
            (button_rect, (200, 200, 200), "Buttons"),
        ):
            pygame.draw.rect(self.screen, color, rect, 2)
            self._draw_panel_label(rect, label, color)

        self._draw_vertical_primer_pitch_roll(
            self.primer_pitches, self.primer_dtimes, primer_pitch_rect, self.primer_colors)
        if self.primer_buttons is not None:
            self._draw_vertical_primer_button_roll(
                self.primer_buttons, self.primer_dtimes, primer_button_rect, self.primer_colors)
        self._draw_vertical_primer_playback_cursor(primer_pitch_rect, primer_button_rect)

        all_pitches = [n['pitch'] for n in self.notes] if self.notes else [60]
        min_pitch = min(all_pitches)
        max_pitch = max(all_pitches)
        self._draw_vertical_chord_overlay(pitch_rect, min_pitch, max_pitch)
        self._draw_vertical_roll(self.notes, pitch_rect, min_pitch, max_pitch, is_button=False)
        self._draw_vertical_joker_panel(joker_rect)
        self._draw_vertical_roll(self.buttons, button_rect, 0, self.button_slots - 1, is_button=True)

    def _draw_vertical_primer_playback_cursor(self, pitch_rect: pygame.Rect, button_rect: pygame.Rect) -> None:
        if self.primer_playback_elapsed is None:
            return
        total = max(self.primer_playback_total, 1e-6)
        frac = max(0.0, min(1.0, float(self.primer_playback_elapsed) / total))
        cursor_color = (255, 255, 60)
        y_pitch = pitch_rect.y + int(frac * pitch_rect.height)
        y_button = button_rect.y + int(frac * button_rect.height)
        pygame.draw.line(self.screen, cursor_color, (pitch_rect.x, y_pitch), (pitch_rect.right, y_pitch), 3)
        pygame.draw.line(self.screen, cursor_color, (button_rect.x, y_button), (button_rect.right, y_button), 3)

        remaining_du = max(0.0, total - float(self.primer_playback_elapsed))
        sp = max(self.primer_playback_speed, 1e-6)
        remaining_s = remaining_du * self.primer_playback_ms_per_unit / 1000.0 / sp
        label = f"{remaining_s:.1f}s"
        surf = self._primer_font.render(label, True, (255, 255, 120))
        self.screen.blit(surf, (pitch_rect.x + 6, pitch_rect.bottom - surf.get_height() - 4))

    def _draw_vertical_primer_pitch_roll(
        self,
        pitches: List[int],
        dtimes: List[int],
        rect: pygame.Rect,
        colors: List[Tuple[int, int, int]],
    ) -> None:
        if not pitches or not dtimes or len(pitches) != len(dtimes):
            return
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        if max_pitch == min_pitch:
            min_pitch -= 1
            max_pitch += 1
        x_scale = rect.width / float(max_pitch - min_pitch + 1)
        times = [0]
        for d in dtimes[:-1]:
            times.append(times[-1] + d)
        total_time = times[-1] + (dtimes[-1] if dtimes else 0)
        y_scale = rect.height / float(max(total_time, 1))
        for idx, pitch in enumerate(pitches):
            x_pos = int(rect.x + (pitch - min_pitch) * x_scale)
            y_pos = int(rect.y + times[idx] * y_scale)
            h = int(max(dtimes[idx] * y_scale, 2))
            color = colors[idx] if idx < len(colors) else (200, 200, 200)
            pygame.draw.rect(self.screen, color, pygame.Rect(x_pos, y_pos, int(max(x_scale, 2)), h))

    def _draw_vertical_primer_button_roll(
        self,
        buttons: List[int],
        dtimes: List[int],
        rect: pygame.Rect,
        colors: List[Tuple[int, int, int]],
    ) -> None:
        if not buttons or not dtimes or len(buttons) != len(dtimes):
            return
        x_scale = rect.width / float(max(self.button_slots, 1))
        times = [0]
        for d in dtimes[:-1]:
            times.append(times[-1] + d)
        total_time = times[-1] + (dtimes[-1] if dtimes else 0)
        y_scale = rect.height / float(max(total_time, 1))
        for idx, button in enumerate(buttons):
            slot = int(button) % self.button_slots
            x_pos = int(rect.x + slot * x_scale)
            y_pos = int(rect.y + times[idx] * y_scale)
            h = int(max(dtimes[idx] * y_scale, 2))
            color = colors[idx] if idx < len(colors) else (200, 200, 200)
            pygame.draw.rect(self.screen, color, pygame.Rect(x_pos, y_pos, int(max(x_scale, 2)), h))

    def _draw_vertical_chord_overlay(self, rect: pygame.Rect, min_pitch: int, max_pitch: int) -> None:
        if not self.chord_active or not self.chord_pcs:
            return
        if max_pitch == min_pitch:
            min_pitch -= 1
            max_pitch += 1
        x_scale = rect.width / float(max_pitch - min_pitch + 1)
        band_w = int(max(x_scale, 2))
        band = pygame.Surface((band_w, rect.height), pygame.SRCALPHA)
        band.fill((255, 255, 255, 80))
        for pitch in range(min_pitch, max_pitch + 1):
            if (pitch % 12) in self.chord_pcs:
                x_pos = int(rect.x + (pitch - min_pitch) * x_scale)
                self.screen.blit(band, (x_pos, rect.y))

    def _draw_vertical_joker_panel(self, rect: pygame.Rect) -> None:
        for item in self.jokers:
            elapsed = max(0.0, self.current_time() - item['start_time'])
            if item['duration'] > 0.0:
                length = int(item['duration'] * self.scroll_speed)
            else:
                length = int(elapsed * self.scroll_speed)
            start_y = int(rect.bottom - elapsed * self.scroll_speed)
            pygame.draw.rect(
                self.screen,
                (255, 0, 255),
                pygame.Rect(rect.x + 3, start_y, max(2, rect.width - 6), max(length, 2)),
            )

    def _draw_vertical_roll(
        self,
        items: List[Dict[str, Any]],
        rect: pygame.Rect,
        min_val: int,
        max_val: int,
        is_button: bool = False,
    ) -> None:
        if max_val == min_val:
            min_val -= 1
            max_val += 1
        if is_button:
            x_scale = rect.width / float(max(self.button_slots, 1))
        else:
            x_scale = rect.width / float(max_val - min_val + 1)
        for item in items:
            elapsed = max(0.0, self.current_time() - item['start_time'])
            if item['duration'] > 0.0:
                length = int(item['duration'] * self.scroll_speed)
            else:
                length = int(elapsed * self.scroll_speed)
            start_y = int(rect.bottom - elapsed * self.scroll_speed)
            color = self._velocity_color(item['velocity'], is_button=is_button)
            if is_button:
                slot = int(item['pitch']) % self.button_slots
                x_pos = int(rect.x + slot * x_scale)
            else:
                x_pos = int(rect.x + (int(item['pitch']) - min_val) * x_scale)
            pygame.draw.rect(
                self.screen,
                color,
                pygame.Rect(x_pos, start_y, int(max(x_scale, 2)), max(length, 2)),
            )

    def _harmony_display_bounds(self) -> Tuple[float, float, float, float]:
        corners = [
            lattice_to_display(self.harmony_grid_min, self.harmony_grid_min),
            lattice_to_display(self.harmony_grid_min, self.harmony_grid_max),
            lattice_to_display(self.harmony_grid_max, self.harmony_grid_min),
            lattice_to_display(self.harmony_grid_max, self.harmony_grid_max),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        padding = 0.8
        return min(xs) - padding, max(xs) + padding, min(ys) - padding, max(ys) + padding

    def _harmony_display_to_pixel(self, disp_x: float, disp_y: float, rect: pygame.Rect) -> Tuple[int, int]:
        min_x, max_x, min_y, max_y = self._harmony_display_bounds()
        norm_x = (disp_x - min_x) / max(max_x - min_x, 1e-9)
        norm_y = (disp_y - min_y) / max(max_y - min_y, 1e-9)
        px = int(rect.x + norm_x * rect.width)
        py = int(rect.y + (1.0 - norm_y) * rect.height)
        return px, py

    def _harmony_lattice_to_pixel(self, lat_x: float, lat_y: float, rect: pygame.Rect) -> Tuple[int, int]:
        disp_x, disp_y = lattice_to_display(float(lat_x), float(lat_y))
        return self._harmony_display_to_pixel(disp_x, disp_y, rect)

    def _harmony_radius_pixels(self, rect: pygame.Rect, radius: float) -> int:
        cx, cy = self._harmony_lattice_to_pixel(0.0, 0.0, rect)
        rx, ry = self._harmony_lattice_to_pixel(1.0, 0.0, rect)
        scale = math.hypot(rx - cx, ry - cy)
        return int(max(0.0, radius) * scale)

    def _draw_harmony_dashed_circle(
        self,
        center: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int],
        width: int = 2,
    ) -> None:
        if radius < 2:
            return
        segments = max(32, int(2.0 * math.pi * radius / 6.0))
        for idx in range(segments):
            if idx % 2 != 0:
                continue
            a0 = 2.0 * math.pi * idx / segments
            a1 = 2.0 * math.pi * (idx + 1) / segments
            p0 = (int(center[0] + radius * math.cos(a0)), int(center[1] + radius * math.sin(a0)))
            p1 = (int(center[0] + radius * math.cos(a1)), int(center[1] + radius * math.sin(a1)))
            pygame.draw.line(self.screen, color, p0, p1, width)

    def _harmony_pc_to_lattice(self, pc: int, key_root: int) -> Tuple[float, float]:
        key_x, key_y = TONNETZ_COORDS[key_root % 12]
        rel_x, rel_y = TONNETZ_COORDS[(int(pc) - key_root) % 12]
        return key_x + rel_x, key_y + rel_y

    def _draw_harmony_polygon(
        self,
        pcs: List[int],
        key_root: int,
        rect: pygame.Rect,
        fill_color: Tuple[int, int, int, int],
        point_color: Tuple[int, int, int],
        point_radius: int,
        outline_color: Tuple[int, int, int, int] = (255, 255, 255, 120),
    ) -> None:
        if len(pcs) < 1:
            return
        points = [
            self._harmony_lattice_to_pixel(*self._harmony_pc_to_lattice(pc, key_root), rect)
            for pc in pcs
        ]
        if len(points) >= 3:
            cx = sum(p[0] for p in points) / float(len(points))
            cy = sum(p[1] for p in points) / float(len(points))
            points = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
            poly_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.polygon(poly_surface, fill_color, points)
            pygame.draw.polygon(poly_surface, outline_color, points, 2)
            self.screen.blit(poly_surface, (0, 0))
        elif len(points) == 2:
            pygame.draw.line(self.screen, outline_color[:3], points[0], points[1], 2)

        for px, py in points:
            pygame.draw.circle(self.screen, point_color, (px, py), point_radius)
            pygame.draw.circle(self.screen, (255, 255, 255), (px, py), max(2, point_radius // 2))

    def _draw_harmony_text(self, text: str, x: int, y: int, color: Tuple[int, int, int],
                           font: Optional[pygame.font.Font] = None) -> int:
        font = font if font is not None else self._harmony_font
        surf = font.render(text, True, color)
        self.screen.blit(surf, (x, y))
        return y + surf.get_height() + 5

    def _pitch_class_list(self, pcs: List[int]) -> str:
        if not pcs:
            return '-'
        return ' '.join(PITCH_NAMES[pc % 12] for pc in pcs)

    def _text_meter(self, value: float, vmin: float, vmax: float, width: int = 8) -> str:
        frac = 0.0 if vmax <= vmin else (value - vmin) / (vmax - vmin)
        frac = max(0.0, min(1.0, frac))
        filled = int(round(frac * width))
        return '[' + ('#' * filled) + ('-' * (width - filled)) + ']'

    def _draw_harmony_panel(self, x: int, y: int, width: int, height: int) -> None:
        panel = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (22, 24, 38), panel)
        pygame.draw.rect(self.screen, (90, 94, 120), panel, 2)

        pad = 10
        title_y = y + pad
        title = self._harmony_font_title.render("Harmony", True, (238, 238, 238))
        self.screen.blit(title, (x + pad, title_y))

        grid_top = title_y + title.get_height() + 10
        # Leave more room below the grid for the extra tension-detail meters.
        grid_frac = 0.50 if self.harmony_tension_detail else 0.64
        grid_size = max(120, min(width - pad * 2, int(height * grid_frac)))
        grid_rect = pygame.Rect(x + pad, grid_top, width - pad * 2, grid_size)
        pygame.draw.rect(self.screen, (18, 24, 48), grid_rect)

        line_color = (58, 58, 90)
        for gy in range(self.harmony_grid_min, self.harmony_grid_max + 1):
            p1 = self._harmony_lattice_to_pixel(self.harmony_grid_min, gy, grid_rect)
            p2 = self._harmony_lattice_to_pixel(self.harmony_grid_max, gy, grid_rect)
            pygame.draw.line(self.screen, line_color, p1, p2, 1)
        for gx in range(self.harmony_grid_min, self.harmony_grid_max + 1):
            p1 = self._harmony_lattice_to_pixel(gx, self.harmony_grid_min, grid_rect)
            p2 = self._harmony_lattice_to_pixel(gx, self.harmony_grid_max, grid_rect)
            pygame.draw.line(self.screen, line_color, p1, p2, 1)
        for diag in range(2 * self.harmony_grid_min, 2 * self.harmony_grid_max + 1):
            points = []
            for gx in range(self.harmony_grid_min, self.harmony_grid_max + 1):
                gy = diag - gx
                if self.harmony_grid_min <= gy <= self.harmony_grid_max:
                    points.append(self._harmony_lattice_to_pixel(gx, gy, grid_rect))
            if len(points) >= 2:
                pygame.draw.lines(self.screen, line_color, False, points, 1)

        for gx in range(self.harmony_grid_min, self.harmony_grid_max + 1):
            for gy in range(self.harmony_grid_min, self.harmony_grid_max + 1):
                px, py = self._harmony_lattice_to_pixel(gx, gy, grid_rect)
                if gx == 0 and gy == 0:
                    color = (255, 215, 0)
                    radius = 7
                elif gx == 1 and gy == 0:
                    color = (255, 107, 107)
                    radius = 6
                elif gx == -1 and gy == 0:
                    color = (78, 205, 196)
                    radius = 6
                elif gy == 1:
                    color = (149, 225, 211)
                    radius = 5
                elif gy == -1:
                    color = (243, 129, 129)
                    radius = 5
                else:
                    color = (122, 122, 154)
                    radius = 4
                pygame.draw.circle(self.screen, color, (px, py), radius)
                label = self._harmony_font_small.render(
                    get_pitch_at_tonnetz_coord(gx, gy, key_root=0), True, color)
                self.screen.blit(label, label.get_rect(center=(px, py - 15)))

        key_root = self.harmony_key_root
        key_x, key_y = TONNETZ_COORDS[key_root]
        key_px, key_py = self._harmony_lattice_to_pixel(key_x, key_y, grid_rect)
        pygame.draw.circle(self.screen, (255, 215, 0), (key_px, key_py), 13, 2)

        if self.harmony_magnitude > 0.01:
            radius = self._harmony_radius_pixels(grid_rect, self.harmony_magnitude)
            self._draw_harmony_dashed_circle((key_px, key_py), radius, (255, 107, 107), 2)

        if len(self.harmony_trail) >= 2:
            trail_points = []
            for rel_x, rel_y in self.harmony_trail:
                trail_points.append(self._harmony_lattice_to_pixel(key_x + rel_x, key_y + rel_y, grid_rect))
            pygame.draw.lines(self.screen, (255, 159, 67), False, trail_points, 2)
            for idx, point in enumerate(trail_points):
                fade = (idx + 1) / float(len(trail_points))
                color = (int(110 + 145 * fade), int(70 + 89 * fade), int(45 + 22 * fade))
                pygame.draw.circle(self.screen, color, point, max(2, int(2 + 4 * fade)))

        self._draw_harmony_polygon(
            self.harmony_active_pcs,
            key_root,
            grid_rect,
            (116, 185, 255, 58),
            (116, 185, 255),
            7,
        )
        self._draw_harmony_polygon(
            self.harmony_target_pcs,
            key_root,
            grid_rect,
            (255, 255, 255, 28),
            (245, 245, 245),
            5,
            (255, 255, 255, 100),
        )

        cx, cy = self.harmony_centroid
        cur_px, cur_py = self._harmony_lattice_to_pixel(key_x + cx, key_y + cy, grid_rect)
        pygame.draw.circle(self.screen, (255, 159, 67), (cur_px, cur_py), 12, 2)
        pygame.draw.circle(self.screen, (255, 159, 67), (cur_px, cur_py), 8)
        pygame.draw.circle(self.screen, (255, 255, 255), (cur_px, cur_py), 4)
        pygame.draw.rect(self.screen, (90, 94, 120), grid_rect, 1)

        text_y = grid_rect.bottom + 12
        text_x = x + pad
        key_label = f"Key: {PITCH_NAMES[key_root]} {self.harmony_key_mode}"
        text_y = self._draw_harmony_text(key_label, text_x, text_y, (255, 215, 0))
        text_y = self._draw_harmony_text(
            f"Centroid: {cx:+.2f} {cy:+.2f}", text_x, text_y, (255, 159, 67))
        text_y = self._draw_harmony_text(
            f"Tension: {self.harmony_magnitude:.2f}", text_x, text_y, (255, 107, 107))
        text_y = self._draw_harmony_text(
            f"Region: {self._pitch_class_list(self.harmony_active_pcs)}",
            text_x, text_y, (116, 185, 255), self._harmony_font_small)
        if self.harmony_movement_label:
            text_y = self._draw_harmony_text(
                f"Move: {self.harmony_movement_label}", text_x, text_y, (220, 220, 220),
                self._harmony_font_small)
        if self.harmony_chord_label:
            text_y = self._draw_harmony_text(
                self.harmony_chord_label[:30], text_x, text_y, (245, 245, 245),
                self._harmony_font_small)
        if self.harmony_tension_detail:
            self._draw_tension_detail(text_x, text_y, width - 2 * pad)

    def _draw_tension_detail(self, x: int, y: int, avail_width: int) -> None:
        """Two-column mini text meters for the tension_extractor component
        breakdown (key confidence/entropy, distances, dissonance, etc.)."""
        y = self._draw_harmony_text("Tension detail:", x, y, (200, 200, 220), self._harmony_font_small)
        col_w = avail_width // 2
        col_x = [x, x + col_w]
        col_y = [y, y]
        col = 0
        for key, label, vmin, vmax in _TENSION_DETAIL_SPECS:
            if key not in self.harmony_tension_detail:
                continue
            value = self.harmony_tension_detail[key]
            meter = self._text_meter(value, vmin, vmax)
            line = f"{label:8s}{meter}{value:+.2f}"
            col_y[col] = self._draw_harmony_text(line, col_x[col], col_y[col], (178, 196, 255), self._harmony_font_small)
            col = 1 - col

    def current_time(self) -> float:
        return time.time() - self.start_time
