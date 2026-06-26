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
from typing import List, Tuple, Optional, Dict, Any, Union, Callable

# Note: For strict TorchScript compatibility, avoid dynamic typing and use explicit types.

class Visualizer:

    def __init__(self, width: int = 1200, height: int = 940, roll_height: int = 180, fps: int = 30, button_slots: int = 12) -> None:
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

    def update(self) -> None:
        now = time.time()
        dt = now - self.last_draw_time
        self.last_draw_time = now
        self.time_offset += self.scroll_speed * dt

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
        # Draw static primer rolls at the top (pitches left, buttons right)
        ref_y = 0
        half_width = self.width // 2
        # Draw frames
        pygame.draw.rect(self.screen, (200, 200, 200), pygame.Rect(0, ref_y, half_width, self.ref_height), 2)  # pitches
        pygame.draw.rect(self.screen, (200, 200, 200), pygame.Rect(half_width, ref_y, self.width - half_width, self.ref_height), 2)  # buttons
        self._draw_primer_pitch_roll(self.primer_pitches, self.primer_dtimes, ref_y, self.ref_height, half_width, self.primer_colors)
        if self.primer_buttons is not None:
            self._draw_primer_button_roll(self.primer_buttons, self.primer_dtimes, ref_y, self.ref_height, half_width, self.primer_colors)
        self._draw_primer_playback_cursor(ref_y, self.ref_height, half_width)
        # Draw frames for piano rolls
        pitch_y = ref_y + self.ref_height + 10
        joker_y = pitch_y + self.pitch_height + 10
        button_y = joker_y + self.joker_height + 10
        pygame.draw.rect(self.screen, (200, 200, 200), pygame.Rect(0, pitch_y, self.width, self.pitch_height), 2)  # pitches frame
        pygame.draw.rect(self.screen, (180, 0, 180), pygame.Rect(0, joker_y, self.width, self.joker_height), 2)  # joker frame (magenta)
        pygame.draw.rect(self.screen, (200, 200, 200), pygame.Rect(0, button_y, self.width, self.button_height), 2)  # buttons frame
        all_pitches = [n['pitch'] for n in self.notes] if self.notes else [60]
        min_pitch = min(all_pitches)
        max_pitch = max(all_pitches)
        # Predicted-chord rows (white) drawn under the notes so generated pitches
        # landing on chord tones are visible on top of the white bands.
        self._draw_chord_overlay(pitch_y, self.pitch_height, min_pitch, max_pitch)
        # Pitch roll: variable height per pitch, double height
        self._draw_roll(self.notes, pitch_y, self.pitch_height, min_pitch, max_pitch, is_button=False)
        # Joker strip: bright magenta bars when joker is active
        self._draw_joker_panel(joker_y, self.joker_height)
        # Button roll: configurable slots, fixed height
        self._draw_roll(self.buttons, button_y, self.button_height, 0, self.button_slots - 1, is_button=True)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _draw_primer_playback_cursor(self, ref_y: int, ref_height: int, half_width: int) -> None:
        """Vertical cursor + remaining-time label during primer playback."""
        if self.primer_playback_elapsed is None:
            return
        total: float = max(self.primer_playback_total, 1e-6)
        frac: float = max(0.0, min(1.0, float(self.primer_playback_elapsed) / total))
        x_left: int = int(frac * float(half_width))
        right_w: int = self.width - half_width
        x_right: int = half_width + int(frac * float(right_w))
        cursor_color: Tuple[int, int, int] = (255, 255, 60)
        pygame.draw.line(self.screen, cursor_color, (x_left, ref_y), (x_left, ref_y + ref_height), 3)
        pygame.draw.line(self.screen, cursor_color, (x_right, ref_y), (x_right, ref_y + ref_height), 3)
        remaining_du: float = max(0.0, total - float(self.primer_playback_elapsed))
        sp: float = max(self.primer_playback_speed, 1e-6)
        remaining_s: float = remaining_du * self.primer_playback_ms_per_unit / 1000.0 / sp
        label: str = f"Primer: {remaining_s:.1f}s left"
        surf = self._primer_font.render(label, True, (255, 255, 120))
        self.screen.blit(surf, (8, ref_y + 4))

    def _draw_primer_pitch_roll(self, pitches: List[int], dtimes: List[int], y_offset: int, height: int, width: int, colors: List[Tuple[int, int, int]]) -> None:
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
            x = int(times[i] * time_scale)
            length = int(dtimes[i] * time_scale)
            color = colors[i] if i < len(colors) else (200, 200, 200)
            y = int(y_offset + height - (pitch - min_pitch + 1) * y_scale)
            rect = pygame.Rect(x, y, max(length, 2), int(max(y_scale, 2)))
            pygame.draw.rect(self.screen, color, rect)

    def _draw_primer_button_roll(self, buttons: List[int], dtimes: List[int], y_offset: int, height: int, width: int, colors: List[Tuple[int, int, int]]) -> None:
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
            x = int(self.width // 2 + times[i] * time_scale)
            length = int(dtimes[i] * time_scale)
            color = colors[i] if i < len(colors) else (200, 200, 200)
            slot = int(button) % self.button_slots
            y = int(y_offset + height - (slot + 1) * y_scale)
            rect = pygame.Rect(x, y, max(length, 2), int(max(y_scale, 2)))
            pygame.draw.rect(self.screen, color, rect)

    def _draw_chord_overlay(self, y_offset: int, height: int, min_pitch: int, max_pitch: int) -> None:
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
        band = pygame.Surface((self.width, band_h), pygame.SRCALPHA)
        band.fill((255, 255, 255, 80))  # translucent white
        for pitch in range(min_pitch, max_pitch + 1):
            if (pitch % 12) in self.chord_pcs:
                y = int(y_offset + height - (pitch - min_pitch + 1) * y_scale)
                self.screen.blit(band, (0, y))

    def _draw_joker_panel(self, y_offset: int, height: int) -> None:
        for item in self.jokers:
            start_x = int(self.width - ((self.current_time() - item['start_time']) * self.scroll_speed))
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

    def _draw_roll(self, items: List[Dict[str, Any]], y_offset: int, height: int, min_val: int, max_val: int, is_button: bool = False) -> None:
        if max_val == min_val:
            min_val -= 1
            max_val += 1
        if is_button:
            slot_count = self.button_slots
            y_scale = height / float(slot_count)
        else:
            y_scale = height / float(max_val - min_val + 1)
        for item in items:
            start_x = int(self.width - ((self.current_time() - item['start_time']) * self.scroll_speed))
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

    def current_time(self) -> float:
        return time.time() - self.start_time