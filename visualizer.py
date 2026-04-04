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
from typing import List, Tuple, Optional, Dict, Any, Union

# Note: For strict TorchScript compatibility, avoid dynamic typing and use explicit types.

class Visualizer:

    def __init__(self, width: int = 1200, height: int = 940, roll_height: int = 180, fps: int = 30, button_slots: int = 12) -> None:
        pygame.init()
        self.width = width
        self.roll_height = roll_height
        self.ref_height = roll_height
        self.button_height = roll_height
        self.pitch_height = roll_height * 2  # Double height for pitches
        self.height = self.ref_height + 10 + self.pitch_height + 10 + self.button_height + 20
        self.fps = fps
        self.button_slots = button_slots
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Real-Time Piano Roll Visualizer')
        self.clock = pygame.time.Clock()
        self.running = True
        self.notes: List[Dict[str, Any]] = []
        self.buttons: List[Dict[str, Any]] = []
        self.primer_pitches: List[int] = []
        self.primer_dtimes: List[int] = []
        self.primer_buttons: Optional[List[int]] = None
        self.primer_colors: List[Tuple[int, int, int]] = []
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

    def primer(self, pitches: Union[List[int], Any], dtimes: Union[List[int], Any], buttons: Optional[Union[List[int], Any]] = None) -> None:
        # Accepts lists or tensors for pitches, dtimes, and optionally buttons
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
        # Assign a random color for each note in the primer
        self.primer_colors = [self._random_color() for _ in self.primer_pitches]

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
        # Draw frames for piano rolls
        pitch_y = ref_y + self.ref_height + 10
        button_y = pitch_y + self.pitch_height + 10
        pygame.draw.rect(self.screen, (200, 200, 200), pygame.Rect(0, pitch_y, self.width, self.pitch_height), 2)  # pitches frame
        pygame.draw.rect(self.screen, (200, 200, 200), pygame.Rect(0, button_y, self.width, self.button_height), 2)  # buttons frame
        all_pitches = [n['pitch'] for n in self.notes] if self.notes else [60]
        min_pitch = min(all_pitches)
        max_pitch = max(all_pitches)
        # Pitch roll: variable height per pitch, double height
        self._draw_roll(self.notes, pitch_y, self.pitch_height, min_pitch, max_pitch, is_button=False)
        # Button roll: configurable slots, fixed height
        self._draw_roll(self.buttons, button_y, self.button_height, 0, self.button_slots - 1, is_button=True)
        pygame.display.flip()
        self.clock.tick(self.fps)

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