#===================================================================================================
# Monster Genie interaction_harmony.py Python module
# Real-time interaction with harmony-conditioned autoencoder
# 
# User controls:
# - QWERTY keyboard: buttons 0-11 for melody control
# - Mouse/trackpad position: Tonnetz X/Y (harm_x, harm_y) for harmony target
# - Trackpad pinch / scroll wheel / +/- keys: tension magnitude (harm_r)
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

import time
import sys
import math
import numpy as np
from typing import Optional, List, Tuple, Dict
from collections import deque

import torch
import pygame
from pygame.locals import *

# Audio
import fluidsynth

# Project imports
from params import VOCAB_SIZE_PITCH
from model_loader import load_model
from models import get_model_hparams
from midiUtils import midi_to_dict, to_device, dict_to_song, ms_SONG_to_MIDI_Converter
from harmony_extractor import chroma_to_tonnetz_tension, pitch_to_tonnetz, TONNETZ_COORDS, PITCH_NAMES

#===================================================================================================
# CONFIGURATION
#===================================================================================================

TRACES = False  # Debug output

# Device
device = torch.device('mps' if torch.backends.mps.is_available() else 
                      'cuda' if torch.cuda.is_available() else 'cpu')

# Model
MODEL_NAME = 'autoenc_no_dtime_harmony_v1'

# Context and generation
CTX_LEN = 128  # Number of notes in context
TOTAL_GEN_LEN = 512  # Max notes to generate
TEMPERATURE = 0.8  # Sampling temperature

# Harmony quantization ranges (same as dataset creation)
X_RANGE = (-3.0, 3.0)  # Tonnetz X (fifths axis)
Y_RANGE = (-2.0, 2.0)  # Tonnetz Y (thirds axis)
R_MAX = 3.0            # Maximum tension magnitude
N_BINS = 128           # Quantization bins (0-127)

# Visualization
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900
TONNETZ_GRID_RANGE = (-3, 3)  # Grid display range

# Harmony analyzer
ROLLING_WINDOW_SIZE = 16  # Number of recent pitches to analyze

# Files
SEED_MIDI_PATH = './samples/clairTester_to_end_monophonic.midi'
OUTPUT_MIDI_NAME = './out/interactive_harmony_performance'
SOUNDFONT_PATH = './piano.sf2'

#===================================================================================================
# KEY MAPPING (12 buttons: 0-11)
#===================================================================================================

KEY_MAPPING: Dict[int, int] = {
    # Number row: buttons 0-9
    K_1: 0, K_2: 1, K_3: 2, K_4: 3, K_5: 4,
    K_6: 5, K_7: 6, K_8: 7, K_9: 8, K_0: 9,
    K_MINUS: 10, K_EQUALS: 11,
    # QWERTY row: alternative mapping
    K_q: 0, K_w: 1, K_e: 2, K_r: 3, K_t: 4,
    K_y: 5, K_u: 6, K_i: 7, K_o: 8, K_p: 9,
    K_LEFTBRACKET: 10, K_RIGHTBRACKET: 11,
}

#===================================================================================================
# COORDINATE CONVERSION
#===================================================================================================

# Tonnetz display geometry (triangular lattice) - from tonnetz.py
_SQRT3 = float(np.sqrt(3.0))


def lattice_to_display(x: float, y: float) -> Tuple[float, float]:
    """
    Map Tonnetz lattice coords (x=fifths, y=thirds) into 2D display coords.
    Creates a triangular/rhombic layout like standard Tonnetz diagrams.
    """
    X = x + 0.5 * y
    Y = -(_SQRT3 / 2.0) * y
    return (X, Y)


def display_to_lattice(X: float, Y: float) -> Tuple[float, float]:
    """
    Inverse of lattice_to_display: convert display coords back to lattice coords.
    """
    # From: X = x + 0.5*y, Y = -sqrt(3)/2 * y
    # Solve: y = -2*Y/sqrt(3), x = X - 0.5*y
    y = -2.0 * Y / _SQRT3
    x = X - 0.5 * y
    return (x, y)


def pixel_to_display(
    pixel_x: int, pixel_y: int,
    window_width: int, window_height: int,
    grid_range: Tuple[int, int] = (-3, 3)
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to display coordinates.
    Maps window center to display origin, with appropriate scaling.
    """
    grid_min, grid_max = grid_range
    
    # Compute display space bounds from lattice corners
    corners = [
        lattice_to_display(grid_min, grid_min),
        lattice_to_display(grid_min, grid_max),
        lattice_to_display(grid_max, grid_min),
        lattice_to_display(grid_max, grid_max),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    
    padding = 0.8
    disp_x_min, disp_x_max = min(xs) - padding, max(xs) + padding
    disp_y_min, disp_y_max = min(ys) - padding, max(ys) + padding
    
    # Map pixel to display coords
    # Pixel (0,0) is top-left, display y increases upward
    norm_x = pixel_x / window_width
    norm_y = 1.0 - (pixel_y / window_height)  # Flip Y
    
    disp_x = disp_x_min + norm_x * (disp_x_max - disp_x_min)
    disp_y = disp_y_min + norm_y * (disp_y_max - disp_y_min)
    
    return (disp_x, disp_y)


def lattice_to_bins(lat_x: float, lat_y: float) -> Tuple[int, int]:
    """
    Convert lattice coordinates to quantized bins (0-127).
    Uses the same quantization as dataset creation.
    """
    x_bin = int(np.clip(
        (lat_x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * N_BINS,
        0, N_BINS - 1
    ))
    y_bin = int(np.clip(
        (lat_y - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]) * N_BINS,
        0, N_BINS - 1
    ))
    return (x_bin, y_bin)


def bins_to_lattice(x_bin: int, y_bin: int) -> Tuple[float, float]:
    """
    Convert bins back to lattice coordinates (approximate inverse).
    """
    lat_x = X_RANGE[0] + (x_bin / (N_BINS - 1)) * (X_RANGE[1] - X_RANGE[0])
    lat_y = Y_RANGE[0] + (y_bin / (N_BINS - 1)) * (Y_RANGE[1] - Y_RANGE[0])
    return (lat_x, lat_y)


def tension_to_bin(tension: float) -> int:
    """Convert tension magnitude (0 to R_MAX) to bin (0-127)."""
    return int(np.clip(tension / R_MAX * N_BINS, 0, N_BINS - 1))


def bin_to_tension(r_bin: int) -> float:
    """Convert bin (0-127) to tension magnitude."""
    return (r_bin / (N_BINS - 1)) * R_MAX


#===================================================================================================
# ROLLING HARMONY ANALYZER
#===================================================================================================

class RollingHarmonyAnalyzer:
    """
    Analyzes harmony from a rolling window of recent pitches.
    Computes Tonnetz centroid and tension magnitude in real-time.
    Also tracks currently active pitches for chord visualization.
    """
    
    def __init__(self, window_size: int = 16, key_root: int = 0):
        self.window_size = window_size
        self.key_root = key_root  # Default to C
        self.pitches: deque = deque(maxlen=window_size)
        self.active_pitches: set = set()  # Currently sounding pitch classes
    
    def add_pitch(self, pitch: int) -> None:
        """Add a new pitch to the rolling window."""
        self.pitches.append(pitch)
        self.active_pitches.add(pitch % 12)
    
    def remove_pitch(self, pitch: int) -> None:
        """Remove a pitch from active set (note-off)."""
        pc = pitch % 12
        if pc in self.active_pitches:
            self.active_pitches.discard(pc)
    
    def get_active_pitch_classes(self) -> List[int]:
        """Get list of currently active pitch classes."""
        return sorted(list(self.active_pitches))
    
    def get_chroma(self) -> np.ndarray:
        """Compute chroma vector from current pitches."""
        chroma = np.zeros(12)
        for pitch in self.pitches:
            pitch_class = pitch % 12
            chroma[pitch_class] += 1.0
        return chroma
    
    def get_tonnetz_position(self) -> Tuple[float, float, float]:
        """
        Get current Tonnetz position (relative to key) and tension.
        
        Returns:
            (centroid_x, centroid_y, tension_magnitude)
        """
        chroma = self.get_chroma()
        if np.sum(chroma) == 0:
            return (0.0, 0.0, 0.0)
        return chroma_to_tonnetz_tension(chroma, self.key_root)
    
    def clear(self) -> None:
        """Clear the pitch buffer."""
        self.pitches.clear()
        self.active_pitches.clear()


#===================================================================================================
# TONNETZ RENDERER (Pygame)
#===================================================================================================

class TonnetzRenderer:
    """
    Renders Tonnetz grid and markers using pygame.
    Shows target harmony (user input) and actual harmony (analyzed output).
    """
    
    def __init__(
        self,
        surface: pygame.Surface,
        grid_range: Tuple[int, int] = (-3, 3),
        key_root: int = 0
    ):
        self.surface = surface
        self.width = surface.get_width()
        self.height = surface.get_height()
        self.grid_min, self.grid_max = grid_range
        self.key_root = key_root
        
        # Colors
        self.bg_color = (26, 26, 46)  # #1a1a2e
        self.grid_color = (58, 58, 90)  # #3a3a5a
        self.text_color = (200, 200, 200)
        self.target_color = (116, 185, 255)  # Blue - user target
        self.actual_color = (255, 159, 67)   # Orange - analyzed actual
        self.tonic_color = (255, 215, 0)     # Gold - tonic
        self.tension_color = (255, 107, 107) # Red - tension circle
        self.chord_color = (116, 185, 255)   # Blue - chord polygon #74b9ff
        self.dominant_color = (255, 107, 107)  # Red for dominant (G)
        self.subdominant_color = (78, 205, 196)  # Teal for subdominant (F)
        self.major_third_color = (149, 225, 211)  # Light green for major third region
        self.minor_third_color = (243, 129, 129)  # Coral for minor region
        
        # Trail effect
        self.trail_positions: deque = deque(maxlen=50)  # Store recent positions
        
        # Pre-compute display bounds
        corners = [
            lattice_to_display(self.grid_min, self.grid_min),
            lattice_to_display(self.grid_min, self.grid_max),
            lattice_to_display(self.grid_max, self.grid_min),
            lattice_to_display(self.grid_max, self.grid_max),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        self.padding = 0.8
        self.disp_x_min = min(xs) - self.padding
        self.disp_x_max = max(xs) + self.padding
        self.disp_y_min = min(ys) - self.padding
        self.disp_y_max = max(ys) + self.padding
        
        # Font
        pygame.font.init()
        self.font_small = pygame.font.SysFont('monospace', 14)
        self.font_medium = pygame.font.SysFont('monospace', 18)
        self.font_large = pygame.font.SysFont('monospace', 24)
    
    def display_to_pixel(self, disp_x: float, disp_y: float) -> Tuple[int, int]:
        """Convert display coordinates to pixel coordinates."""
        norm_x = (disp_x - self.disp_x_min) / (self.disp_x_max - self.disp_x_min)
        norm_y = (disp_y - self.disp_y_min) / (self.disp_y_max - self.disp_y_min)
        
        pixel_x = int(norm_x * self.width)
        pixel_y = int((1.0 - norm_y) * self.height)  # Flip Y
        
        return (pixel_x, pixel_y)
    
    def lattice_to_pixel(self, lat_x: float, lat_y: float) -> Tuple[int, int]:
        """Convert lattice coordinates to pixel coordinates."""
        disp_x, disp_y = lattice_to_display(lat_x, lat_y)
        return self.display_to_pixel(disp_x, disp_y)
    
    def draw_grid(self) -> None:
        """Draw the Tonnetz grid with pitch labels."""
        # Clear background
        self.surface.fill(self.bg_color)
        
        # Draw lattice lines
        # 1. Horizontal lines (constant y in lattice)
        for y in range(self.grid_min, self.grid_max + 1):
            p1 = self.lattice_to_pixel(self.grid_min, y)
            p2 = self.lattice_to_pixel(self.grid_max, y)
            pygame.draw.line(self.surface, self.grid_color, p1, p2, 1)
        
        # 2. Vertical lines (constant x in lattice)
        for x in range(self.grid_min, self.grid_max + 1):
            p1 = self.lattice_to_pixel(x, self.grid_min)
            p2 = self.lattice_to_pixel(x, self.grid_max)
            pygame.draw.line(self.surface, self.grid_color, p1, p2, 1)
        
        # 3. Diagonal lines (x + y = constant)
        for d in range(2 * self.grid_min, 2 * self.grid_max + 1):
            points = []
            for x in range(self.grid_min, self.grid_max + 1):
                y = d - x
                if self.grid_min <= y <= self.grid_max:
                    points.append(self.lattice_to_pixel(x, y))
            if len(points) >= 2:
                pygame.draw.lines(self.surface, self.grid_color, False, points, 1)
        
        # Draw pitch labels at grid nodes with color coding
        for x in range(self.grid_min, self.grid_max + 1):
            for y in range(self.grid_min, self.grid_max + 1):
                # Get pitch name (C at origin)
                semitone = (7 * x + 4 * y) % 12
                pitch_name = PITCH_NAMES[semitone]
                
                px, py = self.lattice_to_pixel(x, y)
                
                # Color-code based on harmonic function
                if x == 0 and y == 0:
                    # Tonic (C)
                    color = self.tonic_color
                    circle_radius = 8
                    circle_width = 0  # Filled
                elif x == 1 and y == 0:
                    # Dominant (G)
                    color = self.dominant_color
                    circle_radius = 6
                    circle_width = 0
                elif x == -1 and y == 0:
                    # Subdominant (F)
                    color = self.subdominant_color
                    circle_radius = 6
                    circle_width = 0
                elif y == 1:
                    # Major third region
                    color = self.major_third_color
                    circle_radius = 5
                    circle_width = 0
                elif y == -1:
                    # Minor third region
                    color = self.minor_third_color
                    circle_radius = 5
                    circle_width = 0
                else:
                    color = (122, 122, 154)  # #7a7a9a
                    circle_radius = 4
                    circle_width = 0
                
                # Draw circle at grid point
                pygame.draw.circle(self.surface, color, (px, py), circle_radius, circle_width)
                if x == 0 and y == 0:
                    # Add ring around tonic
                    pygame.draw.circle(self.surface, color, (px, py), 12, 2)
                
                # Draw label
                text = self.font_small.render(pitch_name, True, color)
                text_rect = text.get_rect(center=(px, py - 18))
                self.surface.blit(text, text_rect)
    
    def draw_dashed_circle(
        self,
        center: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int],
        width: int = 2,
        dash_length: int = 10
    ) -> None:
        """Draw a dashed circle (discontinuous)."""
        if radius < 1:
            return
        
        # Calculate number of segments for smooth circle
        num_segments = max(32, int(2 * np.pi * radius / 5))
        
        for i in range(num_segments):
            angle1 = 2 * np.pi * i / num_segments
            angle2 = 2 * np.pi * (i + 1) / num_segments
            
            # Draw only every other segment for dashed effect
            if i % 2 == 0:
                x1 = int(center[0] + radius * np.cos(angle1))
                y1 = int(center[1] + radius * np.sin(angle1))
                x2 = int(center[0] + radius * np.cos(angle2))
                y2 = int(center[1] + radius * np.sin(angle2))
                pygame.draw.line(self.surface, color, (x1, y1), (x2, y2), width)
    
    def draw_target_marker(
        self,
        harm_x: int, harm_y: int, harm_r: int
    ) -> None:
        """Draw the target harmony marker (user input) - blue with glow."""
        lat_x, lat_y = bins_to_lattice(harm_x, harm_y)
        px, py = self.lattice_to_pixel(lat_x, lat_y)
        
        # Glow effect - outer ring
        pygame.draw.circle(self.surface, self.target_color, (px, py), 18, 3)
        # Inner white dot
        pygame.draw.circle(self.surface, (255, 255, 255), (px, py), 8)
        
        # Draw dashed tension circle around tonic
        tension = bin_to_tension(harm_r)
        if tension > 0.1:
            tonic_px, tonic_py = self.lattice_to_pixel(0, 0)
            # Scale tension to pixels (approximate)
            radius_pixels = int(tension * 50)  # Adjust scaling factor
            self.draw_dashed_circle(
                (tonic_px, tonic_py), radius_pixels, 
                self.tension_color, width=3
            )
    
    def draw_actual_marker(
        self,
        lat_x: float, lat_y: float, tension: float
    ) -> None:
        """Draw the actual harmony marker (analyzed) - orange with glow."""
        px, py = self.lattice_to_pixel(lat_x, lat_y)
        
        # Store position for trail
        self.trail_positions.append((px, py))
        
        # Glow effect - outer ring
        pygame.draw.circle(self.surface, self.actual_color, (px, py), 15, 2)
        # Middle filled circle
        pygame.draw.circle(self.surface, self.actual_color, (px, py), 10)
        # Inner white dot for glow
        pygame.draw.circle(self.surface, (255, 255, 255), (px, py), 6)
    
    def draw_trail(self) -> None:
        """Draw the trail effect showing recent trajectory."""
        if len(self.trail_positions) < 2:
            return
        
        # Draw line connecting trail points
        if len(self.trail_positions) >= 2:
            pygame.draw.lines(
                self.surface, self.actual_color, False, 
                list(self.trail_positions), 2
            )
        
        # Draw fading scatter points along trail
        for i, (px, py) in enumerate(self.trail_positions):
            # Fade from transparent to opaque
            alpha_factor = i / len(self.trail_positions)
            size = int(5 + alpha_factor * 8)
            
            # Create semi-transparent circles for fade effect
            color = (
                int(self.actual_color[0] * (0.3 + 0.7 * alpha_factor)),
                int(self.actual_color[1] * (0.3 + 0.7 * alpha_factor)),
                int(self.actual_color[2] * (0.3 + 0.7 * alpha_factor))
            )
            pygame.draw.circle(self.surface, color, (px, py), size)
    
    def get_chord_vertices(
        self, 
        pitch_classes: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Get pixel coordinates for chord vertices on Tonnetz grid.
        
        Args:
            pitch_classes: List of pitch classes (0-11)
        
        Returns:
            List of (px, py) pixel coordinates
        """
        if not pitch_classes:
            return []
        
        vertices = []
        for pc in pitch_classes:
            # Get relative position to key (always 0 for C)
            interval = (pc - self.key_root) % 12
            rel_x, rel_y = TONNETZ_COORDS[interval]
            
            # Convert to absolute lattice position
            key_x, key_y = TONNETZ_COORDS[self.key_root]
            abs_x = key_x + rel_x
            abs_y = key_y + rel_y
            
            # Convert to pixels
            px, py = self.lattice_to_pixel(abs_x, abs_y)
            vertices.append((px, py))
        
        return vertices
    
    def draw_chord_polygon(
        self, 
        active_pitch_classes: List[int]
    ) -> None:
        """
        Draw polygon connecting active pitches (chord shape).
        Blue semi-transparent triangle/polygon.
        """
        if len(active_pitch_classes) < 2:
            return
        
        vertices = self.get_chord_vertices(active_pitch_classes)
        
        if len(vertices) < 2:
            return
        
        # Draw vertex markers (blue circles)
        for px, py in vertices:
            pygame.draw.circle(self.surface, self.chord_color, (px, py), 8)
            pygame.draw.circle(self.surface, (255, 255, 255), (px, py), 4)
        
        # Draw polygon for 3+ notes
        if len(vertices) >= 3:
            # Sort vertices by angle for proper polygon drawing
            centroid_x = sum(v[0] for v in vertices) / len(vertices)
            centroid_y = sum(v[1] for v in vertices) / len(vertices)
            
            def angle_from_centroid(v):
                return np.arctan2(v[1] - centroid_y, v[0] - centroid_x)
            
            sorted_vertices = sorted(vertices, key=angle_from_centroid)
            
            # Draw semi-transparent polygon
            # Create a temporary surface for alpha blending
            poly_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.polygon(
                poly_surface, 
                (*self.chord_color, 64),  # Blue with alpha=64 (~0.25 opacity)
                sorted_vertices
            )
            # Draw white edges
            pygame.draw.polygon(
                poly_surface,
                (255, 255, 255, 128),  # White edges with alpha
                sorted_vertices,
                2
            )
            self.surface.blit(poly_surface, (0, 0))
    
    def draw_info(
        self,
        harm_x: int, harm_y: int, harm_r: int,
        actual_x: float, actual_y: float, actual_r: float,
        button: int, pitch: int, note_count: int
    ) -> None:
        """Draw info text overlay."""
        y_offset = 10
        line_height = 25
        
        # Target harmony
        target_lat_x, target_lat_y = bins_to_lattice(harm_x, harm_y)
        target_tension = bin_to_tension(harm_r)
        text = self.font_medium.render(
            f"Target: X={target_lat_x:.2f} Y={target_lat_y:.2f} R={target_tension:.2f}",
            True, self.target_color
        )
        self.surface.blit(text, (10, y_offset))
        y_offset += line_height
        
        # Actual harmony
        text = self.font_medium.render(
            f"Actual: X={actual_x:.2f} Y={actual_y:.2f} R={actual_r:.2f}",
            True, self.actual_color
        )
        self.surface.blit(text, (10, y_offset))
        y_offset += line_height
        
        # Current state
        text = self.font_medium.render(
            f"Button: {button}  Pitch: {pitch}  Notes: {note_count}",
            True, self.text_color
        )
        self.surface.blit(text, (10, y_offset))
        
        # Instructions (bottom)
        instructions = [
            "Keys 1-0,-,= or Q-]: Buttons 0-11",
            "Mouse: Harmony X/Y | Pinch/Scroll/+/-: Tension",
            "P: Save | SPACE: Reset | ESC: Quit"
        ]
        y_offset = self.height - len(instructions) * 20 - 10
        for line in instructions:
            text = self.font_small.render(line, True, (150, 150, 150))
            self.surface.blit(text, (10, y_offset))
            y_offset += 20
        
        # Legend (top right)
        legend_x = self.width - 150
        text = self.font_small.render("● Target (input)", True, self.target_color)
        self.surface.blit(text, (legend_x, 10))
        text = self.font_small.render("● Actual (output)", True, self.actual_color)
        self.surface.blit(text, (legend_x, 30))


#===================================================================================================
# FLUIDSYNTH AUDIO
#===================================================================================================

def init_fluidsynth() -> fluidsynth.Synth:
    """Initialize FluidSynth for audio playback."""
    fs = fluidsynth.Synth()
    fs.start()
    sfid = fs.sfload(SOUNDFONT_PATH)
    fs.program_select(0, sfid, 0, 0)
    return fs


def play_note(fs: fluidsynth.Synth, note: int, velocity: int = 100) -> None:
    """Play or stop a note via FluidSynth."""
    if TRACES:
        print(f"FluidSynth: note={note}, velocity={velocity}")
    if velocity > 0:
        fs.noteon(0, note, velocity)
    else:
        fs.noteoff(0, note)


#===================================================================================================
# MAIN INTERACTION LOGIC
#===================================================================================================

def main():
    print("=" * 70)
    print("Monster Genie - Harmony Interaction")
    print("=" * 70)
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Monster Genie - Harmony Controller")
    clock = pygame.time.Clock()
    
    # Initialize audio
    print("Initializing FluidSynth...")
    fs = init_fluidsynth()
    
    # Load model
    print(f"Loading model: {MODEL_NAME}")
    cfg = get_model_hparams(MODEL_NAME)
    model = load_model(model_name=MODEL_NAME, cfg=cfg, set_only=True)
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    # Load seed MIDI for initial context
    print(f"Loading seed MIDI: {SEED_MIDI_PATH}")
    dict_input_tokens, num_notes = midi_to_dict(SEED_MIDI_PATH)
    
    # Initialize buffers
    pitch_buffer: List[int] = dict_input_tokens['pitch'].copy()
    # Extend if needed
    while len(pitch_buffer) < TOTAL_GEN_LEN:
        pitch_buffer.append(0)
    
    # Harmony buffers (start with neutral values: center of range)
    harm_x_buffer: List[int] = [64] * TOTAL_GEN_LEN
    harm_y_buffer: List[int] = [64] * TOTAL_GEN_LEN
    harm_r_buffer: List[int] = [64] * TOTAL_GEN_LEN
    
    # Current harmony target (from mouse/trackpad)
    current_harm_x: int = 64
    current_harm_y: int = 64
    current_harm_r: int = 64
    
    # State
    i: int = 0  # Current position after context
    current_button: int = 6  # Middle button
    last_pitch: int = 60  # Middle C
    first_note: bool = True
    time_last: float = 0.0
    note_on_dict: Dict[int, Tuple[int, float]] = {}  # key -> (pitch, time)
    
    # Harmony analyzer
    analyzer = RollingHarmonyAnalyzer(window_size=ROLLING_WINDOW_SIZE)
    # Prime with seed pitches
    for p in pitch_buffer[:CTX_LEN]:
        analyzer.add_pitch(p)
    
    # Renderer
    renderer = TonnetzRenderer(screen, grid_range=TONNETZ_GRID_RANGE)
    
    print("=" * 70)
    print("Ready! Use keyboard for buttons, mouse for harmony target.")
    print("=" * 70)
    
    running = True
    while running:
        # Get mouse position and update harmony target
        mouse_x, mouse_y = pygame.mouse.get_pos()
        disp_x, disp_y = pixel_to_display(
            mouse_x, mouse_y, WINDOW_WIDTH, WINDOW_HEIGHT, TONNETZ_GRID_RANGE
        )
        lat_x, lat_y = display_to_lattice(disp_x, disp_y)
        current_harm_x, current_harm_y = lattice_to_bins(lat_x, lat_y)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            
            elif event.type == MOUSEWHEEL:
                # Scroll wheel (or trackpad pinch gesture on macOS)
                # Positive y = scroll up = increase tension
                delta = event.y * 5  # Scale factor
                current_harm_r = int(np.clip(current_harm_r + delta, 0, N_BINS - 1))
                if TRACES:
                    print(f"Scroll: harm_r = {current_harm_r}")
            
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                
                elif event.key == K_p:
                    # Save performance
                    print("Saving performance...")
                    context_save = {
                        'dtime': dict_input_tokens['dtime'][:i + CTX_LEN + 1],
                        'pitch': pitch_buffer[:i + CTX_LEN + 1],
                        'dur': dict_input_tokens['dur'][:i + CTX_LEN + 1],
                    }
                    song_d = dict_to_song(context_save)
                    ms_SONG_to_MIDI_Converter(
                        song_d, output_file_name=OUTPUT_MIDI_NAME,
                        timings_multiplier=2
                    )
                    print(f"Saved to {OUTPUT_MIDI_NAME}.mid")
                
                elif event.key == K_SPACE:
                    # Reset context
                    print("Resetting context...")
                    i = 0
                    first_note = True
                    pitch_buffer = dict_input_tokens['pitch'].copy()
                    while len(pitch_buffer) < TOTAL_GEN_LEN:
                        pitch_buffer.append(0)
                    analyzer.clear()
                    for p in pitch_buffer[:CTX_LEN]:
                        analyzer.add_pitch(p)
                    # Clear trail
                    renderer.trail_positions.clear()
                
                elif event.key in (K_PLUS, K_KP_PLUS):
                    # Increase tension
                    current_harm_r = min(current_harm_r + 5, N_BINS - 1)
                
                elif event.key in (K_MINUS, K_KP_MINUS) and event.key not in KEY_MAPPING:
                    # Decrease tension (but not if it's mapped to button 10)
                    current_harm_r = max(current_harm_r - 5, 0)
                
                elif event.key in KEY_MAPPING:
                    # Button press -> generate note
                    button = KEY_MAPPING[event.key]
                    current_button = button
                    
                    time_new = time.perf_counter() * 1000 / 32
                    dtime = max(0, min(127, int(time_new - time_last))) if not first_note else 0
                    first_note = False
                    time_last = time_new
                    
                    # Update dtime buffer
                    if i + CTX_LEN < len(dict_input_tokens['dtime']):
                        dict_input_tokens['dtime'][i + CTX_LEN] = dtime
                    
                    # Update harmony buffers with current target
                    harm_x_buffer[i + CTX_LEN] = current_harm_x
                    harm_y_buffer[i + CTX_LEN] = current_harm_y
                    harm_r_buffer[i + CTX_LEN] = current_harm_r
                    
                    # Build context for model
                    # Encoder sees pitch[1:CTX_LEN+1] -> produces buttons for positions [0:CTX_LEN]
                    # Decoder sees pitch[0:CTX_LEN], button[0:CTX_LEN], harm[1:CTX_LEN+1]
                    
                    # Create context tensors
                    ctx_pitch = torch.tensor(
                        pitch_buffer[i:i + CTX_LEN], dtype=torch.long
                    ).unsqueeze(0).to(device)
                    
                    # Button: we use the current button for all positions (simplified)
                    # In training, buttons come from encoder. Here we use user input directly.
                    ctx_button = torch.full(
                        (1, CTX_LEN), float(button), dtype=torch.float
                    ).to(device)
                    
                    ctx_harm_x = torch.tensor(
                        harm_x_buffer[i + 1:i + CTX_LEN + 1], dtype=torch.long
                    ).unsqueeze(0).to(device)
                    ctx_harm_y = torch.tensor(
                        harm_y_buffer[i + 1:i + CTX_LEN + 1], dtype=torch.long
                    ).unsqueeze(0).to(device)
                    ctx_harm_r = torch.tensor(
                        harm_r_buffer[i + 1:i + CTX_LEN + 1], dtype=torch.long
                    ).unsqueeze(0).to(device)
                    
                    decoder_context = {
                        'pitch': ctx_pitch,
                        'button': ctx_button,
                        'harm_x': ctx_harm_x,
                        'harm_y': ctx_harm_y,
                        'harm_r': ctx_harm_r,
                    }
                    
                    # Generate next pitch
                    with torch.inference_mode():
                        logits = model.decoder(decoder_context)
                        logits_last = logits[0, -1, :]  # Last position
                        
                        # Sample with temperature
                        if TEMPERATURE > 0:
                            probs = torch.softmax(logits_last / TEMPERATURE, dim=-1)
                            new_pitch = torch.multinomial(probs, 1).item()
                        else:
                            new_pitch = torch.argmax(logits_last).item()
                    
                    # Store and play
                    pitch_buffer[i + CTX_LEN] = new_pitch
                    last_pitch = new_pitch
                    play_note(fs, new_pitch, 100)
                    
                    # Update analyzer
                    analyzer.add_pitch(new_pitch)
                    
                    # Track note-on
                    note_on_dict[event.key] = (new_pitch, time_new)
                    
                    i += 1
                    
                    if TRACES:
                        print(f"Generated: pitch={new_pitch}, button={button}, "
                              f"harm=({current_harm_x}, {current_harm_y}, {current_harm_r})")
            
            elif event.type == KEYUP:
                if event.key in KEY_MAPPING and event.key in note_on_dict:
                    # Note off
                    pitch, _ = note_on_dict[event.key]
                    del note_on_dict[event.key]
                    play_note(fs, pitch, 0)
                    # Remove from analyzer's active pitches
                    analyzer.remove_pitch(pitch)
        
        # Get actual harmony from analyzer
        actual_x, actual_y, actual_r = analyzer.get_tonnetz_position()
        active_pcs = analyzer.get_active_pitch_classes()
        
        # Render (order matters for layering)
        renderer.draw_grid()
        renderer.draw_trail()  # Draw trail first (background)
        renderer.draw_chord_polygon(active_pcs)  # Draw chord polygon
        renderer.draw_target_marker(current_harm_x, current_harm_y, current_harm_r)
        renderer.draw_actual_marker(actual_x, actual_y, actual_r)
        renderer.draw_info(
            current_harm_x, current_harm_y, current_harm_r,
            actual_x, actual_y, actual_r,
            current_button, last_pitch, i
        )
        
        pygame.display.flip()
        clock.tick(60)  # 60 FPS
    
    # Cleanup
    print("Shutting down...")
    pygame.quit()
    fs.delete()
    print("Done.")


if __name__ == '__main__':
    main()
