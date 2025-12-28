"""
Tonnetz Visualizer: Animated 2D harmonic space visualization with MIDI playback.

This module provides an animated visualization of harmonic tension trajectories
on a Tonnetz (tone network) grid, showing how the music moves through harmonic space
relative to the detected local key. Optionally plays the MIDI file in sync.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
from typing import List, Tuple, Optional
import os
import time

# Try to import pygame for MIDI playback
try:
    import pygame
    import pygame.midi
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not installed. MIDI playback disabled.")
    print("Install with: pip install pygame")


# Pitch class names for grid labels
PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Tonnetz coordinate lookup (same as in harmony_extractor.py)
TONNETZ_COORDS = [
    (0, 0),    # 0: Unison/Root
    (-1, 2),   # 1: Minor second
    (2, 0),    # 2: Major second
    (1, -1),   # 3: Minor third
    (0, 1),    # 4: Major third
    (-1, 0),   # 5: Perfect fourth
    (2, -2),   # 6: Tritone
    (1, 0),    # 7: Perfect fifth
    (0, -1),   # 8: Minor sixth
    (-1, 1),   # 9: Major sixth
    (-2, 0),   # 10: Minor seventh
    (1, 1),    # 11: Major seventh
]

# --- Tonnetz display geometry (triangular lattice) ---
_SQRT3 = float(np.sqrt(3.0))


def lattice_to_display(x: float, y: float) -> Tuple[float, float]:
    """
    Map Tonnetz lattice coords (x=fifths, y=thirds) into 2D display coords.
    This makes the grid a proper triangular/hex lattice (like standard Tonnetz diagrams).
    
    The transform creates a rhombic/triangular layout where:
    - Moving right (+x) = moving along the circle of fifths
    - Moving up-left (+y) = moving along major thirds
    - Major triads form upward triangles, minor triads form downward triangles
    """
    X = x + 0.5 * y
    Y = -(_SQRT3 / 2.0) * y
    return (X, Y)


def format_key_name(key_root: int, key_mode: str) -> str:
    """Formats a key as a readable string."""
    return f"{PITCH_NAMES[key_root]} {key_mode}"


def get_pitch_at_tonnetz_coord(x: int, y: int, key_root: int) -> str:
    """
    Get the pitch name at a given Tonnetz grid coordinate relative to a key.
    
    The Tonnetz satisfies: semitone = (7*x + 4*y) mod 12
    """
    semitone = (7 * x + 4 * y) % 12
    absolute_pitch = (key_root + semitone) % 12
    return PITCH_NAMES[absolute_pitch]


def get_chord_vertices(pitch_classes: List[int], key_root: int) -> np.ndarray:
    """
    Get the Tonnetz coordinates for a list of pitch classes.
    
    Args:
        pitch_classes: List of absolute pitch classes (0-11)
        key_root: The key root for computing relative positions
    
    Returns:
        Array of shape (N, 2) with [x, y] absolute Tonnetz coordinates
    """
    if not pitch_classes:
        return np.array([])
    
    # Get key position on Tonnetz
    key_x, key_y = TONNETZ_COORDS[key_root]
    
    vertices = []
    for pc in pitch_classes:
        # Get relative position to key
        interval = (pc - key_root) % 12
        rel_x, rel_y = TONNETZ_COORDS[interval]
        # Convert to absolute position
        abs_x = key_x + rel_x
        abs_y = key_y + rel_y
        vertices.append([abs_x, abs_y])
    
    return np.array(vertices)


class MidiPlayer:
    """Handles MIDI playback using pygame."""
    
    def __init__(self, midi_path: str):
        self.midi_path = midi_path
        self.is_playing = False
        self.start_time = None
        
        if not PYGAME_AVAILABLE:
            print("MIDI playback not available (pygame not installed)")
            return
            
        # Initialize pygame mixer for MIDI
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        
    def start(self):
        """Start MIDI playback."""
        if not PYGAME_AVAILABLE or not self.midi_path:
            return
            
        try:
            pygame.mixer.music.load(self.midi_path)
            pygame.mixer.music.play()
            self.start_time = time.time()
            self.is_playing = True
            print(f"Playing: {self.midi_path}")
        except Exception as e:
            print(f"Error starting MIDI playback: {e}")
            self.is_playing = False
    
    def stop(self):
        """Stop MIDI playback."""
        if not PYGAME_AVAILABLE:
            return
            
        try:
            pygame.mixer.music.stop()
            self.is_playing = False
        except Exception:
            pass
    
    def get_current_time(self) -> float:
        """Get current playback position in seconds."""
        if not PYGAME_AVAILABLE or not self.is_playing or self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def cleanup(self):
        """Clean up pygame resources."""
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.quit()
            except Exception:
                pass


class TonnetzVisualizer:
    """
    Animated Tonnetz grid visualizer for harmonic tension trajectories.
    Supports synchronized MIDI playback.
    """
    
    def __init__(
        self,
        tension_xy: np.ndarray,
        global_key: Tuple[int, str],
        fs: int,
        title: str = "Tonnetz Harmonic Trajectory",
        trail_length: int = 50,
        grid_range: Tuple[int, int] = (-3, 3),
        midi_path: Optional[str] = None,
        active_pitches: Optional[List[List[int]]] = None,
        tension_magnitude: Optional[np.ndarray] = None,
        mouse_indicator_enabled: bool = False,
        mouse_indicator_spread: float = 0.3,
    ):
        """
        Initialize the Tonnetz visualizer.
        
        Args:
            tension_xy: Array of shape (N, 2) with [x, y] Tonnetz coordinates (relative to key)
            global_key: Single (root, mode) tuple representing the piece's key
            fs: Sampling frequency (frames per second in the data)
            title: Plot title
            trail_length: Number of past positions to show in the trail
            grid_range: (min, max) range for both axes of the grid
            midi_path: Optional path to MIDI file for synchronized playback
            active_pitches: Optional list of active pitch classes per frame for chord shapes
            tension_magnitude: Optional array of tension magnitudes (mean distance from tonal center)
        """
        self.tension_xy = tension_xy
        self.tension_magnitude = tension_magnitude
        self.global_key = global_key
        self.key_root = global_key[0]
        self.key_mode = global_key[1]
        self.fs = fs
        self.title = title
        self.trail_length = trail_length
        self.grid_min, self.grid_max = grid_range
        self.midi_path = midi_path
        self.active_pitches = active_pitches
        # Mouse indicator overlay
        self.mouse_indicator_enabled = mouse_indicator_enabled
        self.mouse_indicator_spread = mouse_indicator_spread
        self.mouse_pos = None
        
        self.n_frames = len(tension_xy)
        self.fig = None
        self.ax = None
        self.anim = None
        
        # MIDI playback
        self.midi_player = None
        self.playback_started = False
        
        # Plot elements to update
        self.current_point = None
        self.tension_circle = None  # Circle showing tension magnitude
        self.trail_line = None
        self.key_text = None
        self.time_text = None
        self.chord_polygon = None
        self.chord_points = None
        self.mouse_indicator = None
    
    def _on_mouse_move(self, event):
        """Update mouse indicator position when mouse moves over the axes."""
        if not self.mouse_indicator_enabled or self.mouse_indicator is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.mouse_pos = (event.xdata, event.ydata)
        self.mouse_indicator.center = self.mouse_pos
        self.mouse_indicator.set_visible(True)
        self.fig.canvas.draw_idle()
    
    def _on_key_press(self, event):
        """Adjust mouse indicator spread with keyboard (+/-)."""
        if not self.mouse_indicator_enabled or self.mouse_indicator is None:
            return
        if event.key in ['+', '=']:
            self.mouse_indicator_spread *= 1.1
        elif event.key == '-':
            self.mouse_indicator_spread /= 1.1
        # Clamp spread to a small positive range
        self.mouse_indicator_spread = max(0.01, min(self.mouse_indicator_spread, 10.0))
        self.mouse_indicator.set_radius(self.mouse_indicator_spread)
        self.fig.canvas.draw_idle()
        
    def _setup_figure(self):
        """Create the figure and axes with styling."""
        # Dark theme with warm accents
        plt.style.use('dark_background')
        
        self.fig, self.ax = plt.subplots(figsize=(12, 10), facecolor='#1a1a2e')
        self.ax.set_facecolor('#16213e')
        
        # Compute limits in DISPLAY space (skewed triangular lattice)
        corners = [
            lattice_to_display(self.grid_min, self.grid_min),
            lattice_to_display(self.grid_min, self.grid_max),
            lattice_to_display(self.grid_max, self.grid_min),
            lattice_to_display(self.grid_max, self.grid_max),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        padding = 0.8
        self.ax.set_xlim(min(xs) - padding, max(xs) + padding)
        self.ax.set_ylim(min(ys) - padding, max(ys) + padding)
        
        # Labels
        self.ax.set_xlabel('Circle of Fifths (← Subdominant | Dominant →)', 
                          fontsize=12, color='#e8e8e8', fontfamily='monospace')
        self.ax.set_ylabel('Major Thirds (↓ Minor | Major ↑)', 
                          fontsize=12, color='#e8e8e8', fontfamily='monospace')
        self.ax.set_title(self.title, fontsize=14, color='#f8f8f8', 
                         fontfamily='monospace', pad=20)
        
        # Grid styling - disable orthogonal grid (we draw our own lattice lines)
        self.ax.set_aspect('equal')
        self.ax.grid(False)
        
        # Tick styling  
        self.ax.tick_params(colors='#a0a0a0', labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color('#4a4a6a')
        
        # Mouse indicator (optional)
        if self.mouse_indicator_enabled:
            self.mouse_indicator = Circle(
                (0.0, 0.0),
                radius=self.mouse_indicator_spread,
                color='magenta',
                alpha=0.3,
                visible=False,
                zorder=3
            )
            self.ax.add_patch(self.mouse_indicator)
            # Connect mouse and keyboard handlers
            self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            
    def _draw_tonnetz_grid(self):
        """Draw the static Tonnetz grid with pitch labels (C at center/origin)."""
        # The grid is always fixed with C at the origin (0,0)
        # This is drawn once and never changes
        
        # First, draw the lattice lines (triangular grid)
        line_color = '#3a3a5a'
        line_alpha = 0.3
        
        # Draw lines along the three axes of the triangular lattice:
        # 1. Horizontal lines (constant y in lattice = fifths axis)
        for y in range(self.grid_min, self.grid_max + 1):
            x1, y1 = lattice_to_display(self.grid_min, y)
            x2, y2 = lattice_to_display(self.grid_max, y)
            self.ax.plot([x1, x2], [y1, y2], '-', color=line_color, 
                        alpha=line_alpha, linewidth=0.5, zorder=1)
        
        # 2. Vertical lines in lattice (constant x = thirds axis, appears diagonal)
        for x in range(self.grid_min, self.grid_max + 1):
            x1, y1 = lattice_to_display(x, self.grid_min)
            x2, y2 = lattice_to_display(x, self.grid_max)
            self.ax.plot([x1, x2], [y1, y2], '-', color=line_color, 
                        alpha=line_alpha, linewidth=0.5, zorder=1)
        
        # 3. Diagonal lines (x + y = constant, minor third axis)
        for d in range(2 * self.grid_min, 2 * self.grid_max + 1):
            points = []
            for x in range(self.grid_min, self.grid_max + 1):
                y = d - x
                if self.grid_min <= y <= self.grid_max:
                    points.append(lattice_to_display(x, y))
            if len(points) >= 2:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                self.ax.plot(xs, ys, '-', color=line_color, 
                            alpha=line_alpha, linewidth=0.5, zorder=1)
        
        # Now draw the pitch nodes
        for x in range(self.grid_min, self.grid_max + 1):
            for y in range(self.grid_min, self.grid_max + 1):
                # Always use C (key_root=0) as the reference
                pitch_name = get_pitch_at_tonnetz_coord(x, y, key_root=0)
                
                # Transform to display coordinates
                X, Y = lattice_to_display(float(x), float(y))
                
                # Highlight based on position in the grid
                if x == 0 and y == 0:
                    color = '#ffd700'  # Gold for C (tonic/center)
                    fontweight = 'bold'
                    fontsize = 11
                elif x == 1 and y == 0:
                    color = '#ff6b6b'  # Red for G (dominant of C)
                    fontweight = 'normal'
                    fontsize = 10
                elif x == -1 and y == 0:
                    color = '#4ecdc4'  # Teal for F (subdominant of C)
                    fontweight = 'normal'
                    fontsize = 10
                elif y == 1:
                    color = '#95e1d3'  # Light green for major third region
                    fontweight = 'normal'
                    fontsize = 9
                elif y == -1:
                    color = '#f38181'  # Coral for minor region
                    fontweight = 'normal'
                    fontsize = 9
                else:
                    color = '#7a7a9a'
                    fontweight = 'normal'
                    fontsize = 9
                
                # Draw small circle at grid point (in display coords)
                circle = Circle((X, Y), 0.08, color=color, alpha=0.3)
                self.ax.add_patch(circle)
                
                # Draw pitch label (in display coords)
                self.ax.text(
                    X, Y + 0.18, pitch_name,
                    ha='center', va='bottom',
                    fontsize=fontsize, fontweight=fontweight,
                    color=color, fontfamily='monospace'
                )
                
    def _init_animation(self):
        """Initialize animation elements."""
        # Current position marker (larger, glowing effect)
        self.current_point, = self.ax.plot(
            [], [], 'o', markersize=18, 
            color='#ff9f43', markeredgecolor='#fff', 
            markeredgewidth=2, zorder=10
        )
        
        # Inner point for glow effect
        self.inner_point, = self.ax.plot(
            [], [], 'o', markersize=8,
            color='#fff', zorder=11
        )
        
        # Tension magnitude circle (shows true tension even when notes cancel)
        # Uses matplotlib Circle patch for dynamic radius
        from matplotlib.patches import Circle
        self.tension_circle = Circle(
            (0, 0), 0.0,  # Will be updated with key position and radius
            fill=False, edgecolor='#ff6b6b', linewidth=2.5,
            linestyle='--', alpha=0.8, zorder=6
        )
        self.ax.add_patch(self.tension_circle)
        
        # Trail line showing recent trajectory
        self.trail_line, = self.ax.plot(
            [], [], '-', linewidth=2, color='#ff9f43', 
            alpha=0.6, zorder=5
        )
        
        # Trail scatter for fading effect
        self.trail_scatter = self.ax.scatter(
            [], [], s=[], c=[], cmap='YlOrRd', 
            alpha=0.7, zorder=4
        )
        
        # Chord shape polygon (triangle/polygon connecting active notes)
        self.chord_polygon = None  # Will be created dynamically
        
        # Chord vertex markers
        self.chord_points = self.ax.scatter(
            [], [], s=120, c='#74b9ff', 
            marker='o', edgecolors='#fff', linewidths=1.5,
            alpha=0.9, zorder=8
        )
        
        # Harmonic region display
        self.chord_text = self.ax.text(
            0.98, 0.02, '', transform=self.ax.transAxes,
            fontsize=11, color='#74b9ff', fontfamily='monospace',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', 
                     edgecolor='#74b9ff', alpha=0.8)
        )
        
        # Key display text
        self.key_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            fontsize=14, color='#ffd700', fontfamily='monospace',
            fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', 
                     edgecolor='#ffd700', alpha=0.8)
        )
        
        # Time display
        self.time_text = self.ax.text(
            0.98, 0.98, '', transform=self.ax.transAxes,
            fontsize=11, color='#a0a0a0', fontfamily='monospace',
            ha='right', va='top'
        )
        
        # Coordinates display
        self.coord_text = self.ax.text(
            0.02, 0.02, '', transform=self.ax.transAxes,
            fontsize=10, color='#7a7a9a', fontfamily='monospace',
            va='bottom'
        )
        
        # Playback status indicator
        self.playback_text = self.ax.text(
            0.5, 0.02, '', transform=self.ax.transAxes,
            fontsize=10, color='#4ecdc4', fontfamily='monospace',
            ha='center', va='bottom'
        )
        
        # Draw the static Tonnetz grid (only once, C at center)
        self._draw_tonnetz_grid()
        
        # Start MIDI playback if available
        if self.midi_player and not self.playback_started:
            self.midi_player.start()
            self.playback_started = True
        
        return (self.current_point, self.inner_point, self.trail_line, 
                self.key_text, self.time_text, self.coord_text)
    
    def _get_current_frame(self) -> int:
        """Get the current frame based on MIDI playback time or animation frame."""
        if self.midi_player and self.midi_player.is_playing:
            # Sync to MIDI playback time
            current_time = self.midi_player.get_current_time()
            frame = int(current_time * self.fs)
            return min(frame, self.n_frames - 1)
        return None  # Use default frame counter
    
    def _update_animation(self, frame: int):
        """Update animation for a single frame."""
        # Sync to MIDI playback if available
        synced_frame = self._get_current_frame()
        if synced_frame is not None:
            frame = synced_frame
        
        # Clamp frame to valid range
        frame = max(0, min(frame, self.n_frames - 1))
        
        # Get relative tension coordinates (in lattice space)
        rel_x, rel_y = self.tension_xy[frame]
        
        # Convert to absolute lattice position: key_position + relative_offset
        key_x, key_y = TONNETZ_COORDS[self.key_root]
        lat_x = key_x + rel_x
        lat_y = key_y + rel_y
        
        # Transform to display coordinates
        X, Y = lattice_to_display(float(lat_x), float(lat_y))
        
        # Update current position (in display coords)
        self.current_point.set_data([X], [Y])
        self.inner_point.set_data([X], [Y])
        
        # Update tension magnitude circle
        # The circle is centered on the KEY position (not the centroid)
        # Its radius represents the true tension (mean distance from key)
        if self.tension_magnitude is not None and frame < len(self.tension_magnitude):
            # Get key position in display coordinates
            key_disp_x, key_disp_y = lattice_to_display(float(key_x), float(key_y))
            
            # Tension magnitude needs to be scaled to display space
            # We use the same scale factor as the lattice transform (approximately 1)
            mag = float(self.tension_magnitude[frame])
            
            self.tension_circle.set_center((key_disp_x, key_disp_y))
            self.tension_circle.set_radius(mag)
            self.tension_circle.set_visible(True)
        else:
            self.tension_circle.set_visible(False)
        
        # Update trail (last N positions)
        trail_start = max(0, frame - self.trail_length)
        trail_frames = range(trail_start, frame + 1)
        
        if len(trail_frames) > 1:
            # Convert each trail point from relative to absolute lattice coords
            key_pos = TONNETZ_COORDS[self.key_root]
            trail_rel = self.tension_xy[trail_start:frame + 1]
            trail_lattice = trail_rel + np.array([key_pos[0], key_pos[1]])
            
            # Transform to display coordinates
            trail_disp = np.array([lattice_to_display(p[0], p[1]) for p in trail_lattice])
            
            self.trail_line.set_data(trail_disp[:, 0], trail_disp[:, 1])
            
            # Scatter with size/alpha gradient for fading effect
            sizes = np.linspace(5, 40, len(trail_disp))
            colors = np.linspace(0.2, 1.0, len(trail_disp))
            self.trail_scatter.set_offsets(trail_disp)
            self.trail_scatter.set_sizes(sizes)
            self.trail_scatter.set_array(colors)
        
        # Update chord shape polygon
        if self.active_pitches is not None and frame < len(self.active_pitches):
            pitches = self.active_pitches[frame]
            
            # Remove old polygon if exists
            if self.chord_polygon is not None:
                self.chord_polygon.remove()
                self.chord_polygon = None
            
            if len(pitches) >= 2:
                # Get chord vertices in lattice coords
                vertices_lattice = get_chord_vertices(pitches, self.key_root)
                
                # Transform to display coordinates
                verts_disp = np.array([lattice_to_display(v[0], v[1]) for v in vertices_lattice])
                
                # Update chord points scatter (in display coords)
                self.chord_points.set_offsets(verts_disp)
                
                if len(pitches) >= 3:
                    # Draw polygon for chords with 3+ notes
                    # Sort vertices by angle for proper polygon drawing
                    centroid = verts_disp.mean(axis=0)
                    angles = np.arctan2(verts_disp[:, 1] - centroid[1], 
                                       verts_disp[:, 0] - centroid[0])
                    sorted_idx = np.argsort(angles)
                    sorted_vertices = verts_disp[sorted_idx]
                    
                    self.chord_polygon = Polygon(
                        sorted_vertices,
                        fill=True,
                        facecolor='#74b9ff',
                        edgecolor='#fff',
                        alpha=0.25,
                        linewidth=2,
                        zorder=3
                    )
                    self.ax.add_patch(self.chord_polygon)
                
                # Show harmonic region notes
                note_names = [PITCH_NAMES[p] for p in pitches]
                self.chord_text.set_text(f"Region ({len(pitches)}): {', '.join(note_names)}")
            else:
                self.chord_points.set_offsets(np.empty((0, 2)))
                self.chord_text.set_text("")
        
        # Update text displays
        key_name = format_key_name(self.key_root, self.key_mode)
        self.key_text.set_text(f"Key: {key_name}")
        
        time_sec = frame / self.fs
        total_time = self.n_frames / self.fs
        self.time_text.set_text(f"Time: {time_sec:.1f}s / {total_time:.1f}s")
        
        # Show relative tension, magnitude, and lattice position
        if self.tension_magnitude is not None and frame < len(self.tension_magnitude):
            mag = float(self.tension_magnitude[frame])
            self.coord_text.set_text(f"Centroid: ({rel_x:.2f}, {rel_y:.2f}) | Mag: {mag:.2f} | Lattice: ({lat_x:.2f}, {lat_y:.2f})")
        else:
            self.coord_text.set_text(f"Centroid: ({rel_x:.2f}, {rel_y:.2f}) | Lattice: ({lat_x:.2f}, {lat_y:.2f})")
        
        # Update playback status
        if self.midi_player and self.midi_player.is_playing:
            self.playback_text.set_text("♪ Playing MIDI")
        else:
            self.playback_text.set_text("")
        
        return (self.current_point, self.inner_point, self.trail_line,
                self.trail_scatter, self.key_text, self.time_text, self.coord_text)
    
    def animate(
        self, 
        interval: Optional[int] = None,
        save_path: Optional[str] = None,
        fps: int = 30,
        play_midi: bool = True
    ):
        """
        Run the animation with optional MIDI playback.
        
        Args:
            interval: Milliseconds between frames (default: auto from fs)
            save_path: If provided, save animation to this path (mp4 or gif)
            fps: Frames per second for saved animation
            play_midi: If True and midi_path was provided, play MIDI in sync
        """
        self._setup_figure()
        
        # Initialize MIDI player if requested
        if play_midi and self.midi_path and PYGAME_AVAILABLE:
            self.midi_player = MidiPlayer(self.midi_path)
            print(f"MIDI playback enabled: {self.midi_path}")
        
        # Calculate interval from sampling frequency if not provided
        if interval is None:
            interval = max(10, int(1000 / self.fs))  # At least 10ms
        
        # Use a generator for frames when syncing to MIDI
        def frame_generator():
            frame = 0
            while frame < self.n_frames:
                yield frame
                frame += 1
        
        self.anim = FuncAnimation(
            self.fig,
            self._update_animation,
            init_func=self._init_animation,
            frames=frame_generator,
            interval=interval,
            blit=False,  # blit=False for grid redrawing
            repeat=False,  # Don't repeat when using MIDI sync
            cache_frame_data=False
        )
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            # Disable MIDI playback when saving
            if self.midi_player:
                self.midi_player.stop()
                self.midi_player = None
            if save_path.endswith('.gif'):
                self.anim.save(save_path, writer='pillow', fps=fps)
            else:
                self.anim.save(save_path, writer='ffmpeg', fps=fps)
            print("Animation saved!")
        else:
            plt.tight_layout()
            try:
                plt.show()
            finally:
                # Clean up MIDI player when window closes
                if self.midi_player:
                    self.midi_player.stop()
                    self.midi_player.cleanup()
            
    def plot_static_trajectory(self, save_path: Optional[str] = None):
        """
        Create a static plot showing the full trajectory.
        
        Args:
            save_path: If provided, save figure to this path
        """
        self._setup_figure()
        
        # Draw the static Tonnetz grid (C at center)
        self._draw_tonnetz_grid()
        
        # Convert all relative coordinates to absolute lattice coords
        key_pos = TONNETZ_COORDS[self.key_root]
        lattice_coords = self.tension_xy + np.array([key_pos[0], key_pos[1]])
        
        # Transform to display coordinates
        disp_coords = np.array([lattice_to_display(p[0], p[1]) for p in lattice_coords])
        n_points = len(disp_coords)
        
        # Plot full trajectory with color gradient
        colors = plt.cm.YlOrRd(np.linspace(0.3, 1.0, n_points))
        
        # Line segments (in display coords)
        for i in range(n_points - 1):
            self.ax.plot(
                disp_coords[i:i+2, 0],
                disp_coords[i:i+2, 1],
                '-', color=colors[i], linewidth=1, alpha=0.6
            )
        
        # Scatter points (in display coords)
        scatter = self.ax.scatter(
            disp_coords[:, 0], 
            disp_coords[:, 1],
            c=np.arange(n_points), cmap='YlOrRd',
            s=10, alpha=0.7, zorder=5
        )
        
        # Start and end markers (in display coords)
        self.ax.plot(
            disp_coords[0, 0], disp_coords[0, 1],
            'o', markersize=15, color='#4ecdc4', 
            markeredgecolor='#fff', markeredgewidth=2,
            label='Start', zorder=10
        )
        self.ax.plot(
            disp_coords[-1, 0], disp_coords[-1, 1],
            's', markersize=15, color='#ff6b6b',
            markeredgecolor='#fff', markeredgewidth=2,
            label='End', zorder=10
        )
        
        # Colorbar for time
        cbar = plt.colorbar(scatter, ax=self.ax, shrink=0.6, pad=0.02)
        cbar.set_label('Time (frames)', color='#a0a0a0', fontfamily='monospace')
        cbar.ax.yaxis.set_tick_params(color='#a0a0a0')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#a0a0a0')
        
        self.ax.legend(loc='upper right', facecolor='#1a1a2e', 
                       edgecolor='#4a4a6a', labelcolor='#e8e8e8')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#1a1a2e')
            print(f"Static trajectory saved to {save_path}")
        else:
            plt.show()


def visualize_tonnetz(
    tension_xy: np.ndarray,
    global_key: Tuple[int, str],
    fs: int,
    title: str = "Tonnetz Harmonic Trajectory",
    animated: bool = True,
    save_path: Optional[str] = None,
    trail_length: int = 50,
    midi_path: Optional[str] = None,
    play_midi: bool = True,
    active_pitches: Optional[List[List[int]]] = None,
    tension_magnitude: Optional[np.ndarray] = None,
    mouse_indicator_enabled: bool = False,
    mouse_indicator_spread: float = 0.3,
):
    """
    Main function to visualize Tonnetz harmonic trajectory with optional MIDI playback.
    
    Args:
        tension_xy: Array of shape (N, 2) with [x, y] Tonnetz coordinates (relative to key)
        global_key: Single (root, mode) tuple for the piece's key
        fs: Sampling frequency
        title: Plot title
        animated: If True, show animated visualization; else show static trajectory
        save_path: If provided, save to this path
        trail_length: Number of past positions to show in trail (animated only)
        midi_path: Optional path to MIDI file for synchronized playback
        active_pitches: Optional list of active pitch classes per frame for chord shapes
        play_midi: If True and midi_path provided, play MIDI during animation
        tension_magnitude: Optional array of tension magnitudes (mean distance from key)
    """
    viz = TonnetzVisualizer(
        tension_xy=tension_xy,
        global_key=global_key,
        fs=fs,
        title=title,
        trail_length=trail_length,
        midi_path=midi_path,
        active_pitches=active_pitches,
        tension_magnitude=tension_magnitude,
        mouse_indicator_enabled=mouse_indicator_enabled,
        mouse_indicator_spread=mouse_indicator_spread
    )
    
    if animated:
        viz.animate(save_path=save_path, play_midi=play_midi)
    else:
        viz.plot_static_trajectory(save_path=save_path)


# --- DEMO / TEST ---
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # If a MIDI file is provided, use it for playback demo
        midi_file = sys.argv[1]
        print(f"Demo with MIDI file: {midi_file}")
        print("Note: Run harmony_extractor.py for full analysis + visualization")
    else:
        # Generate synthetic test data (circular motion with noise)
        print("Running Tonnetz Visualizer demo with synthetic data...")
        print("Tip: Pass a MIDI file as argument to test MIDI playback")
        
        n_frames = 500
        fs = 10
        
        t = np.linspace(0, 4 * np.pi, n_frames)
        # Simulate relative tension moving around the tonal center
        x = 1.5 * np.sin(t) + 0.3 * np.random.randn(n_frames)
        y = 1.2 * np.cos(t) + 0.3 * np.random.randn(n_frames)
        
        tension_xy = np.column_stack([x, y])
        
        # Single global key for the piece (C major)
        global_key = (0, 'major')
        
        visualize_tonnetz(
            tension_xy, global_key, fs, 
            title="Tonnetz Demo (Synthetic Data in C major)",
            play_midi=False  # No MIDI for synthetic demo
        )

