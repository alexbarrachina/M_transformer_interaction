"""Joker performance with fixed chord suggestions for each primer.

Every MIDI key and Space generate joker notes. Hold number keys to condition
on the primer's fixed chords: 1 tonic, 2 predominant, 4 subdominant,
5 dominant, 6 submediant, 7 subtonic. Shift+2 selects V/V; Satie also has
Shift+5 for a major dominant. The last pressed chord key wins; releasing
all chord keys returns to free generation. Chords take effect immediately.
The visualizer shows chord-note names and highlights the active pitch classes.
R resets; S saves; Escape exits. The model receives no key.
Optional --midi-chords mappings reserve selected MIDI notes for chord controls.
"""

import argparse
import datetime
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from harmony.chord_joker import Chord, ChordControls
from midiUtils import dict_to_song, midi_to_dict, ms_SONG_to_MIDI_Converter
from model_loader import load_model
from models import get_model_hparams


# Get sample seed MIDI path.
sample_midi_path1 = './samples/Bach_Prelude_and_Fugue_in_C_major.mid'
sample_midi_path2 = './samples/clairTester_to_end.midi'
sample_midi_path3 = './samples/Chopin_Nocturnes_Op9No1_In_B_Flat_Minor.mid'
sample_midi_path4 = './samples/Scott_Cyril_Lotus_Land.mid'
sample_midi_path5 = './samples/Satie_Gymnopedie_No1.mid'

sample_midi_path_init = sample_midi_path3
sample_midi_paths = [sample_midi_path1, sample_midi_path2, sample_midi_path3,
                     sample_midi_path4, sample_midi_path5]

# Core chords follow interaction_buttons_style_harm_fixed.py and docs/pieces.
# Subtonic and secondary-dominant entries extend the performance palette;
# bVII in major is borrowed, rather than a claim about the original score.
FIXED_CHORDS: Dict[int, Dict[str, Chord]] = {
    0: {  # Bach Prelude in C major
        '1': Chord('C (I)', 0, (0, 4, 7)),
        '2': Chord('Dm7 (ii7)', 2, (0, 3, 7, 10)),
        '4': Chord('F (IV)', 5, (0, 4, 7)),
        '5': Chord('G7 (V7)', 7, (0, 4, 7, 10)),
        '6': Chord('Am (vi)', 9, (0, 3, 7)),
        '7': Chord('Bb (bVII)', 10, (0, 4, 7)),
        'shift+2': Chord('D7 (V/V)', 2, (0, 4, 7, 10)),
    },
    1: {  # Debussy Clair de lune in Db major
        '1': Chord('Db (I)', 1, (0, 4, 7)),
        '2': Chord('Ebm7 (ii7)', 3, (0, 3, 7, 10)),
        '4': Chord('Gbmaj7 (IVmaj7)', 6, (0, 4, 7, 11)),
        '5': Chord('Ab7 (V7)', 8, (0, 4, 7, 10)),
        '6': Chord('Bbm (vi)', 10, (0, 3, 7)),
        '7': Chord('Cb (bVII)', 11, (0, 4, 7)),
        'shift+2': Chord('Eb7 (V/V)', 3, (0, 4, 7, 10)),
    },
    2: {  # Chopin Nocturne Op.9 No.1 in Bb minor (the docs cover No.2)
        '1': Chord('Bbm (i)', 10, (0, 3, 7)),
        '2': Chord('Cm7b5 (ii half-dim7)', 0, (0, 3, 6, 10)),
        '4': Chord('Ebm (iv)', 3, (0, 3, 7)),
        '5': Chord('F7 (V7)', 5, (0, 4, 7, 10)),
        '6': Chord('Gb (VI)', 6, (0, 4, 7)),
        '7': Chord('Ab (VII)', 8, (0, 4, 7)),
        'shift+2': Chord('C7 (V/V)', 0, (0, 4, 7, 10)),
    },
    3: {  # Cyril Scott Lotus Land in Eb minor
        '1': Chord('Ebm7 (i7)', 3, (0, 3, 7, 10)),
        '2': Chord('Fm7b5 (ii half-dim7)', 5, (0, 3, 6, 10)),
        '4': Chord('Abm7 (iv7)', 8, (0, 3, 7, 10)),
        '5': Chord('Bb7 (V7)', 10, (0, 4, 7, 10)),
        '6': Chord('Cbmaj7 (VImaj7)', 11, (0, 4, 7, 11)),
        '7': Chord('Db (VII)', 1, (0, 4, 7)),
        'shift+2': Chord('F7 (V/V)', 5, (0, 4, 7, 10)),
    },
    4: {  # Satie Gymnopedie No.1 in D major, retaining its modal v7
        '1': Chord('Dmaj7 (Imaj7)', 2, (0, 4, 7, 11)),
        '2': Chord('Em7 (ii7)', 4, (0, 3, 7, 10)),
        '4': Chord('Gmaj7 (IVmaj7)', 7, (0, 4, 7, 11)),
        '5': Chord('Am7 (v7)', 9, (0, 3, 7, 10)),
        '6': Chord('Bm7 (vi7)', 11, (0, 3, 7, 10)),
        '7': Chord('Cmaj7 (bVIImaj7)', 0, (0, 4, 7, 11)),
        'shift+2': Chord('E7 (V/V)', 4, (0, 4, 7, 10)),
        'shift+5': Chord('A7 (V7)', 9, (0, 4, 7, 10)),
    },
}


def read_primer(path: str, length: int) -> Dict[str, List[int]]:
    tokens, _ = midi_to_dict(path)
    output: Dict[str, List[int]] = {key: [] for key in ('pitch', 'dtime', 'dur', 'vel', 'chan')}
    absolute, previous = 0, 0
    for index, pitch in enumerate(tokens['pitch']):
        absolute += tokens['dtime'][index]
        # Filter annotations before using the primer as musical history.
        if tokens['chan'][index] in (3, 4, 9):
            continue
        output['dtime'].append(max(0, absolute - previous) if output['pitch'] else 0)
        for key in ('pitch', 'dur', 'vel', 'chan'):
            output[key].append(tokens[key][index])
        previous = absolute
        if len(output['pitch']) == length:
            break
    if not output['pitch']:
        raise ValueError('The primer must contain performed notes outside annotation channels 4 and 5')
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='D_Chord_Joker_unseen_onset_v1')
    parser.add_argument('--checkpoint', help='Trained chord-adapter checkpoint; defaults to the preset latest')
    parser.add_argument('--primer', default=os.environ.get('PRIMER_MIDI', sample_midi_path_init))
    parser.add_argument('--context-length', type=int, default=300)
    parser.add_argument('--midi-port', type=int, default=0)
    parser.add_argument('--midi-chords', default='',
                        help='Optional MIDI note:chord-key pairs, e.g. 60:1,62:2,67:5,74:shift+2')
    parser.add_argument('--soundfont', default='./piano.sf2')
    parser.add_argument('--output', default='./out/interactive_chords')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--no-preview', action='store_true')
    args = parser.parse_args()
    primer_indices = {Path(path).resolve(): index for index, path in enumerate(sample_midi_paths)}
    piece_index = primer_indices.get(Path(args.primer).resolve())
    if piece_index is None:
        parser.error('--primer must match a configured sample_midi_path with a FIXED_CHORDS entry')
    chord_bank = FIXED_CHORDS[piece_index]
    cfg = get_model_hparams(args.model)
    if cfg.get('model_type') != 'D_Chord_Joker':
        parser.error('--model must be a D_Chord_Joker preset')
    if args.checkpoint:
        cfg['ckpt_file_name'] = args.checkpoint
    if not Path(cfg['ckpt_file_name']).is_file():
        parser.error('No trained chord checkpoint found. Run train_chords.py first or pass --checkpoint.')
    if not 1 <= args.context_length <= cfg['seq_len'] or not 0 < args.temperature < float('inf'):
        parser.error('Invalid context length or temperature')
    primer = read_primer(args.primer, args.context_length)
    context_length = args.context_length
    try:
        entries = [entry.split(':') for entry in args.midi_chords.split(',') if entry.strip()]
        midi_chords = {int(note): key.strip() for note, key in entries}
        if len(midi_chords) != len(entries):
            raise ValueError('Each MIDI chord note must be mapped once')
        controls = ChordControls(len(primer['pitch']), context_length, chord_bank, midi_chords)
    except ValueError as error:
        parser.error(str(error))
    chord_legend = '  |  '.join(f'{key}: {chord.name}' for key, chord in chord_bank.items() if len(key) == 1)
    chord_legend += '\n' + '  |  '.join(f'{key}: {chord.name}' for key, chord in chord_bank.items() if len(key) > 1)
    if midi_chords:
        chord_legend += '\nMIDI chords: ' + '  |  '.join(f'{note}: {chord_bank[key].name}' for note, key in midi_chords.items())
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, cfg=cfg, compile_mode='none').to(device).eval()
    # Hardware/UI imports stay out of headless dataset and control checks.
    import fluidsynth
    import pygame
    import rtmidi
    from visualizer import Visualizer

    synth = fluidsynth.Synth()
    midi = rtmidi.MidiIn()
    visualizer = Visualizer(button_slots=1,
                            controls_legend=[('MIDI / space', 'joker'), ('hold numbers', 'chords'),
                                             ('release numbers', 'free'), ('R / S', 'reset / save')],
                            status_text=controls.status() + '\n' + chord_legend)
    midi_event = pygame.event.custom_type()
    output = {key: list(values) for key, values in primer.items()}
    primer_length = len(primer['pitch'])
    sounding: Dict[str, Tuple[int, float, int]] = {}
    last_onset = 0.0

    def playNote(note: int, velocity: int = 100) -> None:
        if velocity > 0:
            synth.noteon(0, note, velocity)
        else:
            synth.noteoff(0, note)

    def note_off(source: str) -> None:
        held = sounding.pop(source, None)
        if held is not None:
            pitch, started, position = held
            output['dur'][position] = max(1, round((time.perf_counter() - started) * 1000 / 32))
            # Several MIDI keys can generate the same pitch; keep it sounding until the last release.
            if not any(held_pitch == pitch for held_pitch, _, _ in sounding.values()):
                playNote(pitch, 0)
            visualizer.get_note(pitch, 0)
            visualizer.get_joker(0)

    def generate(source: str, velocity: int) -> None:
        nonlocal last_onset
        if source in sounding:
            return
        context = controls.next_tokens(device)
        context['pitch'] = torch.tensor([output['pitch'][-context_length:] + [0]],
                                        dtype=torch.long, device=device)
        pitch = model.gen_pitch_token(context, temperature=args.temperature)
        now = time.perf_counter()
        position = len(output['pitch'])
        output['pitch'].append(pitch)
        output['dtime'].append(0 if not last_onset else max(0, round((now - last_onset) * 1000 / 32)))
        output['dur'].append(1)
        output['vel'].append(velocity)
        output['chan'].append(0)
        controls.commit_note()
        last_onset = now
        sounding[source] = (pitch, now, position)
        playNote(pitch, velocity)
        visualizer.get_note(pitch, velocity, is_joker=True)
        visualizer.get_joker(velocity)

    def reset() -> None:
        nonlocal output, last_onset
        for source in list(sounding):
            note_off(source)
        output = {key: list(values) for key, values in primer.items()}
        last_onset = 0.0
        controls.reset(primer_length)
        visualizer.set_chord_chroma(None, False)
        visualizer.primer(primer['pitch'], primer['dtime'], torch.zeros(primer_length, dtype=torch.long), primer['dur'])

    def save() -> None:
        if len(output['pitch']) == primer_length:
            print('No generated notes to save yet.')
            return
        # Update held-note durations without interrupting their playback.
        for _, started, position in sounding.values():
            output['dur'][position] = max(1, round((time.perf_counter() - started) * 1000 / 32))
        destination = Path(args.output + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        destination.parent.mkdir(parents=True, exist_ok=True)
        continuation = {key: values[primer_length:] for key, values in output.items()}
        ms_SONG_to_MIDI_Converter(dict_to_song(continuation, force_vel=False),
                                 output_file_name=str(destination), timings_multiplier=1)
        print(f'Saved {destination}.mid')

    def midi_callback(event: Tuple[List[int], float], data: object = None) -> None:
        # Keyboard and MIDI share one event queue; process generation on the main thread.
        pygame.event.post(pygame.event.Event(midi_event, message=event[0]))

    try:
        synth.start()
        soundfont = synth.sfload(args.soundfont)
        if soundfont < 0:
            raise ValueError(f'Cannot load soundfont: {args.soundfont}')
        synth.program_select(0, soundfont, 0, 0)
        ports = midi.get_ports()
        if ports:
            if not 0 <= args.midi_port < len(ports):
                parser.error(f'--midi-port is out of range; available ports: {ports}')
            midi.open_port(args.midi_port)
            midi.set_callback(midi_callback)
            print(f'MIDI input: {ports[args.midi_port]}')
        else:
            print('No MIDI input found; keyboard controls remain available.')
        print('Unmapped MIDI keys / Space: joker. Hold chord controls for guidance; release all chord controls for free generation.')
        print(chord_legend)
        reset()
        if not args.no_preview:
            visualizer.play_primer(playNote, last_n=40, playback_speed=1)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == midi_event:
                    if len(event.message) == 3:
                        status, number, velocity = event.message
                        if controls.handle_midi(status, number, velocity):
                            continue
                        kind, channel = status & 0xF0, status & 0x0F
                        source = f'midi:{channel}:{number}'
                        if kind == 0x90 and velocity > 0:
                            generate(source, velocity)
                        elif kind == 0x80 or (kind == 0x90 and velocity == 0):
                            note_off(source)
                elif event.type == pygame.QUIT:
                    running = False
                elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
                    key, pressed = pygame.key.name(event.key), event.type == pygame.KEYDOWN
                    if controls.handle_key(key, pressed, shift=bool(event.mod & pygame.KMOD_SHIFT)):
                        continue
                    if key == 'space':
                        if pressed:
                            generate('keyboard', 100)
                        else:
                            note_off('keyboard')
                    elif pressed and key == 'r':
                        reset()
                    elif pressed and key == 's':
                        save()
                    elif pressed and key == 'escape':
                        running = False
                elif event.type == pygame.WINDOWFOCUSLOST:
                    # A key-up may go to another app after focus changes.
                    note_off('keyboard')
                    controls.release_keyboard()
            visualizer.status_text = controls.status() + '\n' + chord_legend
            visualizer.set_chord_chroma(controls.chroma.float().tolist(), controls.active)
            visualizer.draw(handle_events=False)
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        for source in list(sounding):
            note_off(source)
        midi.close_port()
        synth.delete()
        visualizer.stop()


if __name__ == '__main__':
    main()
