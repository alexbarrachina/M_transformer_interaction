"""One-button performance with held chord suggestions.

Defaults: space or MIDI note 48 generates; hold 1..5, MIDI notes 36..40,
or CC pedals 20..24 for C, F, G7, Am, D7. Release all chord controls for free
generation. R / MIDI 30 resets; S / MIDI 31 saves; Escape exits.

--controls accepts a JSON list of mappings, for example:
[{"name":"C", "pitch_classes":[0,4,7], "anchor_pc":0,
  "key":"1", "midi_note":36, "pedal_cc":20}]
The bank is explicit and can match any primer; the model receives no key.
"""

import argparse
import datetime
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from harmony.chord_joker import ChordControls, read_chord_mappings
from midiUtils import dict_to_song, midi_to_dict, ms_SONG_to_MIDI_Converter
from model_loader import load_model
from models import get_model_hparams


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
    parser.add_argument('--model', default='D_Chord_Joker_little_v1')
    parser.add_argument('--checkpoint', help='Trained chord-adapter checkpoint; defaults to the preset latest')
    parser.add_argument('--primer', default=os.environ.get('PRIMER_MIDI', './samples/Bach_Prelude_and_Fugue_in_C_major.mid'))
    parser.add_argument('--context-length', type=int, default=300)
    parser.add_argument('--controls', help='JSON list of explicit chord mappings (see example above)')
    parser.add_argument('--trigger-note', type=int, default=48)
    parser.add_argument('--midi-port', type=int, default=0)
    parser.add_argument('--soundfont', default='./piano.sf2')
    parser.add_argument('--output', default='./out/interactive_chords')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--no-preview', action='store_true')
    args = parser.parse_args()
    cfg = get_model_hparams(args.model)
    if cfg.get('model_type') != 'D_Chord_Joker':
        parser.error('--model must be a D_Chord_Joker preset')
    if args.checkpoint:
        cfg['ckpt_file_name'] = args.checkpoint
    if not Path(cfg['ckpt_file_name']).is_file():
        parser.error('No trained chord checkpoint found. Run train_chords.py first or pass --checkpoint.')
    if (not 1 <= args.context_length <= cfg['seq_len'] or not 0 <= args.trigger_note <= 127
            or args.trigger_note in (30, 31) or not 0 < args.temperature < float('inf')):
        parser.error('Invalid context length, trigger note or temperature')
    primer = read_primer(args.primer, args.context_length)
    context_length = args.context_length
    controls = ChordControls(len(primer['pitch']), context_length, read_chord_mappings(args.controls))
    if (set(controls.midi_notes) & {args.trigger_note, 30, 31} or set(controls.pedals) & {17, 18}
            or set(controls.keys) & {'space', 'r', 's', 'escape'}):
        parser.error('Chord mappings overlap performance trigger, reset, save or exit controls')
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
                            controls_legend=[('space / MIDI', 'play'), ('hold chord', 'guide'),
                                             ('release', 'free'), ('R / S', 'reset / save')],
                            status_text=controls.status())
    midi_event = pygame.event.custom_type()
    output = {key: list(values) for key, values in primer.items()}
    primer_length = len(primer['pitch'])
    sounding: Dict[str, Tuple[int, float, int]] = {}
    last_onset = 0.0

    def note_off(source: str) -> None:
        held = sounding.pop(source, None)
        if held is not None:
            pitch, started, position = held
            output['dur'][position] = max(1, round((time.perf_counter() - started) * 1000 / 32))
            synth.noteoff(0, pitch)
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
        synth.noteon(0, pitch, velocity)
        visualizer.get_note(pitch, velocity, is_joker=True)
        visualizer.get_joker(velocity)

    def reset() -> None:
        nonlocal output, last_onset
        for source in list(sounding):
            note_off(source)
        output = {key: list(values) for key, values in primer.items()}
        last_onset = 0.0
        controls.reset(primer_length)
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

    def handle_midi(message: List[int]) -> None:
        if len(message) != 3:
            return
        status, number, value = message
        if controls.handle_midi(status, number, value):
            return
        kind = status & 0xF0
        pressed = kind == 0x90 and value > 0
        source = f'note:{status & 0x0F}:{number}'
        if kind in (0x80, 0x90):
            if number == 30 and pressed:
                reset()
            elif number == 31 and pressed:
                save()
            elif number == args.trigger_note:
                if pressed:
                    generate(source, value)
                else:
                    note_off(source)
        elif kind == 0xB0 and value > 0:
            if number == 17:
                reset()
            elif number == 18:
                save()

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
        print('Hold suggestions: ' + ', '.join(
            f'{c.name} [key={c.key or "none"}, MIDI={c.midi_note}, CC={c.pedal_cc}]' for c in controls.mappings))
        reset()
        if not args.no_preview:
            def preview_note(pitch: int, velocity: int = 100) -> None:
                if velocity:
                    synth.noteon(0, pitch, velocity)
                else:
                    synth.noteoff(0, pitch)
            visualizer.play_primer(preview_note, last_n=40, playback_speed=1)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == midi_event:
                    handle_midi(event.message)
                elif event.type == pygame.QUIT:
                    running = False
                elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
                    key, pressed = pygame.key.name(event.key), event.type == pygame.KEYDOWN
                    if controls.handle_key(key, pressed):
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
                    for key in controls.keys:
                        controls.handle_key(key, False)
                    note_off('keyboard')
            visualizer.status_text = controls.status()
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
