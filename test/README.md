# Behaviour tests

Each script directly in `test/` measures one behaviour of a model, such as how strongly a held chord pulls the notes or how evenly the buttons spread over the pitches. A test takes the model to evaluate as an argument, so the same test is run on every new checkpoint of the right type. To see how a new model behaves, run the tests listed for its type in [Which tests for which model](#which-tests-for-which-model). A new test is written only when a behaviour is not measured by any test here.

How to evaluate a new model with these tests, or to add a test, is written in `.claude/skills/behaviour-tests/SKILL.md`.

The first word of a file name is the behaviour it belongs to: `chord_`, `button_`, `align_`, `tension_`, `timing_`, `realism_`. Each script's docstring says the question, the claim and when it is supported or refuted, the data, the method, the measure, the running time and the results so far.

## Running a test

Run from the project root, in the `tgenie` conda environment (the older torch of the base environment stalls on this code):

```bash
conda activate tgenie
python test/button_map_chromatic.py AE_style_tester_v3 AE_style_tester_v2
```

The first argument is the model to evaluate, the second (where the test has one) the reference model it is compared with on the same moments. Depending on the test, a model is a preset name of `models.py` (loaded with its `ckpt_file_name`) or the path of a checkpoint. Giving the same model twice is a check run: every difference must then be exactly 0.

Most tests run in 1 to 5 minutes on the M1 (time in the tables below). The older tests marked "older form" take `--checkpoint` options instead of positional arguments; `--help` lists them.

## Results

Every output of every script in `test/` goes in `test/results/`, never in `out/`: text reports, MIDI files, JSON records, plots, listening pages, audio and raw values. Git ignores the audio renders (`.wav`), which are large; the MIDI files they were rendered from are kept. Every run appends its report to `test/results/<test>__<model>.txt`:

- `<test>` is the script name without `.py`.
- `<model>` is the evaluated model: its preset name, or the checkpoint file name without `.pth` for tests that take a path.
- A comparison run goes in the evaluated model's file and names its reference model inside. A case study with two fixed models joins them with `_and_`.
- Each report gives the date, the git commit and the number of uncommitted files, both models and their checkpoint files, the device, the settings, the running time, the result table and the verdict.
- Any other output has the same name: one file `test/results/<test>__<model>.<extension>` (for example the raw values `chord_strength_many_chords__D_Chord_Joker_clear_latest_best.npz`), or a folder `test/results/<test>__<model>/` when there are several files (MIDI, JSON, plots).
- The older tests and the generation scripts name the folder after the last checkpoint given (the evaluated one; earlier ones are references) or after the model preset they load. `--output` chooses another folder.

So `ls test/results/*__AE_style_tester_v3.txt` lists every behaviour measured on that model, and `ls test/results/button_map_chromatic__*` every model measured on that behaviour.

The older tests write one JSON record per MIDI file and a `summary.json` in their folder and print the summary, but do not append a `.txt` report yet; the two `chord_strength_guidance__*.txt` files were copied by hand from earlier runs.
The only files the tests read from `out/` are the app's recordings studied by two case studies: those are inputs, not outputs.

## Behaviours and their tests

### Chord control: does a held chord or the change key steer the notes?

| Test | Question | Models it takes | Main measure | Time on the M1 |
|---|---|---|---|---|
| `chord_strength_many_chords.py` | Does a chord adapter give the held chord's notes more probability than the app's adapter, over many chords and pieces? | `D_Chord_Joker` checkpoint path; the reference is fixed to `D_Chord_Joker_change_latest_best.pth` | Probability on the chord's notes over 8 generated notes, for the piece's real chord and for all 24 major and minor triads, new minus reference, in percentage points; cost: how unlikely the frozen base finds the notes | 303 s |
| `chord_strength_guidance.py` (older form) | The same for four chords (real, dominant, subdominant, a tritone away), with an optional sweep of the pedal guidance weight | One or more `D_Chord_Joker` or `D_Base_Joker` checkpoints, `--checkpoint` | Probability on the chord's notes with the chord held minus with no chord pressed | not timed |
| `chord_strength_style.py` (older form) | `chord_strength_guidance.py` for the style joker's chord adapters, with the buttons of the real notes or the joker held | `AE_style_jokerParam_chords` checkpoints, `--checkpoint` | As above, per button mode | not timed |
| `chord_change.py` (older form) | Does the change key (Space) make the notes leave the chord that was sounding? | `D_Chord_Joker` change checkpoint, `--checkpoint` | Probability on the previous chord's notes, change pressed minus not pressed | not timed |
| `chord_change_style.py` (older form) | `chord_change.py` for the style joker | `AE_style_jokerParam_change` checkpoint, `--checkpoint` | As above, per button mode | not timed |
| `chord_button_repeats.py` | Does holding a chord make different buttons give the same note? | Two `AE_style_jokerParam_chords` checkpoint paths with the same frozen style joker (new, present) | Number of different most likely notes among the 19 buttons, chord held minus no chord | 274 s |

The four older tests read the MIDI files of `samples/not_seen` by default, and some of them are in the chord adapters' training split: check the overlap they report before citing a number.

### Button control: does the pressed button decide the note as intended?

| Test | Question | Models it takes | Main measure | Time on the M1 |
|---|---|---|---|---|
| `button_control_strength.py` | Does a button tell the model which note to play? | Presets of base style models (`AE_style_*`) and style jokers (`AE_style_jokerParam_*`) | Probability given to the pianist's real next note with its button, new minus reference; reference row: with no button (style jokers only) | 83 s |
| `button_repeat_same_button.py` | Does pressing the previous note's button again give the previous note? | Presets, as above | Probability of the previous note with the previous note's button, new minus reference; control: one button higher | 66 s |
| `button_repeat_neighbour_labels.py` | Do the encoder's training buttons put a repeated note on a neighbouring button, and does the decoder then repeat the note when the button moves one step? | `AE_style_jokerParam_chords` checkpoint path, used with no chord (its frozen style joker) | Share of one-step button moves that repeat the note, minus the share of all note pairs that repeat | 77 s |
| `button_moves_generation.py` | While generating with a fixed button pattern, does the same button repeat the note, does a one-step move wrongly repeat it, and does the note move with the button? | Presets, as above | Share of same-button presses that repeat the previous note, new minus reference; risks: one-step repeats and wrong directions | 93 s |
| `button_future_leak.py` | Do the buttons carry the notes that come after them, so that the decoder follows them worse in play, where a press knows only the notes already played? | Presets, as above | Probability given to the pianist's real next note with the whole passage encoded together minus with the primer encoded alone and buttons knowing only the past, new minus reference | 52 s |

`button_control_strength.py` encodes the whole passage together, as in training, so buttons that know the coming notes raise its measure; `button_future_leak.py` measures by how much.

### Button map: how are the pitches spread over the 19 buttons?

| Test | Question | Models it takes | Main measure | Time on the M1 |
|---|---|---|---|---|
| `button_map_chromatic.py` | Is the map chromatic, and does the encoder crowd the notes onto the end buttons 0 and 18? | Presets, as above | Share of the 19 buttons whose most likely note is outside the passage's scale; share of the passage's notes the encoder puts on button 0 or 18 | 83 s |
| `button_map_compression.py` | How faithfully does the encoder compress the pitches into buttons: does it keep the melodic contour, the notes' relative heights, and avoid piling notes on the end buttons? Also draws, for every MIDI file in `test/samples`, its pitches above the model's buttons (`.jpg`) | Presets, as above (encoder only) | Share of consecutive note pairs moving the same way in buttons as in pitch; distance between a note's height in pitch and in buttons within 32-note windows; share of notes on the first or last button. Reference: a linear map of the passage's pitch range onto the buttons | 17 s |

`button_control_strength.py` also reports, as exploratory measures, how many different notes the 19 buttons give, how often neighbouring buttons come in the wrong order and the span in semitones from button 0 to button 18.

### Alignment: do buttons point at musical landmarks?

| Test | Question | Models it takes | Main measure | Time on the M1 |
|---|---|---|---|---|
| `align_tonic_button_7.py` | Does button 7 give the tonic of the passage's key, in the octave of the Reins layout? | Presets, as above | Probability of that tonic with button 7 pressed, new minus reference; control: the note's own button | 63 s |

`button_map_chromatic.py` measures alignment to the passage's scale (its first claim).

### Other model families

| Test | Question | Models it takes | Output | Time |
|---|---|---|---|---|
| `tension_joker.py` (older form) | Do tension checkpoints follow the requested tension over long paired continuations? | `D_Tension_Joker` checkpoints, positional | `test/results/tension_joker__<last checkpoint>/` | not timed |
| `timing_satie.py` (older form) | Does a style joker that sees note timing use the rhythm of a Satie-like bass and chord pattern to put the bass note below the chord, also at other tempos and chord spreads? | Style joker checkpoint with timing input (`AE_style_jokerParam_dtime_*`), `--checkpoint` | `test/results/timing_satie__<checkpoint>/` (MIDI, plots, report) | not timed |
| `realism_fmd.py` (older form) | How far are generated MIDI files from GiantMIDI (Frechet Music Distance, lower is closer)? | The MIDI files in `metrics/samples`, not a model | `test/results/realism_fmd__metrics_samples.csv` | not timed |

## Which tests for which model

| Model type | Tests |
|---|---|
| Base style model (`AE_style_*` preset: buttons and style, no joker) | `button_control_strength`, `button_future_leak`, `button_repeat_same_button`, `button_moves_generation`, `button_map_chromatic`, `button_map_compression`, `align_tonic_button_7` |
| Style joker (`AE_style_jokerParam_*` preset) | The same seven; `button_control_strength` adds the no-button reference |
| Style chord adapter (`AE_style_jokerParam_chords` checkpoint) | `chord_button_repeats`, `chord_strength_style`, `button_repeat_neighbour_labels` (its frozen style joker) |
| Style change adapter (`AE_style_jokerParam_change` checkpoint) | `chord_change_style` |
| Chord adapter of the base joker (`D_Chord_Joker` checkpoint) | `chord_strength_many_chords`, `chord_strength_guidance`, `chord_change` (change checkpoints) |
| Tension joker (`D_Tension_Joker`) | `tension_joker` |
| Style joker with timing input (`AE_style_jokerParam_dtime_*`) | `timing_satie` |
| Generated MIDI files | `realism_fmd` |

## Shared moments and helpers

The newer tests share the same 60 moments, so their numbers can be compared: one chord change in each of 60 pieces of `Training-Data/giantmidi_full_w_chords_unseen_test`, chosen with seed 1729 by `choose_moments` in `chord_strength_many_chords.py`, leaving out pieces with over 10 % of their 8-note sequences in the training data. Neither the style models nor the chord adapters trained on this split; the chord adapters used it for validation.

Code to reuse instead of copying:

| Helper | Where | What it does |
|---|---|---|
| `bootstrap_ci`, `permutation_p` | `chord_strength_guidance.py` | 95 % range of a mean; paired test of whether a difference is zero |
| `read_events`, `overlap_per_piece`, `choose_moments`, `chords_at` | `chord_strength_many_chords.py` | Read the test pickle and choose the 60 moments |
| `load_style_model`, `read_passages`, `passage_inputs`, `last_logits`, `git_line` | `button_control_strength.py` | Load a style preset; the passages, style and next-note logits at the moments; the git line of a report |
| `style_inputs`, `player_buttons`, `mode_buttons`, `load_style_checkpoint` | `chord_strength_style.py` | Style context of a passage; the encoder's buttons; load a style chord checkpoint |

## Other folders

These scripts are not behaviour tests and are not run on every new model.

- `case_studies/` answers one question about one recording or one primer, with fixed models. Run them as `python test/case_studies/<name>.py`.
  - `style_chord_recording_buttons.py`: in one recording, did the held chord make different buttons give the same note?
  - `style_button_pattern_repeats.py`: does the recording's button pattern repeat the previous note on a changed button, over 60 pieces?
  - `style_button_pattern_chopin.py`: does the app's Chopin primer explain the repeats of that recording?
  - `button_repeats_recordings.py`: do the repeats heard in six recordings come from the buttons the models were trained with?
- `generation/` writes MIDI (and for two of them listening pages and control checks) to listen to, in `test/results/<script>__<model>/`; it measures nothing that is compared between models: `1_oracle_control_stream.py`, `1_oracle_control_stream_9_tests.py`, `automatic_button_gen.py`, `automatic_inference_harm.py`, `automatic_inference_harmony.py`, `automatic_inference_melody.py`, `test_base_joker.py`, `eval_chord_joker.py`, `eval_style_chord_joker.py`.
- `code_checks/` checks that the code is correct (shapes, masks, short training runs), not how a model behaves: `python test/code_checks/test_style_joker_param.py`, `test_train_style_joker.py`, `test_timing.py`.

## Names before 2026-11-13

Docs, memory notes and old reports may use these names.

| Old name | New name |
|---|---|
| `test/eval_chord_strength.py` | `test/chord_strength_guidance.py` |
| `test/eval_style_chord_strength.py` | `test/chord_strength_style.py` |
| `test/eval_chord_change.py` | `test/chord_change.py` |
| `test/eval_style_chord_change.py` | `test/chord_change_style.py` |
| `test/style_chord_button_repeats.py` | `test/chord_button_repeats.py` |
| `test/repeated_button_note.py` | `test/button_repeat_same_button.py` |
| `test/style_button_repeat_cause.py` | `test/button_repeat_neighbour_labels.py` |
| `test/tonic_on_button_7.py` | `test/align_tonic_button_7.py` |
| `test/eval_tension_joker.py` | `test/tension_joker.py` |
| `test/evaluate_timing.py` | `test/timing_satie.py` |
| `test/run_fmd_samples.py` | `test/realism_fmd.py` |
| `test/results/<test>.txt` | `test/results/<new test name>__<model>.txt`, one file per model, split run by run |
| `test/results/eval_chord_strength_guidance.txt` | `test/results/chord_strength_guidance__D_Chord_Joker_change_latest_best.txt` |
| `test/results/eval_chord_strength_training_data.txt` | `test/results/chord_strength_guidance__D_Chord_Joker_clear_latest_best.txt` |

The case studies, generation scripts and code checks kept their names and moved into their folders.
