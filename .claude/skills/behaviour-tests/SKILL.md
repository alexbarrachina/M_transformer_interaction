---
name: behaviour-tests
description: How the monster_genie tests in ./test form a collection of behaviour evaluators that are reused across models. Use when a new model or checkpoint has to be evaluated (choose and run the existing tests for its model type, name and report the results), or when a behaviour that no existing test measures needs a new test (decide that it is new, name it, make it take any model of its type, add it to test/README.md). Not needed for a one-off code check or for reading old results.
---

# Behaviour Tests

`test/` is a collection of behaviour evaluators. Each script directly in `test/` measures one behaviour of a model, such as how strongly a held chord pulls the notes, how chromatic the button map is or whether button 7 gives the tonic, and takes the model to evaluate as an argument. A new model is evaluated by running the tests that already exist; a new test is written only for a behaviour that none of them measures. `test/README.md` is the index: behaviours, tests, the model types each test takes, running times, shared moments and helpers.

This skill says how the collection is organised and used. How to design, size and report a single test stays in simple-evidence-tests, and how to word the report in clear-explanations.

## Evaluating a new model

1. Find the model type: `model_type` of its preset in `models.py`, or `cfg['model_type']` saved in the checkpoint. Look up its tests in the README table "Which tests for which model".
2. Choose the reference model: the one the new model should improve on or replace, usually the previous checkpoint or the model the app uses now. If the user has not said, use the test's default reference and say which it is. The test runs both on the same moments, so the comparison is paired.
3. Check that the new model did not train on the test pieces. The shared moments come from `Training-Data/giantmidi_full_w_chords_unseen_test`; a model trained on that split cannot be evaluated on them, so say so instead of running.
4. Add up the running times from the README and ask before starting when the total is over one minute, giving the list of tests and the estimate. Run them one at a time (they share the M1's memory), in the `tgenie` environment, in the background when they take minutes.
5. Run each test as `python test/<test>.py <model> [reference]`. The test appends its report to `test/results/<test>__<model>.txt` by itself.
6. Add a `Result (date, <model> against <reference>): ...` line to each test's docstring with the verdict and one sentence. Keep the earlier Result lines.
7. Report the evaluation as one table: one row per behaviour, the main measure for the new and the reference model, the difference with its 95 % range, the verdict and the results file. Then say what the profile means for playing, following clear-explanations.

When a test does not accept the model (another loader, no joker, a different number of buttons), do not patch around it inside the run. Report it, and if the fix is a small generalisation of the test's loading code, make it, then run the test with the reference model in both roles: every difference must be exactly 0, which shows the change kept the test the same.

## Deciding that a new test is needed

Read `test/README.md` first.

- The same behaviour is measured already, only on another model: run that test. Never copy a test for a new model.
- The same behaviour, measured another way or on another detail: add the measure to the existing test as an exploratory one, with the date it was added, as `test/button_map_chromatic.py` did. Its main measure and verdict rule stay as they were chosen.
- A question about one recording, one primer or one incident: a case study in `test/case_studies/`, with its models fixed.
- A behaviour no test measures: a new test, as below.

## Writing a new behaviour test

Follow simple-evidence-tests for the design, the size and the docstring. In addition:

- Name it `test/<behaviour>_<what_it_measures>.py`, where `<behaviour>` is the prefix of its family: `chord_`, `button_`, `align_`, `tension_`, `timing_` or `realism_`. Use a new prefix only for a new family of behaviours, and add a section for it to the README.
- The evaluated model is the first positional argument and the reference model the second, each with a default: a preset name for presets, a checkpoint path for checkpoints. The same model in both roles must give differences of exactly 0; make that check run first.
- Make it work for every model of its type: load with `model_loader.load_model`, `load_style_model` or `load_style_checkpoint`, and read sizes such as the number of buttons from the model's config instead of writing them as numbers.
- Write the results to `RESULTS = ROOT / f'test/results/<test>__{NEW}.txt'` for a preset, or `{NEW.stem}` for a checkpoint path. Every other output (MIDI files, JSON, plots, audio, raw values) goes in `test/results` too, never in `out/`: one file `test/results/<test>__<model>.<extension>`, or a folder `test/results/<test>__<model>/` for several. The report gives the date, the commit and the number of uncommitted files (`git_line`), both models with their checkpoint files, the device, the settings, the running time, the result table and the verdict.
- Use the shared 60 moments and the helpers listed in the README, so its numbers can be compared with the other tests'. Choose other data only when the question needs it, and say why in the docstring.
- Add `Run:` with its arguments and `Results: test/results/<test>__<model>.txt, one file per evaluated model.` to the docstring.
- Add the test to the README: a row in its behaviour's table, with the question, the models it takes, the main measure and the running time, and its name in the table of model types.

## Keeping the collection tidy

- Generation and listening scripts go in `test/generation/`, checks of code correctness in `test/code_checks/`, studies of one recording or primer in `test/case_studies/`. Only behaviour tests stay directly in `test/`.
- Every script in `test/`, case studies and generation scripts included, writes its outputs in `test/results`, named after its test and model; `out/` holds only the app's recordings, which some case studies read as input.
- The tests marked "older form" in the README take `--checkpoint` and write JSON records and a `summary.json` in `test/results/<test>__<model>/`, but no `.txt` report. When one of them is needed for an evaluation, ask whether to make it append to `test/results/<test>__<model>.txt` first, and update the README if it is converted.
- When a test is renamed, update the rename table in the README, the references in the other tests' docstrings, and the docs and memory notes that name it.
