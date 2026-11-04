---
name: simple-evidence-tests
description: How to design, write, run and report tests of the monster_genie models. Use whenever a model's behaviour is measured or compared, such as checking a user control or conditioning, comparing checkpoints, diagnosing a model problem or gathering evidence for the thesis or a paper, even if the request only says "check", "evaluate" or "does it work". Keeps tests small enough for a MacBook Pro M1, and clear, documented and reproducible in ./test.
---

# Simple Evidence Tests

A test in this project answers one question about how a user-conditioned model behaves, and its result may be cited in a thesis or paper. A small test that anyone can read, rerun and defend is worth more than a large one, and the M1 cannot afford large ones.

This skill is for measuring model behaviour. A check that a code change runs correctly (shapes, no crash) stays temporary, as in minimal-clear-code.

## Design first

- Write the question as a claim that could turn out false: "Holding a chord longer weakens its effect on the next note."
- Before running, choose one main measure and say which result would support the claim and which would refute it. Choosing afterwards lets the data pick the story, and a reviewer will rightly distrust that. Report anything else you look at as exploratory.
- Compare with a reference: the same moments (positions in a piece where the test asks for the next note) with the control off, the previous checkpoint, or chance level. A number alone is not evidence.
- Change only one thing between conditions, and keep the notes, moments and seed identical. These paired comparisons cancel most of the differences between moments, which is also why few moments are enough.
- For a user control, measure both sides: how much it steers the model as the user asked, and what that costs in plausibility, for example how likely the model without the control finds the result. Strong steering that produces implausible music is not a success.
- State whether the test pieces were used for training, for validation or for neither, and check for overlap. Pieces seen in training say little about new music, and this project has been misled by it before: a base model had memorised about 80 % of the data used to train its adapter.

## Keep it small

Tests run on the MacBook Pro M1. Training and long runs belong on the Debian RTX 4090 machine.

- Aim for under one minute on the M1, and ask before running anything longer, giving the estimated time. Time one batch first and multiply, so that the estimate is measured.
- Prefer one forward pass over fixed moments: the model's next-note probabilities need no sampling and no repeated seeds. Generate note by note only when the question is about what happens over time, and then use the fewest moments and notes that can answer it.
- Start with 20 to 50 moments. Add more only if the uncertainty range is too wide to answer the question.
- Batch the moments, use `torch.inference_mode()` and load saved checkpoints. A test never trains or fine-tunes a model.
- If the M1 still cannot give a usable answer, first look for a simpler question. If a paper really needs a bigger run, keep the same script, raise its size constant and propose running it on the RTX 4090 machine, saying what the extra data would add.

## Honest evidence

- Count uncertainty per moment or per piece, never per note, because neighbouring notes are not independent. Give the number of pieces too, since many moments from few pieces are still few pieces.
- Reuse `bootstrap_ci` (95 % range of a mean) and `permutation_p` (paired test of whether a difference is zero) from `test/eval_chord_strength.py`. Also report how many moments moved in the expected direction, which needs no statistics to understand.
- Keep tests that refute the claim, and do not drop conditions because they look bad. A negative result is evidence too.
- Keep the claim as small as the test: 40 moments from 5 pieces support "in these pieces". Say when a result is preliminary and what a citable version would need.
- Numbers from temporary scripts are exploratory: say so and do not cite them. If one matters, write it as a test here and rerun it, because scratchpad scripts do not survive the session.

## Files

- One test, one file: `test/<what_it_measures>.py`, readable from top to bottom, with its settings as constants at the top. Do not add a test framework. These rules are for new tests; older scripts in `test/` keep their form.
- Reuse existing code instead of copying it: `model_loader.py` for models and the evaluation helpers already in `test/`. Add `ROOT` to `sys.path`, as the other scripts in `test/` do.
- Run in the `tgenie` conda environment, because the older torch in the base environment stalled on this code.
- Start the file with a docstring like this one, in plain words, so that the test can be understood without reading the code:

```python
"""
Question: Does holding a chord longer weaken its effect on the next note?
Supported if: the chord's notes gain less probability as the hold gets longer.
Refuted if: the gain stays flat or grows.
Model: save_models/AE_style_jokerParam_chords_best_v1.pth
Data: samples/not_seen, 5 pieces used for validation but never for training.
Method: at 40 moments, press the chord of the piece at that moment and change only how many notes it has been held for. The played notes stay the same.
Measure: probability given to the chord's notes for the next note, minus the same probability with no chord pressed, in percentage points. Higher means a stronger effect.
Size and time: 40 moments, one forward pass per condition, about 25 s on the M1.
Run: python test/chord_hold_strength.py
Result (2026-11-04): supported. After 64 notes the effect is about half as strong with a known button and a third as strong without one. See test/results/chord_hold_strength.txt.
"""
```

- Each run prints its result and appends the same text to `test/results/<test_name>.txt`, so earlier runs are kept. Include the date, the git commit and whether there were uncommitted changes, the checkpoint, the device, the settings, the running time and the result table. Do not put results in `out/`, which git ignores.
- After a run, update the docstring's `Result` line with the date and a one-sentence conclusion.

## Reporting

Report results following clear-explanations, and name the results file, so that every number in a paper can be traced back to a test and a run.
