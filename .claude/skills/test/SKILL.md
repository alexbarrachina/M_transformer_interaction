---
name: test
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
8. A conclusion, in one or two plain sentences, that will start with "*** CONCLUSION ***" to be visually clear for the user.

When a test does not accept the model (another loader, no joker, a different number of buttons), do not patch around it inside the run. Report it, and if the fix is a small generalisation of the test's loading code, make it, then run the test with the reference model in both roles: every difference must be exactly 0, which shows the change kept the test the same.

## Terms

- Define every metric, abbreviation, statistic and name at its first use in each reply, in one clause: what it measures, on what, and which direction is better. This includes standard terms (NLL, entropy, log-odds, rank correlation, p-value) and terms defined in earlier replies. A one-clause definition is not a repeated explanation.
- Prefer the description to the label, if the label is not a very standard concept: "the probability the model gives to the notes of the held chord", not "chord mass". Do not coin new labels. If a term has to recur, keep one name for it throughout the reply.
- Names of test conditions ("free", "fresh") are labels too. Say what was done: "no chord pressed", "control applied only to the newest note".
- Name scripts, presets and variables only when the user needs them to find or run something, and say what each one is.
- The user's own terms need no definition, unless I use them differently.

## Numbers

A number is understood only if the reader can say what was measured, on what, and compared with what.

- Give the unit, and say what a percentage is of ("of consecutive note pairs"). Write probabilities as percentages, and differences between them as percentage points.
- Give a reference: no chord pressed, chance level, before and after, the previous model. A number alone does not show whether it is good or bad.
- Translate abstract scores into something concrete. For NLL, give e^(-NLL), the probability typically given to the real note. For a difference, say how many notes out of 100 it amounts to.
- Say how much data there is and whether the difference is larger than the noise, in words: "in 16 of 16 comparisons", "only 8 moments and 1 seed, so this is weak".
- Avoid packed notation such as `.088/.061/.045`, `guided .28 -> .62`, `k >= 16` or `p90`. Write a sentence or a table.
- If two figures come from different data or conditions, say so where they are compared.

## Tables

- Use a table only when several comparisons must be seen together. One or two comparisons fit in a sentence.
- Before the table, say which test it comes from and what the rows and columns change.
- Write headers as whole phrases with units, so that each cell can be read aloud as a sentence.
- After the table, or in the sentence before it, say what to conclude from it.

## Deciding that a new test is needed

Read `test/README.md` first.

- The same behaviour is measured already, only on another model: run that test. Never copy a test for a new model.
- The same behaviour, measured another way or on another detail: add the measure to the existing test as an exploratory one, with the date it was added, as `test/button_map_chromatic.py` did. Its main measure and verdict rule stay as they were chosen.
- A question about one recording, one primer or one incident: a case study in `test/case_studies/`, with its models fixed.
- A behaviour no test measures: a new test, as below.

## Ask for new behaviour test

- Don't create a new beahviour test without asking the user first. 
- Apply \new-test SKILL for the creation of the new test 