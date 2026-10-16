# Harmony and tension

This package contains the live harmony display, the performer's tension controls, and the offline evaluator used to select tension-model checkpoints. Training and scoring use these local files; they do not need `../PROCESS/harmony_detector`.

| File | Role |
| --- | --- |
| `__init__.py` | Marks this directory as a Python package without loading the live tracker when training imports the scorer. |
| `harmony.py` | Causal, live key and tension tracker. `HarmonyTracker` consumes sounding notes and returns the current chord, home/local key, and tension level for the visualizer. Its online smoothing differs from the offline dataset labels. |
| `tension_joker.py` | Tension-key state (`TensionControls`), primer MIDI reading and writing (`PianoPerformance`), home-key parsing, generated-note scoring (`TensionScorer`), and the fixed generation benchmark used for checkpoint selection (`validate_tension_controls`). The scorer uses source chord timing when available, but infers chord tones from played pitches. |
| `build_key_tension_dataset.py` | Offline label builder copied from `../PROCESS/harmony_detector`. `TensionScorer` uses its calibration, chroma reader, key analysis, tension indicators, and level quantization. Its dataset-building command is separate from model training. |
| `key_track.py` | Onset-window key tracking and Viterbi path routines used by the offline builder. |
| `key_detect.py` | Key profiles, chroma features, and collection/mode detection used by `key_track.py` and the offline builder. |
| `key_benchmark.py` | Supplies `parse_ground_truth`, which the offline builder imports to read a key from a piece title; it also contains the original key-detection benchmark. |

The four offline analyzer files are local copies. Their package imports were changed to relative imports; the scoring methods and calibration were retained. `TensionScorer` records hashes of those files, its own source, and its calibration in each training checkpoint so scores from a changed evaluator are not compared as though they came from the same protocol.

`train_tension.py` loads `TensionScorer` only for held-out generation and checkpoint selection. `train.py` keeps its original training path without importing the analyzer. The pitch model learns from the existing tension-labelled pickle; the offline builder is not run to rebuild labels during training. `interaction_1_button_tension.py` imports `HarmonyTracker` for the live display and `TensionControls` for the requested level.

Run the live tracker on a MIDI file with `python -m harmony.harmony path/to/file.mid`. The offline builder and benchmark can likewise be run as `python -m harmony.build_key_tension_dataset` and `python -m harmony.key_benchmark` when their input data is available. Their original corpus paths are defaults for those standalone commands, not training dependencies.
