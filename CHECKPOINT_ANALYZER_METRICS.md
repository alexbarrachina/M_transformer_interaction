# Understanding the Checkpoint Analyzer Metrics

This guide explains the numbers in the Checkpoint Analyzer in everyday language. It is meant to help you decide which checkpoints are worth listening to and comparing. **There is no single “best model” score.** A model can follow buttons well but produce repetitive music, or make interesting music while ignoring its style reference.

## Before reading the numbers

- **Pitch** means how high or low a note is. **Contour** means whether the melody goes up, down, or stays level, even if it chooses different exact notes.
- A **chord** is a group of notes heard as one harmony. A **key** is the broader home area those harmonies suggest. **Cross-attention** is the model's way of looking at a second music excerpt while it generates.
- **Prompt** is the opening music given to the model. **Continuation** is the new music the model generates. The generation scores use the continuation alone; they do not give the model credit for notes in the prompt.
- **Style reference** is the music supplied through cross-attention, when the checkpoint supports it. A prompt or style score uses only the excerpt actually given to the model, not the whole MIDI file.
- **Oracle controls** are the button or arrow instructions extracted from the real continuation. They let us ask whether the model can follow accurate instructions. The future pitches themselves are never handed to the generator.
- A **seed** is a repeatable random starting point. A sampled result can change with the seed. **Greedy** means always taking the model's most likely next pitch.
- The model generates **pitches**. The timing, note lengths, and loudness in a preview are inherited from the source MIDI. A good-sounding rhythm or thick chord texture in that preview is therefore not proof that the model created those aspects.

Look at the number of **cases**, **notes**, **windows**, or **seeds** beside a result. A striking score from one short example is weaker evidence than a similar result across many pieces. In the overview, **mean** is the average across compatible cases; **standard deviation** tells you how much those cases differ. A large standard deviation means the model is inconsistent. Example pieces and test-split pieces are kept separate.

**Baseline Difference** is the selected checkpoint's number minus the baseline checkpoint's number, for matching cases. Positive means the selected checkpoint scored higher on that particular metric. It does **not** always mean better: lower collision and reconstruction-surprise scores are usually desirable, while more harmonic movement can mean development or merely wandering.

## 1. Coherence and reconstruction

This tab asks two different questions: *Can the model predict the real next notes?* and *What happens when it must generate on its own?* Prediction scores do not, by themselves, judge whether an alternative continuation is musically good.

| On-screen metric | Plain-language meaning | How to read it |
| --- | --- | --- |
| `reconstruction_cross_entropy` | How surprised the model was by the real next pitch while reading the real previous notes. | Lower means the real continuation was more predictable to the model. It is not a listening-quality score. |
| `teacher_forced_accuracy` | Share of times its single most likely next pitch was the exact real pitch, while it was given the real previous notes. | Higher means better exact prediction. |
| `reconstruction_notes` | Number of next-note predictions used for those two scores. | Larger counts make the result more informative. |
| `pitch_agreement` | Share of generated pitches that exactly match the human continuation at the same positions. | Higher means closer copying; a different but convincing melody can score low. |
| `contour_agreement` | Share of melodic moves that go in the same direction as the human continuation: up, down, or level. | Higher means a similar melodic shape, even when exact notes differ. |
| `repeated_pitch_fraction` | Share of consecutive generated notes that repeat the same pitch. | Very high can point to getting stuck. Some repetition is normal. |
| `longest_repeated_pitch_run` | Longest run of the same pitch. | A long run deserves a listen, especially if it was not requested. |
| `pitch_class_entropy` | Variety of note names after ignoring octave. | Higher means more variety, but random notes also score high. |
| `distinct_pitches` | Number of different exact pitches in the generated continuation. | Shows range of notes used; it is not a quality score. |
| `piano_range_violations` | Share of generated pitches outside a normal 88-key piano's range. | Lower is usually better for piano output. |
| `melodic_leaps_over_octave` | Share of melodic jumps larger than an octave. | Many large jumps can sound erratic, but some are intentional. |
| `mean_pitch` | Average height of generated notes. | Use it to notice an unexpectedly high or low register. |
| `register_drift` | Change in average pitch from the early part of the continuation to the late part. | Positive means the register rises; negative means it falls. Large unplanned drift may matter. |
| `prompt_affinity` | Similarity between the generated and prompt note-name mixtures, ignoring octave and order. | Higher means similar overall pitch classes, not necessarily the same chord sequence or better music. |
| `style_reference_affinity` | The same kind of similarity to the style excerpt used in that case. | Useful only alongside matched style conditions; a high value alone does not prove cross-attention caused it. |
| `arrow_adherence` | Share of generated melodic steps that follow the instructed up, down, or level arrows. | Higher means better response to arrows. Only meaningful for checkpoints with arrow controls. |
| `button_direction_agreement` | Share of changing ordered-button steps whose pitch moves in the same direction. | Higher means buttons generally respect their order; see the full button sweep in tab 5. |
| `button_change_same_pitch` | Share of button changes that produced the same pitch as before. | Higher suggests some button changes have no audible pitch effect. |
| `button_changes_tested` | Number of button changes behind those two scores. | Check this before trusting a percentage. |

The **conditions** matter as much as the number. `oracle` uses correct controls. `unconditioned` uses the model's native free-generation mode. `joker_half` releases some button instructions; `joker` releases all of them, for checkpoints that support jokers. Compare conditions for the *same* piece and seed. A change in exact-pitch agreement under a joker does not automatically mean the music got worse.

### Music-fingerprint distances

| On-screen metric | Plain-language meaning | How to read it |
| --- | --- | --- |
| `reference_squared_embedding_distance` | Distance between one generated clip and its matching human clip in a learned music fingerprint. | Lower means closer in that fingerprint. This single-clip number is **not** FMD. |
| `giantMIDI_mean_squared_distance` | Distance from the average fingerprint of a separate giantMIDI reference collection. | An outlier warning only; it is not a measure of faithfulness to this piece. |
| `fmd` | Distance between a whole collection of generated clips and an equally sized human collection. | Lower means the two collections look more alike to the fingerprint model. Available in the Full benchmark, not as a score for one clip. |
| `pieces` / `seeds` | Number of clips or random starts behind the FMD result. | More evidence gives a more dependable comparison. |
| FMD seed `mean` / `std` | Average FMD across seeds and how much it changes between seeds. | Lower average is closer; a large spread means the result depends strongly on randomness. |

These distances have no universal “good” cutoff. Compare runs with the same benchmark inputs, clip lengths, and settings, then listen to representative examples.

## 2. Button collisions

Here the analyzer keeps the musical history and style fixed, changes **only the next button**, and checks how much the next-pitch predictions change. It asks whether different buttons really offer different choices.

| On-screen metric | Plain-language meaning | How to read it |
| --- | --- | --- |
| `pair_collision` | Share of pairs of different buttons whose most likely next pitch is identical. | Lower means fewer buttons collapse onto the same favorite pitch. |
| `adjacent_collision` | Share of neighboring button numbers whose most likely pitch is identical. | Lower is especially useful if buttons are meant to form an ordered scale. |
| `distribution_overlap` | How similar the *full sets of possible next pitches* are for different buttons. | Near 1 means buttons behave almost alike, even if their favorite pitches differ; near 0 means more distinct choices. |
| `distinct_pitches` | Number of different favorite next pitches reached by trying every button. | Higher means broader immediate pitch coverage. This is different from `distinct_pitches` in tab 1, which counts notes across a generated continuation. |
| `button_change_same_pitch` / `button_changes_tested` | During actual generation, how often changing a button leaves the pitch unchanged, and how many changes were checked. | Complements the fixed-context test: a button can differ in isolation yet have little effect in a longer passage. |

A collision is not necessarily a defect: two buttons can legitimately choose the same pitch in one context. Look for a persistent pattern over many contexts.

## 3. Ordered compression

The encoder turns pitches into button values. This tab asks whether the button sequence preserves the **shape** of the original music. The aligned pitch and button piano rolls show the relationship directly. These scores concern the encoder's mapping, not the quality of a generated performance.

| On-screen metric | Plain-language meaning | How to read it |
| --- | --- | --- |
| `rank_correlation` | Whether higher pitches generally receive higher button numbers across the whole excerpt. | Close to 1 follows pitch order; near 0 shows little ordering; negative means mostly reversed. |
| `local_rank_correlation` | The same ordering check over short stretches. | High local scores mean small melodic shapes survive, even if the whole piece changes register. |
| `melodic_direction_reversal` | Share of pitch changes whose button move points the opposite way. | Lower means fewer reversed melodic steps. |
| `collapsed_pitch_changes` | Share of pitch changes that leave the button unchanged. | Lower means fewer musical changes disappear in compression. |
| `repeated_note_consistency` | Share of repeated pitches that keep the same button. | Higher means a stable mapping for repeated notes. |
| `interval_rank_correlation` | Whether bigger pitch jumps tend to produce bigger button jumps. | Higher means the buttons preserve some sense of interval size. |
| `extreme_button_fraction` | Share of notes mapped to the lowest or highest button. | Very high can mean the encoder is hitting its limits and losing distinctions. |
| `button_occupancy` | Fraction of available buttons used at least once. | Shows how much of the control range the excerpt uses; full use is not required. |
| `within_chord_order_preservation` | Share of neighboring notes at the same moment whose low-to-high order is kept in their buttons. | Higher means chord-note ordering survives compression. |
| `changed_button_same_pitch` | Share of melodic button changes that leave the pitch unchanged. | Lower means fewer button moves without a pitch change. |
| `button_histogram` | Count of uses for each button. | Spot buttons that never appear or dominate the mapping. |
| `notes`, `melodic_transitions`, `chord_transitions` | How many notes, successive-note melodic changes, and same-moment note pairs were examined. | These are evidence counts. `chord_transitions` here means note pairs *inside chords*, not changes from one harmony to another. |

Arrow controls are directions, not numbered pitches. They are checked for whether the melody moves as instructed; they are not given ordered-button scores.

## 4. Cross-attention influence

This experiment generates from the **same prompt, controls, and random seed** three times: once with style A, once with contrasting style B, and once with cross-attention switched off. It asks whether changing the reference actually changes the generated harmony.

| On-screen metric | Plain-language meaning | How to read it |
| --- | --- | --- |
| `target_B_affinity` | How similar the generated clip's overall note-name mixture is to style B. | Higher means more similar to B's overall pitch classes. It does not tell you whether B's chords appear in the same order. |
| `target_B_affinity_gain` | Style-B result's affinity to B minus style-A result's affinity to B. | Positive suggests supplying B moved the output toward B compared with supplying A. |
| `style_B_vs_disabled` | Style-B result's affinity to B minus the result with cross-attention off. | Positive suggests the style path influenced the output beyond the prompt alone. |

These are **paired effects**, not claims that style B is better music. Check the audio, chord timeline, and harmonic-evolution tab to see *how* the result changed. A model without cross-attention cannot run this comparison.

## 5. Oracle robustness and the consecutive-button test

Oracle robustness asks what happens when the model's instructions are changed. `shuffled` locally scrambles controls; `ascending` and `descending` request directed movement; `noise_0.05`, `noise_0.15`, and `noise_0.3` replace about 5%, 15%, or 30% of controls with wrong ones; `splice` borrows half the controls from a different piece. `recovery_oracle` first repeats a control, then restores correct controls; `recovery_joker` releases control after that repeated stretch. Joker cases appear only when the model supports them.

The **changes from matched oracle controls** table subtracts the original oracle case from the modified case using the same piece and seed. For example, a negative change in `contour_agreement` means the intervention followed the human contour less closely. For a recovery case, `restricted_...` scores describe its first half and `recovery_...` scores describe its second half. They are the same measurements listed in tab 1, applied to different parts of the continuation.

The **button sweep** is the direct contour test: press every consecutive button upward, then downward. With 19 buttons that is **0 → 18 → 0**. It is run in greedy and sampled modes. The repeated top button marks the turn; it is excluded from the directional percentages.

| On-screen metric | Plain-language meaning | How to read it |
| --- | --- | --- |
| `ascending_strict_fraction` / `descending_strict_fraction` | Share of upward/downward button steps where pitch moves strictly in that direction. | 1 means every changing-button step passed. These are the clearest “does pitch always rise/fall?” scores. |
| `ascending_nondecreasing_fraction` / `descending_nonincreasing_fraction` | Share that move in the intended direction **or** stay on the same pitch. | Higher than the strict score reveals flat steps rather than reversed steps. |
| `ascending_perfect` / `descending_perfect` | Whether **every** step in that direction passes the strict test. | `true` means no repeats or reversals in that half of this particular sweep. |
| `repeated_pitch_steps` | Number of button changes that kept the same pitch. | Lower means fewer flat spots. |
| `reversed_direction_steps` | Number of button changes that moved the pitch the wrong way. | Zero is the ideal for a strictly ordered button mapping. |
| `ascending_mean_semitones` / `descending_mean_semitones` | Average size and direction of each pitch step. | Upward should be positive; downward should be negative. Very large steps may still sound awkward. |
| `tested_button_changes` | Number of changing-button steps checked. | For 19 buttons, there are 36: 18 up and 18 down. |
| `violation_positions` | Positions in the pitch sequence where a repeated or reversed step occurred. | Use these to find the exact spots on the roll or in the step table. |

Greedy tests the model's favorite response; sampled tests what may happen in normal random generation. A checkpoint can pass one and fail the other. The reference roll for this experiment shows the requested up/down **shape**, not exact pitches the model is expected to copy.

## 6. Harmonic evolution

This tab asks whether the generated music keeps developing its harmony at a degree comparable with the **actual prompt and style excerpts** supplied to the model. An evolving reference might change chords, move to another key, or transpose a motif. The tab compares the amount of development, not whether the model copies the exact same progression.

The analyzer looks at overlapping groups of roughly **16 notes** for local chord movement, advancing four notes each time. For key changes it needs broader groups of roughly **64 notes**, advancing eight notes each time. Notes played together stay in the same group, and notes held longer have more influence. A chord or key change is counted only when a new, reasonably certain label lasts for two analysis windows. Short or ambiguous music may still show continuous movement even when it cannot support a reliable chord or key label.

| On-screen metric | Plain-language meaning | How to read it |
| --- | --- | --- |
| `movement_per_100_notes` | Total travel through different harmonic areas, adjusted to 100 notes so excerpt lengths are comparable. | Higher means more harmonic movement, not necessarily better music. |
| `mean_harmonic_change` | Average difference between one nearby harmonic window and the next. | Higher means stronger moment-to-moment change. |
| `tonal_spread` | How widely the excerpt explores different harmonic areas. | Higher means a broader harmonic palette. |
| `maximum_excursion` | Furthest the harmony gets from where it began. | Shows whether it leaves its starting area, even if it later returns. |
| `chord_changes_per_100_notes` | Number of reliable, lasting chord-label changes, adjusted to 100 notes. | Higher means more active chord progression. Uncertain chord guesses are not counted. |
| `distinct_chord_regions` | Number of different reliably identified chord areas. | More regions can mean development; cycling through the same few also counts as activity. |
| `mean_chord_dwell_notes` | Average number of notes spent in a chord area before an accepted change. | Longer means slower harmonic turnover. |
| `confident_chord_fraction` | Share of short windows where the analyzer has a reasonably clear chord label. | Low confidence means chord-change counts need caution. |
| `key_changes_per_100_notes` | Reliable, lasting changes of key, adjusted to 100 notes. | Use only when there is enough music and clear enough key evidence. |
| `confident_key_fraction` | Share of long windows with a reasonably clear key label. | Low confidence makes apparent modulation unreliable. |
| `mean_key_tonal_distance` | Average size of accepted moves between estimated keys. | Larger means farther key moves; unavailable if no reliable key change occurs. |
| `stagnation_fraction` | Share of nearby windows with almost no harmonic movement. | Higher means more static harmony. |
| `longest_plateau_notes` | Longest nearly unchanged harmonic stretch, measured in notes. | A long plateau may be fine for a static prompt but suspicious against an evolving one. |
| `recurrence_fraction` | Share of windows that resemble an earlier, non-neighboring harmonic area. | Higher suggests returns or repeated cycles; low recurrence can mean development **or** aimless wandering. |
| `key_evidence_windows` | Number of broad windows available for judging key. | Few windows mean little evidence for modulation. |

In other tabs these names may start with `harmonic_`; the meaning is the same. Chord and key names are approximate. An octave change alone does not count as a key change, and moving an entire passage up or down together should not change its measured *degree* of harmonic movement.

For each score, the comparison table shows **generated**, **prompt**, and **style reference** values. **Difference** means generated minus reference. **Ratio** means generated divided by reference: for example, movement ratio `0.5` means the generated continuation moved about half as much as that reference. A ratio is shown as `—` when the reference is effectively zero; use the absolute values then. Higher stagnation or a longer plateau than an evolving reference suggests the continuation became more static.

The four-condition experiment combines a **static or evolving prompt** with a **static or evolving style reference**. It holds the generation setup fixed so the conditions can be compared. The Full benchmark also tries ordinary chord progressions, sustained key changes, transposed motifs, and natural MIDI excerpts. An **order ablation** reorders harmonic blocks while keeping their contents; its `difference` tells you how much a score changes because of *order*, rather than simply which notes were present. Models without style use the prompt-only comparison.

Read the visuals together: the timeline shows *when* harmony changes and highlights quiet plateaus; the path plot shows where it travels; the self-similarity image shows returns or cycles. An active path with weak chord/key confidence, poor coherence, or random-sounding audio is **not** strong evidence of successful harmonic development.

## Missing, partial, and unsupported results

| Display | Meaning |
| --- | --- |
| **Queued / running** | The analysis is waiting to start or is still working. Numbers may continue to appear. |
| **Complete** | All applicable cases for that run finished. |
| `—` | No meaningful number is available: for example, the clip is too short, there is no variation to compare, there is no confident key change, or a ratio would divide by nearly zero. |
| **Not applicable** | The checkpoint does not support the feature, such as buttons, jokers, or cross-attention. |
| **Insufficient evidence** | The test could run, but the supplied excerpt or its confidence is inadequate for a fair conclusion. |
| **Configuration required** | The checkpoint is selectable, but its model setup must be identified before it can be loaded and evaluated. |
| **Failed** | That test encountered an error. Other completed cases can still be useful. |
| **Partial / cancelled / interrupted** | Only some cases finished. Completed cases are kept for resumption; treat the aggregate as incomplete. |

Quick is a screen using the existing Bach, Chopin, Debussy, and Satie examples. Full adds fixed test-split pieces, more seeds, longer continuations, and the broader harmonic experiments. **Smoke** is a short functionality check, not a benchmark. Results from different inputs or settings should not be treated as direct checkpoint comparisons.
