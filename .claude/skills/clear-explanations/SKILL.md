---
name: clear-explanations
description: How to explain things to the user in the monster_genie project. Use for every reply that reports test, evaluation or training results, presents a plan, diagnoses a problem, explains an idea or answers a question, so that the conclusion comes first in plain words, every term is defined and every number says what it measures. Also covers error explanations and corrections of the user's terminology.
---

# Clear Explanations

The user reads a reply without having seen the scripts, the output or my reasoning. Working labels and shorthand feel shared to me only because I have just been using them. Write so that a reply can be understood on the first reading, without decoding it and without remembering earlier replies.

Clear does not mean long. Use plain words and full sentences, and leave out findings that do not bear on the question instead of compressing them into shorthand.

## Order of a result

1. The answer to what was asked, in one or two plain sentences, including what it means for the model or for playing and how sure it is.
2. What was done to find out: which model, which data, how many examples, what changed and what stayed the same.
3. The evidence: only the numbers that support the answer.
4. What the test does not show, and the next decision if there is one.

## Terms

- Define every metric, abbreviation, statistic and name at its first use in each reply, in one clause: what it measures, on what, and which direction is better. This includes standard terms (NLL, entropy, log-odds, rank correlation, p-value) and terms defined in earlier replies. A one-clause definition is not a repeated explanation.
- Prefer the description to the label: "the probability the model gives to the notes of the held chord", not "chord mass". Do not coin new labels. If a term has to recur, keep one name for it throughout the reply.
- Names of test conditions ("free", "fresh", "sustained", "tritone") are labels too. Say what was done: "no chord pressed", "control applied only to the newest note".
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

## Ideas and plans

- Start with the problem as the user sees or hears it, then the idea, then a concrete example (one chord press with its actual notes) before the general description.
- For each step of a plan, say what changes, why, what it costs in time or training, and how we will know it worked.
- Separate what was measured, what I infer from it, and what is only a guess.

## Examples

Unclear:

> Guided NLL is 0.9–1.6 in both, and joker NLL is 2.7–3.4. Earlier unseen-test figures, from a different sample, were about 0.48 and 2.1.

Clear:

> Your recordings surprise the model more than the test pieces do, whether or not a chord is held. I measured the probability the model gave to each note you really played (NLL, negative log-likelihood: lower is better, and e^(-NLL) is the typical probability). When it knows the button for that note (guided), it typically gives the note 20–40 %. With no button (joker), it gives 3–7 %. On test pieces it gives about 62 % and 12 %, but those came from a different sample, so the comparison is rough.

Unclear:

> Same notes, only hold length L changes (40 moments): gain over free at L=1/16/64: guided real +.088/+.061/+.045, joker real +.132/+.059/+.043.

Clear:

> Holding a chord longer weakens its effect: after 64 notes the push is half as strong with a known button and a third as strong without one. At 40 moments from pieces outside the training data, I pressed the real chord and changed only how many notes the key had already been held, keeping the played notes the same. The table shows how much more probability the chord's notes get than with no chord pressed, in percentage points.
>
> | Button input | Held for 1 note | Held for 16 notes | Held for 64 notes |
> |---|---|---|---|
> | Button known (guided) | +8.8 | +6.1 | +4.5 |
> | No button (joker) | +13.2 | +5.9 | +4.3 |

## Focused communication and corrections

- For a specific error, explain its concrete cause and the smallest fix. Avoid repeated explanations, broad tutorials, and speculative future steps.
- Explain the changed logic briefly enough that the user can understand it, including non-obvious tensor dimensions or types when relevant.
- Point out every identifiable error in the user's terminology, conventions, or understanding, even when unrelated to the immediate task. Keep corrections concise, distinguish definite mistakes from uncertainty or preference, and do not invent errors.
- Report the result and relevant verification accurately. Keep unrelated corrections brief so they do not obscure the requested solution.

## Final check

Before sending, take each term and number in the reply and read it as someone who has not seen the run. If it cannot be said aloud as a full sentence ("X % of what, measured on what, compared with what"), rewrite it.
