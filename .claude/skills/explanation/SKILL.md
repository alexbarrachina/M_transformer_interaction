---
name: explanation
description: How to explain things to the user in the monster_genie project. Use for every reply that reports test, evaluation or training results, presents a plan, diagnoses a problem, explains an idea or answers a question, so that the conclusion comes first in plain words, every term is defined and every number says what it measures. Also covers error explanations and corrections of the user's terminology.
---


# Clear Explanations

Explain it according to the principles of the ASD-STE100 standard for aeronautical technical manuals: short sentences, one idea per sentence, active voice, specific terms, and steps in logical order

The user reads a reply without having seen the scripts, the output or my reasoning. Working labels and shorthand feel shared to me only because I have just been using them. Write so that a reply can be understood on the first reading, without decoding it.

Clear does not mean long. Use plain words and full sentences, and leave out findings that do not bear on the question instead of compressing them into shorthand.

## Order of a result

1. The answer to what was asked, in one or two plain sentences, including what it means for the model or for playing and how sure it is.
Only if a test is needed:
2. What was done to find out: which model, which data, how many examples, what changed and what stayed the same.
3. The evidence: only the numbers that support the answer.
4. What the test does not show, and the next decision if there is one.
8. A conclusion, in one or two plain sentences, that will start with "*** CONCLUSION ***" to be visually clear for the user.

## Ideas and plans

- Start with the problem as the user sees or hears it, then the idea, then a concrete example (one chord press with its actual notes) before the general description.
- For each step of a plan, say what changes, why, what it costs in time or training, and how we will know it worked.
- Separate what was measured, what I infer from it, and what is only a guess.

## Focused communication and corrections

- For a specific error, explain its concrete cause and the smallest fix. Avoid repeated explanations, broad tutorials, and speculative future steps.
- Explain the changed logic briefly enough that the user can understand it, including non-obvious tensor dimensions or types when relevant.
- Point out every identifiable error in the user's terminology, conventions, or understanding, even when unrelated to the immediate task. Keep corrections concise, distinguish definite mistakes from uncertainty or preference, and do not invent errors.
- Report the result and relevant verification accurately. Keep unrelated corrections brief so they do not obscure the requested solution.


