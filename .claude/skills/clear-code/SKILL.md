---
name: clear-code
description: Coding preferences for the monster_genie project. Use for every code change in this project — new code, bug fixes, PyTorch/GPU code — to keep changes minimal and readable and tensor code batched. Don't assume.
---
# Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

Ask me any questions you need to refine the task—no more than 5, and one at a time. Give me some answer options and the option to add my own.

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.

# Simplicity first

Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

## Minimal, understandable code

- Preserve the original structure, naming, interfaces, behavior, and formatting wherever the requested change permits. Avoid unrelated refactoring, cleanup, abstractions, dependencies, and speculative checks.
- Use only the code necessary for the task. Favor readable expressions and descriptive names over compressed or clever code.
- Keep existing comments, including explanations of previous ideas. If a changed behavior makes a comment inaccurate, update the explanation while preserving useful historical context; do not delete comments as cleanup.
- Add concise comments where they help the user understand non-obvious intent, tensor shapes, or necessary tradeoffs. Avoid comments that merely repeat the code.
- Do not create extra documentation, Markdown files, or test files unless explicitly requested. Use existing focused checks or a temporary in-memory check when verification is needed, without adding a testing framework or expanding the task. Tests that measure model behaviour follow the simple-evidence-tests skill.

## GPU computation

- Use tensor operations instead of Python loops over tensor elements or batches. Prefer broadcasting, indexing, masking, reductions, and batched operations that preserve the intended result.
- Preserve device, dtype, shape, and gradient behavior. Avoid unnecessary CPU transfers, scalar extraction, synchronization, and large temporary tensors.
- If dependencies between iterations prevent a correct tensor formulation, explain that specific limitation before choosing an alternative. Do not claim a speed or memory improvement without evidence.
