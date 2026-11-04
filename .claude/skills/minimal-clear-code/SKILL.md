---
name: minimal-clear-code
description: Coding preferences for the monster_genie project. Use for every code change in this project — new code, bug fixes, PyTorch/GPU code — to keep changes minimal and readable and tensor code batched.
---

# Minimal Clear Code

Make the smallest clear change that fully solves the requested problem. Apply these preferences within the user's requested scope; a more specific instruction for the current task takes precedence.

## Minimal, understandable code

- Preserve the original structure, naming, interfaces, behavior, and formatting wherever the requested change permits. Avoid unrelated refactoring, cleanup, abstractions, dependencies, and speculative checks.
- Use only the code necessary for the task. Favor readable expressions and descriptive names over compressed or clever code.
- Keep existing comments, including explanations of previous ideas. If a changed behavior makes a comment inaccurate, update the explanation while preserving useful historical context; do not delete comments as cleanup.
- Add concise comments where they help the user understand non-obvious intent, tensor shapes, or necessary tradeoffs. Avoid comments that merely repeat the code.
- Do not create extra documentation, Markdown files, or test files unless explicitly requested. Use existing focused checks or a temporary in-memory check when verification is needed, without adding a testing framework or expanding the task.

## GPU computation

- Use tensor operations instead of Python loops over tensor elements or batches. Prefer broadcasting, indexing, masking, reductions, and batched operations that preserve the intended result.
- Preserve device, dtype, shape, and gradient behavior. Avoid unnecessary CPU transfers, scalar extraction, synchronization, and large temporary tensors.
- If dependencies between iterations prevent a correct tensor formulation, explain that specific limitation before choosing an alternative. Do not claim a speed or memory improvement without evidence.
