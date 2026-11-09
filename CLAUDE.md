# STArK: notes for agents

STArK is the official code for "STArK: Towards Synthesizing Articulatory Kinematics from Text"
(Interspeech 2026). Changes go in `CHANGELOG.md`; this file holds current state and direction.

## Current state (2026-11-09)

- The paper checkpoint is step 32000 of `model=large_model train=train_large`. It is published as
  `checkpoint.ckpt` on `main` of the Hugging Face repo `Lab-MSP/stark-large` (tag
  `paper-step-32k`), which `scripts/stark_cli.py` downloads by default.
- A separate 100,000-step run was published there from 2026-08-23 to 2026-11-05 by mistake. Users
  reported it does not match the paper. It is kept only at the Hub tag `step-100k`. Do not push it
  back to `main`.
- `samples/` and the project site's demo audio were generated with the step-32000 checkpoint.
- Known open bug: `stark_cli.py --device cpu` crashes because SPARC is always loaded on CUDA
  (Lab-MSP/STArK#11).
