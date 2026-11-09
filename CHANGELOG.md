# Changelog

## 2026-11-09

- The Hugging Face model `Lab-MSP/stark-large` now serves the Interspeech 2026 paper checkpoint
  (step 32000) on `main` (tag `paper-step-32k`). It previously served a separate 100,000-step
  run, which is kept at tag `step-100k`.
- `TRAINING.md`: documents which checkpoint the Hub repo serves and how to get the 100k run.
- `samples/`: regenerated all six STArK samples with the step-32000 checkpoint, using the same
  sentences and reference speakers as before. The `*_gt.wav` ground-truth files are unchanged.
- `scripts/eval_and_push.sh` now pushes the step-32000 paper checkpoint instead of the 100k run, so
  running it can no longer put the 100k model back on `Lab-MSP/stark-large`.
