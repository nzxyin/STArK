# STArK: Towards Synthesizing Articulatory Kinematics from Text

Official implementation of **STArK**, a non-autoregressive text-to-articulation (TTA) model that
generates articulatory kinematics (EMA + pitch + loudness, in the [SPARC](https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding)
feature space) directly from text, and resynthesizes speech from those features via a frozen
SPARC vocoder.

> STArK: Towards Synthesizing Articulatory Kinematics from Text
> Xavier Yin, Carlos Busso — Interspeech 2026
> [Paper (PDF)](https://lab-msp.com/MSP/publications/Yin_2026.pdf) — temporary link, pending official proceedings

If you use this code, please cite:

```bibtex
@inproceedings{yin2026stark,
  title     = {{STArK}: Towards Synthesizing Articulatory Kinematics from Text},
  author    = {Yin, Xavier and Busso, Carlos},
  booktitle = {Proc. Interspeech 2026},
  year      = {2026}
}
```

STArK builds directly on SPARC; if you use this code, please also cite the SPARC paper (see
[License](#license) below):

```bibtex
@article{cho2024coding,
  title   = {Coding Speech through Vocal Tract Kinematics},
  author  = {Cho, Cheol Jun and Wu, Peter and Prabhune, Tejas S. and Agarwal, Dhruv and Anumanchipalli, Gopala K.},
  journal = {IEEE Journal of Selected Topics in Signal Processing},
  volume  = {18},
  number  = {8},
  pages   = {1427--1440},
  year    = {2024}
}
```

## Setup

Tested on Python 3.11–3.12 (`pyproject.toml` requires `>=3.11,<3.13`).

```bash
uv sync
```

Phonemization uses [Phonemizer](https://github.com/bootphon/phonemizer) with the
[eSpeak NG](https://github.com/espeak-ng/espeak-ng) backend, which is a system binary and must be
installed separately:

```bash
# Debian/Ubuntu
sudo apt-get install espeak-ng
# macOS
brew install espeak-ng
```

## Reproducing the paper

The exact configuration used for the paper's reported results is `model=large_model
train=train_large` (with the default `preprocess=default_preprocess`), evaluated at
**checkpoint step 32000**. This also matches the paper's stated 10,000-step warmup:
`train_large`'s `max_steps=500000` × the model's 2% warmup fraction = 10,000 steps.

```bash
python train.py train=train_large model=large_model
```

`train_preempt.sh` runs this exact command on the `preempt` SLURM partition (checkpointed every
2000 steps, auto-resumes from the last checkpoint so it's safe to preempt/requeue). Everything
else under `conf/` — `small_model`, `spk_dur_cond_model`, `spk_full_cond_model`,
`libritts_clean_360_preprocess`, `ljspeech_preprocess`, `ljspeech_large` (and `train.sh`, which
runs the LJSpeech config) — are separate, unpublished experiments in this same codebase, **not**
used for the Interspeech 2026 paper.

Inference and all reported metrics (DTW/Pearson correlation against SPARC-extracted ground
truth, DNSMOS, WER, SECS, human NMOS) are computed in [`testing.ipynb`](testing.ipynb); see the
top of that notebook for the one place to set your local data/checkpoint paths.

## Data

The paper trains and evaluates only on LibriTTS-R's `train-clean-100` (+ `dev-clean`/`test-clean`
for evaluation), using SPARC-derived pseudo-labels as targets.

### Quick start: precomputed features (recommended)

A precomputed, training-ready feature set (phoneme ids, alignment priors, speaker embeddings,
and normalized EMA/pitch/loudness targets — everything except the raw audio, which is already
public) is published as a Hugging Face dataset repo. On the Hub the files are sharded into 256
hash-bucket subdirectories per modality (the Hub's git backend rejects any single directory with
more than 10,000 files, and `train-clean-100` alone has ~33k utterances), so after downloading,
run `scripts/materialize_hf_dataset.py` to symlink it into the flat layout `LibriTTSDataset`
actually expects (see [Preprocessed data layout](#preprocessed-data-layout) below) — no data is
duplicated, and no custom `datasets` loading script is needed:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="nzxyin/libritts-r-stark",  # will move to a Lab-MSP namespace later
    repo_type="dataset",
    local_dir="/path/to/libritts-r-stark-download",
)
```

```bash
python scripts/materialize_hf_dataset.py \
    --downloaded_root /path/to/libritts-r-stark-download \
    --dataset_root /path/to/LibriTTS_R
```

Raw LibriTTS-R audio itself doesn't need to be re-downloaded through this dataset — it's already
public (CC-BY-4.0) as [`mythicinfinity/libritts_r`](https://huggingface.co/datasets/mythicinfinity/libritts_r)
if you need the `.wav` files for anything beyond training (e.g. listening to ground truth).

Once downloaded, point `preprocess.dataset.dataset_root` (in `conf/preprocess/default_preprocess.yaml`,
or via `preprocess.dataset.dataset_root=...` on the command line) at that directory — it already
contains `train-clean-100-preprocessed/`, `dev-clean-preprocessed/`, `test-clean-preprocessed/`,
and the three `{split}.json` split manifests.

### Preprocessed data layout

Each `{split}-preprocessed/` directory (`train-clean-100`, `dev-clean`, `test-clean`) is flat —
there's no `sparc/` sub-nesting:

```
LibriTTS_R/
├── train-clean-100.json          # canonical list of usable utterance ids for this split
├── dev-clean.json
├── test-clean.json
├── train-clean-100-preprocessed/
│   ├── wav/                      # {id}.wav
│   ├── original_txt/             # {id}.txt
│   ├── normalized_txt/           # {id}.txt
│   ├── phn/                      # {id}.phones.txt   — phoneme sequence (text)
│   ├── phn_ids/                  # {id}.phones.npy   — phoneme ids, shape (T_text,)
│   ├── dur/                      # {id}.dur.npy      — beta-binomial alignment prior, shape (T_sparc, T_text)
│   └── emasrc/                   # {id}.ema.npy      — 14-dim EMA+pitch+loudness targets, shape (T_sparc, 14)
├── dev-clean-preprocessed/       # same structure
└── test-clean-preprocessed/      # same structure
```

`dur/` is **not** a ground-truth forced alignment — it's the beta-binomial prior
(`src/tts/dataset.py`'s `BetaBinomialInterpolator`) fed to STArK's unsupervised aligner during
training, matching the paper's "One TTS Alignment"-style approach (no external aligner
dependency).

### Regenerating the data from scratch

Not required to train (the Hugging Face dataset above is training-ready), but documented here
for provenance/verification. Starting from raw LibriTTS-R audio + transcripts:

1. **SPARC feature extraction** (raw audio → 15-dim EMA/pitch/periodicity/loudness + 64-dim
   speaker embeddings): run the [SPARC](https://github.com/nzxyin/Speech-Articulatory-Coding)
   `en+` model over each split's audio to produce a `{split}-sparc/emasrc/` and
   `{split}-sparc/spk_emb/` directory (15-dim EMA features per utterance, before pitch
   normalization). This step isn't scripted in this repo — see the SPARC repo directly.
2. **Pitch normalization + speaker embedding staging** ([`process_sparc.py`](process_sparc.py),
   driven by [`preprocess_sparc_libritts.sh`](preprocess_sparc_libritts.sh) as a SLURM array job
   over all four LibriTTS-R splits): reads `{split}-sparc/emasrc/` (15-dim), log-normalizes pitch
   by each utterance's median pitch (Eq. 1 in the paper), drops the periodicity channel to
   produce the 14-dim target, and writes `{split}-preprocessed/emasrc/` +
   `{split}-preprocessed/spk_emb/` + a `pitch_stats.json` (per-utterance median pitch, needed to
   denormalize predicted pitch back to a target speaker's range at inference time).
3. **Phonemization** ([`ipa.py`](ipa.py) defines the IPA symbol set and phoneme id mapping; see
   `preprocess.ipynb` for example driver code): text → phoneme sequence via Phonemizer/eSpeak-NG
   → `{split}-preprocessed/phn/` (text) and `phn_ids/` (ids).
4. **Alignment prior** (`src/tts/dataset.py`'s `BetaBinomialInterpolator`; see `preprocess.ipynb`
   for example driver code): computed from `phn_ids/` length and `{split}-sparc/emasrc/` length
   → `{split}-preprocessed/dur/`.
5. Raw `.wav`/`original_txt`/`normalized_txt` are staged into `{split}-preprocessed/` directly
   from the LibriTTS-R corpus (same per-utterance ids).
6. Each `{split}.json` is the list of utterance ids that have **all** of `dur`/`emasrc`/`spk_emb`
   successfully produced (a small number of utterances — a few dozen out of ~33k for
   `train-clean-100` — fail SPARC extraction/alignment and are excluded).

`process_sparc.py` is shared between the LibriTTS-R and LJSpeech pipelines; the two `Dataset`
classes expect different EMA output directory names (`emasrc` vs `ema_preprocessed`), so pass
`--ema_output_dirname` accordingly (`preprocess_sparc_libritts.sh` uses the `emasrc` default;
`preprocess_sparc_ljspeech.sh` passes `--ema_output_dirname ema_preprocessed`).

## Inference

See [`testing.ipynb`](testing.ipynb) for the full inference + evaluation pipeline: loading a
trained checkpoint, predicting articulatory features, synthesizing speech via the frozen SPARC
vocoder, and computing all metrics reported in the paper. Set the data/output paths in the first
few cells before running.

## License

This repository's original code is MIT-licensed. `src/tts/sparc/` is adapted from the SPARC
codebase (Cho et al., Berkeley Speech Group), used and modified with the original authors'
permission — see [`LICENSE`](LICENSE) for the full terms and citation requirements.
