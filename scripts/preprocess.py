"""Preprocessing steps 3, 4, and 6 of TRAINING.md's "Regenerating the data from scratch"
pipeline (steps 1/2/5 — SPARC feature extraction, pitch normalization via `process_sparc.py`,
and raw audio/text staging — are covered elsewhere; see TRAINING.md).

Replaces the old exploratory `preprocess.ipynb` with a `uv`-runnable script. Only implements the
LibriTTS-R path (the paper's only dataset) — the notebook it replaces also had ad-hoc,
never-documented cells for an unrelated GLOBE-corpus experiment and for LJSpeech (a separate,
unpublished experiment not part of the Interspeech 2026 paper; see TRAINING.md's history if you
need that code — it was intentionally dropped here, not migrated).

Subcommands, run in this order for each split (dev-clean, test-clean, train-clean-100):

  phonemize   normalized_txt/{id}.txt -> phn/{id}.phones.txt (text) and phn_ids/{id}.phones.npy
              (ids), via src/tts/g2p.py's Phonemizer/eSpeak-NG reimplementation. NOTE: this is a
              best-effort reconstruction of the original (unshipped) phonemization pipeline
              (same caveat as g2p.py's docstring) — if you have your own phn/ directory already
              (e.g. from the original pipeline), skip this and go straight to `phn-to-ids`.
  phn-to-ids  phn/{id}.phones.txt -> phn_ids/{id}.phones.npy only, for an existing phn/ directory
              not produced by this script's `phonemize` step.
  durations   phn_ids/ + emasrc/ (both under --preprocessed_dir, i.e. *after* process_sparc.py
              has already run) -> dur/{id}.dur.npy, the beta-binomial alignment prior fed to
              STArK's unsupervised aligner (BetaBinomialInterpolator). Run this after
              process_sparc.py, not before — it reads the normalized 14-dim emasrc/, not the
              raw 15-dim SPARC output.
  splits      scans a {split}-preprocessed/ directory and writes {split}.json: the list of
              utterance ids with all of dur/, emasrc/, and spk_emb/ successfully produced.

Usage:
    uv run scripts/preprocess.py phonemize \
        --preprocessed_dir /data/user_data/xoy/LibriTTS_R/train-clean-100-preprocessed
    uv run scripts/preprocess.py durations \
        --preprocessed_dir /data/user_data/xoy/LibriTTS_R/train-clean-100-preprocessed
    uv run scripts/preprocess.py splits \
        --preprocessed_dir /data/user_data/xoy/LibriTTS_R/train-clean-100-preprocessed \
        --dataset_root /data/user_data/xoy/LibriTTS_R \
        --split_name train-clean-100
"""
import argparse
import json
import os
import sys

# tts/dataset.py and ipa.py are bare-imported off the repo root (`from utils import ...`,
# `from ipa import ...`) — only resolves if the repo root is on sys.path. Running this script
# directly off the root (`python preprocess.py`) gets this for free; running it from scripts/
# (`python scripts/preprocess.py`) does not, since Python puts the *script's* directory on
# sys.path[0], not the repo root. Same pattern as scripts/eval_and_push.py etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm


def cmd_phonemize(args):
    from ipa import get_id
    from tts.g2p import text_to_phonemes

    text_dir = os.path.join(args.preprocessed_dir, "normalized_txt")
    phn_dir = os.path.join(args.preprocessed_dir, "phn")
    phn_ids_dir = os.path.join(args.preprocessed_dir, "phn_ids")
    os.makedirs(phn_dir, exist_ok=True)
    os.makedirs(phn_ids_dir, exist_ok=True)

    failed = []
    for filename in tqdm(sorted(os.listdir(text_dir)), desc="phonemizing"):
        basename = filename.rsplit(".", 1)[0]
        try:
            text = open(os.path.join(text_dir, filename)).read().strip()
            symbols = text_to_phonemes(text)
            with open(os.path.join(phn_dir, f"{basename}.phones.txt"), "w") as f:
                f.write(" ".join(symbols))
            ids = np.array([get_id(s) for s in symbols], dtype=np.int64)
            np.save(os.path.join(phn_ids_dir, f"{basename}.phones.npy"), ids)
        except Exception as e:
            failed.append(filename)
            print(f"  failed on {filename}: {e}")
    print(f"Phonemized {len(os.listdir(text_dir)) - len(failed)}/{len(os.listdir(text_dir))} "
          f"utterances into {phn_dir} + {phn_ids_dir}")
    if failed:
        print(f"  {len(failed)} failures: {failed}")


def cmd_phn_to_ids(args):
    from ipa import get_id

    phn_dir = os.path.join(args.preprocessed_dir, "phn")
    phn_ids_dir = os.path.join(args.preprocessed_dir, "phn_ids")
    os.makedirs(phn_ids_dir, exist_ok=True)

    failed = []
    for filename in tqdm(sorted(os.listdir(phn_dir)), desc="converting phn -> ids"):
        basename = filename.rsplit(".phones.txt", 1)[0]
        try:
            symbols = open(os.path.join(phn_dir, filename)).read().strip().split()
            symbols = [s for s in symbols if s != ""]
            ids = np.array([get_id(s) for s in symbols], dtype=np.int64)
            np.save(os.path.join(phn_ids_dir, f"{basename}.phones.npy"), ids)
        except Exception as e:
            failed.append(filename)
            print(f"  failed on {filename}: {e}")
    print(f"Converted {len(os.listdir(phn_dir)) - len(failed)}/{len(os.listdir(phn_dir))} "
          f"files into {phn_ids_dir}")
    if failed:
        print(f"  {len(failed)} failures: {failed}")


def cmd_durations(args):
    from tts.dataset import BetaBinomialInterpolator

    phn_ids_dir = os.path.join(args.preprocessed_dir, "phn_ids")
    emasrc_dir = os.path.join(args.preprocessed_dir, "emasrc")
    durations_dir = os.path.join(args.preprocessed_dir, "dur")
    os.makedirs(durations_dir, exist_ok=True)

    binomial_interpolator = BetaBinomialInterpolator()
    failed = []
    filenames = sorted(os.listdir(phn_ids_dir))
    for filename in tqdm(filenames, desc="computing alignment priors"):
        # phn_ids/{id}.phones.npy <-> emasrc/{id}.ema.npy (process_sparc.py's normalized
        # 14-dim output — this must run *after* process_sparc.py, not on the raw SPARC dir).
        basename = filename.replace(".phones.npy", "")
        sparc_filename = f"{basename}.ema.npy"
        try:
            text = np.load(os.path.join(phn_ids_dir, filename))
            sparc = np.load(os.path.join(emasrc_dir, sparc_filename))
            assert sparc.shape[1] == 14, f"expected normalized 14-dim emasrc, got {sparc.shape}"
            # Shape (text_len, sparc_len), matching LibriTTSDataset's own assertion
            # (duration.shape == (phone.shape[0], sparc.shape[0])) — LitTTS's collate_fn
            # transposes back to (sparc_len, text_len) at batch time.
            durations = binomial_interpolator(len(text), len(sparc))
            np.save(os.path.join(durations_dir, f"{basename}.dur.npy"), durations)
        except FileNotFoundError:
            failed.append(filename)
            print(f"  {sparc_filename} not found in {emasrc_dir}, skipping")
        except Exception as e:
            failed.append(filename)
            print(f"  failed on {filename}: {e}")
    print(f"Computed durations for {len(filenames) - len(failed)}/{len(filenames)} utterances "
          f"into {durations_dir}")
    if failed:
        print(f"  {len(failed)} failures/skips: {failed}")


def cmd_splits(args):
    dur_ids = {f.replace(".dur.npy", "") for f in os.listdir(os.path.join(args.preprocessed_dir, "dur"))}
    emasrc_ids = {f.replace(".ema.npy", "") for f in os.listdir(os.path.join(args.preprocessed_dir, "emasrc"))}
    spk_emb_ids = {f.replace(".npy", "") for f in os.listdir(os.path.join(args.preprocessed_dir, "spk_emb"))}

    complete_ids = sorted(dur_ids & emasrc_ids & spk_emb_ids)
    incomplete = (dur_ids | emasrc_ids | spk_emb_ids) - set(complete_ids)

    out_path = os.path.join(args.dataset_root, f"{args.split_name}.json")
    with open(out_path, "w") as f:
        json.dump(complete_ids, f)
    print(f"Wrote {len(complete_ids)} complete utterance ids to {out_path}")
    if incomplete:
        print(f"  {len(incomplete)} utterance(s) missing dur/emasrc/spk_emb (excluded): "
              f"{sorted(incomplete)[:10]}{'...' if len(incomplete) > 10 else ''}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("phonemize", help="normalized_txt/ -> phn/ + phn_ids/ via g2p.py")
    p.add_argument("--preprocessed_dir", required=True, help="e.g. .../train-clean-100-preprocessed")
    p.set_defaults(func=cmd_phonemize)

    p = sub.add_parser("phn-to-ids", help="existing phn/ -> phn_ids/ only (ipa.py id mapping)")
    p.add_argument("--preprocessed_dir", required=True)
    p.set_defaults(func=cmd_phn_to_ids)

    p = sub.add_parser("durations", help="phn_ids/ + emasrc/ -> dur/ (beta-binomial alignment prior)")
    p.add_argument("--preprocessed_dir", required=True,
                    help="e.g. .../train-clean-100-preprocessed; must already have emasrc/ (i.e. run process_sparc.py first)")
    p.set_defaults(func=cmd_durations)

    p = sub.add_parser("splits", help="preprocessed dir -> {split}.json manifest")
    p.add_argument("--preprocessed_dir", required=True, help="e.g. .../train-clean-100-preprocessed")
    p.add_argument("--dataset_root", required=True, help="where to write {split_name}.json, e.g. .../LibriTTS_R")
    p.add_argument("--split_name", required=True, help="e.g. train-clean-100")
    p.set_defaults(func=cmd_splits)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
