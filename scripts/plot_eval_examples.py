"""Generate visualizations from a checkpoint's test-clean evaluation:

  1. Per-utterance articulatory trace plots — predicted vs ground-truth EMA/pitch/loudness over
     time, for a handful of example test-clean utterances. Follows the same visual convention
     already established in testing.ipynb's plot_sparc() (14 stacked channel subplots; green =
     Setup 1 predicted-duration prediction, orange dashed = Setup 2 aligner-duration prediction,
     purple dotted = ground truth), cleaned up for standalone PNG output.
  2. A summary bar chart of the aggregate DTW/PCC metrics (reads eval_full_testset.py's results
     JSON if available, otherwise recomputes for just the plotted examples).

Usage:
    uv run scripts/plot_eval_examples.py \
        --checkpoint /data/user_data/xoy/articulatory-tts/stark_large_100k/ckpt/last.ckpt \
        --dataset_root /data/user_data/xoy/LibriTTS_R/ \
        --results_json /data/user_data/xoy/articulatory-tts/stark_large_100k/testset_eval_20260818_122038.json \
        --n 4 \
        --out_dir /data/user_data/xoy/articulatory-tts/stark_large_100k/plots
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import hydra
from hydra.core.global_hydra import GlobalHydra
import lightning as L
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tts

NAMES = ["ULX", "ULY", "LLX", "LLY", "LIX", "LIY", "TTX", "TTY", "TBX", "TBY", "TDX", "TDY", "Pitch", "Loudness"]


def build_config(overrides):
    GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path="../conf"):
        config = hydra.compose(config_name="config", overrides=overrides)
    config["train"]["logger"]["save_dir"] = config["train"]["logger"]["save_dir"].format(experiment_name=config["train"]["experiment_name"])
    config["train"]["checkpoint"]["dirpath"] = config["train"]["checkpoint"]["dirpath"].format(experiment_name=config["train"]["experiment_name"])
    return config


@torch.no_grad()
def run_pipeline(checkpoint_path, base_config, use_aligner, n, batch_size=8):
    """First n test-clean utterances (dataset order), same code path as eval_full_testset.py."""
    config = dict(base_config)
    config["model"] = dict(config["model"]); config["model"]["tts"] = dict(config["model"]["tts"])
    config["model"]["tts"]["use_aligner_durations_if_possible"] = use_aligner
    config["train"] = dict(config["train"]); config["train"]["datamodule"] = dict(config["train"]["datamodule"])
    config["train"]["datamodule"]["batch_size"] = batch_size
    config["train"]["datamodule"]["num_workers"] = 0
    config["train"]["trainer"] = dict(config["train"]["trainer"])
    config["train"]["trainer"]["devices"] = 1
    config["train"]["trainer"]["strategy"] = "auto"
    config["train"]["trainer"]["limit_predict_batches"] = max(1, n // batch_size + 1)

    model = tts.LitTTS.load_from_checkpoint(checkpoint_path, config=config, map_location="cpu",
                                             weights_only=False, strict=False)
    trainer = L.Trainer(**config["train"]["trainer"])
    datamodule = tts.LibriTTSDataModule(config)
    outputs = trainer.predict(model, datamodule)

    predictions = {}
    count = 0
    for batch in outputs:
        ids = batch[0]
        sparc, mask = batch[1][0], batch[1][1]
        for i in range(len(ids)):
            if count >= n:
                break
            predictions[ids[i]] = sparc[i][: (~mask[i]).sum(), :].cpu().numpy()
            count += 1
        if count >= n:
            break
    return predictions


def plot_traces(uid, text, setup1_pred, setup2_pred, ground_truth, out_path):
    """14 stacked channel subplots: prediction (Setup 1, predicted durations) vs aligned
    (Setup 2, aligner ground-truth durations) vs ground truth, matching testing.ipynb's
    plot_sparc() color convention."""
    fig, axes = plt.subplots(14, 1, figsize=(11, 16), sharex=False)
    fig.suptitle(f"{uid}\n\"{text}\"", fontsize=11, y=0.995)

    for i, name in enumerate(NAMES):
        ax = axes[i]
        ax.plot(ground_truth[:, i], color="#6b3fa0", linestyle=":", linewidth=2.2, label="ground truth")
        # Setup 2 (aligner durations) has the same frame count as ground truth by construction,
        # so it shares the ground-truth x-axis; Setup 1 (predicted durations) generally has a
        # different length (it's the true end-to-end scenario) and gets its own x-axis.
        ax.plot(setup2_pred[:, i], color="#dd8452", linestyle="--", linewidth=1.6, label="predicted (aligner durations)")
        ax2 = ax.twiny()
        ax2.plot(setup1_pred[:, i], color="#55a868", linewidth=1.6, alpha=0.85, label="predicted (predicted durations)")
        ax2.set_xticks([])
        ax.set_ylabel(name, rotation=0, labelpad=22, fontsize=9, va="center")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7, ncol=1)
        if i < 13:
            ax.set_xticklabels([])
    axes[-1].set_xlabel("frame (ground truth / Setup 2 axis)", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_summary(results_json, out_path):
    with open(results_json) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    s1 = results["setup1_predicted_durations"]
    s2 = results["setup2_aligner_durations"]
    dtw_labels = ["EMA", "Pitch", "Loudness"]
    dtw_s1 = [s1["dtw_ema"]["mean"], s1["dtw_pitch"]["mean"], s1["dtw_loudness"]["mean"]]
    dtw_s1_ci = [s1["dtw_ema"]["ci95"], s1["dtw_pitch"]["ci95"], s1["dtw_loudness"]["ci95"]]
    dtw_s2 = [s2["dtw_ema"]["mean"], s2["dtw_pitch"]["mean"], s2["dtw_loudness"]["mean"]]
    dtw_s2_ci = [s2["dtw_ema"]["ci95"], s2["dtw_pitch"]["ci95"], s2["dtw_loudness"]["ci95"]]
    x = np.arange(3)
    w = 0.35
    ax = axes[0]
    ax.bar(x - w / 2, dtw_s1, w, yerr=dtw_s1_ci, label="Setup 1 (predicted durations)", color="#55a868", capsize=3)
    ax.bar(x + w / 2, dtw_s2, w, yerr=dtw_s2_ci, label="Setup 2 (aligner durations)", color="#dd8452", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(dtw_labels)
    ax.set_ylabel("DTW distance (lower = better)")
    ax.set_title(f"DTW vs ground truth (N={s1['dtw_ema']['n']})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    pcc_labels = ["EMA", "Pitch", "Loudness"]
    pcc_vals = [s2["pearson_ema"]["mean"], s2["pearson_pitch"]["mean"], s2["pearson_loudness"]["mean"]]
    pcc_ci = [s2["pearson_ema"]["ci95"], s2["pearson_pitch"]["ci95"], s2["pearson_loudness"]["ci95"]]
    paper_vals = [0.905, 0.533, 0.796]
    x = np.arange(3)
    ax.bar(x - w / 2, pcc_vals, w, yerr=pcc_ci, label="this checkpoint", color="#4c72b0", capsize=3)
    ax.bar(x + w / 2, paper_vals, w, label="paper (Table 2, converged)", color="#c44e52", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(pcc_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pearson correlation (higher = better)")
    ax.set_title("PCC vs ground truth, Setup 2 (aligner durations)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    if "audio_quality" in results:
        aq = results["audio_quality"]
        print("(audio_quality present in results but not plotted here — see JSON directly)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--split", default="test-clean")
    parser.add_argument("--overrides", nargs="*", default=["model=large_model", "train=train_large"])
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--results_json", default=None, help="eval_full_testset.py output, for the summary bar chart")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    preprocessed = os.path.join(args.dataset_root, f"{args.split}-preprocessed")
    ids = json.load(open(os.path.join(args.dataset_root, f"{args.split}.json")))
    sample_ids = ids[: args.n]
    print(f"Using first {args.n} utterance ids (dataset order): {sample_ids}")

    overrides = list(args.overrides) + [f"preprocess.dataset.dataset_root={args.dataset_root}"]
    base_config = build_config(overrides)

    ground_truth = {uid: np.load(os.path.join(preprocessed, "emasrc", f"{uid}.ema.npy")) for uid in sample_ids}
    texts = {uid: open(os.path.join(preprocessed, "normalized_txt", f"{uid}.txt")).read().strip() for uid in sample_ids}

    print("Running Setup 1 (predicted durations)...")
    setup1 = run_pipeline(args.checkpoint, base_config, use_aligner=False, n=args.n)
    print("Running Setup 2 (aligner durations)...")
    setup2 = run_pipeline(args.checkpoint, base_config, use_aligner=True, n=args.n)

    for uid in sample_ids:
        if uid not in setup1 or uid not in setup2:
            print(f"  {uid}: missing from predictions, skipping")
            continue
        out_path = os.path.join(args.out_dir, f"trace_{uid}.png")
        plot_traces(uid, texts[uid], setup1[uid], setup2[uid], ground_truth[uid], out_path)

    if args.results_json and os.path.exists(args.results_json):
        plot_summary(args.results_json, os.path.join(args.out_dir, "eval_summary.png"))
    else:
        print(f"No results_json found at {args.results_json!r} -- skipping summary bar chart")


if __name__ == "__main__":
    main()
