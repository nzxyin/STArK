"""Push a trained checkpoint to the Hugging Face Hub, then load it back *from the Hub*
(round-trip, not just reading the local file) and evaluate it on the LibriTTS-R test-clean
split, reporting the same PCC/DTW metrics as `testing.ipynb`.

Usage:
    python scripts/eval_and_push.py \
        --ckpt_path /data/user_data/xoy/articulatory-tts/ddp_slurm_large_model/ckpt/last.ckpt \
        --repo_id nzxyin/stark-large \
        --overrides model=large_model train=train_large \
                    preprocess.dataset.dataset_root=/data/user_data/xoy/LibriTTS_R/
"""
import argparse
import json
import os
import sys
import tempfile

# tts/model.py does `from utils import ...`, a bare import of the repo-root utils.py — that
# only resolves if the repo root is on sys.path. Running scripts directly (`python train.py`)
# gets this for free; running a script from scripts/ (`python scripts/eval_and_push.py`) does
# not, since Python puts the *script's* directory on sys.path[0], not the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import hydra
from hydra.core.global_hydra import GlobalHydra
import lightning as L
from scipy.stats import pearsonr
from tslearn.metrics import dtw
from tqdm import tqdm

import tts

NAMES = ["ULX", "ULY", "LLX", "LLY", "LIX", "LIY", "TTX", "TTY", "TBX", "TBY", "TDX", "TDY", "Pitch", "Loudness"]


def compute_pearson(prediction, ground_truth):
    corr = {n: float(v) for n, v in zip(NAMES, pearsonr(prediction, ground_truth, axis=0).statistic)}
    return sum(v for k, v in corr.items() if k not in ("Pitch", "Loudness")) / (len(corr) - 2), corr["Pitch"], corr["Loudness"]


def compute_dtw(prediction, ground_truth):
    dists = {n: float(dtw(prediction[:, i], ground_truth[:, i])) for i, n in enumerate(NAMES)}
    return sum(v for k, v in dists.items() if k not in ("Pitch", "Loudness")) / (len(dists) - 2), dists["Pitch"], dists["Loudness"]


def build_config(overrides):
    GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path="../conf"):
        config = hydra.compose(config_name="config", overrides=overrides)
    config["train"]["logger"]["save_dir"] = config["train"]["logger"]["save_dir"].format(experiment_name=config["train"]["experiment_name"])
    config["train"]["checkpoint"]["dirpath"] = config["train"]["checkpoint"]["dirpath"].format(experiment_name=config["train"]["experiment_name"])
    return config


def push_checkpoint(ckpt_path, repo_id):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=ckpt_path,
        path_in_repo="checkpoint.ckpt",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Pushed {ckpt_path} to https://huggingface.co/{repo_id}")


def download_checkpoint(repo_id):
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(repo_id=repo_id, filename="checkpoint.ckpt", repo_type="model")
    print(f"Downloaded checkpoint from {repo_id} to {local_path}")
    return local_path


@torch.no_grad()
def evaluate(config, downloaded_ckpt_path, use_aligner_durations, dataset_root, batch_size=16, num_workers=8):
    config = dict(config)
    config["model"] = dict(config["model"])
    config["model"]["tts"] = dict(config["model"]["tts"])
    config["model"]["tts"]["use_aligner_durations_if_possible"] = use_aligner_durations
    config["train"] = dict(config["train"])
    config["train"]["datamodule"] = dict(config["train"]["datamodule"])
    config["train"]["datamodule"]["batch_size"] = batch_size
    config["train"]["datamodule"]["num_workers"] = num_workers
    config["train"]["trainer"] = dict(config["train"]["trainer"])
    config["train"]["trainer"]["devices"] = 1
    config["train"]["trainer"]["strategy"] = "auto"

    # weights_only=False: the checkpoint's saved hyperparameters include a Hydra/OmegaConf
    # DictConfig, which PyTorch >=2.6's weights_only=True default refuses to unpickle.
    model = tts.LitTTS.load_from_checkpoint(downloaded_ckpt_path, config=config, map_location="cpu", weights_only=False)
    trainer = L.Trainer(**config["train"]["trainer"])
    datamodule = tts.LibriTTSDataModule(config)
    outputs = trainer.predict(model, datamodule)

    # PCC (pearsonr) requires equal-length arrays. Only aligner ground-truth durations
    # (use_aligner_durations_if_possible=True) guarantee the predicted frame count matches
    # ground truth's -- predicted durations (the "stark" tag, False) generally do not, so PCC
    # is skipped there (DTW, which tolerates length mismatch, is still computed for both).
    # Confirmed via a real run crashing here: ValueError, shape (147,14) vs (213,14).
    compute_pcc = use_aligner_durations

    pearson_ema, dtw_ema = [], []
    pearson_pitch, dtw_pitch = [], []
    pearson_loudness, dtw_loudness = [], []
    for batch in tqdm(outputs, desc="scoring"):
        ids = batch[0]
        sparc, mask = batch[1][0], batch[1][1]  # (pred_sparc, sparc_mask, durations, attn_soft, attn_hard, attn_logprob)
        for i in range(len(ids)):
            pred = sparc[i][: (~mask[i]).sum(), :].cpu().numpy()
            gt = np.load(os.path.join(dataset_root, "test-clean-preprocessed", "emasrc", f"{ids[i]}.ema.npy"))
            d_ema, d_pitch, d_loud = compute_dtw(pred, gt)
            dtw_ema.append(d_ema); dtw_pitch.append(d_pitch); dtw_loudness.append(d_loud)
            if compute_pcc:
                p_ema, p_pitch, p_loud = compute_pearson(pred, gt)
                pearson_ema.append(p_ema); pearson_pitch.append(p_pitch); pearson_loudness.append(p_loud)

    def summarize(values):
        values = np.array(values)
        return {"mean": float(values.mean()), "ci95": float(1.96 * values.std() / np.sqrt(len(values)))}

    result = {
        "dtw_ema": summarize(dtw_ema),
        "dtw_pitch": summarize(dtw_pitch),
        "dtw_loudness": summarize(dtw_loudness),
    }
    if compute_pcc:
        result.update({
            "pearson_ema": summarize(pearson_ema),
            "pearson_pitch": summarize(pearson_pitch),
            "pearson_loudness": summarize(pearson_loudness),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", required=True, help="Local path to the trained .ckpt to push")
    parser.add_argument("--repo_id", required=True, help="HF model repo, e.g. nzxyin/stark-large")
    parser.add_argument("--dataset_root", required=True, help="Path to preprocessed LibriTTS_R (for ground-truth emasrc)")
    parser.add_argument("--overrides", nargs="*", default=["model=large_model", "train=train_large"],
                         help="Additional Hydra overrides (model/train config selection)")
    parser.add_argument("--results_path", default=None, help="Where to write the metrics JSON (default: alongside ckpt_path)")
    args = parser.parse_args()

    overrides = list(args.overrides) + [f"preprocess.dataset.dataset_root={args.dataset_root}"]
    config = build_config(overrides)

    push_checkpoint(args.ckpt_path, args.repo_id)
    downloaded_path = download_checkpoint(args.repo_id)

    results = {}
    for tag, use_aligner in [("stark", False), ("stark_align", True)]:
        print(f"=== Evaluating variant: {tag} (use_aligner_durations_if_possible={use_aligner}) ===")
        results[tag] = evaluate(config, downloaded_path, use_aligner, args.dataset_root)
        print(json.dumps(results[tag], indent=2))

    results_path = args.results_path or (os.path.splitext(args.ckpt_path)[0] + ".eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results to {results_path}")


if __name__ == "__main__":
    main()
