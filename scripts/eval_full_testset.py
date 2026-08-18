"""Evaluate a checkpoint (any checkpoint — including a mid-training one, not just a final push
candidate) on the *full* LibriTTS-R test-clean split:

  1. Predicted EMA/pitch/loudness vs ground truth, using the same dataset-driven pipeline and
     both setups as `eval_and_push.py`/the paper's Table 2: Setup 1 (predicted durations, the
     true end-to-end scenario) scored via DTW only, and Setup 2 (aligner ground-truth durations,
     the paper's reported anchor) scored via both DTW and Pearson correlation (PCC). PCC is
     skipped for Setup 1 because pearsonr requires equal-length arrays, and only aligner
     ground-truth durations guarantee the predicted frame count matches ground truth's.
  2. Audio-domain quality of the Setup 1 (predicted-duration) resynthesis: DNSMOS and UTMOSv2,
     alongside ground-truth audio and an oracle resynthesis (ground-truth SPARC features decoded
     through the same vocoder) as reference points, matching paper Table 1's methodology.

Deliberately reuses the dataset's precomputed `phn_ids` (not `g2p.py`) for the text input, same
as `eval_and_push.py` and the paper — this isolates checkpoint/model quality from the G2P-pipeline
quality issue that was already diagnosed separately in `diagnose_inference.py`.

This does NOT push anything to the Hub — it's a read-only quality check on a local checkpoint.

Usage:
    uv run --extra eval scripts/eval_full_testset.py \
        --checkpoint /data/user_data/xoy/articulatory-tts/stark_large_100k/ckpt/last.ckpt \
        --dataset_root /data/user_data/xoy/LibriTTS_R/ \
        --results_path /data/user_data/xoy/articulatory-tts/stark_large_100k/testset_eval.json
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
from scipy.stats import pearsonr
from tslearn.metrics import dtw
from tqdm import tqdm

import tts
from tts import sparc as tts_sparc

NAMES = ["ULX", "ULY", "LLX", "LLY", "LIX", "LIY", "TTX", "TTY", "TBX", "TBY", "TDX", "TDY", "Pitch", "Loudness"]


def compute_pearson(prediction, ground_truth):
    corr = {n: float(v) for n, v in zip(NAMES, pearsonr(prediction, ground_truth, axis=0).statistic)}
    return sum(v for k, v in corr.items() if k not in ("Pitch", "Loudness")) / (len(corr) - 2), corr["Pitch"], corr["Loudness"]


def compute_dtw(prediction, ground_truth):
    dists = {n: float(dtw(prediction[:, i], ground_truth[:, i])) for i, n in enumerate(NAMES)}
    return sum(v for k, v in dists.items() if k not in ("Pitch", "Loudness")) / (len(dists) - 2), dists["Pitch"], dists["Loudness"]


def summarize(values):
    values = np.array(values)
    return {"mean": float(values.mean()), "ci95": float(1.96 * values.std() / np.sqrt(len(values))), "n": len(values)}


def build_config(overrides):
    GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path="../conf"):
        config = hydra.compose(config_name="config", overrides=overrides)
    config["train"]["logger"]["save_dir"] = config["train"]["logger"]["save_dir"].format(experiment_name=config["train"]["experiment_name"])
    config["train"]["checkpoint"]["dirpath"] = config["train"]["checkpoint"]["dirpath"].format(experiment_name=config["train"]["experiment_name"])
    return config


@torch.no_grad()
def run_dataset_pipeline(checkpoint_path, base_config, use_aligner, batch_size, num_workers):
    """Full test-clean pass -> {utt_id: predicted_sparc (T,14) ndarray}. Same code path as
    eval_and_push.py, just without limit_predict_batches (the whole split, not a sample)."""
    config = dict(base_config)
    config["model"] = dict(config["model"]); config["model"]["tts"] = dict(config["model"]["tts"])
    config["model"]["tts"]["use_aligner_durations_if_possible"] = use_aligner
    config["train"] = dict(config["train"]); config["train"]["datamodule"] = dict(config["train"]["datamodule"])
    config["train"]["datamodule"]["batch_size"] = batch_size
    config["train"]["datamodule"]["num_workers"] = num_workers
    config["train"]["trainer"] = dict(config["train"]["trainer"])
    config["train"]["trainer"]["devices"] = 1
    config["train"]["trainer"]["strategy"] = "auto"

    model = tts.LitTTS.load_from_checkpoint(checkpoint_path, config=config, map_location="cpu",
                                             weights_only=False, strict=False)
    trainer = L.Trainer(**config["train"]["trainer"])
    datamodule = tts.LibriTTSDataModule(config)
    outputs = trainer.predict(model, datamodule)

    predictions = {}
    for batch in outputs:
        ids = batch[0]
        sparc, mask = batch[1][0], batch[1][1]
        for i in range(len(ids)):
            predictions[ids[i]] = sparc[i][: (~mask[i]).sum(), :].cpu().numpy()
    return predictions


def score_ema(predictions, ground_truth, compute_pcc):
    """compute_pcc must be False for predicted-duration (Setup 1) output: pearsonr requires
    equal-length arrays, but predicted durations generally do NOT produce the same frame count
    as ground truth (only aligner ground-truth durations, i.e. Setup 2, guarantee that) — this
    is exactly why the paper reports DTW (length-tolerant) for Setup 1 and PCC only for Setup 2,
    a distinction diagnose_inference.py already encodes but this script initially missed,
    crashing on a real length mismatch (173 vs 213 frames) partway through a full-test-set run."""
    dtw_ema, dtw_pitch, dtw_loudness = [], [], []
    pearson_ema, pearson_pitch, pearson_loudness = [], [], []
    for uid, pred in tqdm(predictions.items(), desc="scoring EMA (PCC+DTW)" if compute_pcc else "scoring EMA (DTW)"):
        gt = ground_truth[uid]
        d_ema, d_pitch, d_loud = compute_dtw(pred, gt)
        dtw_ema.append(d_ema); dtw_pitch.append(d_pitch); dtw_loudness.append(d_loud)
        if compute_pcc:
            p_ema, p_pitch, p_loud = compute_pearson(pred, gt)
            pearson_ema.append(p_ema); pearson_pitch.append(p_pitch); pearson_loudness.append(p_loud)
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


@torch.no_grad()
def score_audio(setup1_predictions, ground_truth, preprocessed_dir, device):
    """DNSMOS + UTMOSv2 on ground-truth audio, oracle resynthesis (gt features -> vocoder), and
    the model's own Setup-1 (predicted-duration) resynthesis, matching paper Table 1's rows."""
    import soundfile as sf
    import librosa
    from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score as dnsmos
    import utmosv2

    sparc_config = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conf", "sparc", "model_englishplus_2M.yaml")
    sparc_model = tts_sparc.load_model(model_name="en+", config=sparc_config)
    sparc_model.to(device)

    utmos_model = utmosv2.create_model(pretrained=True)

    pitch_stats = json.load(open(os.path.join(preprocessed_dir, "pitch_stats.json")))

    def dnsmos_score(wav, sr):
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        p808, sig, bak, ovr = dnsmos(torch.from_numpy(wav).float().to(device), 16000, personalized=False, num_threads=4)
        return {"p808": float(p808), "sig": float(sig), "bak": float(bak), "ovr": float(ovr)}

    def utmos_score(wav, sr):
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        # predict() returns shape (batch_size,) or (1,), not a bare scalar (per its own docs),
        # so a plain float(...) crashes with "only 0-dimensional arrays can be converted to
        # Python scalars" -- confirmed on a real run that got this far. Squeeze to the one value.
        result = utmos_model.predict(data=wav.astype(np.float32), sr=16000)
        return float(np.asarray(result).reshape(-1)[0])

    results = {"gt": {"dnsmos": [], "utmosv2": []}, "oracle": {"dnsmos": [], "utmosv2": []}, "stark": {"dnsmos": [], "utmosv2": []}}
    for uid in tqdm(setup1_predictions, desc="scoring audio (DNSMOS + UTMOSv2)"):
        spk_emb = np.load(os.path.join(preprocessed_dir, "spk_emb", f"{uid}.npy"))
        median_pitch = float(pitch_stats[uid])
        gt = ground_truth[uid]
        pred = setup1_predictions[uid]

        gt_wav, gt_sr = sf.read(os.path.join(preprocessed_dir, "wav", f"{uid}.wav"))
        gt_wav = gt_wav.astype(np.float32)
        oracle_wav = sparc_model.decode(gt[:, :12], np.exp(gt[:, 12]) * median_pitch, gt[:, 13], spk_emb)
        stark_wav = sparc_model.decode(pred[:, :12], np.exp(pred[:, 12]) * median_pitch, pred[:, 13], spk_emb)

        results["gt"]["dnsmos"].append(dnsmos_score(gt_wav, gt_sr))
        results["gt"]["utmosv2"].append(utmos_score(gt_wav, gt_sr))
        results["oracle"]["dnsmos"].append(dnsmos_score(oracle_wav, sparc_model.sr))
        results["oracle"]["utmosv2"].append(utmos_score(oracle_wav, sparc_model.sr))
        results["stark"]["dnsmos"].append(dnsmos_score(stark_wav, sparc_model.sr))
        results["stark"]["utmosv2"].append(utmos_score(stark_wav, sparc_model.sr))

    summary = {}
    for tag, scores in results.items():
        dnsmos_keys = scores["dnsmos"][0].keys() if scores["dnsmos"] else []
        summary[tag] = {
            **{f"dnsmos_{k}": summarize([s[k] for s in scores["dnsmos"]]) for k in dnsmos_keys},
            "utmosv2": summarize(scores["utmosv2"]),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--split", default="test-clean")
    parser.add_argument("--overrides", nargs="*", default=["model=large_model", "train=train_large"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--skip_audio", action="store_true", help="Skip DNSMOS/UTMOSv2 scoring, EMA PCC/DTW only")
    parser.add_argument("--results_path", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessed = os.path.join(args.dataset_root, f"{args.split}-preprocessed")
    ids = json.load(open(os.path.join(args.dataset_root, f"{args.split}.json")))
    print(f"Evaluating checkpoint {args.checkpoint} on {len(ids)} {args.split} utterances\n")

    overrides = list(args.overrides) + [f"preprocess.dataset.dataset_root={args.dataset_root}"]
    base_config = build_config(overrides)

    ground_truth = {uid: np.load(os.path.join(preprocessed, "emasrc", f"{uid}.ema.npy")) for uid in ids}

    results = {"checkpoint": args.checkpoint, "split": args.split, "n_utterances": len(ids)}

    print("=== Setup 1: predicted durations (true end-to-end scenario) ===")
    setup1_predictions = run_dataset_pipeline(args.checkpoint, base_config, use_aligner=False,
                                               batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"  got predictions for {len(setup1_predictions)}/{len(ids)} utterances")
    results["setup1_predicted_durations"] = score_ema(setup1_predictions, ground_truth, compute_pcc=False)
    print(json.dumps(results["setup1_predicted_durations"], indent=2))

    print("\n=== Setup 2: aligner ground-truth durations (paper Table 2 anchor) ===")
    setup2_predictions = run_dataset_pipeline(args.checkpoint, base_config, use_aligner=True,
                                               batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"  got predictions for {len(setup2_predictions)}/{len(ids)} utterances")
    results["setup2_aligner_durations"] = score_ema(setup2_predictions, ground_truth, compute_pcc=True)
    print(json.dumps(results["setup2_aligner_durations"], indent=2))
    print("  paper Table 2 reference (full test-clean, N=302x8): PCC ema=0.905 pitch=0.533 loudness=0.796")

    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote (partial) results to {args.results_path}")

    if not args.skip_audio:
        print("\n=== Audio quality: DNSMOS + UTMOSv2 (Setup 1 resynthesis vs ground truth vs oracle) ===")
        results["audio_quality"] = score_audio(setup1_predictions, ground_truth, preprocessed, device)
        print(json.dumps(results["audio_quality"], indent=2))
        print("  paper Table 1 DNSMOS reference (full test-clean, N=302x8):")
        print("    GT:    P.808=3.71 SIG=3.465 BAK=3.925 OVR=3.135")
        print("    SPARC: P.808=3.77 SIG=3.543 BAK=4.024 OVR=3.240")
        print("    STArK: P.808=3.71 SIG=3.504 BAK=3.902 OVR=3.142")

        with open(args.results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote final results to {args.results_path}")


if __name__ == "__main__":
    main()
