"""Diagnose whether bad-sounding demo output is a checkpoint/training problem or a bug in the
new inference.py/g2p.py pipeline, by comparing three things on real test-clean utterances (which
have ground truth to compare against, unlike the site's demo sentences, which were made-up text
with nothing to check against):

  (A) The existing, already-tested dataset-driven evaluation path (LibriTTSDataModule ->
      trainer.predict(), same as eval_and_push.py/testing.ipynb) — Setup 1 (predicted durations,
      DTW only) and Setup 2 (aligner ground-truth durations, PCC) exactly reproducing the
      paper's own methodology. If this looks like Table 2, the checkpoint + core model are
      healthy.
  (B) The same model, same Setup 1 (predicted durations), but fed phoneme ids from *my* new
      g2p.py instead of the dataset's precomputed phn_ids. Comparing this DTW against (A)'s
      Setup 1 DTW for the *same utterances* isolates whether my G2P specifically is the problem.
  (C) Audio-domain DNSMOS on: real ground-truth audio, an oracle resynthesis (ground-truth SPARC
      features decoded through the SPARC vocoder with the utterance's own speaker embedding —
      isolates vocoder/decode-call correctness from the TTA model entirely), and my G2P-fed
      synthesis.

Usage:
    python scripts/diagnose_inference.py --checkpoint <path> --dataset_root <path> --n 5
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import lightning as L
from scipy.stats import pearsonr
from tslearn.metrics import dtw

import tts
from tts.inference import STArKInference

NAMES = ["ULX", "ULY", "LLX", "LLY", "LIX", "LIY", "TTX", "TTY", "TBX", "TBY", "TDX", "TDY", "Pitch", "Loudness"]


def compute_pearson(prediction, ground_truth):
    corr = {n: float(v) for n, v in zip(NAMES, pearsonr(prediction, ground_truth, axis=0).statistic)}
    return sum(v for k, v in corr.items() if k not in ("Pitch", "Loudness")) / (len(corr) - 2), corr["Pitch"], corr["Loudness"]


def compute_dtw(prediction, ground_truth):
    dists = {n: float(dtw(prediction[:, i], ground_truth[:, i])) for i, n in enumerate(NAMES)}
    return sum(v for k, v in dists.items() if k not in ("Pitch", "Loudness")) / (len(dists) - 2), dists["Pitch"], dists["Loudness"]


def build_config(overrides):
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path="../conf"):
        config = hydra.compose(config_name="config", overrides=overrides)
    config["train"]["logger"]["save_dir"] = config["train"]["logger"]["save_dir"].format(experiment_name=config["train"]["experiment_name"])
    config["train"]["checkpoint"]["dirpath"] = config["train"]["checkpoint"]["dirpath"].format(experiment_name=config["train"]["experiment_name"])
    return config


@torch.no_grad()
def run_dataset_pipeline(checkpoint_path, dataset_root, use_aligner, n, batch_size=8):
    """(A): same code path as eval_and_push.py, restricted to the first `n` test-clean examples."""
    config = build_config(["model=large_model", "train=train_large",
                            f"preprocess.dataset.dataset_root={dataset_root}"])
    config = dict(config)
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

    results = {}
    count = 0
    for batch in outputs:
        ids = batch[0]
        sparc, mask = batch[1][0], batch[1][1]
        for i in range(len(ids)):
            if count >= n:
                break
            pred = sparc[i][: (~mask[i]).sum(), :].cpu().numpy()
            results[ids[i]] = pred
            count += 1
        if count >= n:
            break
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--split", default="test-clean")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    preprocessed = os.path.join(args.dataset_root, f"{args.split}-preprocessed")
    ids = json.load(open(os.path.join(args.dataset_root, f"{args.split}.json")))
    # LibriTTSDataset/predict_dataloader iterates in this exact (unshuffled) order, so taking
    # the first n here is what limit_predict_batches below will actually cover — a random
    # sample wouldn't necessarily line up with whichever batches get predicted.
    sample_ids = ids[: args.n]
    print(f"Using first {args.n} utterance ids (dataset order): {sample_ids}\n")

    ground_truth = {uid: np.load(os.path.join(preprocessed, "emasrc", f"{uid}.ema.npy")) for uid in sample_ids}

    # (A) Setup 1 (predicted durations, dataset-fed phonemes) — the paper's own DTW methodology
    print("=== (A) Dataset-driven pipeline, Setup 1 (predicted durations) ===")
    setup1_pred = run_dataset_pipeline(args.checkpoint, args.dataset_root, use_aligner=False, n=args.n)
    dtw_ema_list, dtw_pitch_list, dtw_loud_list = [], [], []
    for uid in sample_ids:
        if uid not in setup1_pred:
            print(f"  {uid}: not in the first batch(es), skipping Setup 1 DTW for this id")
            continue
        d_ema, d_pitch, d_loud = compute_dtw(setup1_pred[uid], ground_truth[uid])
        dtw_ema_list.append(d_ema); dtw_pitch_list.append(d_pitch); dtw_loud_list.append(d_loud)
        print(f"  {uid}: DTW ema={d_ema:.3f} pitch={d_pitch:.3f} loudness={d_loud:.3f}")
    if dtw_ema_list:
        print(f"  mean: DTW ema={np.mean(dtw_ema_list):.3f} pitch={np.mean(dtw_pitch_list):.3f} loudness={np.mean(dtw_loud_list):.3f}")
        print(f"  (paper Table 1 doesn't report Setup 1 DTW directly; Table 2's Setup-1-vs-2 DTW deltas: EMA -0.81, Pitch -0.04, Loudness -0.20)")

    # (A) Setup 2 (aligner ground-truth durations) — reproduces paper Table 2's PCC exactly
    print("\n=== (A) Dataset-driven pipeline, Setup 2 (aligner durations) — paper Table 2 anchor ===")
    setup2_pred = run_dataset_pipeline(args.checkpoint, args.dataset_root, use_aligner=True, n=args.n)
    pcc_ema_list, pcc_pitch_list, pcc_loud_list = [], [], []
    for uid in sample_ids:
        if uid not in setup2_pred:
            continue
        p_ema, p_pitch, p_loud = compute_pearson(setup2_pred[uid], ground_truth[uid])
        pcc_ema_list.append(p_ema); pcc_pitch_list.append(p_pitch); pcc_loud_list.append(p_loud)
        print(f"  {uid}: PCC ema={p_ema:.3f} pitch={p_pitch:.3f} loudness={p_loud:.3f}")
    if pcc_ema_list:
        print(f"  mean: PCC ema={np.mean(pcc_ema_list):.3f} pitch={np.mean(pcc_pitch_list):.3f} loudness={np.mean(pcc_loud_list):.3f}")
        print(f"  paper Table 2 (full test-clean, N=302x8): PCC ema=0.905 pitch=0.533 loudness=0.796")

    # (B) My g2p.py-fed pipeline, Setup 1 — isolates whether *my* G2P is the problem
    print("\n=== (B) My g2p.py-fed pipeline (STArKInference), Setup 1 ===")
    engine = STArKInference(checkpoint_path=args.checkpoint)
    dtw_ema_g2p, dtw_pitch_g2p, dtw_loud_g2p = [], [], []
    my_articulations = {}
    for uid in sample_ids:
        norm_text = open(os.path.join(preprocessed, "normalized_txt", f"{uid}.txt")).read().strip()
        art = engine.text_to_articulation(norm_text)
        pred = np.concatenate([art["ema"], art["pitch"][:, None], art["loudness"][:, None]], axis=1)
        my_articulations[uid] = art
        d_ema, d_pitch, d_loud = compute_dtw(pred, ground_truth[uid])
        dtw_ema_g2p.append(d_ema); dtw_pitch_g2p.append(d_pitch); dtw_loud_g2p.append(d_loud)
        print(f"  {uid}: text={norm_text!r}")
        print(f"    phonemes: {' '.join(art['phonemes'])}")
        print(f"    DTW ema={d_ema:.3f} pitch={d_pitch:.3f} loudness={d_loud:.3f}  (pred_len={pred.shape[0]}, gt_len={ground_truth[uid].shape[0]})")
    print(f"  mean: DTW ema={np.mean(dtw_ema_g2p):.3f} pitch={np.mean(dtw_pitch_g2p):.3f} loudness={np.mean(dtw_loud_g2p):.3f}")

    # (C) Audio-domain DNSMOS: ground truth vs oracle resynthesis vs my synthesis
    print("\n=== (C) Audio quality (DNSMOS) ===")
    try:
        from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score as dnsmos
        import soundfile as sf
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def score(wav, sr):
            if sr != 16000:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            p808, sig, bak, ovr = dnsmos(torch.from_numpy(wav).float().to(device), 16000, personalized=False, num_threads=4)
            return {"p808": float(p808), "sig": float(sig), "bak": float(bak), "ovr": float(ovr)}

        pitch_stats = json.load(open(os.path.join(preprocessed, "pitch_stats.json")))
        for uid in sample_ids:
            spk_emb = np.load(os.path.join(preprocessed, "spk_emb", f"{uid}.npy"))
            median_pitch = float(pitch_stats[uid])
            gt = ground_truth[uid]

            gt_wav, gt_sr = sf.read(os.path.join(preprocessed, "wav", f"{uid}.wav"))
            oracle_wav = engine.sparc_model.decode(gt[:, :12], np.exp(gt[:, 12]) * median_pitch, gt[:, 13], spk_emb)
            art = my_articulations[uid]
            my_wav = engine.sparc_model.decode(art["ema"], np.exp(art["pitch"]) * median_pitch, art["loudness"], spk_emb)

            print(f"  {uid}:")
            print(f"    ground truth audio:    {score(gt_wav.astype(np.float32), gt_sr)}")
            print(f"    oracle resynthesis:    {score(oracle_wav, engine.sparc_model.sr)}  (gt SPARC features -> vocoder, isolates vocoder/decode correctness)")
            print(f"    my synthesis (G2P):    {score(my_wav, engine.sparc_model.sr)}")
    except Exception as e:
        print(f"  DNSMOS scoring failed: {type(e).__name__}: {e}")

    print("\n=== paper Table 1 reference (full test-clean, N=302x8) ===")
    print("  GT:       DNSMOS P.808=3.71 SIG=3.465 BAK=3.925 OVR=3.135")
    print("  SPARC:    DNSMOS P.808=3.77 SIG=3.543 BAK=4.024 OVR=3.240")
    print("  STArK:    DNSMOS P.808=3.71 SIG=3.504 BAK=3.902 OVR=3.142  (Setup 1, predicted durations — matches (B)/(C) above)")


if __name__ == "__main__":
    main()
