#!/usr/bin/env python3
"""STArK inference CLI: text-to-speech synthesis and articulatory kinematics visualization.

By default, downloads the trained checkpoint from the Hugging Face Hub (nzxyin/stark-large) —
no local training run needed. Use --checkpoint to point at a local .ckpt instead (e.g. one you
trained yourself).

Examples:
    # Synthesize speech, cloning a voice from a short reference clip
    python scripts/stark_cli.py synthesize "Hello, this is a test." \\
        --reference-audio my_voice.wav --output out.wav

    # Synthesize using a speaker from the packaged LibriTTS-R-STArK eval set instead of your
    # own reference clip (see README for downloading that dataset)
    python scripts/stark_cli.py synthesize "Hello, this is a test." \\
        --reference-utt-id 1580_141083_000057_000000 --dataset-root /path/to/LibriTTS_R \\
        --output out.wav

    # Plot the predicted articulatory trajectories for some text
    python scripts/stark_cli.py visualize "Hello, this is a test." --output trajectories.png
"""
import argparse
import os
import sys

# tts/model.py does `from utils import ...`, a bare import of the repo-root utils.py — only
# resolves if the repo root is on sys.path, which running from scripts/ doesn't get for free.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _build_inference(args):
    from tts.inference import STArKInference

    return STArKInference(
        checkpoint_path=args.checkpoint,
        checkpoint_repo=args.hf_repo,
        checkpoint_filename=args.hf_filename,
        device=args.device,
    )


def _load_reference(engine, args):
    if args.reference_audio:
        return engine.load_reference_speaker(args.reference_audio)
    if args.reference_utt_id:
        if not args.dataset_root:
            sys.exit("--reference-utt-id requires --dataset-root (see README for the "
                      "LibriTTS-R-STArK dataset)")
        return engine.load_reference_speaker_from_dataset(args.reference_utt_id, args.dataset_root)
    sys.exit("Provide either --reference-audio <wav> or --reference-utt-id <id> --dataset-root <path>")


def cmd_synthesize(args):
    import soundfile as sf

    engine = _build_inference(args)
    reference = _load_reference(engine, args)
    wav, sr, _articulation = engine.synthesize(args.text, reference)
    sf.write(args.output, wav, sr)
    print(f"Wrote {len(wav) / sr:.2f}s of audio to {args.output}")


def cmd_visualize(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tts.inference import EMA_DIMENSION_NAMES

    engine = _build_inference(args)
    articulation = engine.text_to_articulation(args.text)
    prediction = np.concatenate(
        [articulation["ema"], articulation["pitch"][:, None], articulation["loudness"][:, None]],
        axis=1,
    )

    reference_trajectory = None
    if args.reference_audio:
        encoded = engine.sparc_model.encode([args.reference_audio], split_batch=True, reduce=True)
        reference_trajectory = np.concatenate(
            [encoded["ema"], encoded["pitch"][:, None], encoded["loudness"][:, None]], axis=1,
        )

    fig, axes = plt.subplots(14, 1, figsize=(14, 15), sharex=True)
    for i, name in enumerate(EMA_DIMENSION_NAMES):
        ax = axes[i]
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])
        line, = ax.plot(prediction[:, i], color="green", linewidth=3)
        if i == 0:
            line.set_label("STArK (predicted)")
        if reference_trajectory is not None:
            line2, = ax.plot(reference_trajectory[:, i], color="purple", linestyle=":", linewidth=3)
            if i == 0:
                line2.set_label("Reference audio (SPARC-encoded)")
        ax.set_ylabel(name, fontsize=20, rotation=0, labelpad=0, ha="right", va="center")
    fig.legend(loc="upper right", fontsize=16)
    fig.suptitle(args.text, fontsize=12)
    plt.tight_layout(pad=0.5)
    plt.savefig(args.output, dpi=150)
    print(f"Wrote articulatory trajectory plot to {args.output}")
    print(f"Phonemes used: {' '.join(articulation['phonemes'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", default=None, help="Local .ckpt path (default: download from Hugging Face)")
    common.add_argument("--hf-repo", default="nzxyin/stark-large", help="HF Hub model repo to download from")
    common.add_argument("--hf-filename", default="checkpoint.ckpt", help="Checkpoint filename within the HF repo")
    common.add_argument("--device", default=None, help="torch device (default: cuda if available, else cpu)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_synth = subparsers.add_parser("synthesize", parents=[common], help="Text-to-speech synthesis")
    p_synth.add_argument("text")
    p_synth.add_argument("--reference-audio", default=None, help="Reference wav to clone the voice/pitch range from")
    p_synth.add_argument("--reference-utt-id", default=None, help="Utterance id to pull a precomputed speaker from the LibriTTS-R-STArK dataset instead")
    p_synth.add_argument("--dataset-root", default=None, help="Path to the LibriTTS_R dataset root (required with --reference-utt-id)")
    p_synth.add_argument("--output", default="stark_output.wav", help="Output wav path")
    p_synth.set_defaults(func=cmd_synthesize)

    p_viz = subparsers.add_parser("visualize", parents=[common], help="Plot predicted articulatory trajectories")
    p_viz.add_argument("text")
    p_viz.add_argument("--reference-audio", default=None, help="Optional reference wav to overlay its own SPARC-encoded trajectory for comparison")
    p_viz.add_argument("--output", default="stark_trajectories.png", help="Output image path")
    p_viz.set_defaults(func=cmd_visualize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
