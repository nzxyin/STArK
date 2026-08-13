"""Standalone STArK inference: load a trained checkpoint from the Hugging Face Hub (no local
training run or Hydra config required) and run text -> articulatory kinematics -> speech.

Voice identity is *not* something the STArK model itself conditions on (per the paper, it's
trained without speaker embeddings) — it predicts speaker-independent normalized articulatory
kinematics from text, and voice cloning happens at the SPARC vocoder stage: any reference
speaker's embedding + pitch range (extracted from a short reference clip, or pulled from the
precomputed LibriTTS-R-STArK dataset) controls the resynthesized voice.
"""
import os

import numpy as np
import torch

import tts
from tts import sparc as tts_sparc
from tts.g2p import text_to_phoneme_ids

DEFAULT_CHECKPOINT_REPO = "nzxyin/stark-large"
DEFAULT_CHECKPOINT_FILENAME = "checkpoint.ckpt"
DEFAULT_SPARC_MODEL = "en+"

EMA_DIMENSION_NAMES = ["ULX", "ULY", "LLX", "LLY", "LIX", "LIY",
                        "TTX", "TTY", "TBX", "TBY", "TDX", "TDY", "Pitch", "Loudness"]


def _repo_root():
    # src/tts/inference.py -> repo root is two levels up
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_checkpoint(repo_id=DEFAULT_CHECKPOINT_REPO, filename=DEFAULT_CHECKPOINT_FILENAME):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")


class STArKInference:
    def __init__(self, checkpoint_path=None, checkpoint_repo=DEFAULT_CHECKPOINT_REPO,
                 checkpoint_filename=DEFAULT_CHECKPOINT_FILENAME,
                 sparc_model_name=DEFAULT_SPARC_MODEL, sparc_config=None,
                 device=None):
        """
        checkpoint_path: use a local .ckpt instead of downloading one from the Hub.
        checkpoint_repo/checkpoint_filename: HF Hub model repo to download from otherwise.
        sparc_config: defaults to conf/sparc/model_englishplus_2M.yaml relative to the repo root.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if checkpoint_path is None:
            checkpoint_path = download_checkpoint(checkpoint_repo, checkpoint_filename)
        self.checkpoint_path = checkpoint_path

        # weights_only=False: the checkpoint's saved hyperparameters include a Hydra/OmegaConf
        # DictConfig, which PyTorch >=2.6's weights_only=True default refuses to unpickle. Only
        # do this for checkpoints you trust (it allows arbitrary unpickling).
        # strict=False: rotary_emb.freqs is a deterministic, non-learned buffer (derived from
        # model_dim, not trained) whose presence/absence in the state_dict has varied across
        # rotary-embedding-torch versions — checkpoints saved with an older version are missing
        # or mismatched on this key even though the actual learned weights are unaffected.
        self.model = tts.LitTTS.load_from_checkpoint(
            checkpoint_path, map_location=self.device, weights_only=False, strict=False,
        )
        self.model.eval()
        self.model.to(self.device)
        # Always use predicted (not ground-truth-aligned) durations for real inference — there's
        # no ground truth to align to when synthesizing from raw text.
        self.model.model.use_aligner_durations_if_possible = False

        if sparc_config is None:
            sparc_config = os.path.join(_repo_root(), "conf", "sparc", "model_englishplus_2M.yaml")
        self.sparc_model = tts_sparc.load_model(model_name=sparc_model_name, config=sparc_config)
        self.sparc_model.to(self.device)

    @torch.no_grad()
    def text_to_articulation(self, text, accent_id=0):
        """Text -> predicted normalized articulatory kinematics.

        Returns a dict: 'ema' (T,12) EMA channels, 'pitch' (T,) log-normalized pitch,
        'loudness' (T,), 'phonemes' (the G2P symbol sequence used), 'durations' (per-phoneme
        frame counts).
        """
        symbols, phone_ids = text_to_phoneme_ids(text)
        phones = torch.from_numpy(phone_ids).unsqueeze(0).to(self.device)
        phone_lens = torch.tensor([len(phone_ids)], device=self.device)
        accent_ids = torch.tensor([accent_id], dtype=torch.int, device=self.device)

        pred_sparc, sparc_mask, durations, *_ = self.model.model(
            phones, phone_lens, max_phone_lens=len(phone_ids), accent_ids=accent_ids,
        )
        sparc = pred_sparc[0][: (~sparc_mask[0]).sum(), :].cpu().numpy()
        return {
            "ema": sparc[:, :12],
            "pitch": sparc[:, 12],
            "loudness": sparc[:, 13],
            "phonemes": symbols,
            "durations": durations[0].cpu().numpy(),
        }

    def load_reference_speaker(self, reference_wav_path):
        """Extract a speaker embedding + median pitch from a reference audio clip, for voice
        cloning at the SPARC vocoder stage."""
        encoded = self.sparc_model.encode([reference_wav_path], split_batch=True, reduce=True)
        median_pitch = float(np.median(encoded["pitch"]))
        return {"spk_emb": encoded["spk_emb"], "median_pitch": median_pitch}

    def load_reference_speaker_from_dataset(self, utt_id, dataset_root, split="test-clean"):
        """Pull a precomputed speaker embedding + median pitch from the LibriTTS-R-STArK
        dataset (see README) instead of a fresh reference clip."""
        import json
        preprocessed = os.path.join(dataset_root, f"{split}-preprocessed")
        spk_emb = np.load(os.path.join(preprocessed, "spk_emb", f"{utt_id}.npy"))
        pitch_stats = json.load(open(os.path.join(preprocessed, "pitch_stats.json")))
        return {"spk_emb": spk_emb, "median_pitch": float(pitch_stats[utt_id])}

    def synthesize(self, text, reference_speaker, accent_id=0):
        """Text + a reference speaker (from load_reference_speaker[_from_dataset]) -> waveform.

        Returns (wav: np.ndarray, sample_rate: int, articulation: dict — see
        text_to_articulation).
        """
        articulation = self.text_to_articulation(text, accent_id=accent_id)
        ema, pitch_norm, loudness = articulation["ema"], articulation["pitch"], articulation["loudness"]
        pitch = np.exp(pitch_norm) * reference_speaker["median_pitch"]
        wav = self.sparc_model.decode(ema, pitch, loudness, reference_speaker["spk_emb"])
        return wav, self.sparc_model.sr, articulation
