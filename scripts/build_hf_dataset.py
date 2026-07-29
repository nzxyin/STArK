"""Stage the LibriTTS-R-STArK feature dataset and (optionally) upload it to Hugging Face.

Copies, for each split's canonical utterance ids (from `{split}.json`), everything needed to
train STArK except raw audio: `dur`, `spk_emb`, `phn`, `phn_ids`, `normalized_txt`,
`original_txt`, `emasrc`. The output layout is identical to `{split}-preprocessed/` on disk, so
it can be pointed at directly as `preprocess.dataset.dataset_root` after download — no custom
`datasets` loading script needed.

Usage:
    python scripts/build_hf_dataset.py \
        --source_root /data/user_data/xoy/LibriTTS_R \
        --staging_root /data/user_data/xoy/libritts-r-stark-staging \
        --repo_id nzxyin/libritts-r-stark \
        --upload
"""
import argparse
import json
import os
import shutil

SPLITS = ["train-clean-100", "dev-clean", "test-clean"]

# modality -> filename suffix appended to the utterance id
MODALITY_SUFFIXES = {
    "dur": ".dur.npy",
    "spk_emb": ".npy",
    "phn": ".phones.txt",
    "phn_ids": ".phones.npy",
    "normalized_txt": ".txt",
    "original_txt": ".txt",
    "emasrc": ".ema.npy",
}


def stage_split(source_root, staging_root, split):
    manifest_path = os.path.join(source_root, f"{split}.json")
    with open(manifest_path) as f:
        ids = json.load(f)

    src_preprocessed = os.path.join(source_root, f"{split}-preprocessed")
    dst_preprocessed = os.path.join(staging_root, f"{split}-preprocessed")

    staged_ids = []
    for utt_id in ids:
        # Skip ids missing any of the SPARC-derived files (a small number of
        # utterances fail SPARC extraction/alignment upstream).
        paths = {
            modality: os.path.join(src_preprocessed, modality, f"{utt_id}{suffix}")
            for modality, suffix in MODALITY_SUFFIXES.items()
        }
        if not all(os.path.exists(p) for p in paths.values()):
            continue
        for modality, src_path in paths.items():
            dst_dir = os.path.join(dst_preprocessed, modality)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src_path, os.path.join(dst_dir, os.path.basename(src_path)))
        staged_ids.append(utt_id)

    os.makedirs(staging_root, exist_ok=True)
    with open(os.path.join(staging_root, f"{split}.json"), "w") as f:
        json.dump(staged_ids, f)

    print(f"{split}: staged {len(staged_ids)}/{len(ids)} utterances")
    return len(staged_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True, help="Path to the live LibriTTS_R directory (e.g. /data/user_data/<user>/LibriTTS_R)")
    parser.add_argument("--staging_root", required=True, help="Where to stage the filtered, feature-only dataset before upload")
    parser.add_argument("--repo_id", required=True, help="Target HF dataset repo, e.g. nzxyin/libritts-r-stark")
    parser.add_argument("--upload", action="store_true", help="Upload staging_root to the HF Hub after staging (requires HF_TOKEN)")
    args = parser.parse_args()

    for split in SPLITS:
        stage_split(args.source_root, args.staging_root, split)

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=args.staging_root,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded {args.staging_root} to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
