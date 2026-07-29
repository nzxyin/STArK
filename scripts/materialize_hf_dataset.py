"""Turn a downloaded (sharded) nzxyin/libritts-r-stark snapshot into the flat
`{split}-preprocessed/{modality}/{id}.ext` layout `LibriTTSDataset` expects, via symlinks
(no data is duplicated).

Usage:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="nzxyin/libritts-r-stark", repo_type="dataset",
                       local_dir="/path/to/downloaded")

    python scripts/materialize_hf_dataset.py \
        --downloaded_root /path/to/downloaded \
        --dataset_root /path/to/LibriTTS_R
"""
import argparse
import json
import os

SPLITS = ["train-clean-100", "dev-clean", "test-clean"]
MODALITIES = ["dur", "spk_emb", "phn", "phn_ids", "normalized_txt", "original_txt", "emasrc"]


def materialize_split(downloaded_root, dataset_root, split):
    src_preprocessed = os.path.join(downloaded_root, f"{split}-preprocessed")
    dst_preprocessed = os.path.join(dataset_root, f"{split}-preprocessed")

    for modality in MODALITIES:
        src_modality_dir = os.path.join(src_preprocessed, modality)
        dst_modality_dir = os.path.join(dst_preprocessed, modality)
        os.makedirs(dst_modality_dir, exist_ok=True)
        for bucket in os.listdir(src_modality_dir):
            bucket_dir = os.path.join(src_modality_dir, bucket)
            if not os.path.isdir(bucket_dir):
                continue
            for filename in os.listdir(bucket_dir):
                link_path = os.path.join(dst_modality_dir, filename)
                if not os.path.exists(link_path):
                    os.symlink(os.path.join(bucket_dir, filename), link_path)

    manifest_src = os.path.join(downloaded_root, f"{split}.json")
    manifest_dst = os.path.join(dataset_root, f"{split}.json")
    if not os.path.exists(manifest_dst):
        with open(manifest_src) as f:
            ids = json.load(f)
        with open(manifest_dst, "w") as f:
            json.dump(ids, f)

    print(f"{split}: materialized {len(os.listdir(os.path.join(dst_preprocessed, MODALITIES[0])))} symlinks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloaded_root", required=True, help="Local dir passed to snapshot_download")
    parser.add_argument("--dataset_root", required=True, help="Flat dataset_root to create (same value as preprocess.dataset.dataset_root)")
    args = parser.parse_args()

    for split in SPLITS:
        materialize_split(args.downloaded_root, args.dataset_root, split)


if __name__ == "__main__":
    main()
