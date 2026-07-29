# Python script to preprocess SPARC data

# given SPARC data directory, preprocessed directory, preprocess config
# read each SPARC file, normalize pitch, loudness, remove periodicity, and save to preprocessed directory with per-speaker median values for pitch and loudness
import os
import numpy as np
import argparse
import json

EPS = 1e-8

def process_sparc_data(sparc_dir, preprocessed_dir, ema_output_dirname="emasrc"):
    ema_dir = os.path.join(sparc_dir, "emasrc")
    spk_emb_dir = os.path.join(sparc_dir, "spk_emb")
    pitch_stats_output = {}
    for file in os.listdir(ema_dir):
        if file.endswith('.npy'):
            sparc_path = os.path.join(ema_dir, file)
            sparc_data = np.load(sparc_path)

            assert sparc_data.shape[1] == 15, "SPARC data must have 15 dimensions"
            pitch = sparc_data[:, 12]
            # loudness = sparc_data[:, 13]
            
            median_pitch = np.median(pitch)  # Consider only periodic frames for median calculation
            pitch_stats_output[file.replace('.npy', '')] = float(median_pitch)

            normalized_pitch = np.log(pitch / median_pitch + EPS)

            # weighted_mean_loudness = (loudness * periodicity).sum() / periodicity.sum()
            # normalized_loudness = np.log(loudness / weighted_mean_loudness + EPS)

            processed_sparc = np.copy(sparc_data[:,:14]) # exclude periodicity
            processed_sparc[:, 12] = normalized_pitch
            # processed_sparc[:, 13] = normalized_loudness

            assert processed_sparc.shape[1] == 14, "Processed SPARC data must have 14 dimensions"

            save_path = os.path.join(preprocessed_dir, ema_output_dirname, file.replace('.npy', '.ema.npy'))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, processed_sparc)

            print(f"Processed and saved: {save_path}")
    # Save pitch stats
    stats_save_path = os.path.join(preprocessed_dir, "pitch_stats.json")
    os.makedirs(os.path.dirname(stats_save_path), exist_ok=True)
    with open(stats_save_path, 'w') as f:
        json.dump(pitch_stats_output, f, indent=4)
    print(f"Saved pitch stats to: {stats_save_path}")
    for file in os.listdir(spk_emb_dir):
        if file.endswith('.npy'):
            spk_emb_path = os.path.join(spk_emb_dir, file)
            spk_emb_data = np.load(spk_emb_path)

            save_path = os.path.join(preprocessed_dir, "spk_emb", file)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, spk_emb_data)

            print(f"Copied speaker embedding: {save_path}")
            

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Process SPARC data")
    arg_parser.add_argument("--sparc_dir", type=str, required=True, help="Path to the SPARC data directory")
    arg_parser.add_argument("--preprocessed_dir", type=str, required=True, help="Path to the preprocessed data directory")
    arg_parser.add_argument("--ema_output_dirname", type=str, default="emasrc",
                             help="Subdirectory name (under preprocessed_dir) to write normalized EMA features to. "
                                  "LibriTTSDataset expects 'emasrc' (the default); LJSpeechDataset expects 'ema_preprocessed'.")
    args = arg_parser.parse_args()
    process_sparc_data(args.sparc_dir, args.preprocessed_dir, args.ema_output_dirname)