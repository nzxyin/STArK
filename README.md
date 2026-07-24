# Articulatory TTS: Non-Autoregressive Articulation-based Text-to-Speech

## Preprocess structure
Save LibriTTS_R
- train-clean-100
- train-clean-360
- dev-clean
- test-clean

After preprocessing, you should have a directory that has the following structure
```
LibriTTS_R_preprocessed
├── dev-clean-preprocessed
├── test-clean-preprocessed
│   ├── dur
│   ├── normalized_txt
│   ├── original_txt
│   ├── phn
│   ├── phn_ids
│   ├── sparc
│   |   ├── emasrc
│   |   |   └── 1089_134686_000001_000001.npy
│   |   └── spk_emb
│   └── wav
├── train-clean-100-preprocessed
└── train-clean-360-preprocessed
```

## Setup
Note that this repo has been tested on Python >3.11.
```
conda create -n <env_name> python=3.13
pip install -r requirements.txt
```
## Training
See SPARC paper for details on how to generate SPARC features.
See UniG2P repo for generating Unilex phonemes.

Once both of these are created, run training with the appropriate model, preprocess, and training parameters.

Example command:
```
python train.py -p config/single_accent_config/preprocess_config.yaml -m config/single_accent_config/model_config.yaml -t config/single_accent_config/train_config.yaml --save_name american_english --restore_step 150000
```
## Inference
See `testing.ipynb`.

Audio samples can be found in the `audio_samples` directory.


python train.py -p config/2_accent_config_arpa/preprocess_config.yaml -m config/2_accent_config_arpa/model_config.yaml -t config/2_accent_config_arpa/train_config.yaml --save_name 2_accent_arpa_globe

python train.py -p config/2_accent_config_unilex/preprocess_config.yaml -m config/2_accent_config_unilex/model_config.yaml -t config/2_accent_config_unilex/train_config.yaml --save_name 2_accent_unilex_globe

python train.py -p config/10_accent_config/preprocess_config.yaml -m config/10_accent_config/model_config.yaml -t config/10_accent_config/train_config.yaml --save_name 10_accent_globe