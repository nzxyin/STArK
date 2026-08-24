import os
import sys

# tts/model.py, tts/dataset.py etc. do `from utils import ...` / `from ipa import ...`, bare
# imports off the repo root — only resolves if the repo root is on sys.path. Running this
# script directly off the root used to get that for free; now that it lives in scripts/, Python
# puts scripts/ on sys.path[0] instead, so it has to be added back explicitly. Same pattern as
# the rest of scripts/*.py (e.g. eval_and_push.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tts
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary


def train(config: DictConfig):
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')
    config['train']['logger']['save_dir'] = config['train']['logger']['save_dir'].format(experiment_name=config['train']['experiment_name'])
    config['train']['checkpoint']['dirpath'] = config['train']['checkpoint']['dirpath'].format(experiment_name=config['train']['experiment_name'])
    trainer = L.Trainer(
        **config['train']['trainer'],
        # Lightning's default progress bar (RichProgressBar, since `rich` is
        # installed) redraws in place via terminal control codes, which is
        # unreadable in a SLURM .out file — swap it for a callback that prints
        # plain log lines instead, so losses are easy to tail/grep.
        enable_progress_bar=False,
        callbacks=[ModelCheckpoint(**config['train']['checkpoint']), tts.PlainTextProgressCallback()],
        logger=TensorBoardLogger(**config['train']['logger']),
    )
    dataset_name = config['preprocess']['dataset']['dataset_name']
    if dataset_name == 'libritts':
        datamodule = tts.LibriTTSDataModule(config)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    model = tts.LitTTS(config)
    # Auto-resume from the last checkpoint if one exists, so this is safe to
    # requeue on a preemptible partition (a preempted job restarts this
    # script from scratch, but should pick training back up, not discard it).
    last_ckpt_path = f'{config["train"]["checkpoint"]["dirpath"]}/last.ckpt'
    ckpt_path = last_ckpt_path if os.path.exists(last_ckpt_path) else None
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    train(config)

if __name__ == "__main__":
    main()