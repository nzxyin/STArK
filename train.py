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
        callbacks=[ModelCheckpoint(**config['train']['checkpoint']),],
        logger=TensorBoardLogger(**config['train']['logger']),
    )
    datamodule = tts.LibriTTSDataModule(config)
    datamodule = tts.LJSpeechDataModule(config)
    model = tts.LitTTS(config)
    # trainer.fit(model, datamodule=datamodule, ckpt_path=f'{config["train"]["checkpoint"]["dirpath"]}/last.ckpt', weights_only=False)
    trainer.fit(model, datamodule=datamodule, ckpt_path=None, weights_only=False)

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    train(config)

if __name__ == "__main__":
    main()