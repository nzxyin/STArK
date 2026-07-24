import argparse
import os
import io

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from deprecated.train_util import eval_log, get_model, to_device, plot_alignment_to_numpy
from tts.loss import TTSLoss
from tts.dataset import SingleAccentDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, logger=None):
    preprocess_config, _, train_config = configs

    # Get dataset
    dataset = MultiAccentDataset(
        "val.json", preprocess_config, train_config, sort=True, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = TTSLoss().to(device)

    visulation_sample = None

    # Evaluation
    loss_sum = 0
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(*(batch[1:]))

                if visulation_sample is None:
                    visulation_sample = output[-1][0][0]

                # Cal Loss
                loss, _ = Loss(output, batch[5])
                loss_sum += loss.item() * len(batch[0])

    loss_mean = loss_sum / len(dataset)

    message = f"Validation Step {step}, Total Loss: {loss_mean:.4f}"

    if logger is not None:
        alignment_image = plot_alignment_to_numpy(visulation_sample)
        logger.add_image("Alignment/Example", alignment_image, step, dataformats='HWC')
        eval_log(logger, step, loss_mean)

    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs)
    print(message)