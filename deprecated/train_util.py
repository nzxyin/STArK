from fastaccent.model import FastAccentTTS
from deprecated.optimizer import ScheduledOptim

import os
import json
import io
import matplotlib.pyplot as plt
from PIL import Image

import torch
import numpy as np

def to_device(data, device):
    assert len(data) == 9, "incorrect data format"
    (
        ids,
        texts,
        text_lens,
        max_text_len,
        durs_padded,
        sparcs,
        sparc_lens,
        max_sparc_len,
        accents,
    ) = data
    texts = torch.from_numpy(texts).long().to(device)
    text_lens = torch.from_numpy(text_lens).to(device)
    sparcs = torch.from_numpy(sparcs).float().to(device)
    sparc_lens = torch.from_numpy(sparc_lens).to(device)
    durs_padded = torch.from_numpy(durs_padded).long().to(device)
    accents = torch.from_numpy(accents).long().to(device)
    return (
        ids,
        texts,
        text_lens,
        max_text_len,
        durs_padded,
        sparcs,
        sparc_lens,
        max_sparc_len,
        accents,
    )

def get_model(args, configs, device, train=False):
    (_, model_config, train_config) = configs

    model = FastAccentTTS(**model_config).to(device)
    if args.load_path:
        ckpt_path = args.load_path
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        msg = model.load_state_dict(ckpt["model"])
        print(msg)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, weights_only=False)
        print(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def log(logger, step, meta):
    logger.add_scalar("loss/total_loss", meta['loss'], step)
    logger.add_scalar("loss/sparc_loss", meta['sparc_loss'], step)
    logger.add_scalar("loss/duration_loss", meta['duration_loss'], step)
    logger.add_scalar("loss/attn_loss", meta['attn_loss'], step)
    logger.add_scalar("loss/bin_loss", meta['bin_loss'], step)

def eval_log(logger, step, loss):
    logger.add_scalar("loss/eval_loss", loss, step)

def plot_alignment_to_numpy(alignment, title="Alignment", xlabel="Text", ylabel="SPARC Frames"):
    """
    Converts an alignment matrix into a numpy image for TensorBoard.
    
    alignment: 2D numpy array or torch.Tensor (mel_len, text_len)
    """
    if isinstance(alignment, torch.Tensor):
        alignment = alignment.detach().cpu().numpy()
        
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Save plot to a numpy array (for TensorBoard)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = np.array(image)
    plt.close(fig)
    
    return image