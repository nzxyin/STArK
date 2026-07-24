import torch
import json
import math
import os
from omegaconf import DictConfig, OmegaConf

import numpy as np
from torch.utils.data import Dataset
import functools
from scipy import ndimage
from scipy.stats import betabinom

from utils import pad_1D, pad_2D

import lightning as L

class BetaBinomialInterpolator:
    """Interpolates alignment prior matrices to save computation.

    Calculating beta-binomial priors is costly. Instead cache popular sizes
    and use img interpolation to get priors faster.
    """
    def __init__(self, round_sparc_len_to=100, round_text_len_to=20):
        self.round_sparc_len_to = round_sparc_len_to
        self.round_text_len_to = round_text_len_to
        self.bank = functools.lru_cache(beta_binomial_prior_distribution)

    def round(self, val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, w, h):
        bw = self.round(w, to=self.round_sparc_len_to)
        bh = self.round(h, to=self.round_text_len_to)
        ret = ndimage.zoom(self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1)
        assert len(ret.shape) == 2 and ret.shape[0] == w and ret.shape[1] == h, ret.shape
        return ret


def beta_binomial_prior_distribution(phoneme_count, sparc_count, scaling=1.0):
    P = phoneme_count
    M = sparc_count
    x = np.arange(0, P)
    sparc_text_probs = []
    for i in range(1, M+1):
        a, b = scaling * i, scaling * (M + 1 - i)
        rv = betabinom(P, a, b)
        sparc_i_prob = rv.pmf(x)
        sparc_text_probs.append(sparc_i_prob)
    return np.array(sparc_text_probs)


def extract_duration(text_len, sparc_len):
    binomial_interpolator = BetaBinomialInterpolator()
    attn_prior = binomial_interpolator(sparc_len, text_len)
    assert sparc_len == attn_prior.shape[0]
    return attn_prior


class MultiAccentDataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.splits_path = preprocess_config["path"]["splits_path"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.name_accent = self.process_meta(filename)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.name_accent)

    def __getitem__(self, idx):
        basename, accent = self.name_accent[idx]
        phone = np.load(os.path.join(self.preprocessed_path, "text_ids", f"{basename}.npy"))
        sparc = np.load(os.path.join(self.preprocessed_path, "sparc", f"{basename}.npy"))
        duration = np.load(os.path.join(self.preprocessed_path, "durations", f"{basename}.npy"))
        assert sparc.shape[1] == 15
        assert duration.shape == (phone.shape[0], sparc.shape[0])

        sample = {
            "id": basename,
            "text": phone,
            "sparc": sparc,
            "duration": duration,
            "accent": accent,
        }
        return sample

    def process_meta(self, filename):
        with open(os.path.join(self.splits_path, filename), "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                name_accent = [(name, 0) for name in data]
            elif isinstance(data, dict):
                name_accent = [(name, accent) for name, accent in data.items()]
            else:
                raise TypeError(f"invalid type for {filename}, {type(data)}")
        return name_accent

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        sparcs = [data[idx]["sparc"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        sparc_lens = np.array([sparc.shape[0] for sparc in sparcs])
        accents = np.array([data[idx]["accent"] for idx in idxs])

        texts = pad_1D(texts)
        sparcs = pad_2D(sparcs)
        
        durs_padded = np.zeros((len(idxs), max(sparc_lens), max(text_lens)))
        for i, dur in enumerate(durations):
            durs_padded[i, :dur.shape[1], :dur.shape[0]] = dur.T

        assert texts.shape[1] == max(text_lens)
        assert sparcs.shape[1] == max(sparc_lens)
        assert (durs_padded.shape[1], durs_padded.shape[2]) == (max(sparc_lens), max(text_lens))

        return (
            ids,
            texts,
            text_lens,
            max(text_lens),
            durs_padded,
            sparcs,
            sparc_lens,
            max(sparc_lens),
            accents,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output

class LibriTTSDataset(Dataset):
    def __init__(
        self, filename: str, config: DictConfig, sort=False, drop_last=False
    ):
        self.dataset_name = config['preprocess']["dataset"]["dataset_name"]
        self.dataset_root = config['preprocess']["dataset"]["dataset_root"]
        self.subset = filename.split(".")[0]
        self.names = self.process_meta(os.path.join(self.dataset_root, filename))
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        basename = self.names[idx]
        phone = np.load(os.path.join(self.dataset_root, self.subset+'-preprocessed', 'phn_ids', f"{basename}.phones.npy"))
        sparc = np.load(os.path.join(self.dataset_root, self.subset+'-preprocessed', 'emasrc', f"{basename}.ema.npy"))
        duration = np.load(os.path.join(self.dataset_root, self.subset+'-preprocessed', 'dur', f"{basename}.dur.npy"))
        assert sparc.shape[1] == 14
        assert duration.shape == (phone.shape[0], sparc.shape[0])

        sample = {
            "id": basename,
            "text": phone,
            "sparc": sparc,
            "duration": duration,
        }
        return sample
    
    def process_meta(self, filename):
        with open(os.path.join(self.dataset_root, filename), "r") as f:
            data = json.load(f)
        return data

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        sparcs = [data[idx]["sparc"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        sparc_lens = np.array([sparc.shape[0] for sparc in sparcs])

        texts = torch.from_numpy(pad_1D(texts))
        sparcs = torch.from_numpy(pad_2D(sparcs))

        durs_padded = torch.zeros(len(idxs), max(sparc_lens), max(text_lens))
        for i, dur in enumerate(durations):
            dur = torch.from_numpy(dur)
            durs_padded[i, :dur.shape[1], :dur.shape[0]] = dur.T

        assert texts.shape[1] == max(text_lens)
        assert sparcs.shape[1] == max(sparc_lens)
        assert (durs_padded.shape[1], durs_padded.shape[2]) == (max(sparc_lens), max(text_lens))

        text_lens = torch.from_numpy(text_lens)
        sparc_lens = torch.from_numpy(sparc_lens)

        return (
            ids,
            texts,
            text_lens,
            max(text_lens),
            durs_padded,
            sparcs,
            sparc_lens,
            max(sparc_lens),
            torch.zeros(len(idxs), dtype=torch.int),
        )
    
    def collate_fn(self, data):
        data_size = len(data)

        idx_arr = np.arange(data_size)

        # tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        # idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        # idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        # if not self.drop_last and len(tail) > 0:
        #     idx_arr += [tail.tolist()]

        output = self.reprocess(data, idx_arr)

        return output

class LibriTTSDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.dataset_class = LibriTTSDataset
        self.config = config
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = self.dataset_class(
                self.config['preprocess']["dataset"]["train_filename"],
                self.config,
            )
        if stage == "fit" or stage == "validate" or stage is None:
            self.dev = self.dataset_class(
                self.config['preprocess']["dataset"]["dev_filename"],
                self.config,
            )
        if stage == "test" or stage == "predict" or stage is None:
            self.test = self.dataset_class(
                self.config['preprocess']["dataset"]["test_filename"],
                self.config,
            )
            self.predict = self.test

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.config["train"]["datamodule"]["batch_size"],
            shuffle=self.config["train"]["datamodule"]["shuffle"],
            pin_memory=self.config["train"]["datamodule"]["pin_memory"],
            drop_last=self.config["train"]["datamodule"]["drop_last"],
            collate_fn=self.train.collate_fn,
            num_workers=self.config["train"]["datamodule"]["num_workers"],
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev,
            batch_size=self.config["train"]["datamodule"]["batch_size"],
            shuffle=False,
            collate_fn=self.dev.collate_fn,
            num_workers=self.config["train"]["datamodule"]["num_workers"],
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.config["train"]["datamodule"]["batch_size"],
            shuffle=False,
            collate_fn=self.test.collate_fn,
            num_workers=self.config["train"]["datamodule"]["num_workers"],
        )

    def predict_dataloader(self):
        return self.test_dataloader()

class LJSpeechDataset(Dataset):
    def __init__(
        self, filename: str, config: DictConfig, sort=False, drop_last=False
    ):
        self.dataset_name = config['preprocess']["dataset"]["dataset_name"]
        self.dataset_root = config['preprocess']["dataset"]["dataset_root"]
        self.subset = filename.split(".")[0]
        self.names = self.process_meta(os.path.join(self.dataset_root, filename))
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        basename = self.names[idx]
        phone = np.load(os.path.join(self.dataset_root, 'phn_ids', f"{basename}.phn.npy"))
        sparc = np.load(os.path.join(self.dataset_root, 'ema_preprocessed', f"{basename}.ema.npy"))
        duration = np.load(os.path.join(self.dataset_root, 'dur', f"{basename}.npy"))
        assert sparc.shape[1] == 14
        assert duration.shape == (phone.shape[0], sparc.shape[0])

        sample = {
            "id": basename,
            "text": phone,
            "sparc": sparc,
            "duration": duration,
        }
        return sample
    
    def process_meta(self, filename):
        with open(os.path.join(self.dataset_root, filename), "r") as f:
            data = json.load(f)
        return data

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        sparcs = [data[idx]["sparc"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        sparc_lens = np.array([sparc.shape[0] for sparc in sparcs])

        texts = torch.from_numpy(pad_1D(texts))
        sparcs = torch.from_numpy(pad_2D(sparcs))

        durs_padded = torch.zeros(len(idxs), max(sparc_lens), max(text_lens))
        for i, dur in enumerate(durations):
            dur = torch.from_numpy(dur)
            durs_padded[i, :dur.shape[1], :dur.shape[0]] = dur.T

        assert texts.shape[1] == max(text_lens)
        assert sparcs.shape[1] == max(sparc_lens)
        assert (durs_padded.shape[1], durs_padded.shape[2]) == (max(sparc_lens), max(text_lens))

        text_lens = torch.from_numpy(text_lens)
        sparc_lens = torch.from_numpy(sparc_lens)

        return (
            ids,
            texts,
            text_lens,
            max(text_lens),
            durs_padded,
            sparcs,
            sparc_lens,
            max(sparc_lens),
            torch.zeros(len(idxs), dtype=torch.int),
        )
    
    def collate_fn(self, data):
        data_size = len(data)

        idx_arr = np.arange(data_size)

        # tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        # idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        # idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        # if not self.drop_last and len(tail) > 0:
        #     idx_arr += [tail.tolist()]

        output = self.reprocess(data, idx_arr)

        return output

class LJSpeechDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.dataset_class = LJSpeechDataset
        self.config = config
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = self.dataset_class(
                self.config['preprocess']["dataset"]["train_filename"],
                self.config,
            )
        if stage == "fit" or stage == "validate" or stage is None:
            self.dev = self.dataset_class(
                self.config['preprocess']["dataset"]["dev_filename"],
                self.config,
            )
        if stage == "test" or stage == "predict" or stage is None:
            self.test = self.dataset_class(
                self.config['preprocess']["dataset"]["test_filename"],
                self.config,
            )
            self.predict = self.test

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.config["train"]["datamodule"]["batch_size"],
            shuffle=self.config["train"]["datamodule"]["shuffle"],
            pin_memory=self.config["train"]["datamodule"]["pin_memory"],
            drop_last=self.config["train"]["datamodule"]["drop_last"],
            collate_fn=self.train.collate_fn,
            num_workers=self.config["train"]["datamodule"]["num_workers"],
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev,
            batch_size=self.config["train"]["datamodule"]["batch_size"],
            shuffle=False,
            collate_fn=self.dev.collate_fn,
            num_workers=self.config["train"]["datamodule"]["num_workers"],
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.config["train"]["datamodule"]["batch_size"],
            shuffle=False,
            collate_fn=self.test.collate_fn,
            num_workers=self.config["train"]["datamodule"]["num_workers"],
        )

    def predict_dataloader(self):
        return self.test_dataloader()