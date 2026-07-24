from .model import TTS
from .loss import TTSLoss
import math
import torch
import torch.nn as nn
import lightning as L

class LitTTS(L.LightningModule):
    def __init__(self, config):
        super(LitTTS, self).__init__()
        self.loss = TTSLoss(**config['model']['loss'])
        self.config = config
        self.model = TTS(**config['model']['tts'])
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        (
            _,
            phones,
            phone_lens,
            max_phone_lens,
            durs_padded,
            sparcs,
            sparc_lens,
            max_sparc_lens,
            accent_ids
        ) = batch
        enc_out, log_dur_pred, enc_mask = self.model.encoder(phones, phone_lens, max_phone_lens, accent_ids)
        attn_soft, attn_hard, attn_logprob, attn_hard_dur = self.model.aligner(enc_out, phone_lens, enc_mask, sparcs, sparc_lens, durs_padded)
        dec_out, sparc_mask = self.model.decoder(enc_out, sparc_lens, max_sparc_lens, attn_hard_dur)
        loss, metrics = self.loss(log_dur_pred, attn_hard_dur, phone_lens, enc_mask, dec_out, sparc_lens, sparc_mask, attn_soft, attn_hard, attn_logprob, sparcs)
        for metric_name, metric_value in metrics.items():
            self.log(f"train/{metric_name}", metric_value, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True, batch_size=self.config['train']['datamodule']['batch_size'])
        return loss

    def validation_step(self, batch, batch_idx):
        (
            _,
            phones,
            phone_lens,
            max_phone_lens,
            durs_padded,
            sparcs,
            sparc_lens,
            max_sparc_lens,
            accent_ids
        ) = batch
        enc_out, log_dur_pred, enc_mask = self.model.encoder(phones, phone_lens, max_phone_lens, accent_ids)
        attn_soft, attn_hard, attn_logprob, attn_hard_dur = self.model.aligner(enc_out, phone_lens, enc_mask, sparcs, sparc_lens, durs_padded)
        dec_out, sparc_mask = self.model.decoder(enc_out, sparc_lens, max_sparc_lens, attn_hard_dur)
        loss, metrics = self.loss(log_dur_pred, attn_hard_dur, phone_lens, enc_mask, dec_out, sparc_lens, sparc_mask, attn_soft, attn_hard, attn_logprob, sparcs)
        for metric_name, metric_value in metrics.items():
            self.log(f"val/{metric_name}", metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.config['train']['datamodule']['batch_size'])
        return loss

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Test step is not implemented yet.")
        # enc_out = self.model.encoder(batch)
        # dec_out = self.model.decoder(enc_out, batch)
        # loss = self.loss(batch, enc_out, dec_out)
        # print(loss)
        # return loss
    
    def forward(self, 
                phones,
                phone_lens,
                max_phone_lens=None,
                durs_padded=None,
                sparcs=None,
                sparc_lens=None,
                max_sparc_lens=5000,
                accent_ids=None, ):
        return self.model(phones, phone_lens, max_phone_lens, durs_padded, sparcs, sparc_lens, max_sparc_lens, accent_ids)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        (
            ids,
            phones,
            phone_lens,
            max_phone_lens,
            durs_padded,
            sparcs,
            sparc_lens,
            max_sparc_lens,
            accent_ids
        ) = batch

        if self.model.use_aligner_durations_if_possible:
            return (ids, self(phones, phone_lens, max_phone_lens, durs_padded, sparcs, sparc_lens, max_sparc_lens, accent_ids))
        
        return (ids, self(phones, phone_lens, max_phone_lens, max_sparc_lens=max_sparc_lens, accent_ids=accent_ids))

    def configure_optimizers(self):
        decay = []
        no_decay = []

        for module in self.modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                if module.weight is not None:
                    no_decay.append(module.weight)
                if module.bias is not None:
                    no_decay.append(module.bias)
            else:
                for param in module.parameters(recurse=False):
                    decay.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.config['train']['optimizer']['weight_decay']},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.config['train']['optimizer']['lr'],
        )

        total_steps = self.config['train']['trainer']['max_steps']
        warmup_steps = int(0.02 * total_steps)  # 2% warmup

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
