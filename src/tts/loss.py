import torch
import torch.nn as nn

import torch.nn.functional as F

class AttentionCTCLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(AttentionCTCLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.blank_logprob = blank_logprob
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)

    # def forward(self, attn_logprob, text_lens, mel_lens):
    #     """Calculate CTC alignment loss between embedded texts and mel features

    #     Args:
    #       attn_logprob: batch x 1 x max(mel_lens) x max(text_lens)
    #         Batched tensor of attention log probabilities, padded to length of
    #         longest sequence in each dimension.
    #       text_lens: batch-D vector of lengths of each text sequence
    #       mel_lens: batch-D vector of lengths of each mel sequence

    #     Returns:
    #       cost: Average CTC loss over batch
    #     """
    #     # Add blank token to attention matrix, with small emission probability
    #     # at all timesteps
    #     attn_logprob = F.pad(
    #         attn_logprob, pad=(1, 0, 0, 0, 0, 0), value=self.blank_logprob)

    #     cost = 0.0
    #     for bid in range(attn_logprob.shape[0]):
    #         # Construct target sequence: each text token is mapped to its
    #         # sequence index, enforcing monotonicity constraint
    #         target_seq = torch.arange(1, text_lens[bid] + 1).unsqueeze(0)

    #         curr_logprob = attn_logprob[bid].permute(1, 0, 2)
    #         curr_logprob = curr_logprob[:mel_lens[bid], :, :text_lens[bid] + 1]
    #         curr_logprob = self.log_softmax(curr_logprob[None])[0]
    #         cost += self.CTCLoss(curr_logprob, target_seq,
    #             input_lengths=mel_lens[bid], target_lengths=text_lens[bid])

    #     cost = cost / attn_logprob.shape[0]
    #     return cost

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        max_key_len = attn_logprob.size(-1)

        # Reorder input to [query_len, batch_size, key_len]
        attn_logprob = attn_logprob.squeeze(1)
        attn_logprob = attn_logprob.permute(1, 0, 2)

        # Add blank label
        attn_logprob = F.pad(
            input=attn_logprob,
            pad=(1, 0, 0, 0, 0, 0),
            value=self.blank_logprob)

        # Convert to log probabilities
        # Note: Mask out probs beyond key_len
        key_inds = torch.arange(
            max_key_len+1,
            device=attn_logprob.device,
            dtype=torch.long)
        attn_logprob.masked_fill_(
            key_inds.view(1,1,-1) > key_lens.view(1,-1,1), # key_inds >= key_lens+1
            -1e8)
        attn_logprob = self.log_softmax(attn_logprob)

        # Target sequences
        target_seqs = key_inds[1:].unsqueeze(0)
        target_seqs = target_seqs.repeat(key_lens.numel(), 1)

        # Evaluate CTC loss
        cost = self.CTCLoss(
            attn_logprob, target_seqs,
            input_lengths=query_lens, target_lengths=key_lens)
        return cost


class AttentionBinarizationLoss(torch.nn.Module):
    def __init__(self):
        super(AttentionBinarizationLoss, self).__init__()

    def forward(self, hard_attention, soft_attention, eps=1e-8):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=eps)).sum()
        return -log_sum / hard_attention.sum()

class TTSLoss(nn.Module):
    def __init__(self, dur_loss_weight=1.0, sparc_loss_weight=1.0, ctc_loss_weight=0.1, bin_loss_weight=1.0):
        super(TTSLoss, self).__init__()
        self.dur_loss_weight = dur_loss_weight
        self.sparc_loss_weight = sparc_loss_weight
        self.ctc_loss_weight = ctc_loss_weight
        self.bin_loss_weight = bin_loss_weight
        self.mse_loss = nn.MSELoss()
        self.attn_ctc_loss = AttentionCTCLoss()
        self.attn_bin_loss = AttentionBinarizationLoss()
    
    def forward(self, 
            log_dur_pred,
            dur_tgt,
            phone_lens,
            enc_mask,
            pred_sparc, 
            target_lens,
            sparc_masks,
            attn_soft,
            attn_hard,
            attn_logprob, 
            target_sparcs):


        target_sparcs = target_sparcs

        sparc_masks = ~sparc_masks

        pred_sparc = pred_sparc.masked_select(sparc_masks.unsqueeze(-1))
        target_sparcs = target_sparcs.masked_select(sparc_masks.unsqueeze(-1))

        log_dur_tgt = torch.log(dur_tgt.float() + 1).masked_select(~enc_mask)
        log_dur_pred = log_dur_pred.masked_select(~enc_mask)

        dur_loss = self.mse_loss(log_dur_tgt, log_dur_pred)

        attn_loss = self.attn_ctc_loss(attn_logprob, phone_lens, target_lens)

        bin_loss = self.attn_bin_loss(attn_hard, attn_soft)

        sparc_loss = self.mse_loss(pred_sparc, target_sparcs)

        loss = self.dur_loss_weight * dur_loss + self.ctc_loss_weight * attn_loss + self.bin_loss_weight * bin_loss + self.sparc_loss_weight * sparc_loss

        return loss, {
            'loss': loss.clone().detach().item(),
            'sparc_loss': sparc_loss.clone().detach().item(),
            'duration_loss': dur_loss.clone().detach().item(),
            'attn_loss': attn_loss.clone().detach().item(),
            'bin_loss': bin_loss.clone().detach().item(),
        }

