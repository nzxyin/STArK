import numba
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import get_mask_from_lengths
from .conformer import Conformer
from .layers import SepConv1d, ConvNorm
# from symbols import symbols
# from symbols_arpa import symbols as arpa_symbols
from ipa import symbols

def _regulate_len(durations, enc_out, pace=1.0, max_dec_len=None):
    """If target=None, then predicted durations are applied"""

    dtype = enc_out.dtype
    reps = durations.float() / pace
    reps = (reps + 0.5).long()
    dec_lens = reps.sum(dim=1)

    max_len = dec_lens.max()
    reps_cumsum = torch.cumsum(F.pad(reps, (1, 0, 0, 0), value=0.0),
                               dim=1)[:, None, :]
    reps_cumsum = reps_cumsum.to(dtype)

    range_ = torch.arange(max_len).to(enc_out.device)[None, :, None]
    mult = ((reps_cumsum[:, :, :-1] <= range_) &
            (reps_cumsum[:, :, 1:] > range_))
    mult = mult.to(dtype)
    enc_rep = torch.matmul(mult, enc_out)

    if max_dec_len is not None:
        if max_dec_len < enc_rep.shape[1]:
            enc_rep = enc_rep[:, :max_dec_len]
            dec_lens = torch.clip(dec_lens, max_dec_len)
        else:
            enc_rep = F.pad(enc_rep, (0,0,0,max_dec_len - enc_rep.shape[1]), value=0.0)
            assert enc_rep.shape[1] == max_dec_len, f"{max_dec_len}, {enc_rep.shape}"
        
    return enc_rep, dec_lens

@numba.njit(cache=True)
def _mas_width1_numba(log_attn_map: np.ndarray) -> np.ndarray:
    # log_attn_map: [mel, text], float32. Same recurrence as the original
    # pure-Python/TorchScript _mas_width1 below (kept for reference and as a correctness
    # cross-check): log_p[i,j] = log_attn_map[i,j] + max(log_p[i-1,j-1], log_p[i-1,j]), first
    # row constrained to column 0, then backtrack the optimal monotonic path.
    mel, text = log_attn_map.shape
    neg_inf = np.float32(-1e8)
    log_p = log_attn_map.copy()
    for j in range(1, text):
        log_p[0, j] = neg_inf
    for i in range(1, mel):
        prev_log1 = neg_inf
        for j in range(text):
            prev_log2 = log_p[i - 1, j]
            if prev_log1 > prev_log2:
                log_p[i, j] += prev_log1
            else:
                log_p[i, j] += prev_log2
            prev_log1 = prev_log2

    opt = np.zeros_like(log_p)
    j = text - 1
    for i in range(mel - 1, 0, -1):
        opt[i, j] = 1.0
        if j > 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
            j -= 1
            if j == 0:
                for k in range(1, i):
                    opt[k, j] = 1.0
                break
    opt[0, j] = 1.0
    return opt


@numba.njit(cache=True, parallel=True)
def _binarize_attention_numba_kernel(log_attn: np.ndarray, in_lens: np.ndarray, out_lens: np.ndarray,
                                      max_mel: int, max_text: int) -> np.ndarray:
    B = log_attn.shape[0]
    attn_out = np.zeros((B, max_mel, max_text), dtype=np.float32)
    for b in numba.prange(B):
        mel_len = out_lens[b]
        text_len = in_lens[b]
        opt = _mas_width1_numba(log_attn[b, :mel_len, :text_len])
        attn_out[b, :mel_len, :text_len] = opt
    return attn_out


# Taken from https://github.com/dan-wells/fastpitch
def _mas_width1(log_attn_map: torch.Tensor) -> torch.Tensor:
    """Reference (pure Python/TorchScript) implementation -- kept only as a correctness
    cross-check for _mas_width1_numba, no longer called from _binarize_attention. See that
    function's docstring for why it was replaced."""
    # log_attn_map: [mel, text]
    mel = log_attn_map.size(0)
    text = log_attn_map.size(1)

    neg_inf = log_attn_map.new_full((), -1e8)

    # clone so we don't modify input
    log_p = log_attn_map.detach().clone()

    # first row constraint
    for j in range(1, text):
        log_p[0, j] = neg_inf

    # forward DP
    for i in range(1, mel):
        prev_log1 = neg_inf
        for j in range(text):
            prev_log2 = log_p[i - 1, j]
            if prev_log1 > prev_log2:
                log_p[i, j] += prev_log1
            else:
                log_p[i, j] += prev_log2
            prev_log1 = prev_log2

    # backtracking
    opt = torch.zeros_like(log_p)
    j = text - 1

    for i in range(mel - 1, 0, -1):
        opt[i, j] = 1.0
        if j > 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
            j -= 1
            if j == 0:
                for k in range(1, i):
                    opt[k, j] = 1.0
                break

    opt[0, j] = 1.0
    return opt

# Taken from https://github.com/dan-wells/fastpitch
def _binarize_attention(
    attn: torch.Tensor,
    in_lens: torch.Tensor,
    out_lens: torch.Tensor,) -> torch.Tensor:
    """For training purposes only. Binarizes attention with MAS.
        These will no longer recieve a gradient.

    Args:
        attn: B x 1 x max_mel_len x max_text_len
        in_lens: B
        out_lens: B

    Numba-accelerated (ported from a sibling repo's unpushed local fix, 2026-08-21) -- was a
    hand-written, doubly-nested PURE PYTHON DP loop (_mas_width1 above, originally
    @torch.jit.script'd, now kept only as a reference/correctness cross-check), run via a
    Python-level `for b in range(B)` loop, ONE BATCH ELEMENT AT A TIME ON A SINGLE CPU CORE,
    every training step. Benchmarked there on a realistic curriculum batch (B=48, max_mel=981,
    max_text=297, ~3.7M mel*text work units total): 16,626 ms/call for the original TorchScript
    version vs. 42.1 ms/call with numba njit + prange parallelizing across the batch dimension
    (actually using the multiple CPUs the job is allocated instead of one) -- ~395x, bit-identical
    output (torch.allclose, verified there against 5 random-shape cases incl. batch_size=1 and
    length-1 edges, bf16 input, repeated calls, and full Aligner.forward integration). This was
    almost certainly a dominant training-throughput bottleneck, not merely a contributor, and
    likely explains some of the throughput variance seen independently while investigating this
    model's DDP hang (see train_large_100k.sh) -- the old implementation's cost scales directly
    with each batch's mel*text size, with zero parallelism to absorb the variance.

    `.float()` before `.numpy()` is required, not defensive-only: numpy has no bfloat16 dtype,
    and this model trains under bf16-mixed precision -- attn can genuinely arrive here as bf16
    (the original version never hit this because it stayed in pure PyTorch/TorchScript tensor
    ops throughout, never converting to numpy).
    """
    B = attn.size(0)
    max_mel = attn.size(2)
    max_text = attn.size(3)

    attn_cpu = attn.detach().float().cpu()
    log_attn = torch.log(attn_cpu).squeeze(1).contiguous().numpy()

    in_lens_np = in_lens.detach().cpu().numpy().astype(np.int64)
    out_lens_np = out_lens.detach().cpu().numpy().astype(np.int64)

    attn_out_np = _binarize_attention_numba_kernel(log_attn, in_lens_np, out_lens_np, max_mel, max_text)

    return torch.from_numpy(attn_out_np).unsqueeze(1).to(dtype=attn.dtype)

class ConvAttention(nn.Module):
    def __init__(self, n_sparc_channels=14, n_text_channels=256, n_att_channels=64):
        super(ConvAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.key_proj = nn.Sequential(
            ConvNorm(n_text_channels,
                     n_text_channels * 2,
                     kernel_size=3,
                     padding=1),
            nn.ReLU(),
            ConvNorm(n_text_channels * 2,
                     n_att_channels,
                     kernel_size=1,)
        )
        self.query_proj = nn.Sequential(
            ConvNorm(n_sparc_channels,
                        n_sparc_channels * 2,
                        kernel_size=3,
                        padding=1),
            nn.ReLU(),
            ConvNorm(n_sparc_channels * 2,
                        n_sparc_channels * 2,
                        kernel_size=1,),
            nn.ReLU(),
            ConvNorm(n_sparc_channels * 2,
                        n_att_channels,
                        kernel_size=1,)
        )

    def forward(self, queries, keys, mask=None, attn_prior=None):
        """Attention mechanism for flowtron parallel
        Unlike in Flowtron, we have no restrictions such as causality etc,
        since we only need this during training.

        Args:
            queries (torch.tensor): B x T1 x C1 tensor (sparc data)
            keys (torch.tensor): B x T2 x C2 tensor (text data)
            mask (torch.tensor): B x T2 uint8 binary mask for variable length entries
                (should be in the T2 domain)
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask.
                Final dim T2 should sum to 1
        """
        keys_enc = self.key_proj(keys).transpose(1,2)  # B x n_attn_dims x T2

        # Beware can only do this since query_dim = attn_dim = n_mel_channels
        queries_enc = self.query_proj(queries).transpose(1,2)

        # different ways of computing attn,
        # one is isotopic gaussians (per phoneme)
        # Simplistic Gaussian Isotopic Attention

        # B x n_attn_dims x T1 x T2
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2
        # compute log likelihood from a gaussian
        attn = -0.0005 * attn.sum(1, keepdim=True)
        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None]+1e-8)

        attn_logprob = attn.clone()

        if mask is not None:
            attn.data.masked_fill_(mask[:, None, None, :], -1e8)

        attn = self.softmax(attn)  # Softmax along T2
        return attn, attn_logprob

class TemporalPredictor(nn.Module):

    def __init__(self, input_size, filter_size, kernel_size, dropout):
        super(TemporalPredictor, self).__init__()

        self.conv_1 = SepConv1d(input_size, filter_size, kernel_size, stride=1, padding=kernel_size//2)
        self.norm_1 = nn.LayerNorm(filter_size)
        self.conv_2 = SepConv1d(filter_size, filter_size, kernel_size, stride=1, padding=kernel_size//2)
        self.norm_2 = nn.LayerNorm(filter_size)

        self.fc = nn.Linear(filter_size, 1)
        self.dropout = nn.Dropout(dropout)


    def forward(self, input, mask):
        """
        input: torch.tensor (B, L, d_model)
        mask: torch.tensor uint8 (B, L)
        ---
        output: torch.tensor (B, L)
        """
        out = input.masked_fill(mask.unsqueeze(-1), 0.)
        out = self.dropout(self.norm_1(F.relu(self.conv_1(out.transpose(1,2)).transpose(1,2))))
        out = self.dropout(self.norm_2(F.relu(self.conv_2(out.transpose(1,2)).transpose(1,2))))
        out = self.fc(out).masked_fill(mask.unsqueeze(-1), 0.)
        return out.squeeze(-1)
    
class Encoder(nn.Module):
    def __init__(self, num_phonemes, num_accents, model_dim=256, n_layers=4, ff_dim=1024, kernel_size=9, ff_kernel_size=3, n_heads=2, dropout=0.1, 
                 predictor_filter_size=256, predictor_kernel_size=3, predictor_dropout=0.5, speaker_emb_dim=None, speaker_emb_dur_only=False):
        super(Encoder, self).__init__()
        self.phone_embedding = nn.Embedding(num_phonemes, model_dim, padding_idx=0)
        self.accent_embedding = nn.Embedding(num_accents, model_dim)
        if speaker_emb_dim is not None:
            self.speaker_projection = nn.Linear(speaker_emb_dim, model_dim, bias=False)
        self.conformer = Conformer(n_layers=n_layers, d_model=model_dim, d_ff=ff_dim, 
                                kernel_size=kernel_size, ff_kernel_size=ff_kernel_size, n_head=n_heads, dropout=dropout, position_emb='rotary')
        self.duration_predictor = TemporalPredictor(model_dim, predictor_filter_size, predictor_kernel_size, predictor_dropout)

    def forward(self, phones, phone_lens, max_phone_len, accent_id, speaker_emb=None):

        enc_mask = get_mask_from_lengths(phone_lens, max_phone_len)
        x = self.phone_embedding(phones)
        accent_emb = self.accent_embedding(accent_id)
        if speaker_emb is not None:
            speaker_emb = self.speaker_projection(speaker_emb)
            x = x + speaker_emb[:, None, :]
        x = x + accent_emb[:, None, :]
        x = self.conformer(x, key_padding_mask=enc_mask)
        log_dur_pred = self.duration_predictor(x, enc_mask)
        return x, log_dur_pred, enc_mask
    
class Decoder(nn.Module):
    def __init__(self, sparc_dim=15, model_dim=256, n_layers=6, ff_dim=1024, kernel_size=9, ff_kernel_size=3, n_heads=2, dropout=0.1):
        super(Decoder, self).__init__()
        self.conformer = Conformer(n_layers=n_layers, d_model=model_dim, d_ff=ff_dim, 
                                kernel_size=kernel_size, ff_kernel_size=ff_kernel_size, n_head=n_heads, dropout=dropout, position_emb=None)
        self.sparc_proj = nn.Linear(model_dim, sparc_dim)

    def forward(self, x, sparc_lens, max_sparc_len, durations):
        x, _ = _regulate_len(durations, x, max_dec_len=max_sparc_len)
        sparc_mask = get_mask_from_lengths(sparc_lens, max_sparc_len)
        x = self.conformer(x, key_padding_mask=sparc_mask)
        x = self.sparc_proj(x)
        return x, sparc_mask

class Aligner(nn.Module):
    def __init__(self, n_sparc_channels=14, n_text_channels=256, n_att_channels=64):
        super(Aligner, self).__init__()
        self.attention = ConvAttention(n_text_channels=n_text_channels, n_sparc_channels=n_sparc_channels, n_att_channels=n_att_channels)

    def forward(self, enc_out, phone_lens, enc_mask, sparc, sparc_lens, attn_prior=None):
        
        attn_soft, attn_logprob = self.attention(sparc.float(), enc_out.float(), enc_mask, attn_prior=attn_prior)
        attn_hard = _binarize_attention(attn_soft, phone_lens, sparc_lens).to(attn_soft.device)
        attn_hard_dur = attn_hard.sum(2)[:, 0, :]
        assert torch.all(torch.eq(attn_hard_dur.sum(dim=1), sparc_lens))
        return attn_soft, attn_hard, attn_logprob, attn_hard_dur

class TTS(nn.Module):
    def __init__(self, num_accents=1, num_phonemes=len(symbols), use_unilex=True, sparc_dim=14, model_dim=512, 
                 encoder_n_layers=4, encoder_ff_dim=1024, encoder_kernel_size=9, encoder_ff_kernel_size=3, encoder_n_heads=2, 
                 n_aligner_channels=64, predictor_filter_size=256, predictor_kernel_size=3, predictor_dropout=0.5,
                 decoder_n_layers=4, decoder_ff_dim=1024, decoder_kernel_size=9, decoder_ff_kernel_size=3, decoder_n_heads=2, dropout=0.2, use_aligner_durations_if_possible=False):
        super(TTS, self).__init__()

        # if use_unilex:
        #     num_phonemes = len(symbols)
        #     print('model is using unilex phoneme set')
        # else:
        #     num_phonemes = len(arpa_symbols)
        #     print('model is using arpabet phoneme set')

        self.encoder = Encoder(num_phonemes, num_accents, model_dim, encoder_n_layers, encoder_ff_dim,
                               encoder_kernel_size, encoder_ff_kernel_size, encoder_n_heads, dropout,
                               predictor_filter_size, predictor_kernel_size, predictor_dropout)
        
        self.aligner = Aligner(n_sparc_channels=sparc_dim, n_text_channels=model_dim, n_att_channels=n_aligner_channels)

        self.decoder = Decoder(sparc_dim, model_dim, decoder_n_layers, decoder_ff_dim,
                               decoder_kernel_size, decoder_ff_kernel_size, decoder_n_heads, dropout)
        
        self.use_aligner_durations_if_possible = use_aligner_durations_if_possible
    
    def forward(self,
                phones,
                phone_lens,
                max_phone_lens=None,
                durs_padded=None,
                sparcs=None,
                sparc_lens=None,
                max_sparc_lens=5000,
                accent_ids=None, ):
        enc_out, log_dur_pred, enc_mask = self.encoder(phones, phone_lens, max_phone_lens, accent_ids)
        if self.use_aligner_durations_if_possible and sparcs is not None and sparc_lens is not None and durs_padded is not None:
            attn_soft, attn_hard, attn_logprob, attn_hard_dur = self.aligner(enc_out, phone_lens, enc_mask, sparcs, sparc_lens, durs_padded)
            durations = attn_hard_dur
            pred_sparc, sparc_mask = self.decoder(enc_out, sparc_lens, max_sparc_lens, durations)
        else:
            if self.use_aligner_durations_if_possible:
                print("Warning: Aligner durations not used during inference since aligner inputs not provided.")
            durations = torch.clip(torch.exp(log_dur_pred) - 1, 0).long()
            durations = durations.masked_fill(enc_mask, 0)
            sparc_lens = durations.sum(dim=1).long()
            sparc_lens = torch.clip(sparc_lens, 0, max_sparc_lens)
            pred_sparc, sparc_mask = self.decoder(enc_out, sparc_lens, max_sparc_lens, durations)
            attn_soft, attn_hard, attn_logprob = None, None, None
        
        return (
            pred_sparc,
            sparc_mask,
            durations,
            attn_soft,
            attn_hard,
            attn_logprob,
        )

# class TTS(nn.Module):
#     def __init__(self, num_accents=2, use_unilex=True, sparc_dim=15, model_dim=256, encoder_n_layers=4, encoder_ff_dim=1024, encoder_kernel_size=9, encoder_ff_kernel_size=3, encoder_n_heads=2, 
#                  n_aligner_channels=64, predictor_filter_size=256, predictor_kernel_size=3, predictor_dropout=0.5,
#                  decoder_n_layers=6, decoder_ff_dim=1024, decoder_kernel_size=9, decoder_ff_kernel_size=3, decoder_n_heads=2, dropout=0.2):
#         super(TTS, self).__init__()

#         self.model_dim = model_dim

#         if use_unilex:
#             self.phone_embedding = nn.Embedding(len(symbols), model_dim, padding_idx=0)
#             print('model is using unilex phoneme set')
#         else:
#             self.phone_embedding = nn.Embedding(len(arpa_symbols), model_dim, padding_idx=0)
#             print('model is using arpabet phoneme set')

#         self.accent_embedding = nn.Embedding(num_accents, model_dim)

#         self.phone_encoder = Conformer(n_layers=encoder_n_layers, d_model=model_dim, d_ff=encoder_ff_dim, 
#                                 kernel_size=encoder_kernel_size, ff_kernel_size=encoder_ff_kernel_size, n_head=encoder_n_heads, dropout=dropout)

#         self.sparc_decoder = Conformer(decoder_n_layers, model_dim, decoder_ff_dim, decoder_kernel_size, decoder_ff_kernel_size, decoder_n_heads, dropout)

#         self.attention = ConvAttention(n_text_channels=model_dim, n_sparc_channels=sparc_dim, n_att_channels=n_aligner_channels)

#         self.duration_predictor = TemporalPredictor(model_dim, predictor_filter_size, predictor_kernel_size, predictor_dropout)
#         # self.loudness_predictor = TemporalPredictor(model_dim, predictor_filter_size, predictor_kernel_size, predictor_dropout)
#         # self.pitch_predictor = TemporalPredictor(model_dim, predictor_filter_size, predictor_kernel_size, predictor_dropout)

#         self.sparc_proj = nn.Linear(model_dim, sparc_dim)

#     # Taken from https://github.com/dan-wells/fastpitch
#     def _binarize_attention(self, attn, in_lens, out_lens):
#         """For training purposes only. Binarizes attention with MAS.
#            These will no longer recieve a gradient.

#         Args:
#             attn: B x 1 x max_mel_len x max_text_len
#         """
#         b_size = attn.shape[0]
#         with torch.no_grad():
#             attn_out_cpu = np.zeros(attn.data.shape, dtype=np.float32)
#             log_attn_cpu = torch.log(attn.data).to(device='cpu', dtype=torch.float32)
#             log_attn_cpu = log_attn_cpu.numpy()
#             out_lens_cpu = out_lens.cpu()
#             in_lens_cpu = in_lens.cpu()
#             for ind in range(b_size):
#                 hard_attn = mas_width1(
#                     log_attn_cpu[ind, 0, :out_lens_cpu[ind], :in_lens_cpu[ind]])
#                 attn_out_cpu[ind, 0, :out_lens_cpu[ind], :in_lens_cpu[ind]] = hard_attn
#             attn_out = torch.tensor(
#                 attn_out_cpu, device=attn.get_device(), dtype=attn.dtype)
#         return attn_out

#     def forward(self, phones, phone_lens, max_phone_len, attn_prior, sparc, target_lens, max_target_len, accent_label):
#         # phones -> masked -> embeddings
#         if torch.isnan(phones).any() or torch.isinf(phones).any():
#             raise TypeError("NaN or Inf in input data detected!")
#         if torch.isnan(attn_prior).any() or torch.isinf(attn_prior).any():
#             raise TypeError("NaN or Inf in input data detected!")
#         if torch.isnan(sparc).any() or torch.isinf(sparc).any():
#             raise TypeError("NaN or Inf in input data detected!")
#         if (phone_lens < 1).any():
#             raise TypeError("Length zero input detected.")

#         enc_mask = get_mask_from_lengths(phone_lens, max_phone_len)
        
#         embs = self.phone_embedding(phones)

#         accent_emb = self.accent_embedding(accent_label)
#         # embeddings -> encoded_embs
#         assert embs.shape == (phones.shape[0], max_phone_len, self.model_dim)

#         enc_out = self.phone_encoder(embs + accent_emb[:, None, :], key_padding_mask=enc_mask)
#         # _regulate_lengths(encoded_embs, durations) -> expanded_embs
#         # expanded_embs -> decoded_embs
#         log_dur_pred = self.duration_predictor(enc_out, enc_mask)
#         assert log_dur_pred.shape == (phones.shape[0], max_phone_len)

#         # attn_soft, attn_logprob = self.attention(
#         #     sparc.float(), enc_out.float(), enc_mask, attn_prior=attn_prior)
        
#         attn_soft, attn_logprob = self.attention(
#             sparc.float(), embs.float(), enc_mask, attn_prior=attn_prior)
        
#         attn_hard = self._binarize_attention(attn_soft, phone_lens, target_lens)

#         # Viterbi --> durations
#         attn_hard_dur = attn_hard.sum(2)[:, 0, :]
#         dur_tgt = attn_hard_dur
#         assert torch.all(torch.eq(dur_tgt.sum(dim=1), target_lens))

#         expanded_embs, dec_lens = _regulate_len(dur_tgt, enc_out, max_dec_len=max_target_len)
#         sparc_masks = get_mask_from_lengths(dec_lens, max_target_len)
#         decoded_embs = self.sparc_decoder(expanded_embs, key_padding_mask=sparc_masks)
#         # decoded_embs -> pred_sparc
#         pred_sparc = self.sparc_proj(decoded_embs)


#         return (
#             log_dur_pred,
#             dur_tgt,
#             phone_lens,
#             enc_mask,
#             pred_sparc, 
#             dec_lens,
#             sparc_masks,
#             attn_soft,
#             attn_hard,
#             attn_logprob,
#         )
    
#     @torch.no_grad()
#     def inference(self, phones, phone_lens, max_phone_len, max_target_len, accent_label):
#         enc_mask = get_mask_from_lengths(phone_lens, max_phone_len)
        
#         embs = self.phone_embedding(phones)

#         accent_emb = self.accent_embedding(accent_label)
#         # embeddings -> encoded_embs
#         assert embs.shape == (phones.shape[0], max_phone_len, self.model_dim)

#         enc_out = self.phone_encoder(embs + accent_emb[:, None, :], key_padding_mask=enc_mask)
#         # _regulate_lengths(encoded_embs, durations) -> expanded_embs
#         # expanded_embs -> decoded_embs
#         log_dur_pred = self.duration_predictor(enc_out, enc_mask)
#         assert log_dur_pred.shape == (phones.shape[0], max_phone_len)
#         dur_pred = torch.clip(torch.exp(log_dur_pred) - 1, 0)
#         expanded_embs, dec_lens = _regulate_len(dur_pred, enc_out, max_dec_len=max_target_len)
#         sparc_masks = get_mask_from_lengths(dec_lens, max_target_len)
#         decoded_embs = self.sparc_decoder(expanded_embs, key_padding_mask=sparc_masks)
#         # decoded_embs -> pred_sparc
#         pred_sparc = self.sparc_proj(decoded_embs)
#         return (
#             pred_sparc, 
#             dur_pred,
#             sparc_masks,
#         )
    

