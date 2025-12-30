#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer-based Chinese→English NMT training script (PyTorch, from scratch)

Data format matches train_rnn_nmt.py:
  preprocessed_data/
    vocab_zh.pkl
    vocab_en.pkl
    train_ids.pkl
    valid_ids.pkl
    test_ids.pkl
    config.pkl (optional)

Meets assignment knobs:
- Positional encoding: absolute (sinusoidal) vs relative (T5-style relative position bias)
- Normalization: LayerNorm vs RMSNorm
- Hyperparams: batch size, learning rate
- Model scale: tiny/small/base/large (overridable by explicit d_model/n_heads/d_ff/num_layers)

Extra for convenience (also helpful for your Analysis section):
- decoding: greedy vs beam (beam supports batch_size=1 like your RNN script)
- BLEU eval via sacrebleu (+ optional sentencepiece detok)

Install:
  pip install torch tqdm sacrebleu sentencepiece

Example:
  python train_transformer_nmt.py --data_dir preprocessed_data \
    --pos_enc absolute --norm layernorm --model_scale small \
    --batch_size 64 --lr 3e-4 --epochs 10 --decode greedy \
    --use_test_for_final --spm_model preprocessed_data/spm_en.model
"""

import os
import math
import time
import json
import pickle
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sacrebleu


# -------------------------
# Config / Utils
# -------------------------
@dataclass
class PreprocessConfig:
    max_len: int
    min_len: int
    truncate: bool
    zh_min_freq: int
    en_min_freq: int
    zh_max_vocab: Optional[int]
    en_max_vocab: Optional[int]
    train_path: str
    valid_path: str
    test_path: str
    seed: int


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -------------------------
# Dataset + Collate
# -------------------------
class NMTIdsDataset(Dataset):
    """Each item: (src_ids: List[int], tgt_ids: List[int])"""
    def __init__(self, pairs: List[Tuple[List[int], List[int]]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


def collate_fn_factory(src_pad_id: int, tgt_pad_id: int):
    def pad_1d(x: List[int], pad_id: int, L: int) -> List[int]:
        if len(x) >= L:
            return x[:L]
        return x + [pad_id] * (L - len(x))

    def collate(batch):
        src_lens = [len(s) for s, _ in batch]
        tgt_lens = [len(t) for _, t in batch]
        src_max = max(src_lens)
        tgt_max = max(tgt_lens)

        src = [pad_1d(s, src_pad_id, src_max) for s, _ in batch]
        tgt = [pad_1d(t, tgt_pad_id, tgt_max) for _, t in batch]

        src = torch.tensor(src, dtype=torch.long)  # (B,S)
        tgt = torch.tensor(tgt, dtype=torch.long)  # (B,T)
        src_lens = torch.tensor(src_lens, dtype=torch.long)
        tgt_lens = torch.tensor(tgt_lens, dtype=torch.long)
        return src, tgt, src_lens, tgt_lens

    return collate

class LabelSmoothingLoss(nn.Module):
    """
    Standard label smoothing for seq2seq.
    Uses KLDivLoss between log-probs and a smoothed target distribution.
    Ignores pad positions.
    """
    def __init__(self, vocab_size: int, pad_id: int, smoothing: float = 0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.kl = nn.KLDivLoss(reduction="sum")  # we'll normalize by #non-pad tokens

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (N, V)
        target: (N,)
        """
        log_probs = torch.log_softmax(logits, dim=-1)  # (N,V)

        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
            # place confidence on true labels
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            # pad positions -> zero contribution
            pad_mask = target.eq(self.pad_id)  # (N,)
            true_dist[pad_mask] = 0.0

        loss = self.kl(log_probs, true_dist)
        denom = (~target.eq(self.pad_id)).sum().clamp_min(1)
        return loss / denom


class LinearWarmupInverseSqrtScheduler:
    """
    Linear warmup then inverse sqrt decay.
    lr(step) = peak_lr * step/warmup_steps                     if step <= warmup_steps
             = peak_lr * sqrt(warmup_steps) / sqrt(step)       else
    This keeps lr continuous at warmup boundary.
    """
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, peak_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = max(1, int(warmup_steps))
        self.peak_lr = float(peak_lr)
        self._step = 0

        # initialize lr very small to avoid huge first updates
        self._set_lr(self.peak_lr / self.warmup_steps)

    def _get_lr(self, step: int) -> float:
        if step <= self.warmup_steps:
            return self.peak_lr * step / self.warmup_steps
        return self.peak_lr * math.sqrt(self.warmup_steps) / math.sqrt(step)

    def _set_lr(self, lr: float):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self):
        self._step += 1
        lr = self._get_lr(self._step)
        self._set_lr(lr)
        return lr

    @property
    def step_num(self):
        return self._step

    @property
    def lr(self):
        return self.optimizer.param_groups[0]["lr"]


# -------------------------
# Norms
# -------------------------
class RMSNorm(nn.Module):
    """RMSNorm: normalize by RMS (no mean subtraction), then scale."""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        return x_norm * self.weight


def build_norm(norm_type: str, dim: int):
    norm_type = norm_type.lower()
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"Unknown norm: {norm_type}")


# -------------------------
# Positional encoding
# -------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Classic absolute sinusoidal positional encoding (Vaswani et al.)."""
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0).to(x.dtype)


class T5RelativePositionBias(nn.Module):
    """
    T5-style relative position bias (used as "relative position encoding" via attention logits bias).
    Produces bias of shape: (1, n_heads, qlen, klen)
    """
    def __init__(self, n_heads: int, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.n_heads = n_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """
        relative_position: (qlen, klen) with values in [-inf, +inf]
        Returns bucket ids in [0, num_buckets)
        Following T5 bucketing idea: exact for small distances, logarithmic for large.
        """
        num_buckets = self.num_buckets
        max_distance = self.max_distance

        # sign: negative => key is ahead? Here we use bidirectional in encoder self-attn,
        # but for decoder self-attn we only attend to previous positions (causal mask),
        # still fine to bucket generally.
        ret = 0
        n = -relative_position  # make "distance" positive when key is in the past
        # n can be negative for future positions; we still bucket them (but they will be masked in decoder)
        sign = (n < 0).to(torch.long)
        n = n.abs()

        half = num_buckets // 2
        # smaller buckets for small distances
        max_exact = half
        is_small = n < max_exact

        val_if_large = max_exact + (
            (torch.log(n.float() / max_exact + 1e-6) / math.log(max_distance / max_exact))
            * (half - 1)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, half - 1))

        bucket = torch.where(is_small, n.to(torch.long), val_if_large)  # [0, half)
        # separate buckets by sign
        bucket = bucket + sign * half  # [0, num_buckets)
        bucket = torch.clamp(bucket, 0, num_buckets - 1)
        return bucket

    def forward(self, qlen: int, klen: int, device: torch.device) -> torch.Tensor:
        context_position = torch.arange(qlen, device=device)[:, None]
        memory_position = torch.arange(klen, device=device)[None, :]
        relative_position = memory_position - context_position  # (qlen, klen)
        rp_bucket = self._relative_position_bucket(relative_position)
        # (qlen, klen, n_heads) -> (1, n_heads, qlen, klen)
        values = self.relative_attention_bias(rp_bucket)  # (qlen, klen, n_heads)
        return values.permute(2, 0, 1).unsqueeze(0)


# -------------------------
# Multi-head attention (with optional relative bias)
# -------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,               # (B, Q, D)
        k: torch.Tensor,               # (B, K, D)
        v: torch.Tensor,               # (B, K, D)
        attn_mask: Optional[torch.Tensor] = None,     # (1 or B, 1 or H, Q, K) bool where True means keep
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, K) bool where True means keep (not pad)
        rel_pos_bias: Optional[torch.Tensor] = None,  # (1, H, Q, K)
    ) -> torch.Tensor:
        B, QL, _ = q.shape
        Bk, KL, _ = k.shape
        assert B == Bk

        q = self.Wq(q).view(B, QL, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,Q,dh)
        k = self.Wk(k).view(B, KL, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,K,dh)
        v = self.Wv(v).view(B, KL, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,K,dh)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,Q,K)

        if rel_pos_bias is not None:
            scores = scores + rel_pos_bias.to(scores.dtype)

        # key padding mask: True for valid (non-pad)
        if key_padding_mask is not None:
            # mask pad positions => -inf
            # key_padding_mask: (B,K) True means keep
            pad_mask = ~key_padding_mask  # True for PAD
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))

        # attn_mask: True means keep
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B,H,Q,dh)
        out = out.transpose(1, 2).contiguous().view(B, QL, self.d_model)
        out = self.Wo(out)
        return out


# -------------------------
# Transformer blocks (Pre-LN style for stability)
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, norm_type: str):
        super().__init__()
        self.norm1 = build_norm(norm_type, d_model)
        self.norm2 = build_norm(norm_type, d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor, rel_pos_bias: Optional[torch.Tensor]):
        # Pre-Norm
        h = self.norm1(x)
        x = x + self.attn(h, h, h, key_padding_mask=src_key_padding_mask, rel_pos_bias=rel_pos_bias)
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, norm_type: str):
        super().__init__()
        self.norm1 = build_norm(norm_type, d_model)
        self.norm2 = build_norm(norm_type, d_model)
        self.norm3 = build_norm(norm_type, d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_attn_mask: torch.Tensor,          # (1,1,T,T) causal keep-mask
        tgt_key_padding_mask: torch.Tensor,   # (B,T) True keep
        src_key_padding_mask: torch.Tensor,   # (B,S) True keep
        rel_pos_bias: Optional[torch.Tensor], # (1,H,T,T) for decoder self-attn
    ):
        h = self.norm1(x)
        x = x + self.self_attn(
            h, h, h,
            attn_mask=tgt_attn_mask,
            key_padding_mask=tgt_key_padding_mask,
            rel_pos_bias=rel_pos_bias
        )
        h = self.norm2(x)
        x = x + self.cross_attn(
            h, memory, memory,
            key_padding_mask=src_key_padding_mask,
            rel_pos_bias=None
        )
        h = self.norm3(x)
        x = x + self.ffn(h)
        return x


# -------------------------
# Full Transformer NMT
# -------------------------
class TransformerNMT(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        src_pad_id: int,
        tgt_pad_id: int,
        tgt_bos_id: int,
        tgt_eos_id: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
        pos_enc: str,
        norm_type: str,
        max_len: int = 2048,
        rel_num_buckets: int = 32,
        rel_max_distance: int = 128,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.tgt_bos_id = tgt_bos_id
        self.tgt_eos_id = tgt_eos_id

        self.d_model = d_model
        self.pos_enc = pos_enc.lower()
        self.norm_type = norm_type.lower()

        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=src_pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=tgt_pad_id)

        self.emb_dropout = nn.Dropout(dropout)

        # absolute pos enc (sinusoidal); relative uses bias per layer (shared module is OK)
        self.abs_pos = SinusoidalPositionalEncoding(d_model, max_len=max_len) if self.pos_enc == "absolute" else None
        self.rel_bias_enc = T5RelativePositionBias(n_heads, rel_num_buckets, rel_max_distance) if self.pos_enc == "relative" else None
        self.rel_bias_dec = T5RelativePositionBias(n_heads, rel_num_buckets, rel_max_distance) if self.pos_enc == "relative" else None

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, norm_type) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, norm_type) for _ in range(num_layers)
        ])

        self.final_norm = build_norm(norm_type, d_model)
        self.lm_head = nn.Linear(d_model, tgt_vocab, bias=False)

        if tie_embeddings:
            if self.lm_head.weight.shape == self.tgt_emb.weight.shape:
                self.lm_head.weight = self.tgt_emb.weight

    def make_key_padding_mask(self, ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        # True for valid tokens (keep), False for pad
        return ids.ne(pad_id)

    def make_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        # keep-mask: True for allowed attention
        # (1,1,T,T)
        mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        return mask.view(1, 1, T, T)

    def encode(self, src_ids: torch.Tensor) -> torch.Tensor:
        # src_ids: (B,S)
        x = self.src_emb(src_ids) * math.sqrt(self.d_model)
        if self.abs_pos is not None:
            x = self.abs_pos(x)
        x = self.emb_dropout(x)

        src_kpm = self.make_key_padding_mask(src_ids, self.src_pad_id)  # (B,S)
        rel = None
        if self.rel_bias_enc is not None:
            rel = self.rel_bias_enc(qlen=src_ids.size(1), klen=src_ids.size(1), device=src_ids.device)

        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_kpm, rel_pos_bias=rel)
        return x

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        """
        src_ids: (B,S)
        tgt_ids: (B,T) includes <bos> ... <eos> <pad> ...
        Predict next tokens for positions 1..T-1
        Returns logits: (B, T-1, V)
        """
        B, S = src_ids.shape
        B2, T = tgt_ids.shape
        assert B == B2

        memory = self.encode(src_ids)  # (B,S,D)

        # decoder input: y_{<bos> ... y_{T-2}}, i.e., tgt_ids[:, :-1]
        dec_in = tgt_ids[:, :-1]  # (B, T-1)
        x = self.tgt_emb(dec_in) * math.sqrt(self.d_model)
        if self.abs_pos is not None:
            x = self.abs_pos(x)
        x = self.emb_dropout(x)

        tgt_kpm = self.make_key_padding_mask(dec_in, self.tgt_pad_id)  # (B, T-1)
        src_kpm = self.make_key_padding_mask(src_ids, self.src_pad_id) # (B, S)
        causal = self.make_causal_mask(dec_in.size(1), device=src_ids.device)  # (1,1,T-1,T-1)

        rel = None
        if self.rel_bias_dec is not None:
            rel = self.rel_bias_dec(qlen=dec_in.size(1), klen=dec_in.size(1), device=src_ids.device)

        for layer in self.decoder_layers:
            x = layer(
                x, memory,
                tgt_attn_mask=causal,
                tgt_key_padding_mask=tgt_kpm,
                src_key_padding_mask=src_kpm,
                rel_pos_bias=rel
            )

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T-1, V)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src_ids: torch.Tensor, max_len: int = 120) -> torch.Tensor:
        """
        Batched greedy decoding.
        src_ids: (B,S)
        returns: (B, L) token ids (excluding BOS), stops at EOS per sample.
        """
        self.eval()
        device = src_ids.device
        B = src_ids.size(0)
        memory = self.encode(src_ids)

        src_kpm = self.make_key_padding_mask(src_ids, self.src_pad_id)

        ys = torch.full((B, 1), self.tgt_bos_id, dtype=torch.long, device=device)  # (B,1)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            dec_in = ys  # includes BOS and generated tokens
            x = self.tgt_emb(dec_in) * math.sqrt(self.d_model)
            if self.abs_pos is not None:
                x = self.abs_pos(x)
            x = self.emb_dropout(x)

            tgt_kpm = self.make_key_padding_mask(dec_in, self.tgt_pad_id)
            causal = self.make_causal_mask(dec_in.size(1), device=device)

            rel = None
            if self.rel_bias_dec is not None:
                rel = self.rel_bias_dec(qlen=dec_in.size(1), klen=dec_in.size(1), device=device)

            h = x
            for layer in self.decoder_layers:
                h = layer(
                    h, memory,
                    tgt_attn_mask=causal,
                    tgt_key_padding_mask=tgt_kpm,
                    src_key_padding_mask=src_kpm,
                    rel_pos_bias=rel
                )
            h = self.final_norm(h)
            logits = self.lm_head(h[:, -1:, :])  # (B,1,V) last step
            next_tok = torch.argmax(logits.squeeze(1), dim=-1)  # (B,)

            # once finished, keep EOS
            next_tok = torch.where(finished, torch.full_like(next_tok, self.tgt_eos_id), next_tok)
            finished = finished | next_tok.eq(self.tgt_eos_id)

            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            if finished.all():
                break

        # remove BOS
        return ys[:, 1:]

    @torch.no_grad()
    def beam_search_decode(
        self,
        src_ids: torch.Tensor,
        beam_size: int = 5,
        max_len: int = 120,
        length_norm_alpha: float = 0.6,
    ) -> torch.Tensor:
        """
        Beam search, simplified: supports batch_size=1 (like your RNN script).
        Returns (1, L) excluding BOS.
        """
        self.eval()
        if src_ids.size(0) != 1:
            raise ValueError("beam_search_decode supports batch_size=1. Use greedy for batching.")

        device = src_ids.device
        memory = self.encode(src_ids)  # (1,S,D)
        src_kpm = self.make_key_padding_mask(src_ids, self.src_pad_id)

        # expand memory for beams
        memory = memory.repeat(beam_size, 1, 1)
        src_kpm = src_kpm.repeat(beam_size, 1)

        beams = [[self.tgt_bos_id] for _ in range(beam_size)]
        scores = torch.full((beam_size,), float("-inf"), device=device)
        scores[0] = 0.0
        finished = torch.zeros((beam_size,), dtype=torch.bool, device=device)

        for step in range(max_len):
            # build decoder input tensor (K, t)
            max_t = max(len(b) for b in beams)
            ys = torch.full((beam_size, max_t), self.tgt_pad_id, dtype=torch.long, device=device)
            for i, b in enumerate(beams):
                ys[i, :len(b)] = torch.tensor(b, dtype=torch.long, device=device)

            x = self.tgt_emb(ys) * math.sqrt(self.d_model)
            if self.abs_pos is not None:
                x = self.abs_pos(x)
            x = self.emb_dropout(x)

            tgt_kpm = self.make_key_padding_mask(ys, self.tgt_pad_id)
            causal = self.make_causal_mask(ys.size(1), device=device)

            rel = None
            if self.rel_bias_dec is not None:
                rel = self.rel_bias_dec(qlen=ys.size(1), klen=ys.size(1), device=device)

            h = x
            for layer in self.decoder_layers:
                h = layer(
                    h, memory,
                    tgt_attn_mask=causal,
                    tgt_key_padding_mask=tgt_kpm,
                    src_key_padding_mask=src_kpm,
                    rel_pos_bias=rel
                )
            h = self.final_norm(h)
            logits = self.lm_head(h[:, -1, :])  # (K,V)
            logp = torch.log_softmax(logits, dim=-1)

            # if finished, force EOS only
            if finished.any():
                logp = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(logp, float("-inf")),
                    logp
                )
                logp[finished, self.tgt_eos_id] = 0.0

            total = scores.unsqueeze(1) + logp  # (K,V)
            flat = total.view(-1)
            topk_scores, topk_idx = torch.topk(flat, k=beam_size)

            beam_idx = topk_idx // logp.size(1)
            tok_idx = topk_idx % logp.size(1)

            new_beams = []
            for i in range(beam_size):
                b = beam_idx[i].item()
                t = tok_idx[i].item()
                new_beams.append(beams[b] + [t])
            beams = new_beams
            scores = topk_scores
            finished = finished | tok_idx.eq(self.tgt_eos_id)
            if finished.all():
                break

        def length_norm(score, length, alpha):
            lp = ((5 + length) / 6) ** alpha
            return score / lp

        best_i = 0
        best_val = float("-inf")
        for i in range(beam_size):
            length = len(beams[i]) - 1
            val = length_norm(scores[i].item(), max(length, 1), length_norm_alpha)
            if val > best_val:
                best_val = val
                best_i = i

        out = beams[best_i][1:]  # remove BOS
        return torch.tensor(out, dtype=torch.long, device=device).unsqueeze(0)


# -------------------------
# Loss / BLEU
# -------------------------
# def compute_loss(logits: torch.Tensor, tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
#     """
#     logits: (B, T-1, V) for predicting tgt[:, 1:]
#     tgt:    (B, T)
#     """
#     B, Tp, V = logits.shape
#     gold = tgt[:, 1:Tp+1].contiguous()
#     loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
#     return loss_fn(logits.view(-1, V), gold.view(-1))

def compute_loss(
    logits: torch.Tensor,
    tgt: torch.Tensor,
    pad_id: int,
    loss_fn: Optional[nn.Module] = None
) -> torch.Tensor:
    """
    logits: (B, T-1, V) predicting tgt[:, 1:]
    tgt:    (B, T)
    """
    B, Tp, V = logits.shape
    gold = tgt[:, 1:Tp+1].contiguous()   # (B, T-1)

    logits_2d = logits.view(-1, V)
    gold_1d = gold.view(-1)

    if loss_fn is not None:
        return loss_fn(logits_2d, gold_1d)

    # fallback to CE (no smoothing)
    ce = nn.CrossEntropyLoss(ignore_index=pad_id)
    return ce(logits_2d, gold_1d)



@torch.no_grad()
def decode_ids_to_text(ids: List[int], itos: List[str], eos_id: int, pad_id: int) -> str:
    toks = []
    for i in ids:
        if i == eos_id:
            break
        if i == pad_id:
            continue
        toks.append(itos[i] if i < len(itos) else "<unk>")
    return " ".join(toks)


@torch.no_grad()
def evaluate_bleu(
    model: TransformerNMT,
    loader: DataLoader,
    spm_model_path: str,
    zh_pad_id: int,
    en_pad_id: int,
    en_itos: List[str],
    decode: str,
    beam_size: int,
    max_len: int,
    device: torch.device
) -> float:
    model.eval()
    hyps, refs = [], []

    sp = None
    if spm_model_path:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(spm_model_path)

    def maybe_detok_sp(text: str) -> str:
        if sp is None:
            return text
        pieces = text.split()
        return sp.decode(pieces)

    for src, tgt, _, _ in tqdm(loader, desc="eval", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)

        if decode == "greedy":
            pred = model.greedy_decode(src, max_len=max_len)  # (B,L)
            for b in range(pred.size(0)):
                hyp_tok = decode_ids_to_text(pred[b].tolist(), en_itos, model.tgt_eos_id, en_pad_id)
                ref_tok = decode_ids_to_text(tgt[b, 1:].tolist(), en_itos, model.tgt_eos_id, en_pad_id)
                hyps.append(maybe_detok_sp(hyp_tok))
                refs.append(maybe_detok_sp(ref_tok))
        else:
            # beam supports batch_size=1; we loop over samples
            for b in range(src.size(0)):
                pred1 = model.beam_search_decode(src[b:b+1], beam_size=beam_size, max_len=max_len)
                hyp_tok = decode_ids_to_text(pred1[0].tolist(), en_itos, model.tgt_eos_id, en_pad_id)
                ref_tok = decode_ids_to_text(tgt[b, 1:].tolist(), en_itos, model.tgt_eos_id, en_pad_id)
                hyps.append(maybe_detok_sp(hyp_tok))
                refs.append(maybe_detok_sp(ref_tok))

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    return float(bleu)

def train_one_epoch(
    model: TransformerNMT,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[LinearWarmupInverseSqrtScheduler],
    loss_fn: Optional[nn.Module],
    en_pad_id: int,
    grad_clip: float,
    device: torch.device
):
    model.train()
    total_loss, total_tokens = 0.0, 0

    for src, tgt, _, _ in tqdm(loader, desc="train", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(src, tgt)
        # loss = compute_loss(logits, tgt, pad_id=en_pad_id)
        loss = compute_loss(logits, tgt, pad_id=en_pad_id, loss_fn=loss_fn)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        gold = tgt[:, 1:1+logits.size(1)]
        tok = gold.ne(en_pad_id).sum().item()
        total_loss += loss.item() * max(tok, 1)
        total_tokens += max(tok, 1)

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


# -------------------------
# Args / Main
# -------------------------
def model_scale_preset(scale: str) -> Dict[str, int]:
    scale = scale.lower()
    # You can tweak these to match your class baseline; these are reasonable "coursework" presets.
    if scale == "tiny":
        return {"d_model": 256, "n_heads": 4, "d_ff": 1024, "num_layers": 2}
    if scale == "small":
        return {"d_model": 384, "n_heads": 6, "d_ff": 1536, "num_layers": 4}
    if scale == "base":
        return {"d_model": 512, "n_heads": 8, "d_ff": 2048, "num_layers": 6}
    if scale == "large":
        return {"d_model": 768, "n_heads": 12, "d_ff": 3072, "num_layers": 8}
    raise ValueError(f"Unknown model_scale: {scale}")


def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_dir", type=str, default="preprocessed_data")
    p.add_argument("--use_test_for_final", action="store_true")
    p.add_argument("--spm_model", type=str, default="",
                   help="Path to SentencePiece .model for English detokenization during BLEU eval.")

    # ablations
    p.add_argument("--pos_enc", type=str, default="absolute", choices=["absolute", "relative"],
                   help="absolute=sinusoidal abs pos enc; relative=T5-style relative position bias (self-attn)")
    p.add_argument("--norm", type=str, default="layernorm", choices=["layernorm", "rmsnorm"])

    # model scale
    p.add_argument("--model_scale", type=str, default="small", choices=["tiny", "small", "base", "large"])
    p.add_argument("--d_model", type=int, default=0, help="Override preset if >0")
    p.add_argument("--n_heads", type=int, default=0, help="Override preset if >0")
    p.add_argument("--d_ff", type=int, default=0, help="Override preset if >0")
    p.add_argument("--num_layers", type=int, default=0, help="Override preset if >0")
    p.add_argument("--dropout", type=float, default=0.1)

    # relative pos bias params
    p.add_argument("--rel_num_buckets", type=int, default=32)
    p.add_argument("--rel_max_distance", type=int, default=128)

    # optimization
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # decoding (optional but useful for your Comparison section)
    p.add_argument("--decode", type=str, default="greedy", choices=["greedy", "beam"])
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--max_decode_len", type=int, default=120)

    # logging / saving
    p.add_argument("--save_dir", type=str, default="runs_transformer_nmt")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=4000)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    os.makedirs(args.save_dir, exist_ok=True)
    run_name = args.run_name
    if not run_name:
        run_name = f"tf_pos={args.pos_enc}_norm={args.norm}_scale={args.model_scale}_bs={args.batch_size}_lr={args.lr}_dec={args.decode}"
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Load artifacts
    vocab_zh = load_pickle(os.path.join(args.data_dir, "vocab_zh.pkl"))
    vocab_en = load_pickle(os.path.join(args.data_dir, "vocab_en.pkl"))
    train_ids = load_pickle(os.path.join(args.data_dir, "train_ids.pkl"))
    valid_ids = load_pickle(os.path.join(args.data_dir, "valid_ids.pkl"))
    test_ids  = load_pickle(os.path.join(args.data_dir, "test_ids.pkl"))
    cfg = None
    cfg_path = os.path.join(args.data_dir, "config.pkl")
    if os.path.exists(cfg_path):
        cfg = load_pickle(cfg_path)

    zh_stoi, zh_itos = vocab_zh["stoi"], vocab_zh["itos"]
    en_stoi, en_itos = vocab_en["stoi"], vocab_en["itos"]

    zh_pad_id = zh_stoi["<pad>"]
    en_pad_id = en_stoi["<pad>"]
    en_bos_id = en_stoi["<bos>"]
    en_eos_id = en_stoi["<eos>"]

    # DataLoader
    train_ds = NMTIdsDataset(train_ids)
    valid_ds = NMTIdsDataset(valid_ids)
    test_ds  = NMTIdsDataset(test_ids)

    collate = collate_fn_factory(zh_pad_id, en_pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=1 if args.decode == "beam" else args.batch_size,
                              shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds, batch_size=1 if args.decode == "beam" else args.batch_size,
                              shuffle=False, collate_fn=collate)

    # Model config from scale + overrides
    preset = model_scale_preset(args.model_scale)
    d_model = args.d_model if args.d_model > 0 else preset["d_model"]
    n_heads = args.n_heads if args.n_heads > 0 else preset["n_heads"]
    d_ff = args.d_ff if args.d_ff > 0 else preset["d_ff"]
    num_layers = args.num_layers if args.num_layers > 0 else preset["num_layers"]

    model = TransformerNMT(
        src_vocab=len(zh_itos),
        tgt_vocab=len(en_itos),
        src_pad_id=zh_pad_id,
        tgt_pad_id=en_pad_id,
        tgt_bos_id=en_bos_id,
        tgt_eos_id=en_eos_id,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=args.dropout,
        pos_enc=args.pos_enc,
        norm_type=args.norm,
        max_len=2048,
        rel_num_buckets=args.rel_num_buckets,
        rel_max_distance=args.rel_max_distance,
        tie_embeddings=True
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-9)
    # label smoothing loss
    loss_fn = None
    if args.label_smoothing > 0:
        loss_fn = LabelSmoothingLoss(
            vocab_size=len(en_itos),
            pad_id=en_pad_id,
            smoothing=args.label_smoothing
        ).to(device)

    # warmup + inverse sqrt scheduler (step-level)
    scheduler = LinearWarmupInverseSqrtScheduler(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        peak_lr=args.lr
    )


    # Save run config
    meta = {
        "args": vars(args),
        "preprocess_cfg": asdict(cfg) if cfg is not None else None,
        "vocab_sizes": {"zh": len(zh_itos), "en": len(en_itos)},
        "special_ids": {"zh_pad": zh_pad_id, "en_pad": en_pad_id, "en_bos": en_bos_id, "en_eos": en_eos_id},
        "model_cfg": {"d_model": d_model, "n_heads": n_heads, "d_ff": d_ff, "num_layers": num_layers},
    }
    save_json(meta, os.path.join(run_dir, "run_meta.json"))

    best_bleu = -1.0
    best_path = os.path.join(run_dir, "best.pt")
    log_path = os.path.join(run_dir, "log.jsonl")

    print(f"[Run] {run_name}")
    print(f"[Device] {device}")
    print(f"[Train] {len(train_ds)}  [Valid] {len(valid_ds)}  [Test] {len(test_ds)}")
    print(f"[Model] layers={num_layers} d_model={d_model} heads={n_heads} d_ff={d_ff} dropout={args.dropout}")
    print(f"[Ablation] pos_enc={args.pos_enc}  norm={args.norm}")
    print(f"[Opt] bs={args.batch_size} lr={args.lr} wd={args.weight_decay}")
    print(f"[Decode] {args.decode} (beam={args.beam_size}) max_len={args.max_decode_len}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # tr_loss, tr_ppl = train_one_epoch(
        #     model, train_loader, optimizer,
        #     en_pad_id=en_pad_id,
        #     grad_clip=args.grad_clip,
        #     device=device
        # )
        tr_loss, tr_ppl = train_one_epoch(
            model, train_loader, optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            en_pad_id=en_pad_id,
            grad_clip=args.grad_clip,
            device=device
        )


        val_bleu = evaluate_bleu(
            model, valid_loader,
            spm_model_path=args.spm_model,
            zh_pad_id=zh_pad_id,
            en_pad_id=en_pad_id,
            en_itos=en_itos,
            decode=args.decode,
            beam_size=args.beam_size,
            max_len=args.max_decode_len,
            device=device
        )

        dt = time.time() - t0
        record = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_ppl": tr_ppl,
            "valid_bleu": val_bleu,
            "seconds": dt
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Epoch {epoch:02d}/{args.epochs} | loss={tr_loss:.4f} ppl={tr_ppl:.2f} | valid BLEU={val_bleu:.2f} | {dt:.1f}s")

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "meta": meta,
                "epoch": epoch,
                "best_bleu": best_bleu
            }, best_path)
            print(f"  ✓ saved best to {best_path} (BLEU={best_bleu:.2f})")

    # Final test evaluation with best checkpoint
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    if args.use_test_for_final:
        test_bleu = evaluate_bleu(
            model, test_loader,
            spm_model_path=args.spm_model,
            zh_pad_id=zh_pad_id,
            en_pad_id=en_pad_id,
            en_itos=en_itos,
            decode=args.decode,
            beam_size=args.beam_size,
            max_len=args.max_decode_len,
            device=device
        )
        print(f"[Final] Test BLEU = {test_bleu:.2f}")
        save_json({"test_bleu": test_bleu, "best_valid_bleu": best_bleu}, os.path.join(run_dir, "final_metrics.json"))

    print(f"Done. Artifacts in: {run_dir}")


if __name__ == "__main__":
    main()
