#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN-based Chinese→English NMT training script (PyTorch)

Meets assignment requirements:
1) GRU or LSTM, encoder/decoder are 2-layer unidirectional RNNs
2) Attention with alignment functions: dot-product, multiplicative (general), additive (Bahdanau)
3) Training policy: Teacher Forcing vs Free Running via --teacher_forcing_ratio
4) Decoding policy: greedy vs beam search via --decode and --beam_size

Assumes you already ran preprocessing and saved:
  preprocessed_data/
    config.pkl
    vocab_zh.pkl
    vocab_en.pkl
    train_ids.pkl
    valid_ids.pkl
    test_ids.pkl

Install:
  pip install torch tqdm sacrebleu

Run example:
  python train_rnn_nmt.py \
    --data_dir preprocessed_data \
    --rnn_type gru \
    --attn additive \
    --teacher_forcing_ratio 1.0 \
    --decode greedy \
    --epochs 10 \
    --batch_size 64

Compare experiments by changing args:
  --rnn_type {gru,lstm}
  --attn {dot,multiplicative,additive}
  --teacher_forcing_ratio 1.0 vs 0.0 (free running)
  --decode greedy vs beam --beam_size 5
"""

import os
import math
import time
import json
import pickle
import random
import argparse
from dataclasses import asdict
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import sacrebleu
from dataclasses import dataclass
from typing import Optional

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

# -------------------------
# Utils
# -------------------------
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
    """
    Each item: (src_ids: List[int], tgt_ids: List[int])
    """
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
        # batch: list[(src_ids, tgt_ids)]
        src_lens = [len(s) for s, _ in batch]
        tgt_lens = [len(t) for _, t in batch]
        src_max = max(src_lens)
        tgt_max = max(tgt_lens)

        src = [pad_1d(s, src_pad_id, src_max) for s, _ in batch]
        tgt = [pad_1d(t, tgt_pad_id, tgt_max) for _, t in batch]

        src = torch.tensor(src, dtype=torch.long)  # (B, S)
        tgt = torch.tensor(tgt, dtype=torch.long)  # (B, T)
        src_lens = torch.tensor(src_lens, dtype=torch.long)
        tgt_lens = torch.tensor(tgt_lens, dtype=torch.long)
        return src, tgt, src_lens, tgt_lens

    return collate


# -------------------------
# Attention modules
# -------------------------
class Attention(nn.Module):
    """
    Alignment functions:
      - dot:        score = s_t · h_i              (requires dec_dim == enc_dim)
      - multiplicative (general): score = s_t^T W h_i
      - additive:   score = v^T tanh(Ws s_t + Wh h_i)   (Bahdanau)
    """
    def __init__(self, attn_type: str, enc_dim: int, dec_dim: int):
        super().__init__()
        attn_type = attn_type.lower()
        assert attn_type in {"dot", "multiplicative", "additive"}
        self.attn_type = attn_type
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim

        if attn_type == "dot":
            if enc_dim != dec_dim:
                raise ValueError("dot attention requires enc_dim == dec_dim")
        elif attn_type == "multiplicative":
            self.W = nn.Linear(enc_dim, dec_dim, bias=False)  # W h_i -> dec_dim
        else:  # additive
            self.Ws = nn.Linear(dec_dim, dec_dim, bias=True)
            self.Wh = nn.Linear(enc_dim, dec_dim, bias=False)
            self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, dec_state: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor):
        """
        dec_state: (B, dec_dim)      decoder current hidden (top layer)
        enc_out:   (B, S, enc_dim)   encoder outputs
        src_mask:  (B, S)            True for valid positions, False for PAD
        returns:
          context: (B, enc_dim)
          attn_w:  (B, S)
        """
        B, S, _ = enc_out.shape

        if self.attn_type == "dot":
            # score: (B, S) = enc_out · dec_state
            score = torch.bmm(enc_out, dec_state.unsqueeze(2)).squeeze(2)  # (B,S)
        elif self.attn_type == "multiplicative":
            # score = (W h_i) · s_t
            proj = self.W(enc_out)  # (B,S,dec_dim)
            score = torch.bmm(proj, dec_state.unsqueeze(2)).squeeze(2)  # (B,S)
        else:
            # additive
            # tanh(Ws s_t + Wh h_i)
            s_proj = self.Ws(dec_state).unsqueeze(1)  # (B,1,dec_dim)
            h_proj = self.Wh(enc_out)                 # (B,S,dec_dim)
            e = torch.tanh(s_proj + h_proj)           # (B,S,dec_dim)
            score = self.v(e).squeeze(2)              # (B,S)

        # mask pads -> -inf
        score = score.masked_fill(~src_mask, float("-inf"))
        attn_w = torch.softmax(score, dim=1)  # (B,S)
        context = torch.bmm(attn_w.unsqueeze(1), enc_out).squeeze(1)  # (B,enc_dim)
        return context, attn_w


# -------------------------
# Encoder / Decoder
# -------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, rnn_type: str, num_layers: int,
                 pad_id: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn_type = rnn_type.lower()
        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM # GRU和LSTM由nn实现
        # dropout applies between layers when num_layers > 1
        self.rnn = rnn_cls(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, src_ids: torch.Tensor, src_lens: torch.Tensor):
        """
        src_ids: (B,S)
        src_lens: (B,)
        returns:
          enc_out: (B,S,H)
          enc_final: hidden (and cell for LSTM)
        """
        emb = self.embedding(src_ids)  # (B,S,E)

        # pack for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, enc_final = self.rnn(packed) # packed_out: (B,S,H) 是 RNN 输出的 hidden state; enc_final: (L,B,H) or (2L,B,H)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B,S,H)
        # enc_out 是 packed_out 经过 pad_packed_sequence 之后还原成 (B, T, H) 的张量，
        # 其中 <pad> 对应的位置的 hidden state 是“无效填充”，通常为全 0（或等价的占位值），并且这些位置会在 attention 中被 mask 掉，不参与任何语义计算。
        return enc_out, enc_final


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, enc_dim: int, hidden_dim: int, rnn_type: str, num_layers: int,
                 pad_id: int, dropout: float, attn_type: str):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn_type = rnn_type.lower()
        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM

        # Decoder RNN input will be [emb(y_{t-1}); context]
        self.rnn = rnn_cls(
            input_size=emb_dim + enc_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = Attention(attn_type=attn_type, enc_dim=enc_dim, dec_dim=hidden_dim)

        # output projection: use [dec_hidden; context] -> vocab
        self.out = nn.Linear(hidden_dim + enc_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, prev_y: torch.Tensor, dec_state, enc_out: torch.Tensor, src_mask: torch.Tensor):
        """
        One decoding step.
        prev_y:   (B,) token ids at t-1
        dec_state: hidden (and cell for LSTM), shapes:
            GRU:  h: (L,B,H)
            LSTM: (h,c) each (L,B,H)
        enc_out:  (B,S,enc_dim)
        src_mask: (B,S)
        returns:
          logits: (B,V)
          dec_state: updated
          attn_w: (B,S)
        """
        emb = self.dropout(self.embedding(prev_y))  # (B,E)

        # use top layer hidden to compute attention,下一时间步的计算需要“所有层的 state”，而 attention 只需要“最顶层的 state”
        if self.rnn_type == "gru":
            top_h = dec_state[-1]  # (B,H) 最顶层dec_state
        else:
            top_h = dec_state[0][-1]  # h from (h,c)

        context, attn_w = self.attn(top_h, enc_out, src_mask)  # (B,enc_dim)

        rnn_in = torch.cat([emb, context], dim=1).unsqueeze(1)  # (B,1,E+enc_dim)
        rnn_out, dec_state = self.rnn(rnn_in, dec_state)        # rnn_out: (B,1,H)
        rnn_out = rnn_out.squeeze(1)                            # (B,H)

        logits = self.out(torch.cat([rnn_out, context], dim=1))  # (B,V)
        return logits, dec_state, attn_w


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, tgt_bos_id: int, tgt_eos_id: int, tgt_pad_id: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_bos_id = tgt_bos_id
        self.tgt_eos_id = tgt_eos_id
        self.tgt_pad_id = tgt_pad_id

    def make_src_mask(self, src_ids: torch.Tensor, src_pad_id: int):
        # True for real tokens
        return src_ids.ne(src_pad_id)

    def forward(self, src_ids: torch.Tensor, src_lens: torch.Tensor, tgt_ids: torch.Tensor,
                teacher_forcing_ratio: float, src_pad_id: int):
        """
        Training forward with optional teacher forcing.
        tgt_ids: (B,T) contains <bos> ... <eos> <pad> ...
        We predict y_t for t=1..T-1 (next token), ignoring pads.
        """
        device = src_ids.device
        B, T = tgt_ids.shape

        enc_out, enc_final = self.encoder(src_ids, src_lens)
        src_mask = self.make_src_mask(src_ids, src_pad_id)

        # init decoder state from encoder final
        dec_state = enc_final

        # first input to decoder is BOS
        prev_y = tgt_ids[:, 0]  # should be <bos>
        logits_all = []

        for t in range(1, T):
            logits, dec_state, _ = self.decoder.forward_step(prev_y, dec_state, enc_out, src_mask)
            logits_all.append(logits.unsqueeze(1))  # (B,1,V)

            use_tf = random.random() < teacher_forcing_ratio
            if use_tf:
                prev_y = tgt_ids[:, t]  # teacher forcing
            else:
                prev_y = torch.argmax(logits, dim=-1)  # free running

        logits_all = torch.cat(logits_all, dim=1)  # (B,T-1,V) 因为少了<bos>
        return logits_all

    @torch.no_grad()
    def translate_greedy(self, src_ids: torch.Tensor, src_lens: torch.Tensor, src_pad_id: int,
                         max_len: int = 80):
        self.eval()
        device = src_ids.device
        B = src_ids.size(0)

        enc_out, enc_final = self.encoder(src_ids, src_lens)
        src_mask = self.make_src_mask(src_ids, src_pad_id)

        dec_state = enc_final
        prev_y = torch.full((B,), self.tgt_bos_id, dtype=torch.long, device=device)

        outputs = []
        finished = torch.zeros((B,), dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits, dec_state, _ = self.decoder.forward_step(prev_y, dec_state, enc_out, src_mask)
            next_y = torch.argmax(logits, dim=-1)  # (B,)
            outputs.append(next_y.unsqueeze(1))

            # once eos generated, keep eos
            next_y = torch.where(finished, torch.full_like(next_y, self.tgt_eos_id), next_y)
            finished = finished | next_y.eq(self.tgt_eos_id)

            prev_y = next_y
            if finished.all():
                break

        if outputs:
            out = torch.cat(outputs, dim=1)  # (B,<=max_len)
        else:
            out = torch.empty((B, 0), dtype=torch.long, device=device)
        return out

    @torch.no_grad()
    def translate_beam(self, src_ids: torch.Tensor, src_lens: torch.Tensor, src_pad_id: int,
                       beam_size: int = 5, max_len: int = 80, length_norm_alpha: float = 0.6):
        """
        Beam search (batch size must be 1 for simplicity in coursework script).
        If you need true batched beam search, tell me and I’ll provide it.
        """
        self.eval()
        if src_ids.size(0) != 1:
            raise ValueError("translate_beam in this script supports batch_size=1. Use greedy for batching.")

        device = src_ids.device
        enc_out, enc_final = self.encoder(src_ids, src_lens)
        src_mask = self.make_src_mask(src_ids, src_pad_id)
        enc_out = enc_out.repeat(beam_size, 1, 1)     # (K,S,H)
        src_mask = src_mask.repeat(beam_size, 1)      # (K,S)

        # init beam
        beams = [[self.tgt_bos_id] for _ in range(beam_size)]
        scores = torch.full((beam_size,), float("-inf"), device=device)
        scores[0] = 0.0  # only first beam active initially

        # decoder states per beam
        dec_state = enc_final
        if isinstance(dec_state, tuple):
            h, c = dec_state
            h = h.repeat(1, beam_size, 1)  # (L,K,H)
            c = c.repeat(1, beam_size, 1)
            dec_state = (h, c)
        else:
            dec_state = dec_state.repeat(1, beam_size, 1)  # (L,K,H)

        prev_y = torch.tensor([self.tgt_bos_id] * beam_size, dtype=torch.long, device=device)
        finished = torch.zeros((beam_size,), dtype=torch.bool, device=device)

        for step in range(max_len):
            logits, dec_state, _ = self.decoder.forward_step(prev_y, dec_state, enc_out, src_mask)
            logp = torch.log_softmax(logits, dim=-1)  # (K,V)

            # if finished, force only eos
            if finished.any():
                logp = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(logp, float("-inf")),
                    logp
                )
                logp[finished, self.tgt_eos_id] = 0.0

            # expand
            total = scores.unsqueeze(1) + logp  # (K,V)
            flat = total.view(-1)               # (K*V,)
            topk_scores, topk_idx = torch.topk(flat, k=beam_size)

            beam_idx = topk_idx // logp.size(1)  # which old beam
            tok_idx = topk_idx % logp.size(1)    # which token

            # reorder beams
            new_beams = []
            for i in range(beam_size):
                b = beam_idx[i].item()
                t = tok_idx[i].item()
                new_beams.append(beams[b] + [t])
            beams = new_beams
            scores = topk_scores

            # reorder decoder states
            def index_state(st):
                if isinstance(st, tuple):
                    h, c = st
                    h = h.index_select(1, beam_idx)
                    c = c.index_select(1, beam_idx)
                    return (h, c)
                else:
                    return st.index_select(1, beam_idx)

            dec_state = index_state(dec_state)
            prev_y = tok_idx

            finished = finished | tok_idx.eq(self.tgt_eos_id)
            if finished.all():
                break

        # length normalization (common in beam search)
        def length_norm(score, length, alpha):
            lp = ((5 + length) / 6) ** alpha
            return score / lp

        best_i = 0
        best_val = float("-inf")
        for i in range(beam_size):
            length = len(beams[i]) - 1  # exclude BOS
            val = length_norm(scores[i].item(), max(length, 1), length_norm_alpha)
            if val > best_val:
                best_val = val
                best_i = i

        out = beams[best_i][1:]  # remove BOS
        return torch.tensor(out, dtype=torch.long, device=device).unsqueeze(0)  # (1,L)


# -------------------------
# Training + Evaluation
# -------------------------
def compute_loss(logits: torch.Tensor, tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    logits: (B,T-1,V)
    tgt:    (B,T) with BOS ... EOS PAD
    predict tgt[:, 1:] aligned with logits
    """
    B, Tp, V = logits.shape
    gold = tgt[:, 1:Tp+1].contiguous()  # (B,T-1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    return loss_fn(logits.view(-1, V), gold.view(-1))


@torch.no_grad()
def decode_ids_to_text(ids: List[int], itos: List[str], eos_id: int, pad_id: int) -> str:
    toks = []
    for i in ids:
        if i == eos_id:
            break
        if i == pad_id:
            continue
        toks.append(itos[i] if i < len(itos) else "<unk>")
    # For English tokens (NLTK word_tokenize), a simple join is OK for coursework baseline.
    # If you want prettier detokenization (punctuation spacing), tell me.
    return " ".join(toks)


@torch.no_grad()
def evaluate_bleu(model: Seq2Seq,
                  loader: DataLoader,
                  spm_model_path: str,
                  zh_pad_id: int,
                  en_pad_id: int,
                  en_itos: List[str],
                  decode: str,
                  beam_size: int,
                  max_len: int,
                  device: torch.device) -> float:
    model.eval()
    hyps = []
    refs = []

    # Optional: SentencePiece detokenization
    sp = None
    if spm_model_path:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(spm_model_path)

    def maybe_detok_sp(text: str) -> str:
        """
        If using SentencePiece, decode pieces back to natural text.
        Our decode_ids_to_text returns a whitespace-joined token string,
        so we split it into pieces and decode.
        """
        if sp is None:
            return text
        pieces = text.split()
        # SentencePiece can decode pieces directly
        return sp.decode(pieces)

    for src, tgt, src_lens, _ in tqdm(loader, desc="eval", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)
        src_lens = src_lens.to(device)

        if decode == "greedy":
            pred = model.translate_greedy(src, src_lens, zh_pad_id, max_len=max_len)  # (B,L)

            for b in range(pred.size(0)):
                hyp_tok = decode_ids_to_text(pred[b].tolist(), en_itos, model.tgt_eos_id, en_pad_id)
                ref_tok = decode_ids_to_text(tgt[b, 1:].tolist(), en_itos, model.tgt_eos_id, en_pad_id)  # skip BOS

                hyp = maybe_detok_sp(hyp_tok)
                ref = maybe_detok_sp(ref_tok)

                hyps.append(hyp)
                refs.append(ref)
        else:
            # beam in this script supports batch_size=1
            for b in range(src.size(0)):
                pred1 = model.translate_beam(
                    src[b:b+1], src_lens[b:b+1], zh_pad_id,
                    beam_size=beam_size, max_len=max_len
                )
                hyp_tok = decode_ids_to_text(pred1[0].tolist(), en_itos, model.tgt_eos_id, en_pad_id)
                ref_tok = decode_ids_to_text(tgt[b, 1:].tolist(), en_itos, model.tgt_eos_id, en_pad_id)

                hyp = maybe_detok_sp(hyp_tok)
                ref = maybe_detok_sp(ref_tok)

                hyps.append(hyp)
                refs.append(ref)

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    return float(bleu)



def train_one_epoch(model: Seq2Seq,
                    loader: DataLoader,
                    optimizer: optim.Optimizer,
                    zh_pad_id: int,
                    en_pad_id: int,
                    teacher_forcing_ratio: float,
                    grad_clip: float,
                    device: torch.device):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt, src_lens, _ in tqdm(loader, desc="train", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)
        src_lens = src_lens.to(device) # 一个 batch 中，每个源语言输入句子（中文）的真实 token 数（不包含 padding）

        optimizer.zero_grad(set_to_none=True)
        logits = model(src, src_lens, tgt, teacher_forcing_ratio=teacher_forcing_ratio, src_pad_id=zh_pad_id)
        loss = compute_loss(logits, tgt, pad_id=en_pad_id)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # token count for ppl (exclude pad)
        gold = tgt[:, 1:1+logits.size(1)]
        tok = gold.ne(en_pad_id).sum().item()

        total_loss += loss.item() * max(tok, 1)
        total_tokens += max(tok, 1)

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_dir", type=str, default="preprocessed_data",
                   help="Directory containing vocab_*.pkl and *_ids.pkl")
    p.add_argument("--use_test_for_final", action="store_true",
                   help="If set, compute BLEU on test set at the end (recommended).")
    p.add_argument("--spm_model", type=str, default="",
                    help="Path to SentencePiece .model for English detokenization during BLEU eval. "
                    "If provided, will decode pieces to text before BLEU.")


    # model
    p.add_argument("--rnn_type", type=str, default="gru", choices=["gru", "lstm"])
    p.add_argument("--attn", type=str, default="additive", choices=["dot", "multiplicative", "additive"])
    p.add_argument("--emb_dim", type=int, default=256) ###
    p.add_argument("--hidden_dim", type=int, default=512) ###
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)

    # training policy
    p.add_argument("--teacher_forcing_ratio", type=float, default=1.0,
                   help="1.0=Teacher Forcing; 0.0=Free Running; (0..1)=mixed")

    # optimization
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # decoding policy
    p.add_argument("--decode", type=str, default="greedy", choices=["greedy", "beam"])
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--max_decode_len", type=int, default=80)

    # logging / saving
    p.add_argument("--save_dir", type=str, default="runs_rnn_nmt")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    os.makedirs(args.save_dir, exist_ok=True)
    run_name = args.run_name
    if not run_name:
        run_name = f"rnn={args.rnn_type}_attn={args.attn}_tf={args.teacher_forcing_ratio}_dec={args.decode}"
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

    collate = collate_fn_factory(zh_pad_id, en_pad_id) # 一个返回函数的函数，给定pad_id，返回一个collate函数，用于DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=1 if args.decode == "beam" else args.batch_size,
                              shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds, batch_size=1 if args.decode == "beam" else args.batch_size,
                              shuffle=False, collate_fn=collate)

    # Build model
    enc = Encoder(
        vocab_size=len(zh_itos),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        rnn_type=args.rnn_type,
        num_layers=args.num_layers,
        pad_id=zh_pad_id,
        dropout=args.dropout
    )
    dec = Decoder(
        vocab_size=len(en_itos),
        emb_dim=args.emb_dim,
        enc_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        rnn_type=args.rnn_type,
        num_layers=args.num_layers,
        pad_id=en_pad_id,
        dropout=args.dropout,
        attn_type=args.attn
    )
    model = Seq2Seq(enc, dec, tgt_bos_id=en_bos_id, tgt_eos_id=en_eos_id, tgt_pad_id=en_pad_id).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Save run config
    meta = {
        "args": vars(args),
        "preprocess_cfg": asdict(cfg) if cfg is not None else None,
        "vocab_sizes": {"zh": len(zh_itos), "en": len(en_itos)},
        "special_ids": {"zh_pad": zh_pad_id, "en_pad": en_pad_id, "en_bos": en_bos_id, "en_eos": en_eos_id}
    }
    save_json(meta, os.path.join(run_dir, "run_meta.json"))

    best_bleu = -1.0
    best_path = os.path.join(run_dir, "best.pt")
    log_path = os.path.join(run_dir, "log.jsonl")

    print(f"[Run] {run_name}")
    print(f"[Device] {device}")
    print(f"[Train] {len(train_ds)}  [Valid] {len(valid_ds)}  [Test] {len(test_ds)}")
    print(f"[Model] {args.rnn_type.upper()} layers={args.num_layers} hidden={args.hidden_dim} emb={args.emb_dim} attn={args.attn}")
    print(f"[Policy] teacher_forcing_ratio={args.teacher_forcing_ratio}  decode={args.decode} (beam={args.beam_size})")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_ppl = train_one_epoch(
            model, train_loader, optimizer,
            zh_pad_id=zh_pad_id, en_pad_id=en_pad_id,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            grad_clip=args.grad_clip,
            device=device
        )

        val_bleu = evaluate_bleu(
            model, valid_loader,
            zh_pad_id=zh_pad_id, en_pad_id=en_pad_id, en_itos=en_itos,
            decode=args.decode, beam_size=args.beam_size, max_len=args.max_decode_len,
            device=device,
            spm_model_path=args.spm_model
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
            zh_pad_id=zh_pad_id, en_pad_id=en_pad_id, en_itos=en_itos,spm_model_path=args.spm_model,
            decode=args.decode, beam_size=args.beam_size, max_len=args.max_decode_len,
            device=device
        )
        print(f"[Final] Test BLEU = {test_bleu:.2f}")
        save_json({"test_bleu": test_bleu, "best_valid_bleu": best_bleu}, os.path.join(run_dir, "final_metrics.json"))

    print(f"Done. Artifacts in: {run_dir}")


if __name__ == "__main__":
    main()
