#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional

import torch

# Reuse training definitions to minimize mismatch risk
# This also avoids pickle PreprocessConfig issues by ensuring the class exists in an importable module.
from train_rnn_nmt import (
    PreprocessConfig,          # optional, but keeps pickle-safe if you ever load config.pkl
    load_pickle,
    Encoder,
    Decoder,
    Seq2Seq,
    decode_ids_to_text,
)


def parse_args():
    p = argparse.ArgumentParser("RNN NMT prediction script (from train_rnn_nmt.py artifacts)")

    p.add_argument("--data_dir", type=str, default="preprocessed_data_hy",
                   help="Directory containing vocab_zh.pkl and vocab_en.pkl (same as training).")

    p.add_argument("--ckpt", type=str, default="runs_rnn_nmt/rnn=gru_attn=additive_tf=1.0_dec=beam/best.pt",
                   help="Path to checkpoint saved by train_rnn_nmt.py (e.g., runs_rnn_nmt/<run>/best.pt)")

    p.add_argument("--test_jsonl", type=str, default="data_hy/test_retranslated_hunyuan.jsonl",
                   help="Path to raw test.jsonl (each line: {zh,en,index}).")

    p.add_argument("--out_jsonl", type=str, default="runs_rnn_nmt/rnn=gru_attn=additive_tf=1.0_dec=beam/predictions.jsonl",
                   help="Output jsonl containing {index, zh, pred_en} per line.")
    p.add_argument("--out_txt", type=str, default="runs_rnn_nmt/rnn=gru_attn=additive_tf=1.0_dec=beam/predictions.txt",
                   help="Output plain text (one prediction per line).")
    p.add_argument("--spm_model", type=str, default="preprocessed_data_hy/spm_en.model",
                    help="Path to SentencePiece .model for English detokenization in prediction (e.g., preprocessed_data/spm_en.model)")
    p.add_argument("--warmup", type=int, default=10,
                    help="Warmup steps for timing (only affects speed report)")

    # decoding policy (match training)
    p.add_argument("--decode", type=str, default="beam", choices=["greedy", "beam"])
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--max_decode_len", type=int, default=60)

    # batching (greedy supports batch; beam in training script supports batch_size=1)
    p.add_argument("--batch_size", type=int, default=64)

    # tokenization for zh -> ids (must match your preprocessing as closely as possible)
    p.add_argument("--zh_tokenizer", type=str, default="jieba", choices=["jieba", "space", "char"],
                   help="Tokenizer used to convert raw zh text into tokens before vocab lookup.")

    # optional: detokenize punctuation for English output
    p.add_argument("--detok_en", action="store_true",
                   help="Apply a simple English detokenization (punctuation spacing).")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def zh_tokenize(text: str, mode: str) -> List[str]:
    text = text.strip()
    if mode == "space":
        return text.split()
    if mode == "char":
        # keep non-space chars
        return [ch for ch in text if not ch.isspace()]
    # default jieba
    import jieba
    return list(jieba.cut(text, cut_all=False))


def detokenize_en_simple(s: str) -> str:
    # Simple punctuation fixes for common tokenized outputs: "word ." -> "word."
    # Good enough for coursework display; BLEU eval can use Moses detok if desired.
    s = s.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    s = s.replace(" :", ":").replace(" ;", ";")
    s = s.replace(" n't", "n't").replace(" 's", "'s").replace(" 're", "'re").replace(" 've", "'ve").replace(" 'll", "'ll")
    s = s.replace(" ( ", " (").replace(" )", ")")
    s = s.replace(" \" ", "\"")
    return s.strip()
    
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_spm(spm_model_path: str):
    if not spm_model_path:
        return None
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model_path)
    return sp

def detok_en_with_spm(sp, token_text: str) -> str:
    """
    token_text: output of decode_ids_to_text -> whitespace-joined tokens/pieces
    """
    if sp is None:
        return token_text
    pieces = token_text.split()
    return sp.decode(pieces).strip()


def pad_batch(seqs: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    seqs: list of variable-length id lists
    returns:
      src_ids: (B, Smax)
      src_lens: (B,)
    """
    lens = torch.tensor([len(x) for x in seqs], dtype=torch.long)
    Smax = int(lens.max().item()) if len(seqs) else 0
    padded = []
    for x in seqs:
        if len(x) < Smax:
            padded.append(x + [pad_id] * (Smax - len(x)))
        else:
            padded.append(x[:Smax])
    src_ids = torch.tensor(padded, dtype=torch.long)
    return src_ids, lens


def build_model_from_ckpt(ckpt: Dict[str, Any], vocab_zh: Dict[str, Any], vocab_en: Dict[str, Any], device: torch.device) -> Seq2Seq:
    # args used in training are stored in checkpoint under "args"
    train_args = ckpt["args"]

    zh_stoi, zh_itos = vocab_zh["stoi"], vocab_zh["itos"]
    en_stoi, en_itos = vocab_en["stoi"], vocab_en["itos"]

    zh_pad_id = zh_stoi["<pad>"]
    en_pad_id = en_stoi["<pad>"]
    en_bos_id = en_stoi["<bos>"]
    en_eos_id = en_stoi["<eos>"]

    enc = Encoder(
        vocab_size=len(zh_itos),
        emb_dim=int(train_args["emb_dim"]),
        hidden_dim=int(train_args["hidden_dim"]),
        rnn_type=str(train_args["rnn_type"]),
        num_layers=int(train_args["num_layers"]),
        pad_id=zh_pad_id,
        dropout=float(train_args["dropout"]),
    )
    dec = Decoder(
        vocab_size=len(en_itos),
        emb_dim=int(train_args["emb_dim"]),
        enc_dim=int(train_args["hidden_dim"]),
        hidden_dim=int(train_args["hidden_dim"]),
        rnn_type=str(train_args["rnn_type"]),
        num_layers=int(train_args["num_layers"]),
        pad_id=en_pad_id,
        dropout=float(train_args["dropout"]),
        attn_type=str(train_args["attn"]),
    )

    model = Seq2Seq(decoder=dec, encoder=enc, tgt_bos_id=en_bos_id, tgt_eos_id=en_eos_id, tgt_pad_id=en_pad_id)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def translate_batch_greedy(
    model: Seq2Seq,
    batch_src_ids: List[List[int]],
    zh_pad_id: int,
    max_len: int,
    device: torch.device
) -> List[List[int]]:
    src, src_lens = pad_batch(batch_src_ids, zh_pad_id)
    src = src.to(device)
    src_lens = src_lens.to(device)
    pred = model.translate_greedy(src, src_lens, zh_pad_id, max_len=max_len)  # (B, L)
    return [pred[i].tolist() for i in range(pred.size(0))]


@torch.no_grad()
def translate_one_beam(
    model: Seq2Seq,
    src_ids: List[int],
    zh_pad_id: int,
    beam_size: int,
    max_len: int,
    device: torch.device
) -> List[int]:
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)      # (1,S)
    src_lens = torch.tensor([len(src_ids)], dtype=torch.long).to(device)       # (1,)
    pred = model.translate_beam(src, src_lens, zh_pad_id, beam_size=beam_size, max_len=max_len)  # (1,L)
    return pred[0].tolist()


def main():
    import time

    args = parse_args()
    device = torch.device(args.device)

    # ---- optional args (backward-compatible) ----
    spm_model_path = getattr(args, "spm_model", "")
    warmup_steps = int(getattr(args, "warmup", 10))

    def sync_if_cuda():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def count_parameters(m: torch.nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    # ---- optional: load SentencePiece for EN detokenization ----
    sp = None
    if spm_model_path:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(spm_model_path)
        print("[SPM] Loaded:", spm_model_path)

    def detok_en_with_spm(token_text: str) -> str:
        """
        token_text is produced by decode_ids_to_text (whitespace-joined tokens).
        If using SentencePiece, decode pieces into natural string.
        """
        if sp is None:
            return token_text
        pieces = token_text.split()
        return sp.decode(pieces).strip()

    # Load vocabs (same format as training)
    vocab_zh = load_pickle(os.path.join(args.data_dir, "vocab_zh.pkl"))
    vocab_en = load_pickle(os.path.join(args.data_dir, "vocab_en.pkl"))

    zh_stoi, zh_itos = vocab_zh["stoi"], vocab_zh["itos"]
    en_stoi, en_itos = vocab_en["stoi"], vocab_en["itos"]

    zh_pad_id = zh_stoi["<pad>"]
    en_pad_id = en_stoi["<pad>"]

    # Load checkpoint saved by train_rnn_nmt.py (best.pt)
    ckpt = torch.load(args.ckpt, map_location=device)

    # Build model exactly as training used (from ckpt["args"])
    model = build_model_from_ckpt(ckpt, vocab_zh, vocab_en, device)

    # Print model parameter count
    num_params = count_parameters(model)
    print(f"[Model] Trainable parameters: {num_params:,}")

    # Read raw test.jsonl
    samples: List[Dict[str, Any]] = []
    with open(args.test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    print(f"[Data] Loaded {len(samples)} samples from: {args.test_jsonl}")

    # ---- warmup for timing (important for CUDA) ----
    # does NOT affect outputs; just stabilizes timing and GPU clocks
    warmup_n = min(warmup_steps, len(samples))
    if warmup_n > 0:
        with torch.no_grad():
            for s in samples[:warmup_n]:
                toks = zh_tokenize(s["zh"], args.zh_tokenizer)
                src_ids = [zh_stoi.get(t, zh_stoi["<unk>"]) for t in toks]
                if args.decode == "greedy":
                    _ = translate_batch_greedy(model, [src_ids], zh_pad_id, args.max_decode_len, device)
                else:
                    _ = translate_one_beam(model, src_ids, zh_pad_id, args.beam_size, args.max_decode_len, device)

    # ---- start timing ----
    sync_if_cuda()
    t0 = time.perf_counter()

    # Predict
    out_rows: List[Dict[str, Any]] = []
    out_lines_txt: List[str] = []

    total_gen_tokens = 0  # count generated tokens (excluding pads)
    total_sents = 0

    if args.decode == "greedy":
        # Batch greedy decoding
        batch_src_ids: List[List[int]] = []
        batch_meta: List[Tuple[Optional[int], str]] = []  # (index, zh)

        for s in samples:
            idx = s.get("index", None)
            zh_text = s["zh"]

            toks = zh_tokenize(zh_text, args.zh_tokenizer)
            src_ids = [zh_stoi.get(t, zh_stoi["<unk>"]) for t in toks]

            batch_src_ids.append(src_ids)
            batch_meta.append((idx, zh_text))

            if len(batch_src_ids) >= args.batch_size:
                preds = translate_batch_greedy(model, batch_src_ids, zh_pad_id, args.max_decode_len, device)
                for (i, zh_), pred_ids in zip(batch_meta, preds):
                    # token ids -> token string
                    hyp_tok = decode_ids_to_text(pred_ids, en_itos, model.tgt_eos_id, en_pad_id)

                    # SentencePiece detok (if provided)
                    hyp = detok_en_with_spm(hyp_tok)

                    # optional extra punctuation detok (legacy)
                    if args.detok_en:
                        hyp = detokenize_en_simple(hyp)

                    out_rows.append({"index": i, "zh": zh_, "pred_en": hyp})
                    out_lines_txt.append(hyp)

                    # speed stats
                    total_gen_tokens += sum(1 for x in pred_ids if x != en_pad_id)
                    total_sents += 1

                batch_src_ids.clear()
                batch_meta.clear()

        # flush remainder
        if batch_src_ids:
            preds = translate_batch_greedy(model, batch_src_ids, zh_pad_id, args.max_decode_len, device)
            for (i, zh_), pred_ids in zip(batch_meta, preds):
                hyp_tok = decode_ids_to_text(pred_ids, en_itos, model.tgt_eos_id, en_pad_id)
                hyp = detok_en_with_spm(hyp_tok)
                if args.detok_en:
                    hyp = detokenize_en_simple(hyp)

                out_rows.append({"index": i, "zh": zh_, "pred_en": hyp})
                out_lines_txt.append(hyp)

                total_gen_tokens += sum(1 for x in pred_ids if x != en_pad_id)
                total_sents += 1

    else:
        # Beam decoding: training implementation supports batch_size=1, so loop
        for s in samples:
            idx = s.get("index", None)
            zh_text = s["zh"]

            toks = zh_tokenize(zh_text, args.zh_tokenizer)
            src_ids = [zh_stoi.get(t, zh_stoi["<unk>"]) for t in toks]

            pred_ids = translate_one_beam(
                model, src_ids, zh_pad_id,
                beam_size=args.beam_size,
                max_len=args.max_decode_len,
                device=device
            )

            hyp_tok = decode_ids_to_text(pred_ids, en_itos, model.tgt_eos_id, en_pad_id)
            hyp = detok_en_with_spm(hyp_tok)
            if args.detok_en:
                hyp = detokenize_en_simple(hyp)

            out_rows.append({"index": idx, "zh": zh_text, "pred_en": hyp})
            out_lines_txt.append(hyp)

            total_gen_tokens += sum(1 for x in pred_ids if x != en_pad_id)
            total_sents += 1

    # ---- end timing ----
    sync_if_cuda()
    t1 = time.perf_counter()
    dt = max(t1 - t0, 1e-9)

    sps = total_sents / dt
    tps = total_gen_tokens / dt
    print(f"[Speed] decode={args.decode}  sents={total_sents}  gen_tokens={total_gen_tokens}  time={dt:.3f}s  sent/s={sps:.2f}  tok/s={tps:.2f}")

    # Save outputs
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(args.out_txt, "w", encoding="utf-8") as f:
        for line in out_lines_txt:
            f.write(line + "\n")

    print(f"[OK] Wrote {len(out_rows)} lines")
    print(f"  jsonl: {args.out_jsonl}")
    print(f"  txt : {args.out_txt}")


if __name__ == "__main__":
    main()
