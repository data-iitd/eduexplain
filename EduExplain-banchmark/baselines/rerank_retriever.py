#!/usr/bin/env python3
"""Cross-encoder reranking of the dense pool. 

SCORE SCALE. The reranker emits a probability; the dense retriever emits a
cosine. Top-n and Modality-Capped read only the ORDER, so they do not care.
MMR does: it computes lambda*Rel - (1-lambda)*Sim with Rel taken raw from the
file, and a saturated probability (most candidates at ~1.0 or ~0.0) makes
lambda=0.8 mean something different from what it means in every other row --
the relevance term goes flat and the diversity term decides alone. So by
default the scores are mapped RANK-PRESERVINGLY onto the per-query range of
the dense scores they replace: the order is untouched, and lambda keeps its
meaning across the table. The raw probability is kept as `rerank_prob` in
every record. --score-scale prob writes the probability instead; use it only
if you intend to report that the MMR cells of this row are not comparable
with the others.

Usage
-----
  python rerank_retriever.py --gold $Q --resources $R \\
      --candidates pools/dense_top100.json \\
      --model /path/to/qwen3-reranker --output pools/rerank_top100.json \\
      --diagnose
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from retrieval_io import (index_text, is_figure, load_corpus, query_text,
                          require_local_model)

# The Qwen3-Reranker chat format, verbatim from its model card's usage.
QWEN_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    "only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
QWEN_SUFFIX = ("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
DEFAULT_INSTRUCT = ("Given a learner's question, judge whether the document "
                    "is part of a set of resources that together explain it")

PLAIN_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query. "
    "Answer only \"yes\" or \"no\".\n\n"
    "<Query>: {query}\n\n<Document>: {doc}\n\nAnswer:")


def doc_text(res: Dict, max_chars: int) -> str:
    t, _ = index_text(res)
    head = " / ".join(str(res.get(k)) for k in ("subject", "chapter_title")
                      if res.get(k) and str(res.get(k)) != "None")
    body = (head + ". " + t) if head else t
    return body[:max_chars]


def to_range(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Rank-preserving map of x onto [lo, hi]. Order is exactly preserved."""
    if x.size == 0:
        return x
    a, b = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.full_like(x, hi)
    if b - a < 1e-12:
        return np.full_like(x, (lo + hi) / 2.0)
    return lo + (x - a) * (hi - lo) / (b - a)


def load_reranker(path: str, dtype, force_vl: bool = False):
    """Load the checkpoint under whatever class its own config calls for.

    A VL checkpoint carries a Qwen3VLConfig, which AutoModelForCausalLM
    refuses with a 200-line list of the configs it does accept -- a message
    that says nothing about what to do. The config is read here first and the
    class chosen from it, so the checkpoint decides instead of a flag the
    caller has to know to pass.

    padding_side="left" is REQUIRED, not a preference: the score is read at
    the last position, and left padding is what puts every sequence's last
    real token there.
    """
    import json as _json

    cfg_path = os.path.join(path, "config.json")
    model_type, arches = "", []
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = _json.load(fh)
        model_type = str(cfg.get("model_type") or "")
        arches = [str(a) for a in (cfg.get("architectures") or [])]
    is_vl = force_vl or "vl" in model_type.lower() or \
        any("ConditionalGeneration" in a or "ImageText" in a for a in arches)
    print(f"  config: model_type={model_type or '?'} "
          f"architectures={arches or '?'} -> "
          f"{'vision-language' if is_vl else 'text'} class")

    # Tokenizer: a VL checkpoint may expose it only through the processor.
    from transformers import AutoTokenizer
    tok = None
    try:
        tok = AutoTokenizer.from_pretrained(path, padding_side="left",
                                            trust_remote_code=True)
    except Exception as exc:
        print(f"  AutoTokenizer: {type(exc).__name__}; trying the processor")
    if tok is None:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        tok = getattr(proc, "tokenizer", proc)
    tok.padding_side = "left"

    # Model: try the classes that can hold this config, most specific first.
    import transformers as T
    names = (["AutoModelForImageTextToText", "Qwen3VLForConditionalGeneration",
              "AutoModelForVision2Seq", "AutoModelForCausalLM"] if is_vl else
             ["AutoModelForCausalLM", "AutoModelForImageTextToText"])
    if arches:
        names = [a for a in arches if hasattr(T, a)] + names

    errors = []
    for name in dict.fromkeys(names):
        cls = getattr(T, name, None)
        if cls is None:
            continue
        for kw in ({"dtype": dtype}, {"torch_dtype": dtype}):
            try:
                m = cls.from_pretrained(path, device_map="auto",
                                        trust_remote_code=True, **kw)
                print(f"  loaded with {name}")
                return tok, m
            except TypeError:
                continue                 # old/new transformers dtype kwarg
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: "
                              f"{str(exc).splitlines()[0][:150]}")
                break
    raise SystemExit(
        "could not load the reranker under any model class.\n  "
        + "\n  ".join(errors[:4])
        + "\n  If this is a Qwen3-VL checkpoint, the installed transformers "
          "must be new enough\n  to expose Qwen3VLForConditionalGeneration.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--resources", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True,
                    help="local directory of the reranker")
    ap.add_argument("--prompt-style", default="qwen3",
                    choices=["qwen3", "plain"],
                    help="qwen3 = the model card's chat template (correct for "
                         "a Qwen3 reranker). plain = a bare instruction, for "
                         "comparison only.")
    ap.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    ap.add_argument("--score-scale", default="dense-range",
                    choices=["dense-range", "prob"])
    ap.add_argument("--rerank-n", type=int, default=0,
                    help="rerank the first N of each pool; 0 = all")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-doc-chars", type=int, default=1200)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--vl", action="store_true",
                    help="force the vision-language model class. Normally "
                         "unnecessary: the class is chosen from the "
                         "checkpoint's own config.json.")
    ap.add_argument("--diagnose", action="store_true",
                    help="report the score distribution and how far the "
                         "ranking actually moved")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N queries only, for a smoke test")
    ns = ap.parse_args()

    ns.model = require_local_model(ns.model, "--model (reranker)")

    gold, resources = load_corpus(ns.gold, ns.resources)
    by_id = {str(r.get("resource_id")): r for r in resources}
    with open(ns.candidates, encoding="utf-8") as f:
        pools = json.load(f)
    if ns.limit:
        pools = pools[:ns.limit]
    print(f"{len(pools)} query pools from {ns.candidates}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {ns.model} on {device}")
    dtype = (torch.float16 if ns.dtype == "float16" else
             torch.bfloat16 if ns.dtype == "bfloat16" else "auto")
    tok, model = load_reranker(ns.model, dtype, force_vl=ns.vl)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if tok.padding_side != "left":
        raise SystemExit("could not set padding_side='left'; the last-token "
                         "read below assumes it.")

    def one_id(word: str) -> int:
        i = tok.convert_tokens_to_ids(word)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            return i
        for form in (word, " " + word, word.capitalize()):
            ids = tok.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        raise SystemExit(
            f"'{word}' is not a single token under {ns.model}; the yes/no "
            f"read needs one id each.")
    yes_id, no_id = one_id("yes"), one_id("no")
    print(f"prompt style: {ns.prompt_style}   yes={yes_id} no={no_id}   "
          f"padding={tok.padding_side}")

    pre_ids = tok.encode(QWEN_PREFIX, add_special_tokens=False)
    suf_ids = tok.encode(QWEN_SUFFIX, add_special_tokens=False)
    budget = ns.max_len - len(pre_ids) - len(suf_ids)
    if ns.prompt_style == "qwen3" and budget < 64:
        raise SystemExit(f"--max-len {ns.max_len} leaves {budget} tokens for "
                         f"the query and document; raise it.")

    def build(q: str, d: str) -> List[int]:
        if ns.prompt_style == "plain":
            return tok.encode(PLAIN_PROMPT.format(query=q, doc=d),
                              add_special_tokens=False)[-ns.max_len:]
        body = f"<Instruct>: {ns.instruct}\n<Query>: {q}\n<Document>: {d}"
        # The DOCUMENT is what gets truncated. Truncating the whole string
        # would cut the assistant suffix off the end, and then the last
        # position is not the answer slot at all.
        mid = tok.encode(body, add_special_tokens=False)[:budget]
        return pre_ids + mid + suf_ids

    @torch.no_grad()
    def score(batches: Sequence[List[int]]) -> np.ndarray:
        out = []
        for b in range(0, len(batches), ns.batch_size):
            chunk = batches[b:b + ns.batch_size]
            enc = tok.pad({"input_ids": chunk}, padding=True,
                          return_tensors="pt").to(model.device)
            logits = model(**enc).logits[:, -1, :]      # left-padded: -1 is
            two = torch.stack([logits[:, no_id],        # the last real token
                               logits[:, yes_id]], dim=-1)
            out.append(torch.softmax(two.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    qmap = {str(g["query_id"]): query_text(g) for g in gold}
    n_missing = 0
    probs_all: List[float] = []
    moved_top3, moved_rank = [], []

    for qi, entry in enumerate(pools):
        qid = str(entry["query_id"])
        q = qmap.get(qid, entry.get("query", ""))
        cands = entry.get("retrieved_resources") or []
        depth = len(cands) if ns.rerank_n <= 0 else min(ns.rerank_n, len(cands))
        head, tail = cands[:depth], cands[depth:]

        prompts = []
        for c in head:
            r = c.get("resource") or by_id.get(str(c.get("resource_id")))
            if r is None:
                n_missing += 1
                r = {"resource_id": c.get("resource_id")}
            prompts.append(build(q, doc_text(r, ns.max_doc_chars)))
        p = score(prompts)
        probs_all.extend(p.tolist())

        dense = np.array([float(c.get("score") or 0.0) for c in head])
        if ns.score_scale == "dense-range" and dense.size:
            s = to_range(p, float(dense.min()), float(dense.max()))
        else:
            s = p

        order = np.argsort(-p, kind="stable")           # ORDER from the model
        if ns.diagnose:
            before = [str(c.get("resource_id")) for c in head[:3]]
            after = [str(head[int(j)].get("resource_id")) for j in order[:3]]
            moved_top3.append(len(set(before) & set(after)))
            pos = {int(j): r for r, j in enumerate(order)}
            moved_rank.append(float(np.mean(
                [abs(pos[j] - j) for j in range(min(10, len(head)))])))

        new = []
        for rank, j in enumerate(order, start=1):
            c = dict(head[int(j)])
            c["prompt_index"] = rank
            c["dense_score"] = c.get("score")
            c["rerank_prob"] = float(p[int(j)])
            c["score"] = float(s[int(j)])
            new.append(c)
        for rank, c in enumerate(tail, start=len(new) + 1):
            c = dict(c)
            c["prompt_index"] = rank
            c["dense_score"] = c.get("score")
            c["rerank_prob"] = None
            # Below every reranked item on whatever scale is in use.
            c["score"] = float(min(s.min(), 0.0) - 1.0) if len(s) else -1.0
            new.append(c)
        entry["retrieved_resources"] = new
        entry["retriever"] = f"rerank:{os.path.basename(ns.model)}"
        if (qi + 1) % 10 == 0 or qi + 1 == len(pools):
            print(f"  [{qi + 1}/{len(pools)}]")

    if n_missing:
        print(f"  [WARN] {n_missing} candidates were not in --resources and "
              f"were scored on their id alone.")

    os.makedirs(os.path.dirname(ns.output) or ".", exist_ok=True)
    with open(ns.output, "w", encoding="utf-8") as f:
        json.dump(pools, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {ns.output}")

    if ns.diagnose and probs_all:
        a = np.array(probs_all)
        sat = float(np.mean((a > 0.95) | (a < 0.05)))
        print(f"\n--- diagnose ---")
        print(f"P(yes): mean {a.mean():.3f}  median {np.median(a):.3f}  "
              f"p5 {np.percentile(a, 5):.3f}  p95 {np.percentile(a, 95):.3f}")
        print(f"saturated (>0.95 or <0.05): {sat:.1%}")
        print(f"top-3 kept from the dense order: "
              f"{np.mean(moved_top3):.2f} / 3")
        print(f"mean |rank change| of the dense top-10: "
              f"{np.mean(moved_rank):.1f}")
        if a.std() < 1e-3:
            print("  [!] the scores are nearly constant. The model is not "
                  "discriminating --\n      check --prompt-style and that "
                  "yes/no are the right token ids.")
        if sat > 0.9 and ns.score_scale == "prob":
            print("  [!] almost every score is 0 or 1. Under --score-scale "
                  "prob the MMR rows\n      will be decided by the diversity "
                  "term alone; dense-range avoids that.")
        if np.mean(moved_top3) > 2.9:
            print("  [!] the top-3 is essentially unchanged; this row will "
                  "differ from the dense\n      row only in the tail.")

    print(f"Pool membership unchanged, so Recall@K must equal the input "
          f"pool's.\nScore scale: {ns.score_scale}"
          + ("  (order from the model, magnitudes mapped onto the dense "
             "range so MMR's\n  lambda means what it means in the other rows)"
             if ns.score_scale == "dense-range" else
             "  (raw probability; MMR's lambda is NOT comparable with the "
             "other rows)"))


if __name__ == "__main__":
    main()
