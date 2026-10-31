#!/usr/bin/env python3
"""Cut a candidate file down to the FIRST-STAGE top-K, for the K sweep.

Usage
-----
  python pool_at_k.py --candidates pools/rerank_top200.json \\
      --k 25,50,100,200 --out-dir pools/sweep --assume-pointwise
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--k", default="25,50,100,200")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default=None,
                    help="basename for the outputs; default is the input's")
    ap.add_argument("--first-stage-key", default="auto",
                    help="field holding the FIRST-STAGE score. 'auto' uses "
                         "dense_score when present, else score.")
    ap.add_argument("--assume-pointwise", action="store_true",
                    help="confirm the reranker scores each document "
                         "independently, so one pass over the largest K "
                         "serves every smaller K")
    ns = ap.parse_args()

    with open(os.path.expanduser(ns.candidates), encoding="utf-8") as f:
        pools = json.load(f)
    ks = sorted({int(x) for x in ns.k.split(",") if x.strip()})
    name = ns.name or os.path.splitext(os.path.basename(ns.candidates))[0]
    os.makedirs(os.path.expanduser(ns.out_dir), exist_ok=True)

    sample = (pools[0].get("retrieved_resources") or [{}])[0]
    key = ns.first_stage_key
    if key == "auto":
        key = "dense_score" if sample.get("dense_score") is not None else "score"
    reranked = any(c.get("dense_score") is not None
                   for e in pools for c in (e.get("retrieved_resources") or []))
    print(f"{len(pools)} pools from {os.path.basename(ns.candidates)}   "
          f"first-stage score: '{key}'   reranked: {reranked}")
    if reranked and not ns.assume_pointwise:
        print("  [!] This file was reranked. Restricting it to the "
              "first-stage top-K reproduces a\n      per-K rerank only if the "
              "reranker is POINTWISE. Pass --assume-pointwise once you\n"
              "      have confirmed that, or rerank each K separately.")

    sizes = [len(e.get("retrieved_resources") or []) for e in pools]
    print(f"pool sizes: min {min(sizes)}  max {max(sizes)}  "
          f"mean {np.mean(sizes):.1f}")
    for k in ks:
        short = sum(1 for s in sizes if s < k)
        out = []
        for e in pools:
            cands = e.get("retrieved_resources") or []
            # The first-stage top-K. Ties broken by the recorded order, so the
            # cut is reproducible rather than dependent on sort stability
            # across runs.
            idx = sorted(range(len(cands)),
                         key=lambda i: (-float(cands[i].get(key) or 0.0), i))[:k]
            keep = [cands[i] for i in idx]
            # Inside the pool the CURRENT order stands: on a reranked file
            # that is the reranker's order, which is what the selector reads.
            keep = sorted(keep, key=lambda c: -float(c.get("score") or 0.0))
            for r, c in enumerate(keep, start=1):
                c = dict(c)
                c["prompt_index"] = r
                keep[r - 1] = c
            out.append({**e, "retrieved_resources": keep,
                        "pool_k": k})
        p = os.path.join(os.path.expanduser(ns.out_dir), f"{name}_K{k}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        msg = f"  K={k:<4d} -> {os.path.basename(p)}"
        if short:
            msg += (f"   [!] {short} pools held fewer than {k} candidates; "
                    f"those rows are not really K={k}")
        print(msg)

    print("\nNext: run a selector on each file with the MATCHING --top_k, so "
          "each row is\na pool of that size rather than a bigger pool with a "
          "filter in front of it.")


if __name__ == "__main__":
    main()
