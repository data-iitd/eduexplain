#!/usr/bin/env python3
"""Dense retrieval with BGE-M3. 

REUSES the corpus embeddings the benchmark repo already ships
(dense_retrieval/resource_embeddings.npy + resource_ids.json, 6921 x 1024,
L2-normalised) rather than recomputing them, so this row is the same dense
space the benchmark was built in. Re-encoding would be a different row wearing
the same name.

The QUERY embeddings cannot be reused: dense_retrieval/query_embeddings.npy
was built for main_queries.json and covers only a couple of dozen of the 321
gold-set queries. They are encoded here.

With --resource-emb omitted the corpus is encoded too, which is what you want
if the corpus changed since those .npy files were written.

Usage
-----
  python dense_retriever.py --gold $Q --resources $R \\
      --resource-emb dense_retrieval/resource_embeddings.npy \\
      --resource-ids dense_retrieval/resource_ids.json \\
      --output pools/dense_top100.json
"""
from __future__ import annotations

import argparse
import json
from typing import List, Sequence

import numpy as np

from retrieval_io import (load_corpus, query_text, report_texts,
                          require_local_model, write_candidates)


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


class Encoder:
    """BGE-M3 through FlagEmbedding when available, else sentence-transformers.

    The benchmark's own embeddings came from FlagEmbedding's BGEM3FlagModel.
    Its dense vector and sentence-transformers' pooled output for bge-m3 are
    the same thing, but only when both L2-normalise -- which is enforced here
    rather than assumed, because an unnormalised query against normalised
    resources ranks by a scaled cosine and the scores stop being comparable
    across queries.
    """

    def __init__(self, model: str, batch_size: int = 16,
                 max_len: int = 512) -> None:
        model = require_local_model(model, "--model (BGE-M3)")
        self.batch_size, self.max_len = batch_size, max_len
        self.backend = None
        try:
            from FlagEmbedding import BGEM3FlagModel
            self.m = BGEM3FlagModel(model, use_fp16=True)
            self.backend = "FlagEmbedding"
        except Exception:
            from sentence_transformers import SentenceTransformer
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.m = SentenceTransformer(model, device=dev)
            self.m.max_seq_length = max_len
            self.backend = "sentence-transformers"
        print(f"encoder: {model} via {self.backend}")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.backend == "FlagEmbedding":
            v = self.m.encode(list(texts), batch_size=self.batch_size,
                              max_length=self.max_len)["dense_vecs"]
        else:
            v = self.m.encode(list(texts), batch_size=self.batch_size,
                              convert_to_numpy=True,
                              normalize_embeddings=True,
                              show_progress_bar=True)
        return l2(np.asarray(v, dtype=np.float32))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--resources", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--resource-emb", default=None,
                    help="precomputed resource_embeddings.npy to reuse")
    ap.add_argument("--resource-ids", default=None,
                    help="resource_ids.json matching --resource-emb, row for "
                         "row")
    ap.add_argument("--emb-out", default=None,
                    help="write the resource embeddings here (the MMR "
                         "selectors need them)")
    ap.add_argument("--ids-out", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=100)
    ns = ap.parse_args()

    gold, resources = load_corpus(ns.gold, ns.resources)
    texts = report_texts(resources)
    rids = [str(r.get("resource_id")) for r in resources]

    enc = None
    if ns.resource_emb and ns.resource_ids:
        E = np.load(ns.resource_emb).astype(np.float32)
        with open(ns.resource_ids, encoding="utf-8") as f:
            emb_ids = [str(x) for x in json.load(f)]
        if len(emb_ids) != E.shape[0]:
            raise SystemExit(f"{ns.resource_emb} has {E.shape[0]} rows but "
                             f"{ns.resource_ids} lists {len(emb_ids)} ids")
        pos = {rid: i for i, rid in enumerate(emb_ids)}
        missing = [r for r in rids if r not in pos]
        if missing:
            raise SystemExit(
                f"{len(missing)} resources have no precomputed embedding "
                f"(e.g. {missing[:3]}).\n  The corpus and the .npy are out of "
                f"step -- drop --resource-emb to re-encode, rather than\n  "
                f"retrieving over a subset of the corpus without saying so.")
        # Reordered to the CORPUS order, so every index in this script means
        # the same resource as it does in the candidates file.
        R = l2(E[[pos[r] for r in rids]])
        print(f"reusing {ns.resource_emb}: {R.shape[0]} x {R.shape[1]}")
    else:
        enc = Encoder(ns.model, ns.batch_size, ns.max_seq_len)
        print(f"encoding {len(texts)} resources...")
        R = enc.encode(texts)
        print(f"resource matrix: {R.shape}")

    qtexts = [query_text(it) for it in gold]
    n_empty = sum(1 for q in qtexts if not q.strip())
    if n_empty:
        print(f"[WARN] {n_empty} queries have empty text")
    if enc is None:
        enc = Encoder(ns.model, ns.batch_size, ns.max_seq_len)
    print(f"encoding {len(qtexts)} queries...")
    Q = enc.encode(qtexts)
    if Q.shape[1] != R.shape[1]:
        raise SystemExit(
            f"query dim {Q.shape[1]} != resource dim {R.shape[1]}. The "
            f"precomputed resource\n  embeddings came from a different model "
            f"than --model; they cannot be compared.")

    S = Q @ R.T
    k = min(ns.top_k, R.shape[0])
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]
    order, scores = [], []
    for i in range(len(qtexts)):
        idx = part[i][np.argsort(-S[i, part[i]], kind="stable")]
        order.append(idx.tolist())
        scores.append(S[i, idx].tolist())

    write_candidates(ns.output, gold, qtexts, resources, order, scores,
                     f"dense:{ns.model}")

    if ns.emb_out:
        np.save(ns.emb_out, R.astype(np.float32))
        print(f"resource embeddings -> {ns.emb_out}  {R.shape}")
    if ns.ids_out:
        with open(ns.ids_out, "w", encoding="utf-8") as f:
            json.dump(rids, f)
        print(f"resource ids -> {ns.ids_out}")
    if bool(ns.emb_out) != bool(ns.ids_out):
        print("  [!] the MMR selectors need BOTH --emb-out and --ids-out; an "
              "embedding matrix\n      without its id list cannot be aligned "
              "to the candidates.")


if __name__ == "__main__":
    main()
