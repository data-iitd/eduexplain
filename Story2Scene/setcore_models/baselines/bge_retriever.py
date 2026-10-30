"""
BGE retrieval

  * resource text  = index_text(), exactly as above
  * query text     = the first present of query_text / query / text
  * model          = whatever --bge-model points at (use bge-large-en-v1.5)
  * max_seq_length = 512
  * L2-normalised embeddings, cosine == dot product
  * no instruction prefix on either side (icra adds none)

Usage:
    python bge_retriever.py \\
        --gold path/to/gold/set \\
        --resources path/to/all/resources \\
        --bge-model BAAI/bge-large-en-v1.5 \\
        --output bge_top100_matched.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def _first(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def index_text(resource):
    for key in ("summary", "description", "image_caption"):
        v = resource.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip(), key
    v = (resource.get("text") or "").strip()[:2000]
    if v:
        return v, "text"
    return str(resource.get("resource_id", "")), "resource_id"


def query_text(item):
    return str(_first(item, "query_text", "query", "text", default=""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="benchmark JSON with the queries")
    ap.add_argument("--resources", required=True)
    ap.add_argument("--bge-model", required=True,
                    help="local path or hub id, e.g. BAAI/bge-large-en-v1.5")
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args()

    with open(args.gold, encoding="utf-8") as f:
        gold_data = json.load(f)
    with open(args.resources, encoding="utf-8") as f:
        resources = json.load(f)
    if isinstance(resources, dict):
        resources = resources.get("resources", resources)

    print(f"Loaded {len(gold_data)} queries, {len(resources)} resources.")

    texts, sources = [], Counter()
    for r in resources:
        t, src = index_text(r)
        texts.append(t)
        sources[src] += 1
    print("resource text drawn from:")
    for k, v in sources.most_common():
        print(f"  {k:<14s} {v:>7d}  ({v/len(resources):.1%})")
    empty = sum(1 for t in texts if not t.strip())
    if empty:
        print(f"  [WARN] {empty} resources have no usable text at all")
    print(f"mean characters per resource: "
          f"{sum(len(t) for t in texts)/max(1,len(texts)):.0f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {args.bge_model} on {device}")
    model = SentenceTransformer(args.bge_model, device=device)
    model.max_seq_length = args.max_seq_len

    print(f"Embedding {len(texts)} resources...")
    R = model.encode(texts, batch_size=args.batch_size, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=True)
    R = np.asarray(R, dtype=np.float32)
    print(f"Embedding matrix: {R.shape}")

    qtexts = [query_text(it) for it in gold_data]
    n_empty_q = sum(1 for q in qtexts if not q.strip())
    if n_empty_q:
        print(f"[WARN] {n_empty_q} queries have empty text")
    print(f"Embedding {len(qtexts)} queries...")
    Q = model.encode(qtexts, batch_size=args.batch_size, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=True)
    Q = np.asarray(Q, dtype=np.float32)

    S = Q @ R.T                                  # cosine, both normalised
    k = min(args.top_k, R.shape[0])
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]

    all_results = []
    for i, item in enumerate(gold_data):
        idx = part[i][np.argsort(-S[i, part[i]], kind="stable")]
        retrieved = []
        for rank, j in enumerate(idx, start=1):
            res = resources[int(j)]
            retrieved.append({
                "prompt_index": rank,
                "resource_id": res.get("resource_id"),
                "score": float(S[i, j]),
                "resource_type": res.get("resource_type") or res.get("type"),
                "concepts": res.get("concepts", []),
                "resource": res,
            })
        all_results.append({
            "query_id": item["query_id"],
            "query": qtexts[i],
            "retrieved_resources": retrieved,
        })
        if (i + 1) % 50 == 0 or i + 1 == len(gold_data):
            print(f"  [{i+1}/{len(gold_data)}]")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"Saved to: {out}")
    print(f"Queries: {len(all_results)}   candidates per query: {k}")
    print("Resource representation: icra's index_text() -- the first non-empty "
          "of\nsummary / description / image_caption, else text[:2000].")
    print("Run tools_pool_audit.py on this file and on the old one with "
          "--compare_candidates_path\nto see what the extra fields were worth.")
    print("=" * 70)


if __name__ == "__main__":
    main()
