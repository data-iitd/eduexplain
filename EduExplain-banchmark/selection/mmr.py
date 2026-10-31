
"""
selection/mmr.py
=================
Plain greedy MMR for EduExplain ESR -- the baseline exactly as the paper states
it, with no reweighting and no normalisation:

    r* = argmax_{r in C_q \\ R_q} [ lambda * Rel(q,r)
                                   - (1 - lambda) * max_{r' in R_q} Sim(r,r') ]

    Rel(q,r)  the retrieval score, as it appears in the candidate file
    Sim(r,r') cosine between BGE-M3 summary embeddings (pre-normalised, so a
              dot product)
    lambda    0.8

Deliberately NOT done here:
  * no min-max normalisation of Rel. The two terms are on different scales and
    that does change the selections, but the equation above says "the retrieval
    score", and the benchmark's published MMR row was produced this way. Making
    the baseline stronger than the paper that defines it is not a fix, it is a
    different method. Keep this one for comparability.
  * no modality term -- that is Modality-Aware MMR, a separate row.

Semantics that are easy to get wrong, and are correct here:
  * With an empty selection the redundancy term is 0. After that it is the true
    max cosine, WHICH MAY BE NEGATIVE: a candidate pointing away from everything
    already chosen earns a bonus. Clamping that max at 0 silently tilts the
    objective toward relevance and inflates the baseline.
  * Ties go to the earlier candidate in the pool, so the output is deterministic.

Usage:
    python selection/mmr.py \\
        --gold_path data/gold_sets.json \\
        --candidates_path data/bge_top200_candidates.json \\
        --embeddings_path data/resource_embeddings.npy \\
        --resource_ids_path data/resource_ids.json \\
        --sizes_path sizes_mean.json \\
        --merged_resources_path data/resources.json \\
        --output_dir results/mmr/ --retrieval_name bge
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluator import build_gold_lookup, evaluate, write_results_csv

TOP_K = 100
LAMBDA_VALUES = [0.8]
TAU = 0.8


def load_embeddings(embeddings_path, resource_ids_path):
    emb = np.load(embeddings_path).astype(np.float32)
    with open(resource_ids_path, encoding="utf-8") as f:
        rids = json.load(f)
    if len(rids) != emb.shape[0]:
        raise ValueError(f"resource_ids ({len(rids)}) != embeddings rows "
                         f"({emb.shape[0]})")
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(n == 0, 1.0, n)
    print(f"  Embeddings loaded: shape={emb.shape}, normalized=True")
    return emb, {r: i for i, r in enumerate(rids)}


def select_mmr(candidates, embeddings, top_k, n, lam):
    """Greedy MMR. Returns n candidate dicts in selection order."""
    pool = sorted(candidates, key=lambda c: -float(c["score"]))[:top_k]
    if not pool:
        return []

    rel = np.array([float(c["score"]) for c in pool])
    idx = [c.get("embedding_index") for c in pool]
    has = np.array([i is not None and 0 <= i < embeddings.shape[0] for i in idx])
    EMB = np.zeros((len(pool), embeddings.shape[1]), dtype=np.float32)
    for j, i in enumerate(idx):
        if has[j]:
            EMB[j] = embeddings[i]

    chosen = []
    alive = np.ones(len(pool), dtype=bool)
    max_sim = None                       # None == empty selection -> term is 0
    for _ in range(min(n, len(pool))):
        pen = np.zeros(len(pool)) if max_sim is None else max_sim
        score = lam * rel - (1.0 - lam) * pen
        score[~alive] = -np.inf
        j = int(np.argmax(score))        # ties -> earlier candidate
        chosen.append(j)
        alive[j] = False
        if has[j]:
            sims = EMB @ EMB[j]          # cosine; may be negative
            sims[~has] = 0.0             # no embedding -> no redundancy signal
            max_sim = sims if max_sim is None else np.maximum(max_sim, sims)
    return [pool[j] for j in chosen]


def build_candidate_lookup(candidates_data, emb_map):
    lookup, missing, total = {}, 0, 0
    for entry in candidates_data:
        qid = entry["query_id"]
        out = []
        for cand in entry.get("retrieved_resources", []):
            total += 1
            if "score" not in cand:
                raise SystemExit(
                    f"candidate for {qid} has no 'score'; MMR cannot rank "
                    f"without it. Keys: {sorted(cand)}")
            c = dict(cand)
            rid = c["resource_id"]
            if rid in emb_map:
                c["embedding_index"] = emb_map[rid]
            else:
                c["embedding_index"] = None
                missing += 1
            out.append(c)
        lookup[qid] = out
    if missing:
        print(f"  [WARN] {missing}/{total} candidates have no embedding; kept "
              f"(dropping them would also change Recall@K)")
    return lookup


def main():
    p = argparse.ArgumentParser(description="Plain MMR selector for EduExplain ESR")
    p.add_argument("--gold_path", required=True)
    p.add_argument("--candidates_path", required=True)
    p.add_argument("--embeddings_path", required=True)
    p.add_argument("--resource_ids_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--retrieval_name", required=True)
    p.add_argument("--sizes_path", required=True,
                   help="JSON mapping query_id -> selection budget")
    p.add_argument("--merged_resources_path", required=True,
                   help="resource JSON carrying `concepts` per resource_id")
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--top_k", type=int, default=TOP_K)
    p.add_argument("--lambdas", default=None,
                   help=f"comma-separated lambdas (default {LAMBDA_VALUES})")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lambdas = ([float(x) for x in args.lambdas.split(",")] if args.lambdas
               else LAMBDA_VALUES)

    with open(args.gold_path, encoding="utf-8") as f:
        gold_data = json.load(f)
    with open(args.sizes_path, encoding="utf-8") as f:
        sizes = json.load(f)
    with open(args.candidates_path, encoding="utf-8") as f:
        candidates_data = json.load(f)
    with open(args.merged_resources_path, encoding="utf-8") as f:
        merged = json.load(f)
    if isinstance(merged, dict):
        merged = merged.get("resources", merged)
    concepts = {r["resource_id"]: r.get("concepts", []) for r in merged}
    print(f"Loaded concepts for {len(concepts):,} resources.")

    embeddings, emb_map = load_embeddings(args.embeddings_path,
                                          args.resource_ids_path)
    gold_lookup = build_gold_lookup(gold_data)
    candidate_lookup = build_candidate_lookup(candidates_data, emb_map)
    meta = {e["query_id"]: e for e in candidates_data}

    qids = sorted(set(gold_lookup) & set(candidate_lookup))
    no_budget = [q for q in qids if q not in sizes]
    if no_budget:
        print(f"  [WARN] {len(no_budget)} of {len(qids)} queries have no budget "
              f"and are skipped; averages are over {len(qids) - len(no_budget)} "
              f"queries.")
    print(f"Queries to evaluate: {len(qids) - len(no_budget)}")

    for lam in lambdas:
        preds = []
        for qid in qids:
            if qid not in sizes:
                continue
            sel = select_mmr(candidate_lookup[qid], embeddings,
                             top_k=args.top_k, n=int(sizes[qid]), lam=lam)
            for r in sel:
                r["concepts"] = concepts.get(r["resource_id"], [])
            e = meta.get(qid, {})
            preds.append({"query_id": qid, "resources": sel,
                          "n_predicted": len(sel),
                          "query_type": e.get("query_type"),
                          "variant_style": e.get("variant_style"),
                          "bundle_type": e.get("bundle_type")})

        results = evaluate(predictions=preds, candidate_lookup=candidate_lookup,
                           gold_lookup=gold_lookup, tau=args.tau)

        print("\n" + "=" * 70)
        print(f"MMR  TopK={args.top_k}  lambda={lam}  budget=given")
        print(f"Queries: {len(results)}")
        print("=" * 70)
        for m in ("precision", "recall", "f1", "consistent_f1",
                  "concept_coverage", "n_predicted"):
            v = [float(r[m]) for r in results if r.get(m) is not None]
            if v:
                print(f"{m:20s}: {np.mean(v):.4f}")
        print("=" * 70)

        write_results_csv(
            results,
            os.path.join(args.output_dir,
                         f"mmr_{args.retrieval_name}_topk{args.top_k}"
                         f"_lam{lam}_given_budget.csv"),
            extra_meta={"selector": "mmr",
                        "retrieval_system": args.retrieval_name,
                        "top_k": args.top_k, "lambda": lam,
                        "budget": "query-specific",
                        "sizes_path": args.sizes_path, "tau": args.tau})

    print(f"\nDone. Written to {args.output_dir}")


if __name__ == "__main__":
    main()
