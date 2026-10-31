"""
selection/top_n.py
===================
Top-n selector for EduExplain ESR -- take the first n of the ranked pool.
No re-ranking, nothing to tune, fully deterministic.

Usage:
    python selection/top_n.py \\
        --gold_path data/gold_sets.json \\
        --candidates_path data/bge_top100_candidates.json \\
        --merged_resources_path data/resources.json \\
        --sizes_path sizes_mean.json \\
        --output_dir results/top_n/ --retrieval_name bge
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline_predictions_out import write_predictions_json
from evaluation.evaluator import build_gold_lookup, evaluate, write_results_csv


def cfield(c, key, default=None):
    """Read a field from a candidate dict, falling back to its nested
    'resource' sub-dict.

    The BGE retriever writes each candidate as
        {"prompt_index": .., "resource_id": .., "score": .., "resource": {...}}
    so resource_type, concepts and every other resource attribute live one level
    down. A plain c.get("resource_type") returns None for every candidate, which
    folds the whole pool to one modality and silently disables the modality
    logic. Always go through this.
    """
    v = c.get(key)
    if v in (None, "", []):
        r = c.get("resource")
        if isinstance(r, dict):
            v = r.get(key, default)
    return default if v in (None, "") else v


TOP_K = 100
TAU   = 0.8


def select_top_n(candidates, top_k, n):
    pool = sorted(candidates, key=lambda c: -float(c.get("score", 0.0)))[:top_k]
    return pool[:n]


def build_candidate_lookup(candidates_data):
    lookup, key_used = {}, None
    n_noscore = 0
    for entry in candidates_data:
        raw = None
        for k in ("retrieved_resources", "candidates", "resources"):
            if isinstance(entry.get(k), list):
                raw, key_used = entry[k], k
                break
        raw = raw or []
        n_noscore += sum(1 for c in raw if "score" not in c)
        lookup[entry["query_id"]] = raw
    if key_used is None:
        raise SystemExit("no candidate list found in the candidates file; "
                         "expected 'retrieved_resources' or 'candidates'")
    empty = sum(1 for v in lookup.values() if not v)
    print(f"  candidate lists read from '{key_used}'"
          + (f"; {empty} queries have an EMPTY pool" if empty else ""))
    if n_noscore:
        print(f"  [WARN] {n_noscore} candidates have no 'score'; they sort as "
              f"0.0, so the\n         file's own order decides the ranking for "
              f"them.")
    return lookup


def main():
    p = argparse.ArgumentParser(description="Top-n selector for EduExplain ESR")
    p.add_argument("--gold_path", required=True)
    p.add_argument("--candidates_path", required=True)
    p.add_argument("--merged_resources_path", required=True,
                   help="resource JSON carrying `concepts` per resource_id; "
                        "without it Concept Coverage silently scores 0")
    p.add_argument("--sizes_path", required=True,
                   help="JSON mapping query_id -> selection budget n")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--predictions_out", default=None,
                   help="also write per-query predictions as JSON, "
                        "for tools_paired_test.py")
    p.add_argument("--retrieval_name", required=True)
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--top_k", type=int, default=TOP_K)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.gold_path, encoding="utf-8") as f:
        gold_data = json.load(f)
    with open(args.candidates_path, encoding="utf-8") as f:
        candidates_data = json.load(f)
    with open(args.sizes_path, encoding="utf-8") as f:
        sizes = json.load(f)
    with open(args.merged_resources_path, encoding="utf-8") as f:
        merged = json.load(f)
    if isinstance(merged, dict):
        merged = merged.get("resources", merged)
    concepts = {r["resource_id"]: r.get("concepts", []) for r in merged}
    print(f"Loaded concepts for {len(concepts):,} resources.")

    gold_lookup = build_gold_lookup(gold_data)
    candidate_lookup = build_candidate_lookup(candidates_data)
    meta = {e["query_id"]: e for e in candidates_data}

    qids = sorted(set(gold_lookup) & set(candidate_lookup))
    no_budget = [q for q in qids if q not in sizes]
    if no_budget:
        print(f"  [WARN] {len(no_budget)} of {len(qids)} queries have no budget "
              f"and are skipped; averages are over "
              f"{len(qids) - len(no_budget)} queries.")
    print(f"Queries in gold: {len(gold_lookup)}   in candidates: "
          f"{len(candidate_lookup)}   evaluated: {len(qids) - len(no_budget)}")
    print(f"TopK={args.top_k}  n = per-query budget from "
          f"{os.path.basename(args.sizes_path)}")

    preds, short, n_missing_con = [], 0, 0
    for qid in qids:
        if qid not in sizes:
            continue
        n = max(1, int(sizes[qid]))
        sel = select_top_n(candidate_lookup[qid], args.top_k, n)
        if len(sel) < n:
            short += 1
        for r in sel:
            r["concepts"] = (concepts.get(r["resource_id"])
                             or cfield(r, "concepts", []) or [])
            if r["resource_id"] not in concepts:
                n_missing_con += 1
        e = meta.get(qid, {})
        preds.append({"query_id": qid, "resources": sel,
                      "n_predicted": len(sel),
                      "query_type": e.get("query_type"),
                      "variant_style": e.get("variant_style"),
                      "bundle_type": e.get("bundle_type")})

    if args.predictions_out:
        write_predictions_json(
            preds, args.predictions_out,
            meta={"selector": "top_n",
                  "candidates_path": args.candidates_path,
                  "sizes_path": getattr(args, "sizes_path", None),
                  "top_k": getattr(args, "top_k", None)})

    results = evaluate(predictions=preds, candidate_lookup=candidate_lookup,
                       gold_lookup=gold_lookup, tau=args.tau)

    print("\n" + "=" * 70)
    print(f"TOP-N  TopK={args.top_k}  budget=given")
    print(f"Queries: {len(results)}")
    print("=" * 70)
    for m in ("precision", "recall", "f1", "consistent_f1",
              "concept_coverage", "n_predicted"):
        v = [float(r[m]) for r in results if r.get(m) is not None]
        if v:
            print(f"{m:20s}: {np.mean(v):.4f}")
    print("=" * 70)

    if short:
        print(f"[WARN] {short} queries returned fewer than their budget -- the "
              f"pool was shorter than n.")
    if n_missing_con:
        print(f"[WARN] {n_missing_con} selected resources are absent from the "
              f"resource file, so they\n       carry no concepts and depress "
              f"Concept Coverage. Check the id spaces match.")

    write_results_csv(
        results,
        os.path.join(args.output_dir,
                     f"top_n_{args.retrieval_name}_topk{args.top_k}"
                     f"_given_budget.csv"),
        extra_meta={"selector": "top_n",
                    "retrieval_system": args.retrieval_name,
                    "top_k": args.top_k, "budget": "query-specific",
                    "sizes_path": args.sizes_path, "tau": args.tau})
    print(f"\nDone. Written to {args.output_dir}")


if __name__ == "__main__":
    main()
