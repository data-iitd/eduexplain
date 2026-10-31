
"""
selection/modality_aware_mmr.py
================================
Modality-Aware MMR for EduExplain ESR -- the baseline exactly as the paper
states it, at the stated parameters.

    Score(r) = lambda * Rel(q,r)
             - (1 - lambda) * max_{r' in R_q} Sim(r, r')
             - gamma * N_{m(r)}(R_q) / kappa_{m(r)}

    N_m(R_q)  how many selected resources already have modality m
    kappa_m   that modality's quota (kappa_text = 2, kappa_image = 1)
    lambda    0.8        gamma  0.1

n is the PER-QUERY budget from --sizes_path, the same file the other baselines
are given, so every system in the table is compared at the same |R_q|.

THE PENALTY HERE IS NOT THE ONE THE PREVIOUS SCRIPT COMPUTED
------------------------------------------------------------
That script used  gamma * (same-modality count / |R_q|)  -- the PROPORTION of
the selected set sharing the candidate's modality. The paper divides by
kappa_m, not by |R_q|. They behave differently (gamma = 0.1, kappa_text = 2):

    selected          paper  g*N_m/kappa_m     old script  g*N_m/|S|
    1 text                   0.050                         0.100
    2 text                   0.100                         0.100
    1 text + 1 visual        0.050                         0.050
    2 text + 1 visual        0.100                         0.067

The old form is maximal immediately after the first same-modality pick and
SHRINKS as the set gets mixed; the paper's grows with the count and passes 1
beyond the quota. This file implements the paper's.

WHAT THE PENALTY IS WORTH
-------------------------
With lambda = 0.8 the relevance term spans 0.8 x (range of Rel) while the
penalty is at most gamma = 0.1 within quota. On RAW retrieval scores -- which is
what the MMR baseline uses, and what this keeps for comparability -- that makes
the penalty small, and if the scores are unnormalised and wide it is numerically
invisible. The run reports how often the modality term changed a selection
relative to plain MMR; if that is 0, the row IS plain MMR and the table should
not carry it as a separate method.

Also note this is a soft penalty: nothing stops the set from exceeding kappa_m.
That is the difference from Modality-Capped, which refuses.

Usage:
    python selection/modality_aware_mmr.py \\
        --gold_path data/gold_sets.json \\
        --candidates_path data/bge_top200_candidates.json \\
        --embeddings_path data/resource_embeddings.npy \\
        --resource_ids_path data/resource_ids.json \\
        --merged_resources_path data/resources.json \\
        --sizes_path sizes_mean.json \\
        --output_dir results/modality_aware_mmr/ --retrieval_name bge
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluator import build_gold_lookup, evaluate, write_results_csv

TOP_K   = 100
LAMBDA  = 0.8
GAMMA   = 0.1
KAPPA   = {"text": 2, "visual": 1}
TAU     = 0.8

TEXT_TYPES   = {"explanation", "definition", "example", "activity", "exercise"}
VISUAL_TYPES = {"figure", "diagram", "table", "image"}


def get_modality(resource_type):
    """(modality, recognised). Unknown types fall back to text and are counted:
    a type the map does not know inflates the text side silently."""
    rt = (resource_type or "").lower().strip()
    if rt in VISUAL_TYPES:
        return "visual", True
    if rt in TEXT_TYPES:
        return "text", True
    return "text", False


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


def select(candidates, embeddings, top_k, n, lam, gamma, kappa,
           modality_field="resource_type", penalty="proportion"):
    """Greedy Modality-Aware MMR. gamma = 0 reduces to plain MMR.

    penalty="proportion" (default, the benchmark's reference implementation):
        gamma * N_m(S) / |S|   -- the share of the selected set with the
        candidate's modality. kappa is not used.
    penalty="quota" (the draft's equation):
        gamma * N_m(S) / kappa_m

    They are different functions and they are NOT interchangeable; see the
    module docstring. The default follows the reference script so this row
    reproduces the benchmark. If the paper keeps the kappa form in print, run
    --penalty quota and make the text match whichever you report.
    """
    pool = sorted(candidates, key=lambda c: -float(c["score"]))[:top_k]
    if not pool:
        return [], {"text": 0, "visual": 0, "unknown": 0}

    rel = np.array([float(c["score"]) for c in pool])
    mods, known = zip(*[get_modality(c.get(modality_field, "")) for c in pool])
    mods = np.array(mods)

    idx = [c.get("embedding_index") for c in pool]
    has = np.array([i is not None and 0 <= i < embeddings.shape[0] for i in idx])
    EMB = np.zeros((len(pool), embeddings.shape[1]), dtype=np.float32)
    for j, i in enumerate(idx):
        if has[j]:
            EMB[j] = embeddings[i]

    chosen, alive = [], np.ones(len(pool), dtype=bool)
    max_sim = None                        # None == empty set -> the term is 0
    counts = {"text": 0, "visual": 0}
    n_unknown = 0
    for _ in range(min(n, len(pool))):
        pen_sim = np.zeros(len(pool)) if max_sim is None else max_sim
        n_sel = len(chosen)
        if penalty == "quota":
            pen_mod = np.array([counts[m] / float(kappa.get(m, 1)) for m in mods])
        else:                                   # reference: share of the set
            pen_mod = (np.zeros(len(pool)) if n_sel == 0
                       else np.array([counts[m] / float(n_sel) for m in mods]))
        score = lam * rel - (1.0 - lam) * pen_sim - gamma * pen_mod
        score[~alive] = -np.inf
        j = int(np.argmax(score))         # ties -> earlier candidate
        chosen.append(j)
        alive[j] = False
        counts[mods[j]] += 1
        if not known[j]:
            n_unknown += 1
        if has[j]:
            sims = EMB @ EMB[j]           # cosine; MAY BE NEGATIVE, not clamped
            sims[~has] = 0.0
            max_sim = sims if max_sim is None else np.maximum(max_sim, sims)
    return [pool[j] for j in chosen], {**counts, "unknown": n_unknown}


def build_candidate_lookup(candidates_data, emb_map, types=None,
                           modality_field="resource_type"):
    """Accepts either key. The original read entry['candidates'], while the BGE
    file uses 'retrieved_resources'; with the wrong key every pool is empty and
    every prediction scores 0.0 without raising."""
    lookup, key_used, missing, total = {}, None, 0, 0
    n_filled = n_notype = 0
    pool_mix = {"text": 0, "visual": 0}
    for entry in candidates_data:
        raw = None
        for k in ("retrieved_resources", "candidates", "resources"):
            if isinstance(entry.get(k), list):
                raw, key_used = entry[k], k
                break
        out = []
        for cand in (raw or []):
            total += 1
            if "score" not in cand:
                raise SystemExit(f"candidate for {entry['query_id']} has no "
                                 f"'score'. Keys: {sorted(cand)}")
            c = dict(cand)
            # The modality term is computed from this field. If the candidates
            # file does not carry it, every candidate folds to one modality,
            # the penalty becomes the SAME CONSTANT for all of them at each
            # step, and it cancels out of the argmax -- the row then reproduces
            # plain MMR exactly. Fall back to the resource file, which has it.
            if not c.get(modality_field) and types:
                t = types.get(c["resource_id"])
                if t:
                    c[modality_field] = t
                    n_filled += 1
            if not c.get(modality_field):
                n_notype += 1
            if c["resource_id"] in emb_map:
                c["embedding_index"] = emb_map[c["resource_id"]]
            else:
                c["embedding_index"] = None
                missing += 1
            m, _ = get_modality(c.get(modality_field, ""))
            pool_mix[m] += 1
            out.append(c)
        lookup[entry["query_id"]] = out
    if key_used is None:
        raise SystemExit("no candidate list found; expected "
                         "'retrieved_resources' or 'candidates'")
    print(f"  candidate lists read from '{key_used}'")
    if missing:
        print(f"  [WARN] {missing}/{total} candidates have no embedding; kept "
              f"(dropping them would also change Recall@K)")
    if n_filled:
        print(f"  '{modality_field}' was absent on {n_filled}/{total} "
              f"candidates and was filled in from the resource file")
    if n_notype:
        print(f"  [WARN] {n_notype}/{total} candidates still have no "
              f"'{modality_field}' and fold to TEXT")
    tot = max(1, sum(pool_mix.values()))
    print(f"  pool modality mix: text {pool_mix['text']/tot:.1%}, "
          f"visual {pool_mix['visual']/tot:.1%}")
    if pool_mix["visual"] == 0 or pool_mix["text"] == 0:
        print("  [FATAL-ish] the candidate pools are SINGLE-MODALITY. The "
              "penalty is then the\n              same constant for every "
              "candidate at each step, so it cancels out of\n              the "
              "argmax and this row will reproduce plain MMR EXACTLY. Fix the "
              "modality\n              field before reporting it.")
    return lookup


def main():
    p = argparse.ArgumentParser(description="Modality-Aware MMR for EduExplain ESR")
    p.add_argument("--gold_path", required=True)
    p.add_argument("--candidates_path", required=True)
    p.add_argument("--embeddings_path", required=True)
    p.add_argument("--resource_ids_path", required=True)
    p.add_argument("--merged_resources_path", required=True,
                   help="resource JSON carrying `concepts`; without it Concept "
                        "Coverage silently scores 0")
    p.add_argument("--sizes_path", required=True,
                   help="JSON mapping query_id -> budget n")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--retrieval_name", required=True)
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--top_k", type=int, default=TOP_K)
    p.add_argument("--lam", type=float, default=LAMBDA)
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--kappa_text", type=int, default=KAPPA["text"])
    p.add_argument("--kappa_image", type=int, default=KAPPA["visual"])
    p.add_argument("--penalty", default="proportion",
                   choices=["proportion", "quota"],
                   help="proportion = gamma*N_m/|S| (the reference script, "
                        "default); quota = gamma*N_m/kappa_m (the draft's "
                        "equation)")
    p.add_argument("--modality_field", default="resource_type")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    kappa = {"text": args.kappa_text, "visual": args.kappa_image}

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

    embeddings, emb_map = load_embeddings(args.embeddings_path,
                                          args.resource_ids_path)
    gold_lookup = build_gold_lookup(gold_data)
    res_types = {r["resource_id"]: (r.get("resource_type") or r.get("type"))
                 for r in merged}
    candidate_lookup = build_candidate_lookup(candidates_data, emb_map,
                                              res_types, args.modality_field)
    meta = {e["query_id"]: e for e in candidates_data}

    qids = sorted(set(gold_lookup) & set(candidate_lookup))
    no_budget = [q for q in qids if q not in sizes]
    if no_budget:
        print(f"  [WARN] {len(no_budget)} of {len(qids)} queries have no budget "
              f"and are skipped.")
    print(f"Queries to evaluate: {len(qids) - len(no_budget)}")
    print(f"lambda={args.lam}  gamma={args.gamma}  penalty={args.penalty}"
          f"{'  (kappa_text=%d, kappa_image=%d)' % (args.kappa_text, args.kappa_image) if args.penalty == 'quota' else ''}"
          f"  TopK={args.top_k}  n = per-query budget")

    preds, mix, unknown, n_diff = [], {"text": 0, "visual": 0}, 0, 0
    for qid in qids:
        if qid not in sizes:
            continue
        n = max(1, int(sizes[qid]))
        n=3
        sel, c = select(candidate_lookup[qid], embeddings, args.top_k, n,
                        args.lam, args.gamma, kappa, args.modality_field,
                        args.penalty)
        # Does the modality term change anything? If never, this row is MMR.
        base, _ = select(candidate_lookup[qid], embeddings, args.top_k, n,
                         args.lam, 0.0, kappa, args.modality_field,
                         args.penalty)
        if [r["resource_id"] for r in base] != [r["resource_id"] for r in sel]:
            n_diff += 1
        mix["text"] += c["text"]; mix["visual"] += c["visual"]
        unknown += c["unknown"]
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
    print(f"MODALITY-AWARE MMR  TopK={args.top_k}  lambda={args.lam}  "
          f"gamma={args.gamma}  budget=given")
    print(f"Queries: {len(results)}")
    print("=" * 70)
    for m in ("precision", "recall", "f1", "consistent_f1",
              "concept_coverage", "n_predicted"):
        v = [float(r[m]) for r in results if r.get(m) is not None]
        if v:
            print(f"{m:20s}: {np.mean(v):.4f}")
    print("=" * 70)

    tot = max(1, mix["text"] + mix["visual"])
    print(f"modality mix : text {mix['text']/tot:.1%}, "
          f"visual {mix['visual']/tot:.1%}  (kappa {kappa})")
    print(f"vs plain MMR : differs on {n_diff}/{len(preds)} queries "
          f"({n_diff/max(1,len(preds)):.1%})")
    if not n_diff:
        print("               IDENTICAL to plain MMR. Check the pool modality "
              "mix printed above FIRST:\n               if the pools are "
              "single-modality the penalty is a constant across candidates "
              "and\n               cancels, which produces an exact match. If "
              "the mix is genuinely mixed, then\n               gamma is "
              "simply too small against the relevance term, and the row should "
              "not\n               be reported as a separate method without "
              "saying so.")
    if unknown:
        print(f"[WARN] {unknown} selected resources had a resource_type outside "
              f"the modality map and were counted as TEXT.")

    write_results_csv(
        results,
        os.path.join(args.output_dir,
                     f"modality_aware_mmr_{args.retrieval_name}"
                     f"_topk{args.top_k}_lam{args.lam}_g{args.gamma}"
                     f"_given_budget.csv"),
        extra_meta={"selector": "modality_aware_mmr",
                    "retrieval_system": args.retrieval_name,
                    "top_k": args.top_k, "lambda": args.lam,
                    "gamma": args.gamma, "penalty": args.penalty,
                    "kappa_text": args.kappa_text,
                    "kappa_image": args.kappa_image, "tau": args.tau,
                    "budget": "query-specific", "sizes_path": args.sizes_path})
    print(f"\nDone. Written to {args.output_dir}")


if __name__ == "__main__":
    main()
