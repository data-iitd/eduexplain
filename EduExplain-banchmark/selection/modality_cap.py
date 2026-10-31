

"""
selection/modality_cap.py
==========================
Modality-Capped selection for EduExplain ESR -- the baseline exactly as the
paper states it, at the stated parameters.

    Partition the pool by modality, M = {text, image}, and set a quota kappa_m
    per modality with sum_m kappa_m = n. Scanning candidates in RANKED order, a
    resource r is added iff |R_q^(m(r))| < kappa_{m(r)}; selection stops once
    |R_q| = n.

    kappa_text = 2, kappa_image = 1, over the top-100 pool.

n is the PER-QUERY budget from --sizes_path -- the same file every other
baseline is given -- so the systems in the table are compared at the same
|R_q|. The paper sets sum_m kappa_m = n = 3; with a per-query budget those two
can disagree, and --scale_caps decides which one gives way (see caps_for).

This preserves the retrieval ranking and only prevents one modality from
filling the set. There is no scoring and nothing to tune: given the pool and
the caps the output is determined, so there are no seeds and no variance.

Fixed here rather than swept: the original looped TopK x text_cap x visual_cap
over 24 cells, which is a grid to choose from, not a baseline. The paper names
one setting; this runs that one. --text_cap / --visual_cap / --top_k remain if
you want the grid back.

Usage:
    python selection/modality_cap.py \\
        --gold_path data/gold_sets.json \\
        --candidates_path data/bge_top200_candidates.json \\
        --merged_resources_path data/resources.json \\
        --sizes_path sizes_mean.json \\
        --output_dir results/modality_cap/ --retrieval_name bge
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluator import (build_gold_lookup, compute_best_metrics,
                                  evaluate, write_results_csv)

TOP_K      = 100
TEXT_CAP   = 2        # kappa_text
VISUAL_CAP = 1        # kappa_image
TAU        = 0.8

# Extended to the vocabulary this corpus actually uses. The original map was
# missing "text"/"passage"/"explanatory" on the text side and
# "illustration"/"photo" on the visual side; anything unrecognised folds to
# TEXT, so a corpus typed with those names comes out 100% text and the visual
# quota never applies.
TEXT_TYPES   = {"explanation", "explanatory", "definition", "example",
                "activity", "exercise", "text", "passage", "worked_example",
                "solved_example", "note", "summary"}
VISUAL_TYPES = {"figure", "diagram", "table", "image", "illustration",
                "photo", "chart", "graph", "picture", "visual"}


def get_modality(resource_type: str) -> tuple[str, bool]:
    """(modality, was_recognised). Unknown types fall back to text, but the
    caller counts them: a type the map does not know inflates the text side
    silently, and on this corpus that would quietly change the modality mix."""
    rt = (resource_type or "").lower().strip()
    if rt in TEXT_TYPES:
        return "text", True
    if rt in VISUAL_TYPES:
        return "visual", True
    return "text", False


def caps_for(n, text_cap, visual_cap, scale):
    """Per-modality quotas for a query whose budget is n.

    scale=False (default): the paper's kappa_text=2, kappa_image=1 regardless
        of n. Faithful to the stated parameters, but a query whose budget is 4
        or more can never be filled -- the caps sum to 3 -- so the row is short
        on exactly the queries with the largest gold sets. The run reports how
        often that happens.
    scale=True: keep the 2:1 ratio and spread it over n, so the budget is always
        reachable and every system in the table returns the same |R_q|.
    """
    if not scale:
        return text_cap, visual_cap
    vis = max(1, int(round(n * visual_cap / float(text_cap + visual_cap))))
    vis = min(vis, max(1, n - 1)) if n > 1 else 0
    return n - vis, vis


def select_modality_cap(candidates, top_k, text_cap, visual_cap, max_n=None,
                        modality_field="resource_type"):
    """Ranked scan with per-modality quotas. Returns the selected dicts."""
    max_n = (text_cap + visual_cap) if max_n is None else max_n
    pool = sorted(candidates, key=lambda c: -float(c.get("score", 0.0)))[:top_k]

    selected, n_text, n_vis, n_unknown = [], 0, 0, 0
    for cand in pool:
        if len(selected) >= max_n:
            break
        if n_text >= text_cap and n_vis >= visual_cap:
            break
        mod, known = get_modality(cand.get(modality_field, ""))
        if not known:
            n_unknown += 1
        if mod == "text" and n_text < text_cap:
            selected.append(cand)
            n_text += 1
        elif mod == "visual" and n_vis < visual_cap:
            selected.append(cand)
            n_vis += 1
    return selected, {"text": n_text, "visual": n_vis, "unknown": n_unknown}


def build_candidate_lookup(candidates_data, types=None,
                           modality_field="resource_type"):
    """Accepts either key. The original read entry['candidates'], while the BGE
    file written for this benchmark uses 'retrieved_resources' -- with the wrong
    key every pool is empty and every prediction comes back empty, which scores
    0.0 without raising anything."""
    from collections import Counter
    lookup, key_used = {}, None
    seen, n_filled, n_blank = Counter(), 0, 0
    for entry in candidates_data:
        raw = None
        for k in ("retrieved_resources", "candidates", "resources"):
            if isinstance(entry.get(k), list):
                raw, key_used = entry[k], k
                break
        out = []
        seen_resource_ids = set()

        for cand in (raw or []):
            c = dict(cand)

            rid = c.get("resource_id")
            if rid is None:
                raise ValueError(
                    f"Candidate without resource_id for query {entry['query_id']}"
                )

            # Deduplicate by resource_id, preserving the first occurrence
            # (which is the highest-ranked occurrence if the input is ranked).
            if rid in seen_resource_ids:
                continue

            seen_resource_ids.add(rid)

            # Fill modality if missing
            if not c.get(modality_field) and types:
                t = types.get(rid)
                if t:
                    c[modality_field] = t
                    n_filled += 1

            v = str(c.get(modality_field) or "").strip().lower()
            seen[v or "(missing)"] += 1

            if not v:
                n_blank += 1

            out.append(c)

        lookup[entry["query_id"]] = out
    if key_used is None:
        raise SystemExit("no candidate list found in the candidates file; "
                         "expected 'retrieved_resources' or 'candidates'")
    empty = sum(1 for v in lookup.values() if not v)
    print(f"  candidate lists read from '{key_used}'"
          + (f"; {empty} queries have an EMPTY pool" if empty else ""))
    if n_filled:
        print(f"  '{modality_field}' was absent on {n_filled} candidates and "
              f"was filled in from the resource file")
    # Show the RAW values actually present, so an unrecognised vocabulary is
    # visible instead of being folded silently into TEXT.
    print("  resource_type values in the pools (raw -> modality):")
    for v, k in seen.most_common(15):
        m, known = get_modality(v)
        print(f"      {v:<22s} {k:>7d}  -> {m}"
              + ("" if known else "   <-- NOT IN THE MAP"))
    unk = sum(k for v, k in seen.items() if not get_modality(v)[1])
    if unk:
        print(f"  [WARN] {unk} of {sum(seen.values())} pool entries have a type "
              f"outside the map and fold to TEXT.\n         Add the names "
              f"marked above to TEXT_TYPES / VISUAL_TYPES before reporting "
              f"this row.")
    return lookup

def deduplicate_candidates(candidates):
    seen = set()
    result = []

    for c in candidates:
        rid = c["resource_id"]
        if rid in seen:
            continue
        seen.add(rid)
        result.append(c)

    return result

def main():
    p = argparse.ArgumentParser(description="Modality-capped selector for EduExplain ESR")
    p.add_argument("--gold_path", required=True)
    p.add_argument("--candidates_path", required=True)
    p.add_argument("--merged_resources_path", required=True,
                   help="resource JSON carrying `concepts` per resource_id; "
                        "without it Concept Coverage silently scores 0")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--retrieval_name", required=True)
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--top_k", type=int, default=TOP_K)
    p.add_argument("--text_cap", type=int, default=TEXT_CAP)
    p.add_argument("--visual_cap", type=int, default=VISUAL_CAP)
    p.add_argument("--sizes_path", required=True,
                   help="JSON mapping query_id -> selection budget n, the same "
                        "file the other baselines are given")
    p.add_argument("--scale_caps", action="store_true",
                   help="spread the 2:1 text:image ratio over each query's "
                        "budget instead of holding kappa at 2/1, so a budget "
                        "above 3 is reachable")
    p.add_argument("--modality_field", default="resource_type")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.gold_path, encoding="utf-8") as f:
        gold_data = json.load(f)
    with open(args.candidates_path, encoding="utf-8") as f:
        candidates_data = json.load(f)
    with open(args.merged_resources_path, encoding="utf-8") as f:
        merged = json.load(f)
    if isinstance(merged, dict):
        merged = merged.get("resources", merged)
    concepts = {r["resource_id"]: r.get("concepts", []) for r in merged}
    print(f"Loaded concepts for {len(concepts):,} resources.")

    with open(args.sizes_path, encoding="utf-8") as f:
        sizes = json.load(f)
    print(f"Loaded budgets for {len(sizes)} queries.")

    gold_lookup = build_gold_lookup(gold_data)
    res_types = {r["resource_id"]: (r.get("resource_type") or r.get("type"))
                 for r in merged}
    candidate_lookup = build_candidate_lookup(candidates_data, res_types,
                                              args.modality_field)
    meta = {e["query_id"]: e for e in candidates_data}

    qids = sorted(set(gold_lookup) & set(candidate_lookup))
    assert set(gold_lookup) == set(candidate_lookup)
    assert set(gold_lookup) == set(sizes)
    no_budget = [q for q in qids if q not in sizes]
    if no_budget:
        print(f"  [WARN] {len(no_budget)} of {len(qids)} queries have no budget "
              f"and are skipped; averages are over "
              f"{len(qids) - len(no_budget)} queries.")
    print(f"Queries to evaluate: {len(qids) - len(no_budget)}")
    print(f"kappa_text={args.text_cap}  kappa_image={args.visual_cap}  "
          f"n = per-query budget from {os.path.basename(args.sizes_path)}"
          f"{' (caps scaled to n)' if args.scale_caps else ' (caps fixed)'}"
          f"  TopK={args.top_k}")

    preds, mix = [], {"text": 0, "visual": 0}
    short = unknown = no_vis = capped = 0
    dup = reordered = truncated_only = 0
    
    ctl_budget, ctl_matched = [], []      # the two size controls
    for qid in qids:
        if qid not in sizes:
            continue
        n = max(1, int(sizes[qid]))
        kt, kv = caps_for(n, args.text_cap, args.visual_cap, args.scale_caps)
        if kt + kv < n:
            capped += 1
        sel, c = select_modality_cap(
            candidate_lookup[qid], top_k=args.top_k,
            text_cap=kt, visual_cap=kv, max_n=n,
            modality_field=args.modality_field)
        mix["text"] += c["text"]
        mix["visual"] += c["visual"]
        unknown += c["unknown"]
        if len(sel) < n:
            short += 1
            if kv > 0 and c["visual"] < min(kv, 1):
                no_vis += 1
        ids = [r["resource_id"] for r in sel]
        if len(set(ids)) != len(ids):
            dup += 1

        # ---- the control that matters.
        # A cap can change a set in two ways: REORDER it (skip a candidate and
        # reach further down the ranking) or merely TRUNCATE it (stop early
        # because a quota is full or the pool lacks a modality). Truncation
        # alone raises precision mechanically -- a smaller set is a easier
        # denominator -- so a row that only truncates is not a better selector,
        # it is a shorter one. Scoring Top-n at the SAME REALISED SIZE separates
        # the two; if that control matches this row, the cap did nothing.
        ranked = [c["resource_id"] for c in
                  sorted(candidate_lookup[qid],
                         key=lambda c: -float(c.get("score", 0.0)))]
        if ids == ranked[:len(ids)]:
            if len(ids) < n:
                truncated_only += 1
        else:
            reordered += 1
        gs = gold_lookup.get(qid, [])
        ctl_budget.append(compute_best_metrics(set(ranked[:n]), gs))
        ctl_matched.append(compute_best_metrics(set(ranked[:len(ids)]), gs))

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
    print(f"MODALITY-CAPPED  TopK={args.top_k}  "
          f"kappa_text={args.text_cap}  kappa_image={args.visual_cap}  "
          f"budget=given{'  (caps scaled)' if args.scale_caps else ''}")
    print(f"Queries: {len(results)}")
    print("=" * 70)
    for m in ("precision", "recall", "f1", "consistent_f1",
              "concept_coverage", "n_predicted"):
        v = [float(r[m]) for r in results if r.get(m) is not None]
        if v:
            print(f"{m:20s}: {np.mean(v):.4f}")
    print("=" * 70)

    # ---- size controls, printed next to the row they qualify
    def avg(rows, k):
        return sum(r[k] for r in rows) / max(1, len(rows))
    own = {k: sum(float(r[k]) for r in results) / max(1, len(results))
           for k in ("precision", "recall", "f1", "n_predicted")}
    print(f"\n{'control':<38s} {'GS-P':>8s} {'GS-R':>8s} {'GS-F1':>8s} "
          f"{'|R_q|':>7s}")
    print(f"{'this row (modality-capped)':<38s} {own['precision']:>8.4f} "
          f"{own['recall']:>8.4f} {own['f1']:>8.4f} {own['n_predicted']:>7.2f}")
    print(f"{'Top-n at the SAME budget':<38s} {avg(ctl_budget,'precision'):>8.4f} "
          f"{avg(ctl_budget,'recall'):>8.4f} {avg(ctl_budget,'f1'):>8.4f} "
          f"{sum(min(int(sizes[q]), len(candidate_lookup[q])) for q in sizes if q in candidate_lookup)/max(1,len(ctl_budget)):>7.2f}")
    print(f"{'Top-n cut to THIS row size':<38s} "
          f"{avg(ctl_matched,'precision'):>8.4f} {avg(ctl_matched,'recall'):>8.4f} "
          f"{avg(ctl_matched,'f1'):>8.4f} {own['n_predicted']:>7.2f}")
    print("\nIf the third row matches the first, the cap only SHORTENED the "
          "set and changed\nnothing about which resources were picked -- the "
          "precision gain is a size effect,\nnot selection quality. Compare "
          "against the third row, not the second.")
    print(f"\ncap effect   : reordered {reordered}/{len(preds)} queries, "
          f"truncated-only {truncated_only}/{len(preds)}, "
          f"no effect {len(preds) - reordered - truncated_only}/{len(preds)}")
    if dup:
        print(f"[WARN] {dup} queries selected the SAME resource_id twice. The "
              f"evaluator scores a SET,\n       so precision is divided by the "
              f"deduplicated size while n_predicted counts the\n       list -- "
              f"that inflates precision. Deduplicate the candidate pool.")

    tot = max(1, mix["text"] + mix["visual"])
    print(f"modality mix : text {mix['text']/tot:.1%}, "
          f"visual {mix['visual']/tot:.1%}")
    print(f"short sets   : {short}/{len(preds)} queries returned fewer than "
          f"their budget")
    if capped:
        print(f"  cap-limited: {capped} queries have a budget above "
              f"kappa_text+kappa_image={args.text_cap + args.visual_cap} and "
              f"CANNOT reach it.\n               Those are the large-gold-set "
              f"queries, so this row is handicapped exactly where recall is "
              f"hardest.\n               Pass --scale_caps to keep the 2:1 "
              f"ratio and make every budget reachable.")
    if no_vis:
        print(f"  pool-limited: {no_vis} could not fill the image quota -- no "
              f"visual resource in the top-{args.top_k} pool. That, not the "
              f"selector,\n               is why mean |R_q| comes out below "
              f"{n}; it is a first-stage property.")
    if unknown:
        print(f"[WARN] {unknown} candidates SCANNED during selection had a "
              f"resource_type outside the\n       modality map and were "
              f"treated as TEXT. See the raw-value table above.")

    write_results_csv(
        results,
        os.path.join(args.output_dir,
                     f"modality_cap_{args.retrieval_name}_topk{args.top_k}"
                     f"_text{args.text_cap}_vis{args.visual_cap}"
                     f"{'_scaled' if args.scale_caps else ''}"
                     f"_given_budget.csv"),
        extra_meta={"selector": "modality_cap",
                    "retrieval_system": args.retrieval_name,
                    "top_k": args.top_k, "text_cap": args.text_cap,
                    "visual_cap": args.visual_cap, "tau": args.tau,
                    "budget": "query-specific", "sizes_path": args.sizes_path,
                    "scale_caps": args.scale_caps})
    print(f"\nDone. Written to {args.output_dir}")


if __name__ == "__main__":
    main()
