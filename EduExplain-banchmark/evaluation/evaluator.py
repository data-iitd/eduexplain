"""
evaluation/evaluator.py
========================
EduExplain Explanation Set Retrieval (ESR) evaluation framework.

Implements:
    - GoldSet Precision  (max over gold sets, independent maximization)
    - GoldSet Recall     (max over gold sets, independent maximization)
    - GoldSet F1         (max over gold sets, independent maximization)
    - Concept Coverage   (core concepts at configurable tau threshold)
    - Recall@K           (K = 25, 50, 100, 200)

IMPORTANT: Precision, Recall, F1 are maximized INDEPENDENTLY.
Each metric finds the gold set that maximizes IT specifically.
This means reported F1 is not necessarily harmonic_mean(reported_P, reported_R).
This matches the benchmark spec in Section 4.1 of the paper.

The evaluator does NOT load any JSON files.
All data is passed in already loaded.
"""

import csv
import os
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Type aliases (readability)
# ---------------------------------------------------------------------------
ResourceID   = str
ConceptID    = str
QueryID      = str

# A gold resource dict as it appears in gold_sets.json
GoldResource = dict[str, Any]

# A gold set: list of gold resource dicts
GoldSet = list[GoldResource]

# gold_lookup: query_id -> list of gold sets
# Each gold set is a list of gold resource dicts (already contain concepts)
GoldLookup = dict[QueryID, list[GoldSet]]

# candidate_lookup: query_id -> list of candidate dicts (full dicts, sorted by score)
CandidateLookup = dict[QueryID, list[dict[str, Any]]]

# A prediction: {"query_id": ..., "resources": [candidate_dict, ...]}
Prediction = dict[str, Any]

# Per-query metric result
QueryMetrics = dict[str, Any]


# ---------------------------------------------------------------------------
# build_gold_lookup
# ---------------------------------------------------------------------------

def build_gold_lookup(gold_data: list[dict]) -> GoldLookup:
    """
    Convert raw gold_sets.json data into a lookup dict.

    gold_data: list of objects, each with keys:
        query_id    str
        gold_sets   list of gold set objects, each with:
            resources_used  list of resource dicts (each has resource_id, concepts, ...)
            answer          str
            rationale       str

    Returns:
        { query_id: [ [gold_resource, ...], [gold_resource, ...], ... ] }

    Each inner list is one complete gold set (list of gold resource dicts).
    Sanity checks: warns if a query has zero gold sets or any gold set is empty.
    """
    lookup: GoldLookup = {}

    for entry in gold_data:
        qid = entry["query_id"]
        raw_gold_sets = entry.get("gold_sets", [])

        if not raw_gold_sets:
            print(f"[WARN] build_gold_lookup: query {qid} has zero gold sets")
            lookup[qid] = []
            continue

        parsed: list[GoldSet] = []
        for gs_idx, gs in enumerate(raw_gold_sets):
            resources = gs.get("resources_used", [])
            if not resources:
                print(f"[WARN] build_gold_lookup: query {qid} gold set {gs_idx} is empty")
            parsed.append(resources)

        lookup[qid] = parsed

    return lookup


# ---------------------------------------------------------------------------
# compute_prf  (for one prediction against one gold set)
# ---------------------------------------------------------------------------

def compute_prf(
    pred_ids: set[ResourceID],
    gold_ids: set[ResourceID],
) -> tuple[float, float, float]:
    """
    Compute Precision, Recall, and F1 for one prediction vs one gold set.

    Precision = |Pred ∩ Gold| / |Pred|
    Recall    = |Pred ∩ Gold| / |Gold|
    F1        = 2|Pred ∩ Gold| / (|Pred| + |Gold|)

    F1 here is the set-overlap Dice formulation, consistent with the paper spec.
    It equals 2*P*R/(P+R) when P and R are computed from the same intersection,
    which is always the case within a single gold set comparison.

    Returns (precision, recall, f1). All values in [0, 1].
    Handles edge cases: empty prediction or empty gold set returns 0.0 for
    the affected metric.
    """
    if not pred_ids or not gold_ids:
        return 0.0, 0.0, 0.0

    intersection = len(pred_ids & gold_ids)
    precision    = intersection / len(pred_ids)
    recall       = intersection / len(gold_ids)
    denom        = len(pred_ids) + len(gold_ids)
    f1           = (2 * intersection / denom) if denom > 0 else 0.0

    return precision, recall, f1


# ---------------------------------------------------------------------------
# compute_best_metrics
# ---------------------------------------------------------------------------

def compute_best_metrics(
    pred_ids: set[ResourceID],
    gold_sets: list[GoldSet],
) -> dict[str, float]:
    """
    Compute best Precision, Recall, and F1 over all gold sets.

    CRITICALLY: each metric is maximized INDEPENDENTLY.
    best_precision = max(precision_i  for each gold set i)
    best_recall    = max(recall_i     for each gold set i)
    best_f1        = max(f1_i         for each gold set i)

    This means the three metrics may come from different gold sets.
    The reported F1 is NOT necessarily harmonic_mean(best_P, best_R).
    This is intentional and matches the benchmark spec.

    Also computes consistent_f1 = 2*P*R/(P+R) using best_precision and
    best_recall, for reference in the results table.

    Returns dict with keys:
        precision, recall, f1, consistent_f1,
        best_precision_gold_idx, best_recall_gold_idx, best_f1_gold_idx
    """
    if not gold_sets:
        return {
            "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "consistent_f1": 0.0,
            "best_precision_gold_idx": -1,
            "best_recall_gold_idx": -1,
            "best_f1_gold_idx": -1,
        }

    best_precision      = 0.0
    best_recall         = 0.0
    best_f1             = 0.0
    best_p_idx          = 0
    best_r_idx          = 0
    best_f1_idx         = 0

    for gs_idx, gold_set in enumerate(gold_sets):
        gold_ids = {r["resource_id"] for r in gold_set}
        p, r, f  = compute_prf(pred_ids, gold_ids)

        if p > best_precision:
            best_precision = p
            best_p_idx     = gs_idx
        if r > best_recall:
            best_recall = r
            best_r_idx  = gs_idx
        if f > best_f1:
            best_f1    = f
            best_f1_idx = gs_idx

    # Consistent F1: harmonic mean of independently-maximized P and R
    pr_sum        = best_precision + best_recall
    consistent_f1 = (2 * best_precision * best_recall / pr_sum) if pr_sum > 0 else 0.0

    return {
        "precision":             best_precision,
        "recall":                best_recall,
        "f1":                    best_f1,
        "consistent_f1":         consistent_f1,
        "best_precision_gold_idx": best_p_idx,
        "best_recall_gold_idx":    best_r_idx,
        "best_f1_gold_idx":        best_f1_idx,
    }


# ---------------------------------------------------------------------------
# compute_core_concepts
# ---------------------------------------------------------------------------

def compute_core_concepts(
    gold_sets: list[GoldSet],
    tau: float = 0.8,
) -> set[ConceptID]:
    """
    Compute core concepts for a query.

    For each concept, count in how many gold sets it appears.
    freq(c) = |{gold sets containing c}| / |total gold sets|
    Core concepts = { c : freq(c) >= tau }

    Special case: if there is only one gold set, all its concepts are core
    (freq = 1.0 >= any tau in (0,1]). This is by design.

    Args:
        gold_sets: list of gold sets, each a list of gold resource dicts
        tau:       frequency threshold (default 0.8)

    Returns:
        set of concept IDs that are core for this query
    """
    if not gold_sets:
        return set()

    num_gold_sets = len(gold_sets)

    # Count how many gold sets contain each concept
    concept_gold_set_count: dict[ConceptID, int] = defaultdict(int)

    for gold_set in gold_sets:
        # Collect concepts appearing in this gold set (deduplicated)
        concepts_in_this_gs: set[ConceptID] = set()
        for resource in gold_set:
            for c in resource.get("concepts", []):
                concepts_in_this_gs.add(c)

        # Increment count for each concept appearing in this gold set
        for c in concepts_in_this_gs:
            concept_gold_set_count[c] += 1

    # Apply threshold
    core: set[ConceptID] = {
        c for c, count in concept_gold_set_count.items()
        if (count / num_gold_sets) >= tau
    }

    return core


# ---------------------------------------------------------------------------
# compute_concept_coverage
# ---------------------------------------------------------------------------

def compute_concept_coverage(
    pred_resources: list[dict[str, Any]],
    gold_sets: list[GoldSet],
    tau: float = 0.8,
) -> dict[str, Any]:
    """
    Compute concept coverage of the prediction against core concepts.

    Coverage = |Pred_concepts ∩ Core_concepts| / |Core_concepts|

    Pred_concepts = union of concepts from all predicted resources.
    Core_concepts = concepts appearing in >= tau fraction of gold sets.

    Returns dict with keys:
        concept_coverage    float in [0, 1]
        core_concept_count  int
        pred_concept_count  int
        covered_count       int
        tau                 float
    """
    core_concepts = compute_core_concepts(gold_sets, tau=tau)

    # Collect concepts from prediction
    pred_concepts: set[ConceptID] = set()
    for r in pred_resources:
        for c in r.get("concepts", []):
            pred_concepts.add(c)

    if not core_concepts:
        # No core concepts: coverage is undefined; treat as 1.0
        # (prediction cannot fail to cover concepts that don't exist)
        return {
            "concept_coverage":   1.0,
            "core_concept_count": 0,
            "pred_concept_count": len(pred_concepts),
            "covered_count":      0,
            "tau":                tau,
        }

    covered      = pred_concepts & core_concepts
    coverage     = len(covered) / len(core_concepts)

    return {
        "concept_coverage":   coverage,
        "core_concept_count": len(core_concepts),
        "pred_concept_count": len(pred_concepts),
        "covered_count":      len(covered),
        "tau":                tau,
    }


# ---------------------------------------------------------------------------
# compute_recall_at_k
# ---------------------------------------------------------------------------

def compute_recall_at_k(
    all_candidates: list[dict[str, Any]],
    gold_sets: list[GoldSet],
    k_values: list[int] = (25, 50, 100, 200),
) -> dict[str, float]:
    """
    Compute Recall@K for K in k_values.

    Relevant resources = UNION of resource_ids across ALL gold sets.
    This means a resource is relevant if it appears in ANY gold set.

    all_candidates: full ranked list of candidate dicts for this query,
                    sorted by retrieval score (descending).
                    These are the Top-200 candidates from the candidate file.

    Recall@K = |Top-K candidates ∩ Relevant| / |Relevant|

    Returns dict: { "recall@25": float, "recall@50": float, ... }
    """
    # Build relevant set = union over all gold sets
    relevant: set[ResourceID] = set()
    for gold_set in gold_sets:
        for r in gold_set:
            relevant.add(r["resource_id"])

    if not relevant:
        return {f"recall@{k}": 0.0 for k in k_values}

    results: dict[str, float] = {}
    candidate_ids = [c["resource_id"] for c in all_candidates]

    for k in k_values:
        top_k_ids   = set(candidate_ids[:k])
        hits        = top_k_ids & relevant
        results[f"recall@{k}"] = len(hits) / len(relevant)

    return results


# ---------------------------------------------------------------------------
# evaluate  (main entry point)
# ---------------------------------------------------------------------------

def evaluate(
    predictions: list[Prediction],
    candidate_lookup: CandidateLookup,
    gold_lookup: GoldLookup,
    tau: float = 0.8,
    k_values: list[int] = (25, 50, 100, 200),
) -> list[QueryMetrics]:
    """
    Run all metrics for every prediction.

    Args:
        predictions:       list of prediction dicts
                           { "query_id": str, "resources": [candidate_dict, ...] }
        candidate_lookup:  query_id -> full ranked candidate list (Top-200)
        gold_lookup:       query_id -> list of gold sets
        tau:               core concept threshold for Concept Coverage
        k_values:          K values for Recall@K

    Returns:
        list of per-query metric dicts, one per prediction.
        Does NOT average — averaging is the caller's responsibility.

    Sanity checks:
        - Warns if a query_id in predictions is missing from gold_lookup.
        - Warns if a query_id in predictions is missing from candidate_lookup.
        - Warns if a prediction has zero resources.
    """
    results: list[QueryMetrics] = []

    for pred in predictions:
        qid           = pred["query_id"]
        pred_resources = pred.get("resources", [])

        # Sanity checks
        if qid not in gold_lookup:
            print(f"[WARN] evaluate: query {qid} not found in gold_lookup — skipping")
            continue
        if qid not in candidate_lookup:
            print(f"[WARN] evaluate: query {qid} not found in candidate_lookup — skipping")
            continue
        if not pred_resources:
            print(f"[WARN] evaluate: query {qid} has zero predicted resources")

        gold_sets     = gold_lookup[qid]
        all_candidates = candidate_lookup[qid]
        pred_ids      = {r["resource_id"] for r in pred_resources}

        # --- GoldSet Precision / Recall / F1 ---
        prf_metrics = compute_best_metrics(pred_ids, gold_sets)

        # --- Concept Coverage ---
        cc_metrics = compute_concept_coverage(pred_resources, gold_sets, tau=tau)

        # --- Recall@K ---
        rk_metrics = compute_recall_at_k(all_candidates, gold_sets, k_values=k_values)

        query_result: QueryMetrics = {
            "query_id":               qid,
            "n_predicted":            len(pred_resources),
            "n_gold_sets":            len(gold_sets),
            # GoldSet metrics
            "precision":              prf_metrics["precision"],
            "recall":                 prf_metrics["recall"],
            "f1":                     prf_metrics["f1"],
            "consistent_f1":          prf_metrics["consistent_f1"],
            "best_precision_gold_idx": prf_metrics["best_precision_gold_idx"],
            "best_recall_gold_idx":    prf_metrics["best_recall_gold_idx"],
            "best_f1_gold_idx":        prf_metrics["best_f1_gold_idx"],
            # Concept Coverage
            "concept_coverage":       cc_metrics["concept_coverage"],
            "core_concept_count":     cc_metrics["core_concept_count"],
            "pred_concept_count":     cc_metrics["pred_concept_count"],
            "covered_count":          cc_metrics["covered_count"],
            "tau":                    cc_metrics["tau"],
        }

        # Merge Recall@K into result
        query_result.update(rk_metrics)

        results.append(query_result)

    return results


# ---------------------------------------------------------------------------
# write_results_csv
# ---------------------------------------------------------------------------

def write_results_csv(
    results: list[QueryMetrics],
    output_path: str,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """
    Write per-query metrics to a CSV file.

    Args:
        results:      list of per-query metric dicts from evaluate()
        output_path:  path to write CSV
        extra_meta:   optional dict of extra columns to append to every row
                      (e.g. {"selector": "top_n", "top_k": 50, "n": 3})

    Creates parent directories if they don't exist.
    Overwrites existing file.
    """
    if not results:
        print(f"[WARN] write_results_csv: no results to write to {output_path}")
        return

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Determine column order: metric keys first, then extra_meta keys
    metric_keys = list(results[0].keys())
    meta_keys   = list(extra_meta.keys()) if extra_meta else []
    fieldnames  = metric_keys + [k for k in meta_keys if k not in metric_keys]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            full_row = dict(row)
            if extra_meta:
                full_row.update(extra_meta)
            writer.writerow(full_row)

    print(f"  Written: {output_path}  ({len(results)} rows)")
