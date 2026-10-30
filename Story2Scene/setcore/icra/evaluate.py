"""Metrics.

Set quality is scored best-match against the m acceptable gold sets. Alongside
it we report the two things the method actually claims: *sufficiency* (does the
returned set cover the concepts the gold sets cover) and *redundancy* (is the
returned set free of overlap). Without a redundancy number "minimal overlap"
stays an aspiration rather than a result.

Gold concept annotations are read here and nowhere else in the inference path.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import numpy as np


def prf(pred: Set[str], gold: Set[str]) -> Dict[str, float]:
    if not pred and not gold:
        return {"p": 1.0, "r": 1.0, "f1": 1.0}
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"p": p, "r": r, "f1": f}


# ===================================================================
# OFFICIAL EduExplain ESR metrics (evaluation/evaluator.py)
# Reimplemented to match the benchmark spec exactly. Differences from the
# generic helpers below:
#   * P, R and F1 are each maximised INDEPENDENTLY over gold sets, so they may
#     come from different gold sets and F1 != harmonic_mean(P, R).
#   * Concept Coverage is measured against CORE concepts (present in >= tau of
#     the gold sets), not the union of all gold concepts.
#   * Recall@K is measured over the full ranked candidate list.
# ===================================================================
def official_prf(pred_ids: Set[str], gold_ids: Set[str]):
    """Dice-form P/R/F1 for one prediction against one gold set."""
    if not pred_ids or not gold_ids:
        return 0.0, 0.0, 0.0
    inter = len(pred_ids & gold_ids)
    denom = len(pred_ids) + len(gold_ids)
    return (inter / len(pred_ids), inter / len(gold_ids),
            (2 * inter / denom) if denom else 0.0)


def official_best_metrics(pred_ids: Set[str],
                          gold_sets: Sequence[Sequence[str]]) -> Dict[str, float]:
    """Each of P, R, F1 maximised independently over the gold sets."""
    if not gold_sets:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "consistent_f1": 0.0,
                "best_precision_gold_idx": -1, "best_recall_gold_idx": -1,
                "best_f1_gold_idx": -1}
    bp = br = bf = 0.0
    ip = ir = if_ = 0
    for i, g in enumerate(gold_sets):
        p, r, f = official_prf(pred_ids, set(g))
        if p > bp:
            bp, ip = p, i
        if r > br:
            br, ir = r, i
        if f > bf:
            bf, if_ = f, i
    tot = bp + br
    return {"precision": bp, "recall": br, "f1": bf,
            "consistent_f1": (2 * bp * br / tot) if tot > 0 else 0.0,
            "best_precision_gold_idx": ip, "best_recall_gold_idx": ir,
            "best_f1_gold_idx": if_}


def core_concepts(gold_concept_sets: Sequence[Sequence[str]],
                  tau: float = 0.8) -> Set[str]:
    """Concepts appearing in at least tau of the gold sets.

    With a single gold set every concept is core (freq = 1.0), by design.
    """
    if not gold_concept_sets:
        return set()
    n = len(gold_concept_sets)
    cnt: Dict[str, int] = {}
    for cs in gold_concept_sets:
        for c in set(cs):
            cnt[c] = cnt.get(c, 0) + 1
    return {c for c, k in cnt.items() if k / n >= tau}


def official_concept_coverage(pred_concepts: Sequence[str],
                              gold_concept_sets: Sequence[Sequence[str]],
                              tau: float = 0.8) -> Dict[str, float]:
    core = core_concepts(gold_concept_sets, tau)
    pred = set(pred_concepts)
    if not core:
        # Undefined; a prediction cannot fail to cover concepts that don't exist.
        return {"concept_coverage": 1.0, "core_concept_count": 0,
                "pred_concept_count": len(pred), "covered_count": 0}
    cov = pred & core
    return {"concept_coverage": len(cov) / len(core),
            "core_concept_count": len(core),
            "pred_concept_count": len(pred), "covered_count": len(cov)}


def core_cc_defined(gold_concept_sets: Sequence[Sequence[str]],
                    tau: float = 0.8) -> bool:
    """Is Core Concept Coverage well posed for this query?

    The paper reports CoreCC only where it is: at least two gold sets AND a
    non-empty core. With a single gold set every concept clears tau trivially,
    so the notion is vacuous there and averaging it in inflates the mean.
    """
    return len(gold_concept_sets) >= 2 and bool(core_concepts(gold_concept_sets, tau))


def core_cc_included(gold_concept_sets: Sequence[Sequence[str]], tau: float,
                     population: str = "defined") -> bool:
    """Whether this query enters the Core CC average, under either convention."""
    if population == "official":
        return True          # benchmark evaluator: every query, 1.0 if no core
    return core_cc_defined(gold_concept_sets, tau)


def set_concept_coverage(pred_concepts: Sequence[str],
                         gold_concept_sets: Sequence[Sequence[str]]) -> Dict[str, float]:
    """SetCC (Eq. 12): per gold set, the fraction of ITS concepts recovered,
    averaged over the query's gold sets.

    Unlike CoreCC there is no cross-set frequency threshold, so SetCC is defined
    on every query. It is the measure to quote when CoreCC's 271-query subset
    would make two methods incomparable.
    """
    if not gold_concept_sets:
        return {"set_concept_coverage": 0.0}
    pred = set(pred_concepts)
    fracs = []
    for cs in gold_concept_sets:
        g = set(cs)
        if g:
            fracs.append(len(pred & g) / len(g))
    return {"set_concept_coverage": float(np.mean(fracs)) if fracs else 0.0}


def attainment(pred_ids: Set[str], gold_sets: Sequence[Sequence[str]],
               f1: float) -> Dict[str, float]:
    """GoldSet F1 divided by the best value the set size allows.

    MaxF1(q) = max_i 2*min(|R_q|,|G_i|) / (|R_q| + |G_i|), which is what a
    system could score if every returned resource were correct. Dividing by it
    separates "picked the wrong resources" from "returned the wrong number",
    the confound that otherwise dominates any comparison at differing set
    sizes. Report it only when all systems share one size operating point.
    """
    n = len(pred_ids)
    if not gold_sets or n == 0:
        return {"max_f1": 0.0, "attainment": 0.0}
    mx = max(2 * min(n, len(g)) / (n + len(g)) for g in gold_sets if g)
    return {"max_f1": mx, "attainment": (f1 / mx) if mx > 0 else 0.0}


def official_recall_at_k(ranked_ids: Sequence[str],
                         gold_sets: Sequence[Sequence[str]],
                         k_values=(25, 50, 100, 200)) -> Dict[str, float]:
    """Relevant = union of resource ids over ALL gold sets."""
    relevant = {x for g in gold_sets for x in g}
    if not relevant:
        return {f"recall@{k}": 0.0 for k in k_values}
    out = {}
    for k in k_values:
        out[f"recall@{k}"] = len(set(ranked_ids[:k]) & relevant) / len(relevant)
    return out


def best_match(pred: Sequence[str], gold_sets: Sequence[Sequence[str]]) -> Dict[str, float]:
    """max over gold sets of F1, with that set's P/R and exact-match flag."""
    if not gold_sets:
        return {"p": 0.0, "r": 0.0, "f1": 0.0, "em": 0.0, "best_i": -1}
    ps = set(pred)
    scored = [(prf(ps, set(g)), i) for i, g in enumerate(gold_sets)]
    m, i = max(scored, key=lambda x: x[0]["f1"])
    return {**m, "em": float(any(ps == set(g) for g in gold_sets)), "best_i": i}


def jaccard(pred: Sequence[str], gold_sets: Sequence[Sequence[str]]) -> float:
    ps = set(pred)
    best = 0.0
    for g in gold_sets:
        gs = set(g)
        u = len(ps | gs)
        if u:
            best = max(best, len(ps & gs) / u)
    return best


# ------------------------------------------------------------- sufficiency
def concept_sufficiency(pred_concepts: List[List[str]],
                        gold_concepts: List[List[str]]) -> Dict[str, float]:
    """Fraction of the gold sets' concept mass the returned set supplies."""
    need: Set[str] = set()
    for cs in gold_concepts:
        need |= set(cs)
    have: Set[str] = set()
    for cs in pred_concepts:
        have |= set(cs)
    if not need:
        return {"concept_recall": 1.0, "concept_precision": 1.0, "n_required": 0}
    return {
        "concept_recall": len(need & have) / len(need),
        "concept_precision": (len(need & have) / len(have)) if have else 0.0,
        "n_required": len(need),
    }


# --------------------------------------------------------------- redundancy
def redundancy(pred_concepts: List[List[str]],
               emb: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Two views of overlap inside R_q.

    concept_redundancy : 1 - |union| / sum|.|   (0 = disjoint, ->1 = duplicated)
    unique_per_resource: |union| / |R_q|
    mean_pairwise_sim  : mean cosine between selected resource embeddings
    """
    n = len(pred_concepts)
    tot = sum(len(c) for c in pred_concepts)
    uni = len({c for cs in pred_concepts for c in cs})
    out = {
        "concept_redundancy": (1.0 - uni / tot) if tot else 0.0,
        "unique_concepts_per_resource": (uni / n) if n else 0.0,
    }
    if emb is not None and len(emb) > 1:
        S = emb @ emb.T
        iu = np.triu_indices(len(emb), k=1)
        out["mean_pairwise_sim"] = float(S[iu].mean())
    else:
        out["mean_pairwise_sim"] = 0.0
    return out


def source_diversity(resources) -> Dict[str, float]:
    """Diversity of a set at the SOURCE level rather than the concept level.

    Gold sets can be redundant in concepts while still being complementary in
    presentation: the same idea from a different textbook, a different grade
    level, or as a figure rather than prose. Concept-redundancy cannot see
    that, so if gold sets score high here and low on concept diversity, the
    coverage objective is optimising the wrong kind of diversity.
    """
    n = len(resources)
    if n == 0:
        return {}
    def frac(vals):
        vals = [v for v in vals if v is not None]
        return (len(set(vals)) / n) if vals else 0.0
    return {
        "distinct_documents_per_resource": frac([r.document_id for r in resources]),
        "distinct_chapters_per_resource": frac([r.chapter_title for r in resources]),
        "distinct_grades_per_resource": frac([r.class_no for r in resources]),
        "distinct_types_per_resource": frac([r.type for r in resources]),
    }


# ---------------------------------------------------------------- diagnostics
def pool_recall(pool_ids: Sequence[str], gold_sets: Sequence[Sequence[str]]) -> Dict[str, float]:
    """First-stage ceiling. Everything downstream is capped by this."""
    ps = set(pool_ids)
    if not gold_sets:
        return {"pool_recall_best": 0.0, "pool_recall_any_complete": 0.0}
    recs = [len(ps & set(g)) / max(1, len(g)) for g in gold_sets]
    return {
        "pool_recall_best": float(max(recs)),
        "pool_recall_any_complete": float(any(set(g) <= ps for g in gold_sets)),
    }


def profile_metrics(pred_c: int, pred_f: int, gold_c: List[int],
                    gold_f: List[int]) -> Dict[str, float]:
    return {
        "count_exact": float(pred_c in gold_c),
        "count_within1": float(any(abs(pred_c - g) <= 1 for g in gold_c)),
        "count_abs_err": float(min(abs(pred_c - g) for g in gold_c)),
        "figure_correct": float(pred_f in gold_f),
        "pred_c": pred_c, "pred_f": pred_f,
    }


def aggregate(rows: List[Dict]) -> Dict[str, float]:
    keys, out = set(), {}
    for r in rows:
        keys |= {k for k, v in r.items() if isinstance(v, (int, float, bool))}
    for k in sorted(keys):
        vals = [float(r[k]) for r in rows if k in r and isinstance(r[k], (int, float, bool))]
        if vals:
            out[k] = float(np.mean(vals))
    out["n_queries"] = len(rows)
    return out


def binary_f1(pred: Sequence[int], gold: Sequence[int]) -> Dict[str, float]:
    pred, gold = np.asarray(pred), np.asarray(gold)
    tp = int(((pred == 1) & (gold == 1)).sum())
    fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "accuracy": float((pred == gold).mean()) if len(gold) else 0.0}
