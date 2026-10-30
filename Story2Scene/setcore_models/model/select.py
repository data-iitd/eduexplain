"""Constrained set selection.

Objective :   max_R  ConceptCov(R) + beta * sum_{r in R} s(q,r)
subject to           |{r in R : type(r) = t}| <= pi_q[t]   for all t

"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .concepts import ConceptSignal


# ------------------------------------------------------------------ coverage
class CoverageState:
    """Incrementally maintained coverage vector over U_q.
    """

    def __init__(self, P: np.ndarray, w: np.ndarray, mode: str = "graded_max"):
        self.mode = mode
        self.w = w.astype(np.float64)
        self.P = (P > 0).astype(np.float64) if mode == "binary" else P.astype(np.float64)
        self.cur = np.zeros(P.shape[1], dtype=np.float64)   # binary/graded_max
        self.prod = np.ones(P.shape[1], dtype=np.float64)   # noisy_or

    def gain(self, idx: np.ndarray) -> np.ndarray:
        """Marginal coverage gain (Eq. 9, first term) for each candidate in idx."""
        Pi = self.P[idx]
        if self.mode == "noisy_or":
            new = 1.0 - self.prod[None, :] * (1.0 - Pi)
            delta = new - (1.0 - self.prod)[None, :]
        else:
            delta = np.maximum(Pi - self.cur[None, :], 0.0)
        return delta @ self.w

    def add(self, i: int) -> None:
        if self.mode == "noisy_or":
            self.prod *= (1.0 - self.P[i])
        else:
            self.cur = np.maximum(self.cur, self.P[i])

    def value(self) -> float:
        c = (1.0 - self.prod) if self.mode == "noisy_or" else self.cur
        return float(c @ self.w)


def coverage_of(P: np.ndarray, w: np.ndarray, sel: Sequence[int], mode: str) -> float:
    st = CoverageState(P, w, mode)
    for i in sel:
        st.add(i)
    return st.value()


def objective(P, w, s, sel, mode, beta, relevance="linear") -> float:
    rel = float(np.sum(s[list(sel)])) if len(sel) else 0.0
    if relevance == "sqrt":
        rel = float(np.sqrt(rel))
    return coverage_of(P, w, sel, mode) + beta * rel


# ------------------------------------------------------------------ relevance
def normalise_scores(s: np.ndarray) -> np.ndarray:
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-9:
        return np.ones_like(s) * 0.5
    return (s - lo) / (hi - lo)


def _rel_gain(s: np.ndarray, idx: np.ndarray, acc: float, relevance: str) -> np.ndarray:
    if relevance == "sqrt":
        # Saturating relevance keeps late slots from collapsing to top-k.
        return np.sqrt(acc + s[idx]) - np.sqrt(acc)
    return s[idx]


# -------------------------------------------------------------------- result
@dataclass
class Selection:
    indices: List[int]
    order: List[Tuple[int, str, float]]      # (index, type, marginal gain)
    objective: float
    coverage: float
    slots_requested: Dict[str, int]
    slots_filled: Dict[str, int]
    stopped_early: bool = False


# -------------------------------------------------------------------- greedy
def greedy_select(types: Sequence[str], s: np.ndarray, sig: ConceptSignal,
                  profile: Dict[str, int], cfg) -> Selection:
    s = normalise_scores(s) if cfg.normalize_relevance else s
    types = np.asarray(types)
    st = CoverageState(sig.P, sig.w, cfg.coverage)
    if not getattr(cfg, "type_constraint", True):
        types = np.array(["_any"] * len(types))
        profile = {"_any": int(sum(profile.values()))}
    remaining = {t: int(profile.get(t, 0)) for t in profile}
    chosen: List[int] = []
    order: List[Tuple[int, str, float]] = []
    acc_rel, stopped = 0.0, False
    total = sum(remaining.values())

    for _ in range(total):
        feasible = np.array(
            [i for i in range(len(types))
             if i not in chosen and remaining.get(types[i], 0) > 0], dtype=int)
        if feasible.size == 0:
            break
        g = st.gain(feasible) + cfg.beta * _rel_gain(s, feasible, acc_rel, cfg.relevance)
        j = int(np.argmax(g))
        if cfg.stop_on_nonpositive and g[j] <= 0:
            stopped = True
            break
        i = int(feasible[j])
        chosen.append(i)
        order.append((i, str(types[i]), float(g[j])))
        st.add(i)
        acc_rel += float(s[i])
        remaining[types[i]] -= 1

    filled = {t: sum(1 for i in chosen if types[i] == t) for t in profile}
    return Selection(chosen, order,
                     objective(sig.P, sig.w, s, chosen, cfg.coverage, cfg.beta, cfg.relevance),
                     st.value(), dict(profile), filled, stopped)


def greedy_select_block(types: Sequence[str], s: np.ndarray, sig: ConceptSignal,
                        profile: Dict[str, int], cfg) -> Selection:
    s = normalise_scores(s) if cfg.normalize_relevance else s
    types = np.asarray(types)
    st = CoverageState(sig.P, sig.w, cfg.coverage)
    if not getattr(cfg, "type_constraint", True):
        types = np.array(["_any"] * len(types))
        profile = {"_any": int(sum(profile.values()))}
    chosen: List[int] = []
    order: List[Tuple[int, str, float]] = []
    acc_rel, stopped = 0.0, False

    for t in profile:
        pool = np.array([i for i in range(len(types)) if types[i] == t], dtype=int)
        for _ in range(int(profile.get(t, 0))):
            cand = np.array([i for i in pool if i not in chosen], dtype=int)
            if cand.size == 0:
                break
            g = st.gain(cand) + cfg.beta * _rel_gain(s, cand, acc_rel, cfg.relevance)
            j = int(np.argmax(g))
            if cfg.stop_on_nonpositive and g[j] <= 0:
                stopped = True
                break
            i = int(cand[j])
            chosen.append(i); order.append((i, str(t), float(g[j])))
            st.add(i); acc_rel += float(s[i])

    filled = {t: sum(1 for i in chosen if types[i] == t) for t in profile}
    return Selection(chosen, order,
                     objective(sig.P, sig.w, s, chosen, cfg.coverage, cfg.beta, cfg.relevance),
                     st.value(), dict(profile), filled, stopped)


def select(types, s, sig, profile, cfg) -> Selection:
    fn = greedy_select_block if cfg.greedy == "block" else greedy_select
    sel = fn(types, s, sig, profile, cfg)
    if cfg.backfill == "explanation":
        short = sum(sel.slots_requested.values()) - len(sel.indices)
        if short > 0:
            p2 = {"explanation": sel.slots_filled.get("explanation", 0) + short}
            sel = fn(types, s, sig, p2, cfg)
    return sel


# --------------------------------------------------------- optimality gap
def exact_select(types: Sequence[str], s: np.ndarray, sig: ConceptSignal,
                 profile: Dict[str, int], cfg, topn: int = 25) -> Tuple[List[int], float]:
    s_n = normalise_scores(s) if cfg.normalize_relevance else s
    types = np.asarray(types)
    per_type = {}
    for t, k in profile.items():
        pool = [i for i in range(len(types)) if types[i] == t]
        pool = sorted(pool, key=lambda i: -s_n[i])[:topn]
        per_type[t] = list(itertools.combinations(pool, min(int(k), len(pool))))
        if not per_type[t]:
            per_type[t] = [()]
    best, best_v = [], -np.inf
    for combo in itertools.product(*[per_type[t] for t in profile]):
        sel = [i for part in combo for i in part]
        v = objective(sig.P, sig.w, s_n, sel, cfg.coverage, cfg.beta, cfg.relevance)
        if v > best_v:
            best, best_v = sel, v
    return best, float(best_v)


def select_topn(types, s, emb, n: int, **kw) -> Selection:
    """Top-n: the highest-ranked candidates. The control the method must beat."""
    idx = list(np.argsort(-np.asarray(s))[:n])
    return Selection([int(i) for i in idx], [], 0.0, 0.0, {"topn": n},
                     {"topn": len(idx)})


def select_modality_capped(types, s, emb, n: int, fig_cap: int = 1, **kw) -> Selection:
    """Modality-Capped: top-n under a per-modality quota.
    """
    types = np.asarray(types)
    order = np.argsort(-np.asarray(s))
    out, n_fig = [], 0
    for i in order:
        if len(out) >= n:
            break
        if types[i] == "figure":
            if n_fig >= fig_cap:
                continue
            n_fig += 1
        out.append(int(i))
    return Selection(out, [], 0.0, 0.0, {"n": n, "fig_cap": fig_cap},
                     {"figure": n_fig, "explanation": len(out) - n_fig})


def select_mmr(types, s, emb, n: int, lam: float = 0.5,
               modality_aware: bool = False, mod_penalty: float = 0.5,
               **kw) -> Selection:
    """Maximum Marginal Relevance: relevance traded against pairwise redundancy.
    """
    s = normalise_scores(np.asarray(s))
    E = np.asarray(emb)
    types = np.asarray(types)
    S = E @ E.T
    out: List[int] = []
    for _ in range(n):
        cand = [i for i in range(len(s)) if i not in out]
        if not cand:
            break
        if not out:
            g = s[cand]
        else:
            red = S[np.ix_(cand, out)].max(axis=1)
            g = lam * s[cand] - (1 - lam) * red
            if modality_aware:
                chosen_t = types[out]
                share = np.array([(chosen_t == types[i]).mean() for i in cand])
                g = g - mod_penalty * share
        out.append(int(cand[int(np.argmax(g))]))
    filled = {t: int((types[out] == t).sum()) for t in ("explanation", "figure")} \
        if out else {}
    return Selection(out, [], 0.0, 0.0, {"n": n}, filled)


CONTROLS = {
    "topn": select_topn,
    "modality_capped": select_modality_capped,
    "mmr": select_mmr,
    "mmr_modality": lambda *a, **k: select_mmr(*a, modality_aware=True, **k),
}
