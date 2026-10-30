"""The concept layer.

Two objects are needed by the coverage term:
  * Con(r)  -- which vocabulary concepts each candidate covers, built from the
               candidate's stored summary/description ONLY.
  * U_q     -- which concepts the query requires, built from the query ONLY
               (never from the retrieved pool), so coverage cannot be
               trivially satisfied by whatever the retriever returned.

"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .data import Resource, Vocabulary


@dataclass
class ConceptSignal:
    """Per-query ground set handed to the selector.

    units : identifiers of the concept units in U_q
    w     : |U_q| weights (>= 0)
    P     : |C_q| x |U_q| coverage strengths in [0,1].
            Binary coverage uses P rounded to {0,1}; graded modes use it raw.
    """
    units: List[str]
    w: np.ndarray
    P: np.ndarray

    def __len__(self):
        return len(self.units)


# --------------------------------------------------------------- Con(r)
def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def build_concept_matrix(res_emb: np.ndarray, vocab_emb: np.ndarray, cfg) -> np.ndarray:
    """P_full[r, v] in [0,1]: strength with which resource r covers concept v.

    Column-wise z-scoring (cfg.tag_zscore) corrects for hubness: some concept
    descriptors sit near everything in embedding space and would otherwise be
    tagged on most of the corpus, contributing coverage mass without
    discriminating between candidate sets.
    """
    S = res_emb @ vocab_emb.T                      # cosine, both L2-normalised
    if cfg.tag_zscore:
        mu = S.mean(axis=0, keepdims=True)
        sd = S.std(axis=0, keepdims=True) + 1e-6
        Z = (S - mu) / sd
        P = 1.0 / (1.0 + np.exp(-Z))               # squash to [0,1]
        zmax = np.maximum(Z.max(axis=1, keepdims=True), 1e-6)
        gate = (Z >= cfg.tag_rel * zmax) & (Z >= cfg.tag_z_threshold)
    else:
        P = np.clip((S + 1.0) / 2.0, 0.0, 1.0)
        gate = P >= cfg.tag_threshold

    if cfg.tag_topm and cfg.tag_topm < P.shape[1]:
        keep = np.argsort(-P, axis=1)[:, : cfg.tag_topm]
        mask = np.zeros_like(P, dtype=bool)
        np.put_along_axis(mask, keep, True, axis=1)
        gate &= mask
    return np.where(gate, P, 0.0).astype(np.float32)


def annotated_concept_matrix(resources, vocab, P_embed=None, mode="annotated"):
    """Con(r) read from the corpus `concepts` annotation instead of predicted.

    The annotation is resource-side corpus metadata, available to any
    system at indexing time -- it is not the query-level `concepts` field,
    which stays evaluation-only.

    mode="both" takes the elementwise max, i.e. the annotation is trusted
    absolutely and the embedding tagger may only add concepts on top.
    """
    from .data import slugify
    pos = {c: j for j, c in enumerate(vocab.ids)}
    P = np.zeros((len(resources), len(vocab.ids)), dtype=np.float32)
    for i, r in enumerate(resources):
        for c in (r.gold_concepts or []):
            j = pos.get(slugify(c))
            if j is not None:
                P[i, j] = 1.0
    if mode == "both" and P_embed is not None:
        P = np.maximum(P, P_embed)
    return P


def learned_concept_matrix(PROB: np.ndarray, cfg,
                           target_size: Optional[float] = None) -> np.ndarray:
    """Sparsify a classifier's |R| x |V| probability matrix into Con(r).

    con_learned_size="topm"  keep the top-m concepts above the floor, with m =
                             con_learned_topm (or tag_topm), so the learned and
                             embedding taggers produce Con(r) of the same size
                             and the comparison is not a sparsity confound.
    con_learned_size="match" keep the top-ceil(target_size) above the floor,
                             where target_size is the mean |annotation| -- the
                             learned tagger then matches the ANNOTATION's
                             sparsity instead of the embedding tagger's.

    """
    m = cfg.con_learned_topm or cfg.tag_topm
    if cfg.con_learned_size == "match" and target_size:
        m = max(1, int(np.ceil(target_size)))
    gate = PROB >= cfg.con_learned_tau
    if m and m < PROB.shape[1]:
        keep = np.argpartition(-PROB, m - 1, axis=1)[:, :m]
        mask = np.zeros_like(PROB, dtype=bool)
        np.put_along_axis(mask, keep, True, axis=1)
        gate &= mask
    empty = ~gate.any(axis=1)
    if empty.any():
        top1 = PROB[empty].argmax(axis=1)
        gate[np.nonzero(empty)[0], top1] = True
    return np.where(gate, PROB, 0.0).astype(np.float32)


def concept_sets(P_full: np.ndarray, vocab: Vocabulary) -> List[List[str]]:
    """Con(r) as explicit id lists (for logging / the binary covering relation)."""
    return [[vocab.ids[j] for j in np.nonzero(P_full[i])[0]] for i in range(P_full.shape[0])]


# ------------------------------------------------------------------- U_q
def _apply_weights(a: np.ndarray, df: Optional[np.ndarray], cfg, n_res: int) -> np.ndarray:
    if cfg.w_mode == "uniform":
        w = np.ones_like(a)
    elif cfg.w_mode == "aq":
        w = np.clip(a, 1e-6, None)
    elif cfg.w_mode == "aq_idf":
        idf = np.log(n_res / (1.0 + (df if df is not None else np.ones_like(a))))
        w = np.clip(a, 1e-6, None) * np.clip(idf, 0.0, None)
    else:
        raise ValueError(cfg.w_mode)
    if cfg.normalize_weights and w.sum() > 0:
        w = w / w.sum()
    return w.astype(np.float32)


def uq_match(q_emb: np.ndarray, vocab_emb: np.ndarray, P_pool: np.ndarray,
             vocab: Vocabulary, cfg, df: Optional[np.ndarray] = None,
             n_res: int = 1) -> ConceptSignal:
    """ query-to-vocabulary matching, U_q = {v : a_q(v) >= gamma}.

    With gamma_relative the cutoff is gamma * max_v a_q(v). A fixed absolute
    cosine cutoff is brittle because cosine magnitudes drift with query and
    descriptor length; uq_min guarantees U_q is never empty (an empty ground
    set would silently reduce the whole method to top-k by relevance).
    """
    a = vocab_emb @ q_emb                                   # a_q(v)
    if cfg.uq_abs_tau > 0:
        idx = np.nonzero(a >= cfg.uq_abs_tau)[0]
    else:
        thr = cfg.gamma * float(a.max()) if cfg.gamma_relative else cfg.gamma
        idx = np.nonzero(a >= thr)[0]
    if len(idx) < cfg.uq_min:
        idx = np.argsort(-a)[: cfg.uq_min]
    if len(idx) > cfg.kappa:                                # |U_q| <= kappa
        idx = idx[np.argsort(-a[idx])[: cfg.kappa]]
    idx = np.sort(idx)
    w = _apply_weights(a[idx], None if df is None else df[idx], cfg, n_res)
    return ConceptSignal([vocab.ids[j] for j in idx], w, P_pool[:, idx].copy())


def uq_learned(rho: np.ndarray, vocab: Vocabulary, P_pool: np.ndarray, cfg,
               a: Optional[np.ndarray] = None, df: Optional[np.ndarray] = None,
               n_res: int = 1) -> ConceptSignal:
    """Learned concept head: U_q = {v : rho_q(v) >= gamma'}.

    """
    if cfg.concept_topk > 0:
        k = min(cfg.concept_topk, len(rho))
        idx = np.argpartition(-rho, k - 1)[:k]
        idx = np.sort(idx)
        base = rho[idx] if a is None else a[idx]
        w = _apply_weights(base, None if df is None else df[idx], cfg, n_res)
        return ConceptSignal([vocab.ids[j] for j in idx], w, P_pool[:, idx].copy())
    idx = np.nonzero(rho >= cfg.gamma_learned)[0]
    if len(idx) < cfg.uq_min:
        idx = np.argsort(-rho)[: cfg.uq_min]
    if len(idx) > cfg.kappa:
        idx = idx[np.argsort(-rho[idx])[: cfg.kappa]]
    idx = np.sort(idx)
    base = rho[idx] if a is None else a[idx]
    w = _apply_weights(base, None if df is None else df[idx], cfg, n_res)
    return ConceptSignal([vocab.ids[j] for j in idx], w, P_pool[:, idx].copy())


_STOP = set("""a an the of in on for to and or is are was were be been being with without
by from as at into about over under this that these those it its do does did how what why
when which who whom can could should would may might will shall not no if then than there
here your you we they he she i me my our their explain describe give tell show define""".split())


def noun_phrases(text: str, max_phrases: int = 12) -> List[str]:
    """Cheap phrase chunker for the query-anchored baseline.

    Splits on stopwords/punctuation and keeps the surviving 1-3 word runs.
    Uses spaCy noun_chunks when spaCy is importable with a loaded model.
    """
    try:  # optional, better chunks
        import spacy  # type: ignore
        nlp = noun_phrases._nlp  # type: ignore[attr-defined]
        if nlp is None:
            raise RuntimeError
        return [c.text.strip() for c in nlp(text).noun_chunks][:max_phrases]
    except Exception:
        pass
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-']+", (text or "").lower())
    runs, cur = [], []
    for t in toks:
        if t in _STOP or len(t) <= 2:
            if cur:
                runs.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        runs.append(cur)
    out, seen = [], set()
    for run in runs:
        for n in (3, 2, 1):
            for i in range(len(run) - n + 1):
                p = " ".join(run[i:i + n])
                if p not in seen:
                    seen.add(p)
                    out.append(p)
    return out[:max_phrases]


noun_phrases._nlp = None  # type: ignore[attr-defined]


def uq_anchored(query_text: str, pool_emb: np.ndarray, encoder, cfg,
                sim_threshold: float = 0.5) -> ConceptSignal:
    """Baseline: ground set is the query's own noun phrases.

    """
    phrases = noun_phrases(query_text, max_phrases=cfg.kappa)
    if not phrases:
        phrases = [query_text[:64] or "query"]
    pe = encoder.encode(phrases)                 # |U| x d
    S = pool_emb @ pe.T                          # |C_q| x |U|, cosine
    keep = S >= cfg.anchored_rel * np.maximum(S.max(axis=0, keepdims=True), 1e-6)
    if cfg.anchored_cos > 0:
        keep &= S >= cfg.anchored_cos
    P = np.where(keep, np.clip((S + 1.0) / 2.0, 0.0, 1.0), 0.0).astype(np.float32)
    w = np.ones(len(phrases), dtype=np.float32)
    if cfg.normalize_weights:
        w = w / w.sum()
    return ConceptSignal(phrases, w, P)


def uq_prf(q_emb: np.ndarray, pool_emb: np.ndarray, s: np.ndarray,
           vocab_emb: np.ndarray, P_pool: np.ndarray, vocab: Vocabulary, cfg,
           df: Optional[np.ndarray] = None, n_res: int = 1) -> ConceptSignal:
    """Pseudo-relevance feedback: match V against the query blended with the
    centroid of the top-n first-stage hits.
    """
    n = min(cfg.prf_docs, len(s))
    top = np.argsort(-s)[:n]
    centroid = pool_emb[top].mean(axis=0)
    blend = (1.0 - cfg.prf_alpha) * q_emb + cfg.prf_alpha * centroid
    blend = blend / (np.linalg.norm(blend) + 1e-9)
    a = vocab_emb @ blend
    thr = cfg.gamma * float(a.max()) if cfg.gamma_relative else cfg.gamma
    idx = np.nonzero(a >= thr)[0]
    if len(idx) < cfg.uq_min:
        idx = np.argsort(-a)[: cfg.uq_min]
    if len(idx) > cfg.kappa:
        idx = idx[np.argsort(-a[idx])[: cfg.kappa]]
    idx = np.sort(idx)
    w = _apply_weights(a[idx], None if df is None else df[idx], cfg, n_res)
    return ConceptSignal([vocab.ids[j] for j in idx], w, P_pool[:, idx].copy())


def uq_from_units(unit_ids: Sequence[str], vocab: Vocabulary, P_pool: np.ndarray,
                  cfg, a: Optional[np.ndarray] = None,
                  df: Optional[np.ndarray] = None, n_res: int = 1) -> ConceptSignal:
    """Build a ConceptSignal from an explicit list of vocabulary concept ids."""
    pos = {c: i for i, c in enumerate(vocab.ids)}
    idx = np.sort(np.array([pos[c] for c in unit_ids if c in pos], dtype=int))
    if len(idx) == 0:
        idx = np.arange(min(cfg.uq_min, P_pool.shape[1]))
    base = a[idx] if a is not None else np.ones(len(idx))
    w = _apply_weights(base, None if df is None else df[idx], cfg, n_res)
    return ConceptSignal([vocab.ids[j] for j in idx], w, P_pool[:, idx].copy())


def mix_units(pred_units: Sequence[str], oracle_units: Sequence[str],
              mix: float, rng: np.random.Generator) -> List[str]:
    """Interpolate the ground set between predicted and gold concepts.
    """
    pred, orac = list(pred_units), list(oracle_units)
    if not orac or mix <= 0:
        return pred
    n = len(pred) if pred else len(orac)
    k = int(round(mix * n))
    if k <= 0:
        return pred
    missing = [c for c in orac if c not in set(pred)]
    rng.shuffle(missing)
    take = missing[:k]
    if len(take) < k:                       # gold exhausted; top up from gold
        take += [c for c in orac if c not in take][: k - len(take)]
    keep = [c for c in pred if c not in set(take)][: max(0, n - len(take))]
    return keep + take


def uq_from_texts(texts: Sequence[str], encoder, vocab_emb: np.ndarray,
                  vocab: Vocabulary, P_pool: np.ndarray, cfg,
                  per_text: int = 2, df: Optional[np.ndarray] = None,
                  n_res: int = 1) -> ConceptSignal:
    """Project free-text information requirements onto the concept vocabulary.
    """
    if not texts:
        idx = np.arange(min(cfg.uq_min, P_pool.shape[1]))
        w = np.ones(len(idx), dtype=np.float32)
        return ConceptSignal([vocab.ids[j] for j in idx],
                             w / w.sum() if cfg.normalize_weights else w,
                             P_pool[:, idx].copy())
    TE = encoder.encode(list(texts))                  # |texts| x d
    S = TE @ vocab_emb.T                              # |texts| x |V|
    order = np.argsort(-S, axis=1)[:, :per_text]
    best: Dict[int, float] = {}
    for r in range(order.shape[0]):
        for j in order[r]:
            best[int(j)] = max(best.get(int(j), -1e9), float(S[r, j]))
    idx = np.array(sorted(best, key=lambda j: -best[j])[: cfg.kappa], dtype=int)
    idx = np.sort(idx)
    a = np.array([best[int(j)] for j in idx])
    w = _apply_weights(a, None if df is None else df[idx], cfg, n_res)
    return ConceptSignal([vocab.ids[j] for j in idx], w, P_pool[:, idx].copy())


def uq_oracle(gold_concept_ids: Sequence[str], vocab: Vocabulary,
              P_pool: np.ndarray, cfg) -> ConceptSignal:
    """Diagnostic only: U_q read straight off the gold sets' concepts.
    """
    pos = {c: i for i, c in enumerate(vocab.ids)}
    idx = np.sort(np.array([pos[c] for c in gold_concept_ids if c in pos], dtype=int))
    if len(idx) == 0:
        idx = np.arange(min(cfg.uq_min, P_pool.shape[1]))
    w = np.ones(len(idx), dtype=np.float32)
    if cfg.normalize_weights:
        w = w / w.sum()
    return ConceptSignal([vocab.ids[j] for j in idx], w, P_pool[:, idx].copy())
