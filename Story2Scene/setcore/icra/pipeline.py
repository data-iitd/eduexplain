"""End-to-end runner.

    python -m icra.pipeline run     --resources resources.json --queries queries.json
    python -m icra.pipeline check   --resources resources.json --queries queries.json
    python -m icra.pipeline ablate  --resources resources.json --queries queries.json

Generalisation is estimated by k-fold cross-validation over queries: the role
head (and the learned concept head) are fit on the training folds only, and
every reported number comes from held-out folds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import concepts as C
from . import evaluate as E
from . import select as S
from .config import Config
from .data import (Query, Resource, TYPES, assign_groups, build_vocabulary,
                   load_candidates, load_queries, load_resources, slugify,
                   validate)
from .encoders import build_encoder
from .heads import (ConceptHead, MajorityRoleBaseline, RoleHead,
                    group_kfold_indices, kfold_indices)


# --------------------------------------------------------------------- setup
def _annotation_matrix(resources, vocab):
    """|R| x |V| indicator of the corpus `concepts` annotation."""
    from .data import slugify
    pos = {c: j for j, c in enumerate(vocab.ids)}
    Y = np.zeros((len(resources), len(vocab.ids)), dtype=np.float32)
    for i, r in enumerate(resources):
        for c in (r.gold_concepts or []):
            j = pos.get(slugify(c))
            if j is not None:
                Y[i, j] = 1.0
    return Y


def _res_groups(resources, mode: str) -> List[str]:
    """Grouping for the resource-side folds.

    Resources from one chapter overlap heavily in wording and in concepts, so
    plain k-fold would put near-duplicates of a held-out resource in its own
    training fold and overstate how well the tagger generalises. Only fields on
    the allowed list are used (subject, chapter_title, document_id).
    """
    if mode == "document":
        return [str(getattr(r, "document_id", None) or r.resource_id)
                for r in resources]
    if mode == "chapter":
        return [f"{r.subject or '-'}|{r.chapter_title or r.document_id or r.resource_id}"
                for r in resources]
    return [r.resource_id for r in resources]


def _learned_con_prob(resources, R_emb, vocab, cfg, quiet: bool = False
                      ) -> np.ndarray:
    """Out-of-fold |R| x |V| concept probabilities for every resource.

    Supervised by the corpus `concepts` annotation, which is resource-side
    indexing-time metadata (the same thing con_source="annotated" reads
    directly). The point of learning it is that every prediction is OUT OF
    FOLD: the tagger is scored, and used, on resources whose annotation it
    never saw, so the row answers "would this work on a corpus that is not
    annotated" rather than "can a model memorise a lookup table".

    Cached, because the decode (top-m vs size-matched) is free to vary once
    these are in hand and the fit is not.
    """
    from .heads import ConceptHead
    Y = _annotation_matrix(resources, vocab)
    has = Y.sum(axis=1) > 0
    nR, nV = Y.shape
    if not has.any():
        raise SystemExit("--con-source learned needs resource-level `concepts` "
                         "annotations to train on; this corpus has none.")

    key = hashlib.sha1(json.dumps({
        # v2: the minibatch generator moved into ConceptHead.__init__, which
        # changes the shuffle sequence, so v1 caches are not interchangeable.
        "v": 2,
        "res": cfg.resources_path, "enc": cfg.encoder_name,
        "nR": nR, "vocab": hashlib.sha1("|".join(vocab.ids).encode()).hexdigest(),
        "folds": cfg.con_folds, "group": cfg.con_group, "seed": cfg.seed,
        "h": cfg.res_concept_hidden, "lr": cfg.res_concept_lr,
        "ep": cfg.res_concept_epochs, "bs": cfg.res_concept_batch,
        "wd": cfg.res_concept_wd,
    }, sort_keys=True).encode()).hexdigest()[:16]
    cache = os.path.join(cfg.cache_dir, f"conprob_{key}.npy")

    if os.path.exists(cache):
        PROB = np.load(cache)
        if not quiet:
            print(f"Con(r) classifier: out-of-fold predictions from cache "
                  f"({os.path.basename(cache)})")
    else:
        groups = _res_groups(resources, cfg.con_group)
        k = min(cfg.con_folds, max(2, len(set(groups))))
        folds = group_kfold_indices(groups, k, cfg.seed)
        PROB = np.zeros((nR, nV), dtype=np.float32)
        for fi, (tr, te) in enumerate(folds):
            tr = tr[has[tr]]
            if len(tr) == 0:
                continue
            head = ConceptHead(R_emb.shape[1], nV, cfg.res_concept_hidden,
                               cfg.res_concept_lr, cfg.res_concept_epochs,
                               "sigmoid", cfg.seed,
                               batch_size=cfg.res_concept_batch,
                               wd=cfg.res_concept_wd)
            head.fit(R_emb[tr], Y[tr])
            PROB[te] = head.predict(R_emb[te]).astype(np.float32)
            if not quiet:
                print(f"Con(r) classifier fold {fi}: trained on {len(tr)} "
                      f"annotated resources, tagged {len(te)} held out")
        os.makedirs(cfg.cache_dir, exist_ok=True)
        np.save(cache, PROB)
    return PROB


def _learned_con_matrix(resources, R_emb, vocab, cfg, quiet: bool = False
                        ) -> Tuple[np.ndarray, Dict]:
    """Con(r) from the out-of-fold classifier, decoded and scored."""
    PROB = _learned_con_prob(resources, R_emb, vocab, cfg, quiet)
    Y = _annotation_matrix(resources, vocab)
    has = Y.sum(axis=1) > 0

    target = float(Y[has].sum(axis=1).mean())
    P = C.learned_concept_matrix(PROB, cfg, target)

    # ---- how good is the tagging, against the annotation it is predicting?
    B = (P > 0)
    A = Y > 0
    tp = float((B[has] & A[has]).sum())
    diag = {
        "con_source": "learned",
        "con_group": cfg.con_group,
        "con_decode": cfg.con_learned_size,
        "con_annotated_resources": int(has.sum()),
        "con_micro_precision": tp / max(1.0, float(B[has].sum())),
        "con_micro_recall": tp / max(1.0, float(A[has].sum())),
        "con_size_pred": float(B.sum(axis=1).mean()),
        "con_size_annotated": target,
    }
    p, r = diag["con_micro_precision"], diag["con_micro_recall"]
    diag["con_micro_f1"] = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    per = [(len(set(np.nonzero(B[i])[0]) & set(np.nonzero(A[i])[0])),
            int(B[i].sum()), int(A[i].sum()))
           for i in np.nonzero(has)[0]]
    diag["con_macro_f1"] = float(np.mean(
        [0.0 if (b + a) == 0 else 2 * t / (b + a) for t, b, a in per]))
    if not quiet:
        print(f"Con(r) classifier: micro-F1 {diag['con_micro_f1']:.4f} "
              f"(P {p:.4f} / R {r:.4f}), macro-F1 {diag['con_macro_f1']:.4f}, "
              f"|Con(r)| {diag['con_size_pred']:.2f} vs annotated "
              f"{target:.2f}")
    return P, diag


def _con_matrix(resources, R_emb, V_emb, vocab, cfg, quiet: bool = False):
    """Dispatch the covering relation Con(r) on cfg.con_source."""
    _con_matrix.last_diag = {}      # cleared, so a sweep cannot carry one over
    if cfg.con_source == "annotated":
        return C.annotated_concept_matrix(resources, vocab, mode="annotated")
    if cfg.con_source == "learned":
        P, diag = _learned_con_matrix(resources, R_emb, vocab, cfg, quiet)
        _con_matrix.last_diag = diag
        return P
    P = C.build_concept_matrix(R_emb, V_emb, cfg)
    if cfg.con_source == "both":
        return C.annotated_concept_matrix(resources, vocab, P, mode="both")
    return P


_con_matrix.last_diag = {}


class Index:
    """Everything that is computed once, independently of any query split."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.resources: List[Resource] = load_resources(cfg.resources_path)
        self.queries: List[Query] = load_queries(cfg.queries_path)
        assign_groups(self.queries, cfg.cv_group, cfg.cv_group_prefix_len)
        self.train_queries: List[Query] = (
            load_queries(cfg.train_queries_path) if cfg.train_queries_path else [])
        if self.train_queries:
            assign_groups(self.train_queries, cfg.cv_group, cfg.cv_group_prefix_len)
        self.rid2i = {r.resource_id: i for i, r in enumerate(self.resources)}
        self.vocab = build_vocabulary(
            self.resources, cfg.vocab_source, cfg.vocab_min_df,
            cfg.vocab_max_df_frac, cfg.vocab_drop_structural,
            cfg.vocab_drop_generic, cfg.vocab_context_descriptors,
            cfg.vocab_structural_frac)

        self.enc = build_encoder(cfg)
        self.R = self.enc.encode([r.index_text() for r in self.resources])
        self.V = self.enc.encode(self.vocab.descriptors)
        self.Q = self.enc.encode([q.text for q in self.queries])
        self.Q_train = (self.enc.encode([q.text for q in self.train_queries])
                        if self.train_queries else np.zeros((0, self.R.shape[1]),
                                                            dtype=np.float32))
        self.enc.flush()

        # Con(r): tagging of every resource against V, from stored text only.
        self.P_full = _con_matrix(self.resources, self.R, self.V, self.vocab, cfg)
        self.con_diag = dict(_con_matrix.last_diag)
        self.Con = C.concept_sets(self.P_full, self.vocab)
        self.df = (self.P_full > 0).sum(axis=0).astype(np.float64)  # for aq_idf
        self.types = np.array([r.type for r in self.resources])
        self.candidates = (load_candidates(cfg.candidates_path)
                           if cfg.candidates_path else None)
        self.uq_ext: Optional[Dict[str, List[str]]] = None
        if cfg.uq_file:
            with open(cfg.uq_file) as f:
                raw = json.load(f)
            if isinstance(raw, list):
                raw = {str(d.get("query_id")): (d.get("concepts")
                       or d.get("units") or d.get("subqueries") or []) for d in raw}
            self.uq_ext = {str(k): [str(x) for x in (v or [])]
                           for k, v in raw.items()}
            n_miss = sum(1 for q in self.queries if q.query_id not in self.uq_ext)
            vocab_ids = set(self.vocab.ids)
            n_id = sum(1 for v in self.uq_ext.values() for x in v if x in vocab_ids)
            n_tot = sum(len(v) for v in self.uq_ext.values())
            print(f"U_q file: {len(self.uq_ext)} queries, {n_miss} eval queries "
                  f"missing (those fall back to the matcher); "
                  f"{n_id}/{n_tot} entries are vocabulary ids, the rest are "
                  f"treated as free text and projected onto V")
        if self.candidates is not None:
            miss = sum(1 for q in self.queries if q.query_id not in self.candidates)
            if miss:
                print(f"WARNING: {miss}/{len(self.queries)} eval queries have no "
                      f"entry in the candidates file; those fall back to dense "
                      f"retrieval, so the first stage is mixed. Check the ids match.")

    # ------------------------------------------------------------- rebinding
    # Params that change Con(r) and therefore require re-tagging.
    _TAG_KEYS = ("vocab_source", "tag_threshold", "tag_topm", "tag_zscore",
                 "tag_z_threshold", "tag_rel", "con_source",
                 "vocab_min_df", "prf_docs", "uq_file_per_text", "e2e_epochs", "e2e_d_model",
                 "e2e_layers", "e2e_heads", "e2e_batch_size", "e2e_patience",
                 "e2e_silver_epochs", "vocab_max_df_frac", "vocab_drop_structural",
                 "vocab_drop_generic", "vocab_context_descriptors",
                 # learned Con(r): all of these change the tagging
                 "con_group", "con_folds", "con_learned_topm", "con_learned_tau",
                 "con_learned_size", "res_concept_hidden", "res_concept_lr",
                 "res_concept_epochs", "res_concept_batch", "res_concept_wd")

    def rebind(self, cfg: Config) -> "Index":
        """Return an Index sharing the (expensive) embeddings under a new config.

        Only the concept-tagging stage is recomputed, and only when a parameter
        that affects it actually changed. Used by the ablation sweep so a
        14-cell grid does not re-encode the corpus 14 times.
        """
        if any(getattr(cfg, k) != getattr(self.cfg, k) for k in self._TAG_KEYS):
            if any(getattr(cfg, k) != getattr(self.cfg, k) for k in
                   ("vocab_source", "vocab_min_df", "vocab_max_df_frac",
                    "vocab_drop_structural", "vocab_drop_generic",
                    "vocab_context_descriptors")):
                return Index(cfg)                       # V itself changed
            new = object.__new__(Index)
            new.__dict__.update(self.__dict__)
            new.cfg = cfg
            # quiet: a sweep re-tags once per row and the tagger diagnostic is
            # identical every time; it is kept in each row's metrics.json.
            new.P_full = _con_matrix(self.resources, self.R, self.V,
                                     self.vocab, cfg, quiet=True)
            new.con_diag = dict(_con_matrix.last_diag)
            new.Con = C.concept_sets(new.P_full, self.vocab)
            new.df = (new.P_full > 0).sum(axis=0).astype(np.float64)
            return new
        new = object.__new__(Index)
        new.__dict__.update(self.__dict__)
        new.cfg = cfg
        return new

    def full_ranking(self, qi: int, n: int = 200) -> List[str]:
        """Top-n resource ids over the WHOLE corpus, for Recall@K.

        Independent of cfg.K so recall@200 stays meaningful with a 100-doc pool.
        """
        if self.candidates is not None:
            lst = self.candidates.get(self.queries[qi].query_id)
            if lst:
                return [rid for rid, _ in lst[:n]]
        s_all = self.R @ self.Q[qi]
        idx = np.argsort(-s_all)[:n]
        return [self.resources[i].resource_id for i in idx]

    # ------------------------------------------------------------ first stage
    def retrieve_for(self, q_emb: np.ndarray,
                     q: Optional[Query] = None) -> Tuple[np.ndarray, np.ndarray]:
        """First-stage pool for an arbitrary query embedding.

        Same logic as retrieve(), but usable for TRAINING queries, which are
        not in self.queries. The E2E variant needs pools for both files.
        """
        cfg = self.cfg
        s_all = self.R @ q_emb
        mask = np.ones(len(self.resources), dtype=bool)
        if cfg.restrict_to_subject and q is not None:
            for j, r in enumerate(self.resources):
                if q.subject and r.subject and r.subject != q.subject:
                    mask[j] = False
                if q.class_no is not None and r.class_no is not None \
                        and r.class_no != q.class_no:
                    mask[j] = False
            if mask.sum() < cfg.K:
                mask = np.ones(len(self.resources), dtype=bool)
        cand = np.nonzero(mask)[0]
        if cfg.stratified_first_stage:
            out = []
            for t, k in (("figure", cfg.K_figure),
                         ("explanation", max(0, cfg.K - cfg.K_figure))):
                sub = cand[self.types[cand] == t]
                sub = sub[np.argsort(-s_all[sub])][:k]
                out.append(sub)
            pool = np.concatenate(out)
        else:
            pool = cand[np.argsort(-s_all[cand])][: cfg.K]
        return pool, s_all[pool]

    def _from_candidates(self, qid: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        lst = (self.candidates or {}).get(qid)
        if not lst:
            return None
        idx, sc = [], []
        for rid, s_ in lst[: self.cfg.K]:
            if rid in self.rid2i:
                idx.append(self.rid2i[rid])
                sc.append(s_)
        if not idx:
            return None
        return np.array(idx, dtype=int), np.array(sc, dtype=float)

    def retrieve(self, qi: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.candidates is not None:
            got = self._from_candidates(self.queries[qi].query_id)
            if got is not None:
                return got
        cfg = self.cfg
        q = self.queries[qi]
        s_all = self.R @ self.Q[qi]
        mask = np.ones(len(self.resources), dtype=bool)
        if cfg.restrict_to_subject:
            for j, r in enumerate(self.resources):
                if q.subject and r.subject and r.subject != q.subject:
                    mask[j] = False
                if q.class_no is not None and r.class_no is not None and r.class_no != q.class_no:
                    mask[j] = False
            if mask.sum() < cfg.K:
                mask = np.ones(len(self.resources), dtype=bool)
        cand = np.nonzero(mask)[0]

        if cfg.stratified_first_stage:
            # Figures are the type a text retriever handles worst (short,
            # stylistically unlike explanatory prose), so an unstratified pool
            # often contains none and the figure slot silently goes unfilled.
            out = []
            for t, k in (("figure", cfg.K_figure),
                         ("explanation", max(0, cfg.K - cfg.K_figure))):
                sub = cand[self.types[cand] == t]
                sub = sub[np.argsort(-s_all[sub])][:k]
                out.append(sub)
            pool = np.concatenate(out)
        else:
            pool = cand[np.argsort(-s_all[cand])][: cfg.K]
        return pool, s_all[pool]

    # --------------------------------------------------------- role targets
    def targets_for(self, q: Query) -> Tuple[List[int], List[int]]:
        """Admissible (count, figure) profiles, one entry per gold set.

        Read purely from gold-set resource TYPES -- never from bundle_type,
        which counts query concepts, not resources.
        """
        cfg = self.cfg
        cs, fs = [], []
        for g in q.gold_sets:
            idx = [self.rid2i[x] for x in g if x in self.rid2i]
            c = sum(1 for i in idx if self.types[i] == "explanation")
            f = int(any(self.types[i] == "figure" for i in idx))
            cs.append(min(c, cfg.n_max)); fs.append(f)
        if not cs:
            cs, fs = [0], [0]
        return cs, fs

    def role_targets(self, qi: int) -> Tuple[List[int], List[int]]:
        return self.targets_for(self.queries[qi])

    def mean_targets_for(self, q: Query) -> Tuple[int, int]:
        """Eq. (4): rounded mean count, majority figure decision."""
        cs, fs = self.targets_for(q)
        return min(int(round(float(np.mean(cs)))), self.cfg.n_max), \
            int(float(np.mean(fs)) >= 0.5)

    def mean_targets(self, qi: int) -> Tuple[int, int]:
        return self.mean_targets_for(self.queries[qi])

    def concept_targets_for(self, q: Query) -> List[str]:
        """Training target for the concept head g_psi.

        concept_target="query": the query-level `concepts` annotation, which is
        a direct gold U_q and a far cleaner target than reconstructing it from
        the gold resources. Construction-time, so it is a TARGET only -- never
        an inference input.
        concept_target="gold_con": union of Con(r) over the gold resources.
            NOTE this MOVES WITH --con-source, because Con(r) does. Under
            --con-source learned the head is then trained to predict the
            resource classifier's tags rather than the annotation, so a
            query-side comparison across different con_source values is
            confounded: both the covering relation and the supervision changed.
        concept_target="gold_con_ann": union of the resources' stored `concepts`
            ANNOTATION over the gold sets, whatever --con-source is. Same
            quantity as gold_con under --con-source annotated, but held fixed
            when Con(r) is learned -- use this to vary one thing at a time.
        concept_target="both": union of the query annotation and gold_con.
        """
        pos = set(self.vocab.ids)
        q_units = [c for c in q.gold_concepts if c in pos]
        mode = self.cfg.concept_target
        if mode == "query" and q_units:
            return q_units
        if mode == "gold_con_ann":
            return self.gold_annotation_units_for(q)
        if mode == "both":
            return sorted(set(q_units) | set(self.gold_con_units_for(q)))
        return self.gold_con_units_for(q)

    def gold_annotation_units_for(self, q: Query) -> List[str]:
        """Union over the gold sets of the resources' stored `concepts`.

        Independent of cfg.con_source, which is the whole point: it is the same
        target in every cell of the con_source x uq_mode table.
        """
        pos = set(self.vocab.ids)
        out = set()
        for g in q.gold_sets:
            for x in g:
                if x in self.rid2i:
                    for c in (self.resources[self.rid2i[x]].gold_concepts or []):
                        s = slugify(c)
                        if s in pos:
                            out.add(s)
        return sorted(out)

    def oracle_units_for(self, q: Query) -> List[str]:
        """Best available gold U_q, for measuring how good the predicted one is."""
        pos = set(self.vocab.ids)
        qu = [c for c in q.gold_concepts if c in pos]
        return qu if qu else self.gold_con_units_for(q)

    def gold_con_units_for(self, q: Query) -> List[str]:
        """Union over gold sets of Con(r). Uses the predicted tagging of gold
        resources, so no query-level annotation is involved."""
        out = set()
        for g in q.gold_sets:
            for x in g:
                if x in self.rid2i:
                    out |= set(self.Con[self.rid2i[x]])
        return sorted(out)

    def gold_con_units(self, qi: int) -> List[str]:
        """Union over gold sets of Con(r) -- targets for the learned head.

        Note this uses the *predicted* tagging of gold resources, not the
        corpus `concepts` field, so the leakage boundary holds: gold concept
        annotations stay exclusively in evaluate.py.
        """
        out = set()
        for g in self.queries[qi].gold_sets:
            for x in g:
                if x in self.rid2i:
                    out |= set(self.Con[self.rid2i[x]])
        return sorted(out)


# ---------------------------------------------------------------------- run
def _check_inputs(cfg: Config) -> None:
    """Refuse oracle modes under strict_inputs unless explicitly allowed."""
    if not cfg.strict_inputs:
        return
    bad = []
    if cfg.profile_mode == "intent":
        bad.append("profile_mode='intent' reads query_type")
    if cfg.uq_oracle:
        bad.append("uq_oracle reads gold concepts")
    if cfg.restrict_to_subject:
        bad.append("restrict_to_subject reads query-level grade/subject")
    if bad and not cfg.allow_oracles:
        raise ValueError(
            "strict_inputs=True but " + "; ".join(bad) +
            ". These are construction-time fields. Pass --allow-oracles to run "
            "them as labelled oracle ablations, and never report them as "
            "system numbers.")


def _concept_targets_matrix(ix: "Index", queries, pos) -> np.ndarray:
    Y = np.zeros((len(queries), len(pos)), dtype=np.float32)
    for k, q in enumerate(queries):
        for u in ix.concept_targets_for(q):
            j = pos.get(u)
            if j is not None:
                Y[k, j] = 1.0
    return Y


def _concept_quality(RHO: np.ndarray, Y: np.ndarray, cfg) -> Dict:
    """Micro/macro F1 of the head's own decode, before kappa and the weights.

    Measured at whatever decode the selector will use (top-k if concept_topk is
    set, else the gamma_learned probability threshold), so this number and the
    reported uq_f1 differ only by the kappa cap and the uq_min floor.
    """
    if cfg.concept_topk > 0:
        k = min(cfg.concept_topk, RHO.shape[1])
        B = np.zeros_like(RHO, dtype=bool)
        np.put_along_axis(B, np.argpartition(-RHO, k - 1, axis=1)[:, :k], True, 1)
    else:
        B = RHO >= cfg.gamma_learned
    A = Y > 0
    tp = float((B & A).sum())
    p = tp / max(1.0, float(B.sum()))
    r = tp / max(1.0, float(A.sum()))
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    mac = [0.0 if (int(B[i].sum()) + int(A[i].sum())) == 0 else
           2 * float((B[i] & A[i]).sum()) / (int(B[i].sum()) + int(A[i].sum()))
           for i in range(RHO.shape[0])]
    return {"head_micro_precision": p, "head_micro_recall": r,
            "head_micro_f1": f, "head_macro_f1": float(np.mean(mac)),
            "head_size": float(B.sum(axis=1).mean()),
            "head_target_size": float(A.sum(axis=1).mean())}


def _fit_concept_head(ix: "Index", cfg: Config, quiet: bool = False
                      ) -> Tuple[np.ndarray, Dict]:
    """rho_q over V for every evaluation query, under one of three protocols.

    The only protocols that are defensible here are the two the silver/gold
    distribution gap allows: grouped k-fold over the gold queries, or pre-train
    on silver and fine-tune on the gold folds. Fitting on silver alone is kept
    as a labelled row because it is the cheap option and the comparison is the
    point -- it is NOT cross-validated on the evaluation set, it is a transfer
    row.
    """
    nq, nV = len(ix.queries), len(ix.vocab)
    pos = {v: j for j, v in enumerate(ix.vocab.ids)}
    RHO = np.zeros((nq, nV))

    mode = cfg.concept_train
    if mode == "auto":
        mode = ("silver" if (ix.train_queries and cfg.role_train != "cv")
                else "cv")
    if mode in ("silver", "silver_then_cv") and not ix.train_queries:
        raise SystemExit(f"--concept-train {mode} needs --train-queries "
                         "(the silver file)")

    def _new_head(epochs=None, lr=None):
        return ConceptHead(ix.Q.shape[1], nV, cfg.concept_hidden,
                           cfg.concept_lr if lr is None else lr,
                           cfg.concept_epochs if epochs is None else epochs,
                           cfg.concept_activation, cfg.seed)

    info: Dict = {"concept_train": mode, "concept_target": cfg.concept_target}

    # ---- optional silver phase, fitted once
    pre_state = None
    if mode in ("silver", "silver_then_cv"):
        Ys = _concept_targets_matrix(ix, ix.train_queries, pos)
        head = _new_head().fit(ix.Q_train, Ys)
        info["concept_silver_queries"] = len(ix.train_queries)
        if mode == "silver":
            RHO = head.predict(ix.Q)
            info.update(_concept_quality(
                RHO, _concept_targets_matrix(ix, ix.queries, pos), cfg))
            if not quiet:
                print(f"g_psi: fitted on {len(ix.train_queries)} silver "
                      f"queries, transferred to {nq} gold queries "
                      f"(head micro-F1 {info['head_micro_f1']:.4f})")
            return RHO, info
        pre_state = head.state()
        if not quiet:
            print(f"g_psi: pre-trained on {len(ix.train_queries)} silver "
                  f"queries; fine-tuning per fold")

    # ---- grouped k-fold over the gold queries
    k = min(cfg.folds, max(2, nq))
    folds = group_kfold_indices([q.group_id for q in ix.queries], k, cfg.seed)
    Y = _concept_targets_matrix(ix, ix.queries, pos)
    for fi, (tr, te) in enumerate(folds):
        head = _new_head()
        if pre_state is not None:
            # Warm start. A fresh Adam state and a lower lr for the gold phase:
            # 321 queries after 992 silver ones will otherwise undo the
            # pre-training in the first few steps.
            head.load_state(pre_state, lr=cfg.concept_ft_lr,
                            epochs=cfg.concept_ft_epochs)
        head.fit(ix.Q[tr], Y[tr])
        RHO[te] = head.predict(ix.Q[te])
        if not quiet:
            print(f"g_psi fold {fi}: {len(tr)} train / {len(te)} held out")
    info.update(_concept_quality(RHO, Y, cfg))
    info["concept_folds"] = k
    if not quiet:
        print(f"g_psi: grouped {k}-fold over the gold queries, head micro-F1 "
              f"{info['head_micro_f1']:.4f} (P {info['head_micro_precision']:.4f} "
              f"/ R {info['head_micro_recall']:.4f}), |pred| "
              f"{info['head_size']:.2f} vs |target| {info['head_target_size']:.2f}")
    return RHO, info


def run(cfg: Config, ix: Optional[Index] = None, quiet: bool = False) -> Dict:
    _check_inputs(cfg)
    _t_start = time.perf_counter()
    _t_index = 0.0
    if ix is None:
        _t0 = time.perf_counter()
        ix = Index(cfg)
        _t_index = time.perf_counter() - _t0
    nq = len(ix.queries)
    two_file = bool(ix.train_queries)
    # The lightweight heads (role, concept, intent) and the end-to-end model do
    # not have to be fitted on the same data. --role-train cv keeps the heads on
    # grouped k-fold CV over the evaluation queries while the e2e model still
    # trains on the separate training file. That matters when the two files'
    # gold sets differ in size: silver proposals are smaller than the
    # human-validated sets (annotators ADD resources during validation), so a
    # role head fitted on silver systematically under-predicts |R_q| here.
    role_two_file = two_file and cfg.role_train != "cv"
    if role_two_file:
        # Heads are fit once on the separate training file; the evaluation
        # queries are never trained on, so no folds are needed.
        rng_folds = [(np.arange(len(ix.train_queries)), np.arange(nq))]
    else:
        k = min(cfg.folds, max(2, nq))
        rng_folds = group_kfold_indices([q.group_id for q in ix.queries], k, cfg.seed)

    # External size budget, shared with the baselines.
    ext_sizes: Optional[Dict[str, int]] = None
    if cfg.sizes_file:
        with open(cfg.sizes_file) as _f:
            _raw = json.load(_f)
        ext_sizes = {str(k): int(v) for k, v in _raw.items()}
        _have = sum(1 for q in ix.queries if str(q.query_id) in ext_sizes)
        if not quiet:
            print(f"budget composition: {cfg.sizes_composition}"
                  + ("  (role head keeps the figure decision; only |R_q| is "
                     "pinned)" if cfg.sizes_composition == "role" else
                     "  (type constraint dropped for budgeted queries, as the "
                     "LLM baselines consume the same file)"))
        if not quiet:
            print(f"size budget from {cfg.sizes_file}: "
                  f"{_have}/{nq} queries matched"
                  + ("" if _have == nq else
                     "; unmatched queries fall back to the predicted profile"))

    # Used only when --sizes supplies a total and --sizes-composition free
    # drops the partition for those queries.
    _cfg_free_types = Config(**{**cfg.__dict__, "type_constraint": False})

    pred_c = np.zeros(nq, dtype=int)
    pred_f = np.zeros(nq, dtype=int)
    pred_intent = np.array([""] * nq, dtype=object)
    base_c = np.zeros(nq, dtype=int)
    base_f = np.zeros(nq, dtype=int)
    rho_all: Optional[np.ndarray] = None
    concept_info: Dict = {}

    if cfg.uq_mode == "learned" and cfg.concept_target == "query":
        n_tr = sum(1 for q in (ix.train_queries or ix.queries) if q.gold_concepts)
        n_ev = sum(1 for q in ix.queries if q.gold_concepts)
        if n_tr and not n_ev:
            print("WARNING: --concept-target query trains g_psi on the "
                  "query-level `concepts` annotation, but the evaluation file "
                  "has none, so U_q is scored against Con(r) of the gold "
                  "resources instead. Those are different ground sets (query "
                  "topic vs what an explanation must cover) and the head will "
                  "look far worse than it is. Use --concept-target gold_con.")

    if cfg.uq_mode == "learned":
        rho_all, concept_info = _fit_concept_head(ix, cfg, quiet)

    TRQ = ix.train_queries if two_file else ix.queries
    TRX = ix.Q_train if two_file else ix.Q
    # Head-fitting source, which --role-train can decouple from TRQ.
    RQ = ix.train_queries if role_two_file else ix.queries
    RX = ix.Q_train if role_two_file else ix.Q
    for tr, te in rng_folds:
        tq = [RQ[i] for i in tr]
        _multi = cfg.role_loss in ("best_match", "single_first", "single_random", "average")
        ct = [ix.targets_for(q)[0] if _multi
              else [ix.mean_targets_for(q)[0]] for q in tq]
        ft = [ix.targets_for(q)[1] if _multi
              else [ix.mean_targets_for(q)[1]] for q in tq]
        head = RoleHead(ix.Q.shape[1], cfg.n_max, cfg.role_hidden, cfg.role_lr,
                        cfg.role_weight_decay, cfg.role_epochs, cfg.lam,
                        cfg.role_loss, cfg.seed).fit(RX[tr], ct, ft)
        c, f, _ = head.predict(ix.Q[te], cfg.count_decode, cfg.eta)
        pred_c[te], pred_f[te] = c, f

        if cfg.profile_mode == "predicted_intent":
            # Table 4 row: instead of predicting the profile straight from the
            # query, predict the INTENT from the query and read off that
            # intent's modal profile. Intent labels train the classifier but are
            # never an inference input, so the leakage boundary holds; the
            # comparison with profile_mode="intent" (gold label) separates
            # intent-prediction error from the intent->profile mapping itself.
            from .heads import IntentHead
            labels = sorted({str(q.intent) for q in tq})
            li = {l: i for i, l in enumerate(labels)}
            ih = IntentHead(ix.Q.shape[1], len(labels), cfg.role_hidden,
                            cfg.role_lr, cfg.role_epochs, cfg.seed).fit(
                RX[tr], np.array([li[str(q.intent)] for q in tq]))
            tab2: Dict[str, List[Tuple[int, int]]] = {}
            for q in tq:
                tab2.setdefault(str(q.intent), []).append(ix.mean_targets_for(q))
            glob2 = [ix.mean_targets_for(q) for q in tq]
            pi = ih.predict(ix.Q[te])
            for k_, i in enumerate(te):
                rows_ = tab2.get(labels[int(pi[k_])], glob2) or glob2
                pred_c[i] = int(np.bincount([x[0] for x in rows_]).argmax())
                pred_f[i] = int(np.mean([x[1] for x in rows_]) >= 0.5)
                pred_intent[i] = labels[int(pi[k_])]

        if cfg.profile_mode == "intent":
            # Ablation promised in the write-up: give the system the
            # construction-time intent label and read off the modal profile of
            # that intent on the TRAINING folds only. Upper bound on what
            # "knowing the intent" is worth.
            tab: Dict[str, List[Tuple[int, int]]] = {}
            for q in tq:
                tab.setdefault(f"{q.intent}", []).append(ix.mean_targets_for(q))
            glob = [ix.mean_targets_for(q) for q in tq]
            for i in te:
                rows_ = tab.get(f"{ix.queries[i].intent}", glob) or glob
                pred_c[i] = int(np.bincount([x[0] for x in rows_]).argmax())
                pred_f[i] = int(np.mean([x[1] for x in rows_]) >= 0.5)

        mb = MajorityRoleBaseline().fit(RX[tr], [[x[0]] for x in ct], [[x[0]] for x in ft])
        bc, bf, _ = mb.predict(ix.Q[te])
        base_c[te], base_f[te] = bc, bf

    # ------------------------------------------------ end-to-end variant
    e2e_pred: Optional[List[List[int]]] = None
    e2e_info: Dict = {}
    if cfg.selector == "e2e":
        from . import e2e as E2

        # The pointer's candidate features are [is_explanation, is_figure,
        # normalised first-stage score, rank] plus the embeddings -- see
        # e2e.build_example. The concept layer is NOT among them, so U_q and
        # Con(r) do not enter the e2e objective at any point: changing how they
        # are built changes the logged U_q and nothing the model selects.
        # Verified by construction and by running both -- identical sets, all
        # end metrics identical to 4 dp.
        if cfg.uq_mode == "learned" or cfg.con_source == "learned":
            print("WARNING: --selector e2e ignores the concept layer entirely. "
                  "The pointer reads type, first-stage score, rank and the "
                  "embeddings -- never U_q or Con(r) -- so --uq-mode learned "
                  "and --con-source learned change nothing it predicts. "
                  "The concept-classifier experiments belong to the "
                  "lightweight (submodular) selector; to make them bear on "
                  "the e2e variant, the coverage signal has to be added to "
                  "build_example or to L_match first.")

        def _examples(queries, embs):
            out = []
            for k, qq in enumerate(queries):
                pl, sc = ix.retrieve_for(embs[k], qq)
                out.append(E2.build_example(ix, qq, embs[k], pl, sc))
            return out

        def _dev_split(idx, groups_all, seed):
            """Hold out whole GROUPS for early stopping, never whole queries.

            Surface-form variants of one item share gold sets and near-identical
            embeddings, so splitting them across train/dev leaks.
            """
            g = np.array([groups_all[i] for i in idx])
            uniq = np.unique(g)
            rs = np.random.default_rng(seed)
            rs.shuffle(uniq)
            n_dev = int(round(cfg.e2e_dev_frac * len(uniq)))
            dev_g = set(uniq[:n_dev].tolist()) if n_dev > 0 else set()
            tr = [i for i in idx if groups_all[i] not in dev_g]
            dv = [i for i in idx if groups_all[i] in dev_g]
            return tr, dv

        # Hybrid: the lightweight role head decides HOW MANY (and of which
        # type); the pointer decides WHICH. The two components are independent
        # -- the role head reads only the query, the pointer only the pool --
        # so composing them is not double-counting any signal.
        _ov = None
        if ext_sizes is not None:
            _ov = [{"size": max(1, min(int(ext_sizes[str(q.query_id)]), cfg.n_max))}
                   if str(q.query_id) in ext_sizes else {}
                   for q in ix.queries]
        elif cfg.e2e_size_source in ("role", "role_profile", "oracle", "oracle_profile"):
            if cfg.e2e_size_source.startswith("oracle") and not cfg.allow_oracles:
                raise SystemExit("--e2e-size-source oracle needs --allow-oracles")
            _ov = []
            for i in range(len(ix.queries)):
                if cfg.e2e_size_source.startswith("oracle"):
                    cq, fq = ix.mean_targets(i)   # rounded mean gold composition
                else:
                    cq, fq = int(pred_c[i]), int(pred_f[i])
                cq = max(0, min(cq, cfg.n_max))
                if cq + fq > cfg.n_max:
                    fq = max(0, cfg.n_max - cq)
                if cfg.e2e_size_source in ("role", "oracle"):
                    _ov.append({"size": max(1, cq + fq)})
                else:
                    pool_i, _ = ix.retrieve(i)
                    _ov.append({"profile": {"explanation": cq, "figure": fq},
                                "types": [str(t) for t in ix.types[pool_i]]})

        # ---- optional silver pre-training, done ONCE and reused.
        # The silver file shares no query with the evaluation set, so one
        # pre-trained checkpoint can seed every fold without leaking: what
        # differs between folds is only which gold queries are fine-tuned on.
        pre_state = None
        if cfg.silver_queries_path:
            # The pre-trained checkpoint depends on the silver file, the seed,
            # the architecture and the LOSS -- but NOT on how |R_q| is decoded,
            # because the size regime only affects inference. So one checkpoint
            # serves every size regime, and caching it turns 4 pre-trains per
            # (loss, seed) into 1. Key on everything that does affect it.
            import torch as _torch
            _pk = hashlib.sha1(json.dumps({
                "v": 1, "silver": cfg.silver_queries_path,
                "enc": cfg.encoder_name, "seed": cfg.seed,
                "loss": cfg.e2e_gold_match, "ep": cfg.e2e_silver_epochs,
                "d": cfg.e2e_d_model, "L": cfg.e2e_layers, "H": cfg.e2e_heads,
                "do": cfg.e2e_dropout, "lr": cfg.e2e_lr,
                "wd": cfg.e2e_weight_decay, "bs": cfg.e2e_batch_size,
                "in": int(ix.Q.shape[1]), "K": cfg.K, "Kf": cfg.K_figure,
                "strat": cfg.stratified_first_stage, "nmax": cfg.n_max,
                "subj": cfg.restrict_to_subject, "cand": cfg.candidates_path,
                "lossnorm": cfg.e2e_loss_norm, "lamt": cfg.e2e_lam_type,
                "nullw": cfg.e2e_null_weight,
            }, sort_keys=True).encode()).hexdigest()[:16]
            _pp = os.path.join(cfg.cache_dir, f"e2e_silver_{_pk}.pt")
            if os.path.exists(_pp):
                pre_state = _torch.load(_pp, map_location=cfg.e2e_device)
                if not quiet:
                    print(f"E2E: silver pre-train from cache "
                          f"({os.path.basename(_pp)})")
                e2e_info["silver_pretrain_cached"] = True
            else:
                silver = load_queries(cfg.silver_queries_path)
                assign_groups(silver, cfg.cv_group, cfg.cv_group_prefix_len)
                s_emb = ix.enc.encode([q.text for q in silver])
                ix.enc.flush()
                silver_ex = _examples(silver, s_emb)
                if not quiet:
                    print(f"E2E: silver pre-training on {len(silver_ex)} queries")
                pre = E2.E2ETrainer(cfg, ix.Q.shape[1], cfg.e2e_device)
                pre.fit(silver_ex, None, cfg.e2e_silver_epochs, "silver", quiet)
                pre_state = {k: v.detach().clone()
                             for k, v in pre.model.state_dict().items()}
                os.makedirs(cfg.cache_dir, exist_ok=True)
                _torch.save(pre_state, _pp)
                e2e_info["silver_queries"] = len(silver_ex)
                e2e_info["silver_pretrain_cached"] = False

        if cfg.e2e_cv:
            # ---- grouped k-fold over the EVALUATION queries.
            # Required whenever the model is fine-tuned on the human-validated
            # set: every prediction must come from a model that never saw that
            # query. Folds are by group, so paraphrase variants cannot straddle.
            eval_ex = _examples(ix.queries, ix.Q)
            gall = [q.group_id for q in ix.queries]
            folds = group_kfold_indices(gall, min(cfg.folds, max(2, nq)), cfg.seed)
            e2e_pred = [[] for _ in range(nq)]
            fold_log = []
            for fi, (tr_f, te_f) in enumerate(folds):
                tr_i, dv_i = _dev_split(list(tr_f), gall, cfg.seed + fi)
                trainer = E2.E2ETrainer(cfg, ix.Q.shape[1], cfg.e2e_device)
                if pre_state is not None:
                    trainer.model.load_state_dict(pre_state)
                if not quiet:
                    print(f"E2E fold {fi}: fine-tuning on {len(tr_i)} gold "
                          f"queries ({len(dv_i)} dev, {len(te_f)} held out)")
                trainer.fit([eval_ex[i] for i in tr_i],
                            [eval_ex[i] for i in dv_i] or None,
                            tag=f"fold{fi}", quiet=quiet)
                sub = trainer.predict([eval_ex[i] for i in te_f],
                                      [_ov[i] for i in te_f] if _ov else None)
                for j, i in enumerate(te_f):
                    e2e_pred[i] = sub[j]
                _trp = trainer.predict([eval_ex[i] for i in tr_i])
                fold_log.append({
                    "fold": fi, "n_train": len(tr_i), "n_dev": len(dv_i),
                    "n_test": int(len(te_f)),
                    "best_dev_f1": getattr(trainer, "best_dev", None),
                    "best_epoch": getattr(trainer, "best_epoch", None),
                    "train_f1": float(np.mean([
                        E.official_best_metrics(
                            {eval_ex[i].pool_ids[j] for j in _trp[k]},
                            ix.queries[i].gold_sets)["f1"]
                        for k, i in enumerate(tr_i)])) if tr_i else 0.0,
                    "mean_predicted_size": float(np.mean([len(x) for x in sub])),
                })
            _be = [f["best_epoch"] for f in fold_log if f["best_epoch"] is not None]
            _bd = [f["best_dev_f1"] for f in fold_log if f["best_dev_f1"] is not None]
            e2e_info.update({
                "protocol": ("silver_pretrain+gold_cv" if pre_state is not None
                             else "gold_cv"),
                "folds": fold_log,
                "best_dev_f1": float(np.mean(_bd)) if _bd else None,
                "best_epoch": float(np.mean(_be)) if _be else None,
                "train_f1": float(np.mean([f["train_f1"] for f in fold_log])),
                "train_mean_size": float(np.mean(
                    [f["mean_predicted_size"] for f in fold_log])),
                "train_queries": int(np.mean([f["n_train"] for f in fold_log])),
                "dev_queries": int(np.mean([f["n_dev"] for f in fold_log])),
                "n_params": sum(p_.numel() for p_ in trainer.model.parameters()),
                "mean_predicted_size": float(np.mean([len(x) for x in e2e_pred])),
                "frac_empty_predictions": float(
                    np.mean([len(x) == 0 for x in e2e_pred])),
            })
        else:
            # ---- single fixed split: train on --train-queries, evaluate on
            # --queries. No fold is needed because the two files are disjoint.
            train_ex = _examples(TRQ, TRX)
            gall = [q.group_id for q in TRQ]
            tr_idx, dv_idx = _dev_split(list(range(len(TRQ))), gall, cfg.seed)
            tr_data = [train_ex[i] for i in tr_idx]
            dv_data = [train_ex[i] for i in dv_idx] or None

            trainer = E2.E2ETrainer(cfg, ix.Q.shape[1], cfg.e2e_device)
            if pre_state is not None:
                trainer.model.load_state_dict(pre_state)
            if not quiet:
                _what = "fine-tuning" if pre_state is not None else "training"
                _src = ("the separate training file" if two_file
                        else "the EVALUATION queries -- this leaks; use --e2e-cv")
                print(f"E2E: {_what} on {len(tr_data)} queries from {_src} "
                      f"({len(dv_data or [])} held out for early stopping)")
            trainer.fit(tr_data, dv_data, tag="gold", quiet=quiet)
            e2e_pred = trainer.predict([
                E2.build_example(ix, q, ix.Q[i], *ix.retrieve(i))
                for i, q in enumerate(ix.queries)], _ov)
            # Fit diagnostics: train-set F1 alongside the eval F1 separates
            # "did not learn" from "learned but did not generalise". A pointer
            # decoder on a few hundred queries can do either.
            _tr_pred = trainer.predict(tr_data)
            _tr_f1 = float(np.mean([
                E.official_best_metrics({tr_data[i].pool_ids[j] for j in _tr_pred[i]},
                                        TRQ[tr_idx[i]].gold_sets)["f1"]
                for i in range(len(tr_data))])) if tr_data else 0.0
            e2e_info.update({
                "protocol": ("silver_pretrain+train_file" if pre_state is not None
                             else ("train_file" if two_file else "eval_queries_LEAKY")),
                "best_dev_f1": getattr(trainer, "best_dev", None),
                "best_epoch": getattr(trainer, "best_epoch", None),
                "train_f1": _tr_f1,
                "train_mean_size": float(np.mean([len(x) for x in _tr_pred])) if _tr_pred else 0.0,
                "train_queries": len(tr_data), "dev_queries": len(dv_data or []),
                "n_params": sum(p_.numel() for p_ in trainer.model.parameters()),
                "mean_predicted_size": float(np.mean([len(x) for x in e2e_pred])),
                "frac_empty_predictions": float(np.mean([len(x) == 0 for x in e2e_pred])),
            })

        if e2e_info["frac_empty_predictions"] > 0.2 and not quiet:
            print(f"WARNING: the E2E model returns an EMPTY set for "
                  f"{e2e_info['frac_empty_predictions']:.0%} of queries. That is "
                  f"the NULL-collapse failure mode: lower --e2e-null-weight "
                  f"(try 0.05), or reduce --n-max toward the mean gold set size.")

    # A top-k reference at a FIXED size, constant across variants. The main
    # topk_baseline is size-matched to whatever the system returned, so a
    # variant that under-predicts size is compared against a crippled baseline
    # and two variants are not comparable to each other. This one is not.
    _gsz = [len(g) for q in ix.queries for g in q.gold_sets]
    fixed_k = max(1, int(round(float(np.mean(_gsz)) if _gsz else 2)))

    rows, topk_rows, fixedk_rows, preds, optiset = [], [], [], [], []
    gaps = []
    _t_select = time.perf_counter()
    for qi, q in enumerate(ix.queries):
        pool, s = ix.retrieve(qi)
        pool_ids = [ix.resources[i].resource_id for i in pool]
        ptypes = ix.types[pool]
        P_pool = ix.P_full[pool]

        # ---- profile pi_q
        gold_c, gold_f = ix.role_targets(qi)
        if cfg.profile_mode == "oracle":
            cq, fq = ix.mean_targets(qi)
        elif cfg.profile_mode in ("intent", "predicted_intent"):
            cq, fq = int(pred_c[qi]), int(pred_f[qi])
        elif cfg.profile_mode == "fixed":
            cq, fq = cfg.parsed_fixed_profile()
        else:
            cq, fq = int(pred_c[qi]), int(pred_f[qi])
        cq = max(0, min(cq, cfg.n_max))
        if cq + fq > cfg.n_max:      # drop the figure slot last
            fq = max(0, cfg.n_max - cq)
        profile = {"explanation": cq, "figure": fq}
        cfg_q = cfg
        if ext_sizes is not None and str(q.query_id) in ext_sizes:
            # The file gives a TOTAL, not a split.
            #
            # [FIX] This used to set {"explanation": n, "figure": 0}, which does
            # NOT let the selector choose the composition -- the partition
            # matroid then has no figure slot, so a --sizes run could never
            # select a figure at all. Measured: 0 figures over the whole corpus.
            # Two defensible readings, and they are different experiments:
            #   sizes_composition="role" (default) -- keep the role head's
            #       figure decision and spend the remaining budget on
            #       explanations. The method stays intact; only |R_q| is pinned.
            #   sizes_composition="free" -- drop the type constraint for these
            #       queries, which is how OptiSet/SetR consume the same file:
            #       one budget, any types.
            _n = max(1, min(int(ext_sizes[str(q.query_id)]), cfg.n_max))
            if cfg.sizes_composition == "free":
                cfg_q = _cfg_free_types
                profile = {"explanation": _n}      # select() folds this to _any
            else:
                _f = 1 if (int(pred_f[qi]) and _n >= 2) else 0
                profile = {"explanation": _n - _f, "figure": _f}

        # ---- concept signal U_q
        if cfg.uq_oracle:
            sig = C.uq_oracle(ix.gold_con_units(qi), ix.vocab, P_pool, cfg)
        elif cfg.uq_mode == "anchored":
            sig = C.uq_anchored(q.text, ix.R[pool], ix.enc, cfg)
        elif cfg.uq_mode == "prf":
            sig = C.uq_prf(ix.Q[qi], ix.R[pool], s, ix.V, P_pool, ix.vocab, cfg,
                           df=ix.df, n_res=len(ix.resources))
        elif cfg.uq_mode == "learned":
            sig = C.uq_learned(rho_all[qi], ix.vocab, P_pool, cfg,
                               a=ix.V @ ix.Q[qi], df=ix.df, n_res=len(ix.resources))
        elif ix.uq_ext is not None and ix.uq_ext.get(q.query_id):
            ents = ix.uq_ext[q.query_id]
            vocab_ids = set(ix.vocab.ids)
            direct = [x for x in ents if x in vocab_ids]
            free = [x for x in ents if x not in vocab_ids]
            # A file may mix the two: concepts that already match the
            # vocabulary are taken as-is, and anything else is projected onto
            # it. Both contribute -- dropping either half would silently
            # discard part of the supplied signal.
            units = list(direct)
            if free:
                units += C.uq_from_texts(
                    free, ix.enc, ix.V, ix.vocab, P_pool, cfg,
                    cfg.uq_file_per_text, df=ix.df,
                    n_res=len(ix.resources)).units
            seen, merged = set(), []
            for u in units:
                if u not in seen:
                    seen.add(u)
                    merged.append(u)
            sig = C.uq_from_units(merged[: cfg.kappa], ix.vocab, P_pool, cfg,
                                  a=ix.V @ ix.Q[qi], df=ix.df,
                                  n_res=len(ix.resources))
        else:
            sig = C.uq_match(ix.Q[qi], ix.V, P_pool, ix.vocab, cfg,
                             df=ix.df, n_res=len(ix.resources))

        # Interpolate U_q toward the gold ground set. Applied here, before the
        # selector dispatch, so every selector sees the same signal quality.
        if cfg.uq_oracle_mix > 0 and not cfg.uq_oracle:
            sig = C.uq_from_units(
                C.mix_units(sig.units, ix.oracle_units_for(q), cfg.uq_oracle_mix,
                            np.random.default_rng(cfg.seed + qi)),
                ix.vocab, P_pool, cfg, a=ix.V @ ix.Q[qi], df=ix.df,
                n_res=len(ix.resources))

        if cfg.selector in S.CONTROLS:
            # Non-learned control. Size comes from the same predicted profile
            # the method uses, so Table 1 rows are size-comparable and any
            # difference is selection, not budget.
            n_sel = max(1, cq + fq)
            sel = S.CONTROLS[cfg.selector](ptypes, s, ix.R[pool], n_sel,
                                           **({"fig_cap": max(fq, 1)}
                                              if cfg.selector == "modality_capped"
                                              else {"lam": cfg.mmr_lambda}
                                              if "mmr" in cfg.selector else {}))
            sel.slots_requested = dict(profile)
        elif e2e_pred is not None:
            idx = e2e_pred[qi]
            filled = {t: sum(1 for i in idx if ptypes[i] == t) for t in TYPES}
            sel = S.Selection(indices=idx, order=[], objective=0.0,
                              coverage=S.coverage_of(sig.P, sig.w, idx, cfg.coverage),
                              slots_requested=dict(profile), slots_filled=filled)
        else:
            sel = S.select(ptypes, s, sig, profile, cfg_q)
        chosen_ids = [pool_ids[i] for i in sel.indices]
        chosen_res = [ix.resources[pool[i]] for i in sel.indices]

        # ---- metrics
        gold_sets = q.gold_sets
        gold_con = [[ix.resources[ix.rid2i[x]].gold_concepts
                     for x in g if x in ix.rid2i] for g in gold_sets]
        gold_con_flat = [c for gs in gold_con for cs in gs for c in cs]
        row = {}
        # ---- OFFICIAL EduExplain ESR metrics ----
        row.update(E.official_best_metrics(set(chosen_ids), gold_sets))
        gold_concept_sets = [[c for x in g if x in ix.rid2i
                              for c in ix.resources[ix.rid2i[x]].gold_concepts]
                             for g in gold_sets]
        _pc = [c for r in chosen_res for c in r.gold_concepts]
        _cc = E.official_concept_coverage(_pc, gold_concept_sets, cfg.tau)
        # CoreCC is reported ONLY where it is well posed (>=2 gold sets and a
        # non-empty core). Leaving the key absent otherwise makes aggregate()
        # average over exactly that subset, instead of folding in a vacuous 1.0.
        if E.core_cc_included(gold_concept_sets, cfg.tau, cfg.cc_population):
            row.update(_cc)
            row["core_cc_defined"] = 1.0
        else:
            # Leave core_concept_count absent too, not just concept_coverage.
            # Setting it here averaged it over all queries while the baselines
            # averaged it over the defined subset only, so the two columns were
            # not comparable. core_cc_defined still reports the subset size.
            row["pred_concept_count"] = _cc["pred_concept_count"]
            row["core_cc_defined"] = 0.0
        row.update(E.set_concept_coverage(_pc, gold_concept_sets))
        row.update(E.attainment(set(chosen_ids), gold_sets, row["f1"]))
        row.update(E.official_recall_at_k(ix.full_ranking(qi, max(cfg.k_values)),
                                          gold_sets, cfg.k_values))
        row["n_predicted"] = len(chosen_ids)
        row["n_gold_sets"] = len(gold_sets)
        # ---- supplementary diagnostics (not part of the benchmark spec) ----
        _bm = E.best_match(chosen_ids, gold_sets)
        row["bm_precision"], row["bm_recall"] = _bm["p"], _bm["r"]
        row["em"] = _bm["em"]
        row["jaccard"] = E.jaccard(chosen_ids, gold_sets)
        row.update(E.concept_sufficiency([r.gold_concepts for r in chosen_res],
                                         [gold_con_flat]))
        # Redundancy is only defined for sets of 2+ resources: a singleton has
        # nothing to overlap with and scores 0, which would drag the mean down.
        # Gold sets are filtered the same way below, so the comparison is
        # like-for-like. Keys are simply absent for singletons and the
        # aggregator skips them.
        if len(chosen_res) > 1:
            row.update(E.redundancy([r.gold_concepts for r in chosen_res],
                                    ix.R[pool][sel.indices]))
            row["scored_for_redundancy"] = 1.0
        else:
            row["scored_for_redundancy"] = 0.0
        # How redundant are the GOLD sets themselves? If the human-validated
        # sets carry as much overlap as top-k, then driving redundancy below
        # that level moves AWAY from the target and costs F1 for a property the
        # metric does not reward. This decides whether "minimal overlap" is a
        # goal the benchmark can measure or one it penalises.
        _gr = []
        for g in gold_sets:
            gres = [ix.resources[ix.rid2i[x]] for x in g if x in ix.rid2i]
            if len(gres) > 1:
                _gr.append(E.redundancy([r.gold_concepts for r in gres],
                                        ix.R[[ix.rid2i[r.resource_id] for r in gres]]))
        if _gr:
            row["gold_concept_redundancy"] = float(np.mean([x["concept_redundancy"] for x in _gr]))
            row["gold_mean_pairwise_sim"] = float(np.mean([x["mean_pairwise_sim"] for x in _gr]))
            row["gold_unique_concepts_per_resource"] = float(
                np.mean([x["unique_concepts_per_resource"] for x in _gr]))
        row.update(E.source_diversity(chosen_res))
        _sd = [E.source_diversity([ix.resources[ix.rid2i[x]] for x in g if x in ix.rid2i])
               for g in gold_sets]
        _sd = [d for d in _sd if d]
        if _sd:
            for k2 in _sd[0]:
                row["gold_" + k2] = float(np.mean([d[k2] for d in _sd]))
        row.update(E.pool_recall(pool_ids, gold_sets))
        row.update(E.profile_metrics(cq, fq, gold_c, gold_f))
        row["set_size"] = len(chosen_ids)
        row["figure_in_pool"] = float((ptypes == "figure").any())
        row["figure_slot_filled"] = float(
            sel.slots_filled.get("figure", 0) >= profile.get("figure", 0))
        row["uq_size"] = len(sig)
        # Direct quality of the concept signal. With the oracle-U_q run scoring
        # far above the predicted-U_q run, this is the number to optimise --
        # it is measurable without rerunning selection.
        _or = set(ix.oracle_units_for(q))
        _pr = set(sig.units)
        if _or:
            _tp = len(_or & _pr)
            row["uq_precision"] = _tp / max(1, len(_pr))
            row["uq_recall"] = _tp / len(_or)
            row["uq_f1"] = (2 * _tp / (len(_pr) + len(_or))) if (_pr or _or) else 0.0
            row["uq_oracle_size"] = len(_or)
        row["objective"] = sel.objective
        row["coverage_value"] = sel.coverage
        rows.append(row)

        # ---- top-k baseline at the same size (beta -> infinity)
        k = len(chosen_ids) or 1
        tk = [pool_ids[i] for i in np.argsort(-s)[:k]]
        tkr = E.official_best_metrics(set(tk), gold_sets)
        _tpc = [c for x in tk for c in ix.resources[ix.rid2i[x]].gold_concepts]
        if E.core_cc_included(gold_concept_sets, cfg.tau, cfg.cc_population):
            tkr.update(E.official_concept_coverage(_tpc, gold_concept_sets, cfg.tau))
        tkr.update(E.set_concept_coverage(_tpc, gold_concept_sets))
        tkr.update(E.attainment(set(tk), gold_sets, tkr["f1"]))
        if len(tk) > 1:
            tkr.update(E.redundancy([ix.resources[ix.rid2i[x]].gold_concepts for x in tk],
                                    ix.R[[ix.rid2i[x] for x in tk]]))
        tkr.update(E.concept_sufficiency(
            [ix.resources[ix.rid2i[x]].gold_concepts for x in tk], [gold_con_flat]))
        tkr.update(E.source_diversity([ix.resources[ix.rid2i[x]] for x in tk]))
        topk_rows.append(tkr)

        # constant-size reference
        fk = [pool_ids[i] for i in np.argsort(-s)[:fixed_k]]
        fkr = E.official_best_metrics(set(fk), gold_sets)
        _fpc = [c for x in fk for c in ix.resources[ix.rid2i[x]].gold_concepts]
        if E.core_cc_included(gold_concept_sets, cfg.tau, cfg.cc_population):
            fkr.update(E.official_concept_coverage(_fpc, gold_concept_sets, cfg.tau))
        fkr.update(E.set_concept_coverage(_fpc, gold_concept_sets))
        fkr.update(E.attainment(set(fk), gold_sets, fkr["f1"]))
        fkr["n_predicted"] = len(fk)
        fixedk_rows.append(fkr)

        # ---- greedy optimality gap
        if cfg.exact_gap_check and qi < cfg.exact_gap_queries:
            _, best = S.exact_select(ptypes, s, sig, profile, cfg, cfg.exact_gap_topn)
            gaps.append(sel.objective / best if best > 1e-12 else 1.0)

        optiset.append({
            "query_id": q.query_id, "query": q.text,
            "predicted_set": {"resources_used": [
                {"resource_id": r.resource_id, "resource_type": r.raw_type,
                 "document_id": r.document_id, "concepts": r.gold_concepts}
                for r in chosen_res]},
        })
        preds.append({
            "query_id": q.query_id, "query": q.text, "intent": q.intent,
            "profile": profile, "selected": chosen_ids,
            "selected_types": [r.type for r in chosen_res],
            "U_q": sig.units, "best_match_f1": row["f1"],
            "objective": sel.objective, "stopped_early": sel.stopped_early,
        })
        row["_intent"] = q.intent
        row["_bundle"] = q.bundle_type

    _t_sel_total = time.perf_counter() - _t_select

    by_intent = {}
    for it in sorted({str(r.get("_intent")) for r in rows}):
        sub = [r for r in rows if str(r.get("_intent")) == it]
        a = E.aggregate(sub)
        # Carries the paper's metric set so Table 5 (per-intent, control vs
        # ours) can be read straight off this block, and so the D3 diagnostic
        # has a SetCC to subtract from.
        by_intent[it] = {k2: a[k2] for k2 in
                         ("f1", "precision", "recall", "concept_coverage",
                          "set_concept_coverage", "attainment", "n_predicted",
                          "concept_recall", "concept_redundancy", "set_size",
                          # D2 needs the COUNT head per intent as well as the
                          # figure head: "profile predictability" is both axes
                          # of pi_q, not just figure inclusion.
                          "count_exact", "count_within1", "count_abs_err",
                          "figure_correct", "figure_in_pool", "n_queries")
                         if k2 in a}
    by_bundle = {}
    for bt in sorted({str(r.get("_bundle")) for r in rows}):
        sub = [r for r in rows if str(r.get("_bundle")) == bt]
        a = E.aggregate(sub)
        by_bundle[bt] = {k2: a[k2] for k2 in
                         ("f1", "precision", "recall", "concept_coverage",
                          "set_concept_coverage", "attainment", "n_predicted",
                          "n_queries") if k2 in a}

    metrics = {
        "system": E.aggregate(rows),
        "by_intent": by_intent,
        "by_bundle_type": by_bundle,
        "topk_baseline": E.aggregate(topk_rows),
        "topk_fixed_k_baseline": {**E.aggregate(fixedk_rows), "k": fixed_k},
        "topk_fixed_k_baseline": {**E.aggregate(fixedk_rows), "k": fixed_k},
        "role_head_vs_majority": {
            "head_count_exact": float(np.mean([
                pred_c[i] in ix.role_targets(i)[0] for i in range(nq)])),
            "majority_count_exact": float(np.mean([
                base_c[i] in ix.role_targets(i)[0] for i in range(nq)])),
            "head_count_within1": float(np.mean([
                any(abs(pred_c[i] - g) <= 1 for g in ix.role_targets(i)[0])
                for i in range(nq)])),
            "head_figure": E.binary_f1(pred_f, [ix.mean_targets(i)[1] for i in range(nq)]),
            "majority_figure": E.binary_f1(base_f, [ix.mean_targets(i)[1] for i in range(nq)]),
        },
        "vocabulary_size": len(ix.vocab),
        **({"intent_classifier_accuracy": float(np.mean(
            [pred_intent[i] == str(ix.queries[i].intent) for i in range(nq)]))}
           if cfg.profile_mode == "predicted_intent" else {}),
        **({"e2e": e2e_info} if e2e_info else {}),
        **({"concept_head": concept_info} if concept_info else {}),
        **({"con_classifier": ix.con_diag} if getattr(ix, "con_diag", None) else {}),
        "mean_Con_size": float(np.mean([len(c) for c in ix.Con])),
        # Efficiency. No LLM is involved at any stage, so the call counts are
        # structurally zero -- that is the comparison, not an omission. Encoder
        # passes are counted as the embedding work the method actually needs.
        "efficiency": {
            "wall_clock_total_s": round(time.perf_counter() - _t_start, 3),
            "wall_clock_index_s": round(_t_index, 3),
            "wall_clock_selection_s": round(_t_sel_total, 3),
            "selection_s_per_query": round(_t_sel_total / max(1, nq), 5),
            "llm_calls_total": 0, "llm_calls_per_query": 0.0,
            "llm_prompt_tokens": 0, "llm_generated_tokens": 0,
            "encoder_passes_per_query": 1,
            "n_queries": nq,
        },
    }
    if gaps:
        metrics["greedy_optimality"] = {
            "mean_ratio_vs_restricted_exact": float(np.mean(gaps)),
            "min_ratio": float(np.min(gaps)), "n_checked": len(gaps),
        }

    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.out_dir, "config.json"))
    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(cfg.out_dir, "predictions.jsonl"), "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    # optiset format, so evaluation/evaluator.py can be run on this directly
    # without depending on the reimplementation above.
    with open(os.path.join(cfg.out_dir, "optiset_predictions.json"), "w") as f:
        json.dump(optiset, f, indent=1)
    if not quiet:
        print(json.dumps(metrics, indent=2))
    return metrics


# ------------------------------------------------------------------ ablations
ABLATIONS = {
    # --- Table 4 rows that remove a component outright ---
    # "- role prediction": composition is no longer inferred per query; the
    # profile is held constant at the benchmark's mean gold set size.
    "no_role":          dict(profile_mode="fixed", fixed_profile="2,1"),
    # "- concept-coverage term": beta so large that the objective is pure
    # relevance under the predicted slot constraints.
    "no_coverage":      dict(beta=1e6),
    # --- non-learned selection controls (Table 1) at the same size ---
    "ctrl_topn":            dict(selector="topn"),
    "ctrl_modality_capped": dict(selector="modality_capped"),
    "ctrl_mmr":             dict(selector="mmr"),
    "ctrl_mmr_modality":    dict(selector="mmr_modality"),
    "uq_anchored":      dict(uq_mode="anchored"),
    "uq_match":         dict(uq_mode="match"),
    "uq_learned":       dict(uq_mode="learned", concept_target="gold_con"),
    "uq_learned_query": dict(uq_mode="learned", concept_target="query"),
    # --- U_q from a trained classifier instead of cosine matching ---
    # Both protocols the silver/gold distribution gap leaves open. The
    # size-matched decode (concept_topk = kappa) is what makes these comparable
    # with uq_match at the same kappa.
    "uq_clf_cv": dict(uq_mode="learned", concept_train="cv",
                      concept_target="gold_con", concept_topk=12),
    "uq_clf_silver_cv": dict(uq_mode="learned", concept_train="silver_then_cv",
                             concept_target="gold_con", concept_topk=12),
    "uq_clf_silver": dict(uq_mode="learned", concept_train="silver",
                          concept_target="gold_con", concept_topk=12),
    # --- both sides learned: rho_q for U_q, a classifier for Con(r) ---
    "both_clf_cv": dict(uq_mode="learned", concept_train="cv",
                        concept_target="gold_con", concept_topk=12,
                        con_source="learned"),
    "both_clf_silver_cv": dict(uq_mode="learned",
                               concept_train="silver_then_cv",
                               concept_target="gold_con", concept_topk=12,
                               con_source="learned"),
    # --- resource side only, for the 2x2 against con_source ---
    "con_learned": dict(con_source="learned"),
    "uq_prf":           dict(uq_mode="prf"),
    "uq_kappa_20":      dict(kappa=20),
    "uq_kappa_30":      dict(kappa=30),
    "uq_gamma_0.70":    dict(gamma=0.70),
    "uq_gamma_0.90":    dict(gamma=0.90),
    "cov_binary":       dict(coverage="binary"),
    "cov_graded":       dict(coverage="graded_max"),
    "cov_noisy_or":     dict(coverage="noisy_or"),
    "greedy_block":     dict(greedy="block"),
    "greedy_global":    dict(greedy="global"),
    "beta_0":           dict(beta=0.0),
    "beta_10":          dict(beta=10.0),
    "profile_oracle":   dict(profile_mode="oracle", allow_oracles=True),
    "profile_intent":   dict(profile_mode="intent", allow_oracles=True),
    "profile_pred_intent": dict(profile_mode="predicted_intent"),
    "oracle_both":      dict(profile_mode="oracle", uq_oracle=True,
                             allow_oracles=True),

    "profile_fixed_31": dict(profile_mode="fixed", fixed_profile="3,1"),
    "uq_oracle":        dict(uq_oracle=True, allow_oracles=True),
    "stratified_pool":  dict(stratified_first_stage=True),
    # Eq. 9 aggregation over the gold sets, for the role head and the e2e loss.
    "gold_best_match":   dict(role_loss="best_match"),
    "gold_single_first": dict(role_loss="single_first"),
    "gold_single_random": dict(role_loss="single_random"),
    "gold_average":      dict(role_loss="average"),
    "no_type_constraint": dict(type_constraint=False),
}


def audit(cfg: Config) -> Dict:
    """Leakage audit between the training file and the evaluation file.

    Uses construction-time fields (bundle_id, variant_style, gold sets) that the
    model never sees. Auditing a split with them is not the same as feeding them
    to a predictor -- this is how you demonstrate the split is clean.
    """
    tr = load_queries(cfg.train_queries_path) if cfg.train_queries_path else []
    te = load_queries(cfg.queries_path)
    if not tr:
        return {"error": "no --train-queries given; nothing to audit"}

    def sig(q):
        return tuple(sorted(tuple(sorted(s)) for s in q.gold_sets))

    # --- distribution shift: the model is fit on `tr` and scored on `te`, so
    # an intent present in the evaluation set but absent from training is a
    # direct explanation for that slice underperforming.
    def dist(qs, key):
        d = {}
        for q in qs:
            d[str(key(q))] = d.get(str(key(q)), 0) + 1
        return {k: round(v / max(1, len(qs)), 4) for k, v in sorted(d.items())}

    def gold_stats(qs):
        sz, ex, fg, nc = [], [], [], []
        for q in qs:
            nc.append(len(q.gold_concepts))
            for g in q.gold_sets:
                sz.append(len(g))
        return {"mean_gold_set_size": round(float(np.mean(sz)) if sz else 0, 3),
                "mean_query_concepts": round(float(np.mean(nc)) if nc else 0, 3)}

    tr_int, te_int = dist(tr, lambda q: q.intent), dist(te, lambda q: q.intent)
    missing = sorted(k for k in te_int if tr_int.get(k, 0.0) == 0.0)
    thin = sorted(k for k in te_int
                  if 0.0 < tr_int.get(k, 0.0) < 0.5 * te_int[k])

    # --- vocabulary coverage of the query-level concept annotations
    res = load_resources(cfg.resources_path)
    v = build_vocabulary(res, cfg.vocab_source, cfg.vocab_min_df,
                         cfg.vocab_max_df_frac, cfg.vocab_drop_structural,
                         cfg.vocab_drop_generic, cfg.vocab_context_descriptors,
                         cfg.vocab_structural_frac)
    vs = set(v.ids)

    def vcov(qs):
        tot = [c for q in qs for c in q.gold_concepts]
        if not tot:
            return {"n_query_concepts": 0, "frac_in_vocabulary": None}
        return {"n_query_concepts": len(tot),
                "frac_in_vocabulary": round(sum(c in vs for c in tot) / len(tot), 4),
                "examples_missing": sorted({c for c in tot if c not in vs})[:10]}

    tb = {q.bundle_id for q in tr if q.bundle_id}
    eb = {q.bundle_id for q in te if q.bundle_id}
    ts, es = {sig(q) for q in tr}, {sig(q) for q in te}
    tqid, eqid = {q.query_id for q in tr}, {q.query_id for q in te}
    ttxt = {" ".join(q.text.lower().split()) for q in tr}
    etxt = {" ".join(q.text.lower().split()) for q in te}
    tres = {r for q in tr for g in q.gold_sets for r in g}
    eres = {r for q in te for g in q.gold_sets for r in g}

    rep = {
        "n_train": len(tr), "n_eval": len(te),
        "shared_query_ids": len(tqid & eqid),
        "shared_query_text": len(ttxt & etxt),
        "shared_bundle_ids": len(tb & eb),
        "frac_eval_bundles_seen_in_train": (len(tb & eb) / len(eb)) if eb else None,
        "shared_gold_set_signatures": len(ts & es),
        "frac_eval_signatures_seen_in_train": len(ts & es) / max(1, len(es)),
        "shared_gold_resources": len(tres & eres),
        "frac_eval_resources_seen_in_train": len(tres & eres) / max(1, len(eres)),
        "examples_shared_bundles": sorted(tb & eb)[:10],
        "distribution_shift": {
            "train_intents": tr_int, "eval_intents": te_int,
            "eval_intents_absent_from_train": missing,
            "eval_intents_underrepresented_in_train": thin,
            "train_variant_styles": dist(tr, lambda q: q.variant_style),
            "eval_variant_styles": dist(te, lambda q: q.variant_style),
            "train_gold": gold_stats(tr), "eval_gold": gold_stats(te),
        },
        "vocabulary_coverage_of_query_concepts": {
            "vocabulary_size": len(v),
            "train": vcov(tr), "eval": vcov(te),
        },
    }
    # Shared gold RESOURCES are expected and fine -- both files index the same
    # corpus. Only shared queries/items are leakage.
    problems = []
    if ttxt & etxt:
        problems.append(f"{len(ttxt & etxt)} identical query texts across files")
    if tb & eb:
        problems.append(f"{len(tb & eb)} bundle_ids in both files")
    rep["verdict"] = "CLEAN" if not problems else "LEAK: " + "; ".join(problems)
    if missing:
        rep["warning"] = (
            "intents in the evaluation set with NO training examples: "
            + ", ".join(missing) + ". Anything trained (the role head, and the "
            "concept head under --uq-mode learned) is extrapolating on those "
            "slices. Note the default --uq-mode match is training-free, so it "
            "is NOT affected by this.")
    # Duplicate ids WITHIN one file DO matter -- predictions are keyed by them.
    rep["duplicate_query_ids_within_train"] = len(tr) - len(tqid)
    rep["duplicate_query_ids_within_eval"] = len(te) - len(eqid)
    rep["note"] = (
        "shared_gold_resources is expected (same corpus) and is NOT leakage. "
        "shared_query_ids across files is informational only -- ids generated "
        "independently per file can collide by chance; shared query TEXT and "
        "shared gold-set signatures are the real signals. Duplicate ids WITHIN "
        "a file do matter, since predictions are keyed by query_id.")
    print(json.dumps(rep, indent=2))
    return rep


def eval_file(cfg: Config, pred_path: str, label: str = "") -> Dict:
    """Score an EXTERNAL predictions file with exactly the metric code the
    pipeline uses for its own output.

    Takes the optiset format that `run` also writes:

        [{"query_id": "...",
          "predicted_set": {"resources_used": [{"resource_id": "..."}, ...]}}]

    A flat {"query_id": ..., "resources": [...]} or {"selected": [id, ...]} is
    accepted too. Concepts are read from resources.json rather than trusted
    from the prediction file, so a competitor cannot score differently just by
    carrying different concept annotations.

    Use this for SetR, OptiSet and any other external system: every row of
    Table 3 is then computed by one implementation, and differences are the
    methods rather than the evaluators.
    """
    resources = load_resources(cfg.resources_path)
    queries = load_queries(cfg.queries_path)
    by_id = {r.resource_id: r for r in resources}
    gold = {q.query_id: q for q in queries}

    with open(pred_path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("predictions", raw.get("results", []))

    rows, missing, unknown = [], 0, 0
    for item in raw:
        qid = str(item.get("query_id"))
        if qid not in gold:
            missing += 1
            continue
        q = gold[qid]
        ru = (item.get("predicted_set") or {}).get("resources_used")
        if ru is None:
            ru = item.get("resources") or item.get("selected") or []
        ids = [str(r.get("resource_id") if isinstance(r, dict) else r) for r in ru]
        unknown += sum(1 for x in ids if x not in by_id)
        ids = [x for x in ids if x in by_id]

        gold_sets = q.gold_sets
        gcs = [[c for x in g if x in by_id for c in by_id[x].gold_concepts]
               for g in gold_sets]
        pc = [c for x in ids for c in by_id[x].gold_concepts]

        row = E.official_best_metrics(set(ids), gold_sets)
        if E.core_cc_included(gcs, cfg.tau, cfg.cc_population):
            row.update(E.official_concept_coverage(pc, gcs, cfg.tau))
            row["core_cc_defined"] = 1.0
        else:
            row["core_cc_defined"] = 0.0
        row.update(E.set_concept_coverage(pc, gcs))
        row.update(E.attainment(set(ids), gold_sets, row["f1"]))
        row["n_predicted"] = len(ids)
        row["_intent"] = q.intent
        rows.append(row)

    agg = E.aggregate(rows)
    by_intent = {}
    for it in sorted({str(r.get("_intent")) for r in rows}):
        sub = [r for r in rows if str(r.get("_intent")) == it]
        a = E.aggregate(sub)
        by_intent[it] = {k: a[k] for k in
                         ("f1", "precision", "recall", "concept_coverage",
                          "set_concept_coverage", "n_predicted", "n_queries")
                         if k in a}
    out = {"label": label or os.path.basename(pred_path),
           "predictions_file": pred_path, "system": agg, "by_intent": by_intent,
           "queries_not_in_gold": missing, "unknown_resource_ids": unknown}
    print(json.dumps(out, indent=2))
    return out


def vocab_report(cfg: Config) -> Dict:
    """Diagnose V before running anything: size, df distribution, hubs, junk."""
    res = load_resources(cfg.resources_path)
    v = build_vocabulary(res, cfg.vocab_source, cfg.vocab_min_df,
                         cfg.vocab_max_df_frac, cfg.vocab_drop_structural,
                         cfg.vocab_drop_generic, cfg.vocab_context_descriptors,
                         cfg.vocab_structural_frac)
    raw = build_vocabulary(res, cfg.vocab_source, 1, 1.0, False, False,
                           cfg.vocab_context_descriptors, cfg.vocab_structural_frac)
    n = max(1, len(res))
    df = np.array(v.df, dtype=float) if v.df else np.zeros(1)
    order = np.argsort(-df)
    rep = {
        "vocab_source": cfg.vocab_source,
        "size_before_filtering": len(raw),
        "size_after_filtering": len(v),
        "n_structural_kept": int(sum(v.is_structural)),
        "frac_structural_kept": float(np.mean(v.is_structural)) if len(v) else 0.0,
        "df_mean": float(df.mean()), "df_median": float(np.median(df)),
        "df_p95": float(np.percentile(df, 95)), "df_max": float(df.max()),
        "singletons": int((df <= 1).sum()),
        "concepts_per_resource": float(np.mean([len(r.gold_concepts) for r in res])),
        "dropped": {k: (x[:20] + ["..."] if len(x) > 20 else x)
                    for k, x in v.dropped.items()} if v.dropped else {},
        "n_dropped_by_reason": {k: len(x) for k, x in v.dropped.items()},
        "top_df_concepts": [
            {"concept": v.ids[i], "df": int(df[i]),
             "frac_corpus": round(float(df[i] / n), 3),
             "structural": bool(v.is_structural[i]),
             "struct_frac": float(v.structural_frac[i]) if v.structural_frac else 0.0}
            for i in order[:20]],
        "sample_descriptors": v.descriptors[:8],
    }
    print(json.dumps(rep, indent=2))
    return rep


def _hdr() -> None:
    print(f"{'variant':22s} {'GS-F1':>7s} {'GS-P':>7s} {'GS-R':>7s} "
          f"{'CoreCC':>7s} {'SetCC':>7s} {'attain':>7s} {'size':>6s}")


def _line(name: str, m: Dict) -> None:
    print(f"{name:22s} {m['f1']:7.4f} {m['precision']:7.4f} {m['recall']:7.4f} "
          f"{m.get('concept_coverage', 0.0):7.4f} {m['set_concept_coverage']:7.4f} "
          f"{m.get('attainment', 0.0):7.4f} {m['n_predicted']:6.2f}")


def ablate(cfg: Config, only: Optional[Sequence[str]] = None) -> Dict:
    """Run the ablation table, or just the named rows with --only.

    --only also accepts rows that are not in ABLATIONS at all, as long as the
    name is a preset: that is how the concept-classifier rows get run next to
    `full` without re-running the whole 29-row table.
    """
    ix = Index(cfg)
    out = {}
    if only:
        # "full" is not an ablation -- it is the unmodified config -- but being
        # able to name it here is what makes a two-row comparison one command.
        ABLATIONS.setdefault("full", dict())
        miss = [n for n in only if n not in ABLATIONS]
        if miss:
            raise SystemExit("unknown ablation row(s): " + ", ".join(miss)
                             + "\nknown rows: " + ", ".join(ABLATIONS))
        rows = [(n, ABLATIONS[n]) for n in only]
    else:
        rows = list(ABLATIONS.items())
    _hdr()
    for name, over in rows:
        c = Config(**{**cfg.__dict__, **over,
                      "out_dir": os.path.join(cfg.out_dir, "ablate", name)})
        out[name] = run(c, ix.rebind(c), quiet=True)["system"]
        _line(name, out[name])
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "ablations.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


_INTENT_PRETTY = {
    "why_how": "Why/How", "whyhow": "Why/How", "why": "Why/How",
    "visual": "Visual", "visual_aid": "Visual", "diagram": "Visual",
    "example": "Example", "practice": "Practice", "exercise": "Practice",
    "explanation": "Explanation", "conceptual": "Explanation",
    "revision": "Revision", "review": "Revision", "summary": "Revision",
    "misconception": "Misconception", "misconceptions": "Misconception",
}


def _pretty_intent(raw: str) -> str:
    k = str(raw).strip().lower()
    if k in _INTENT_PRETTY:
        return _INTENT_PRETTY[k]
    return " ".join(w.capitalize() for w in k.replace("-", "_").split("_"))


def diagnostics(cfg: Config, n_samples: int = 60,
                intent_order: Optional[Sequence[str]] = None,
                latex: bool = False) -> Dict:
    """Table 2: the three premise checks, per intent.

    D1 -- correlation between a candidate set's mean pairwise DISSIMILARITY and
          its pedagogical value (GS-F1, and SetCC). Sampled over random subsets
          of the pool at the query's gold size, Spearman within a query, then
          averaged. A correlation near zero or negative is the evidence that
          content distance is the wrong signal, which is what motivates
          replacing diversity with role composition.
    D2 -- how predictable the composition is from the query: the gold
          figure-inclusion rate and the role head's accuracy, per intent.
    D3 -- the SetCC minus GS-F1 gap, which localises queries where the pipeline
          recovers the right concepts but not the annotator's exact resource.
    """
    from scipy.stats import spearmanr

    ix = Index(cfg)
    rng = np.random.default_rng(cfg.seed)
    per_intent: Dict[str, Dict[str, List[float]]] = {}
    # Pooled over ALL queries, for the "All" row. Averaging the per-intent means
    # instead would weight a 24-query intent like a 66-query one.
    pooled: Dict[str, List[float]] = {"r_f1": [], "r_cc": [], "fig": []}

    for qi, q in enumerate(ix.queries):
        pool, s = ix.retrieve(qi)
        pool_ids = [ix.resources[i].resource_id for i in pool]
        gold_sets = q.gold_sets
        if not gold_sets:
            continue
        n = max(1, int(round(np.mean([len(g) for g in gold_sets]))))
        Emb = ix.R[pool]
        gcs = [[c for x in g if x in ix.rid2i
                for c in ix.resources[ix.rid2i[x]].gold_concepts] for g in gold_sets]

        dis, f1s, ccs = [], [], []
        for _ in range(n_samples):
            pick = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
            ids = {pool_ids[i] for i in pick}
            if len(pick) > 1:
                Sub = Emb[pick] @ Emb[pick].T
                iu = np.triu_indices(len(pick), k=1)
                dis.append(1.0 - float(Sub[iu].mean()))
            else:
                dis.append(0.0)
            f1s.append(E.official_best_metrics(ids, gold_sets)["f1"])
            pc = [c for x in ids for c in ix.resources[ix.rid2i[x]].gold_concepts]
            ccs.append(E.set_concept_coverage(pc, gcs)["set_concept_coverage"])

        key = str(q.intent)
        d = per_intent.setdefault(key, {"r_f1": [], "r_cc": [], "fig": [],
                                        "gap": [], "n": []})
        _fig = float(np.mean(
            [any(ix.resources[ix.rid2i[x]].type == "figure"
                 for x in g if x in ix.rid2i) for g in gold_sets]))
        if np.std(dis) > 1e-9 and np.std(f1s) > 1e-9:
            r = float(spearmanr(dis, f1s).statistic)
            d["r_f1"].append(r)
            pooled["r_f1"].append(r)
        if np.std(dis) > 1e-9 and np.std(ccs) > 1e-9:
            r = float(spearmanr(dis, ccs).statistic)
            d["r_cc"].append(r)
            pooled["r_cc"].append(r)
        d["fig"].append(_fig)
        pooled["fig"].append(_fig)
        d["n"].append(1.0)

    # D2/D3 come from a normal run, so the numbers match the main table exactly.
    m = run(cfg, ix.rebind(cfg), quiet=True)
    bi = m["by_intent"]

    rep = {}
    print(f"\nfirst stage: {cfg.encoder_name}"
          f"{'' if cfg.encoder_backend == 'sentence_transformers' else ' [backend: %s]' % cfg.encoder_backend}"
          f", K = {cfg.K}"
          f"{' (stratified, K_figure=%d)' % cfg.K_figure if cfg.stratified_first_stage else ''}"
          f"   |  subsets sampled per query: {n_samples}"
          f"   |  Con(r): {cfg.con_source}")
    print(f"raw intent labels found: {sorted(per_intent)}\n")

    if intent_order:
        known = [i for i in intent_order if i in per_intent]
        order = known + [i for i in sorted(per_intent) if i not in known]
        missing = [i for i in intent_order if i not in per_intent]
        if missing:
            print(f"WARNING: --intent-order names not present in the data: "
                  f"{missing}\n")
    else:
        order = sorted(per_intent)

    print(f"{'Intent':14s} {'D1 r(dis,F1)':>13s} {'D1 r(dis,CC)':>13s} "
          f"{'D2 fig-rate':>12s} {'D2 fig-acc':>11s} {'D2 cnt-exact':>13s} "
          f"{'D2 cnt-+-1':>11s} {'D3 SetCC-F1':>12s} {'n':>4s}")
    for it in order:
        d = per_intent[it]
        b = bi.get(it, {})
        row = {
            "d1_corr_dissim_f1": float(np.mean(d["r_f1"])) if d["r_f1"] else None,
            "d1_corr_dissim_setcc": float(np.mean(d["r_cc"])) if d["r_cc"] else None,
            "d2_gold_figure_rate": float(np.mean(d["fig"])),
            "d2_figure_accuracy": b.get("figure_correct"),
            "d2_count_exact": b.get("count_exact"),
            "d2_count_within1": b.get("count_within1"),
            "d3_setcc_minus_f1": (
                (b["set_concept_coverage"] - b["f1"])
                if ("set_concept_coverage" in b and "f1" in b) else None),
            "n_queries": int(sum(d["n"])),
        }
        rep[it] = row
        _d3 = row["d3_setcc_minus_f1"]
        _d3 = float("nan") if _d3 is None else _d3
        print(f"{it:14s} {row['d1_corr_dissim_f1']:>13.4f} "
              f"{row['d1_corr_dissim_setcc']:>13.4f} "
              f"{row['d2_gold_figure_rate']:>12.4f} "
              f"{(row['d2_figure_accuracy'] or 0):>11.4f} "
              f"{(row['d2_count_exact'] or 0):>13.4f} "
              f"{(row['d2_count_within1'] or 0):>11.4f} "
              f"{_d3:>12.4f} {row['n_queries']:>4d}")

    # ---- the "All" row, pooled over every query rather than averaged over
    # the per-intent means, which would weight a 24-query intent like a 66.
    sysm = m["system"]
    allrow = {
        "d1_corr_dissim_f1": float(np.mean(pooled["r_f1"])) if pooled["r_f1"] else None,
        "d1_corr_dissim_setcc": float(np.mean(pooled["r_cc"])) if pooled["r_cc"] else None,
        "d2_gold_figure_rate": float(np.mean(pooled["fig"])) if pooled["fig"] else None,
        "d2_figure_accuracy": sysm.get("figure_correct"),
        "d2_count_exact": sysm.get("count_exact"),
        "d2_count_within1": sysm.get("count_within1"),
        "d3_setcc_minus_f1": (
            (sysm["set_concept_coverage"] - sysm["f1"])
            if ("set_concept_coverage" in sysm and "f1" in sysm) else None),
        "n_queries": int(sum(len(per_intent[i]["n"]) for i in per_intent)),
    }
    rep["__all__"] = allrow
    print("-" * 104)
    print(f"{'All':14s} {allrow['d1_corr_dissim_f1']:>13.4f} "
          f"{allrow['d1_corr_dissim_setcc']:>13.4f} "
          f"{allrow['d2_gold_figure_rate']:>12.4f} "
          f"{(allrow['d2_figure_accuracy'] or 0):>11.4f} "
          f"{(allrow['d2_count_exact'] or 0):>13.4f} "
          f"{(allrow['d2_count_within1'] or 0):>11.4f} "
          f"{(allrow['d3_setcc_minus_f1'] or 0):>12.4f} "
          f"{allrow['n_queries']:>4d}")

    if latex:
        print("\n% ---- Table 1 body. Column order: D1 vs GS-F1, D1 vs SetCC,")
        print("%      D2 fig. rate, fig. acc., count acc., D3 SetCC-GS-F1, n.")
        def _f(v, sgn=False):
            if v is None:
                return "--"
            return (f"${v:+.3f}$" if sgn else f"{v:.3f}")
        for it in order:
            r = rep[it]
            print(f"{_pretty_intent(it):<14s} & {_f(r['d1_corr_dissim_f1'], True)} "
                  f"& {_f(r['d1_corr_dissim_setcc'], True)} "
                  f"& {_f(r['d2_gold_figure_rate'])} "
                  f"& {_f(r['d2_figure_accuracy'])} "
                  f"& {_f(r['d2_count_exact'])} "
                  f"& {_f(r['d3_setcc_minus_f1'])} "
                  f"& {r['n_queries']} \\\\")
        print("\\midrule")
        r = allrow
        print(f"{'All':<14s} & {_f(r['d1_corr_dissim_f1'], True)} "
              f"& {_f(r['d1_corr_dissim_setcc'], True)} "
              f"& {_f(r['d2_gold_figure_rate'])} "
              f"& {_f(r['d2_figure_accuracy'])} "
              f"& {_f(r['d2_count_exact'])} "
              f"& {_f(r['d3_setcc_minus_f1'])} "
              f"& {r['n_queries']} \\\\")

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "diagnostics.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {os.path.join(cfg.out_dir, 'diagnostics.json')}")
    return rep


def sweep_e2e(cfg: Config) -> Dict:
    """Small capacity/regularisation sweep for the end-to-end variant.

    The configuration is chosen by DEV F1 -- a slice held out of the TRAINING
    queries -- and only that one configuration's evaluation number is meant to
    be reported. Picking the config with the best evaluation F1 would be
    test-set tuning, and it would also hollow out the negative result: "the
    heavy model does not pay off at this data scale" is only credible if the
    heavy model was given a fair search, and only honest if the search never
    saw the test queries.
    """
    grid = [
        dict(e2e_d_model=128, e2e_layers=1, e2e_dropout=0.3, e2e_lr=3e-4),
        dict(e2e_d_model=128, e2e_layers=1, e2e_dropout=0.5, e2e_lr=3e-4),
        dict(e2e_d_model=128, e2e_layers=1, e2e_dropout=0.3, e2e_lr=1e-4),
        dict(e2e_d_model=64,  e2e_layers=1, e2e_dropout=0.3, e2e_lr=3e-4),
        dict(e2e_d_model=256, e2e_layers=2, e2e_dropout=0.3, e2e_lr=3e-4),
        dict(e2e_d_model=128, e2e_layers=1, e2e_dropout=0.3, e2e_lr=3e-4,
             e2e_null_weight=0.5),
        dict(e2e_d_model=128, e2e_layers=1, e2e_dropout=0.3, e2e_lr=3e-4,
             e2e_null_weight=2.0),
    ]
    ix = Index(cfg)
    out, best = {}, None
    print(f"{'d_model':>8s} {'layers':>7s} {'drop':>5s} {'lr':>7s} {'null_w':>7s} "
          f"{'DEV F1':>7s} {'ep':>4s} {'train':>7s} {'EVAL':>7s} {'|R|':>5s}")
    for i, g in enumerate(grid):
        c = Config(**{**cfg.__dict__, **g, "selector": "e2e",
                      "out_dir": os.path.join(cfg.out_dir, "e2e", str(i))})
        m = run(c, ix.rebind(c), quiet=True)
        e, sysm = m.get("e2e", {}), m["system"]
        row = {"config": g, "dev_f1": e.get("best_dev_f1"),
               "best_epoch": e.get("best_epoch"), "train_f1": e.get("train_f1"),
               "eval_f1": sysm["f1"], "set_size": sysm["n_predicted"]}
        out[str(i)] = row
        print(f"{g['e2e_d_model']:>8d} {g['e2e_layers']:>7d} "
              f"{g['e2e_dropout']:>5.1f} {g['e2e_lr']:>7.0e} "
              f"{g.get('e2e_null_weight', cfg.e2e_null_weight):>7.2f} "
              f"{(row['dev_f1'] or 0):>7.4f} {(row['best_epoch'] or -1):>4d} "
              f"{(row['train_f1'] or 0):>7.4f} {row['eval_f1']:>7.4f} "
              f"{row['set_size']:>5.2f}")
        if best is None or (row["dev_f1"] or -1) > (out[best]["dev_f1"] or -1):
            best = str(i)
    out["selected_by_dev"] = best
    print(f"\nselected on DEV F1: config {best} -> report EVAL F1 "
          f"{out[best]['eval_f1']:.4f} at |R|={out[best]['set_size']:.2f}")
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "e2e_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def sweep_k(cfg: Config, ks: Sequence[int] = (25, 50, 100, 200)) -> Dict:
    """Behaviour as the candidate pool grows, for the RQ4 pool-size check.

    A larger pool raises the recall ceiling but also hands the selector more
    ways to go wrong, so the two curves separate: Recall@K keeps climbing while
    GS-F1 need not. Where GS-F1 flattens or turns over is the point past which
    more retrieval stops helping this selector.
    """
    ix = Index(cfg)
    out: Dict[str, Dict] = {}
    print(f"{'K':>5s} {'GS-F1':>7s} {'top-n':>7s} {'delta':>7s} {'CoreCC':>7s} "
          f"{'pool_rec':>9s} {'|R|':>5s}")
    for k in ks:
        c = Config(**{**cfg.__dict__, "K": int(k),
                      "out_dir": os.path.join(cfg.out_dir, "K", str(k))})
        m = run(c, ix.rebind(c), quiet=True)
        sysm, tk = m["system"], m["topk_baseline"]
        out[str(k)] = {"system": sysm, "topk": tk}
        print(f"{k:>5d} {sysm['f1']:>7.4f} {tk['f1']:>7.4f} "
              f"{sysm['f1'] - tk['f1']:>+7.4f} "
              f"{sysm.get('concept_coverage', 0.0):>7.4f} "
              f"{sysm['pool_recall_best']:>9.4f} {sysm['n_predicted']:>5.2f}")
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "k_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def sweep_uq(cfg: Config,
             mixes: Sequence[float] = tuple(round(0.1 * i, 1) for i in range(11))
             ) -> Dict:
    """GS-F1 against concept-signal quality, with top-n as the crossing line.

    The ablation shows the coverage term helping with gold concepts and hurting
    with predicted ones, so the open question is not whether the mechanism works
    but how accurate the concept signal has to be before it pays. This sweep
    answers that directly: it interpolates U_q from predicted to gold and
    reports, at each step, the achieved U_q F1 alongside GS-F1. The mix where
    the method's curve crosses top-n is the accuracy target a deployable
    concept predictor has to hit.
    """
    ix = Index(cfg)
    out: Dict[str, Dict] = {}
    base = None
    print(f"{'mix':>5s} {'U_q F1':>7s} {'U_q P':>7s} {'U_q R':>7s} "
          f"{'GS-F1':>7s} {'top-n':>7s} {'delta':>7s} {'CoreCC':>7s}")
    for mx in mixes:
        c = Config(**{**cfg.__dict__, "uq_oracle_mix": float(mx),
                      "out_dir": os.path.join(cfg.out_dir, "uqmix", str(mx))})
        m = run(c, ix.rebind(c), quiet=True)
        sysm, tk = m["system"], m["topk_baseline"]
        base = base if base is not None else tk["f1"]
        out[str(mx)] = {"system": sysm, "topk": tk}
        print(f"{mx:>5.1f} {sysm['uq_f1']:>7.4f} {sysm['uq_precision']:>7.4f} "
              f"{sysm['uq_recall']:>7.4f} {sysm['f1']:>7.4f} {tk['f1']:>7.4f} "
              f"{sysm['f1'] - tk['f1']:>+7.4f} "
              f"{sysm.get('concept_coverage', 0.0):>7.4f}")
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "uq_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def error_breakdown(cfg: Config, run_dir: str) -> Dict:
    """Where precision is lost: wrong size, or wrong resource at the right size.

    Reads a finished run's predictions.jsonl, so it works on any selector
    (lightweight, control, e2e, hybrid) with no re-encoding.

    Each predicted resource that is not in the best-matching gold set G* is put
    in exactly one bucket:

      alt_gold     -- it IS in another gold set for this query. The metric scores
                      against one gold set at a time, so a set assembled from two
                      equally valid explanations is penalised as if it were wrong.
                      This is a property of the metric, not an error of the model.
      concept_ok   -- not in any gold set, but it covers at least one concept the
                      gold sets require. "Right concept, wrong resource."
      concept_miss -- covers none of the required concepts. A genuine miss.

    Two counterfactuals separate the size question from the choice question:

      P@|G*|  -- precision after truncating the prediction to the size of the
                 best gold set, keeping the selector's own ordering. The gap to
                 actual precision is what over-prediction costs.
      P_union -- precision against the UNION of all gold sets. The gap to actual
                 precision is what assembling across gold sets costs.
    """
    res = load_resources(cfg.resources_path)
    qs = load_queries(cfg.queries_path)
    rid2i = {r.resource_id: i for i, r in enumerate(res)}
    path = os.path.join(run_dir, "predictions.jsonl")
    preds = [json.loads(l) for l in open(path)]
    if len(preds) != len(qs):
        raise SystemExit(f"{path} has {len(preds)} rows but the query file has "
                         f"{len(qs)}; they must come from the same evaluation file.")

    def con(rid):
        i = rid2i.get(rid)
        return {slugify(c) for c in (res[i].gold_concepts or [])} if i is not None else set()

    rows = []
    for q, pr in zip(qs, preds):
        pred = list(pr.get("selected") or [])
        gold_sets = [list(g) for g in q.gold_sets if g]
        if not gold_sets:
            continue
        gsets = [set(g) for g in gold_sets]
        union = set().union(*gsets)
        gold_con_sets = [set().union(*[con(x) for x in g]) if g else set() for g in gold_sets]
        need = set().union(*gold_con_sets) if gold_con_sets else set()
        core = {c for c in E.core_concepts([sorted(x) for x in gold_con_sets], cfg.tau)}

        P = set(pred)
        # G* = the gold set this prediction matches best on F1, the same set the
        # official GoldSet-F1 reports against.
        def f1(g):
            inter = len(P & g)
            return (2 * inter / (len(P) + len(g))) if (P or g) else 0.0
        gstar = max(gsets, key=f1)

        tp = P & gstar
        fps = [x for x in pred if x not in gstar]
        alt = [x for x in fps if x in union]
        rest = [x for x in fps if x not in union]
        cok = [x for x in rest if con(x) & need]
        cmiss = [x for x in rest if not (con(x) & need)]
        ccore = [x for x in rest if con(x) & core]

        trunc = pred[: len(gstar)]                 # selector's own order
        row = {
            "n_pred": len(pred), "n_gstar": len(gstar),
            "oversize": max(0, len(pred) - len(gstar)),
            "undersize": max(0, len(gstar) - len(pred)),
            "is_oversized": float(len(pred) > len(gstar)),
            "is_undersized": float(len(pred) < len(gstar)),
            "is_rightsize": float(len(pred) == len(gstar)),
            "precision": (len(tp) / len(P)) if P else 0.0,
            "precision_union": (len(P & union) / len(P)) if P else 0.0,
            "precision_trunc": (len(set(trunc) & gstar) / len(trunc)) if trunc else 0.0,
            "f1": f1(gstar),
            "f1_trunc": max((2 * len(set(trunc) & g) / (len(trunc) + len(g))
                             if (trunc or g) else 0.0) for g in gsets),
            "n_fp": len(fps), "fp_alt_gold": len(alt),
            "fp_concept_ok": len(cok), "fp_concept_miss": len(cmiss),
            "fp_core_ok": len(ccore),
            "_intent": str(q.intent),
        }
        rows.append(row)

    def agg(rs, keys):
        return {k: round(float(np.mean([r[k] for r in rs])), 4) for k in keys}

    main = ["precision", "precision_union", "precision_trunc", "f1", "f1_trunc",
            "n_pred", "n_gstar", "oversize", "undersize",
            "is_oversized", "is_rightsize", "is_undersized"]
    out = {"n_queries": len(rows), "overall": agg(rows, main)}

    nfp = sum(r["n_fp"] for r in rows)
    out["false_positives"] = {
        "total": nfp, "per_query": round(nfp / max(1, len(rows)), 3),
        "alt_gold": sum(r["fp_alt_gold"] for r in rows),
        "concept_ok": sum(r["fp_concept_ok"] for r in rows),
        "concept_miss": sum(r["fp_concept_miss"] for r in rows),
        "core_ok_of_non_gold": sum(r["fp_core_ok"] for r in rows),
        "share": {k: (round(sum(r["fp_" + k] for r in rows) / max(1, nfp), 4))
                  for k in ("alt_gold", "concept_ok", "concept_miss")},
    }
    out["by_intent"] = {it: {**agg([r for r in rows if r["_intent"] == it], main),
                             "n": sum(1 for r in rows if r["_intent"] == it)}
                        for it in sorted({r["_intent"] for r in rows})}

    o, fp = out["overall"], out["false_positives"]
    print(f"queries {len(rows)}   |R_q| {o['n_pred']:.2f}   |G*| {o['n_gstar']:.2f}")
    print(f"\nsize    right {o['is_rightsize']:.1%}   over {o['is_oversized']:.1%} "
          f"(mean +{o['oversize']:.2f})   under {o['is_undersized']:.1%} "
          f"(mean -{o['undersize']:.2f})")
    print(f"\nprecision          {o['precision']:.4f}")
    print(f"  vs union of gold  {o['precision_union']:.4f}  "
          f"(+{o['precision_union'] - o['precision']:.4f} -- cost of mixing gold sets)")
    print(f"  truncated to |G*| {o['precision_trunc']:.4f}  "
          f"(+{o['precision_trunc'] - o['precision']:.4f} -- cost of over-predicting)")
    print(f"\nF1 {o['f1']:.4f}   F1 truncated to |G*| {o['f1_trunc']:.4f}  "
          f"({o['f1_trunc'] - o['f1']:+.4f})")
    print(f"\nfalse positives: {fp['total']} total, {fp['per_query']:.2f} per query")
    print(f"  in another gold set      {fp['alt_gold']:>5d}  {fp['share']['alt_gold']:.1%}")
    print(f"  right concept, not gold  {fp['concept_ok']:>5d}  {fp['share']['concept_ok']:.1%}"
          f"   (of which core-concept: {fp['core_ok_of_non_gold']})")
    print(f"  wrong concept            {fp['concept_miss']:>5d}  {fp['share']['concept_miss']:.1%}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "errors.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def sweep_uq_k(cfg: Config, ks: Sequence[int] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20)
               ) -> Dict:
    """U_q construction treated as its own retrieval problem.

    For each k, U_q is simply the top-k vocabulary concepts by query-concept
    cosine -- no threshold, no cap interaction -- and is scored against the
    query's gold concept union (the union of Con(r) over the gold explanation
    sets, or the query-level `concepts` annotation when present). This isolates
    the concept matcher from the selector: if concept-F1 is low at every k, the
    embedding similarity is the problem, not the composition objective.

    k is chosen by grouped 5-fold CV: within each fold the best k is picked on
    the four training folds and scored on the held-out fold, so the reported
    held-out F1 is not the number k was selected on. Paraphrase variants share
    a group, so a variant cannot sit on both sides of a split.

    The target is built from gold explanation sets, which a deployed system
    would not have. It is a tuning signal for one scalar, not an input: the
    selector never sees it. Note the final k is still chosen using gold-set
    information from these same queries, so the downstream GS-F1 carries a mild
    optimism -- run this on the silver queries to remove it entirely.

    A random-concept control at each k is reported alongside. If the matcher is
    not clearly above it, the embedding space is not separating query-relevant
    concepts at all.
    """
    ix = Index(cfg)
    A = ix.V @ ix.Q.T                        # |V| x n_q cosine
    order = np.argsort(-A, axis=0)           # concepts ranked per query
    vid = ix.vocab.ids
    vset = set(vid)

    def target_for(q) -> set:
        """What U_q is supposed to recover. Three defensible definitions:

        union -- every concept carried by any gold resource. Large (~13 on
                 EduExplain) and includes prerequisites the query never names.
        core  -- concepts in >= tau of the gold sets, i.e. exactly the Core(q)
                 the benchmark's own Concept Coverage metric scores against.
                 This is the target the end task actually rewards.
        query -- the query-level `concepts` annotation, when the file has one.

        These differ by nearly an order of magnitude in size, so a k tuned
        against one does not transfer to another. Always report which was used.
        """
        if cfg.uq_target == "core":
            gcs = [[slugify(c) for x in g if x in ix.rid2i
                    for c in ix.resources[ix.rid2i[x]].gold_concepts]
                   for g in q.gold_sets]
            return {c for c in E.core_concepts(gcs, cfg.tau) if c in vset}
        if cfg.uq_target == "query":
            return {c for c in q.gold_concepts if c in vset}
        return set(ix.oracle_units_for(q))

    targets = [target_for(q) for q in ix.queries]
    keep = [i for i, t in enumerate(targets) if t]
    rng = np.random.default_rng(cfg.seed)

    def per_query(k: int, random_ctrl: bool = False) -> np.ndarray:
        """Per-query (P, R, F1) at top-k; rows aligned with `keep`."""
        out = np.zeros((len(keep), 3))
        for n, qi in enumerate(keep):
            if random_ctrl:
                pred = {vid[j] for j in rng.choice(len(vid), size=min(k, len(vid)),
                                                   replace=False)}
            else:
                pred = {vid[j] for j in order[:k, qi]}
            tgt = targets[qi]
            tp = len(pred & tgt)
            p = tp / max(1, len(pred))
            r = tp / len(tgt)
            out[n] = (p, r, (2 * p * r / (p + r)) if (p + r) > 0 else 0.0)
        return out

    table = {k: per_query(k) for k in ks}
    rand = {k: per_query(k, True) for k in ks}

    tsz = np.array([len(targets[i]) for i in keep], dtype=float)
    print(f"target = {cfg.uq_target}")
    print(f"queries scored: {len(keep)} / {len(ix.queries)}   "
          f"|V| = {len(vid)}   target size: mean {tsz.mean():.2f} "
          f"median {np.median(tsz):.0f} p10 {np.percentile(tsz, 10):.0f} "
          f"p90 {np.percentile(tsz, 90):.0f}")
    print(f"\n{'k':>4s} {'P':>7s} {'R':>7s} {'F1':>7s} | {'randF1':>7s} {'lift':>7s}")
    for k in ks:
        m, rm = table[k].mean(axis=0), rand[k].mean(axis=0)
        print(f"{k:>4d} {m[0]:>7.4f} {m[1]:>7.4f} {m[2]:>7.4f} | "
              f"{rm[2]:>7.4f} {m[2] - rm[2]:>+7.4f}")

    # ---- grouped CV selection of k
    groups = [ix.queries[i].group_id for i in keep]
    folds = group_kfold_indices(groups, cfg.folds, cfg.seed)
    picks, held = [], []
    print(f"\ngrouped {cfg.folds}-fold selection of k")
    print(f"{'fold':>4s} {'n_val':>6s} {'k*':>4s} {'train F1':>9s} {'held-out F1':>12s}")
    for fi, (tr, va) in enumerate(folds):
        means = {k: float(table[k][tr, 2].mean()) for k in ks}
        kstar = max(means, key=lambda k: (means[k], -k))
        h = float(table[kstar][va, 2].mean())
        picks.append(kstar)
        held.append(h)
        print(f"{fi:>4d} {len(va):>6d} {kstar:>4d} {means[kstar]:>9.4f} {h:>12.4f}")
    final = int(np.bincount(picks).argmax())
    print(f"\nselected k = {final}  (per-fold picks {picks}, "
          f"mean held-out concept-F1 {np.mean(held):.4f} "
          f"+/- {np.std(held):.4f})")
    best_all = max(ks, key=lambda k: table[k][:, 2].mean())
    if best_all != final:
        print(f"note: k maximising F1 on ALL queries is {best_all} "
              f"({table[best_all][:, 2].mean():.4f}) -- the CV pick is the "
              f"honest one.")

    out = {"per_k": {str(k): {"precision": float(table[k][:, 0].mean()),
                              "recall": float(table[k][:, 1].mean()),
                              "f1": float(table[k][:, 2].mean()),
                              "random_f1": float(rand[k][:, 2].mean())}
                     for k in ks},
           "cv": {"folds": int(cfg.folds), "picks": picks,
                  "held_out_f1": held, "selected_k": final},
           "target": cfg.uq_target,
           "target_size_mean": float(tsz.mean()),
           "n_scored": len(keep), "vocabulary_size": len(vid)}
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "uq_k_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def _uq_eval_target(ix: "Index", q, cfg) -> set:
    """What U_q is scored against while tuning. Deliberately con_source-free.

    "union" here means the union of the gold resources' stored ANNOTATION, not
    `oracle_units_for`, which falls back through Con(r) and would therefore move
    with --con-source. A tuning target that shifts when an unrelated flag
    changes is not a target.
    """
    vset = set(ix.vocab.ids)
    if cfg.uq_target == "core":
        gcs = [[slugify(c) for x in g if x in ix.rid2i
                for c in ix.resources[ix.rid2i[x]].gold_concepts]
               for g in q.gold_sets]
        return {c for c in E.core_concepts(gcs, cfg.tau) if c in vset}
    if cfg.uq_target == "query":
        return {c for c in q.gold_concepts if c in vset}
    return set(ix.gold_annotation_units_for(q))


def tune_uq_head(cfg: Config,
                 hiddens: Optional[Sequence[int]] = None,
                 lrs: Optional[Sequence[float]] = None,
                 epochs_grid: Sequence[int] = (100, 200, 300, 600, 1000),
                 topks: Sequence[int] = (3, 5, 8, 12, 16, 20),
                 targets: Sequence[str] = ("gold_con_ann", "query"),
                 tune_on: str = "silver",
                 metric: str = "micro") -> Dict:
    """Tune the query-side concept head g_psi.

    This is NOT symmetric with contune, and the asymmetry is the whole point.
    The Con(r) tagger could be tuned freely because its criterion lives on
    resources, where no query appears. Here the criterion is concept-F1 on
    QUERIES, so where those queries come from decides whether the tuning is
    honest.

    --tune-on silver (default). Grouped k-fold INSIDE the silver file; the 321
        gold queries are never touched, so the selection cannot inform anything
        about the test set. This is the same device the kappa calibration used.
        The chosen flags are then applied to E1 and E2 unchanged.

    --tune-on gold-nested. Nested CV over the gold queries: the production folds
        are the outer loop, a grouped inner split selects, the winner is refit
        on the outer training part and scored on the outer test fold. The
        reported nested F1 is unbiased, but REUSING the pick for a production
        run is mildly optimistic -- the pick saw these queries' gold sets. Use
        it as a cross-check on the silver pick, or when there is no silver file,
        and say which in the paper.

    Free axes. epochs is checkpointed along one training run per
    (target, hidden, lr) cell, which is exact because the minibatch generator
    persists; topk only re-decodes probabilities already computed. So the fit
    count is |targets| x |hiddens| x |lrs| x folds. The query head is small and
    the files are a few hundred to a thousand rows, so a fit is seconds.
    """
    from .heads import ConceptHead
    hiddens = list(hiddens or [cfg.concept_hidden])
    lrs = list(lrs or [cfg.concept_lr])
    epochs_grid = sorted(set(int(e) for e in epochs_grid))
    topks = sorted(set(int(k) for k in topks))
    targets = list(targets)
    key = "micro_f1" if metric == "micro" else "macro_f1"

    ix = Index(cfg)
    pos = {v: j for j, v in enumerate(ix.vocab.ids)}
    warm = cfg.concept_train == "silver_then_cv"

    if tune_on == "silver":
        if not ix.train_queries:
            raise SystemExit("--tune-on silver needs --train-queries")
        if warm:
            raise SystemExit(
                "--concept-train silver_then_cv cannot be tuned on silver: the "
                "pre-training and the fine-tuning would share queries, so the "
                "inner score is meaningless. Tune the base head here (it is "
                "shared by both protocols), or use --tune-on gold-nested to "
                "tune the fine-tuning itself.")
        Q, X = ix.train_queries, ix.Q_train
        where = f"{len(Q)} SILVER queries (gold set untouched)"
    else:
        Q, X = ix.queries, ix.Q
        where = f"{len(Q)} GOLD queries, nested"

    ev = [_uq_eval_target(ix, q, cfg) for q in Q]
    keep = [i for i, t in enumerate(ev) if t]
    if not keep:
        raise SystemExit("no query has a non-empty concept target")
    groups = [Q[i].group_id for i in keep]
    Xk = X[keep]
    evk = [ev[i] for i in keep]
    def target_matrix(name: str) -> np.ndarray:
        c = Config(**{**cfg.__dict__, "concept_target": name})
        ix.cfg = c                               # concept_targets_for reads cfg
        M = np.zeros((len(keep), len(pos)), dtype=np.float32)
        for r, i in enumerate(keep):
            for u in ix.concept_targets_for(Q[i]):
                j = pos.get(u)
                if j is not None:
                    M[r, j] = 1.0
        ix.cfg = cfg
        return M

    Ys = {t: target_matrix(t) for t in targets}
    # Two "different" targets can resolve to the SAME matrix: concept_target
    # "query" falls back to gold_con when a query file has no query-level
    # `concepts` field, and under --con-source annotated that fallback equals
    # gold_con_ann. The axis is then inert and the identical scores look like a
    # finding instead of a no-op.
    for i, t1 in enumerate(targets):
        for t2 in targets[i + 1:]:
            if np.array_equal(Ys[t1], Ys[t2]):
                print(f"WARNING: targets '{t1}' and '{t2}' are the SAME matrix "
                      f"on these queries, so that axis tests nothing. Most "
                      f"likely this file has no query-level `concepts` field, "
                      f"so 'query' fell back to gold_con. Check the file, or "
                      f"drop one value from --target-grid.")

    def prf(RHO: np.ndarray, rows: Sequence[int], k: int) -> Dict:
        kk = min(k, RHO.shape[1])
        tp = bn = an = 0
        mac = []
        for r, i in enumerate(rows):
            pred = {ix.vocab.ids[j]
                    for j in np.argpartition(-RHO[r], kk - 1)[:kk]}
            t = evk[i]
            inter = len(pred & t)
            tp += inter; bn += len(pred); an += len(t)
            mac.append(2 * inter / (len(pred) + len(t)) if (pred or t) else 0.0)
        p, r_ = tp / max(1, bn), tp / max(1, an)
        return {"micro_f1": 0.0 if p + r_ == 0 else 2 * p * r_ / (p + r_),
                "macro_f1": float(np.mean(mac)), "precision": p, "recall": r_}

    def new_head(h, lr, ep):
        return ConceptHead(X.shape[1], len(pos), h, lr, ep,
                           cfg.concept_activation, cfg.seed)

    # silver pre-training, once per (target, hidden, lr), for the warm protocol
    pre: Dict = {}
    if warm:
        Ysil = {}
        for t in targets:
            c = Config(**{**cfg.__dict__, "concept_target": t})
            ix.cfg = c
            Ysil[t] = _concept_targets_matrix(ix, ix.train_queries, pos)
            ix.cfg = cfg
        for t in targets:
            for h in hiddens:
                for lr in lrs:
                    hd = new_head(h, lr, cfg.concept_epochs)
                    hd.fit(ix.Q_train, Ysil[t])
                    pre[(t, h, lr)] = hd.state()

    print(f"tuning the query head on {where}")
    print(f"scored queries: {len(keep)}/{len(Q)}   |V| = {len(pos)}   "
          f"eval target = {cfg.uq_target}   "
          f"mean |target| = {np.mean([len(t) for t in evk]):.2f}")
    print(f"grid: target {targets} x hidden {hiddens} x lr {lrs} x "
          f"epochs {epochs_grid} x topk {topks}"
          + ("   (fine-tuning a silver pre-train)" if warm else ""))
    print(f"protocol: {cfg.concept_train}   selecting on {key}\n")

    def select(tr: np.ndarray, va: np.ndarray) -> Dict:
        """Best cell by validation score, fitting on tr and scoring on va."""
        best = None
        for t in targets:
            for h in hiddens:
                for lr in lrs:
                    hd = new_head(h, lr, max(epochs_grid))
                    if warm:
                        hd.load_state(pre[(t, h, lr)], lr=lr)
                    done = 0
                    for ep in epochs_grid:
                        hd.fit(Xk[tr], Ys[t][tr], epochs=ep - done)
                        done = ep
                        R = hd.predict(Xk[va])
                        for k in topks:
                            m = prf(R, va, k)
                            cand = (m[key], -ep,
                                    {"target": t, "hidden": h, "lr": lr,
                                     "epochs": ep, "topk": k, "val": m})
                            if best is None or cand[:2] > best[:2]:
                                best = cand
        return best[2]

    folds = group_kfold_indices(groups, min(cfg.folds, max(2, len(keep))),
                                cfg.seed)
    picks, outer = [], []
    if tune_on == "silver":
        # One selection over the silver folds: score every cell by its mean
        # held-out F1 across folds, which is steadier than picking per fold.
        acc: Dict = {}
        for fi, (tr, va) in enumerate(folds):
            for t in targets:
                for h in hiddens:
                    for lr in lrs:
                        hd = new_head(h, lr, max(epochs_grid))
                        if warm:
                            hd.load_state(pre[(t, h, lr)], lr=lr)
                        done = 0
                        for ep in epochs_grid:
                            hd.fit(Xk[tr], Ys[t][tr], epochs=ep - done)
                            done = ep
                            R = hd.predict(Xk[va])
                            for k in topks:
                                m = prf(R, va, k)
                                acc.setdefault((t, h, lr, ep, k), []).append(
                                    m[key])
            print(f"  fold {fi} done ({len(tr)} train / {len(va)} val)")
        means = {c: float(np.mean(v)) for c, v in acc.items()}
        bestc = max(means, key=lambda c: (means[c], -c[3]))
        chosen = {"target": bestc[0], "hidden": bestc[1], "lr": bestc[2],
                  "epochs": bestc[3], "topk": bestc[4]}
        sel = means[bestc]
        print(f"\nbest silver-CV {key}: {sel:.4f} at {chosen}")
        top = sorted(means, key=lambda c: -means[c])[:8]
        print(f"\n{'target':<14s} {'hid':>5s} {'lr':>8s} {'ep':>6s} "
              f"{'topk':>5s} {'silver ' + key:>14s}")
        for c in top:
            print(f"{c[0]:<14s} {c[1]:>5d} {c[2]:>8g} {c[3]:>6d} {c[4]:>5d} "
                  f"{means[c]:>14.4f}")
        nested = None
    else:
        for fi, (tr, va) in enumerate(folds):
            inner = group_kfold_indices([groups[i] for i in tr], 4,
                                        cfg.seed + 200 + fi)
            itr, iva = tr[inner[0][0]], tr[inner[0][1]]
            p = select(itr, iva)
            hd = new_head(p["hidden"], p["lr"], p["epochs"])
            if warm:
                hd.load_state(pre[(p["target"], p["hidden"], p["lr"])],
                              lr=p["lr"])
            hd.fit(Xk[tr], Ys[p["target"]][tr])
            m = prf(hd.predict(Xk[va]), va, p["topk"])
            picks.append(p)
            outer.append(m)
            print(f"fold {fi}: chose target {p['target']}, hidden {p['hidden']},"
                  f" lr {p['lr']:g}, epochs {p['epochs']}, topk {p['topk']}"
                  f"   inner {key} {p['val'][key]:.4f} -> outer "
                  f"{m[key]:.4f} (P {m['precision']:.4f} / R {m['recall']:.4f})")

        def mode(vals):
            c: Dict = {}
            for v in vals:
                c[v] = c.get(v, 0) + 1
            return max(c, key=lambda v: (c[v], -list(vals).index(v)))
        chosen = {k2: mode([p[k2] for p in picks])
                  for k2 in ("target", "hidden", "lr", "epochs", "topk")}
        nested = float(np.mean([m[key] for m in outer]))
        print(f"\nnested held-out {key} {nested:.4f} "
              f"+/- {np.std([m[key] for m in outer]):.4f}")
        print("That nested figure is unbiased. Reusing the chosen setting for a "
              "production run is NOT -- the pick saw these queries' gold sets. "
              "Prefer the silver pick for the reported rows.")
        sel = nested

    # edge-of-grid guards
    if chosen["epochs"] == max(epochs_grid):
        print(f"WARNING: epochs picked the largest offered "
              f"({max(epochs_grid)}); extend --epochs-grid upward.")
    if len(topks) > 1 and chosen["topk"] in (min(topks), max(topks)):
        print(f"WARNING: topk = {chosen['topk']} is on the edge of "
              f"--topk-grid; widen it.")
    if len(hiddens) > 1 and chosen["hidden"] == max(hiddens):
        print(f"WARNING: hidden = {chosen['hidden']} is the largest offered.")

    print("\nproduction flags (paste into the E1 / E2 / E3 commands):")
    if warm:
        # In warm mode the tuned lr/epochs are the FINE-TUNING budget, and the
        # production run reads those from --concept-ft-*; --concept-epochs is
        # the silver PRE-TRAINING budget, which this run held at its default.
        # Printing --concept-lr here would silently retune the wrong phase.
        print(f"  --concept-target {chosen['target']} "
              f"--concept-hidden {chosen['hidden']} \\\n"
              f"  --concept-ft-lr {chosen['lr']:g} "
              f"--concept-ft-epochs {chosen['epochs']} "
              f"--concept-topk {chosen['topk']}")
        print(f"  (plus --concept-epochs <silver pre-train budget>, which this "
              f"run left at {cfg.concept_epochs}; take it from the "
              f"--tune-on silver run.)")
    else:
        print(f"  --concept-target {chosen['target']} "
              f"--concept-hidden {chosen['hidden']} "
              f"--concept-lr {chosen['lr']:g} \\\n"
              f"  --concept-epochs {chosen['epochs']} "
              f"--concept-topk {chosen['topk']}")
    if chosen["topk"] != cfg.kappa:
        print(f"note: topk {chosen['topk']} != kappa {cfg.kappa}. The learned "
              f"U_q is then a different SIZE from the matcher's, so row 0 is no "
              f"longer size-matched. Either set --kappa {chosen['topk']} in "
              f"every row of the table, or report both sizes.")

    out = {"chosen": chosen, "tune_on": tune_on, "protocol": cfg.concept_train,
           "selection_metric": key, "selected_score": sel,
           "nested_held_out": nested, "per_fold": picks,
           "outer": outer, "n_scored": len(keep),
           "eval_target": cfg.uq_target,
           "grid": {"target": targets, "hidden": hiddens, "lr": lrs,
                    "epochs": epochs_grid, "topk": topks}}
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "uq_tune.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(cfg.out_dir, 'uq_tune.json')}")
    return out


def tune_con_classifier(cfg: Config,
                        epochs_grid: Sequence[int] = (15, 30, 60, 120, 200),
                        taus: Sequence[float] = (0.1, 0.2, 0.3, 0.5, 0.7),
                        hiddens: Optional[Sequence[int]] = None,
                        lrs: Optional[Sequence[float]] = None,
                        wds: Optional[Sequence[float]] = None,
                        size_modes: Sequence[str] = ("topm", "match"),
                        metric: str = "micro") -> Dict:
    """Choose the Con(r) classifier's hyperparameters by NESTED CV over resources.

    Why this is leakage-free, and worth saying so in the paper: the selection
    criterion is concept-F1 against resource annotations on held-out RESOURCES.
    No query, no gold set and no evaluation query enters it at any point, so
    tuning here cannot inform anything about the query test set. That is a
    stronger position than the kappa calibration, which had to be routed
    through the silver queries to stay honest.

    Protocol. The outer folds are the production folds (grouped by
    --con-group). Inside each outer training part a grouped inner split is held
    out; the grid is scored on it; the winner is refitted on the whole outer
    training part and scored on the outer test fold. So the reported Con(r)
    quality is the quality of the whole select-then-fit procedure, not of the
    best cell of a grid -- which is the number that would be optimistic.

    Cost. The epoch axis is nearly free: each (hidden, lr) pair trains ONCE to
    max(epochs_grid) and is evaluated at every checkpoint along the way, which
    is exact because the minibatch generator persists across fit() calls. The
    tau and decode axes are free too -- they only re-threshold probabilities
    that are already computed. So the number of fits is
    len(hiddens) x len(lrs) x n_folds, plus one refit per fold.
    """
    from .heads import ConceptHead
    hiddens = list(hiddens or [cfg.res_concept_hidden])
    lrs = list(lrs or [cfg.res_concept_lr])
    wds = list(wds or [cfg.res_concept_wd])
    epochs_grid = sorted(set(int(e) for e in epochs_grid))

    ix = Index(cfg)
    Y = _annotation_matrix(ix.resources, ix.vocab)
    has = Y.sum(axis=1) > 0
    if not has.any():
        raise SystemExit("contune needs resource-level `concepts` annotations")
    tgt_size = float(Y[has].sum(axis=1).mean())
    groups = np.array(_res_groups(ix.resources, cfg.con_group))

    def score(PROB: np.ndarray, rows: np.ndarray, tau: float, size: str) -> Dict:
        c = Config(**{**cfg.__dict__, "con_learned_tau": float(tau),
                      "con_learned_size": size})
        B = C.learned_concept_matrix(PROB, c, tgt_size) > 0
        A = Y[rows] > 0
        tp = float((B & A).sum())
        p = tp / max(1.0, float(B.sum()))
        r = tp / max(1.0, float(A.sum()))
        mac = [0.0 if (int(B[i].sum()) + int(A[i].sum())) == 0 else
               2 * float((B[i] & A[i]).sum()) / (int(B[i].sum()) + int(A[i].sum()))
               for i in range(len(rows))]
        return {"micro_f1": 0.0 if p + r == 0 else 2 * p * r / (p + r),
                "macro_f1": float(np.mean(mac)) if mac else 0.0,
                "precision": p, "recall": r,
                "size": float(B.sum(axis=1).mean())}

    key = "micro_f1" if metric == "micro" else "macro_f1"
    n_grid = len(hiddens) * len(lrs) * len(wds)
    print(f"annotated resources: {int(has.sum())}/{len(ix.resources)}   "
          f"|V| = {len(ix.vocab)}   mean |annotation| = {tgt_size:.2f}")
    print(f"grid: hidden {hiddens} x lr {lrs} x wd {wds} x "
          f"epochs {epochs_grid} x tau {list(taus)} x decode {list(size_modes)}")
    print(f"selecting on inner-split {key}; {n_grid} fit(s) per outer fold "
          f"plus one refit\n")

    k = min(cfg.con_folds, max(2, len(set(groups.tolist()))))
    folds = group_kfold_indices(groups.tolist(), k, cfg.seed)
    picks, outer = [], []
    for fi, (tr, te) in enumerate(folds):
        tr = tr[has[tr]]
        te_s = te[has[te]]
        if len(tr) == 0 or len(te_s) == 0:
            continue
        # grouped inner split of the outer training part
        inner = group_kfold_indices(groups[tr].tolist(), 4, cfg.seed + 100 + fi)
        itr, iva = tr[inner[0][0]], tr[inner[0][1]]

        best = None
        for h in hiddens:
            for lr in lrs:
                for wd in wds:
                    head = ConceptHead(ix.R.shape[1], len(ix.vocab), h, lr,
                                       max(epochs_grid), "sigmoid", cfg.seed,
                                       batch_size=cfg.res_concept_batch, wd=wd)
                    done = 0
                    for ep in epochs_grid:
                        head.fit(ix.R[itr], Y[itr], epochs=ep - done)
                        done = ep
                        PV = head.predict(ix.R[iva])
                        for tau in taus:
                            for sz in size_modes:
                                m = score(PV, iva, tau, sz)
                                cand = (m[key], -ep,
                                        {"hidden": h, "lr": lr, "wd": wd,
                                         "epochs": ep, "tau": float(tau),
                                         "size": sz, "val": m})
                                if best is None or cand[:2] > best[:2]:
                                    best = cand
        cfgp = best[2]
        # refit on the whole outer training part at the chosen setting
        head = ConceptHead(ix.R.shape[1], len(ix.vocab), cfgp["hidden"],
                           cfgp["lr"], cfgp["epochs"], "sigmoid", cfg.seed,
                           batch_size=cfg.res_concept_batch,
                           wd=cfgp["wd"]).fit(ix.R[tr], Y[tr])
        m = score(head.predict(ix.R[te_s]), te_s, cfgp["tau"], cfgp["size"])
        picks.append(cfgp)
        outer.append(m)
        print(f"fold {fi}: chose hidden {cfgp['hidden']}, lr {cfgp['lr']:g}, "
              f"wd {cfgp['wd']:g}, epochs {cfgp['epochs']}, "
              f"tau {cfgp['tau']:g}, {cfgp['size']}"
              f"   inner {key} {cfgp['val'][key]:.4f}"
              f" -> outer micro-F1 {m['micro_f1']:.4f} "
              f"(macro {m['macro_f1']:.4f}, |Con(r)| {m['size']:.2f})")

    if not picks:
        raise SystemExit("no usable folds")

    def mode(vals):
        c: Dict = {}
        for v in vals:
            c[v] = c.get(v, 0) + 1
        return max(c, key=lambda v: (c[v], -list(vals).index(v)))

    chosen = {"hidden": mode([p["hidden"] for p in picks]),
              "lr": mode([p["lr"] for p in picks]),
              "wd": mode([p["wd"] for p in picks]),
              "epochs": mode([p["epochs"] for p in picks]),
              "tau": mode([p["tau"] for p in picks]),
              "size": mode([p["size"] for p in picks])}
    mi = [m["micro_f1"] for m in outer]
    ma = [m["macro_f1"] for m in outer]
    print(f"\nnested held-out micro-F1 {np.mean(mi):.4f} +/- {np.std(mi):.4f}"
          f"   macro-F1 {np.mean(ma):.4f} +/- {np.std(ma):.4f}")
    print("This is the unbiased number for the tagger: report it, not the best "
          "cell of the grid.")
    # A pick sitting on the edge of the grid means the grid, not the data,
    # chose it. Say so -- this is the most common way a tuning run lies.
    if chosen["epochs"] == max(epochs_grid):
        print(f"WARNING: the epoch pick is the LARGEST value offered "
              f"({max(epochs_grid)}); the head was still improving, so extend "
              f"--epochs-grid upward before using this.")
    if chosen["epochs"] == min(epochs_grid) and len(epochs_grid) > 1:
        print(f"WARNING: the epoch pick is the SMALLEST value offered "
              f"({min(epochs_grid)}); it may be overfitting past that, so "
              f"extend --epochs-grid downward.")
    if len(taus) > 1 and chosen["tau"] in (min(taus), max(taus)):
        print(f"WARNING: tau = {chosen['tau']:g} is on the edge of "
              f"--tau-grid; widen it.")
    if len(hiddens) > 1 and chosen["hidden"] == max(hiddens):
        print(f"WARNING: hidden = {chosen['hidden']} is the largest offered; "
              f"widen --hidden-grid.")
    if len(set(p["epochs"] for p in picks)) > 1:
        print(f"note: the epoch pick is not stable across folds "
              f"({[p['epochs'] for p in picks]}); the modal value is used "
              f"below, and a spread this wide is worth a sentence in the paper.")
    print("\nproduction flags (paste into the E3 / 3c commands):")
    print(f"  --res-concept-hidden {chosen['hidden']} "
          f"--res-concept-lr {chosen['lr']:g} "
          f"--res-concept-wd {chosen['wd']:g} "
          f"--res-concept-epochs {chosen['epochs']} \\\n"
          f"  --con-learned-tau {chosen['tau']:g} "
          f"--con-learned-size {chosen['size']} "
          f"--con-group {cfg.con_group}")

    out = {"chosen": chosen, "per_fold": picks,
           "outer": outer,
           "nested_micro_f1_mean": float(np.mean(mi)),
           "nested_micro_f1_std": float(np.std(mi)),
           "nested_macro_f1_mean": float(np.mean(ma)),
           "nested_macro_f1_std": float(np.std(ma)),
           "grid": {"hidden": hiddens, "lr": lrs, "wd": wds,
                    "epochs": epochs_grid,
                    "tau": list(taus), "size": list(size_modes)},
           "selection_metric": key, "con_group": cfg.con_group,
           "n_annotated": int(has.sum()), "vocabulary_size": len(ix.vocab)}
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "con_tune.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(cfg.out_dir, 'con_tune.json')}")
    return out


def compare_concept_predictors(cfg: Config,
                               ks: Sequence[int] = (3, 5, 8, 12, 16, 20)
                               ) -> Dict:
    """Cosine matching vs a trained classifier, on both sides of the concept layer.

    QUERY SIDE. U_q is currently the top concepts by query-to-concept cosine.
    This scores that against a multi-label classifier h_q -> V trained on the
    concept annotations of the query's explanation sets, under the two
    protocols the silver/gold distribution gap leaves open (grouped k-fold over
    the gold queries, or pre-train on silver then fine-tune per gold fold), plus
    the silver-only transfer row for reference. Every method is decoded at the
    same top-k, so the rows differ in ranking quality and nothing else.

    RESOURCE SIDE. Con(r) is currently the top concepts by resource-to-concept
    cosine. This scores that against a classifier trained on the corpus
    annotation, predicted OUT OF FOLD over resources so no resource's own
    annotation tags it, against the annotation as ground truth.

    Nothing here runs the selector, so it is cheap and it answers the question
    that matters first: is the concept signal itself better. The GS-F1 effect
    then comes from `run` with --uq-mode learned and/or --con-source learned.
    """
    ix = Index(cfg)
    vid = ix.vocab.ids
    vset = set(vid)

    # ------------------------------------------------------- query side
    def target_for(q) -> set:
        if cfg.uq_target == "core":
            gcs = [[slugify(c) for x in g if x in ix.rid2i
                    for c in ix.resources[ix.rid2i[x]].gold_concepts]
                   for g in q.gold_sets]
            return {c for c in E.core_concepts(gcs, cfg.tau) if c in vset}
        if cfg.uq_target == "query":
            return {c for c in q.gold_concepts if c in vset}
        return set(ix.oracle_units_for(q))

    targets = [target_for(q) for q in ix.queries]
    keep = [i for i, t in enumerate(targets) if t]
    tsz = np.array([len(targets[i]) for i in keep], dtype=float)

    scores: Dict[str, np.ndarray] = {"cosine (current)": (ix.V @ ix.Q.T).T}
    head_info: Dict[str, Dict] = {}
    protocols = [("classifier, gold CV", "cv"),
                 ("classifier, silver only", "silver"),
                 ("classifier, silver -> gold CV", "silver_then_cv")]
    for label, mode in protocols:
        if mode != "cv" and not ix.train_queries:
            print(f"skipping '{label}': needs --train-queries (the silver file)")
            continue
        c = Config(**{**cfg.__dict__, "uq_mode": "learned",
                      "concept_train": mode})
        RHO, info = _fit_concept_head(ix, c, quiet=True)
        scores[label] = RHO
        head_info[label] = info

    rng = np.random.default_rng(cfg.seed)
    RND = rng.random((len(ix.queries), len(vid)))
    scores["random control"] = RND

    def prf_at(S: np.ndarray, k: int) -> np.ndarray:
        """Per-query (P, R, F1) of the top-k of S; rows aligned with `keep`."""
        kk = min(k, S.shape[1])
        out = np.zeros((len(keep), 3))
        for n, qi in enumerate(keep):
            top = np.argpartition(-S[qi], kk - 1)[:kk]
            pred = {vid[j] for j in top}
            tgt = targets[qi]
            tp = len(pred & tgt)
            p, r = tp / max(1, len(pred)), tp / len(tgt)
            out[n] = (p, r, (2 * p * r / (p + r)) if (p + r) > 0 else 0.0)
        return out

    per = {lab: {k: prf_at(S, k) for k in ks} for lab, S in scores.items()}

    print("=" * 78)
    print("QUERY SIDE: what concepts does the query require?")
    print("=" * 78)
    print(f"target = {cfg.uq_target} (training target = {cfg.concept_target})")
    print(f"queries scored: {len(keep)}/{len(ix.queries)}   |V| = {len(vid)}   "
          f"target size: mean {tsz.mean():.2f} median {np.median(tsz):.0f}")
    print(f"\n{'method':<32s}" + "".join(f"{'F1@' + str(k):>9s}" for k in ks))
    for lab in scores:
        print(f"{lab:<32s}" + "".join(f"{per[lab][k][:, 2].mean():>9.4f}"
                                      for k in ks))

    kbest = max(ks, key=lambda k: per["cosine (current)"][k][:, 2].mean())
    print(f"\nat k = {kbest} (best k for the cosine matcher):")
    print(f"{'method':<32s} {'P':>8s} {'R':>8s} {'F1':>8s} {'vs cosine':>10s} "
          f"{'win rate':>9s} {'p(boot)':>9s}")
    base = per["cosine (current)"][kbest]
    boot_rng = np.random.default_rng(cfg.seed + 1)
    rows = {}
    for lab in scores:
        m = per[lab][kbest]
        d = m[:, 2] - base[:, 2]
        if lab == "cosine (current)":
            cells = ("", "", "")
        else:
            # paired bootstrap over queries: how often the mean delta flips sign
            idx = boot_rng.integers(0, len(d), size=(2000, len(d)))
            bs = d[idx].mean(axis=1)
            pval = float(np.mean(bs <= 0)) if d.mean() > 0 else float(np.mean(bs >= 0))
            cells = (f"{d.mean():>+10.4f}",
                     f"{np.mean(d > 0):>9.2f}", f"{pval:>9.3f}")
        print(f"{lab:<32s} {m[:, 0].mean():>8.4f} {m[:, 1].mean():>8.4f} "
              f"{m[:, 2].mean():>8.4f} {cells[0]:>10s} {cells[1]:>9s} "
              f"{cells[2]:>9s}")
        rows[lab] = {"precision": float(m[:, 0].mean()),
                     "recall": float(m[:, 1].mean()),
                     "f1": float(m[:, 2].mean()),
                     "delta_f1_vs_cosine": float(d.mean()),
                     "win_rate": float(np.mean(d > 0)),
                     "f1_by_k": {str(k): float(per[lab][k][:, 2].mean())
                                 for k in ks}}

    # ---------------------------------------------------- resource side
    print("\n" + "=" * 78)
    print("RESOURCE SIDE: what concepts does a resource cover?  (Con(r))")
    print("=" * 78)
    Y = _annotation_matrix(ix.resources, ix.vocab)
    has = Y.sum(axis=1) > 0
    res_rows = {}
    if not has.any():
        print("no resource-level `concepts` annotation in this corpus; skipped")
    else:
        def con_quality(P: np.ndarray) -> Dict:
            B, A = (P > 0), (Y > 0)
            tp = float((B[has] & A[has]).sum())
            p = tp / max(1.0, float(B[has].sum()))
            r = tp / max(1.0, float(A[has].sum()))
            mac = [0.0 if (int(B[i].sum()) + int(A[i].sum())) == 0 else
                   2 * float((B[i] & A[i]).sum()) /
                   (int(B[i].sum()) + int(A[i].sum()))
                   for i in np.nonzero(has)[0]]
            return {"precision": p, "recall": r,
                    "f1": 0.0 if p + r == 0 else 2 * p * r / (p + r),
                    "macro_f1": float(np.mean(mac)),
                    "size": float(B.sum(axis=1).mean())}

        tgt_size = float(Y[has].sum(axis=1).mean())
        cand = {"cosine (current)": C.build_concept_matrix(ix.R, ix.V, cfg)}
        # Two fold groupings, because they answer different questions, and two
        # decodes, which are free once the probabilities are fitted.
        #   chapter -> the held-out resource's whole chapter is unseen, so this
        #              is "tag a chapter the tagger has never read"
        #   none    -> a random split, so sibling resources from the same
        #              chapter are in training. Closer to how a tagger is
        #              actually deployed on a growing corpus, and the looser of
        #              the two numbers; report both or say which.
        for grp, glab in (("chapter", "chapter-held-out"),
                          ("none", "random split")):
            cl = Config(**{**cfg.__dict__, "con_source": "learned",
                           "con_group": grp})
            PROB = _learned_con_prob(ix.resources, ix.R, ix.vocab, cl, quiet=True)
            for dec, dlab in (("topm", f"top-{cfg.con_learned_topm or cfg.tag_topm}"),
                              ("match", "size-matched")):
                c2 = Config(**{**cl.__dict__, "con_learned_size": dec})
                cand[f"classifier, {glab} [{dlab}]"] = C.learned_concept_matrix(
                    PROB, c2, tgt_size)
        print(f"\nresources with an annotation: {int(has.sum())}/{len(ix.resources)}"
              f"   mean |annotation| = {tgt_size:.2f}")
        print(f"\n{'tagger':<42s} {'micro-P':>9s} {'micro-R':>9s} "
              f"{'micro-F1':>9s} {'macro-F1':>9s} {'|Con(r)|':>9s}")
        for lab, P in cand.items():
            d = con_quality(P)
            res_rows[lab] = d
            print(f"{lab:<42s} {d['precision']:>9.4f} {d['recall']:>9.4f} "
                  f"{d['f1']:>9.4f} {d['macro_f1']:>9.4f} {d['size']:>9.2f}")
        print("\nThe annotation itself is con_source=annotated, i.e. micro-F1 1.0 "
              "by construction.\nThe classifier row is what a corpus WITHOUT "
              "annotations would get.")

    out = {"query_side": {"target": cfg.uq_target,
                          "training_target": cfg.concept_target,
                          "k_reported": kbest, "ks": list(ks),
                          "n_scored": len(keep),
                          "target_size_mean": float(tsz.mean()),
                          "methods": rows,
                          "head_info": head_info},
           "resource_side": res_rows,
           "vocabulary_size": len(vid)}
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "concept_predictors.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(cfg.out_dir, 'concept_predictors.json')}")
    return out


def sweep_tau(cfg: Config, taus: Optional[Sequence[float]] = None,
              kappas: Sequence[int] = (5,)) -> Dict:
    """|U_q| = top-kappa concepts above an absolute similarity floor tau.

    The default construction gates a_q(v) at gamma * max_v a_q(v) on RAW cosine,
    while Con(r) is gated on z-scored similarity. Measured on EduExplain the
    relative gate binds for ~11% of queries and kappa truncates the rest, so
    U_q is a constant 12 concepts and the coverage weights (which sum to 1)
    are spread over roughly twice as many concepts as any valid answer can
    contain. This sweep replaces the relative gate with an absolute floor so
    |U_q| can adapt to query specificity, and reports each setting at beta=0
    (coverage alone) and at the configured beta (full objective).

    beta=0 is the diagnostic that matters: it scores the concept signal with no
    relevance term to mask it, so a tau that improves the full objective but not
    the beta=0 column is improving by getting out of relevance's way rather than
    by selecting better concepts.

    With no --taus given the grid is derived from the data: the kappa-th best
    a_q(v) per query, at the 10/25/50/75/90th percentiles, so each tau binds for
    a known fraction of queries. tau=0 reproduces the current default.
    """
    ix = Index(cfg)
    A = ix.V @ ix.Q.T                        # |V| x n_q
    out: Dict[str, Dict] = {}

    if taus is None:
        k0 = max(kappas)
        kth = np.sort(A, axis=0)[-k0, :] if A.shape[0] >= k0 else A.min(axis=0)
        taus = [0.0] + [round(float(np.percentile(kth, p)), 4)
                        for p in (10, 25, 50, 75, 90)]
        print(f"a_q(v) percentiles of the {k0}-th best concept per query: "
              + ", ".join(f"p{p}={t:.4f}" for p, t in zip((10, 25, 50, 75, 90),
                                                          taus[1:])))
        amax = A.max(axis=0)
        print(f"max_v a_q(v): mean {amax.mean():.4f}  "
              f"p10 {np.percentile(amax, 10):.4f}  p90 {np.percentile(amax, 90):.4f}")

    print(f"\nuq_min={cfg.uq_min} (floor: fewer than this many concepts above tau "
          f"are padded back up; pass --uq-min 0 to let U_q shrink freely)")
    print(f"\n{'tau':>7s} {'kappa':>5s} {'beta':>6s} {'|U_q|':>6s} {'U_q F1':>7s} "
          f"{'U_q P':>7s} {'U_q R':>7s} {'GS-F1':>7s} {'top-n':>7s} "
          f"{'delta':>7s} {'CoreCC':>7s}")
    for kap in kappas:
        for tau in taus:
            for beta in (0.0, float(cfg.beta)):
                tag = f"tau{tau}_k{kap}_b{beta}"
                c = Config(**{**cfg.__dict__, "uq_abs_tau": float(tau),
                              "kappa": int(kap), "beta": beta,
                              "out_dir": os.path.join(cfg.out_dir, "uqtau", tag)})
                m = run(c, ix.rebind(c), quiet=True)
                sysm, tk = m["system"], m["topk_baseline"]
                out[tag] = {"system": sysm, "topk": tk}
                print(f"{tau:>7.4f} {kap:>5d} {beta:>6.2f} "
                      f"{sysm.get('uq_size', 0.0):>6.2f} {sysm['uq_f1']:>7.4f} "
                      f"{sysm['uq_precision']:>7.4f} {sysm['uq_recall']:>7.4f} "
                      f"{sysm['f1']:>7.4f} {tk['f1']:>7.4f} "
                      f"{sysm['f1'] - tk['f1']:>+7.4f} "
                      f"{sysm.get('concept_coverage', 0.0):>7.4f}")
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "uq_tau_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def role_sizes(cfg, out_path: str, compare: Optional[str] = None) -> Dict[str, int]:
    """Write the per-query budget from the ROLE HEAD ALONE, with no selection run.

    `tools_sizes_from_run.py --mode profile` already recovers this number from a
    completed run, and the two agree exactly, because the recorded profile is
    built before U_q and before select(). The reason to have this command as
    well is provenance, not a different answer: extracted from a run, the file
    sits downstream of kappa, the concept configuration, the pool and the
    selector, and a reader has to be told that none of those reach it. Produced
    here, that is manifest -- the only things in the chain are the query
    encoder, the role head and the CV split.

    So the budget is a prediction and not an oracle: for each query the head
    that predicts it was fitted on the other folds, grouped by gold-set
    signature so no near-duplicate query leaks across the split. Nothing about
    which resources are gold enters, only how MANY, and only from the training
    folds.
    """
    ix = Index(cfg)
    nq = len(ix.queries)
    # bool(), not "is not None": Index sets train_queries to [] when no
    # --train-queries is given, and an empty list would silently route us into
    # the two-file branch and fit the head on ZERO queries. run() uses bool();
    # this has to match it exactly or the budget here is not the budget there.
    two_file = bool(ix.train_queries)
    role_two_file = two_file and cfg.role_train != "cv"
    if role_two_file:
        folds = [(np.arange(len(ix.train_queries)), np.arange(nq))]
        RQ, RX = ix.train_queries, ix.Q_train
    else:
        k = min(cfg.folds, max(2, nq))
        folds = group_kfold_indices([q.group_id for q in ix.queries], k, cfg.seed)
        RQ, RX = ix.queries, ix.Q

    pred_c = np.zeros(nq, dtype=int)
    pred_f = np.zeros(nq, dtype=int)
    _multi = cfg.role_loss in ("best_match", "single_first", "single_random",
                               "average")
    for tr, te in folds:
        tq = [RQ[i] for i in tr]
        ct = [ix.targets_for(q)[0] if _multi else [ix.mean_targets_for(q)[0]]
              for q in tq]
        ft = [ix.targets_for(q)[1] if _multi else [ix.mean_targets_for(q)[1]]
              for q in tq]
        head = RoleHead(ix.Q.shape[1], cfg.n_max, cfg.role_hidden, cfg.role_lr,
                        cfg.role_weight_decay, cfg.role_epochs, cfg.lam,
                        cfg.role_loss, cfg.seed).fit(RX[tr], ct, ft)
        c, f, _ = head.predict(ix.Q[te], cfg.count_decode, cfg.eta)
        pred_c[te], pred_f[te] = c, f

    sizes: Dict[str, int] = {}
    for qi, q in enumerate(ix.queries):
        cq, fq = int(pred_c[qi]), int(pred_f[qi])
        cq = max(0, min(cq, cfg.n_max))          # identical clamp to run()
        if cq + fq > cfg.n_max:
            fq = max(0, cfg.n_max - cq)
        sizes[str(q.query_id)] = max(1, cq + fq)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sizes, f, indent=2, sort_keys=True)

    import statistics as _st
    from collections import Counter
    vals = list(sizes.values())
    gold = [len(set().union(*[set(g) for g in q.gold_sets])) if q.gold_sets else 0
            for q in ix.queries]
    tru = [int(round(_st.mean([len(g) for g in q.gold_sets])))
           for q in ix.queries if q.gold_sets]
    print(f"wrote {out_path}: {len(sizes)} queries  (role head only, "
          f"{len(folds)}-fold grouped CV, seed {cfg.seed})")
    print(f"predicted |R_q|: mean {_st.mean(vals):.4f}  median "
          f"{_st.median(vals):.0f}  min {min(vals)}  max {max(vals)}")
    print(f"true mean gold set size: {_st.mean(tru):.4f}   "
          f"gold union size: {_st.mean(gold):.4f}")
    dist = Counter(vals)
    print(f"\n{'n':>4s} {'queries':>8s} {'share':>8s}")
    for kk in sorted(dist):
        print(f"{kk:>4d} {dist[kk]:>8d} {dist[kk]/len(vals):>7.1%}")
    exact = sum(1 for a, b in zip(vals, tru) if a == b)
    print(f"\nbudget == rounded true size on {exact}/{len(tru)} queries "
          f"({exact/max(1,len(tru)):.1%})")
    print(f"mean signed error {_st.mean([a - b for a, b in zip(vals, tru)]):+.4f}"
          f"   mean |error| {_st.mean([abs(a - b) for a, b in zip(vals, tru)]):.4f}")

    if compare:
        with open(os.path.expanduser(compare)) as f:
            other = {str(kk): int(v) for kk, v in json.load(f).items()}
        shared = sorted(set(sizes) & set(other))
        if shared:
            d = [sizes[q] - other[q] for q in shared]
            agree = sum(1 for x in d if x == 0)
            print(f"\nvs {os.path.basename(compare)} over {len(shared)} queries: "
                  f"equal {agree}/{len(shared)} ({agree/len(shared):.1%}), "
                  f"mean signed {_st.mean(d):+.4f}")
            for kk, v in sorted(Counter(d).items()):
                print(f"  {kk:>+6d} {v:>8d}")
        else:
            print(f"\n[WARN] no shared query ids with {compare}")
    return sizes


def sweep_sizes(cfg: Config, sizes: Sequence[int] = (1, 2, 3, 4, 5, 6)) -> Dict:
    """F1 as a function of |R_q|, for top-k and for the method at a fixed size.

    This is the experiment that separates the two things a set-retrieval result
    confounds: how many resources you return, and which ones. Both curves are
    evaluated on the same pools with the same metric, so the vertical gap at a
    given size is selection quality and the shape of each curve is the size
    effect. Run it before concluding anything about the selector -- with Dice
    F1 against gold sets of ~2.5, size alone moves the number a long way.
    """
    ix = Index(cfg)
    out: Dict[str, Dict] = {}
    print(f"{'|R_q|':>6s} {'method_F1':>10s} {'topk_F1':>9s} {'m_P':>7s} {'m_R':>7s} "
          f"{'t_P':>7s} {'t_R':>7s} {'m_cov':>7s} {'t_cov':>7s}")
    for n in sizes:
        # Spend one slot on a figure when the size allows it, matching the
        # composition the role head would ask for at that budget.
        prof = f"{n-1},1" if n > 1 else "1,0"
        c = Config(**{**cfg.__dict__, "selector": "greedy",
                      "profile_mode": "fixed", "fixed_profile": prof,
                      "out_dir": os.path.join(cfg.out_dir, "sizes", str(n))})
        m = run(c, ix.rebind(c), quiet=True)
        sysm, tk = m["system"], m["topk_baseline"]
        out[str(n)] = {"method": sysm, "topk": tk}
        print(f"{sysm['set_size']:>6.2f} {sysm['f1']:>10.4f} {tk['f1']:>9.4f} "
              f"{sysm['precision']:>7.4f} {sysm['recall']:>7.4f} "
              f"{tk['precision']:>7.4f} {tk['recall']:>7.4f} "
              f"{sysm['concept_coverage']:>7.4f} {tk['concept_coverage']:>7.4f}")
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "size_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def sweep_beta(cfg: Config, betas: Sequence[float] = (0.0, 0.05, 0.1, 0.25, 0.5,
                                                      1.0, 2.0, 5.0, 20.0)) -> Dict:
    """The coverage/relevance frontier.

    beta -> 0   : pure coverage (most diverse, least relevance-driven)
    beta -> inf : reproduces top-k, which is the maximally redundant selection
    Report this. It is the cleanest evidence that the coverage term is doing
    work, and it doubles as a sanity check that the top-k limit is recovered.
    """
    ix = Index(cfg)
    out = {}
    _hdr()
    for b in betas:
        c = Config(**{**cfg.__dict__, "beta": float(b),
                      "out_dir": os.path.join(cfg.out_dir, "beta", str(b))})
        out[str(b)] = run(c, ix.rebind(c), quiet=True)["system"]
        _line(f"beta={b}", out[str(b)])
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "beta_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


# ------------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser(description="Intent-conditioned role-aware set retrieval")
    ap.add_argument("command", choices=["run", "check", "vocab", "audit",
                                        "ablate", "sweep", "sizes", "rolesizes",
                                        "uqmix", "uqtau", "uqk", "conclf", "contune", "uqtune", "errors", "ksweep", "e2esweep", "diag",
                                        "eval-file"])
    ap.add_argument("--sizes-out", default=None,
                    help="rolesizes: where to write the budget file")
    ap.add_argument("--compare-sizes", default=None,
                    help="rolesizes: an existing sizes file to diff against")
    ap.add_argument("--resources", dest="resources_path")
    ap.add_argument("--queries", dest="queries_path")
    ap.add_argument("--train-queries", dest="train_queries_path")
    ap.add_argument("--silver-queries", dest="silver_queries_path")
    ap.add_argument("--epochs-grid", default=None,
                    help="contune: comma-separated epoch budgets")
    ap.add_argument("--tau-grid", default=None,
                    help="contune: comma-separated probability floors")
    ap.add_argument("--hidden-grid", default=None,
                    help="contune: comma-separated hidden widths")
    ap.add_argument("--lr-grid", default=None,
                    help="contune: comma-separated learning rates")
    ap.add_argument("--topk-grid", default=None,
                    help="uqtune: comma-separated |U_q| decodes")
    ap.add_argument("--target-grid", default=None,
                    help="uqtune: comma-separated training targets")
    ap.add_argument("--tune-on", default="silver",
                    choices=["silver", "gold-nested"],
                    help="uqtune: which queries the SELECTION may see")
    ap.add_argument("--wd-grid", default=None,
                    help="contune: comma-separated weight decays")
    ap.add_argument("--size-grid", default=None,
                    help="contune: comma-separated decodes (topm,match)")
    ap.add_argument("--tune-metric", default="micro", choices=["micro", "macro"],
                    help="contune: inner-split selection criterion")
    ap.add_argument("--diag-samples", type=int, default=60,
                    help="diag: random subsets sampled per query for D1")
    ap.add_argument("--intent-order", default=None,
                    help="diag: comma-separated RAW intent labels, in the order "
                         "the table should list them")
    ap.add_argument("--latex", action="store_true",
                    help="diag: also print the table body as LaTeX rows")
    ap.add_argument("--only", default=None,
                    help="ablate only: comma-separated row names to run")
    ap.add_argument("--uq-file", dest="uq_file",
                    help="external U_q per query (vocab ids or free text)")
    ap.add_argument("--candidates", dest="candidates_path",
                    help="precomputed first-stage pools (benchmark candidate file)")
    ap.add_argument("--predictions", help="eval-file: external predictions JSON")
    ap.add_argument("--label", help="eval-file: name for this system in the output")
    ap.add_argument("--out", dest="out_dir")
    ap.add_argument("--config", help="JSON file of Config overrides")
    ap.add_argument("--taus", help="uqtau: comma-separated absolute tau grid; "
                    "default is derived from the data")
    ap.add_argument("--kappas", help="uqtau: comma-separated kappa grid (default 5)")
    ap.add_argument("--sizes", dest="sizes_file",
                    help="JSON {query_id: n} from tools_target_sizes.py; makes "
                         "every system use the same per-query set size")
    ap.add_argument("--run", help="errors: a finished run directory containing "
                    "predictions.jsonl")
    ap.add_argument("--no-type-constraint", dest="type_constraint",
                    action="store_false", default=None,
                    help="drop the per-type capacity constraint, keep the size")
    ap.add_argument("--uq-ks", dest="uq_ks",
                    help="uqk: comma-separated k grid (default 1..20)")
    for name, typ in [("encoder_backend", str), ("encoder_name", str),
                      ("cache_dir", str), ("uq_mode", str), ("coverage", str),
                      ("greedy", str), ("relevance", str), ("profile_mode", str),
                      ("role_loss", str), ("count_decode", str), ("vocab_source", str), ("concept_target", str), ("con_source", str),
                      ("fixed_profile", str),
                      ("uq_target", str),
                      ("selector", str), ("e2e_device", str), ("e2e_early_stop", str),
                      ("e2e_decode", str), ("e2e_size_source", str), ("e2e_gold_match", str), ("cc_population", str), ("role_train", str),
                      ("silver_queries_path", str),
                      ("w_mode", str), ("concept_activation", str), ("backfill", str),
                      ("concept_train", str), ("con_group", str),
                      ("con_learned_size", str), ("sizes_composition", str)]:
        ap.add_argument(f"--{name.replace('_', '-')}", dest=name, type=typ)
    for name in ["beta", "gamma", "eta", "lam", "tag_threshold", "gamma_learned",
                 "vocab_max_df_frac", "vocab_structural_frac", "uq_oracle_mix",
                 "tag_z_threshold", "tag_rel", "prf_alpha",
                 "e2e_lr", "e2e_dropout", "e2e_weight_decay", "e2e_dev_frac",
                 "e2e_lam_type", "e2e_null_weight", "e2e_fire_threshold",
                 "mmr_lambda", "anchored_rel", "anchored_cos", "uq_abs_tau",
                 "concept_lr", "concept_ft_lr", "res_concept_lr",
                 "res_concept_wd", "con_learned_tau"]:
        ap.add_argument(f"--{name.replace('_', '-')}", dest=name, type=float)
    for name in ["K", "n_max", "kappa", "uq_min", "folds", "seed", "tag_topm", "role_epochs",
                 "vocab_min_df", "prf_docs", "uq_file_per_text", "e2e_epochs", "e2e_d_model",
                 "e2e_layers", "e2e_heads", "e2e_batch_size", "e2e_patience",
                 "e2e_silver_epochs",
                 "K_figure", "exact_gap_topn", "exact_gap_queries",
                 "e2e_fixed_size",
                 "concept_epochs", "concept_hidden", "concept_topk",
                 "concept_ft_epochs", "res_concept_epochs", "res_concept_hidden",
                 "res_concept_batch", "con_folds", "con_learned_topm"]:
        ap.add_argument(f"--{name.replace('_', '-')}", dest=name, type=int)
    for name in ["stratified_first_stage", "exact_gap_check", "uq_oracle",
                 "vocab_drop_structural", "allow_oracles", "e2e_loss_norm", "e2e_single_gold",
                 "tag_zscore", "gamma_relative", "stop_on_nonpositive",
                 "restrict_to_subject", "type_constraint", "e2e_cv"]:
        ap.add_argument(f"--{name.replace('_', '-')}", dest=name,
                        action="store_true", default=None)
    ns = ap.parse_args()

    cfg = Config.load(ns.config) if ns.config else Config()
    for k, v in vars(ns).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    if ns.command == "check":
        _res, _q = load_resources(cfg.resources_path), load_queries(cfg.queries_path)
        assign_groups(_q, cfg.cv_group, cfg.cv_group_prefix_len)
        rep = validate(_res, _q)
        ix_vocab = build_vocabulary(
            _res, cfg.vocab_source, cfg.vocab_min_df, cfg.vocab_max_df_frac,
            cfg.vocab_drop_structural, cfg.vocab_drop_generic,
            cfg.vocab_context_descriptors, cfg.vocab_structural_frac)
        rep["vocabulary_size"] = len(ix_vocab)
        rep["vocabulary_sample"] = ix_vocab.names[:15]
        print(json.dumps(rep, indent=2))
    elif ns.command == "eval-file":
        eval_file(cfg, ns.predictions, ns.label or "")
    elif ns.command == "audit":
        audit(cfg)
    elif ns.command == "vocab":
        vocab_report(cfg)
    elif ns.command == "run":
        run(cfg)
    elif ns.command == "ablate":
        ablate(cfg, only=([x.strip() for x in ns.only.split(",") if x.strip()]
                          if ns.only else None))
    elif ns.command == "sizes":
        sweep_sizes(cfg)
    elif ns.command == "rolesizes":
        role_sizes(cfg, out_path=(ns.sizes_out or
                                  os.path.join(cfg.out_dir, "sizes_pred.json")),
                   compare=ns.compare_sizes)
    elif ns.command == "uqmix":
        sweep_uq(cfg)
    elif ns.command == "errors":
        if not ns.run:
            raise SystemExit("errors: pass --run <dir with predictions.jsonl>")
        error_breakdown(cfg, ns.run)
    elif ns.command == "uqk":
        sweep_uq_k(cfg, ks=([int(x) for x in ns.uq_ks.split(",")]
                            if ns.uq_ks else (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20)))
    elif ns.command == "uqtune":
        def _g2(v, cast, dflt):
            return [cast(x) for x in v.split(",")] if v else dflt
        tune_uq_head(
            cfg,
            hiddens=_g2(ns.hidden_grid, int, None),
            lrs=_g2(ns.lr_grid, float, None),
            epochs_grid=_g2(ns.epochs_grid, int, (100, 200, 300, 600, 1000)),
            topks=_g2(ns.topk_grid, int, (3, 5, 8, 12, 16, 20)),
            targets=_g2(ns.target_grid, str, ("gold_con_ann", "query")),
            tune_on=ns.tune_on, metric=ns.tune_metric)
    elif ns.command == "contune":
        def _g(v, cast, dflt):
            return [cast(x) for x in v.split(",")] if v else dflt
        tune_con_classifier(
            cfg,
            epochs_grid=_g(ns.epochs_grid, int, (15, 30, 60, 120, 200)),
            taus=_g(ns.tau_grid, float, (0.1, 0.2, 0.3, 0.5, 0.7)),
            hiddens=_g(ns.hidden_grid, int, None),
            lrs=_g(ns.lr_grid, float, None),
            wds=_g(ns.wd_grid, float, None),
            size_modes=_g(ns.size_grid, str, ("topm", "match")),
            metric=ns.tune_metric)
    elif ns.command == "conclf":
        compare_concept_predictors(
            cfg, ks=([int(x) for x in ns.uq_ks.split(",")]
                     if ns.uq_ks else (3, 5, 8, 12, 16, 20)))
    elif ns.command == "uqtau":
        sweep_tau(cfg,
                  taus=([float(x) for x in ns.taus.split(",")] if ns.taus else None),
                  kappas=([int(x) for x in ns.kappas.split(",")]
                          if ns.kappas else (5,)))
    elif ns.command == "ksweep":
        sweep_k(cfg)
    elif ns.command == "e2esweep":
        sweep_e2e(cfg)
    elif ns.command == "diag":
        diagnostics(cfg, n_samples=ns.diag_samples,
                    intent_order=([x.strip() for x in ns.intent_order.split(",")]
                                  if ns.intent_order else None),
                    latex=ns.latex)
    else:
        sweep_beta(cfg)


if __name__ == "__main__":
    main()
