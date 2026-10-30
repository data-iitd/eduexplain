#!/usr/bin/env python3
"""Train once, save to disk, then answer new queries. Both paper models.

`pipeline run` is an EXPERIMENT: it fits under grouped k-fold cross-validation
and reports held-out numbers. There is no single trained model at the end of
it -- there are k fold models, each of which has seen four fifths of the data,
and each query was answered by the one fold that had not seen it. That is the
right protocol for a table and the wrong artifact for deployment: you cannot
hand a fold model a new query, because "which fold has not seen this query" is
undefined for a query that was not in the evaluation set.

This module closes that gap. It fits ONE model on all the labelled data you
give it, writes every piece of state a new query needs into a directory, and
loads it back to answer queries one at a time.

  train    fit and save            minutes (lightweight) / longer (e2e)
  predict  answer a query          one encoder call + the selector

THE TWO PAPER MODELS. --model-type picks the configuration; both match the
README's final rows exactly except for the fitting protocol, which is the
whole point of this file.

  lightweight   cosine U_q against the concept vocabulary, a LEARNED Con(r)
                classifier, the role head deciding the size. Nothing but the
                role head is learned at query time, so the checkpoint is small
                and inference is one encode plus a greedy selection.
                = README "SetCoRe lightweight (cos / clf, free size)"

  e2e           the learned set predictor over candidate embeddings, trained
                with the best-match loss, size from the role head.
                = README "SetCoRe e2e (best-match, gold CV)"

WHAT IS SAVED, AND WHY EACH PIECE

  manifest.json     the full Config, the model type, what it was fitted on,
                    and the encoder it needs at query time
  arrays.npz        R (resource embeddings), V (vocabulary descriptor
                    embeddings), P_full (the tagged Con(r) matrix), df, types
  vocab.json        the concept vocabulary
  resources.json    a COPY of the corpus, so the checkpoint is self-contained
                    and a prediction can return resources, not bare ids
  role_head.npz     the role head's weights
  concept_head.npz  only under --uq-mode learned
  e2e.pt            only under --model-type e2e

P_full is stored rather than the resource-concept classifier that produced it.
Con(r) is a property of the corpus, not of the query, so tagging is a
train-time step; storing the tags means the classifier never has to be
reloaded and the served tags are bit-identical to the trained ones.

THE PROTOCOL CHANGE IS DELIBERATE AND MUST BE REPORTED. A model fitted on all
321 gold queries cannot be evaluated on those queries -- it has seen them. The
paper's numbers stay the CV numbers from `pipeline run`; this checkpoint is
for answering new queries. --fit-on makes the choice explicit and the manifest
records it.

Usage
-----
  # lightweight, fitted on the training queries only (keeps the gold set clean)
  python -m icra.serve train --model-type lightweight \\
      --resources $R --queries $Q --train-queries $T \\
      --encoder-name $BGE --fit-on train --out ~/models/setcore_lw

  # e2e, fitted on everything, for deployment
  python -m icra.serve train --model-type e2e \\
      --resources $R --queries $Q --train-queries $T \\
      --encoder-name $BGE --fit-on all --out ~/models/setcore_e2e

  python -m icra.serve predict --model ~/models/setcore_lw \\
      --query "Explain how a concave mirror forms a real image"

  python -m icra.serve predict --model ~/models/setcore_lw \\
      --queries-file new_queries.json --out answers.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import concepts as C
from . import select as S
from .config import Config
from .data import Query, Vocabulary, assign_groups, load_queries, load_resources
from .encoders import build_encoder
from .heads import ConceptHead, RoleHead
from .pipeline import Index, _concept_targets_matrix

MANIFEST = "manifest.json"
ARRAYS = "arrays.npz"
VOCAB = "vocab.json"
RESOURCES = "resources.json"
ROLE = "role_head.npz"
CONCEPT = "concept_head.npz"
E2E = "e2e.pt"


# ---------------------------------------------------------------- the presets
#
# Exactly the README's two final rows. Anything the CLI sets explicitly wins,
# so these are a starting point rather than a cage -- but changing one means
# the served model is no longer the model the paper reports.
PRESETS: Dict[str, Dict] = {
    "lightweight": {
        "selector": "greedy",
        "uq_mode": "match",
        "kappa": 8,
        "con_source": "learned",
        "con_group": "chapter",
        "con_learned_size": "topm",
        "con_learned_tau": 0.7,
        "res_concept_hidden": 512,
        "res_concept_lr": 1e-3,
        "res_concept_wd": 1e-5,
        "res_concept_epochs": 400,
    },
    "e2e": {
        "selector": "e2e",
        "e2e_d_model": 128,
        "e2e_layers": 1,
        "e2e_dropout": 0.3,
        "e2e_lr": 1e-4,
        "e2e_gold_match": "best",
        "e2e_size_source": "role",
        # e2e_cv is the EXPERIMENT's protocol and is deliberately off here:
        # a served model is fitted once, not once per fold.
        "e2e_cv": False,
    },
}


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


# ================================================================== TRAINING

def _fit_queries(ix: Index, fit_on: str) -> List[Query]:
    """Which labelled queries supervise the heads.

    'train' uses only --train-queries, which leaves the gold set genuinely
    held out, so the paper's CV numbers remain an honest estimate of what this
    checkpoint does. 'all' adds the gold queries: a stronger model and the
    right choice for deployment, but every number measured on those queries is
    now a training number.
    """
    if fit_on == "train":
        if not ix.train_queries:
            raise SystemExit(
                "--fit-on train needs --train-queries. Pass the silver/training "
                "file, or\n  use --fit-on all to fit on the gold queries too "
                "(and stop quoting gold-set\n  numbers for this checkpoint).")
        return list(ix.train_queries)
    if fit_on == "gold":
        return list(ix.queries)
    return list(ix.train_queries) + list(ix.queries)


def _embed_for(ix: Index, queries: Sequence[Query]) -> np.ndarray:
    """Embeddings for a query list, reusing the Index's cached ones when the
    list IS one of the Index's own (the cache makes this cheap either way)."""
    if queries is ix.queries or list(queries) == list(ix.queries):
        return ix.Q
    if ix.train_queries and list(queries) == list(ix.train_queries):
        return ix.Q_train
    X = ix.enc.encode([q.text for q in queries])
    ix.enc.flush()
    return X


def _fit_role_head(ix: Index, cfg: Config, fitq: Sequence[Query],
                   X: np.ndarray) -> RoleHead:
    multi = cfg.role_loss in ("best_match", "single_first", "single_random",
                              "average")
    ct = [ix.targets_for(q)[0] if multi else [ix.mean_targets_for(q)[0]]
          for q in fitq]
    ft = [ix.targets_for(q)[1] if multi else [ix.mean_targets_for(q)[1]]
          for q in fitq]
    return RoleHead(X.shape[1], cfg.n_max, cfg.role_hidden, cfg.role_lr,
                    cfg.role_weight_decay, cfg.role_epochs, cfg.lam,
                    cfg.role_loss, cfg.seed).fit(X, ct, ft)


def _fit_concept_head(ix: Index, cfg: Config, fitq: Sequence[Query],
                      X: np.ndarray) -> ConceptHead:
    """One head on all the fit rows. `pipeline run` cross-validates this too;
    a served head must see everything, for the same reason the role head
    must."""
    pos = {v: j for j, v in enumerate(ix.vocab.ids)}
    Y = _concept_targets_matrix(ix, fitq, pos)
    head = ConceptHead(X.shape[1], len(ix.vocab), cfg.concept_hidden,
                       cfg.concept_lr, cfg.concept_epochs,
                       cfg.concept_activation, cfg.seed)
    head.fit(X, Y)
    return head


def _fit_e2e(ix: Index, cfg: Config, fitq: Sequence[Query], X: np.ndarray,
             quiet: bool):
    from . import e2e as E2

    ex = []
    for k, q in enumerate(fitq):
        pool, sc = ix.retrieve_for(X[k], q)
        ex.append(E2.build_example(ix, q, X[k], pool, sc))

    # Dev split by GROUP, never by query: surface-form variants of one item
    # share gold sets and near-identical embeddings, so splitting them across
    # train and dev leaks and the early stop fires on a memorised score.
    groups = [q.group_id for q in fitq]
    uniq = np.unique(np.array(groups))
    rs = np.random.default_rng(cfg.seed)
    rs.shuffle(uniq)
    n_dev = int(round(cfg.e2e_dev_frac * len(uniq)))
    dev_g = set(uniq[:n_dev].tolist()) if n_dev > 0 else set()
    tr = [ex[i] for i, g in enumerate(groups) if g not in dev_g]
    dv = [ex[i] for i, g in enumerate(groups) if g in dev_g]
    if not quiet:
        print(f"E2E: fitting on {len(tr)} queries ({len(dv)} held out for "
              f"early stopping)")
    trainer = E2.E2ETrainer(cfg, X.shape[1], cfg.e2e_device)
    trainer.fit(tr, dv or None, tag="serve", quiet=quiet)
    return trainer


def train(cfg: Config, out_dir: str, model_type: str, fit_on: str,
          quiet: bool = False) -> Dict:
    t0 = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)

    # Checked BEFORE the index is built. Building it encodes the whole corpus
    # and fits the Con(r) tagger, so finding out afterwards that the fit set
    # is empty wastes the expensive half of the run.
    if fit_on == "train" and not cfg.train_queries_path:
        raise SystemExit(
            "--fit-on train needs --train-queries. Pass the silver/training "
            "file, or\n  use --fit-on all to fit on the gold queries too (and "
            "stop quoting gold-set\n  numbers for this checkpoint).")

    if cfg.candidates_path and not quiet:
        # A precomputed pool is keyed by query_id, and a NEW query has no
        # entry in it, so the served model always retrieves densely. That
        # reproduces the pool only if the file IS the dense top-K under this
        # encoder -- which it is when it came from bge_retrieve_matched.py at
        # the same K, and is not when it came from anywhere else.
        print(f"  [i] --candidates is used for TRAINING only. At query time "
              f"there is no pool file\n      entry for an unseen query, so "
              f"retrieval is dense over the corpus at K={cfg.K}.\n      That "
              f"matches your runs only if the pool file is the dense top-K "
              f"under this\n      same encoder.")

    if not quiet:
        print(f"building the index ({cfg.encoder_name})")
    ix = Index(cfg)
    fitq = _fit_queries(ix, fit_on)
    if not fitq:
        raise SystemExit("no labelled queries to fit on")
    X = _embed_for(ix, fitq)

    n_gold_in_fit = sum(1 for q in fitq
                        if q.query_id in {x.query_id for x in ix.queries})
    if not quiet:
        print(f"fitting on {len(fitq)} queries (--fit-on {fit_on})")
        if n_gold_in_fit:
            print(f"  [!] {n_gold_in_fit} of them are EVALUATION queries. This "
                  f"checkpoint has seen the\n      gold set, so no number "
                  f"measured on those queries describes it. The paper's\n"
                  f"      numbers come from `pipeline run`, which "
                  f"cross-validates; keep quoting those.")

    if not quiet:
        print("fitting the role head")
    role = _fit_role_head(ix, cfg, fitq, X)

    chead = None
    if cfg.uq_mode == "learned":
        if not quiet:
            print("fitting the concept head (U_q)")
        chead = _fit_concept_head(ix, cfg, fitq, X)

    trainer = None
    if cfg.selector == "e2e":
        trainer = _fit_e2e(ix, cfg, fitq, X, quiet)

    # ------------------------------------------------------------- serialise
    np.savez_compressed(
        os.path.join(out_dir, ARRAYS),
        R=ix.R.astype(np.float32), V=ix.V.astype(np.float32),
        P_full=ix.P_full.astype(np.float32),
        df=np.asarray(ix.df, dtype=np.float64),
        types=np.asarray(ix.types, dtype=object).astype(str))

    with open(os.path.join(out_dir, VOCAB), "w", encoding="utf-8") as fh:
        json.dump({"ids": list(ix.vocab.ids), "names": list(ix.vocab.names),
                   "descriptors": list(ix.vocab.descriptors),
                   "df": [int(x) for x in (ix.vocab.df or [])],
                   "is_structural": [bool(x) for x in
                                     (ix.vocab.is_structural or [])],
                   "structural_frac": [float(x) for x in
                                       (ix.vocab.structural_frac or [])]},
                  fh)

    # The corpus is copied in so the checkpoint answers with RESOURCES, not
    # ids a caller then has to resolve against a file that may have moved.
    with open(os.path.expanduser(cfg.resources_path), encoding="utf-8") as fh:
        raw_res = json.load(fh)
    with open(os.path.join(out_dir, RESOURCES), "w", encoding="utf-8") as fh:
        json.dump(raw_res, fh, ensure_ascii=False)

    np.savez_compressed(os.path.join(out_dir, ROLE),
                        **{k: np.asarray(v) for k, v in role.P.items()})
    if chead is not None:
        np.savez_compressed(os.path.join(out_dir, CONCEPT),
                            **{k: np.asarray(v) for k, v in chead.P.items()})
    if trainer is not None:
        import torch
        torch.save(trainer.model.state_dict(), os.path.join(out_dir, E2E))

    manifest = {
        "format": 1,
        "model_type": model_type,
        "fit_on": fit_on,
        "n_fit_queries": len(fitq),
        "n_eval_queries_in_fit": int(n_gold_in_fit),
        "n_resources": len(ix.resources),
        "n_vocab": len(ix.vocab),
        "dim": int(ix.R.shape[1]),
        "has_concept_head": chead is not None,
        "has_e2e": trainer is not None,
        "best_dev_f1": _jsonable(getattr(trainer, "best_dev", None))
        if trainer is not None else None,
        "best_epoch": _jsonable(getattr(trainer, "best_epoch", None))
        if trainer is not None else None,
        "role_head": {"n_max": cfg.n_max, "hidden": cfg.role_hidden,
                      "role_loss": cfg.role_loss},
        "encoder": {"name": cfg.encoder_name, "backend": cfg.encoder_backend,
                    "note": "needed at query time to embed the new query; the "
                            "resource and vocabulary embeddings are stored"},
        "config": {k: _jsonable(v) for k, v in cfg.__dict__.items()},
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_seconds": round(time.perf_counter() - t0, 1),
    }
    with open(os.path.join(out_dir, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    if not quiet:
        mb = sum(os.path.getsize(os.path.join(out_dir, f))
                 for f in os.listdir(out_dir)) / 1e6
        print(f"\nsaved {model_type} -> {out_dir}  ({mb:.1f} MB, "
              f"{manifest['train_seconds']}s)")
        print(f"  predict with:  python -m icra.serve predict --model "
              f"{out_dir} --query \"...\"")
    return manifest


# ================================================================= SERVING

class ServedIndex(Index):
    """An Index restored from disk instead of recomputed.

    It subclasses Index and skips its __init__ on purpose: `retrieve_for` and
    the type/subject masking are the first stage, and a served query must go
    through the SAME code as a trained one. Re-implementing retrieval here is
    how a served model quietly stops being the model that was measured.
    """

    def __init__(self, path: str, encoder=None):
        path = os.path.expanduser(path)
        with open(os.path.join(path, MANIFEST), encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        self.cfg = Config(**{k: v for k, v in self.manifest["config"].items()
                             if k in Config.__dataclass_fields__})

        z = np.load(os.path.join(path, ARRAYS), allow_pickle=False)
        self.R = z["R"]
        self.V = z["V"]
        self.P_full = z["P_full"]
        self.df = z["df"]
        self.types = z["types"].astype(object)

        with open(os.path.join(path, VOCAB), encoding="utf-8") as fh:
            v = json.load(fh)
        self.vocab = Vocabulary(ids=v["ids"], names=v["names"],
                                descriptors=v["descriptors"],
                                df=v.get("df") or [],
                                is_structural=v.get("is_structural") or [],
                                structural_frac=v.get("structural_frac") or [])
        self.Con = C.concept_sets(self.P_full, self.vocab)

        self.resources = load_resources(os.path.join(path, RESOURCES))
        self.rid2i = {r.resource_id: i for i, r in enumerate(self.resources)}

        # Index attributes the inherited methods touch.
        self.queries = []
        self.train_queries = []
        self.Q = np.zeros((0, self.R.shape[1]), dtype=np.float32)
        self.Q_train = self.Q
        self.candidates = None
        self.uq_ext = None
        self.con_diag = {}

        self.enc = encoder if encoder is not None else build_encoder(self.cfg)
        if int(self.enc.dim) != int(self.R.shape[1]):
            raise SystemExit(
                f"encoder dimension {self.enc.dim} does not match the stored "
                f"embeddings ({self.R.shape[1]}).\n  The checkpoint was built "
                f"with {self.manifest['encoder']['name']}; serving it with a "
                f"different\n  encoder would score the new query against "
                f"resource vectors from another space.")


class ServedModel:
    """Load once, then call `.predict(text)` per query."""

    def __init__(self, path: str, encoder=None, quiet: bool = True):
        path = os.path.expanduser(path)
        self.path = path
        self.ix = ServedIndex(path, encoder)
        self.cfg = self.ix.cfg
        self.manifest = self.ix.manifest
        dim = int(self.ix.R.shape[1])

        rp = np.load(os.path.join(path, ROLE))
        self.role = RoleHead(dim, self.cfg.n_max, self.cfg.role_hidden,
                             self.cfg.role_lr, self.cfg.role_weight_decay,
                             self.cfg.role_epochs, self.cfg.lam,
                             self.cfg.role_loss, self.cfg.seed)
        self.role.P = {k: rp[k] for k in rp.files}

        self.chead = None
        if self.manifest.get("has_concept_head"):
            cp = np.load(os.path.join(path, CONCEPT))
            self.chead = ConceptHead(dim, len(self.ix.vocab),
                                     self.cfg.concept_hidden, self.cfg.concept_lr,
                                     self.cfg.concept_epochs,
                                     self.cfg.concept_activation, self.cfg.seed)
            self.chead.P = {k: cp[k] for k in cp.files}

        self.trainer = None
        if self.manifest.get("has_e2e"):
            import torch
            from . import e2e as E2
            self.trainer = E2.E2ETrainer(self.cfg, dim, self.cfg.e2e_device)
            self.trainer.model.load_state_dict(
                torch.load(os.path.join(path, E2E),
                           map_location=self.cfg.e2e_device))
            self.trainer.model.eval()

        if not quiet:
            # stderr, not stdout: --json writes machine-readable answers to
            # stdout and a banner mixed into them makes the whole stream
            # unparseable for the caller.
            import sys
            print(f"loaded {self.manifest['model_type']} from {path}: "
                  f"{self.manifest['n_resources']} resources, "
                  f"{self.manifest['n_vocab']} concepts", file=sys.stderr)

    # ------------------------------------------------------------------ core
    def predict(self, text: str, subject: Optional[str] = None,
                class_no: Optional[int] = None, n: Optional[int] = None,
                query_id: str = "q") -> Dict:
        """Answer one query. Returns the chosen resources and how they were
        chosen: the predicted profile, the concept ground set, and -- for the
        lightweight model -- the per-pick marginal coverage gain."""
        cfg = self.cfg
        ix = self.ix
        q = Query(query_id=str(query_id), text=str(text), gold_sets=[],
                  subject=subject, class_no=class_no)

        t0 = time.perf_counter()
        q_emb = ix.enc.encode([q.text])[0]
        t_enc = time.perf_counter() - t0

        t0 = time.perf_counter()
        pool, s = ix.retrieve_for(q_emb, q)
        t_ret = time.perf_counter() - t0

        # ---- profile pi_q = (c_q, f_q) from the role head
        t0 = time.perf_counter()
        cq, fq, pf = self.role.predict(q_emb[None, :], cfg.count_decode, cfg.eta)
        cq, fq = int(cq[0]), int(fq[0])
        if cfg.profile_mode == "fixed":
            cq, fq = cfg.parsed_fixed_profile()
        cq = max(0, min(cq, cfg.n_max))
        if cq + fq > cfg.n_max:          # the figure slot gives way last
            fq = max(0, cfg.n_max - cq)
        profile = {"explanation": cq, "figure": fq}
        cfg_q = cfg
        if n is not None:
            # A caller-supplied size keeps the role head's FIGURE decision and
            # spends the rest on explanations -- the same rule --sizes uses in
            # the experiment, so an overridden size is still this model's
            # composition.
            n = max(1, min(int(n), cfg.n_max))
            f_ = 1 if (fq and n >= 2) else 0
            profile = {"explanation": n - f_, "figure": f_}

        pool_ids = [ix.resources[i].resource_id for i in pool]
        ptypes = ix.types[pool]
        P_pool = ix.P_full[pool]

        # ---- U_q
        if cfg.uq_mode == "learned" and self.chead is not None:
            rho = self.chead.predict(q_emb[None, :])[0]
            sig = C.uq_learned(rho, ix.vocab, P_pool, cfg,
                               a=ix.V @ q_emb, df=ix.df,
                               n_res=len(ix.resources))
        elif cfg.uq_mode == "anchored":
            sig = C.uq_anchored(q.text, ix.R[pool], ix.enc, cfg)
        else:
            sig = C.uq_match(q_emb, ix.V, P_pool, ix.vocab, cfg,
                             df=ix.df, n_res=len(ix.resources))

        # ---- the selection itself
        if self.trainer is not None:
            from . import e2e as E2
            ex = E2.build_example(ix, q, q_emb, pool, s)
            ov = None
            if cfg.e2e_size_source in ("role", "role_profile"):
                if cfg.e2e_size_source == "role":
                    ov = [{"size": max(1, cq + fq)}]
                else:
                    ov = [{"profile": dict(profile),
                           "types": [str(t) for t in ptypes]}]
            if n is not None:
                ov = [{"size": n}]
            idx = self.trainer.predict([ex], ov)[0]
            filled = {t: sum(1 for i in idx if ptypes[i] == t)
                      for t in ("explanation", "figure")}
            sel = S.Selection(indices=idx, order=[], objective=0.0,
                              coverage=S.coverage_of(sig.P, sig.w, idx,
                                                     cfg.coverage),
                              slots_requested=dict(profile),
                              slots_filled=filled)
        else:
            sel = S.select(ptypes, s, sig, profile, cfg_q)
        t_sel = time.perf_counter() - t0

        chosen = []
        for rank, i in enumerate(sel.indices, start=1):
            r = ix.resources[pool[i]]
            chosen.append({
                "rank": rank,
                "resource_id": r.resource_id,
                "type": r.type,
                "subject": r.subject,
                "chapter_title": r.chapter_title,
                "summary": r.index_text()[:400],
                "retrieval_score": float(s[i]),
                "concepts": [ix.vocab.ids[j]
                             for j in np.nonzero(ix.P_full[pool[i]])[0]][:12],
            })

        return {
            "query_id": q.query_id,
            "query": q.text,
            "predicted_resource_ids": [c["resource_id"] for c in chosen],
            "resources": chosen,
            "profile": {"explanation": int(profile.get("explanation", 0)),
                        "figure": int(profile.get("figure", 0))},
            "n_predicted": len(chosen),
            "figure_prob": float(pf[0]),
            "concepts_U_q": list(sig.units),
            "coverage": float(sel.coverage),
            "pool_size": int(len(pool_ids)),
            "model_type": self.manifest["model_type"],
            "seconds": {"encode": round(t_enc, 4),
                        "retrieve": round(t_ret, 4),
                        "select": round(t_sel, 4),
                        "total": round(t_enc + t_ret + t_sel, 4)},
        }

    def predict_many(self, queries: Sequence[Dict]) -> List[Dict]:
        out = []
        for i, q in enumerate(queries):
            if isinstance(q, str):
                q = {"query": q}
            out.append(self.predict(
                q.get("query") or q.get("text") or "",
                subject=q.get("subject"), class_no=q.get("class_no"),
                n=q.get("n"), query_id=q.get("query_id", f"q{i:05d}")))
        return out


# ====================================================================== CLI

def _add_config_flags(ap: argparse.ArgumentParser) -> None:
    """Every Config field as a flag, so a served model can be trained with the
    same switches as `pipeline run` and nothing is silently un-settable."""
    for name, f in Config.__dataclass_fields__.items():
        if name in ("resources_path", "queries_path", "train_queries_path",
                    "out_dir"):
            continue
        flag = "--" + name.replace("_", "-")
        if name == "candidates_path":
            # --candidates is what the README calls it.
            ap.add_argument("--candidates", flag, dest=name, default=None)
            continue
        if f.type in (bool, "bool"):
            ap.add_argument(flag, dest=name, default=None,
                            type=lambda x: str(x).lower() in ("1", "true", "yes"))
        elif f.type in (int, "int"):
            ap.add_argument(flag, dest=name, default=None, type=int)
        elif f.type in (float, "float"):
            ap.add_argument(flag, dest=name, default=None, type=float)
        else:
            ap.add_argument(flag, dest=name, default=None)


def _build_cfg(ns, model_type: str) -> Config:
    kw = dict(PRESETS[model_type])
    kw.update({"resources_path": os.path.expanduser(ns.resources),
               "queries_path": os.path.expanduser(ns.queries)})
    if ns.train_queries:
        kw["train_queries_path"] = os.path.expanduser(ns.train_queries)
    # Explicit CLI values override the preset.
    for name in Config.__dataclass_fields__:
        v = getattr(ns, name, None)
        if v is not None and name not in ("resources_path", "queries_path",
                                          "train_queries_path", "out_dir"):
            kw[name] = v
    # role_train is NOT forced here. It selects the experiment's protocol
    # (train file vs k-fold CV) and this module never runs that protocol: it
    # always fits one head on --fit-on. The field is carried into the manifest
    # unchanged so the checkpoint still records the configuration it came from.
    return Config(**kw)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="fit one model and save it")
    t.add_argument("--model-type", required=True,
                   choices=sorted(PRESETS), help="which paper model")
    t.add_argument("--resources", required=True)
    t.add_argument("--queries", required=True,
                   help="the gold/evaluation queries. Needed even under "
                        "--fit-on train: the corpus tagging and the "
                        "vocabulary are built from the same Index.")
    t.add_argument("--train-queries", default=None)
    t.add_argument("--out", required=True, help="checkpoint DIRECTORY")
    t.add_argument("--fit-on", default="auto",
                   choices=["auto", "train", "gold", "all"],
                   help="which labelled queries supervise the heads. 'auto' = "
                        "train when --train-queries is given, else all. "
                        "'train' keeps the gold set genuinely held out; 'all' "
                        "is the strongest deployable model and makes every "
                        "gold-set number a training number.")
    t.add_argument("--quiet", action="store_true")
    _add_config_flags(t)

    p = sub.add_parser("predict", help="answer queries with a saved model")
    p.add_argument("--model", required=True, help="checkpoint directory")
    p.add_argument("--query", default=None)
    p.add_argument("--queries-file", default=None,
                   help="JSON list of strings, or of objects with `query` and "
                        "optionally query_id / subject / class_no / n")
    p.add_argument("--subject", default=None)
    p.add_argument("--class-no", type=int, default=None)
    p.add_argument("--n", type=int, default=None,
                   help="force the set size; the role head still decides "
                        "whether one of the slots is a figure")
    p.add_argument("--out", default=None, help="write the answers as JSON")
    p.add_argument("--interactive", action="store_true",
                   help="read queries from stdin, one per line")
    p.add_argument("--json", action="store_true",
                   help="print raw JSON instead of the readable form")

    ns = ap.parse_args()

    if ns.cmd == "train":
        cfg = _build_cfg(ns, ns.model_type)
        fit_on = ns.fit_on
        if fit_on == "auto":
            fit_on = "train" if cfg.train_queries_path else "all"
        train(cfg, os.path.expanduser(ns.out), ns.model_type, fit_on, ns.quiet)
        return

    if not (ns.query or ns.queries_file or ns.interactive):
        raise SystemExit("predict needs --query, --queries-file or --interactive")

    model = ServedModel(ns.model, quiet=False)

    def show(a: Dict) -> None:
        if ns.json:
            print(json.dumps(a, ensure_ascii=False))
            return
        print()
        print(f"Q: {a['query']}")
        print(f"   profile {a['profile']['explanation']} explanation(s) + "
              f"{a['profile']['figure']} figure(s)   "
              f"|R_q| = {a['n_predicted']}   "
              f"{a['seconds']['total']:.2f}s")
        if a["concepts_U_q"]:
            print(f"   U_q: {', '.join(a['concepts_U_q'][:8])}")
        for c in a["resources"]:
            print(f"   {c['rank']}. [{c['type']:<11s}] {c['resource_id']}  "
                  f"({c['retrieval_score']:.3f})")
            print(f"      {c['summary'][:150]}")

    answers: List[Dict] = []
    if ns.query:
        answers.append(model.predict(ns.query, ns.subject, ns.class_no, ns.n))
        show(answers[-1])
    if ns.queries_file:
        with open(os.path.expanduser(ns.queries_file), encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            raw = raw.get("queries", [])
        answers.extend(model.predict_many(raw))
        for a in answers[-len(raw):]:
            show(a)
    if ns.interactive:
        import sys
        print("\nqueries, one per line; ctrl-D to stop")
        for line in sys.stdin:
            line = line.strip()
            if line:
                a = model.predict(line, ns.subject, ns.class_no, ns.n)
                answers.append(a)
                show(a)

    if ns.out:
        op = os.path.expanduser(ns.out)
        os.makedirs(os.path.dirname(op) or ".", exist_ok=True)
        with open(op, "w", encoding="utf-8") as fh:
            json.dump(answers, fh, ensure_ascii=False, indent=2)
        print(f"\nwritten: {op}  ({len(answers)} queries)")


if __name__ == "__main__":
    main()
