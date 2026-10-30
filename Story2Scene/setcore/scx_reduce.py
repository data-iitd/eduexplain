#!/usr/bin/env python3
"""Stages 1-2 of SetCoRe-X, per the method spec: predict structure, reduce pool.

This replaces the earlier "run icra with --sizes = m*n_q" approach, which could
not express the spec and silently clamped the budget to cfg.n_max. Everything
here follows Section: Method directly.

Predicting answer structure
---------------------------
A head over the frozen retrieval embedding predicts an explanatory count c_q
and a figure indicator f_q in {0,1} (threshold 0.5). Decoding is restricted to
NON-EMPTY profiles of total size at most --profile-max (3):

    c_q <- clip to [0, profile_max]
    f_q <- 1[p_f >= 0.5]
    if c_q + f_q > profile_max:  f_q <- profile_max - c_q     (as in SetCoRe)
    if c_q + f_q == 0:           c_q <- 1                     (non-empty)

The head is trained with icra's own n_max so the targets are unchanged; the
restriction is applied at DECODE, which is what the spec says. n_q = c_q + f_q.

Intermediate budget
-------------------
    B_q(m) = min{K, ceil(m * n_q)},   m >= 1   (m may be fractional)

m controls alternatives per predicted slot, not the final answer size.

Composition-guided reduction
----------------------------
Capacities scale the predicted type proportions to B_q by largest-remainder
apportionment:

    exact[t]  = B_q * pi_q[t] / n_q
    b_q[t]    = floor(exact[t]),  then the B_q - sum(floors) spare slots go to
                the largest fractional remainders, ties broken by a fixed type
                order so the result is deterministic.

Then, from D_q = {}, greedily add the feasible resource with the largest
marginal gain in

    F_q(S) = sum_{v in U_q} w_qv * 1[v in Con(S)] + beta * sum_{r in S} s~(q,r)

subject to those capacities, until B_q resources are selected. Coverage is the
BINARY indicator of the spec, not icra's graded default. s~ is min-max
normalised over C_q with ALL-TIED scores mapped to one (icra's own normaliser
maps them to 0.5; the spec says one, so this module normalises itself).
Zero-gain ties break by retrieval rank. If a category lacks enough candidates,
the remaining budget is filled greedily from ANY category, so scarcity cannot
shorten a pool that has at least B_q resources. At B_q >= |C_q| the reduced
pool is exactly C_q, which is asserted rather than assumed.

Both reducers emit D_q in ORIGINAL RETRIEVAL ORDER, so the LLM sees the same
presentation order whichever reducer produced the pool and a difference between
them cannot be an artefact of ordering.

    --reducer setcore   the composition-guided reduction above
    --reducer topn      the first B_q of the ranked pool (the control)

Usage
-----
  python scx_reduce.py --resources $R --queries $Q --train-queries $T \\
      --m 2 --reducer setcore --out ~/scx/shortlists/setcore_m2.json \\
      --encoder-name ~/models/bge-large-en-v1.5 \\
      --con-source learned --con-group chapter --con-learned-tau 0.7 \\
      --res-concept-hidden 512 --res-concept-epochs 400 --kappa 8 --seed 0
"""
from __future__ import annotations

import argparse
import math
import os
import statistics as st
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scx_common import expand_tildes, load_json, write_json

# ------------------------------------------------------------------- spec bits


def normalise_relevance(s: np.ndarray) -> np.ndarray:
    """Min-max over C_q, with ALL-TIED scores mapped to one.

    icra's normalise_scores maps a degenerate range to 0.5. The spec says one,
    and the difference is not cosmetic: with beta = 1 the relevance term then
    contributes |S| rather than |S|/2, which changes the coverage/relevance
    balance on exactly the queries where the ranking is uninformative.
    """
    s = np.asarray(s, dtype=np.float64)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return np.ones_like(s)
    return (s - lo) / (hi - lo)


def concept_weights(a_uq: np.ndarray, df_uq: np.ndarray, n_res: int
                    ) -> np.ndarray:
    """Query-similarity-weighted IDF over U_q, normalised to sum to one.

    Negative similarities are clipped to ZERO, per the spec. icra clips to
    1e-6, which keeps a floor of weight on a concept the query is actively
    dissimilar to; after normalisation that is nearly the same number, but it
    is not the same rule, and it changes what happens when every similarity in
    U_q is negative: here the weights are all zero and coverage is zero, which
    is what the spec prescribes.
    """
    a = np.clip(np.asarray(a_uq, dtype=np.float64), 0.0, None)
    idf = np.log(n_res / (1.0 + np.asarray(df_uq, dtype=np.float64)))
    w = a * np.clip(idf, 0.0, None)
    tot = float(w.sum())
    if tot <= 0.0:
        return np.zeros_like(w)          # coverage is zero, per the spec
    return w / tot


def apportion(budget: int, profile: Dict[str, int], order: Sequence[str]
              ) -> Dict[str, int]:
    """Largest-remainder apportionment of `budget` over the profile's types.

    b[t] = floor(budget * profile[t] / n), spare slots to the largest
    fractional remainders. Ties in the remainder are broken by position in
    `order`, so two runs on the same input always agree.
    """
    n = sum(profile.values())
    if n <= 0 or budget <= 0:
        return {t: 0 for t in profile}
    exact = {t: budget * profile[t] / n for t in profile}
    base = {t: int(math.floor(exact[t])) for t in profile}
    spare = budget - sum(base.values())
    if spare > 0:
        rem = sorted(profile,
                     key=lambda t: (-(exact[t] - base[t]),
                                    order.index(t) if t in order else 99))
        for t in rem[:spare]:
            base[t] += 1
    return base


def reduce_setcore(pool_types: Sequence[str], s_pool: np.ndarray,
                   P_pool: np.ndarray, w: np.ndarray, budget: int,
                   capacities: Dict[str, int], beta: float,
                   type_order: Sequence[str],
                   coverage: str = "binary") -> Tuple[List[int], int]:
    """Composition-guided greedy. Returns (selected pool positions, n_backfill).

    `pool_types`, `s_pool` and `P_pool` are in RETRIEVAL ORDER, so np.argmax
    over a retrieval-ordered candidate array already breaks ties toward the
    better-ranked resource -- which is the spec's tie rule, obtained without a
    second sort.
    """
    from icra.select import CoverageState

    budget = min(budget, len(pool_types))
    st_cov = CoverageState(P_pool, w, coverage)      # spec default: binary
    chosen: List[int] = []
    taken = np.zeros(len(pool_types), dtype=bool)
    remaining = dict(capacities)
    n_backfill = 0

    def step(feasible: np.ndarray) -> Optional[int]:
        if feasible.size == 0:
            return None
        g = st_cov.gain(feasible) + beta * s_pool[feasible]
        j = int(np.argmax(g))            # first max == best retrieval rank
        i = int(feasible[j])
        chosen.append(i)
        taken[i] = True
        st_cov.add(i)
        return i

    # phase 1: respect the expanded capacities
    while len(chosen) < budget:
        feasible = np.array(
            [i for i in range(len(pool_types))
             if not taken[i] and remaining.get(pool_types[i], 0) > 0],
            dtype=int)
        i = step(feasible)
        if i is None:
            break
        remaining[pool_types[i]] -= 1

    # phase 2: scarcity must not shorten the pool -- fill from any category
    while len(chosen) < budget:
        feasible = np.array([i for i in range(len(pool_types)) if not taken[i]],
                            dtype=int)
        if step(feasible) is None:
            break
        n_backfill += 1

    return chosen, n_backfill


# ----------------------------------------------------------------- the driver

def build_config(ns):
    from icra.config import Config
    over = {}
    for k in ("encoder_name", "encoder_backend", "con_source", "con_group",
              "con_learned_size", "con_learned_tau", "res_concept_hidden",
              "res_concept_lr", "res_concept_wd", "res_concept_epochs",
              "kappa", "gamma", "uq_min", "K", "seed", "folds", "role_hidden",
              "role_lr", "role_epochs", "role_weight_decay", "role_loss",
              "count_decode", "eta", "lam", "n_max", "w_mode"):
        v = getattr(ns, k, None)
        if v is not None:
            over[k] = v
    over["resources_path"] = ns.resources
    over["queries_path"] = ns.queries
    if ns.train_queries:
        over["train_queries_path"] = ns.train_queries
    if ns.candidates:
        over["candidates_path"] = ns.candidates
    over["uq_mode"] = "match"
    over["coverage"] = "binary"
    over["role_train"] = "cv"
    return Config(**over)


def predict_profiles(ix, cfg, profile_max: int, cap_drop: str = "figure",
                     quiet: bool = False):
    """Grouped-CV role head -> (c_q, f_q) per evaluation query, decode-capped."""
    from icra.heads import RoleHead, group_kfold_indices

    nq = len(ix.queries)
    k = min(cfg.folds, max(2, nq))
    folds = group_kfold_indices([q.group_id for q in ix.queries], k, cfg.seed)
    pred_c = np.zeros(nq, dtype=int)
    pred_f = np.zeros(nq, dtype=int)
    multi = cfg.role_loss in ("best_match", "single_first", "single_random",
                              "average")
    for tr, te in folds:
        tq = [ix.queries[i] for i in tr]
        ct = [ix.targets_for(q)[0] if multi else [ix.mean_targets_for(q)[0]]
              for q in tq]
        ft = [ix.targets_for(q)[1] if multi else [ix.mean_targets_for(q)[1]]
              for q in tq]
        head = RoleHead(ix.Q.shape[1], cfg.n_max, cfg.role_hidden, cfg.role_lr,
                        cfg.role_weight_decay, cfg.role_epochs, cfg.lam,
                        cfg.role_loss, cfg.seed).fit(ix.Q[tr], ct, ft)
        c, f, _ = head.predict(ix.Q[te], cfg.count_decode, cfg.eta)
        pred_c[te], pred_f[te] = c, f

    out = []
    n_capped = n_clipped = n_empty = 0
    for i in range(nq):
        c_raw, f = int(pred_c[i]), int(pred_f[i])
        c = max(0, min(c_raw, profile_max))
        if c != c_raw:
            n_clipped += 1
        if c + f > profile_max:
            # Which slot gives way is a real choice, not a formality: with
            # cap_drop="figure" (SetCoRe's own rule) a predicted (3, 1) becomes
            # (3, 0) and the figure is lost on every such query, which
            # suppresses figure slots systematically. cap_drop="explanation"
            # keeps the figure and spends one fewer explanation slot instead.
            if cap_drop == "explanation":
                c = max(0, profile_max - f)
            else:
                f = max(0, profile_max - c)
            n_capped += 1
        if c + f == 0:                   # non-empty profiles only
            c, f = 1, 0
            n_empty += 1
        out.append((c, f))
    if not quiet:
        print(f"role head: {n_clipped} counts clipped to {profile_max}, "
              f"{n_capped} profiles capped at total {profile_max} "
              f"(dropping {cap_drop}), {n_empty} empty forced to (1, 0)")
        if n_clipped:
            print(f"  [note] the cap is applied by CLIPPING the count head's "
                  f"argmax, so a query whose\n         argmax was above "
                  f"{profile_max} gets exactly {profile_max} rather than the "
                  f"best count\n         within 0..{profile_max}. Pass "
                  f"--n-max {profile_max} to restrict the head itself "
                  f"instead,\n         at the cost of also clamping the "
                  f"training targets.")
    return out


def add_icra_to_path(path: Optional[str]) -> None:
    """Make `import icra` work, without breaking the standard library.

    Three things go wrong with the obvious sys.path.insert(0, path):

    1. The icra checkout contains select.py, concepts.py, data.py -- names
       that collide with the standard library. At sys.path[0] they WIN, so a
       plain `import select` deep inside torch or httpx resolves to icra's
       select.py and dies with "attempted relative import with no known parent
       package". Appending instead lets the standard library win and still
       finds icra, since nothing else on the path is called that.

    2. If `import icra` already works -- because this script sits next to an
       icra/ directory, which it usually does -- then adding anything at all
       is pure risk for no benefit. So that case is checked first and the path
       is left alone.

    3. A FLAT checkout (~/icra holding __init__.py and the modules directly)
       is the package itself, not a directory containing it, so `import icra`
       needs its PARENT on the path. Adding the package directory would never
       have worked for the import and only ever contributes the collisions in
       (1).
    """
    try:
        import icra  # noqa: F401
        if path:
            print(f"  [i] --icra given but `import icra` already works; "
                  f"leaving sys.path alone")
        return
    except ImportError:
        pass
    if not path:
        raise SystemExit(
            "`import icra` fails and no --icra was given. Point --icra at the "
            "directory that\nCONTAINS the icra package (not at the package "
            "itself).")

    p = os.path.abspath(os.path.expanduser(path))
    if os.path.exists(os.path.join(p, "__init__.py")):
        p = os.path.dirname(p)          # flat checkout: use its parent
    sys.path.append(p)                  # append, never insert(0)
    try:
        import icra  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"added {p} to sys.path but `import icra` still fails: {exc}\n"
            f"  --icra must name the directory CONTAINING the icra package.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resources", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--train-queries", default=None)
    ap.add_argument("--candidates", default=None,
                    help="external first-stage pool; otherwise icra retrieves")
    ap.add_argument("--out", default=None,
                    help="output file for a single (reducer, m). Use --plan "
                         "with --out-dir for several.")
    ap.add_argument("--plan", default=None,
                    help="several candidate sets from ONE process, as "
                         "name:reducer:m[:capacities] separated by commas, "
                         "e.g. "
                         "'topn_nq:topn:1,topB:topn:2,guided:setcore:2'. "
                         "Encoding the corpus, tagging Con(r) and "
                         "cross-validating the role head are independent of "
                         "the reducer and of m, and they dominate the "
                         "runtime; one invocation per candidate set repeats "
                         "all of it. Files land in --out-dir as <name>.json.")
    ap.add_argument("--out-dir", default=None, help="where --plan writes")
    ap.add_argument("--overwrite", action="store_true",
                    help="--plan: rebuild entries whose file already exists")
    ap.add_argument("--pool-out", default=None,
                    help="also write the FIRST-STAGE pool C_q per query, "
                         "{query_id: [ids in rank order]}. scx_ceiling.py and "
                         "scx_eval.py read it with --pool to report the "
                         "retrieval ceiling, which bounds the whole pipeline. "
                         "It does not depend on m, so write it once.")
    ap.add_argument("--m", type=float, default=None,
                    help="shortlist multiplier, may be fractional (>= 1)")
    ap.add_argument("--reducer", default="setcore", choices=["setcore", "topn"])
    ap.add_argument("--capacities", default="proportional",
                    choices=["proportional", "none"],
                    help="proportional (the spec): b_q[t] scales pi_q to B_q "
                         "by largest remainder. none: one pooled capacity of "
                         "B_q over all types, so the greedy is unconstrained "
                         "by composition. The pair isolates what the type "
                         "capacities cost in reachability -- if 'none' has a "
                         "higher ceiling at the same |D_q|, the apportionment "
                         "is spending budget on a type that is not gold.")
    ap.add_argument("--sizes", default=None,
                    help="{query_id: n_hat_q}, e.g. the lightweight model's "
                         "budget_final.json. Overrides the role head's TOTAL "
                         "while keeping its figure decision. Pass the same "
                         "file here, to the Top-n_hat_q row and to the Table 1 "
                         "fixed-oracle column, so one n_hat_q sizes the whole "
                         "paper.")
    ap.add_argument("--profile-max", type=int, default=3,
                    help="decode restriction on c_q + f_q")
    ap.add_argument("--cap-drop", default="figure",
                    choices=["figure", "explanation"],
                    help="which slot gives way when c_q + f_q exceeds "
                         "--profile-max. 'figure' is SetCoRe's rule and "
                         "suppresses figures on every capped query.")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--coverage", default="binary",
                    choices=["binary", "graded_max", "noisy_or"],
                    help="binary is the spec's 1[v in Con(S)] and the default. "
                         "It SATURATES -- a second resource on a covered "
                         "concept gains exactly zero -- so it diversifies "
                         "harder than graded_max, which gives diminishing "
                         "rather than zero returns. If reachability is worse "
                         "than the first implementation's, this is the other "
                         "candidate cause besides the type capacities.")
    ap.add_argument("--icra", default=None,
                    help="directory CONTAINING the icra package. Ignored when "
                         "`import icra` already works, and appended rather "
                         "than prepended so icra's select.py cannot shadow "
                         "the standard library's.")
    # icra config passthrough; None means "leave the Config default"
    # uq-mode is here because paper 1's command states it explicitly. Config
    # happens to default to "match" today, so omitting it gives the same
    # reduction -- but a default is not a guarantee, and a silently different
    # concept layer would change every number in the table without changing
    # any command. role-train is NOT in this list: predict_profiles forces
    # "cv" on line ~218 regardless, so accepting the flag would imply a
    # choice that does not exist.
    for name, typ in [("encoder-name", str), ("encoder-backend", str),
                      ("con-source", str), ("con-group", str),
                      ("con-learned-size", str), ("w-mode", str),
                      ("role-loss", str), ("count-decode", str),
                      ("uq-mode", str)]:
        ap.add_argument(f"--{name}", dest=name.replace("-", "_"), type=typ,
                        default=None)
    for name in ("con-learned-tau", "res-concept-lr", "res-concept-wd",
                 "gamma", "role-lr", "eta", "lam"):
        ap.add_argument(f"--{name}", dest=name.replace("-", "_"), type=float,
                        default=None)
    for name in ("res-concept-hidden", "res-concept-epochs", "kappa",
                 "uq-min", "K", "seed", "folds", "role-hidden", "role-epochs",
                 "n-max"):
        ap.add_argument(f"--{name}", dest=name.replace("-", "_"), type=int,
                        default=None)
    ap.add_argument("--role-weight-decay", dest="role_weight_decay",
                    type=float, default=None)
    ns = ap.parse_args()
    expand_tildes(ns)
    # The plan is parsed BEFORE icra loads and the index is built: a typo in
    # it should cost a second, not the ten minutes of encoding and training
    # that come first.
    plan: List[Tuple[str, str, float, str]] = []
    if ns.plan:
        if ns.m is not None or ns.out:
            raise SystemExit(
                "--plan carries its own reducer and m per entry, so --m and "
                "--out would be\nignored. Remove them, or drop --plan.")
        if not ns.out_dir:
            raise SystemExit("--plan needs --out-dir")
        for part in ns.plan.split(","):
            bits = [b.strip() for b in part.split(":") if b.strip() != ""]
            if len(bits) < 3:
                raise SystemExit(
                    f"--plan entry {part!r} must be name:reducer:m "
                    f"(optionally :capacities)")
            name, red, mtxt = bits[0], bits[1], bits[2]
            if red not in ("setcore", "topn"):
                raise SystemExit(f"--plan: unknown reducer {red!r} in {part!r}")
            try:
                mm = float(mtxt)
            except ValueError:
                raise SystemExit(f"--plan: m must be a number in {part!r}")
            if mm < 1:
                raise SystemExit(f"--plan: m must be >= 1, got {mm}")
            caps = bits[3] if len(bits) > 3 else ns.capacities
            if caps not in ("proportional", "none"):
                raise SystemExit(f"--plan: unknown capacities {caps!r}")
            plan.append((name, red, mm, caps))
        names = [p[0] for p in plan]
        if len(set(names)) != len(names):
            raise SystemExit(f"--plan: duplicate names {names}; each entry "
                             f"writes <name>.json and they would collide")
    else:
        if ns.m is None or not ns.out:
            raise SystemExit("--m and --out are required without --plan")
        if ns.m < 1:
            raise SystemExit("--m must be >= 1")
        plan = [(os.path.splitext(os.path.basename(ns.out))[0],
                 ns.reducer, ns.m, ns.capacities)]
    add_icra_to_path(ns.icra)

    import icra.concepts as C

    from icra.pipeline import Index

    print(f"icra      : {os.path.dirname(os.path.abspath(C.__file__))}")

    cfg = build_config(ns)
    _t0 = time.perf_counter()
    ix = Index(cfg)
    t_index = time.perf_counter() - _t0
    n_res = len(ix.resources)
    TYPE_ORDER = ("explanation", "figure")

    if not ns.plan:
        print(f"reducer   : {ns.reducer}"
              + (f"   capacities: {ns.capacities}" if ns.reducer == "setcore"
                 else ""))
        print(f"m         : {ns.m}")
    print(f"K         : {cfg.K}   profile cap: {ns.profile_max}   "
          f"beta: {ns.beta}")
    print(f"U_q       : match, gamma={cfg.gamma} "
          f"({'relative' if cfg.gamma_relative else 'absolute'}), "
          f"kappa={cfg.kappa}, uq_min={cfg.uq_min}")
    print(f"Con(r)    : {cfg.con_source}")
    if cfg.con_source != "embed":
        print(f"  [!] The method text says Con(r) is 'inferred from resource "
              f"text, using descriptor\n      similarity in the frozen "
              f"embedding space'. That is --con-source embed.\n      "
              f"'{cfg.con_source}' is a different tagger: a multi-label "
              f"classifier over resource\n      embeddings supervised by the "
              f"corpus `concepts` annotation, whose predictions\n      also "
              f"define the document frequencies in the concept weights. Either "
              f"switch to\n      embed or change that sentence -- the two give "
              f"different D_q and different numbers.")
    if cfg.uq_min > 0:
        print(f"  [note] uq_min={cfg.uq_min}, so U_q is never empty and the "
              f"text's 'if U_q is empty,\n         coverage is zero' case "
              f"cannot arise. Pass --uq-min 0 to allow it.")
    print(f"coverage  : {ns.coverage}"
          + ("  (spec Eq. objective)" if ns.coverage == "binary"
             else "  [!] the spec's objective is the BINARY indicator"))

    profiles = predict_profiles(ix, cfg, ns.profile_max, ns.cap_drop)

    # --sizes replaces the role head's TOTAL with a given n_hat_q, keeping its
    # figure decision where the budget can afford one. Table 1's fixed-oracle
    # ceiling, the Top-n_hat_q row and every hybrid row have to be sized from
    # the SAME n_hat_q, or the ceiling does not bound the rows beneath it and
    # "fixed predicted size" means two different numbers in one paper. The
    # role head still supplies the composition; only the size comes from the
    # file.
    if ns.sizes:
        given = {str(k): int(v) for k, v in load_json(ns.sizes).items()}
        n_sub = n_miss = n_fig = 0
        for qi, q in enumerate(ix.queries):
            n = given.get(str(q.query_id))
            if n is None:
                n_miss += 1
                continue
            n = max(1, n)
            c, f = profiles[qi]
            nf = 1 if (f and n >= 2) else 0
            if (c, f) != (n - nf, nf):
                n_sub += 1
            if f and not nf:
                n_fig += 1
            profiles[qi] = (n - nf, nf)
        print(f"--sizes {os.path.basename(ns.sizes)}: n_q taken from the file "
              f"for {len(ix.queries) - n_miss}/{len(ix.queries)} queries "
              f"({n_sub} changed)")
        if n_fig:
            print(f"  {n_fig} queries wanted a figure but have n_hat_q = 1, "
                  f"so the figure slot was dropped")
        if n_miss:
            print(f"  [!] {n_miss} queries are absent from the file and keep "
                  f"the role head's own n_q.\n      Those rows are sized "
                  f"differently from the rest of the table.")

    # ---- the plan: one or more (name, reducer, m) in ONE process.
    #
    # Everything above this line -- encoding the corpus, tagging Con(r),
    # cross-validating the role head -- is independent of the reducer and the
    # multiplier, and it is the overwhelming majority of the runtime. Running
    # scx_reduce.py four times to get four candidate sets repeats all of it
    # four times for nothing.
    # ---- per-query work that no plan entry can change, done once
    # Timed, because the cost figure's "end-to-end" is retrieval + reduction +
    # selection + repairs, and a hybrid row that reports only its LLM call
    # understates its own latency in its own favour. The index build above is
    # OFFLINE (encoding the corpus, tagging Con(r), fitting the role head) and
    # is reported separately rather than amortised into the per-query number.
    t_ret = time.perf_counter()
    POOLS: List[Tuple[List[int], object, List[str], List[str]]] = []
    for qi in range(len(ix.queries)):
        pool, s_raw = ix.retrieve(qi)
        POOLS.append((pool, s_raw,
                      [ix.resources[i].resource_id for i in pool],
                      [ix.resources[i].type for i in pool]))
    t_retrieve = time.perf_counter() - t_ret
    nq = max(1, len(ix.queries))
    print(f"\nretrieval : {t_retrieve:.2f} s total, "
          f"{t_retrieve/nq*1000:.1f} ms/query   (shared by every plan entry)")
    print(f"offline   : {t_index:.2f} s to build the index "
          f"(encode, tag Con(r), fit the role head) -- state this separately, "
          f"not per query")

    pool_out: Dict[str, List[str]] = {}
    if ns.pool_out is not None:
        for qi, q in enumerate(ix.queries):
            pool_out[str(q.query_id)] = list(POOLS[qi][2])

    # The concept signal depends on the query and the POOL, not on m or the
    # reducer, so it is computed at most once per query and only if some plan
    # entry actually needs it. A plan of pure top-n reductions never touches
    # it -- which matters, because top-n is a prefix of the ranking and has no
    # business paying for concept inference at all.
    SIG: Dict[int, Tuple] = {}
    needs_concepts = any(r == "setcore" for _, r, _, _ in plan)

    def concept_signal(qi: int):
        if qi in SIG:
            return SIG[qi]
        pool, s_raw, _, _ = POOLS[qi]
        P_pool = ix.P_full[pool]
        sig = C.uq_match(ix.Q[qi], ix.V, P_pool, ix.vocab, cfg,
                         df=ix.df, n_res=n_res)
        a_all = ix.V @ ix.Q[qi]
        vi = [ix.vocab.ids.index(u) for u in sig.units]
        w = concept_weights(a_all[vi], ix.df[vi], n_res) if vi \
            else np.zeros(0)
        SIG[qi] = (sig, w, normalise_relevance(s_raw), list(sig.units))
        return SIG[qi]

    if len(plan) > 1:
        print(f"\nplan      : {len(plan)} candidate sets in ONE process "
              + ", ".join(f"{n}({r},m={m:g})" for n, r, m, _ in plan))
        print(f"            index, Con(r) and the role head are shared; "
              f"only the reduction repeats")
        if not needs_concepts:
            print(f"            no setcore entry, so the concept signal is "
                  f"never computed")

    for p_name, p_reducer, p_m, p_caps in plan:
        out_path = (os.path.join(ns.out_dir, f"{p_name}.json")
                    if ns.plan else ns.out)
        if ns.plan and os.path.exists(out_path) and not ns.overwrite:
            print(f"\n--- {p_name}: reusing {out_path}")
            continue
        print(f"\n--- {p_name}   reducer={p_reducer}  m={p_m:g}"
              + (f"  capacities={p_caps}" if p_reducer == "setcore" else ""))
        reduce_one(ix, cfg, C, POOLS, profiles, concept_signal, n_res,
                   TYPE_ORDER, p_reducer, p_m, p_caps, ns, out_path,
                   t_retrieve, t_index)

    if ns.pool_out is not None:
        write_json(pool_out, ns.pool_out, "first-stage pool C_q")
        print(f"  |C_q| mean "
              f"{st.mean(len(v) for v in pool_out.values()):.2f}  "
              f"min {min(len(v) for v in pool_out.values())}  "
              f"max {max(len(v) for v in pool_out.values())}")
    print("\nCandidates are emitted in retrieval order, so both reducers "
          "present identically.\nNext: scx_ceiling.py, then scx_llm_select.py.")


def reduce_one(ix, cfg, C, POOLS, profiles, concept_signal, n_res, TYPE_ORDER,
               reducer: str, m: float, capacities: str, ns, out_path: str,
               t_retrieve: float = 0.0, t_index: float = 0.0) -> None:
    """One (reducer, m) candidate set, from already-computed shared state."""
    t_red0 = time.perf_counter()
    rows: List[Dict] = []
    stats: Counter = Counter()
    bq_all, sl_all, uq_all, backfill_all = [], [], [], []
    zero_w = 0

    for qi, q in enumerate(ix.queries):
        pool, s_raw, pool_ids, pool_types = POOLS[qi]
        c_q, f_q = profiles[qi]
        n_q = c_q + f_q
        B_q = min(cfg.K, len(pool), int(math.ceil(m * n_q)))
        bq_all.append(B_q)

        if reducer == "topn":
            sel = list(range(min(B_q, len(pool))))
            nb = 0
            uq_ids: List[str] = []
        else:
            # Shared across plan entries: U_q and the concept weights depend
            # on the query and the pool, not on m or the reducer.
            sig, w, s_norm, uq_ids = concept_signal(qi)
            if w.size == 0 or float(w.sum()) <= 0:
                zero_w += 1
            if capacities == "none":
                # One pooled slot count. Equivalent to the first
                # implementation's --sizes-composition free, and the control
                # for whether composition guidance helps REACHABILITY.
                caps = {"_any": B_q}
                sel, nb = reduce_setcore(["_any"] * len(pool_types), s_norm,
                                         sig.P, w, B_q, caps, ns.beta,
                                         ("_any",), ns.coverage)
            else:
                caps = apportion(B_q, {"explanation": c_q, "figure": f_q},
                                 TYPE_ORDER)
                sel, nb = reduce_setcore(pool_types, s_norm, sig.P, w, B_q,
                                         caps, ns.beta, TYPE_ORDER,
                                         ns.coverage)
        uq_all.append(len(uq_ids))
        backfill_all.append(nb)

        # Emit in ORIGINAL RETRIEVAL ORDER, whichever reducer produced it.
        sel_sorted = sorted(set(sel))
        ids = [pool_ids[i] for i in sel_sorted]
        sl_all.append(len(ids))
        if B_q >= len(pool) and len(ids) != len(pool):
            raise AssertionError(
                f"{q.query_id}: B_q={B_q} >= |C_q|={len(pool)} but the reduced "
                f"pool has {len(ids)} resources; at B_q = K it must be exactly "
                f"C_q.")
        if len(ids) < B_q:
            stats["short_of_budget"] += 1

        rows.append({
            "query_id": str(q.query_id),
            "shortlist": ids,
            "n_shortlist": len(ids),
            "target_n": n_q,
            "profile": {"explanation": c_q, "figure": f_q},
            "B_q": B_q,
            "m": m,
            "reducer": reducer,
            "ranks": [i + 1 for i in sel_sorted],
            "types": [pool_types[i] for i in sel_sorted],
            "capacities": (({"_any": B_q} if capacities == "none"
                            else apportion(B_q,
                                           {"explanation": c_q,
                                            "figure": f_q}, TYPE_ORDER))
                           if reducer == "setcore" else None),
            "capacity_mode": (capacities if reducer == "setcore"
                              else None),
            "n_backfilled": nb,
            "n_pool": len(pool),
            "U_q": uq_ids,
        })

    # C_q does not depend on the reducer or on m, so main() writes it once
    # after the plan finishes rather than each entry rewriting it.
    t_reduce = time.perf_counter() - t_red0
    nq = max(1, len(rows))
    eff = {"reducer": reducer, "m": m, "capacities": capacities,
           "n_queries": nq,
           "index_build_s": round(t_index, 3),
           "retrieval_s_total": round(t_retrieve, 3),
           "retrieval_s_per_query": round(t_retrieve / nq, 6),
           "reduction_s_total": round(t_reduce, 3),
           "reduction_s_per_query": round(t_reduce / nq, 6),
           # What the cost figure should add to the LLM call. The index build
           # is deliberately NOT in it: it is paid once, offline, for every
           # query at once, and amortising it per query would either flatter
           # or punish the method depending on the benchmark's size.
           "online_s_per_query": round((t_retrieve + t_reduce) / nq, 6)}
    write_json(eff, os.path.splitext(out_path)[0] + ".efficiency.json",
               "efficiency")
    print(f"  retrieval+reduction: {eff['online_s_per_query']*1000:.2f} "
          f"ms/query online  (index build {t_index:.1f} s, offline)")
    write_json(rows, out_path, "shortlists")
    pdist = Counter((r["profile"]["explanation"], r["profile"]["figure"])
                    for r in rows)
    print(f"\n{len(rows)} queries")
    print(f"n_q       : mean {st.mean(r['target_n'] for r in rows):.4f}   "
          + "  ".join(f"({c},{f}):{n}" for (c, f), n in sorted(pdist.items())))
    print(f"B_q       : mean {st.mean(bq_all):.4f}   "
          f"min {min(bq_all)}   max {max(bq_all)}")
    print(f"|D_q|     : mean {st.mean(sl_all):.4f}   "
          f"min {min(sl_all)}   max {max(sl_all)}")
    print(f"realised m: {st.mean(a / max(1, b) for a, b in zip(sl_all, [r['target_n'] for r in rows])):.4f}"
          f"   (requested {m})")
    if reducer == "setcore":
        print(f"|U_q|     : mean {st.mean(uq_all):.2f}")
        if capacities == "proportional":
            figs = [(r["capacities"] or {}).get("figure", 0) for r in rows]
            got = [sum(1 for t in r["types"] if t == "figure") for r in rows]
            print(f"figure cap: {st.mean(figs):.2f} of {st.mean(bq_all):.2f} "
                  f"slots reserved "
                  f"({st.mean(figs) / max(1e-9, st.mean(bq_all)):.1%} of D_q)"
                  f"   figures actually in D_q: {st.mean(got):.2f}")
            print(f"  Those reserved slots cannot hold an explanation. If gold "
                  f"sets contain figures\n  less often than that share, the "
                  f"apportionment is spending reachability on them;\n  "
                  f"--capacities none is the control that measures it.")
        nbq = sum(1 for v in backfill_all if v)
        print(f"backfill  : {nbq}/{len(rows)} queries needed cross-category "
              f"fill ({sum(backfill_all)} slots)")
        if zero_w:
            print(f"  [!] {zero_w} queries had all-zero concept weights, so "
                  f"their coverage term is 0\n      and the reduction was "
                  f"driven by relevance alone.")
    if stats["short_of_budget"]:
        print(f"  [!] {stats['short_of_budget']} queries have |D_q| < B_q "
              f"because the POOL itself is smaller\n      than the budget; "
              f"that is the only way scarcity can shorten the pool.")
    dist = Counter(sl_all)
    print(f"\n{'|D_q|':>6s} {'queries':>8s}")
    for k in sorted(dist):
        print(f"{k:>6d} {dist[k]:>8d}")


if __name__ == "__main__":
    main()
