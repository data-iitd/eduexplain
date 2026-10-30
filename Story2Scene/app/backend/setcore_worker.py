"""JSON-lines worker: setCore stages 2-4 (lightweight model -> n_hat_q -> coverage-greedy reduce) for ONE new query.
Runs in set2story's venv (numpy + sentence-transformers, CPU). Reuses setCore's icra / scx_reduce code unchanged.
  in : {"id","query","pool":[{"resource_id","score"},...]}   out: {"id","ok",n_hat,B,shortlist,profile,...}
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json, math, sys, time
sys.path.insert(0, "/mnt/storage/aryan/setCore")
sys.stdout, _real = sys.stderr, sys.stdout          # library prints must not corrupt the protocol


def send(obj):
    _real.write(json.dumps(obj) + "\n"); _real.flush()


def main():
    try:
        import numpy as np
        from scx_lw_common import make_cfg, tiny_queries_file, load_role, ROLE
        if not os.path.exists(ROLE):
            raise RuntimeError(f"role head missing: {ROLE} (run setCore/scx_lw_train.py)")
        import icra.concepts as C
        import icra.select as S
        from icra.config import Config
        from icra.pipeline import Index
        from scx_reduce import concept_weights, normalise_relevance, reduce_setcore
        cfg = make_cfg(tiny_queries_file())                       # lw config (graded_max coverage = icra default)
        cfg_red = Config(**{**cfg.__dict__, "coverage": "binary"})  # scx_reduce.build_config: binary coverage
        ix = Index(cfg)                                           # corpus emb + Con(r) from the persisted caches
        n_res = len(ix.resources)
        head = load_role(ix.Q.shape[1], cfg)
        # warm up the query path
        ix.enc.encode(["warm up"])
    except Exception as exc:
        import traceback; traceback.print_exc()
        send({"ok": False, "error": f"{type(exc).__name__}: {exc}"}); return
    send({"ok": True, "ready": True, "n_resources": n_res})

    for line in sys.stdin:
        if not line.strip():
            continue
        req = {}
        try:
            req = json.loads(line)
            t0 = time.time()
            qv = ix.enc.encode([req["query"]])[0]
            pool_ids = [p["resource_id"] for p in req.get("pool") or []]
            if pool_ids:
                idx = np.array([ix.rid2i[r] for r in pool_ids if r in ix.rid2i], dtype=int)
                s_all = ix.R[idx] @ qv
                order = np.argsort(-s_all, kind="stable")
                pool, s = idx[order], s_all[order]
            else:
                pool, s = ix.retrieve_for(qv)
            ptypes = ix.types[pool]
            P_pool = ix.P_full[pool]
            # ---- stage 2: lightweight model, regime free (identical to icra.pipeline.run's per-query block)
            c, f, _ = head.predict(qv[None], cfg.count_decode, cfg.eta)
            cq, fq = max(0, min(int(c[0]), cfg.n_max)), int(f[0])
            if cq + fq > cfg.n_max:
                fq = max(0, cfg.n_max - cq)
            profile = {"explanation": cq, "figure": fq}
            sig = C.uq_match(qv, ix.V, P_pool, ix.vocab, cfg, df=ix.df, n_res=n_res)
            sel = S.select(ptypes, s, sig, profile, cfg)
            # ---- stage 3: n_hat_q = size RETURNED (scx_budget --variant final, min_n=1)
            n_hat = max(1, len(sel.indices))
            # ---- stage 4: scx_reduce --reducer setcore --m 2 --capacities none
            B = min(cfg.K, len(pool), int(math.ceil(2 * n_hat)))
            sigr = C.uq_match(qv, ix.V, P_pool, ix.vocab, cfg_red, df=ix.df, n_res=n_res)
            vi = [ix.vocab.ids.index(u) for u in sigr.units]
            w = concept_weights((ix.V @ qv)[vi], ix.df[vi], n_res) if vi else np.zeros(0)
            chosen, nb = reduce_setcore(["_any"] * len(pool), normalise_relevance(s), sigr.P, w, B,
                                        {"_any": B}, 1.0, ("_any",), "binary")
            sel_sorted = sorted(set(chosen))                      # emitted in retrieval order
            rids = [ix.resources[pool[i]].resource_id for i in sel_sorted]
            send({"id": req.get("id"), "ok": True, "n_hat": n_hat, "B": B, "shortlist": rids,
                  "ranks": [i + 1 for i in sel_sorted], "profile": profile, "U_q": list(sigr.units),
                  "lw_selected": [ix.resources[pool[i]].resource_id for i in sel.indices],
                  "n_pool": int(len(pool)), "seconds": round(time.time() - t0, 3)})
        except Exception as exc:
            import traceback; traceback.print_exc()
            send({"id": req.get("id"), "ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
