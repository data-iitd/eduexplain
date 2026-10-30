"""
Demo-paper evaluation E1 (component swap): 5 benchmark queries x 7 configurations -> 35 Manim videos.

    tmux new -d -s vx-exp "cd /mnt/storage/aryan/VideoExplain/experiments/demo_eval && \
        /mnt/storage/aryan/VideoExplain/venv/bin/python -u run_demo_eval.py all 2>&1 | tee -a logs/run.log"

Phases (each resumable; finished outputs are skipped): stage1 -> stage2 -> stage3 ("all" runs the three in order).

  Stage 1  gold      benchmark gold set gs0 (the set behind the reference storyboard)
           lw        SetCoRe-LW   (run_setcore.sh, out-of-fold predictions, seed 0)
           e2e       SetCoRe-E2E  (run_setcore.sh, out-of-fold predictions, seed 0)
           x         SetCoRe-LLM    (the web app's Stage 1: SetCoRe shortlist + Llama-3.1-70B on :8003)
           optiset   OptiSet      (optiset_vllm.py, Llama-3.1-70B on :8003)
  Stage 2  s2s       Set2Story (the web app's Stage 2) for every Stage 1 condition
           ref       benchmark reference storyboard of (query, gs0), as is
           ff        FreeForm (freeform_s2s.py, gemma-3-12b-it) on the gold set
  Stage 3  Manim VideoModule (pipeline.py via the backend's stage3.run): 3 rounds, cap pacing, budget T per query.

Configurations: gold+s2s (A0/B1), lw+s2s, e2e+s2s, x+s2s, optiset+s2s, gold+ref (B0), gold+ff (B2).
Video folders: videos/<stage1>__<stage2>__<qid>/. Experiment jobs never appear on the website.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent
BACKEND = Path("/mnt/storage/aryan/VideoExplain/backend")
S2S_PY = "/mnt/storage/aryan/set2story/venv/bin/python"
BENCH_SB = Path("/mnt/storage/aryan/set2story/BenchMark/Storyboard_BenchMark.json")
BENCH_Q = Path("/mnt/storage/aryan/setCore/gold_sets_final_validated_new_merged.json")
QIDS = json.loads((EXP / "queries.json").read_text())
STAGE1 = ["gold", "lw", "e2e", "x", "optiset"]
CONFIGS = [(c, "s2s") for c in STAGE1] + [("gold", "ref"), ("gold", "ff")]

# same GPU-sharing environment as start_all.sh (all four vLLM servers on the one A100). Stage 3 needs only :8000/:8001;
# gemma/Llama may be stopped then to free RAM, and a peer that is down must not be listed (a failed sleep disables
# GPU swapping for the whole run).
G, V, GE, LL = (f"http://127.0.0.1:{p}/v1" for p in (8000, 8001, 8002, 8003))


def _up(url):
    import urllib.request
    try:
        urllib.request.urlopen(url + "/models", timeout=5)
        return True
    except Exception:
        return False


os.environ.update({
    "VLLM_GPU_PEERS": ",".join(u for u in (G, V, GE, LL) if u in (G, V) or _up(u)),
    "VLLM_SLEEP_LEVELS": f"{GE}=2,{LL}=2",
    "S2S_GEN_URL": GE, "S2S_GEN_MODEL": "gemma-3-12b-it", "S2S_SLEEP_URLS": ",".join([G, V, LL]),
    "VX_STAGE1_LLM_URL": LL, "VX_STAGE1_LLM_MODEL": "Llama-3.1-70B-Instruct",
})
sys.path.insert(0, str(BACKEND))
sys.path.append("/mnt/storage/aryan/VideoModule")


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jsave(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def bench():
    sb = {it["query_id"]: it["storyboard"] for it in jload(BENCH_SB) if it["gold_set_index"] == 0}
    qs = {q["query_id"]: q for q in jload(BENCH_Q)}
    return {q: {"query": qs[q]["query"], "grade": qs[q].get("grade"), "ref": sb[q],
                "T": int(sb[q]["target_duration_sec"]), "gold_ids": list(sb[q]["gold_set_resources"])} for q in QIDS}


def setcore_preds(run_dir):
    """query_id -> predicted resource ids, from model.pipeline's predictions.jsonl."""
    out = {}
    for line in (run_dir / "predictions.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            ids = r.get("selected") or r.get("predicted") or r.get("pred") or r.get("predicted_resource_ids") or r.get("resource_ids")
            out[r["query_id"]] = [x if isinstance(x, str) else x.get("resource_id") for x in ids]
    return out


def optiset_preds():
    p = EXP / "stage1" / "optiset" / "optiset_summary.json"
    raw = jload(p)
    raw = raw.get("results", raw) if isinstance(raw, dict) else raw
    return {r["query_id"]: list(r["predicted_resource_ids"]) for r in raw}


_s1 = None


def stage1_service():
    global _s1
    if _s1 is None:
        from model_loader import VLLMClient
        from stage1 import Stage1
        _s1 = Stage1(llm=VLLMClient("stage1", LL, "Llama-3.1-70B-Instruct"), log=lambda m: log(f"[stage1] {m}"))
    return _s1


def phase_stage1(B):
    s1 = stage1_service()
    sources = {}
    for cond in STAGE1:
        for q in QIDS:
            out = EXP / "stage1" / cond / f"{q}.json"
            if out.exists():
                continue
            b = B[q]
            try:
                r1 = _stage1_one(s1, sources, B, cond, q, b)
            except Exception as exc:
                log(f"! stage1 {cond} {q} FAILED: {type(exc).__name__}: {exc}")
                continue
            r1["condition"] = cond
            jsave(out, r1)
            log(f"stage1 {cond} {q}: {r1['selection']['selected']}")


def _stage1_one(s1, sources, B, cond, q, b):
    if cond == "x":
        log(f"stage1 x {q}: SetCoRe-LLM (Llama-3.1-70B)")
        return s1.run(b["query"], query_id=q)
    if cond not in sources:
        sources[cond] = {"gold": lambda: {k: v["gold_ids"] for k, v in B.items()},
                         "lw": lambda: setcore_preds(EXP / "stage1" / "lw_s0"),
                         "e2e": lambda: setcore_preds(EXP / "stage1" / "e2e_s0"),
                         "optiset": optiset_preds}[cond]()
    stub = {"stage": 1, "mode": cond, "query": b["query"], "query_id": q, "retrieved": [], "pool_size": 0,
            "selection": {"shortlist": []}}
    return s1.apply_manual(stub, sources[cond][q])


def phase_stage2(B):
    s2 = None
    for st1, st2 in CONFIGS:
        for q in QIDS:
            name = f"{st1}__{st2}__{q}"
            out = EXP / "stage2" / f"{name}.json"
            if out.exists():
                continue
            b = B[q]
            if not (EXP / "stage1" / st1 / f"{q}.json").exists():
                log(f"! stage2 {name}: no Stage 1 result, skipped")
                continue
            try:
                s2 = _stage2_one(s2, st1, st2, q, b, name, out)
            except Exception as exc:
                log(f"! stage2 {name} FAILED: {type(exc).__name__}: {exc}")


def _stage2_one(s2, st1, st2, q, b, name, out):
    """Write one storyboard; returns the (lazily created) Set2Story service."""
    from model_loader import client_from_env
    from stage2 import Stage2
    r1 = jload(EXP / "stage1" / st1 / f"{q}.json")
    if st2 == "s2s":
        if s2 is None:
            s2 = Stage2(llm=client_from_env("generator", "Qwen3.8-27B", G), log=lambda m: log(f"[stage2] {m}"))
        log(f"stage2 {name}: Set2Story")
        r2 = s2.run(r1, total=b["T"])
        if r2["mode"] != "set2story":
            raise RuntimeError(f"Set2Story fell back to {r2['mode']}; not a valid run")
        sb = r2["storyboard"]
        meta = {k: r2.get(k) for k in ("mode", "narration_trimmed", "seconds")}
    elif st2 == "ref":
        sb, meta = {"storyboard": dict(b["ref"])}, {"mode": "reference"}
    else:
        gpath = EXP / "stage2" / "ff_inputs" / f"{q}.json"
        jsave(gpath, dict(r1["gold_set"], grade=b["grade"]))
        tmp = EXP / "stage2" / "ff_inputs" / f"{name}.out.json"
        log(f"stage2 {name}: FreeForm (gemma-3-12b-it)")
        t = time.time()
        subprocess.run([S2S_PY, str(EXP / "freeform_s2s.py"), str(gpath), str(b["T"]), str(tmp), name],
                       check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        res = jload(tmp)
        sb = {"storyboard": res["storyboard"]}
        meta = {"mode": "freeform", "seconds": round(time.time() - t, 1), **res["freeform"]}
    sb["storyboard"]["storyboard_id"] = name   # unique output folder per configuration
    sb["storyboard"]["query_id"] = q
    jsave(out, {"config": [st1, st2], "query_id": q, "T": b["T"], "meta": meta, **sb})
    segs = sb["storyboard"]["segments"]
    log(f"stage2 {name}: {len(segs)} segments, {sum(s['duration_sec'] for s in segs)} s (T={b['T']})")
    return s2


def phase_stage3(B):
    import stage3
    order = [(q, c) for c in CONFIGS for q in QIDS]   # one configuration across all queries, then the next
    for q, (st1, st2) in order:
        name = f"{st1}__{st2}__{q}"
        vdir = EXP / "videos" / name
        done = vdir / "stage3.json"
        if done.exists() and jload(done).get("ok"):
            continue
        if not (EXP / "stage2" / f"{name}.json").exists():
            log(f"! stage3 {name}: no storyboard, skipped")
            continue
        sb = jload(EXP / "stage2" / f"{name}.json")
        sb_path = vdir / "storyboard.json"
        jsave(sb_path, {"storyboard": sb["storyboard"]})
        log(f"stage3 {name}: Manim, budget {sb['T']} s")
        r3 = stage3.run(sb_path, vdir, "manim", 3, on_line=lambda line: None)
        jsave(done, r3)
        log(f"stage3 {name}: ok={r3['ok']} status={r3.get('status')} {r3['seconds']} s")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    B = bench()
    jsave(EXP / "benchmark_items.json", {q: {k: v for k, v in b.items() if k != "ref"} for q, b in B.items()})
    for ph, fn in (("stage1", phase_stage1), ("stage2", phase_stage2), ("stage3", phase_stage3)):
        if what in (ph, "all"):
            log(f"=== {ph}")
            fn(B)
    log("ALL DONE" if what == "all" else f"{what} done")


if __name__ == "__main__":
    main()
