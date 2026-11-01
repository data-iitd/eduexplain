"""
Alternative Stage 1 / Stage 2 methods for the web app (the demo paper's component switch). The defaults stay
SetCoRe-LLM (selection "auto") and Set2Story; these replace one stage each, and every result keeps the usual
stage1.json / stage2.json schema so the panels and Stage 3 need no changes.

  Stage 1  lw / e2e    SetCoRe-LW / SetCoRe-E2E (SetCoRe repository model/serve.py checkpoints in VideoExplain/models,
                       trained by models/train_setcore.sh), via setcore_runner.py on the CPU
           optiset     OptiSet (SetCoRe repository baseline: BGE top-20 -> sub-queries -> Llama-3.1-70B selection
                       and refinement), via optiset_runner.py on vLLM :8003
           gold        the benchmark gold set gs0 of a benchmark query (EduExplain, 321 queries)
  Stage 2  freeform    FreeForm baseline (set2story Baseline/baselines.py run_freeform: one unconstrained
                       gemma-3-12b-it call), via freeform_worker.py
           reference   the benchmark reference storyboard of (query, gs0), as is; needs the gold Stage 1
Both workers run in set2story's venv, on the CPU; their LLM calls go through model_loader's GPU swap.
"""
import json
import os
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
S2S_PY = os.getenv("S2S_PY", "/mnt/storage/aryan/set2story/venv/bin/python")
SETCORE = Path(os.getenv("SETCORE_DIR", os.getenv("SETCORE", "/mnt/storage/aryan/setCore")))
BENCH_Q = SETCORE / "gold_sets_final_validated_new_merged.json"
BENCH_SB = Path(os.getenv("BENCH_SB", "/mnt/storage/aryan/set2story/BenchMark/Storyboard_BenchMark.json"))
RESOURCES = SETCORE / "all_resources_unique_ids_clean1_merged.json"
BGE = Path(os.getenv("BGE_DIR", str(SETCORE / "models" / "bge-large-en-v1.5")))
STAGE1 = {"auto": "SetCoRe-LLM", "manual": "Manual", "lw": "SetCoRe-LW", "e2e": "SetCoRe-E2E", "optiset": "OptiSet",
          "gold": "Benchmark gold set"}
MODELS = Path(os.getenv("VX_MODELS", str(HERE.parent / "models")))
SETCORE_CKPT = {"lw": MODELS / "setcore_lw", "e2e": MODELS / "setcore_e2e"}
STAGE2 = {"set2story": "Set2Story", "freeform": "FreeForm", "reference": "Benchmark reference"}
_CPU_ENV = {"CUDA_VISIBLE_DEVICES": "", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "16"}


@lru_cache(maxsize=1)
def benchmark():
    """query_id -> {query, grade, T, gold_ids, ref}: every benchmark query with its gs0 reference storyboard."""
    sb = {it["query_id"]: it["storyboard"] for it in json.loads(BENCH_SB.read_text(encoding="utf-8"))
          if it["gold_set_index"] == 0}
    out = {}
    for q in json.loads(BENCH_Q.read_text(encoding="utf-8")):
        s = sb.get(q["query_id"])
        if s:
            out[q["query_id"]] = {"query": q["query"], "grade": q.get("grade"), "T": int(s["target_duration_sec"]),
                                  "gold_ids": list(s["gold_set_resources"]), "ref": s}
    return out


def _run(cmd, log, what):
    """Run a worker, passing its output lines to log; raises with the last lines on failure."""
    t, tail = time.time(), []
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env=dict(os.environ, **_CPU_ENV))
    for line in proc.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-15:]
        if line.startswith(("[", "LLM via", "Resource embeddings", "Encoding", "freeform", "!")):
            log("  " + line[:200])
    if proc.wait() != 0:
        raise RuntimeError(f"{what} failed (exit {proc.returncode}): " + " | ".join(tail[-4:]))
    return round(time.time() - t, 1)


def stage1(s1, method, query, query_id=None, log=print, n=None):
    """Stage 1 result (stage1.json schema) for method lw | e2e | optiset | gold."""
    t = time.time()
    if method == "gold":
        b = benchmark().get(query_id or "")
        if not b:
            raise ValueError(f"the gold set needs a benchmark query (got query_id={query_id!r})")
        ids, info = b["gold_ids"], {"source": "benchmark gold set gs0"}
        query = b["query"]
    elif method == "optiset":
        qid = query_id or f"Q_{int(t)}"
        with tempfile.TemporaryDirectory() as d:
            bench, out = Path(d) / "q.json", Path(d) / "optiset.json"
            bench.write_text(json.dumps([{"query_id": qid, "query": query}]), encoding="utf-8")
            sec = _run([S2S_PY, "-u", str(HERE / "optiset_runner.py"), "--benchmark", str(bench),
                        "--resources", str(RESOURCES), "--embedding-model", str(BGE),
                        "--llm-model", "Llama-3.1-70B-Instruct", "--retrieval-fields", "summary",
                        "--prompt-fields", "summary", "--top-k", "20", "--num-subqueries", "3",
                        "--temperature", "0.2", "--output", str(out)], log, "OptiSet")
            raw = json.loads(out.read_text(encoding="utf-8"))
        raw = raw.get("results", raw) if isinstance(raw, dict) else raw
        r = next(x for x in raw if x["query_id"] == qid)
        ids = list(r["predicted_resource_ids"])
        if not ids:
            raise RuntimeError(f"OptiSet returned no resources ({r.get('error') or 'empty'})")
        # OptiSet's own BGE top-20 (its first stage), shown in the Stage 1 panel
        retrieved = [{"resource_id": x["resource_id"], "score": round(float(x.get("score") or 0), 4),
                      "resource_type": x.get("resource_type"), "text": s1.texts[s1.ids.index(x["resource_id"])][:600]
                      if x["resource_id"] in s1.rmap else ""} for x in r.get("retrieved_resources") or []]
        info = {"llm_sec": sec, "subqueries": r.get("optiset_subqueries")}
    elif method in SETCORE_CKPT:
        ckpt = SETCORE_CKPT[method]
        if not (ckpt / "manifest.json").exists():
            raise RuntimeError(f"{STAGE1[method]} checkpoint missing ({ckpt}): run models/train_setcore.sh")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "setcore.json"
            sec = _run([S2S_PY, "-u", str(HERE / "setcore_runner.py"), "--model", str(ckpt), "--query", query,
                        "--query-id", query_id or f"Q_{int(t)}", "--out", str(out)] + (["--n", str(n)] if n else []),
                       log, STAGE1[method])
            r = json.loads(out.read_text(encoding="utf-8"))
        ids = [i for i in r["predicted_resource_ids"] if i in s1.rmap]
        if not ids:
            raise RuntimeError(f"{STAGE1[method]} returned no known resources")
        # the chosen set with the model's retrieval scores and concepts, shown in the Stage 1 panel
        retrieved = [{"resource_id": c["resource_id"], "score": round(c["retrieval_score"], 4),
                      "resource_type": c["type"], "text": s1.texts[s1.ids.index(c["resource_id"])][:600]}
                     for c in r["resources"] if c["resource_id"] in s1.rmap]
        info = {"model_sec": sec, "profile": r["profile"], "figure_prob": round(r["figure_prob"], 3),
                "concepts": r["concepts_U_q"][:12], "coverage": round(r["coverage"], 3), "fit_on": r.get("fit_on")}
    else:
        raise ValueError(f"unknown Stage 1 method {method!r}")
    if method == "gold":
        retrieved = []
    stub = {"stage": 1, "mode": method, "query": query, "query_id": query_id or f"Q_{int(t)}", "retrieved": retrieved,
            "pool_size": len(retrieved), "selection": {"shortlist": [], "parse_path": method, **info}}
    r1 = s1.apply_manual(stub, ids)
    r1["seconds"] = round(time.time() - t, 1)
    return r1


def stage2(method, r1, total, storyboard_id, log=print):
    """Stage 2 result (stage2.json schema: {"mode", "storyboard": {"storyboard": {...}}}) for freeform | reference."""
    t = time.time()
    qid = r1.get("query_id")
    if method == "reference":
        b = benchmark().get(qid or "")
        if not b or r1.get("mode") != "gold":
            raise ValueError("the reference storyboard belongs to the benchmark gold set: use Stage 1 'gold'")
        sb, extra = json.loads(json.dumps(b["ref"])), {}
    elif method == "freeform":
        grade = (benchmark().get(qid or "") or {}).get("grade") or "high school"
        with tempfile.TemporaryDirectory() as d:
            gpath, out = Path(d) / "gold.json", Path(d) / "ff.json"
            gpath.write_text(json.dumps(dict(r1["gold_set"], grade=grade), ensure_ascii=False), encoding="utf-8")
            _run([S2S_PY, "-u", str(HERE / "freeform_worker.py"), str(gpath), str(int(total)), str(out),
                  storyboard_id], log, "FreeForm")
            res = json.loads(out.read_text(encoding="utf-8"))
        sb = res["storyboard"]
        extra = {"parse_failed": res["freeform"].get("parse_failed")}
        if not sb.get("segments"):
            raise RuntimeError("FreeForm produced no segments" + (" (unparseable output)" if extra["parse_failed"] else ""))
    else:
        raise ValueError(f"unknown Stage 2 method {method!r}")
    sb["storyboard_id"], sb["query_id"] = storyboard_id, qid
    return {"stage": 2, "mode": method, "storyboard": {"storyboard": sb}, "seconds": round(time.time() - t, 1), **extra}
