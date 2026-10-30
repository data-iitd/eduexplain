"""
VideoExplain backend: the 3-stage pipeline as a local web API.

  Stage 1  setCore     query -> retrieved resources -> gold explanation set
  Stage 2  set2story   gold set -> storyboard
  Stage 3  video       storyboard -> narrated video (Manim or HyperFrames)

One job runs at a time (one shared A100: the vLLM generator/verifier swap the GPU). Each stage's output is written
to runs/<job>/stageN.json as soon as it is ready, so the UI shows the pipeline progressing panel by panel.

    venv/bin/uvicorn app:app --host 0.0.0.0 --port 8800      (from backend/)
"""
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(os.getenv("VIDEOMODULE", "/mnt/storage/aryan/VideoModule"))   # stdlib vLLM client + GPU swap

from model_loader import client_from_env, share_gpu  # noqa: E402
import baselines  # noqa: E402
import stage3 as stage3_mod  # noqa: E402
import study  # noqa: E402

app = FastAPI(title="VideoExplain")
app.include_router(study.router)   # user study E2 (tap-only study tab)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

generator = client_from_env("generator", "Qwen3.8-27B", stage3_mod.GEN_URL)
verifier = client_from_env("verifier", "Qwen3.5-9B", stage3_mod.VER_URL)
share_gpu(generator, verifier)


def _stage_llm(stage):
    """Per-stage LLM endpoint. Default: the shared generator (Qwen3.8-27B) standing in for setCore's
    Llama-3.1-70B (stage 1) and set2story's gemma-3-12b-it (stage 2). Override with VX_STAGE{n}_LLM_URL /
    VX_STAGE{n}_LLM_MODEL, e.g. a separately served original model or backend/dummy_llm.py (:8002)."""
    url = os.getenv(f"VX_STAGE{stage}_LLM_URL")
    if not url or url.rstrip("/") == generator.base_url:
        return generator
    from model_loader import VLLMClient
    return VLLMClient(f"stage{stage}", url, os.getenv(f"VX_STAGE{stage}_LLM_MODEL", "dummy"))
_services = {}
_lock = threading.Lock()
jobs = {}
work = queue.Queue()


def services():
    """Stage 1/2 services are created lazily (Stage 1 loads the corpus embeddings once)."""
    with _lock:
        if not _services:
            from stage1 import Stage1
            from stage2 import Stage2
            _services["s1"] = Stage1(llm=_stage_llm(1), log=lambda m: print("[stage1]", m, flush=True))
            _services["s2"] = Stage2(llm=_stage_llm(2), log=lambda m: print("[stage2]", m, flush=True))
        return _services["s1"], _services["s2"]


class JobIn(BaseModel):
    query: str
    engine: str = "manim"          # manim | hyperframes
    n: int | None = None           # gold-set size (default: setCore's typical size)
    duration_sec: int = 180        # storyboard time budget
    max_rounds: int = 3
    stop_after: int = 3            # run stages 1..stop_after
    selection: str = "auto"        # Stage 1: auto (SetCoRe-LLM) | manual (the user ticks resources) | lw | e2e | optiset | gold
    stage2: str = "set2story"      # Stage 2: set2story | freeform | reference (benchmark storyboard, gold Stage 1)
    query_id: str | None = None    # a benchmark query (EduExplain id), needed by the gold set and the reference


class SelectionIn(BaseModel):
    resource_ids: list[str]        # the ticked resources, in the order they should be used


def _save(job):
    d = RUNS / job["id"]
    d.mkdir(exist_ok=True)
    (d / "job.json").write_text(json.dumps({k: v for k, v in job.items() if k != "results"}, indent=2), encoding="utf-8")


def _log(job, msg):
    job["log"].append(f"{time.strftime('%H:%M:%S')} {msg}")
    job["log"] = job["log"][-400:]
    # Stage 3 milestones in the video module's output (stage3.PROGRESS lines) move the progress bar
    if job.get("stage") != 3:
        return
    if "--- round " in msg:
        _phase(job, "round" + msg.split("--- round ")[1].split()[0])
    elif "rendering " in msg:
        _phase(job, "render")


# Progress bar: the pipeline as a sequence of phases with their typical length in seconds (measured on this
# machine). Done phases count fully, the current one fills asymptotically (1 - e^(-t/expected)), so the bar never
# reaches its end before the phase does.
EXPECTED_SEC = {"stage1": {"auto": 150, "manual": 20, "lw": 40, "e2e": 40, "optiset": 600, "gold": 3},
                "stage2": {"set2story": 300, "freeform": 150, "reference": 2}, "tts": 150, "round": 130, "render": 300}


def _phase(job, name):
    job["phase"], job["phase_started"] = name, time.time()


def _phases(spec):
    sel, s2 = spec.get("selection", "auto"), spec.get("stage2", "set2story")
    ph = [("stage1", EXPECTED_SEC["stage1"].get(sel, 150), "Stage 1: choosing resources")]
    if spec.get("stop_after", 3) >= 2:
        ph.append(("stage2", EXPECTED_SEC["stage2"].get(s2, 300), "Stage 2: writing the storyboard"))
    if spec.get("stop_after", 3) >= 3:
        ph.append(("tts", EXPECTED_SEC["tts"], "Stage 3: recording the narration"))
        for k in range(int(spec.get("max_rounds", 3)) + 1):
            ph.append((f"round{k}", EXPECTED_SEC["round"],
                       "Stage 3: writing the animation" if k == 0 else f"Stage 3: fixing the animation (round {k})"))
        ph.append(("render", EXPECTED_SEC["render"], "Stage 3: rendering the video"))
    return ph


def _progress(job):
    status = job.get("status")
    if status == "done":
        return {"pct": 100, "label": "Done", "eta_sec": 0}
    ph = _phases(job.get("spec", {}))
    total = sum(e for _, e, _ in ph)
    names = [n for n, _, _ in ph]
    cur = job.get("phase")
    if status == "queued" or cur not in names:
        return {"pct": 0, "label": "Waiting in the queue" if status == "queued" else str(status), "eta_sec": total}
    i = names.index(cur)
    done = sum(e for _, e, _ in ph[:i])
    exp_sec, label = ph[i][1], ph[i][2]
    frac = 1 - math.exp(-max(0.0, time.time() - job.get("phase_started", time.time())) / exp_sec)
    if status != "running":   # failed / interrupted / awaiting a selection: the bar stops where the job stopped
        frac = 1.0 if status == "awaiting_selection" else frac
        label = {"awaiting_selection": "Waiting for your selection", "failed": "Failed: " + label}.get(status, f"{status}: {label}")
    pct = 100 * (done + exp_sec * frac) / total
    return {"pct": round(min(pct, 99.0), 1), "label": label, "eta_sec": round(max(0.0, total - done - exp_sec * frac))}


def _run(job):
    d = RUNS / job["id"]
    spec = job["spec"]
    try:
        s1, s2 = services()
        sel = spec.get("selection", "auto")
        _phase(job, "stage1")
        if job.get("manual_ids"):   # manual mode, resumed after the user ticked the gold set
            r1 = s1.apply_manual(_read(job["id"], "stage1.json"), job.pop("manual_ids"))
            (d / "stage1.json").write_text(json.dumps(r1, indent=2, ensure_ascii=False), encoding="utf-8")
            job["stages"]["1"] = "done"; _log(job, f"Stage 1 done (manual, {len(r1['selection']['selected'])} resources)")
        elif spec.get("selection") == "manual":
            job["stage"], job["stages"]["1"] = 1, "running"; _log(job, "Stage 1: retrieving candidates for manual selection")
            r1 = s1.candidates(spec["query"], n=spec.get("n"))
            (d / "stage1.json").write_text(json.dumps(r1, indent=2, ensure_ascii=False), encoding="utf-8")
            job["stages"]["1"], job["awaiting"] = "awaiting", True
            _log(job, f"Stage 1: {len(r1['retrieved'])} candidates ready - tick the resources to use, then continue")
            return
        elif sel in ("lw", "e2e", "optiset", "gold"):
            job["stage"], job["stages"]["1"] = 1, "running"; _log(job, f"Stage 1: {baselines.STAGE1[sel]}")
            r1 = baselines.stage1(s1, sel, spec["query"], spec.get("query_id"), log=lambda m: _log(job, m), n=spec.get("n"))
            (d / "stage1.json").write_text(json.dumps(r1, indent=2, ensure_ascii=False), encoding="utf-8")
            job["stages"]["1"] = "done"; _log(job, f"Stage 1 done ({r1['mode']}, {len(r1['selection']['selected'])} resources)")
        else:
            job["stage"], job["stages"]["1"] = 1, "running"; _log(job, "Stage 1: retrieving resources and selecting the gold set")
            r1 = s1.run(spec["query"], n=spec.get("n"), query_id=spec.get("query_id"))
            (d / "stage1.json").write_text(json.dumps(r1, indent=2, ensure_ascii=False), encoding="utf-8")
            job["stages"]["1"] = "done"; _log(job, f"Stage 1 done ({r1['mode']}, {len(r1['selection']['selected'])} resources)")
        if spec["stop_after"] < 2:
            return
        m2 = spec.get("stage2", "set2story")
        job["stage"], job["stages"]["2"] = 2, "running"; _log(job, f"Stage 2: writing the storyboard ({baselines.STAGE2[m2]})")
        _phase(job, "stage2")
        if m2 == "set2story":
            r2 = s2.run(r1, total=spec["duration_sec"])
        else:
            r2 = baselines.stage2(m2, r1, spec["duration_sec"], f"SB_{job['id']}", log=lambda m: _log(job, m))
        (d / "stage2.json").write_text(json.dumps(r2, indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "storyboard.json").write_text(json.dumps(r2["storyboard"], indent=2, ensure_ascii=False), encoding="utf-8")
        job["stages"]["2"] = "done"; _log(job, f"Stage 2 done ({r2['mode']}, {len(r2['storyboard']['storyboard']['segments'])} segments)")
        if spec["stop_after"] < 3:
            return
        job["stage"], job["stages"]["3"] = 3, "running"
        _phase(job, "tts")
        _log(job, f"Stage 3: {stage3_mod.ENGINES[spec['engine']]['label']} (expected {stage3_mod.ENGINES[spec['engine']]['eta_min']} min)")
        r3 = stage3_mod.run(d / "storyboard.json", d / "stage3", spec["engine"], spec["max_rounds"],
                            on_line=lambda line: _log(job, "  " + line))
        (d / "stage3.json").write_text(json.dumps(r3, indent=2), encoding="utf-8")
        job["stages"]["3"] = "done" if r3["ok"] else "failed"
        _log(job, f"Stage 3 {'done' if r3['ok'] else 'FAILED'} ({r3.get('status')}, {r3['seconds']}s)")
    except Exception as exc:
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["stages"][str(job["stage"])] = "failed"
        _log(job, "ERROR " + job["error"])
        (d / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if job.get("awaiting") and not job.get("error"):
            job["status"] = "awaiting_selection"   # frees the worker; POST /api/jobs/{id}/selection resumes it
        else:
            job["finished"] = time.time()
            job["status"] = "failed" if job.get("error") or "failed" in job["stages"].values() else "done"
        _save(job)


def _worker():
    while True:
        job = work.get()
        if job.get("status") == "cancelled":
            continue
        job["status"], job["started"] = "running", time.time()
        _save(job)
        _run(job)


def _recover():
    """The queue lives in memory: jobs a previous backend left queued or running will never finish. Mark them
    interrupted so the UI does not show them as pending forever (awaiting_selection jobs stay resumable)."""
    for p in RUNS.glob("*/job.json"):
        try:
            job = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if job.get("status") in ("queued", "running"):
            job["status"] = "interrupted"
            job["stages"] = {k: ("failed" if v == "running" else v) for k, v in job.get("stages", {}).items()}
            job.setdefault("log", []).append(f"{time.strftime('%H:%M:%S')} interrupted: the backend restarted")
            p.write_text(json.dumps(job, indent=2), encoding="utf-8")


_recover()
threading.Thread(target=_worker, daemon=True).start()


# ---------------------------------------------------------------- API
@app.post("/api/jobs")
def create_job(spec: JobIn):
    if spec.engine not in stage3_mod.ENGINES:
        raise HTTPException(400, f"engine must be one of {list(stage3_mod.ENGINES)}")
    if spec.selection not in baselines.STAGE1:
        raise HTTPException(400, f"selection must be one of {list(baselines.STAGE1)}")
    if spec.stage2 not in baselines.STAGE2:
        raise HTTPException(400, f"stage2 must be one of {list(baselines.STAGE2)}")
    if spec.query_id and spec.query_id not in baselines.benchmark():
        raise HTTPException(400, f"unknown benchmark query {spec.query_id}")
    if spec.selection == "gold" and not spec.query_id:
        raise HTTPException(400, "the benchmark gold set needs a benchmark query (pick one from the list)")
    if spec.stage2 == "reference" and spec.selection != "gold":
        raise HTTPException(400, "the benchmark reference storyboard needs the benchmark gold set in Stage 1")
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    job = {"id": jid, "spec": spec.model_dump(), "status": "queued", "stage": 0, "created": time.time(),
           "stages": {"1": "pending", "2": "pending", "3": "pending"}, "log": [], "queue_position": work.qsize()}
    jobs[jid] = job
    _save(job)
    work.put(job)
    return {"id": jid, "queue_position": job["queue_position"]}


@app.post("/api/jobs/{jid}/selection")
def submit_selection(jid: str, sel: SelectionIn):
    """Manual mode: the ticked resources become the gold set; the job is queued again for stages 2-3."""
    job = jobs.get(jid) or _read(jid, "job.json")
    if not job:
        raise HTTPException(404, "unknown job")
    if job.get("status") != "awaiting_selection":
        raise HTTPException(409, f"job is {job.get('status')}, not awaiting a selection")
    cand = _read(jid, "stage1.json") or {}
    pool = {r["resource_id"] for r in cand.get("retrieved", [])}
    ids = list(dict.fromkeys(sel.resource_ids))
    bad = [i for i in ids if i not in pool]
    if not ids or bad:
        raise HTTPException(400, "select at least one resource" if not ids else f"not in the candidate list: {bad}")
    job.setdefault("log", [])
    job.update(manual_ids=ids, awaiting=False, status="queued", queue_position=work.qsize())
    _log(job, f"Stage 1: {len(ids)} resources selected by the user")
    jobs[jid] = job
    _save(job)
    work.put(job)
    return {"id": jid, "selected": ids, "queue_position": job["queue_position"]}


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    """Drop a queued job (a running job cannot be stopped: one GPU pipeline is in flight)."""
    job = jobs.get(jid)
    if not job or job.get("status") not in ("queued", "awaiting_selection"):
        raise HTTPException(409, f"only queued or awaiting jobs can be cancelled (job is {job and job.get('status')})")
    job["status"] = "cancelled"
    _log(job, "cancelled")
    _save(job)
    return {"id": jid, "status": "cancelled"}


def _read(jid, name):
    p = RUNS / jid / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    job = jobs.get(jid) or _read(jid, "job.json")
    if not job:
        raise HTTPException(404, "unknown job")
    s3 = _read(jid, "stage3.json")
    return {**job, "stage1": _read(jid, "stage1.json"), "stage2": _read(jid, "stage2.json"), "stage3": s3,
            "video_url": f"/api/jobs/{jid}/video" if s3 and s3.get("video") else None, "progress": _progress(job)}


@app.get("/api/jobs")
def list_jobs():
    """Premade results (runs/premade-*, the demo evaluation videos; all of them) and the 50 latest user jobs."""
    names = [p.name for p in RUNS.iterdir() if p.is_dir()]
    premade = sorted(n for n in names if n.startswith(PREMADE))
    ids = premade + sorted((n for n in names if not n.startswith(PREMADE)), reverse=True)[:50]
    out = []
    for jid in ids:
        j = jobs.get(jid) or _read(jid, "job.json") or {}
        sp = j.get("spec", {})
        out.append({"id": jid, "query": sp.get("query"), "engine": sp.get("engine"), "status": j.get("status"),
                    "stages": j.get("stages"), "premade": jid.startswith(PREMADE), "query_id": sp.get("query_id"),
                    "selection": sp.get("selection", "auto"), "stage2": sp.get("stage2", "set2story"),
                    "stage1_label": sp.get("stage1_label"), "stage2_label": sp.get("stage2_label")})
    return out


PREMADE = "premade-"


@app.get("/api/benchmark")
def benchmark_queries():
    """The benchmark queries (EduExplain) that can use the gold set / reference storyboard."""
    return [{"query_id": k, "query": v["query"], "T": v["T"], "n_gold": len(v["gold_ids"])}
            for k, v in sorted(baselines.benchmark().items())]


@app.get("/api/jobs/{jid}/video")
def video(jid: str):
    s3 = _read(jid, "stage3.json")
    if not s3 or not s3.get("video") or not Path(s3["video"]).exists():
        raise HTTPException(404, "no video yet")
    return FileResponse(s3["video"], media_type="video/mp4")


@app.get("/api/figure/{fig_id}")
def figure(fig_id: str):
    """PNG of a textbook figure resource (FIG_xxxx), for the Stage 1/2 panels."""
    from figures import load_mapping   # VideoModule/figures.py (shared figure mapping)
    path = load_mapping().get(fig_id)
    if not path:
        raise HTTPException(404, f"unknown figure {fig_id}")
    return FileResponse(path, media_type="image/png")


@app.get("/api/health")
def health():
    def probe(c):
        try:
            return {"url": c.base_url, "up": True, "sleeping": c.is_sleeping()}
        except Exception as exc:
            return {"url": c.base_url, "up": False, "error": str(exc)[:200]}
    from stage1 import BGE_DIR
    return {"generator": probe(generator), "verifier": probe(verifier),
            "stage1_bge": (BGE_DIR / "onnx" / "model.onnx").exists(),
            "engines": {k: {"label": v["label"], "eta_min": v["eta_min"]} for k, v in stage3_mod.ENGINES.items()},
            "queue": work.qsize(), "busy": any(j["status"] == "running" for j in jobs.values())}
