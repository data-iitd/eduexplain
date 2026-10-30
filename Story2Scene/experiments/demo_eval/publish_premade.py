"""
Publish the 35 demo-evaluation videos as the website's premade results: every existing job in runs/ is moved to
runs_archive/<date>/ (nothing is deleted), and each video becomes a finished job runs/premade-<stage1>__<stage2>__<qid>/
with the usual job.json / stage1.json / stage2.json / stage3.json, so the stage panels show it like any other job.
The videos stay where they are (stage3.json points at experiments/demo_eval/videos/...).

    /mnt/storage/aryan/VideoExplain/venv/bin/python publish_premade.py
"""
import json
import re
import shutil
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent
VX = EXP.parent.parent
RUNS = VX / "runs"
PREMADE = "premade-"
S1 = {"gold": ("gold", "Benchmark gold set"), "lw": ("lw", "SetCoRe-LW"), "e2e": ("e2e", "SetCoRe-E2E"),
      "x": ("auto", "SetCoRe-LLM"), "optiset": ("optiset", "OptiSet")}
JOB_ID = {"x": "llm"}   # job id prefix per Stage 1 dir (x = SetCoRe-LLM, renamed from SetCoRe-X)
S2 = {"s2s": ("set2story", "Set2Story"), "ref": ("reference", "Benchmark reference"), "ff": ("freeform", "FreeForm")}
sys.path.insert(0, str(VX / "backend"))
from stage3 import PROGRESS  # noqa: E402


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jsave(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    bench = jload(EXP / "benchmark_items.json")
    archive = VX / "runs_archive" / time.strftime("%Y%m%d-%H%M%S")
    old = [p for p in RUNS.iterdir()]
    if old:
        archive.mkdir(parents=True)
        for p in old:
            shutil.move(str(p), archive / p.name)
        print(f"archived {len(old)} entries -> {archive}")
    n = 0
    for vdir in sorted((EXP / "videos").iterdir()):
        st1, st2, q = vdir.name.split("__")
        r3 = jload(vdir / "stage3.json")
        if not r3.get("ok") or not Path(r3.get("video") or "").exists():
            print(f"! {vdir.name}: no finished video, skipped")
            continue
        (sel, l1), (m2, l2) = S1[st1], S2[st2]
        b, sb = bench[q], jload(EXP / "stage2" / f"{vdir.name}.json")
        r1 = jload(EXP / "stage1" / st1 / f"{q}.json")
        r1["mode"] = l1
        r2 = {"stage": 2, "mode": l2, "storyboard": {"storyboard": sb["storyboard"]},
              "seconds": (sb.get("meta") or {}).get("seconds")}
        log = [f"premade result (demo evaluation E1): Stage 1 {l1}, Stage 2 {l2}, Manim, budget {b['T']} s"]
        if (vdir / "stage3.log").exists():
            log += ["  " + ln.rstrip() for ln in (vdir / "stage3.log").read_text(encoding="utf-8").splitlines()
                    if PROGRESS.search(ln)][-60:]
        jid = PREMADE + "__".join([JOB_ID.get(st1, st1), st2, q])
        d = RUNS / jid
        d.mkdir(parents=True)
        t = (vdir / "stage3.json").stat().st_mtime
        jsave(d / "job.json", {
            "id": jid, "status": "done", "stage": 3, "created": t, "finished": t,
            "spec": {"query": b["query"], "query_id": q, "engine": "manim", "duration_sec": b["T"], "max_rounds": 3,
                     "stop_after": 3, "selection": sel, "stage2": m2, "stage1_label": l1, "stage2_label": l2},
            "stages": {"1": "done", "2": "done", "3": "done"}, "log": log})
        jsave(d / "stage1.json", r1)
        jsave(d / "stage2.json", r2)
        jsave(d / "storyboard.json", {"storyboard": sb["storyboard"]})
        jsave(d / "stage3.json", r3)
        n += 1
    print(f"published {n} premade results in {RUNS}")


if __name__ == "__main__":
    if not re.fullmatch(r".*/VideoExplain", str(VX)):
        sys.exit(f"unexpected layout: {VX}")
    main()
