"""
Stage 3 - storyboard -> narrated video, with either video module (run as subprocesses in their own venvs):
  manim       /mnt/storage/aryan/VideoModule/pipeline.py        (Manim CE script; ~6-15 min for 45-180 s)
  hyperframes /mnt/storage/aryan/VideoHyperFrames/hf_pipeline.py (HTML+GSAP segments; ~8-20 min, slower)
Both narrate with Parler "Mary", keep the storyboard time budget, show FIG_ resources, and use the shared vLLM
generator (:8000) / vision verifier (:8001).
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

VIDEOMODULE = Path(os.getenv("VIDEOMODULE", "/mnt/storage/aryan/VideoModule"))
HYPERFRAMES = Path(os.getenv("VIDEOHYPERFRAMES", "/mnt/storage/aryan/VideoHyperFrames"))
PY = str(VIDEOMODULE / "venv38" / "bin" / "python")
GEN_URL = os.getenv("VX_GEN_URL", "http://127.0.0.1:8000/v1")
VER_URL = os.getenv("VX_VER_URL", "http://127.0.0.1:8001/v1")

ENGINES = {
    "manim": {"label": "Manim (faster)", "script": VIDEOMODULE / "pipeline.py", "cwd": VIDEOMODULE,
              "folder": "pipeline__Qwen3.8-27B__Qwen3.5-9B", "eta_min": "6-15"},
    "hyperframes": {"label": "HyperFrames (slower, more polished)", "script": HYPERFRAMES / "hf_pipeline.py",
                    "cwd": HYPERFRAMES, "folder": "hf__Qwen3.8-27B__Qwen3.5-9B", "eta_min": "8-25"},
}
PROGRESS = re.compile(r"Narration:|--- round|figure FIG_|segment \d+:|exec_ok=|rendering|render ok|=== |Timing|"
                      r"Traceback|Error|!")


def run(storyboard_path, out_dir, engine="manim", max_rounds=3, on_line=print, timeout=4 * 3600):
    eng = ENGINES[engine]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", str(eng["script"]), "-s", str(storyboard_path), "--gen-url", GEN_URL, "--ver-url", VER_URL,
           "--max-rounds", str(max_rounds), "--out-dir", str(out_dir)]
    if engine == "manim":
        cmd += ["-q", "m"]
    t = time.time()
    log_path = out_dir / "stage3.log"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=eng["cwd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if PROGRESS.search(line):
                on_line(line.rstrip())
            if time.time() - t > timeout:
                proc.kill()
                on_line(f"! stage 3 timed out after {timeout}s")
                break
        proc.wait()
    sb_id = json.loads(Path(storyboard_path).read_text(encoding="utf-8"))
    sb_id = (sb_id.get("storyboard") or sb_id).get("storyboard_id", "storyboard")
    run_dir = out_dir / sb_id / eng["folder"]
    summary = {}
    if (run_dir / "pipeline_summary.json").exists():
        summary = json.loads((run_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
    video = run_dir / "final.mp4"
    return {"stage": 3, "engine": engine, "ok": proc.returncode == 0 and video.exists(), "returncode": proc.returncode,
            "video": str(video) if video.exists() else None, "run_dir": str(run_dir),
            "status": summary.get("status"), "timing_sec": summary.get("timing_sec"),
            "seconds": round(time.time() - t, 1), "log": str(log_path)}
