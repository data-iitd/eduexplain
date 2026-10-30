"""Backend-side adapter for the set2story worker (stdlib only; the worker runs in set2story's own venv)."""
import atexit
import json
import subprocess
import threading
from pathlib import Path

S2S = Path("/mnt/storage/aryan/set2story")
PYTHON = S2S / "venv/bin/python"
WORKER = Path(__file__).resolve().parent / "set2story_worker.py"
REQUIRED = [S2S / "models/all-roberta-large-v1/modules.json", S2S / "models/role_predictor_final.pt",
            S2S / "models/role_fallback.json"]


class Set2StoryAdapter:
    def __init__(self, llm=None, log=print, ready_timeout=900, run_timeout=1800):
        self.llm, self.log = llm, log      # llm unused: the worker talks to the vLLM generator itself
        self.ready_timeout, self.run_timeout = ready_timeout, run_timeout
        self.proc, self._n, self._lock = None, 0, threading.Lock()
        self._check()
        atexit.register(self.close)

    def _check(self):
        if not PYTHON.exists():
            raise RuntimeError(f"set2story venv missing: {PYTHON}")
        missing = [str(p) for p in REQUIRED if not p.exists()]
        if missing:
            raise RuntimeError("set2story models/checkpoint missing: " + ", ".join(missing))

    def _readline(self, timeout):
        res = []
        t = threading.Thread(target=lambda: res.append(self.proc.stdout.readline()), daemon=True)
        t.start(); t.join(timeout)
        if not res:
            self.close()
            raise RuntimeError(f"set2story worker timed out after {timeout}s")
        if not res[0]:
            rc = self.proc.poll()
            self.proc = None
            raise RuntimeError(f"set2story worker exited (rc={rc})")
        return json.loads(res[0])

    def _start(self):
        self.log("starting set2story worker ...")
        self.proc = subprocess.Popen([str(PYTHON), str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=None, text=True, bufsize=1)
        msg = self._readline(self.ready_timeout)
        if not msg.get("ok"):
            self.close()
            raise RuntimeError(f"set2story worker failed to start: {msg.get('error')}")
        self.log("set2story worker ready")

    def run(self, gold_set, total=180):
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            self._n += 1
            rid = f"req{self._n}"
            self.proc.stdin.write(json.dumps({"id": rid, "gold_set": gold_set, "total": total}) + "\n")
            self.proc.stdin.flush()
            msg = self._readline(self.run_timeout)
        if not msg.get("ok"):
            raise RuntimeError(f"set2story failed: {msg.get('error')}")
        self.last = {k: msg.get(k) for k in ("roles", "regen", "llm_calls", "seconds")}
        return msg["storyboard"]

    def close(self):
        p, self.proc = self.proc, None
        if p and p.poll() is None:
            try:
                p.stdin.close(); p.wait(timeout=10)
            except Exception:
                p.kill()
