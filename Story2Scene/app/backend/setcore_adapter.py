"""Backend-side adapter for setCore stages 2-4 (lightweight n_hat_q + coverage-greedy reduce), stdlib only.

The numpy/sentence-transformers work runs in setcore_worker.py under set2story's venv (CPU), as a persistent
JSON-lines subprocess started lazily on first use. Any failure raises RuntimeError so stage1 can fall back.

    r = SetCoreReducer(); r.load()
    out = r.shortlist(query, pool)   # pool: [{"resource_id", "score", ...}, ...] (stage1.retrieve output)
    out -> {"n_hat", "B", "shortlist": [resource_id...], "method": "setcore-reduce", "profile", "U_q", ...}
"""
import atexit
import json
import subprocess
import threading
from pathlib import Path

SETCORE = Path("/mnt/storage/aryan/setCore")
PYTHON = Path("/mnt/storage/aryan/set2story/venv/bin/python")
WORKER = Path(__file__).resolve().parent / "setcore_worker.py"
REQUIRED = [SETCORE / "models/lw/role_head_s0.npz", SETCORE / "models/bge-large-en-v1.5/modules.json",
            SETCORE / "scx_reduce.py", SETCORE / "icra/pipeline.py"]


class SetCoreReducer:
    def __init__(self, log=print, ready_timeout=900, run_timeout=120):
        self.log, self.ready_timeout, self.run_timeout = log, ready_timeout, run_timeout
        self.proc, self._n, self._lock = None, 0, threading.Lock()
        atexit.register(self.close)

    def _check(self):
        if not PYTHON.exists():
            raise RuntimeError(f"setcore worker venv missing: {PYTHON}")
        missing = [str(p) for p in REQUIRED if not p.exists()]
        if missing:
            raise RuntimeError("setCore lw artifacts missing: " + ", ".join(missing))

    def _readline(self, timeout):
        res = []
        t = threading.Thread(target=lambda: res.append(self.proc.stdout.readline()), daemon=True)
        t.start(); t.join(timeout)
        if not res:
            self.close()
            raise RuntimeError(f"setcore worker timed out after {timeout}s")
        if not res[0]:
            rc = self.proc.poll()
            self.proc = None
            raise RuntimeError(f"setcore worker exited (rc={rc})")
        return json.loads(res[0])

    def load(self):
        with self._lock:
            self._load()

    def _load(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        self._check()
        self.log("starting setCore lw worker ...")
        self.proc = subprocess.Popen([str(PYTHON), str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=None, text=True, bufsize=1)
        msg = self._readline(self.ready_timeout)
        if not msg.get("ok"):
            self.close()
            raise RuntimeError(f"setcore worker failed to start: {msg.get('error')}")
        self.log("setCore lw worker ready")

    def shortlist(self, query, pool):
        pool = [{"resource_id": p["resource_id"], "score": p.get("score")} for p in pool]
        with self._lock:
            self._load()
            self._n += 1
            self.proc.stdin.write(json.dumps({"id": f"req{self._n}", "query": query, "pool": pool}) + "\n")
            self.proc.stdin.flush()
            msg = self._readline(self.run_timeout)
        if not msg.get("ok"):
            raise RuntimeError(f"setcore reduce failed: {msg.get('error')}")
        msg["method"] = "setcore-reduce"
        return msg

    def close(self):
        p, self.proc = self.proc, None
        if p and p.poll() is None:
            try:
                p.stdin.close(); p.wait(timeout=10)
            except Exception:
                p.kill()
