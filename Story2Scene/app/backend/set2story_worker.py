"""Persistent set2story worker (run with /mnt/storage/aryan/set2story/venv/bin/python, CPU only).

stdin : one JSON per line {"id", "gold_set": {query_id, query, gold_sets:[{resources_used:[...]}]}, "total": 180}
stdout: one JSON per line {"id", "ok", "storyboard", "roles", "regen", "llm_calls", "seconds", "error"}
        plus {"id": "ready"} once models are loaded. All logs go to stderr.
"""
import json
import os
import re
import sys
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path("/mnt/storage/aryan/set2story")
for d in ("Methodology", "Metrics", "Baseline", "training", "Runs"):
    sys.path.insert(0, str(ROOT / d))

GEN_URL = os.environ.get("S2S_GEN_URL", "http://127.0.0.1:8000/v1")
VER_URL = os.environ.get("S2S_VER_URL", "http://127.0.0.1:8001/v1")
GEN_MODEL = os.environ.get("S2S_GEN_MODEL", "Qwen3.8-27B")
# servers to put to sleep before this worker's LLM calls (all other vLLM servers on the shared GPU)
SLEEP_URLS = [u.strip() for u in os.environ.get("S2S_SLEEP_URLS", VER_URL).split(",") if u.strip()]
MODELS = ROOT / "models"
ENCODER = MODELS / "all-roberta-large-v1"
NLI_GLOB = Path.home() / ".cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-large/snapshots"
CKPT = MODELS / "role_predictor_final.pt"
FALLBACK = MODELS / "role_fallback.json"
FALLBACK_ONSCREEN = "On-screen text:"   # set2story realization.py text_render / extractive fallback prefix


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _http(method, url, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "null")


def _root(url):
    return url[:-3] if url.endswith("/v1") else url.rstrip("/")


def _sleeping(url):
    try:
        v = _http("GET", _root(url) + "/is_sleeping", timeout=10)
        return bool(v.get("is_sleeping")) if isinstance(v, dict) else bool(v)
    except Exception:
        return False


def _level(url):
    """vLLM sleep level for a server, as in VideoModule's model_loader.sleep_level_for (VLLM_SLEEP_LEVELS="url=2,...";
    the 70B on :8003 sleeps at level 2 because its weights cannot be parked in RAM)."""
    for item in os.environ.get("VLLM_SLEEP_LEVELS", "").split(","):
        u, _, lv = item.strip().rpartition("=")
        if u and u.rstrip("/") == url.rstrip("/"):
            return int(lv)
    return 1


def prepare_gpu():
    """Once per storyboard: sleep the verifier, wake the generator (shared GPU)."""
    try:
        for url in SLEEP_URLS:   # every other vLLM server on the shared GPU
            if url.rstrip("/") != GEN_URL.rstrip("/") and _reachable(url) and not _sleeping(url):
                _http("POST", _root(url) + f"/sleep?level={_level(url)}", timeout=120)
        if _sleeping(GEN_URL):
            if _level(GEN_URL) == 2:   # weights were dropped: restore buffers, reload from disk, then the KV cache
                _http("POST", _root(GEN_URL) + "/wake_up?tags=weights", timeout=300)
                _http("POST", _root(GEN_URL) + "/collective_rpc", {"method": "reload_weights"}, timeout=900)
                _http("POST", _root(GEN_URL) + "/wake_up?tags=kv_cache", timeout=300)
            else:
                _http("POST", _root(GEN_URL) + "/wake_up", timeout=300)
    except Exception as exc:
        log("gpu swap warning:", exc)


def _reachable(url):
    try:
        _http("GET", _root(url) + "/is_sleeping", timeout=5)
        return True
    except Exception:
        return False


class VLLMLLM:
    """llm(prompt, max_new_tokens) -> str via the OpenAI-compatible vLLM generator (thinking off, temperature 0)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt, max_new_tokens=400):
        body = {"model": GEN_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "max_tokens": int(max_new_tokens), "chat_template_kwargs": {"enable_thinking": False}}
        out = _http("POST", GEN_URL + "/chat/completions", body)
        self.calls += 1
        text = out["choices"][0]["message"].get("content") or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def apportion(durs, total):
    """Integer durations (>=1) summing exactly to total, largest remainder."""
    s = sum(durs) or 1.0
    raw = [d * total / s for d in durs]
    out = [max(1, int(x)) for x in raw]
    diff = total - sum(out)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - int(raw[i]), reverse=(diff > 0))
    i = 0
    while diff != 0 and order:
        j = order[i % len(order)]
        if diff > 0:
            out[j] += 1; diff -= 1
        elif out[j] > 1:
            out[j] -= 1; diff += 1
        i += 1
    return out


class Worker:
    def __init__(self):
        import torch
        torch.set_num_threads(int(os.environ.get("S2S_THREADS", "8")))
        from backend import BatchedNLI
        from encoder import SentenceTransformerEncoder
        from pipeline import GSCContext
        from realization import NarrationVerifier
        from role_predictor import RoleSequencePredictor
        from Role_Metrics import canonical_role
        from slotfill import role_duration_shares
        self.torch = torch
        for p in (ENCODER, CKPT):
            if not p.exists():
                raise FileNotFoundError(f"missing {p}")
        snaps = sorted(NLI_GLOB.glob("*"))
        if not snaps:
            raise FileNotFoundError(f"NLI model not found under {NLI_GLOB}")
        t = time.time()
        enc = SentenceTransformerEncoder(str(ENCODER), device="cpu")
        self.ctx = GSCContext(enc, json.load(open(ROOT / "inputs/concept_vocab_curriculum.json")))
        self.nli = BatchedNLI(str(snaps[-1]), device="cpu", max_batch=64)
        self.verifier = NarrationVerifier(self.nli, entail_threshold=0.5, contra_threshold=0.2)
        ck = torch.load(CKPT, map_location="cpu")
        c = ck["cfg"]
        self.pred = RoleSequencePredictor(ck["roles"], d=c["d"], hidden=c["hidden"],
                                          role_emb_dim=c["role_emb_dim"], dropout=c["dropout"])
        self.pred.load_state_dict(ck["state_dict"]); self.pred.eval()
        self.fallback = json.load(open(FALLBACK))["role_template"] if FALLBACK.exists() else ["hook", "definition", "example", "summary"]
        boards = []
        for it in json.load(open(ROOT / "inputs/Reference_Storyboard.json")):
            segs = [{"role": canonical_role(s["role"]), "duration_sec": s["duration_sec"]} for s in it["storyboard"]["segments"]]
            boards.append({"segments": segs})
        self.shares = role_duration_shares(boards)
        self.llm = VLLMLLM()
        log(f"loaded in {time.time()-t:.0f}s")

    def run(self, gold_set, total):
        from pipeline import DEFAULT_PARAMS, build_storyboard, predict_role_sequence
        from realization import make_segment_generator
        query, qid = gold_set["query"], gold_set.get("query_id", "Q")
        gold = gold_set["gold_sets"][0]
        p = DEFAULT_PARAMS
        Rq, Uq, prec = self.ctx.prepare(query, gold, p["gamma"], p["kappa"])
        roles = predict_role_sequence(self.pred, self.ctx.query_vec(query))
        if not roles:
            roles = list(self.fallback)
        prepare_gpu()
        c0 = self.llm.calls
        board = build_storyboard(query, roles, Rq, Uq, prec, self.shares, float(total),
                                 make_segment_generator(self.llm, 400), self.verifier, p)
        calls = self.llm.calls - c0
        # intent: cheap single call
        try:
            intent = self.llm("In one short sentence (max 20 words), state the learning goal of a video answering "
                              f"this learner question: {query}\nReturn only the sentence.", 60).strip().split("\n")[0]
        except Exception:
            intent = ""
        intent = intent or f"Explain: {query}"
        durs = apportion([s.duration for s in board.segments], int(total))
        by_id = {r["resource_id"]: r for r in gold["resources_used"]}
        segs = []
        for k, (s, d) in enumerate(zip(board.segments, durs)):
            concepts = []
            for r in s.resources:
                for cc in by_id.get(r.rid, {}).get("concepts", []) or sorted(r.concepts):
                    if cc not in concepts:
                        concepts.append(cc)
            onscreen = s.onscreen or ""
            # set2story's realisation fallback ("On-screen text: <narration or a resource sentence>") turns the
            # screen into a transcript of the voice-over; ask for a title and key-term labels instead
            if FALLBACK_ONSCREEN in onscreen:
                terms = ", ".join(c.replace("_", " ") for c in concepts[:4])
                onscreen = re.sub(re.escape(FALLBACK_ONSCREEN) + r".*", "A short title and labels for the key terms"
                                  + (f": {terms}." if terms else " (a few words each)."), onscreen, flags=re.S).strip()
            fig = s.figure
            if fig is not None and fig.rid not in onscreen:   # fidelity fallback lost it: restore the figure directive
                onscreen = f"Figure {fig.rid} is shown. {onscreen}".strip()
            segs.append({"segment_id": f"SEG_{k+1:03d}", "order": k, "resource_ids": [r.rid for r in s.resources],
                         "role": s.role, "duration_sec": d, "narration": s.narration or "", "on_screen": onscreen,
                         "concepts": concepts[:4]})
        sb = {"storyboard_id": f"SB_{qid}", "query_id": qid, "query": query, "intent": intent,
              "target_duration_sec": int(total), "total_duration_sec": sum(durs),
              "role_template": " -> ".join(s.role for s in board.segments), "segments": segs}
        regen = [len(s.trace.get("attempts", [])) - 1 for s in board.segments if s.trace.get("attempts")]
        return {"storyboard": {"storyboard": sb}, "roles": board.role_sequence(),
                "regen": sum(max(0, x) for x in regen), "llm_calls": calls + 1}


def main():
    try:
        w = Worker()
    except Exception as exc:
        print(json.dumps({"id": "ready", "ok": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 1
    print(json.dumps({"id": "ready", "ok": True}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid, t = None, time.time()
        try:
            req = json.loads(line)
            rid = req.get("id")
            res = w.run(req["gold_set"], req.get("total", 180))
            out = {"id": rid, "ok": True, **res, "seconds": round(time.time() - t, 1), "error": None}
        except Exception as exc:
            log(traceback.format_exc())
            out = {"id": rid, "ok": False, "storyboard": None, "roles": [], "regen": 0, "llm_calls": 0,
                   "seconds": round(time.time() - t, 1), "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(out, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
