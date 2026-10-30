"""
User study E2 (demo paper, experiments_plan.tex): participants watch two unlabeled videos for the same benchmark
question next to the explanation set each was made from (the Stage 1 resources, as readable textbook text and
figures) and answer three questions by tapping one of four choices:

  correct     which video has fewer errors in the concepts it explains
  faithful    which video preserves the meaning of its explanation set
  coherent    which video explains the topic more coherently
  choices     a | b | same | cant  (Video 1, Video 2, the same, can't judge)

  comparison A  SetCoRe (A1, SetCoRe-LW + Set2Story)  vs  OptiSet (A4, OptiSet + Set2Story)   own set per video
  comparison B  Set2Story (B1, gold + Set2Story)      vs  FreeForm (B2, gold + FreeForm)       one shared set

Each session gets PAIRS_PER_SESSION pairs (half A, half B) on distinct questions, chosen so that every
(comparison, question) is answered about equally often; sides and order are random. Videos are served through
/api/study/{sid}/video/{k}/{side}, so nothing in the page names the configuration. Answers are saved as they are
given (study/sessions/<sid>.json); GET /api/study/summary/all aggregates them per question (wins, sign test).
The first design (1-5 ratings, overall preference, SUS) is archived in runs_archive/study-olddesign-*/.

While study/PAUSED exists, requests that come from the public site (through the reverse tunnel, Host port 18800)
cannot start sessions or answer; the local site (Vite on :5173) keeps working, so a new design can be tried there.
"""
import html
import json
import math
import re
import random
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

import baselines

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
STUDY = ROOT / "study" / "sessions"
STUDY.mkdir(parents=True, exist_ok=True)
PAUSED = ROOT / "study" / "PAUSED"
PUBLIC_PORT = ":18800"                 # Host of requests forwarded by the VM (deploy/tunnel.sh -R 18800:8800)
QIDS = ["Q007098", "Q006487", "Q015643", "Q006124", "Q015908"]
COMPARISONS = {
    "A": {"title": "Stage 1: SetCoRe vs OptiSet", "configs": ("lw__s2s", "optiset__s2s"),
          "labels": ("SetCoRe-LW + Set2Story", "OptiSet + Set2Story")},
    "B": {"title": "Stage 2: Set2Story vs FreeForm", "configs": ("gold__s2s", "gold__ff"),
          "labels": ("Gold set + Set2Story", "Gold set + FreeForm")},
}
OURS = {"lw__s2s", "gold__s2s"}
QUESTIONS = ("correct", "faithful", "coherent")
CHOICES = ("a", "b", "same", "cant")
PAIRS_PER_SESSION = 4
ROLES = {"student", "teacher", "other"}
_lock = threading.Lock()
router = APIRouter(prefix="/api/study")


def _open(request: Request):
    if PAUSED.exists() and request.headers.get("host", "").endswith(PUBLIC_PORT):
        raise HTTPException(503, PAUSED.read_text(encoding="utf-8").strip() or "The user study is paused.")


def _path(sid):
    if not sid.isalnum():
        raise HTTPException(400, "bad session id")
    return STUDY / f"{sid}.json"


def _load(sid):
    p = _path(sid)
    if not p.exists():
        raise HTTPException(404, "unknown session")
    return json.loads(p.read_text(encoding="utf-8"))


def _save(s):
    s["updated"] = time.time()
    _path(s["sid"]).write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")


def _sessions():
    out = []
    for p in STUDY.glob("*.json"):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _video(config, qid):
    s3 = RUNS / f"premade-{config}__{qid}" / "stage3.json"
    v = json.loads(s3.read_text(encoding="utf-8")).get("video") if s3.exists() else None
    return v if v and Path(v).exists() else None


def _query(qid):
    j = RUNS / f"premade-lw__s2s__{qid}" / "job.json"
    return json.loads(j.read_text(encoding="utf-8"))["spec"]["query"] if j.exists() else qid


@lru_cache(maxsize=1)
def _resources():
    raw = json.loads(baselines.RESOURCES.read_text(encoding="utf-8"))
    return {r["resource_id"]: r for r in (raw if isinstance(raw, list) else raw.get("resources", []))}


def _val(r, k):
    v = r.get(k)
    return "" if v is None or str(v).strip() in ("", "None", "nan") else _clean(str(v).strip())


def _clean(t):
    """Textbook text with (escaped) HTML from the PDF extraction -> plain text: <sup>2</sup> -> ^2, <sub>2</sub> -> _2."""
    t = html.unescape(html.unescape(t))
    t = re.sub(r"<sup>(.*?)</sup>", r"^\1", t, flags=re.S | re.I)
    t = re.sub(r"<sub>(.*?)</sub>", r"_\1", t, flags=re.S | re.I)
    return re.sub(r"</?[a-zA-Z][^>]{0,40}>", "", t)


def _short(t, n=180):
    """First sentence or two of a resource summary, for the compact list (the full text opens on tap)."""
    t = " ".join(t.split())
    if len(t) <= n:
        return t
    cut = t.rfind(". ", 0, n)
    return t[:cut + 1] if cut > 60 else t[:n].rsplit(" ", 1)[0] + "…"


@lru_cache(maxsize=64)
def _set(config, qid):
    """The explanation set behind one premade video: its Stage 1 resources, in order, as the participant reads them."""
    r1 = json.loads((RUNS / f"premade-{config}__{qid}" / "stage1.json").read_text(encoding="utf-8"))
    out, res = [], _resources()
    for rid in r1["selection"]["selected"]:
        r = res.get(rid, {})
        fig = rid.startswith("FIG")
        where = [f"Class {_val(r, 'class_no')}" if _val(r, "class_no") else "", _val(r, "subject"),
                 _val(r, "chapter_title"), _val(r, "section") if len(_val(r, "section")) <= 60 else "",
                 f"p. {_val(r, 'page_start')}" if _val(r, "page_start") else ""]
        text = (_val(r, "description") or _val(r, "summary")) if fig else (_val(r, "text") or _val(r, "summary"))
        out.append({"id": rid, "kind": "figure" if fig else "text", "source": " · ".join(w for w in where if w),
                    "text": text, "summary": _short(_val(r, "summary") or text), "caption": _val(r, "image_caption")})
    return tuple(out)


def _pick_pairs(used, n_each, exclude=()):
    """n_each pairs per comparison on distinct questions, least-answered (comparison, question) first."""
    counts = {}
    for s in _sessions():
        for p in s.get("pairs", []):
            counts[(p["comparison"], p["query_id"])] = counts.get((p["comparison"], p["query_id"]), 0) + 1
    taken, pairs = set(used), []
    for comp in random.sample(list(COMPARISONS), len(COMPARISONS)):
        cand = [q for q in QIDS if q not in taken and (comp, q) not in exclude
                and all(_video(c, q) for c in COMPARISONS[comp]["configs"])]
        cand.sort(key=lambda q: (counts.get((comp, q), 0), random.random()))
        for q in cand[:n_each]:
            taken.add(q)
            a, b = random.sample(COMPARISONS[comp]["configs"], 2)   # random sides
            pairs.append({"comparison": comp, "query_id": q, "query": _query(q), "a": a, "b": b})
    random.shuffle(pairs)
    return pairs


def _public(s):
    """The session as the page sees it: no configuration names until the participant has finished."""
    done = bool(s.get("finished"))
    pairs = []
    for k, p in enumerate(s["pairs"]):
        sa, sb = list(_set(p["a"], p["query_id"])), list(_set(p["b"], p["query_id"]))
        same = [x["id"] for x in sa] == [x["id"] for x in sb]
        e = {"k": k, "query": p["query"], "answered": str(k) in s["answers"],
             "sets": {"a": sa} if same else {"a": sa, "b": sb}, "same_set": same}
        if done and e["answered"]:   # an extra pair after finishing stays unlabeled until answered
            e.update(a=p["a"], b=p["b"], a_ours=p["a"] in OURS, b_ours=p["b"] in OURS,
                     comparison=COMPARISONS[p["comparison"]]["title"],
                     choices=s["answers"][str(k)]["choices"])
        pairs.append(e)
    return {"sid": s["sid"], "profile": s.get("profile"), "pairs": pairs, "finished": done,
            "n_extra": s.get("n_extra", 0)}


class StartIn(BaseModel):
    role: str
    level: str
    consent: bool


class AnswerIn(BaseModel):
    choices: dict[str, str]                # {question: a | b | same | cant}
    watch: dict[str, float] = {}           # fraction of each video played
    seconds: float | None = None


@router.post("/start")
def start(body: StartIn, request: Request):
    _open(request)
    if not body.consent:
        raise HTTPException(400, "consent is required")
    if body.role not in ROLES:
        raise HTTPException(400, f"role must be one of {sorted(ROLES)}")
    with _lock:
        sid = uuid.uuid4().hex[:12]
        s = {"sid": sid, "design": 2, "created": time.time(),
             "profile": {"role": body.role, "level": body.level[:40]},
             "pairs": _pick_pairs((), PAIRS_PER_SESSION // 2), "answers": {}, "finished": None}
        _save(s)
    return _public(s)


@router.get("/{sid}")
def get_session(sid: str):
    s = _load(sid)
    if s.get("design") != 2:
        raise HTTPException(404, "session from an earlier study design")
    return _public(s)


@router.post("/{sid}/answer/{k}")
def answer(sid: str, k: int, body: AnswerIn, request: Request):
    _open(request)
    if set(body.choices) != set(QUESTIONS) or not all(v in CHOICES for v in body.choices.values()):
        raise HTTPException(400, f"choices needs {list(QUESTIONS)}, each one of {list(CHOICES)}")
    with _lock:
        s = _load(sid)
        if not 0 <= k < len(s["pairs"]):
            raise HTTPException(404, "unknown pair")
        p = s["pairs"][k]
        s["answers"][str(k)] = {"choices": dict(body.choices),
                                "winner": {q: {"a": p["a"], "b": p["b"]}.get(v, v) for q, v in body.choices.items()},
                                "watch": {p[x]: round(float(body.watch.get(x, 0)), 3) for x in ("a", "b")},
                                "seconds": body.seconds, "time": time.time()}
        if len(s["answers"]) >= len(s["pairs"]):
            s["finished"] = s.get("finished") or time.time()
        _save(s)
    return _public(s)


@router.post("/{sid}/more")
def more(sid: str, request: Request):
    """One more pair after finishing (any question / comparison this participant has not seen yet)."""
    _open(request)
    with _lock:
        s = _load(sid)
        seen = {(p["comparison"], p["query_id"]) for p in s["pairs"]}
        extra = _pick_pairs((), 1, exclude=seen)
        if not extra:
            raise HTTPException(409, "you have seen every pair - thank you!")
        s["pairs"].append(extra[0])
        s["n_extra"] = s.get("n_extra", 0) + 1
        _save(s)
    return _public(s)


@router.get("/{sid}/video/{k}/{side}")
def video(sid: str, k: int, side: str):
    s = _load(sid)
    if side not in ("a", "b") or not 0 <= k < len(s["pairs"]):
        raise HTTPException(404, "unknown video")
    p = s["pairs"][k]
    v = _video(p[side], p["query_id"])
    if not v:
        raise HTTPException(404, "video missing")
    return FileResponse(v, media_type="video/mp4")


def sign_test(w, l):
    """Two-sided exact sign test (same / can't judge dropped)."""
    n = w + l
    if n == 0:
        return None
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


@router.get("/summary/all")
def summary():
    ss = [s for s in _sessions() if s.get("design") == 2]
    out = {"sessions": len(ss), "finished": sum(1 for s in ss if s.get("finished")), "roles": {}, "comparisons": {}}
    for s in ss:
        r = (s.get("profile") or {}).get("role", "?")
        out["roles"][r] = out["roles"].get(r, 0) + 1
    for comp, c in COMPARISONS.items():
        ours, theirs = c["configs"]
        per = {q: {"ours": 0, "baseline": 0, "same": 0, "cant": 0} for q in QUESTIONS}
        n = 0
        for s in ss:
            for k, a in s.get("answers", {}).items():
                if s["pairs"][int(k)]["comparison"] != comp:
                    continue
                n += 1
                for q, w in a["winner"].items():
                    per[q][{ours: "ours", theirs: "baseline"}.get(w, w)] += 1
        for q, d in per.items():
            w, l = d["ours"], d["baseline"]
            d["ours_win_rate"] = round(w / (w + l), 3) if w + l else None
            d["sign_test_p"] = sign_test(w, l)
        out["comparisons"][comp] = {"title": c["title"], "ours": c["labels"][0], "baseline": c["labels"][1], "n": n,
                                    "questions": per}
    return out
