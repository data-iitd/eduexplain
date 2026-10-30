#!/usr/bin/env python3
"""Shared loading, parsing and metric code for SetCoRe-X.

Everything here is lifted from the icra tools so that SetCoRe-X scores
identically to the first paper. In particular:

  * GoldSet P, R and F1 are each maximised INDEPENDENTLY over the query's gold
    sets (the benchmark spec), and `best_single_metrics` additionally reports
    the P/R of the one gold set that maximises F1, which is what
    tools_consistent_eval.py reports.
  * Core Concept Coverage is defined only where a query has >= 2 gold sets and
    a non-empty core at tau; it is averaged over that subset, and the subset
    size is always printed so two tables are never compared over different
    denominators.
  * index_text() is byte-identical to icra's Resource.index_text() and to the
    one in baseline_setr.py / baseline_optiset.py, so every stage reads the
    same single field.
  * resolve() expands "~" itself. A quoted "~/runs/x_s*" is expanded by
    neither the shell nor glob, and would silently match nothing.
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------- loading


def load_json(path: str) -> Any:
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        return json.load(f)


def load_records(path: str) -> List[Dict]:
    """A JSON list, a JSON object wrapping one, or JSONL."""
    path = os.path.expanduser(path)
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        try:
            obj = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(l) for l in f if l.strip()]
    if isinstance(obj, list):
        return obj
    for k in ("results", "predictions", "queries", "data", "records"):
        if isinstance(obj.get(k), list):
            return obj[k]
    # A JSONL file holding exactly ONE record parses as a bare dict.
    if any(k in obj for k in ("query_id", "id")):
        return [obj]
    raise SystemExit(f"{path}: no list of records found")


def resolve(spec: str) -> List[str]:
    """A directory, a file, or a glob over either -> prediction file paths."""
    spec = os.path.expanduser(spec)
    out: List[str] = []
    for p in sorted(glob.glob(spec)) or ([spec] if os.path.exists(spec) else []):
        if os.path.isdir(p):
            for name in ("predictions.jsonl", "predictions.json",
                         "shortlists.json"):
                c = os.path.join(p, name)
                if os.path.exists(c):
                    out.append(c)
                    break
        else:
            out.append(p)
    return out


def expand_tildes(ns) -> None:
    """Apply expanduser to every string / list-of-string attribute of a Namespace."""
    for k, v in list(vars(ns).items()):
        if isinstance(v, str) and v.startswith("~"):
            setattr(ns, k, os.path.expanduser(v))
        elif isinstance(v, list):
            setattr(ns, k, [os.path.expanduser(x) if isinstance(x, str)
                            and x.startswith("~") else x for x in v])


# ------------------------------------------------------------------ resources

ALL_FIELDS = [
    ("TYPE", "resource_type"),
    ("SUBJECT", "subject"),
    ("CHAPTER", "chapter_title"),
    ("SUMMARY", "summary"),
    ("DESCRIPTION", "description"),
    ("IMAGE CAPTION", "image_caption"),
    ("TEXT", "text"),
]

VISUAL_TYPES = {"figure", "diagram", "table", "image", "illustration", "photo",
                "chart", "graph", "picture", "visual"}


def cfield(c: Dict, key: str, default=None):
    """Read a field from a candidate dict, falling back to its nested
    'resource' sub-dict. The pool files nest most metadata one level down, and
    reading only the top level silently yields None for resource_type -- which
    is what made Modality-Aware MMR identical to plain MMR."""
    v = c.get(key)
    if v in (None, "", []):
        r = c.get("resource")
        if isinstance(r, dict):
            v = r.get(key, default)
    return default if v in (None, "") else v


def index_text(resource: Dict) -> str:
    """icra's Resource.index_text(): the FIRST NON-EMPTY of summary ->
    description -> image_caption, else text[:2000], else the resource_id.

    One field, not a concatenation, and no resource_type / subject /
    chapter_title. This is the only representation the lightweight selector
    may read, so it is the one that keeps the LLM stage comparable with it.
    """
    for key in ("summary", "description", "image_caption"):
        v = resource.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    v = (resource.get("text") or "").strip()[:2000]
    return v or str(resource.get("resource_id", ""))


def index_text_c(cand: Dict) -> str:
    """index_text() for a candidate dict whose fields may be nested."""
    for key in ("summary", "description", "image_caption"):
        v = cfield(cand, key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    v = str(cfield(cand, "text", "") or "").strip()[:2000]
    return v or str(cfield(cand, "resource_id", "") or "")


def is_visual(type_str: Optional[str]) -> bool:
    return str(type_str or "").strip().lower() in VISUAL_TYPES


# ---------------------------------------------------------------- predictions

PRED_KEYS = ("predicted_resource_ids", "selected", "predicted_set",
             "predicted", "resources_used", "prediction", "final_set",
             "shortlist")


def resource_ids(x: Any) -> List[str]:
    out: List[str] = []
    if x is None:
        return out
    if isinstance(x, dict):
        for k in ("resources_used", "resources", "resource_ids", "selected"):
            if k in x:
                return resource_ids(x[k])
        return out
    if isinstance(x, str):
        return [x]
    for item in x:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            rid = item.get("resource_id") or item.get("id")
            if rid:
                out.append(str(rid))
    return out


def prediction_of(rec: Dict, key: Optional[str] = None) -> List[str]:
    if key:
        return resource_ids(rec.get(key))
    for k in PRED_KEYS:
        if k in rec:
            ids = resource_ids(rec[k])
            if ids or k in ("predicted_resource_ids", "selected", "shortlist"):
                return ids
    return []


def dedupe(ids: Sequence[str]) -> List[str]:
    out, seen = [], set()
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ----------------------------------------------------------------- gold / queries


def gold_sets_of(q: Dict) -> List[List[str]]:
    out = []
    for g in (q.get("gold_sets") or q.get("explanation_sets") or []):
        ids = resource_ids(g if not isinstance(g, dict) else
                           (g.get("resources") or g.get("resources_used")
                            or g.get("resource_ids") or []))
        ids = [str(i) for i in ids if i]
        if ids:
            out.append(ids)
    return out


def norm_text(s: Optional[str]) -> str:
    return " ".join(str(s or "").lower().split())


def query_text_of(d: Dict) -> str:
    for k in ("query", "query_text", "text", "question"):
        if d.get(k):
            return str(d[k])
    return ""


def load_queries(path: str) -> List[Dict]:
    q = load_json(path)
    if isinstance(q, dict):
        q = q.get("queries", q)
    return q


def query_map(queries: Sequence[Dict]) -> Dict[str, Dict]:
    return {str(q.get("query_id") or q.get("id")): q for q in queries}


def load_concepts(resources_path: str) -> Dict[str, Set[str]]:
    res = load_json(resources_path)
    if isinstance(res, dict):
        res = res.get("resources", res)
    return {str(r.get("resource_id")): set(r.get("concepts") or []) for r in res}


def load_resource_map(resources_path: str) -> Dict[str, Dict]:
    res = load_json(resources_path)
    if isinstance(res, dict):
        res = res.get("resources", res)
    return {str(r.get("resource_id")): r for r in res}


# -------------------------------------------------------------------- metrics


def prf(pred: Set[str], gold: Set[str]) -> Tuple[float, float, float]:
    """Dice F1, which within ONE gold set equals the harmonic mean of P and R."""
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    i = len(pred & gold)
    return i / len(pred), i / len(gold), 2 * i / (len(pred) + len(gold))


def best_independent_metrics(pred: Set[str],
                             gold_sets: Sequence[Set[str]]) -> Dict[str, float]:
    """P, R, F1 each maximised INDEPENDENTLY over gold sets (benchmark spec)."""
    bp = br = bf = 0.0
    for g in gold_sets:
        p, r, f = prf(pred, g)
        bp, br, bf = max(bp, p), max(br, r), max(bf, f)
    return {"precision": bp, "recall": br, "f1": bf}


def best_single_metrics(pred: Set[str], gold_sets: Sequence[Set[str]]
                        ) -> Dict[str, float]:
    """Pick the ONE gold set with the highest F1 and report its P, R and F1.

    Reported alongside the independent maxima because they answer different
    questions: the independent form is the benchmark's number, this one is
    internally consistent (its P, R and F1 come from a single gold set, so F1
    always lies between P and R for that query).
    """
    best = (-1.0, 0.0, 0.0, -1)
    for i, g in enumerate(gold_sets):
        p, r, f = prf(pred, g)
        if f > best[0]:
            best = (f, p, r, i)
    f, p, r, i = best
    return {"precision": p, "recall": r, "f1": max(0.0, f), "gold_idx": i}


def core_concepts(gold_concept_sets: Sequence[Set[str]], tau: float) -> Set[str]:
    if not gold_concept_sets:
        return set()
    n = len(gold_concept_sets)
    cnt: Dict[str, int] = defaultdict(int)
    for cs in gold_concept_sets:
        for c in cs:
            cnt[c] += 1
    return {c for c, k in cnt.items() if k / n >= tau}


def set_concept_coverage(pred_concepts: Set[str],
                         gold_concept_sets: Sequence[Set[str]]) -> float:
    """Best coverage of any ONE gold set's concept set."""
    best = 0.0
    for gc in gold_concept_sets:
        if gc:
            best = max(best, len(pred_concepts & gc) / len(gc))
    return best


def gold_concept_sets_of(gold: Sequence[Sequence[str]],
                         con: Dict[str, Set[str]]) -> List[Set[str]]:
    return [set().union(*[con.get(x, set()) for x in g]) if g else set()
            for g in gold]


# ------------------------------------------------------- LLM output parsing

FINAL_RE = re.compile(r"###\s*Final\s+Selection\s*:\s*(.*)",
                      re.IGNORECASE | re.DOTALL)


def parse_selection(response: str, n_candidates: int, max_items: int,
                    stats: Optional[Counter] = None,
                    tag: str = "") -> List[int]:
    """Parse "### Final Selection: [2] [7]" into [2, 7]. 1-based, validated.

    Two fallbacks, both counted rather than silent:

    1. No header -> the LAST bracket-bearing line. Scanning the whole response
       harvests the Step-2 reasoning indices instead of the Step-3 answer: on
       "Req 1 -> [7] and [12] ... the clearest combination is [3], [9], [12]"
       that returns [7, 12, 3, 9].
    2. No brackets anywhere -> bare integers, but only on ONE line. Run over
       the whole response this is actively harmful: the prompt itself contains
       "Step 1." and "Step 2.", and 1 and 2 are valid candidate indices, so a
       response that declined to answer parses as [1, 2].
    """
    m = FINAL_RE.search(response or "")
    if m:
        text = m.group(1)
        text = re.split(r"\n\s*(?:Step|Explanation|Reasoning|Note)\s*[:.]",
                        text, maxsplit=1, flags=re.IGNORECASE)[0]
        path = "header"
    else:
        lines = [l for l in (response or "").strip().splitlines()
                 if re.search(r"\[\d+\]", l)]
        if lines:
            text, path = lines[-1], "fallback_bracket"
        else:
            body = (response or "").strip().splitlines()
            text, path = (body[-1] if body else ""), "fallback_bare"

    nums = re.findall(r"\[(\d+)\]", text)
    if not nums:
        nums = re.findall(r"\b(\d+)\b", text)

    out, seen = [], set()
    for s in nums:
        i = int(s)
        if not (1 <= i <= n_candidates) or i in seen:
            continue
        seen.add(i)
        out.append(i)
        if max_items and len(out) >= max_items:
            break

    if stats is not None:
        stats[f"{tag}{path}"] += 1
        if not out:
            stats[f"{tag}empty"] += 1
    return out


# -------------------------------------------------------------------- display


def ms(vals: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], None
    return st.mean(vals), st.stdev(vals)


def cell(mean: Optional[float], sd: Optional[float], nd: int = 4) -> str:
    if mean is None:
        return "    --   "
    return f"{mean:.{nd}f}" if sd is None else f"{mean:.{nd}f}+/-{sd:.{nd}f}"


def write_json(obj: Any, path: str, label: str = "") -> str:
    path = os.path.expanduser(path)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"{label or 'wrote'} -> {path}")
    return path
