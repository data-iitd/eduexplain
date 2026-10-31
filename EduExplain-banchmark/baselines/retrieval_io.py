"""Shared I/O for the retrievers
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence


def require_local_model(spec: str, what: str) -> str:
    p = os.path.expanduser(spec)
    if os.path.exists(p):
        return p
    looks_like_path = spec.startswith(("/", "~", "./", "../"))
    offline = os.environ.get("HF_HUB_OFFLINE") == "1" or \
        os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    if looks_like_path:
        raise SystemExit(
            f"{what}: {p} does not exist.\n  Download it on a node WITH "
            f"network and point --model / --pretrained at the directory.")
    if offline:
        raise SystemExit(
            f"{what}: {spec!r} is a Hub id, not a local path, and "
            f"HF_HUB_OFFLINE=1.\n  Download it first:\n    hf download "
            f"{spec} --local-dir <dir>\n  then pass that directory.")
    return spec


def first(d: Dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def index_text(resource: Dict) -> tuple:
    for key in ("summary", "description", "image_caption"):
        v = resource.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip(), key
    v = (resource.get("text") or "").strip()[:2000]
    if v:
        return v, "text"
    return str(resource.get("resource_id", "")), "resource_id"


def query_text(item: Dict) -> str:
    return str(first(item, "query_text", "query", "text", default=""))


def load_corpus(gold_path: str, resources_path: str):
    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)
    if isinstance(gold, dict):
        gold = first(gold, "queries", "data", default=[])
    with open(resources_path, encoding="utf-8") as f:
        resources = json.load(f)
    if isinstance(resources, dict):
        resources = resources.get("resources", resources)
    return gold, resources


def report_texts(resources: Sequence[Dict]) -> List[str]:
    texts, sources = [], Counter()
    for r in resources:
        t, src = index_text(r)
        texts.append(t)
        sources[src] += 1
    print(f"Loaded {len(resources)} resources. Text drawn from:")
    for k, v in sources.most_common():
        print(f"  {k:<14s} {v:>7d}  ({v / len(resources):.1%})")
    empty = sum(1 for t in texts if not t.strip())
    if empty:
        print(f"  [WARN] {empty} resources have no usable text at all")
    print(f"mean characters per resource: "
          f"{sum(len(t) for t in texts) / max(1, len(texts)):.0f}")
    return texts


def is_figure(resource: Dict) -> bool:
    t = str(first(resource, "resource_type", "type", default="")).lower()
    return t in {"figure", "image", "diagram", "illustration", "photo", "table"}


def write_candidates(path: str, gold: Sequence[Dict], qtexts: Sequence[str],
                     resources: Sequence[Dict], order: Sequence[Sequence[int]],
                     scores: Sequence[Sequence[float]], retriever: str) -> None:
    """One record per query, candidates in rank order.

    `order[i]` holds resource INDICES for query i, best first; `scores[i]` the
    matching scores. Keeping them parallel rather than re-sorting here means
    the rank the retriever intended is the rank the file records.
    """
    out = []
    for i, item in enumerate(gold):
        retrieved = []
        for rank, (j, sc) in enumerate(zip(order[i], scores[i]), start=1):
            res = resources[int(j)]
            retrieved.append({
                "prompt_index": rank,
                "resource_id": res.get("resource_id"),
                "score": float(sc),
                "resource_type": first(res, "resource_type", "type"),
                "concepts": res.get("concepts", []),
                "resource": res,
            })
        out.append({"query_id": item["query_id"], "query": qtexts[i],
                    "retriever": retriever,
                    "retrieved_resources": retrieved})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    n = len(out[0]["retrieved_resources"]) if out else 0
    print(f"\nSaved: {p}\nQueries: {len(out)}   candidates per query: {n}   "
          f"retriever: {retriever}")
