"""
Stage 1 - SetCoRe-LLM resource retrieval + gold-set selection, as a per-query service around /mnt/storage/aryan/setCore.

setCore is a batch benchmark pipeline (BGE pool -> lightweight heads -> reduce -> LLM select). Here the same steps
run for ONE free-text query:
  1. retrieve   BGE-large-en-v1.5 (setCore's encoder; ONNX on CPU, no GPU memory) over every resource's
                index_text(); corpus embeddings are computed once and cached. Top-K pool (K=100, setCore's K).
  2. shortlist  B = min(K, ceil(2 * n)) top candidates (setCore's budget rule; n = target set size).
  3. select     setCore's own LLM-select prompt / parse_ids / validate / repair / fallback (scx_llm_select.py),
                sent to the shared vLLM generator (Qwen3.8-27B) instead of the in-process Llama-3.1-70B, which
                cannot fit the single 40 GB A100.
Manual mode (the UI's checkboxes): candidates() returns the pool without an LLM call, apply_manual() turns the
user's ticked ids into the gold set. Degraded modes are explicit (result["mode"]): no BGE model -> keyword (BM25) retrieval; no LLM endpoint ->
top-n of the shortlist (setCore's own fallback).
"""
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

SETCORE = Path(os.getenv("SETCORE_DIR", "/mnt/storage/aryan/setCore"))
RESOURCES = SETCORE / "all_resources_unique_ids_clean1_merged.json"
GOLD = SETCORE / "gold_sets_final_validated_new_merged.json"
BGE_DIR = Path(os.getenv("BGE_DIR", str(SETCORE / "models" / "bge-large-en-v1.5")))
CACHE = Path(os.getenv("VX_CACHE", "/mnt/storage/aryan/VideoExplain/cache"))
sys.path.append(str(SETCORE))   # append (setCore has an icra/select.py that would shadow the stdlib module)

from scx_common import index_text  # noqa: E402
import scx_llm_select as sel  # noqa: E402


class BGEEncoder:
    """BAAI/bge-large-en-v1.5 via its bundled ONNX export: CLS pooling + L2 norm (what sentence-transformers does)."""

    def __init__(self, model_dir=BGE_DIR, threads=32):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.tok.enable_truncation(512)
        self.tok.enable_padding()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(Path(model_dir) / "onnx" / "model.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def encode(self, texts, batch=32):
        out = []
        for k in range(0, len(texts), batch):
            enc = self.tok.encode_batch(list(texts[k:k + batch]))
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            if "token_type_ids" in self.inputs:
                feeds["token_type_ids"] = np.zeros_like(ids)
            cls = self.sess.run(None, feeds)[0][:, 0]
            out.append(cls / np.linalg.norm(cls, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float32)


class BM25:
    """Keyword fallback when the BGE model is not available (dummy mode)."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.toks = [re.findall(r"[a-z0-9]+", d.lower()) for d in docs]
        self.df = Counter(t for d in self.toks for t in set(d))
        self.avg = sum(map(len, self.toks)) / max(len(self.toks), 1)
        self.n, self.k1, self.b = len(docs), k1, b
        self.tf = [Counter(d) for d in self.toks]

    def scores(self, query):
        q = re.findall(r"[a-z0-9]+", query.lower())
        out = np.zeros(self.n, dtype=np.float32)
        for t in set(q):
            if t not in self.df:
                continue
            idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(self.toks[i]) / self.avg))
        return out


class Stage1:
    def __init__(self, llm=None, log=print):
        self.log = log
        self.llm = llm
        self.resources = json.loads(RESOURCES.read_text(encoding="utf-8"))
        self.rmap = {r["resource_id"]: r for r in self.resources}
        self.ids = [r["resource_id"] for r in self.resources]
        self.texts = [index_text(r) for r in self.resources]
        self.default_n = self._typical_gold_size()
        self.encoder, self.R, self.bm25 = None, None, None
        self.reducer = None   # setcore_adapter.SetCoreReducer, created on first use
        if (BGE_DIR / "onnx" / "model.onnx").exists():
            self.encoder = BGEEncoder()
            self.R = self._corpus_embeddings()
        else:
            self.log(f"! BGE model missing at {BGE_DIR}: Stage 1 uses keyword (BM25) retrieval (dummy mode)")
            self.bm25 = BM25(self.texts)

    def _typical_gold_size(self):
        try:
            gold = json.loads(GOLD.read_text(encoding="utf-8"))
            sizes = sorted(len(gs.get("resources_used", [])) for q in gold for gs in q.get("gold_sets", []))
            return max(1, sizes[len(sizes) // 2]) if sizes else 4
        except (OSError, json.JSONDecodeError):
            return 4

    def _corpus_embeddings(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(("|".join(self.ids) + "|".join(t[:80] for t in self.texts)).encode()).hexdigest()[:12]
        path = CACHE / f"bge_corpus_{key}.npy"
        if path.exists():
            return np.load(path)
        t = time.time()
        self.log(f"encoding {len(self.texts)} resources with BGE-large (once; cached in {path}) ...")
        R = self.encoder.encode(self.texts)
        np.save(path, R)
        self.log(f"corpus embeddings ready in {time.time() - t:.0f}s")
        return R

    def _reduce(self, query, pool):
        """setCore stages 2-4 (lightweight model -> n_hat, coverage-greedy reduce -> D_q) via setcore_adapter's
        persistent worker. None (and Stage 1 uses top-B with the default n) when it is unavailable."""
        if self.reducer is False:
            return None
        try:
            if self.reducer is None:
                from setcore_adapter import SetCoreReducer
                self.reducer = SetCoreReducer(log=self.log)
            return self.reducer.shortlist(query, pool)
        except Exception as exc:
            self.log(f"! setCore reduce unavailable ({exc}); shortlist = top-B by BGE score, n = {self.default_n}")
            self.reducer = False
            return None

    # ------------------------------------------------------------------ per query
    def retrieve(self, query, k=100):
        if self.encoder is not None:
            s = self.R @ self.encoder.encode([query])[0]
        else:
            s = self.bm25.scores(query)
        top = np.argsort(-s)[:k]
        return [{"resource_id": self.ids[i], "score": round(float(s[i]), 4),
                 "resource_type": self.rmap[self.ids[i]].get("resource_type"),
                 "text": self.texts[i][:600]} for i in top]

    def select(self, query, pool, n, shortlist=None):
        """setCore's LLM selection over the shortlist (exactly n ids); falls back to the top-n of the shortlist.
        shortlist: setCore's coverage-greedy reduce D_q (setcore_adapter); default top-B by BGE score."""
        B = min(len(pool), max(n, math.ceil(2 * n)))
        short = list(shortlist) if shortlist else [p["resource_id"] for p in pool[:B]]
        B = len(short)
        info = {"shortlist": short, "n": n, "B": B}
        if self.llm is None:
            info.update(selected=short[:n], parse_path="no_llm", fell_back=True)
            return info
        prompt = sel.PROMPT.format(question=query, guidance=sel.guidance_block("size", 0, 0, [], False),
                                   context=sel.build_context(short, self.rmap), size_text=sel.size_text("exact", n, n),
                                   example=sel.example_json(short, n))
        t = time.time()
        try:
            raw = self.llm.generate([{"role": "user", "content": prompt}], max_new_tokens=512, temperature=0.0)
        except Exception as exc:
            self.log(f"! Stage 1 LLM unavailable ({exc}); using setCore's fallback (top-n of the shortlist)")
            info.update(selected=short[:n], parse_path="llm_error", fell_back=True, error=str(exc)[:300])
            return info
        ids, how = sel.parse_ids(raw, short)
        ids, problem = sel.validate(ids, short, "exact", n, n)
        repaired = False
        if problem:   # setCore's single repair round
            rep = sel.REPAIR.format(problem=problem, ids="\n".join(short),
                                    size_text_short=f"exactly {n} IDs", example=sel.example_json(short, n))
            try:
                raw2 = self.llm.generate([{"role": "user", "content": prompt}, {"role": "assistant", "content": raw},
                                          {"role": "user", "content": rep}], max_new_tokens=128, temperature=0.0)
                ids, how = sel.parse_ids(raw2, short)
                (ids, problem), repaired = sel.validate(ids, short, "exact", n, n), True
            except Exception:
                pass
        fell_back = bool(problem)
        chosen = short[:n] if fell_back else list(ids)[:n]
        info.update(selected=chosen, parse_path=how, repaired=repaired, fell_back=fell_back, problem=problem,
                    llm_response=raw[:2000], llm_sec=round(time.time() - t, 1))
        return info

    def _candidates(self, query, n=None, k=100):
        """BGE pool + setCore reduce (shortlist D_q and n_hat), shared by the automatic and the manual mode."""
        pool = self.retrieve(query, k)
        reduced = self._reduce(query, pool)
        shortlist = None
        if reduced:
            n = int(n or reduced["n_hat"])
            shortlist = list(reduced["shortlist"])
            for p in pool:   # a user-chosen n larger than n_hat: top up D_q to B = 2n in retrieval order
                if len(shortlist) >= min(len(pool), math.ceil(2 * n)):
                    break
                if p["resource_id"] not in shortlist:
                    shortlist.append(p["resource_id"])
        else:
            n = int(n or self.default_n)
        return pool, reduced, n, shortlist

    def _gold_set(self, query, query_id, ids):
        resources_used = []
        for rid in ids:
            r = self.rmap[rid]
            resources_used.append({k2: r.get(k2) for k2 in ("resource_id", "resource_type", "summary", "description",
                                                            "image_caption", "text", "document_id", "concepts",
                                                            "image_path", "chapter_title", "subject", "class_no")})
        # gold-set record in setCore's gold format (input to Stage 2)
        return {"query_id": query_id, "query": query, "gold_sets": [{"resources_used": resources_used}]}

    def run(self, query, n=None, k=100, query_id=None):
        t = time.time()
        pool, reduced, n, shortlist = self._candidates(query, n, k)
        choice = self.select(query, pool, n, shortlist)
        if reduced:
            choice.update(reducer="setcore-reduce", n_hat=reduced["n_hat"],
                          reducer_info={k2: reduced.get(k2) for k2 in ("B", "profile", "lw_selected", "seconds")})
        mode = (("bge" if self.encoder is not None else "bm25-dummy") + ("+reduce" if reduced else "+topB")
                + ("+llm" if not choice.get("fell_back") else "+fallback"))
        query_id = query_id or f"Q_{int(time.time())}"
        return {
            "stage": 1, "mode": mode, "query": query, "query_id": query_id,
            "retrieved": pool[:25], "pool_size": len(pool), "selection": choice,
            "gold_set": self._gold_set(query, query_id, choice["selected"]),
            "seconds": round(time.time() - t, 1),
        }

    # ------------------------------------------------------------------ manual mode
    def candidates(self, query, n=None, k=100, query_id=None):
        """Manual mode, step 1: the whole retrieval pool for the user to tick, no LLM call. The setCore shortlist
        and n_hat are returned as suggestions only (nothing is pre-selected)."""
        t = time.time()
        pool, reduced, n, shortlist = self._candidates(query, n, k)
        return {
            "stage": 1, "mode": ("bge" if self.encoder is not None else "bm25-dummy") + "+manual",
            "query": query, "query_id": query_id or f"Q_{int(time.time())}",
            "retrieved": pool, "pool_size": len(pool), "awaiting_selection": True,
            "selection": {"shortlist": shortlist or [p["resource_id"] for p in pool[:math.ceil(2 * n)]],
                          "selected": [], "n": n, "suggested_n": n, "parse_path": "manual",
                          **({"n_hat": reduced["n_hat"]} if reduced else {})},
            "seconds": round(time.time() - t, 1),
        }

    def apply_manual(self, cand, ids):
        """Manual mode, step 2: the user's ticked resources (in the order given) become the gold set."""
        ids = list(dict.fromkeys(ids))
        unknown = [i for i in ids if i not in self.rmap]
        if not ids or unknown:
            raise ValueError("select at least one resource" if not ids else f"unknown resource ids: {unknown}")
        r1 = dict(cand, awaiting_selection=False)
        r1["selection"] = dict(cand["selection"], selected=ids, n=len(ids), fell_back=False)
        r1["gold_set"] = self._gold_set(cand["query"], cand["query_id"], ids)
        return r1
