"""Batched inference backends shared by every method.

Storyboards are realised concurrently by many threads within a single
process. Each thread calls the LLM/NLI as if it were alone, and a
DynamicBatcher merges the concurrent calls into one GPU forward pass. The
pipeline code stays sequential per storyboard, while the GPU sees large
batches.
"""
import threading
import queue
from concurrent.futures import Future
from typing import Callable, List, Sequence, Tuple

import numpy as np


class DynamicBatcher:
    def __init__(self, fn: Callable[[list], list], max_batch: int = 32, max_wait: float = 0.05,
                 name: str = "batcher"):
        self.fn, self.max_batch, self.max_wait = fn, max_batch, max_wait
        self.q: "queue.Queue[Tuple[object, Future]]" = queue.Queue()
        threading.Thread(target=self._loop, name=name, daemon=True).start()

    def _loop(self):
        while True:
            items = [self.q.get()]
            try:
                while len(items) < self.max_batch:
                    items.append(self.q.get(timeout=self.max_wait))
            except queue.Empty:
                pass
            try:
                outs = self.fn([x for x, _ in items])
                for (_, fut), o in zip(items, outs):
                    fut.set_result(o)
            except Exception as e:  # propagate to every waiting caller
                for _, fut in items:
                    if not fut.done():
                        fut.set_exception(e)

    def submit_many(self, xs: Sequence) -> list:
        futs = []
        for x in xs:
            f = Future()
            self.q.put((x, f))
            futs.append(f)
        return [f.result() for f in futs]

    def submit(self, x):
        return self.submit_many([x])[0]


class HFChatLLM:
    """Instruction-tuned HF causal LM. Greedy decoding, prompts are wrapped in
    the model's chat template, generation batched with left padding."""

    def __init__(self, model_path: str, device: str = "cuda:0", max_batch: int = 24,
                 dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=getattr(torch, dtype), device_map={"": device})
        self.model.eval()
        self.device = device
        self.calls = 0
        self._lock = threading.Lock()
        self.batcher = DynamicBatcher(self._run, max_batch=max_batch, name="llm")

    def _chat(self, prompt: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

    def _run(self, items: List[Tuple[str, int]]) -> List[str]:
        out = [None] * len(items)
        by_len = {}
        for i, (p, n) in enumerate(items):
            by_len.setdefault(n, []).append(i)
        for n_new, idxs in by_len.items():
            texts = [self._chat(items[i][0]) for i in idxs]
            enc = self.tok(texts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=n_new, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for i, row in zip(idxs, new):
                out[i] = self.tok.decode(row, skip_special_tokens=True)
        with self._lock:
            self.calls += len(items)
        return out

    def __call__(self, prompt: str, max_new_tokens: int = 400) -> str:
        return self.batcher.submit((prompt, max_new_tokens))


class BatchedNLI:
    """nli-deberta-v3-large cross-encoder; returns softmax probabilities
    (contradiction, entailment, neutral) per (premise, hypothesis) pair."""
    LABELS = ("contradiction", "entailment", "neutral")

    def __init__(self, model_path: str, device: str = "cuda:0", max_batch: int = 256,
                 max_cache: int = 500_000):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_path, device=device, local_files_only=True)
        self.cache, self.max_cache = {}, max_cache
        self._lock = threading.Lock()
        self.batcher = DynamicBatcher(self._run, max_batch=max_batch, max_wait=0.01, name="nli")

    def _run(self, pairs):
        logits = np.asarray(self.model.predict(list(pairs), batch_size=len(pairs),
                                               show_progress_bar=False), dtype=np.float64)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return list(e / e.sum(axis=1, keepdims=True))

    def probs(self, pairs: Sequence[Tuple[str, str]]) -> List[np.ndarray]:
        with self._lock:
            got = {p: self.cache[p] for p in set(pairs) if p in self.cache}
        missing = [p for p in dict.fromkeys(pairs) if p not in got]
        if missing:
            res = self.batcher.submit_many(missing)
            got.update(zip(missing, res))
            with self._lock:
                if len(self.cache) > self.max_cache:
                    self.cache.clear()
                self.cache.update(zip(missing, res))
        return [got[p] for p in pairs]

    def entail(self, pairs) -> np.ndarray:
        return np.array([p[1] for p in self.probs(pairs)]) if pairs else np.zeros(0)

    def contra(self, pairs) -> np.ndarray:
        return np.array([p[0] for p in self.probs(pairs)]) if pairs else np.zeros(0)
