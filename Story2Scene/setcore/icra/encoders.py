"""The frozen encoder E_phi.

One encoder is used for everything -- first-stage scoring, the role head's
input h_q, concept tagging, and query-to-vocabulary matching -- so no extra
forward pass is introduced beyond first-stage retrieval. Embeddings are
L2-normalised, so cosine == dot product everywhere downstream.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Dict, List, Sequence

import numpy as np


def l2norm(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


class BaseEncoder:
    name: str = "base"
    dim: int = 0

    def _encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class SentenceTransformerEncoder(BaseEncoder):
    """BGE-M3 (or any sentence-transformers model)."""

    def __init__(self, name: str = "BAAI/bge-m3", batch_size: int = 32,
                 max_seq_len: int = 512):
        from sentence_transformers import SentenceTransformer  # lazy
        self.name = name
        self.model = SentenceTransformer(name)
        self.model.max_seq_length = max_seq_len
        self.batch_size = batch_size
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        v = self.model.encode(list(texts), batch_size=self.batch_size,
                              convert_to_numpy=True, normalize_embeddings=False,
                              show_progress_bar=False)
        return l2norm(np.asarray(v, dtype=np.float32))


class HashEncoder(BaseEncoder):
    """Deterministic offline stub: hashed char-3gram + word unigram bag.

    Not semantically meaningful. Only for smoke-testing the pipeline without
    downloading a model. Never report numbers from this.
    """

    def __init__(self, dim: int = 256):
        self.name = f"hash-{dim}"
        self.dim = dim

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t = (t or "").lower()
            toks = re.findall(r"[a-z0-9]+", t)
            grams = toks + [t[j:j + 3] for j in range(max(0, len(t) - 2))]
            for g in grams:
                h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
        return l2norm(out)


class CachedEncoder:
    """Disk-cached wrapper. Key = sha1(model_name || text)."""

    def __init__(self, base: BaseEncoder, cache_dir: str = ".cache"):
        self.base = base
        self.dim = base.dim
        self.name = base.name
        os.makedirs(cache_dir, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", base.name)
        self.path = os.path.join(cache_dir, f"emb_{safe}.npz")
        self._store: Dict[str, np.ndarray] = {}
        if os.path.exists(self.path):
            with np.load(self.path) as z:
                self._store = {k: z[k] for k in z.files}
        self._dirty = False

    @staticmethod
    def _key(model: str, text: str) -> str:
        return hashlib.sha1(f"{model}\x00{text}".encode("utf-8")).hexdigest()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        keys = [self._key(self.name, t) for t in texts]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._store]
        if missing_idx:
            fresh = self.base._encode([texts[i] for i in missing_idx])
            for j, i in enumerate(missing_idx):
                self._store[keys[i]] = fresh[j]
            self._dirty = True
        return np.stack([self._store[k] for k in keys]).astype(np.float32)

    def flush(self) -> None:
        if self._dirty:
            np.savez_compressed(self.path, **self._store)
            self._dirty = False


def build_encoder(cfg) -> CachedEncoder:
    if cfg.encoder_backend == "hash":
        base: BaseEncoder = HashEncoder(dim=cfg.hash_dim)
    else:
        base = SentenceTransformerEncoder(cfg.encoder_name, cfg.batch_size,
                                          cfg.max_seq_len)
    return CachedEncoder(base, cfg.cache_dir)
