"""
Stage 1 baseline OptiSet (SetCoRe repository, baselines/optiset.py) for the web app, with its LLM calls sent to the
vLLM server instead of loading Llama-3.1-70B-Instruct through Transformers (bf16, ~140 GB: does not fit the A100).

Only make_llm(), generate_one() and encode_resource_corpus() are replaced: the first two send the repository's
prompts to the AWQ 4-bit Llama-3.1-70B-Instruct on :8003 (same sampling and max_tokens), the third caches the BGE
corpus embeddings on disk (the corpus does not change; encoding it on the CPU takes minutes per run). Prompts,
parsing, retrieval and refinement are the repository's.

    CUDA_VISIBLE_DEVICES= set2story/venv/bin/python optiset_runner.py [optiset.py arguments]
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

REPO = os.getenv("OPTISET_REPO", "/mnt/storage/aryan/external/setcore-37E8/baselines")
sys.path.insert(0, REPO)
sys.path.insert(0, "/mnt/storage/aryan/VideoModule")

import optiset  # noqa: E402
from model_loader import VLLMClient  # noqa: E402

URL = os.getenv("OPTISET_LLM_URL", "http://127.0.0.1:8003/v1")
MODEL = os.getenv("OPTISET_LLM_MODEL", "Llama-3.1-70B-Instruct")
CACHE = Path(os.getenv("VX_CACHE", "/mnt/storage/aryan/VideoExplain/cache")) / "optiset"
_encode = optiset.encode_resource_corpus


def make_llm(model_name):
    print(f"LLM via vLLM: {MODEL} at {URL} (instead of Transformers {model_name})")
    return None, VLLMClient("optiset", URL, MODEL)


def generate_one(tokenizer, model, user_prompt, temperature, max_tokens):
    extra = {"top_p": 0.9} if temperature > 0 else {}
    return model.generate([{"role": "user", "content": user_prompt}], max_new_tokens=max_tokens,
                          temperature=temperature, **extra).strip()


def encode_resource_corpus(encoder, resources, batch_size, field_mode="all"):
    texts = [optiset.resource_text(r, field_mode) for r in resources]
    key = hashlib.sha1("\0".join([field_mode] + texts).encode("utf-8")).hexdigest()[:16]
    path = CACHE / f"corpus_{key}.npy"
    if path.exists():
        print(f"Resource embeddings from cache {path}")
        return np.load(path)
    emb = _encode(encoder, resources, batch_size, field_mode)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


optiset.make_llm = make_llm
optiset.generate_one = generate_one
optiset.encode_resource_corpus = encode_resource_corpus

if __name__ == "__main__":
    optiset.main()
