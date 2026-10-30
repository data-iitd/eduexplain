"""
OptiSet (SetCoRe repository, baselines/optiset.py) with its LLM calls sent to the vLLM server instead of loading
Llama-3.1-70B-Instruct through Transformers (bf16, ~140 GB: does not fit the single A100 40GB).

Only make_llm() and generate_one() are replaced; prompts, parsing, retrieval (BGE-large, CPU), sampling
(temperature, top_p 0.9 when sampling) and max_tokens are the repository's. The served model is the AWQ 4-bit
Llama-3.1-70B-Instruct on :8003 (also used by SetCoRe-LLM); that deviation is reported in the paper.

    CUDA_VISIBLE_DEVICES= set2story/venv/bin/python optiset_vllm.py [optiset.py arguments]
"""
import os
import sys

REPO = "/mnt/storage/aryan/external/setcore-37E8/baselines"
sys.path.insert(0, REPO)
sys.path.insert(0, "/mnt/storage/aryan/VideoModule")

import optiset  # noqa: E402
from model_loader import VLLMClient  # noqa: E402

URL = os.getenv("OPTISET_LLM_URL", "http://127.0.0.1:8003/v1")
MODEL = os.getenv("OPTISET_LLM_MODEL", "Llama-3.1-70B-Instruct")


def make_llm(model_name):
    print(f"LLM via vLLM: {MODEL} at {URL} (instead of Transformers {model_name})")
    return None, VLLMClient("optiset", URL, MODEL)


def generate_one(tokenizer, model, user_prompt, temperature, max_tokens):
    extra = {"top_p": 0.9} if temperature > 0 else {}
    return model.generate([{"role": "user", "content": user_prompt}], max_new_tokens=max_tokens,
                          temperature=temperature, **extra).strip()


optiset.make_llm = make_llm
optiset.generate_one = generate_one

if __name__ == "__main__":
    optiset.main()
