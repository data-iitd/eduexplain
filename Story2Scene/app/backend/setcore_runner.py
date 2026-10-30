"""
Stage 1 SetCoRe-LW / SetCoRe-E2E for the web app: loads a checkpoint written by the SetCoRe repository's
model/serve.py (`train`, see VideoExplain/models/train_setcore.sh) and answers one query with ServedModel.predict,
the repository's own serving code (BGE encode -> retrieval over the whole corpus -> role head -> greedy / e2e set).

    CUDA_VISIBLE_DEVICES= set2story/venv/bin/python setcore_runner.py --model DIR --query TEXT --out OUT.json [--n N]
"""
import argparse
import json
import os
import sys

REPO = os.getenv("SETCORE_REPO", "/mnt/storage/aryan/external/setcore-37E8")
sys.path.insert(0, REPO)

from model.serve import ServedModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--query-id", default="q")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = ServedModel(a.model)
    print(f"[setcore] loaded {m.manifest['model_type']} (fit on {m.manifest['fit_on']}, "
          f"{m.manifest['n_resources']} resources)", flush=True)
    r = m.predict(a.query, n=a.n, query_id=a.query_id)
    print(f"[setcore] {r['n_predicted']} resources ({r['profile']['explanation']} explanation + "
          f"{r['profile']['figure']} figure) in {r['seconds']['total']:.2f}s", flush=True)
    r["fit_on"] = m.manifest["fit_on"]
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(r, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
