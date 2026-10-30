"""
Stage 2 FreeForm baseline (set2story Baseline/baselines.py run_freeform): one unconstrained LLM call
(gemma-3-12b-it on vLLM :8002, greedy, 1600 new tokens) writes the whole storyboard. No verification, no budget
enforcement: durations are the model's own, ids outside the gold set are dropped (as in the baseline).
The result is written in the Stage-3 storyboard schema; nothing in the content is rewritten.

    CUDA_VISIBLE_DEVICES= /mnt/storage/aryan/set2story/venv/bin/python freeform_worker.py IN_gold.json T OUT.json SB_ID
IN_gold.json: {"query_id", "query", "grade", "gold_sets": [{"resources_used": [...]}]} (the backend's gold format)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/storage/aryan/VideoExplain/backend")
import set2story_worker as W  # noqa: E402  (puts set2story's Methodology/Baseline dirs on sys.path)
from baselines import run_freeform  # noqa: E402
from encoder import SentenceTransformerEncoder  # noqa: E402
from pipeline import DEFAULT_PARAMS, GSCContext  # noqa: E402


def main(gold_path, total, out_path, sb_id):
    gold_set = json.loads(Path(gold_path).read_text())
    query, qid, gold = gold_set["query"], gold_set["query_id"], gold_set["gold_sets"][0]
    enc = SentenceTransformerEncoder(str(W.ENCODER), device="cpu")
    ctx = GSCContext(enc, json.load(open(W.ROOT / "inputs/concept_vocab_curriculum.json")))
    p = DEFAULT_PARAMS
    Rq, _, _ = ctx.prepare(query, gold, p["gamma"], p["kappa"])
    W.prepare_gpu()
    llm = W.VLLMLLM()
    raw = {}

    def call(prompt, n):
        raw["prompt"], raw["text"] = prompt, llm(prompt, n)
        return raw["text"]

    board = run_freeform(query, Rq, gold_set.get("grade", "high school"), call, float(total))
    by_id = {r["resource_id"]: r for r in gold["resources_used"]}
    segs = []
    for k, s in enumerate(board.segments):
        concepts = []
        for r in s.resources:
            for c in by_id.get(r.rid, {}).get("concepts", []) or sorted(r.concepts):
                if c not in concepts:
                    concepts.append(c)
        segs.append({"segment_id": f"SEG_{k + 1:03d}", "order": k, "resource_ids": [r.rid for r in s.resources],
                     "role": s.role, "duration_sec": max(1, int(round(s.duration))), "narration": s.narration,
                     "on_screen": s.onscreen, "concepts": concepts[:4]})
    sb = {"storyboard_id": sb_id, "query_id": qid, "query": query, "intent": f"Explain: {query}",
          "target_duration_sec": int(total), "total_duration_sec": sum(s["duration_sec"] for s in segs),
          "role_template": " -> ".join(s["role"] for s in segs), "segments": segs}
    out = {"storyboard": sb, "freeform": {"raw": raw.get("text", ""), "parse_failed": bool(
        board.segments and board.segments[0].trace.get("parse_failed")),
        "raw_rids": [s.trace.get("raw_rids", []) for s in board.segments]}}
    Path(out_path).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"freeform {sb_id}: {len(segs)} segments, {sb['total_duration_sec']} s (target {total})")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
