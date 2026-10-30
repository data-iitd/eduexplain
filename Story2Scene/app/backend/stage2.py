"""
Stage 2 - gold set -> storyboard (Stage-3 schema), around /mnt/storage/aryan/set2story.

The set2story adapter (set2story_adapter.py, added once its models are available) is used when it loads; otherwise
the LLM fallback below writes the storyboard with the shared vLLM generator so the 3-stage pipeline always runs.
result["mode"] says which path produced the storyboard.

Output = Stage-3 storyboard: {"storyboard": {storyboard_id, query_id, query, intent, target_duration_sec,
total_duration_sec, role_template, segments: [{segment_id, order, resource_ids, role, duration_sec, narration,
on_screen, concepts}]}}
"""
import json
import re
import time

ROLE_TEMPLATE = ["hook", "definition", "visual", "example", "summary"]

PROMPT = """You write the storyboard for a short educational explainer video (a teacher's voice-over with on-screen
visuals). Use ONLY the facts in the selected resources below; do not invent facts.

Learner query: {query}

Selected resources (the gold explanation set):
{resources}

Write {n_seg} segments in this order of roles: {roles}. Total duration exactly {total} seconds
(segment durations are whole seconds and add up to {total}; hook and summary 15-30 s, others 30-50 s).
For each segment:
- narration: what the teacher says, 2-4 clear sentences, about 2.2 words per second of the segment
- on_screen: what is visible; put any exact on-screen text in 'single quotes'
- resource_ids: the ids of the resources this segment uses (at least one; every resource used somewhere).
  A FIGURE resource (id FIG_...) must be listed in the segment that shows it, and that segment's on_screen must
  say "The figure <id> is shown" and how to annotate it (arrow / highlight / label which parts).
- concepts: 2-4 snake_case concept tags from the resources

Return ONLY JSON:
{{"intent": "<one sentence>", "segments": [{{"role": "...", "duration_sec": 30, "resource_ids": ["..."],
  "narration": "...", "on_screen": "...", "concepts": ["..."]}}]}}"""


WORDS_PER_SEC = 2.3   # teacher pace at Parler speed 0.92, leaving room for the natural pauses


def fit_narration(storyboard, wps=WORDS_PER_SEC, margin=2.0):
    """Time budget first: keep whole sentences of each segment's narration while they fit its duration at the
    teacher's pace (always at least the first sentence). set2story writes narration without a length target, so
    without this a long narration overruns its slot, the voice-over is cut and the video runs long."""
    trimmed = []
    for seg in storyboard["storyboard"]["segments"]:
        budget = max(6.0, (float(seg["duration_sec"]) - margin) * wps)
        sents = [x for x in re.split(r"(?<=[.!?])\s+", seg.get("narration", "").strip()) if x]
        kept, words = [], 0
        for x in sents:
            n = len(x.split())
            if kept and words + n > budget:
                break
            kept.append(x)
            words += n
        if len(kept) < len(sents):
            trimmed.append(f"{seg['segment_id']}: {len(sents) - len(kept)} sentence(s) dropped")
            seg["narration"] = " ".join(kept)
    return trimmed


def _resource_block(gold_set):
    rows = []
    for r in gold_set["gold_sets"][0]["resources_used"]:
        txt = next((str(r[k]) for k in ("summary", "description", "image_caption", "text")
                    if r.get(k) and str(r[k]) != "None"), "")[:700]
        rows.append(f"- {r['resource_id']} ({r.get('resource_type')}): {txt}")
    return "\n".join(rows)


def _parse(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    m = re.search(r"\{.*\}", re.sub(r"```(?:json)?", "", raw), flags=re.S)
    return json.loads(m.group(0)) if m else None


def _normalise(data, gold_set, total, query_id, query):
    valid = [r["resource_id"] for r in gold_set["gold_sets"][0]["resources_used"]]
    segs = []
    for k, s in enumerate(data.get("segments", [])):
        rids = [r for r in s.get("resource_ids", []) if r in valid] or valid[:1]
        segs.append({"segment_id": f"SEG_{k + 1:03d}", "order": k, "resource_ids": rids,
                     "role": str(s.get("role", ROLE_TEMPLATE[min(k, 4)])), "duration_sec": max(8, int(s.get("duration_sec", 30))),
                     "narration": str(s.get("narration", "")).strip(), "on_screen": str(s.get("on_screen", "")).strip(),
                     "concepts": [str(c) for c in s.get("concepts", [])][:4]})
    # every gold resource is used somewhere (figures especially)
    used = {r for s in segs for r in s["resource_ids"]}
    for rid in valid:
        if rid not in used and segs:
            tgt = next((s for s in segs if s["role"] == "visual"), segs[min(2, len(segs) - 1)])
            tgt["resource_ids"].append(rid)
    # exact time budget: scale durations to the target total
    cur = sum(s["duration_sec"] for s in segs) or 1
    for s in segs:
        s["duration_sec"] = max(8, round(s["duration_sec"] * total / cur))
    if segs:
        segs[-1]["duration_sec"] += total - sum(s["duration_sec"] for s in segs)
    return {"storyboard": {
        "storyboard_id": f"SB_{query_id}", "query_id": query_id, "query": query,
        "intent": str(data.get("intent", f"Explain: {query}")), "target_duration_sec": total,
        "total_duration_sec": total, "role_template": " -> ".join(s["role"] for s in segs), "segments": segs}}


class Stage2:
    def __init__(self, llm=None, log=print):
        self.llm, self.log = llm, log
        self.adapter = None
        try:   # real set2story path, when its models/artifacts are present
            from set2story_adapter import Set2StoryAdapter
            self.adapter = Set2StoryAdapter(llm=llm, log=log)
        except Exception as exc:
            self.log(f"! set2story adapter unavailable ({exc}); Stage 2 uses the LLM storyboard fallback")

    def run(self, stage1, total=180):
        t = time.time()
        gold, qid, query = stage1["gold_set"], stage1["query_id"], stage1["query"]
        if self.adapter is not None:
            try:
                sb = self.adapter.run(gold, total=total)
                trimmed = fit_narration(sb)
                if trimmed:
                    self.log("narration fitted to the time budget: " + "; ".join(trimmed))
                return {"stage": 2, "mode": "set2story", "storyboard": sb, "narration_trimmed": trimmed,
                        "seconds": round(time.time() - t, 1)}
            except Exception as exc:
                self.log(f"! set2story failed ({exc}); falling back to the LLM storyboard writer")
        if self.llm is None:
            raise RuntimeError("Stage 2 needs the set2story models or the vLLM generator; neither is available")
        prompt = PROMPT.format(query=query, resources=_resource_block(gold), n_seg=len(ROLE_TEMPLATE),
                               roles=" -> ".join(ROLE_TEMPLATE), total=total)
        data, raw = None, ""
        for attempt in range(2):
            raw = self.llm.generate([{"role": "user", "content": prompt}], max_new_tokens=2500, temperature=0.3)
            try:
                data = _parse(raw)
            except json.JSONDecodeError:
                data = None
            if data and data.get("segments"):
                break
        if not data or not data.get("segments"):
            raise RuntimeError(f"Stage 2 LLM did not return a storyboard: {raw[:300]}")
        sb = _normalise(data, gold, total, qid, query)
        trimmed = fit_narration(sb)
        return {"stage": 2, "mode": "llm-fallback", "storyboard": sb, "narration_trimmed": trimmed,
                "llm_response": raw[:3000],
                "seconds": round(time.time() - t, 1)}
