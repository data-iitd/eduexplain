# Set2Story: Storyboard Generation from Gold Resource Sets

Code, prompt and benchmark for generating grounded educational video storyboards from a query and a gold set of resources (text passages `TXT_*` and figures `FIG_*`).

## Repository layout

```
benchmark/Storyboard_BenchMark.json   # final benchmark: 741 storyboards, 321 queries, 3,369 segments
prompts/storyboard_prompt.txt         # system prompt used for storyboard generation
src/generate_storyboard.py            # Stage 1: generate one storyboard per (query, gold set)
src/json_utils.py                     # response parsing, schema validation, duration normalisation
src/refine_onscreen.py                # Stage 2: rewrite each segment's on_screen field (prompt embedded in the script)
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Download an instruct model in Hugging Face format (we used Qwen3.5-9B-Instruct) to a local folder.

## Pipeline

**Stage 1: storyboard generation.** Each gold set is sent to the model on its own, with `prompts/storyboard_prompt.txt` as the system message. Responses are parsed and validated by `json_utils.parse_and_validate`.

```bash
python src/generate_storyboard.py \
    --input data/gold_sets.json \
    --output outputs/storyboards.json \
    --model-path /path/to/Qwen3.5-9B-Instruct
# defaults: --temperature 0.2 --max-new-tokens 4096
```

The input is a JSON list of `{"query_id", "query", "gold_sets": [...]}` objects.

**Stage 2: on-screen refinement.** Each segment's `on_screen` description is rewritten so that it either references the segment's figure (`figure_reference`), shows a text highlight (`text_highlight`), or uses a simple generic visual (`simple_generated_visual`). The script also writes two working fields per segment, `on_screen_mode` and `on_screen_original`.

```bash
python src/refine_onscreen.py \
    --input outputs/storyboards.json \
    --output outputs/storyboards.refined.json \
    --model-path /path/to/Qwen3.5-9B-Instruct \
    --force
```

**Stage 3: export.** The released benchmark is the refined file with the two working fields removed:

```bash
python -c "import json; d=json.load(open('outputs/storyboards.refined.json')); [s.pop(k,None) for e in d for s in e['storyboard']['segments'] for k in ('on_screen_mode','on_screen_original')]; json.dump(d,open('benchmark/Storyboard_BenchMark.json','w'),indent=2,ensure_ascii=False)"
```

## Benchmark format

`benchmark/Storyboard_BenchMark.json` is a list of entries:

```json
{
  "generation_id": "Q007596::gs0",
  "query_id": "Q007596",
  "gold_set_index": 0,
  "storyboard": {
    "storyboard_id": "...", "query": "...", "gold_set_id": "...",
    "gold_set_resources": ["TXT_002442", "TXT_003306"],
    "intent": "...", "role_template": "hook -> definition -> visual -> example -> summary",
    "target_duration_sec": 180, "total_duration_sec": 180,
    "segments": [
      {
        "segment_id": "SEG_001", "order": 0, "role": "hook",
        "resource_ids": ["TXT_002442"], "duration_sec": 30,
        "narration": "...", "on_screen": "...",
        "concepts": ["..."]
      }
    ],
    "provenance": {...}, "alt_storyboard_ids": []
  }
}
```
