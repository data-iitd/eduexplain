# Story2Scene

Story2Scene turns a learner's question into a short narrated explainer video, built from a textbook corpus in three
inspectable stages:

```
question ──► Stage 1: explanation set ──► Stage 2: grounded storyboard ──► Stage 3: narrated Manim video
             (which resources)            (roles, timing, narration,       (voice-over, scene code,
                                           on-screen plan)                  checks, render)
```

Each stage writes its result as JSON (`stage1.json`, `stage2.json`, `stage3.json`). The web app shows every
stage, so a visitor can swap the method used for one stage and see what changes downstream.

## Repository layout

| Folder | Contents |
|---|---|
| `app/backend/` | FastAPI server: job queue, the three stage modules, method switches, user study |
| `app/frontend/` | React + Vite web interface |
| `app/*.sh` | start-up and model-serving scripts |
| `setcore/` | SetCoRe shortlist and LLM selection (Stage 1, SetCoRe-LLM) |
| `setcore_models/` | SetCoRe-LW / SetCoRe-E2E models and serving (`model/serve.py`); OptiSet baseline |
| `set2story/` | Set2Story storyboard construction (Stage 2); FreeForm baseline |
| `video_module/` | Manim video generation with a generator-verifier loop (Stage 3) |
| `experiments/demo_eval/` | component-swap evaluation (7 configurations x 5 benchmark questions) |

## Stage 1: explanation set

Input: the question. Output: the set of textbook resources (passages and figures) the video will explain from.

1. **Retrieval.** BGE-large-en-v1.5 (ONNX, CPU) embeds the question and returns the top-100 resources from the
   corpus (`app/backend/stage1.py`). The corpus embeddings are computed once and cached.
2. **Selection.** One of six selectors picks the set:

| Selector | How it chooses | Code |
|---|---|---|
| **SetCoRe-LLM** (default) | SetCoRe's lightweight heads predict the set size n̂; a coverage-greedy reduction keeps a shortlist of ⌈2n̂⌉ candidates; Llama-3.1-70B-Instruct picks n̂ of them (falls back to the top n̂ of the shortlist) | `stage1.py`, `setcore_adapter.py`, `setcore_worker.py`, `setcore/` |
| **SetCoRe-LW** | predicts how many explanations and whether a figure are needed, then greedily maximises weighted concept coverage plus relevance; no LLM | `setcore_runner.py`, `setcore_models/model/` |
| **SetCoRe-E2E** | a learned set predictor over the candidate embeddings, sized by the same predictor; no LLM | as above |
| **OptiSet** (baseline) | its own BGE top-20 plus three LLM sub-queries, then 70B selection and refinement | `optiset_runner.py`, `setcore_models/baselines/optiset.py` |
| **Manual** | the user ticks resources from the retrieved pool; SetCoRe's picks are marked as suggestions | `stage1.py` (`candidates`, `apply_manual`) |
| **Benchmark gold set** | the annotated set of a benchmark question | `baselines.py` |

The LW and E2E checkpoints are trained once with `app/models/train_setcore.sh` (CPU, a few minutes).

## Stage 2: grounded storyboard

Input: the question, the Stage 1 set and a time budget. Output: an ordered list of segments, each with a
pedagogical role, its supporting resources, a duration, a narration and an on-screen directive.

Set2Story (`set2story/Methodology/`, run by `app/backend/set2story_worker.py`):

1. **Role sequence.** A query-conditioned GRU decoder predicts the sequence of roles (hook, context, definition,
   mechanism, comparison, example, visual, activity, check, misconception, summary) (`role_predictor.py`).
2. **Slot filling.** Each slot is filled with resources that add uncovered concepts and relevance, with at most
   one figure per slot (`slotfill.py`).
3. **Timing.** Role-specific duration priors divide the budget (`slotfill.py`).
4. **Grounded narration.** gemma-3-12b-it writes each segment's narration and on-screen directive from that
   segment's resources only. An NLI cross-encoder (nli-deberta-v3-large) checks every factual sentence
   (entailment ≥ 0.5). A failing draft is regenerated with the unsupported claims as feedback, up to three
   generations, then unsupported sentences are dropped or an extractive fallback is used. An on-screen directive
   that the narration contradicts is rejected (`realization.py`).

Alternatives: **FreeForm**, one unconstrained LLM call that writes the whole storyboard
(`set2story/Baseline/baselines.py`, `app/backend/freeform_worker.py`), and the benchmark's **reference
storyboard** (`app/backend/baselines.py`).

## Stage 3: narrated video

Input: the storyboard. Output: `final.mp4` with voice-over and subtitles (`video_module/`, started by
`app/backend/stage3.py`).

1. **Voice-over first.** Indic Parler-TTS (speaker "Mary", Indian English) records every segment (`tts.py`,
   `tts_parler.py`, `parler_worker.py`). The planned duration is an upper bound: a segment lasts as long as its
   speech, and over-long speech is squeezed, slightly sped up, or borrows spare time from other segments. The
   measured timeline (segment lengths and beat times) replaces the planned one.
2. **Figures.** A segment that lists a textbook figure must show it. Its parts are located by OCR of the printed
   labels plus the verifier vision-language model, so callouts point at the right places (`figures.py`).
3. **Scene code.** The generator, Qwen3.8-27B (FP8), writes one Manim CE script with one method per segment
   (`prompt.py`).
4. **Checks, up to three rounds** (`pipeline.py`):
   - static checks: script contract, quoted on-screen text, figure use (`checks.py`);
   - a headless execution of the scene, measuring segment timing, narration-animation sync, text overlaps,
     out-of-frame objects and figure size (`exec_check.py`); labels are moved to free space at run time
     (`layout_fix.py`);
   - the verifier, Qwen3.5-9B, reviews the code for layout and sync.

   Critical and major issues go back to the generator as feedback. The round with the fewest issues becomes
   `final_scene.py`.
5. **Render.** The scene is rendered at 720p with each segment's audio anchored at its start and the subtitles
   muxed in. A segment that would overrun is played slightly faster, so the video never exceeds the budget
   (`render.py`).

## Web app

- `app/backend/app.py`, FastAPI on port 8800:
  - `POST /api/jobs` runs stages 1–3 as a background job (one at a time);
  - `GET /api/jobs/{id}` returns the stage outputs and a progress estimate;
  - `POST /api/jobs/{id}/selection` continues a manual Stage 1;
  - also `GET /api/jobs`, `GET /api/benchmark`, `/api/jobs/{id}/video`, `/api/figure/{FIG_id}`,
    `/api/health`.
- `app/frontend/` (Vite, port 5173, proxies `/api` to 8800):
  - the question box and the Stage 1 / Stage 2 method switches;
  - the three stage panels;
  - a gallery of precomputed results;
  - the Compare tab (`app/frontend/src/Compare.tsx`): two finished configurations of one question side by side
    (videos with "Play both", Stage 1 sets with shared / unique resources marked, storyboards);
  - the user-study tab (`app/backend/study.py`, `app/frontend/src/Study.tsx`).

## Models and hardware

Everything runs on one A100 40 GB GPU with open models served by vLLM (OpenAI-compatible API):

| Port | Model | Used by |
|---|---|---|
| 8000 | Qwen3.8-27B (FP8) | Stage 3 generator |
| 8001 | Qwen3.5-9B (vision) | Stage 3 verifier, figure grounding |
| 8002 | gemma-3-12b-it | Stage 2 narration, FreeForm |
| 8003 | Llama-3.1-70B-Instruct (AWQ INT4, partly offloaded to CPU RAM) | SetCoRe-LLM, OptiSet |

The servers share the GPU through vLLM sleep mode: before a client generates, it puts the other servers to sleep
and wakes its own. On the CPU: BGE-large-en-v1.5, all-roberta-large-v1 (Set2Story encoder), nli-deberta-v3-large
and Parler-TTS (GPU when free). `app/download_models.sh` fetches the Hugging Face models; Llama and Gemma are
gated.

## Running

Environments:

| Environment | Packages | Used for |
|---|---|---|
| app | fastapi, uvicorn, pydantic, numpy, onnxruntime, tokenizers | backend |
| set2story | torch (CPU), transformers, sentence-transformers, scikit-learn | Stage 1 / Stage 2 workers |
| video module | see `video_module/requirements.txt` | Manim client, Parler-TTS, vLLM serving |
| frontend | Node.js | `npm install` |

Start the model servers, then the app:

```bash
MODEL_ROOT=<models dir> video_module/serve_models.sh      # :8000 generator, :8001 verifier
app/serve_gemma.sh                                       # :8002 (optional; Stage 2 LLM)
app/serve_llama70b.sh                                    # :8003 (optional; SetCoRe-LLM / OptiSet)
GEMMA=1 LLAMA=1 app/start_all.sh                         # backend :8800 + frontend :5173 in tmux
```

Or run one storyboard through Stage 3 on its own:

```bash
python video_module/pipeline.py -s storyboard.json \
  --gen-url http://127.0.0.1:8000/v1 --ver-url http://127.0.0.1:8001/v1 --max-rounds 3
```

The corpus and the benchmark are in this repository: the resources and the gold sets in
[`../EduExplain-banchmark/data/`](../EduExplain-banchmark/data/), the reference storyboards in
[`../Storyboard_Gen/benchmark/`](../Storyboard_Gen/benchmark/). The figure images and the model weights are not.
Paths default to the original deployment; point them at the repository copies with:

```bash
export SETCORE_DIR=$PWD/../EduExplain-banchmark/data      # resources + gold sets (run from Story2Scene/)
export VIDEO_RESOURCES_JSON=$SETCORE_DIR/all_resources_unique_ids_clean1_merged.json
export BENCH_SB=$PWD/../Storyboard_Gen/benchmark/Storyboard_BenchMark.json
export BGE_DIR=<models dir>/bge-large-en-v1.5             # default: $SETCORE_DIR/models/bge-large-en-v1.5
export VIDEO_FIGURES_DIR=<figures dir>                    # figure images + figure_id_mapping.csv
```

Other paths can be changed the same way, among them `VIDEOMODULE`, `S2S_PY`, `S2S_GEN_URL`,
`VX_STAGE1_LLM_URL` and `VIDEO_FIGURE_MAPPING`.

## Evaluation scripts

`experiments/demo_eval/` reproduces the component-swap study:

- `run_demo_eval.py` runs Stage 1 and Stage 2 for seven configurations on five benchmark questions, then
  Stage 3 for all 35;
- `run_setcore.sh` produces the SetCoRe-LW / E2E predictions;
- `e1_metrics.py` computes GoldSet F1, the NLI-entailed share of narration claims, and the seconds beyond
  budget;
- `publish_premade.py` turns the 35 runs into the web app's gallery.
