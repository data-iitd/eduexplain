# EduExplain

From a learner's question to a textbook-grounded explainer video.

A good answer to a question such as *"How do atomic orbitals overlap to form a sigma bond?"* is rarely one document.
It is a small **set** of complementary textbook resources (a passage, a diagram, an example), organised into a
teaching plan and explained with narration and visuals. This repository holds the benchmark, the methods and the
end-to-end system for that task:

```
learner question
      │
      ▼
 Explanation-set retrieval  ──►  Storyboard construction  ──►  Narrated video
 (which resources)               (roles, order, timing,          (voice-over, animation,
                                  narration, on-screen plan)      textbook figures)

 EduExplain-banchmark/            Storyboard_Gen/                 Story2Scene/
 benchmark + baselines            reference storyboards           full system + web demo
```

**Live demo:** https://mmir.iitd.ac.in/VideoModule/

## Repository layout

| Folder | What it contains | Start here |
|---|---|---|
| [`EduExplain-banchmark/`](EduExplain-banchmark/) | The **EduExplain benchmark** for explanation-set retrieval: 6,921 resources, 321 validated queries with 1–3 gold sets each. Also contains the code that built it (PDF parsing, figure extraction, concept annotation, query and gold-set generation) and the reference baselines (BM25, BGE-M3, OpenCLIP, cross-encoder reranking; top-n, MMR and modality-aware selection) with the evaluation metrics. | [README](EduExplain-banchmark/README.md) |
| [`Storyboard_Gen/`](Storyboard_Gen/) | The **storyboard benchmark**: 741 reference storyboards (3,369 segments) for the 321 queries, with the prompt and scripts used to generate and refine them. | [README](Storyboard_Gen/README.md) |
| [`Story2Scene/`](Story2Scene/) | **Story2Scene**, the end-to-end system: SetCoRe explanation-set selection, Set2Story grounded storyboard planning, the Manim video module with a generator–verifier loop, the FastAPI + React web app (including the Compare and User-study tabs) and the demo evaluation scripts. | [README](Story2Scene/README.md) |
| [`Tech_Report.pdf`](Tech_Report.pdf) | *EduExplain: A Benchmark for Explanation-Set Retrieval and Storyboard Construction in Educational Explanation*: construction pipeline, annotator agreement and baselines. | |

## The data

The corpus is built from NCERT secondary-level science textbooks.

| | Count | File |
|---|---|---|
| Resources | 6,921 (4,594 explanations, 1,717 figures, 495 examples, 115 activities) | `EduExplain-banchmark/data/all_resources_unique_ids_clean1_merged.json` |
| Concepts | 2,123 | tagged on every resource (`concepts` field) |
| Queries with gold explanation sets | 321, across 7 pedagogical categories | `EduExplain-banchmark/data/gold_sets_final_validated_new_merged.json` |
| Reference storyboards | 741 (3,369 segments) | `Storyboard_Gen/benchmark/Storyboard_BenchMark.json` |

Each query has a *set of acceptable sets*, so a system is scored against the gold set it matches best (GoldSet
F1). A storyboard is an ordered list of segments, each with a pedagogical role (hook, definition, mechanism,
example, visual, summary, ...), its supporting resource IDs, a duration, a narration and an on-screen directive.
The figure image files and the cached dense-retrieval embeddings are not included in the repository.

## Story2Scene at a glance

| Stage | Method | Alternatives in the demo |
|---|---|---|
| 1. Explanation set | **SetCoRe**: BGE retrieval of 100 candidates, set-size prediction, coverage-greedy shortlist, LLM selection (Llama-3.1-70B) | SetCoRe-LW and SetCoRe-E2E (no LLM), OptiSet, manual selection, benchmark gold set |
| 2. Storyboard | **Set2Story**: predicted role sequence, concept-covering slot filling, role-based timing, narration checked sentence by sentence with an NLI verifier | FreeForm (one unconstrained LLM call), benchmark reference storyboard |
| 3. Video | Indic Parler-TTS voice-over recorded first; Manim scene code from Qwen3.8-27B, checked by headless execution and a Qwen3.5-9B verifier for up to three rounds; textbook figures with OCR-anchored callouts | |

Everything runs with open models on a single A100 (40 GB) GPU. See
[`Story2Scene/README.md`](Story2Scene/README.md) for setup, model serving and the web app.

## Quick start

```bash
git clone https://github.com/data-iitd/eduexplain.git
cd eduexplain
```

- **Evaluate a retriever or selector on the benchmark:** follow *Running the baselines* in
  [`EduExplain-banchmark/README.md`](EduExplain-banchmark/README.md). `evaluation/evaluator.py` computes GoldSet
  Precision / Recall / F1, Concept Coverage and Recall@K.
- **Generate or use reference storyboards:** see [`Storyboard_Gen/README.md`](Storyboard_Gen/README.md).
- **Run the full system or the web demo:** see *Running* in [`Story2Scene/README.md`](Story2Scene/README.md).

Each part has its own dependencies (`requirements.txt` or environment table in its README).

## Authors

Aryan Ahmed, Riti Verma, Sanskar Singh and Maya Ramanath, IIT Delhi.
