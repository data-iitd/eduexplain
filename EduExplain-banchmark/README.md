# EduExplain

A benchmark for **Explanation-Set Retrieval (ESR)**: given a learner's
question, return a small *set* of NCERT resources that together explain it.
Each query has a **set of acceptable sets** rather than one gold answer, so a
system is scored against whichever gold set it matches best.

- 6,921 resources (4,594 explanations, 1,717 figures, 495 examples, 115
  activities) from NCERT science textbooks
- 321 validated evaluation queries across 7 pedagogical categories
- 1–3 gold explanation sets per query, each of 1–3 resources

---

## Contents

```
data/                 the benchmark itself (resources, queries, gold sets)
parser/               NCERT PDF -> structured chunks
extraction/           pluggable PDF text extractors
image_handling/       figure extraction and VLM descriptions
resource_builder/     chunks -> resources.json
concept_annotation/   concept vocabulary and per-resource tagging
query_generation/     concepts -> queries                    [Qwen2.5-7B]
gold_generation/      queries -> validated gold sets         [Qwen2.5-7B]
baselines/            the four first-stage retrievers
selection/            the four explanation-set selection strategies
evaluation/           the metric implementations
dense_retrieval/      cached BGE-M3 corpus embeddings
documentation/        per-stage design notes
```

---

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`torch`, `transformers` and `sentence-transformers` are needed throughout;
`FlagEmbedding` for BGE-M3 (the dense retriever falls back to
sentence-transformers without it); `open_clip_torch` and `pillow` only for the
multimodal baseline; `scipy` only for significance testing.

### Local models

```bash
M=$HOME/models
hf download Qwen/Qwen2.5-7B-Instruct  --local-dir $M/Qwen2.5-7B-Instruct
hf download BAAI/bge-base-en-v1.5     --local-dir $M/bge-base-en-v1.5
hf download BAAI/bge-m3               --local-dir $M/bge-m3
hf download laion/CLIP-ViT-L-14-laion2B-s32B-b82K \
                                      --local-dir $M/openclip-vit-l-14
hf download Qwen/Qwen3-VL-Reranker-2B --local-dir $M/qwen3-vl-reranker-2b
```

| model | used by |
|---|---|
| Qwen2.5-7B-Instruct | query generation, gold-set generation, summarisation, concept annotation |
| bge-base-en-v1.5 | candidate pool for gold-set construction |
| bge-m3 | the dense retrieval baseline |
| OpenCLIP ViT-L-14 | the multimodal retrieval baseline |
| Qwen3-VL-Reranker-2B | the cross-encoder reranking baseline |

On a compute node with no outbound network, export `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` and pass local directories everywhere. 

---

## The data

| file | what |
|---|---|
| `data/all_resources_unique_ids_clean1_merged.json` | the 6,921-resource corpus |
| `data/gold_sets_final_validated_new_merged.json` | the 321 evaluation queries with their gold sets |

A resource:

```json
{
  "resource_id": "FIG_000001",
  "resource_type": "figure",          // explanation | figure | example | activity
  "document_id": "CLASS_10_SCI_CH_01",
  "class_no": 10, "subject": "Science", "chapter_title": "...",
  "text": "...",                      // raw NCERT text; "None" for figures
  "summary": "...",                   // what every retriever indexes
  "description": "...",               // VLM-generated, for figures
  "image_path": "./figures\\CLASS_10_SCI_CH_01\\..._figure_0.png",
  "concepts": ["magnesium_oxide", "oxygen", "heating"]
}
```

A query:

```json
{
  "query_id": "Q007596",
  "query": "What is a triple bond and how does it work?",
  "query_type": "explanation",        // 7 categories
  "bundle_type": "single",            // single | pair | triple  (concept count)
  "variant_style": "search",          // search | learning_context | chat
  "grade": "11",
  "gold_sets": [{"resources_used": [{"resource_id": "TXT_002442", ...}]}]
}
```

---

## Running the baselines

Set these once:

```bash
export R=data/all_resources_unique_ids_clean1_merged.json
export Q=data/gold_sets_final_validated_new_merged.json
export M=$HOME/models
mkdir -p pools results
```

### 1. First stage: build a candidate pool

Each writes one candidates file: per query, a ranked list of resources with
scores. All four produce the same format, so the selection stage is identical
across them.

```bash
# BM25 (Okapi, k1=1.2 b=0.75, no external dependency)
python baselines/bm25_retriever.py --gold $Q --resources $R \
    --top-k 100 --output pools/bm25_top100.json

# Dense, BGE-M3. Reuses the cached corpus embeddings; query embeddings are
# always recomputed.
python baselines/dense_retriever.py --gold $Q --resources $R \
    --model $M/bge-m3 \
    --resource-emb dense_retrieval/resource_embeddings.npy \
    --resource-ids dense_retrieval/resource_ids.json \
    --top-k 100 --output pools/dense_top100.json

# Multimodal, OpenCLIP ViT-L-14. Needs the figure images.
python baselines/openclip_retriever.py --gold $Q --resources $R \
    --model ViT-L-14 \
    --pretrained $M/openclip-vit-l-14/open_clip_pytorch_model.bin \
    --image-root . --image-map figure_id_map.csv \
    --top-k 100 --output pools/clip_top100.json

# Cross-encoder reranking. Reorders the dense pool; it does not retrieve.
python baselines/rerank_retriever.py --gold $Q --resources $R \
    --candidates pools/dense_top100.json \
    --model $M/qwen3-vl-reranker-2b \
    --output pools/rerank_top100.json 
```

To vary the candidate-pool size, build one deep pool and cut it to each K.
A pointwise cross-encoder scores each document independently, so reranking a
top-200 once gives the scores every smaller K would have:

```bash
python baselines/pool_at_k.py --candidates pools/rerank_top200.json \
    --k 25,50,100,200 --out-dir pools/sweep --assume-pointwise
```

### 2. Selection: pick the explanation set

A per-query budget file first — the benchmark's default is n=3:

```bash
python - $Q sizes_n3.json 3 <<'PY'
import json, sys
qs = json.load(open(sys.argv[1], encoding="utf-8"))
json.dump({str(q["query_id"]): int(sys.argv[3]) for q in qs},
          open(sys.argv[2], "w"))
PY
```

Then any of the four strategies, on any pool. `POOL` and `TAG` are the only
things that change between retrievers:

```bash
POOL=pools/dense_top100.json
TAG=dense

python selection/top_n.py \
    --gold_path $Q --candidates_path $POOL --merged_resources_path $R \
    --sizes_path sizes_n3.json --output_dir results \
    --retrieval_name $TAG --top_k 100

python selection/modality_cap.py \
    --gold_path $Q --candidates_path $POOL --merged_resources_path $R \
    --sizes_path sizes_n3.json --output_dir results \
    --retrieval_name $TAG --top_k 100

# The two MMR variants need resource embeddings for their similarity term.
python selection/mmr.py \
    --gold_path $Q --candidates_path $POOL --merged_resources_path $R \
    --sizes_path sizes_n3.json --output_dir results \
    --retrieval_name $TAG --top_k 100 \
    --embeddings_path dense_retrieval/resource_embeddings.npy \
    --resource_ids_path dense_retrieval/resource_ids.json

python selection/modality_aware_mmr.py \
    --gold_path $Q --candidates_path $POOL --merged_resources_path $R \
    --sizes_path sizes_n3.json --output_dir results \
    --retrieval_name $TAG --top_k 100 \
    --embeddings_path dense_retrieval/resource_embeddings.npy \
    --resource_ids_path dense_retrieval/resource_ids.json
```

Each prints the means and writes a per-query CSV to `results/`, named after
the selector, the retriever tag and the hyperparameters.

### Metrics

`evaluation/evaluator.py` implements GoldSet Precision, Recall and F1 — each
**maximised independently** over the query's gold sets, so P and R in a row do
not combine into that row's F1 — plus Concept Coverage and Recall@K. The
selection scripts call it; to score an external system, pass it a list of
`{"query_id", "resources"}` and the gold and candidate lookups.

`Recall@K` is computed over the candidate list **in file order**. On a
reranked file that order is the reranker's, so pool-quality Recall@K must be
read from the first-stage (dense) file, not the reranked one.

`k_values` is hardcoded to `(25, 50, 100, 200)`; a different K sweep needs
that tuple changed.

---

## Building the benchmark

### 1. PDF → structured chunks

Per chapter:

```bash
python parser/ncert_parser_v3.py \
    --pdf NCERT/class10/ch01.pdf \
    --class_no 10 --chapter_no 1 \
    --chapter_title "Chemical Reactions and Equations" \
    --backend pdfplumber --output parsed/
```

Or in batch, from a manifest of
`[{"file", "class_no", "chapter_no", "chapter_title"}, ...]`:

```bash
python parser/ncert_parser_v3.py --batch --manifest manifest.json \
    --input_dir NCERT/ --output_dir parsed/ --backend pdfplumber
```

Batch mode writes `parsed/class<N>/sci_class<N>_ch<NN>.json` per chapter plus
a combined `parsed/all_chunks.json`. Backends: `pdfplumber` (default),
`marker`, `precomputed` (`--result_path` to reuse an earlier extraction).

### 2. Figures

```bash
python image_handling/pdf_to_images.py        # NCERT/   -> page_images/
python image_handling/extract_figures.py      # page_images/ -> figures/
python image_handling/build_figure_metadata.py # figures/ -> outputs/figure_resources.json
jupyter nbconvert --execute image_handling/image_description.ipynb  # -> description.txt
python image_handling/map_descriptions.py     # -> outputs/updated_figure_resources.json
```

These read and write fixed relative paths (`./NCERT`, `./page_images`,
`./figures`, `./outputs`), so run them from the directory that holds those
folders.

### 3. Chunks → resources

```bash
python resource_builder/build_resources.py \
    --parsed_dir parsed/ --raw_dir raw/ --output_dir data_resources/

python resource_builder/summarise_resources.py
```

`build_resources.py` walks `--parsed_dir` and matches each file to the same
relative path under `--raw_dir`, deriving `document_id` from the folder and
file names. `summarise_resources.py` takes no arguments: it reads
`resource_builder/data_resources/` and writes
`resource_builder/data_resources_summmary/`.

### 4. Concept vocabulary and annotation

```bash
python concept_annotation/build_structural_vocab.py
python concept_annotation/refine_structural_vocab.py
python concept_annotation/annotate_resources.py --max_candidates 40
```

The first two take no arguments. `annotate_resources.py` accepts
`--max_resources N` to cap the run, `--include ID [ID ...]` to restrict it,
and `--max_candidates N` for how many vocabulary entries the LLM sees per
resource (default 40). It skips any resource that already has a `concepts`
field, so it is restartable.

### 5. Query generation

In order, from `query_generation/`. None take arguments — each reads the
previous stage's output and `config.py`:

```bash
cd query_generation
python build_concept_index.py             # -> concept_index.json
python build_pmi_graph.py                 # -> pmi_graph.json
python build_query_generation_input.py    # -> query_generation_input.json
python build_query_generation_bundles.py  # -> query_generation_bundles.json
python generate_queries.py                # Qwen2.5-7B -> generated_queries_hierarchial_new.json
python select_benchmark_queries.py        # -> main_queries_new.json, robustness_queries_new.json
cd ..
```

`config.py` holds the model path, decoding parameters and the prompt files.

### 6. Gold sets

```bash
cd gold_generation
python build_candidates_pool.py   # -> query_candidate_pool_2_new.json
python generate_gold_sets.py      # Qwen2.5-7B -> gold_sets_new.json
python gold_set_correction.py
python gold_set_concept_merge.py  # -> data/gold_sets_final_validated_new_merged.json
cd ..
```

`build_candidates_pool.py` computes its own bge-base embeddings over the
corpus, so `dense_retrieval/` is not a prerequisite — that directory is a
retrieval baseline, independent of construction.

Gold sets were then validated by human annotators; `documentation/` records
the protocol and the agreement figures.

---
