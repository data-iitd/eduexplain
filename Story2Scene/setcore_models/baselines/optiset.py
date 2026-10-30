
#!/usr/bin/env python3
"""
Training-free OptiSet inference for the EduExplain ESR benchmark.

Pipeline
--------
1. BGE-large-en-v1.5 dense retrieval -> initial top-K candidates
2. Llama-3.1-70B-Instruct -> query expansion
3. Llama-3.1-70B-Instruct -> set selection over the top-K candidates
4. Llama-3.1-70B-Instruct -> refinement of the selected set
5. Return at most 4 resources

NO training / fine-tuning is performed.

Resource representation
-----------------------
There are THREE places the resource representation enters this pipeline, not
one, and they decide different things:

  --retrieval-fields   what BGE embeds, i.e. which candidates reach the LLM
  --prompt-fields      what format_candidate() emits, which is read TWICE --
                       once in the selection prompt and once in refinement

Setting only the first leaves the LLM reading seven labelled fields over a
summary-only pool, which is not a summary-only run. Both default to "all" (the
original behaviour); pass "summary" to both for a like-for-like comparison with
icra, whose selector may read only index_text().

In "summary" and "summary_type" modes the `RESOURCE ID:` line is also dropped
from each candidate block. That line is not evidence: on this corpus the id
carries the modality in its prefix (TXT_/IMG_), so leaving it in hands the LLM
a type signal through a field that looks like bookkeeping.

Expected resource corpus
------------------------
A JSON list of resource dictionaries, e.g.

[
  {
    "resource_id": "TXT_002442",
    "resource_type": "explanation",
    "summary": "...",
    "description": "",
    "document_id": "...",
  }
]

or:

{"resources": [ ... ]}

Usage
-----
Single K:

python optiset.py \
    --benchmark /path/to/benchmark.json \
    --resources /path/to/all_resources.json \
    --output optiset_k20.json \
    --top-k 20 \
    --llm-model /path/to/Llama-3.1-70B-Instruct \

Summary-only (comparable with icra and with SetR --*-fields summary):

python optiset.py \
    --benchmark /path/to/benchmark.json \
    --resources /path/to/all_resources.json \
    --output optiset_summary.json \
    --retrieval-fields summary \
    --prompt-fields summary \
    --llm-model /path/to/Llama-3.1-70B-Instruct \

Multiple K values:

python optiset.py \
    --benchmark /path/to/benchmark.json \
    --resources /path/to/all_resources.json \
    --output optiset.json \
    --top-k-list 10,20,40,60 \
    --llm-model /path/to/Llama-3.1-70B-Instruct \

This writes:
    optiset_k10.json
    optiset_k20.json
    optiset_k40.json
    optiset_k60.json

The resource embeddings are computed ONCE and reused for all K values.
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# User-specified models
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "fill bge large v1.5 model path here"

# Change this to your local Llama directory if the model is stored locally.
LLM_MODEL = "meta-llama/Llama-3.1-70B-Instruct"

MAX_FINAL_RESOURCES = 4

# OptiSet expansion is a small set of alternative query perspectives.
DEFAULT_NUM_SUBQUERIES = 3


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_benchmark(path: str) -> list[dict[str, Any]]:
    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError(
            f"Benchmark must be a JSON list. Got: {type(data).__name__}"
        )

    return data


def load_resources(path: str) -> list[dict[str, Any]]:
    """
    Load the complete searchable resource corpus.

    Supported:
        1. [resource1, resource2, ...]
        2. {"resources": [resource1, resource2, ...]}
    """
    data = load_json(path)

    if isinstance(data, list):
        resources = data

    elif isinstance(data, dict) and isinstance(data.get("resources"), list):
        resources = data["resources"]

    else:
        raise ValueError(
            "Resource corpus must be either a JSON list or "
            '{"resources": [...]}'
        )

    output = []
    seen_ids = set()

    for resource in resources:
        if not isinstance(resource, dict):
            continue

        rid = resource.get("resource_id")

        if not rid:
            continue

        if rid in seen_ids:
            continue

        seen_ids.add(rid)
        output.append(resource)

    if not output:
        raise ValueError(
            f"No valid resources with resource_id found in: {path}"
        )

    return output


# ---------------------------------------------------------------------------
# Resource text used by BGE
# ---------------------------------------------------------------------------

ALL_FIELDS = [
    ("TYPE", "resource_type"),
    ("SUBJECT", "subject"),
    ("CHAPTER", "chapter_title"),
    ("SUMMARY", "summary"),
    ("DESCRIPTION", "description"),
    ("IMAGE CAPTION", "image_caption"),
    ("TEXT", "text"),
]


def index_text(resource: dict[str, Any]) -> str:
    for key in ("summary", "description", "image_caption"):
        v = resource.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    v = (resource.get("text") or "").strip()[:2000]
    return v or str(resource.get("resource_id", ""))


def resource_text(resource: dict[str, Any], mode: str = "all") -> str:
    """
    mode="all"     -> the seven labelled fields (the original)
    mode="summary" -> icra's index_text(), for a like-for-like comparison

    """
    if mode == "summary":
        return index_text(resource)

    parts = []

    for label, key in ALL_FIELDS:
        value = resource.get(key)

        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        parts.append(f"{label}: {value}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# BGE dense retrieval
# ---------------------------------------------------------------------------

def encode_resource_corpus(
    encoder: SentenceTransformer,
    resources: list[dict[str, Any]],
    batch_size: int,
    field_mode: str = "all",
) -> np.ndarray:
    texts = [resource_text(r, field_mode) for r in resources]

    print(f"Resource representation for RETRIEVAL: {field_mode}")
    print(f"  mean characters per resource: "
          f"{sum(len(t) for t in texts)/max(1,len(texts)):.0f}")

    print(f"Encoding {len(texts)} resources with BGE...")

    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    return embeddings.astype(np.float32)


def retrieve_top_k(
    encoder: SentenceTransformer,
    resource_embeddings: np.ndarray,
    resources: list[dict[str, Any]],
    query: str,
    k: int,
) -> list[dict[str, Any]]:
    """
    Dense cosine retrieval.

    Because both query and resource embeddings are normalized,
    cosine similarity is simply their dot product.
    """
    query_embedding = encoder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0].astype(np.float32)

    scores = resource_embeddings @ query_embedding

    k = min(k, len(resources))

    ranked_indices = np.argsort(-scores)[:k]

    retrieved = []

    for rank, idx in enumerate(ranked_indices, start=1):
        retrieved.append({
            "resource": resources[int(idx)],
            "score": float(scores[int(idx)]),
            "retrieval_rank": rank,
        })

    return retrieved


# ---------------------------------------------------------------------------
# Llama / Transformers
# ---------------------------------------------------------------------------

def make_llm(model_name: str):
    """
    Load Llama-3.1-70B-Instruct using the same Transformers setup as the
    existing baseline: bfloat16 + device_map="auto" + local_files_only.
    """
    print(f"Loading Llama from: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    model.eval()

    return tokenizer, model


def generate_one(
    tokenizer,
    model,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Generate one response using the same greedy/sampling interface as the baseline."""
    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )

    input_device = next(model.parameters()).device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if temperature <= 0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.9,
        })

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:,
    ]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


# ---------------------------------------------------------------------------
# OptiSet: Expansion
# ---------------------------------------------------------------------------

EXPANSION_PROMPT = """Search Query: {question}

Rules:
- Generate {num_subqueries} standalone sub-queries.
- Each sub-query must capture a different information need or perspective
  required to answer the Search Query.
- Each sub-query must stand alone.
- Do not refer to previous queries.
- Do not use pronouns such as "it", "they", "this", or "that" when the
  antecedent would be the original question or another query.
- Do not answer the queries.
- Output exactly one sub-query per line.
- Do not output numbering, bullets, explanations, headings, or markdown.
"""


def clean_subqueries(
    text: str,
    max_subqueries: int,
) -> list[str]:
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        line = re.sub(r"^\s*[-*•]\s*", "", line)
        line = re.sub(r"^\s*\d+[\.\)]\s*", "", line)
        line = line.strip()

        if line:
            lines.append(line)

    # Deduplicate while preserving order.
    output = []
    seen = set()

    for line in lines:
        key = line.casefold()

        if key in seen:
            continue

        seen.add(key)
        output.append(line)

        if len(output) >= max_subqueries:
            break

    return output


def expand_query(
    tokenizer,
    model,
    question: str,
    num_subqueries: int,
    temperature: float,
) -> tuple[list[str], str]:
    prompt = EXPANSION_PROMPT.format(
        question=question,
        num_subqueries=num_subqueries,
    )

    response = generate_one(
        tokenizer=tokenizer,
        model=model,
        user_prompt=prompt,
        temperature=temperature,
        max_tokens=256,
    )

    subqueries = clean_subqueries(
        response,
        max_subqueries=num_subqueries,
    )

    return subqueries, response


# ---------------------------------------------------------------------------
# Candidate formatting
# ---------------------------------------------------------------------------

def format_candidate(
    prompt_index: int,
    item: dict[str, Any],
    field_mode: str = "all",
) -> str:
    resource = item["resource"]

    block = f"[{prompt_index}]"

    if field_mode in ("summary", "summary_type"):
        if field_mode == "summary_type":
            t = resource.get("resource_type")
            if t:
                block += f"\nTYPE: {t}"
        block += f"\nSUMMARY: {index_text(resource)}"
        return block

    block += f"\nRESOURCE ID: {resource.get('resource_id', '')}"

    for label, key in ALL_FIELDS:
        value = resource.get(key)

        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        block += f"\n{label}: {value}"


    return block


# ---------------------------------------------------------------------------
# OptiSet: Selection
# ---------------------------------------------------------------------------

def make_selection_prompt(
    question: str,
    subqueries: list[str],
    candidates: list[dict[str, Any]],
    field_mode: str = "all",
) -> str:
    subquery_text = "\n".join(
        f"{i}. {sq}"
        for i, sq in enumerate(subqueries, start=1)
    )

    context = "\n\n".join(
        format_candidate(i, item, field_mode)
        for i, item in enumerate(candidates, start=1)
    )

    return f"""I will provide you with {len(candidates)} passages, each
indicated by a numerical identifier [].

Select passages based on their relevance to the search query.

Search Query: {question}

Sub-Queries:
{subquery_text if subquery_text else "(No sub-queries were generated.)"}

Context:
{context}

Please follow the steps below:

Step 1. List the information requirements needed to answer the query and
sub-queries.

Step 2. For each information requirement, identify the passages that contain
the required information.

Step 3. Choose the passages that provide the clearest and most diverse
coverage of the query.

The final set MUST contain at most {MAX_FINAL_RESOURCES} passages.

Output exactly:
### Final Selection: [i] [j] [k]

where i, j, k are the numerical passage identifiers you selected.

Output no explanations after the final selection.
"""


# ---------------------------------------------------------------------------
# OptiSet: Refinement
# ---------------------------------------------------------------------------

def make_refinement_prompt(
    question: str,
    candidates: list[dict[str, Any]],
    field_mode: str = "all",
) -> str:
    context = "\n\n".join(
        format_candidate(i, item, field_mode)
        for i, item in enumerate(candidates, start=1)
    )

    return f"""I will provide you with {len(candidates)} passages, each
indicated by a numerical identifier [].

Select passages based on their relevance to the search query.

Search Query: {question}

Context:
{context}

Step 1. Identify whether the above set of paragraphs contains irrelevant
or repetitive paragraphs.

Step 2. Exclude irrelevant or repetitive information while preserving the
information necessary to answer the search query.

The final set MUST contain at most {MAX_FINAL_RESOURCES} passages.

Output exactly:
### Final Selection: [i] [j] [k]

where i, j, k are the numerical passage identifiers you selected.

Output no explanations after the final selection.
"""


# ---------------------------------------------------------------------------
# Parse LLM selections
# ---------------------------------------------------------------------------

FINAL_SELECTION_RE = re.compile(
    r"###\s*Final\s+Selection\s*:\s*(.*?)(?:\n|$)",
    flags=re.IGNORECASE,
)


def parse_selected_indices(
    text: str,
    n_candidates: int,
    max_items: int = MAX_FINAL_RESOURCES,
    stats: Counter | None = None,
    tag: str = "",
) -> list[int]:
    """
    Parse selections such as:

        ### Final Selection: [2] [7] [4]

    Only valid candidate indices are accepted.

    Two fallbacks, in order, and both are counted rather than silent:

    1. No header. The original set selection_text = the WHOLE response, which
       harvests the Step-2 reasoning indices instead of the Step-3 answer: on
       "Req 1 -> [7] and [12] ... the clearest combination is [3], [9], [12]"
       it returns [7, 12, 3, 9]. Scanning the LAST bracket-bearing line is much
       closer to the model's actual conclusion. Same fix as baseline_setr.py,
       so the two baselines parse identically.

    2. No brackets anywhere. Only here do we fall back to bare integers. Run on
       the whole response this is actively harmful -- the prompt itself contains
       "Step 1." and "Step 2.", and 1 and 2 are valid candidate indices, so a
       response with no selection at all parses as [1, 2]. Restricting it to the
       chosen line keeps that from turning a non-answer into a plausible one.
    """
    match = FINAL_SELECTION_RE.search(text)

    if match:
        selection_text = match.group(1)
        path = "header"
    else:
        lines = [l for l in text.strip().splitlines()
                 if re.search(r"\[\d+\]", l)]
        if lines:
            selection_text = lines[-1]
            path = "fallback_bracket"
        else:
            selection_text = text.strip().splitlines()[-1] if text.strip() else ""
            path = "fallback_bare"

    # Preferred format: [2] [7] [4]
    numbers = re.findall(
        r"\[(\d+)\]",
        selection_text,
    )

    # Fallback for slightly malformed output: "2, 7, 4"
    if not numbers:
        numbers = re.findall(
            r"\b(\d+)\b",
            selection_text,
        )

    selected = []
    seen = set()

    for number in numbers:
        idx = int(number)

        if not (1 <= idx <= n_candidates):
            continue

        if idx in seen:
            continue

        seen.add(idx)
        selected.append(idx)

        if len(selected) >= max_items:
            break

    if stats is not None:
        stats[f"{tag}{path}"] += 1
        if not selected:
            stats[f"{tag}empty"] += 1

    return selected


def unique_items(
    items: list[dict[str, Any]],
    max_items: int = MAX_FINAL_RESOURCES,
) -> list[dict[str, Any]]:
    output = []
    seen_ids = set()

    for item in items:
        rid = item["resource"]["resource_id"]

        if rid in seen_ids:
            continue

        seen_ids.add(rid)
        output.append(item)

        if len(output) >= max_items:
            break

    return output


# ---------------------------------------------------------------------------
# One complete OptiSet run
# ---------------------------------------------------------------------------

def run_optiset_for_query(
    tokenizer,
    model,
    encoder: SentenceTransformer,
    resource_embeddings: np.ndarray,
    resources: list[dict[str, Any]],
    query_id: str,
    question: str,
    top_k: int,
    num_subqueries: int,
    temperature: float,
    prompt_fields: str = "all",
    stats: Counter | None = None,
) -> dict[str, Any]:

    # ================================================================
    # 1. INITIAL RETRIEVAL
    # ================================================================

    retrieved = retrieve_top_k(
        encoder=encoder,
        resource_embeddings=resource_embeddings,
        resources=resources,
        query=question,
        k=top_k,
    )

    # ================================================================
    # 2. EXPAND
    # ================================================================

    subqueries, expansion_response = expand_query(
        tokenizer=tokenizer,
        model=model,
        question=question,
        num_subqueries=num_subqueries,
        temperature=temperature,
    )

    # ================================================================
    # 3. SELECT
    #
    # ================================================================

    selection_prompt = make_selection_prompt(
        question=question,
        subqueries=subqueries,
        candidates=retrieved,
        field_mode=prompt_fields,
    )

    selection_response = generate_one(
        tokenizer=tokenizer,
        model=model,
        user_prompt=selection_prompt,
        temperature=temperature,
        max_tokens=512,
    )

    selected_indices = parse_selected_indices(
        selection_response,
        n_candidates=len(retrieved),
        max_items=MAX_FINAL_RESOURCES,
        stats=stats,
        tag="select_",
    )

    selected_items = [
        retrieved[i - 1]
        for i in selected_indices
    ]

    # Safe fallback for malformed LLM output.
    if not selected_items and retrieved:
        selected_indices = [1]
        selected_items = [retrieved[0]]
        if stats is not None:
            stats["select_defaulted_to_rank1"] += 1

    # ================================================================
    # 4. REFINE
    # ================================================================

    refinement_prompt = make_refinement_prompt(
        question=question,
        candidates=selected_items,
        field_mode=prompt_fields,
    )

    refinement_response = generate_one(
        tokenizer=tokenizer,
        model=model,
        user_prompt=refinement_prompt,
        temperature=temperature,
        max_tokens=256,
    )

    refined_local_indices = parse_selected_indices(
        refinement_response,
        n_candidates=len(selected_items),
        max_items=MAX_FINAL_RESOURCES,
        stats=stats,
        tag="refine_",
    )

    refined_items = [
        selected_items[i - 1]
        for i in refined_local_indices
    ]

    # Safe fallback if refinement output is malformed.
    if not refined_items:
        refined_items = selected_items[:MAX_FINAL_RESOURCES]
        if stats is not None:
            stats["refine_kept_selection"] += 1

    refined_items = unique_items(
        refined_items,
        max_items=MAX_FINAL_RESOURCES,
    )

    selected_resources = [
        item["resource"]
        for item in refined_items
    ]

    # ================================================================
    # 5. Map final resources back to the INITIAL top-K prompt indices
    # ================================================================

    prompt_index_by_resource_id = {
        item["resource"]["resource_id"]: local_idx
        for local_idx, item in enumerate(retrieved, start=1)
    }

    selected_prompt_indices = [
        prompt_index_by_resource_id[r["resource_id"]]
        for r in selected_resources
        if r["resource_id"] in prompt_index_by_resource_id
    ]

    # ================================================================
    # 6. OUTPUT
    #
    # ================================================================

    retrieval_output = []

    for local_idx, item in enumerate(retrieved, start=1):
        resource = item["resource"]

        retrieval_output.append({
            "prompt_index": local_idx,
            "resource_id": resource["resource_id"],
            "resource_type": resource.get("resource_type"),
            "score": item["score"],
            "summary": resource.get("summary", ""),
        })

    result = {
        "query_id": query_id,
        "query": question,

        "predicted_set": {
            "resources_used": selected_resources
        },

        "predicted_resource_ids": [
            r["resource_id"]
            for r in selected_resources
        ],

        "retrieved_resources": retrieval_output,

        # Kept for compatibility with your current result format.
        "setr_response": refinement_response,

        "selected_prompt_indices": selected_prompt_indices,

        # ------------------------------------------------------------
        # OptiSet diagnostics
        # ------------------------------------------------------------

        "optiset_subqueries": subqueries,

        "optiset_expansion_response": expansion_response,

        "optiset_selection_response": selection_response,

        "optiset_refinement_response": refinement_response,

        "initial_top_k": top_k,

        "final_resource_count": len(selected_resources),

        "prompt_fields": prompt_fields,
    }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run training-free OptiSet on the EduExplain ESR benchmark."
        )
    )

    parser.add_argument(
        "--benchmark",
        required=True,
        help=(
            "Benchmark JSON containing query_id and query. "
            "Your gold_sets JSON can also be supplied here if it contains "
            "the complete benchmark query list."
        ),
    )

    parser.add_argument(
        "--resources",
        required=True,
        help=(
            "Full searchable resource corpus JSON. "
            "This is the corpus over which BGE performs initial retrieval."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON filename.",
    )

    # One K.
    parser.add_argument(
        "--top-k", "--top_k",
        dest="top_k",
        type=int,
        default=20,
        help="Initial BGE retrieval K.",
    )

    # Several K values.
    parser.add_argument(
        "--top-k-list", "--top_k_list",
        dest="top_k_list",
        default=None,
        help=(
            "Comma-separated initial retrieval K values, e.g. "
            "10,20,40,60. Each K is run independently."
        ),
    )

    parser.add_argument(
        "--embedding-model", "--embedding_model",
        dest="embedding_model",
        default=EMBEDDING_MODEL,
        help="Path/name of the BGE embedding model.",
    )

    parser.add_argument(
        "--llm-model", "--llm_model",
        dest="llm_model",
        default=LLM_MODEL,
        help="Path/name of Llama-3.1-70B-Instruct.",
    )

    parser.add_argument(
        "--num-subqueries", "--num_subqueries",
        dest="num_subqueries",
        type=int,
        default=DEFAULT_NUM_SUBQUERIES,
        help="Number of expansion queries generated per benchmark query.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help=(
            "LLM sampling temperature. "
            "Use 0 for greedy/deterministic selection."
        ),
    )

    parser.add_argument(
        "--embedding-batch-size", "--embedding_batch_size",
        dest="embedding_batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of benchmark queries to run.",
    )

    parser.add_argument(
        "--query-id", "--query_id",
        dest="query_id",
        default=None,
        help="Run only this query_id.",
    )

    parser.add_argument(
        "--retrieval-fields", "--retrieval_fields",
        dest="retrieval_fields",
        default="all",
        choices=["all", "summary"],
        help="what the BGE index embeds. 'summary' = icra's index_text()",
    )

    parser.add_argument(
        "--prompt-fields", "--prompt_fields",
        dest="prompt_fields",
        default="all",
        choices=["all", "summary", "summary_type"],
        help="what the LLM sees per candidate, in BOTH the selection and the "
             "refinement call. 'summary' = index_text() only and no RESOURCE "
             "ID line; 'summary_type' adds the TYPE line, which icra's "
             "selector also knows",
    )

    args = parser.parse_args()

    for k, v in list(vars(args).items()):
        if isinstance(v, str) and v.startswith("~"):
            setattr(args, k, os.path.expanduser(v))

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---------------------------------------------------------------
    # Resolve K values
    # ---------------------------------------------------------------

    if args.top_k_list:
        top_k_values = [
            int(x.strip())
            for x in args.top_k_list.split(",")
            if x.strip()
        ]
    else:
        top_k_values = [args.top_k]

    if not top_k_values:
        raise ValueError("No top-k values supplied.")

    if any(k <= 0 for k in top_k_values):
        raise ValueError("All top-k values must be > 0.")

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    benchmark = load_benchmark(args.benchmark)
    resources = load_resources(args.resources)

    if args.query_id is not None:
        benchmark = [
            item
            for item in benchmark
            if item.get("query_id") == args.query_id
        ]

    if args.limit is not None:
        benchmark = benchmark[:args.limit]

    if not benchmark:
        raise ValueError("No benchmark queries selected.")

    print()
    print("=" * 80)
    print("TRAINING-FREE OPTISET")
    print("=" * 80)
    print(f"Embedding model : {args.embedding_model}")
    print(f"LLM             : {args.llm_model}")
    print(f"Initial K       : {top_k_values}")
    print(f"Final max       : {MAX_FINAL_RESOURCES}")
    print(f"Subqueries      : {args.num_subqueries}")
    print(f"Temperature     : {args.temperature}")
    print(f"Queries         : {len(benchmark)}")
    print(f"Resources       : {len(resources)}")
    print(f"Retrieval repr  : {args.retrieval_fields}")
    print(f"Prompt repr     : {args.prompt_fields}  "
          f"(selection AND refinement)")
    if args.retrieval_fields != "all" and args.prompt_fields == "all":
        print("  [!] The index is summary-only but the LLM still reads seven "
              "labelled fields.\n      That is not a summary-only run. Pass "
              "--prompt-fields summary as well.")
    print("=" * 80)

    # ---------------------------------------------------------------
    # Load BGE and encode corpus ONCE.
    #
    # This is important for K sweeps:
    # top-10/top-20/top-40/... all use the same corpus embeddings.
    # ---------------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading BGE on {device}...")

    encoder = SentenceTransformer(
        args.embedding_model,
        device=device,
    )

    resource_embeddings = encode_resource_corpus(
        encoder=encoder,
        resources=resources,
        batch_size=args.embedding_batch_size,
        field_mode=args.retrieval_fields,
    )

    # ---------------------------------------------------------------
    # Load Llama once.
    # ---------------------------------------------------------------

    print("\nLoading Llama through Transformers (device_map=auto)...")

    tokenizer, model = make_llm(
        model_name=args.llm_model,
    )

    # ---------------------------------------------------------------
    # Run every K independently.
    # ---------------------------------------------------------------

    for k in top_k_values:

        print()
        print("=" * 80)
        print(f"OPTISET: INITIAL TOP-K = {k}")
        print("=" * 80)

        results = []
        stats: Counter = Counter()
        sizes: list[int] = []

        for query_number, item in enumerate(
            benchmark,
            start=1,
        ):
            query_id = item["query_id"]
            question = item["query"]

            print(
                f"[{query_number}/{len(benchmark)}] "
                f"{query_id} | {question[:100]}"
            )

            try:
                result = run_optiset_for_query(
                    tokenizer=tokenizer,
                    model=model,
                    encoder=encoder,
                    resource_embeddings=resource_embeddings,
                    resources=resources,
                    query_id=query_id,
                    question=question,
                    top_k=k,
                    num_subqueries=args.num_subqueries,
                    temperature=args.temperature,
                    prompt_fields=args.prompt_fields,
                    stats=stats,
                )

                results.append(result)
                sizes.append(len(result["predicted_resource_ids"]))

                print(
                    "    selected:",
                    result["predicted_resource_ids"],
                )

            except Exception as exc:
                # One failed query must not kill the entire benchmark run.
                stats["query_errors"] += 1
                print(
                    f"    [ERROR] {query_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

        # -----------------------------------------------------------
        # Output filename
        # -----------------------------------------------------------

        if len(top_k_values) == 1:
            output_path = args.output

        else:
            base = Path(args.output)

            suffix = base.suffix if base.suffix else ".json"

            output_path = str(
                base.with_name(
                    f"{base.stem}_k{k}{suffix}"
                )
            )

        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                results,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print()
        print(f"Written: {output_path}")
        print(f"Queries completed: {len(results)}")
        print(f"prompt representation : {args.prompt_fields}")
        print(f"mean |R_q|            : "
              f"{sum(sizes)/max(1,len(sizes)):.4f}")

        n = max(1, len(results))
        print(f"\nparse paths over {len(results)} queries")
        for stage in ("select", "refine"):
            hdr = stats[f"{stage}_header"]
            fb = stats[f"{stage}_fallback_bracket"]
            bare = stats[f"{stage}_fallback_bare"]
            print(f"  {stage:<7s} header {hdr:>4d}  "
                  f"last-bracket-line {fb:>4d}  bare-integers {bare:>4d}  "
                  f"empty {stats[f'{stage}_empty']:>4d}")
        if stats["select_fallback_bare"] or stats["refine_fallback_bare"]:
            print("  [WARN] bare-integer parses are guesses about what the "
                  "model meant.\n         Read a few by hand before the "
                  "number goes in a table.")
        if stats["select_defaulted_to_rank1"]:
            print(f"  [WARN] {stats['select_defaulted_to_rank1']} queries "
                  f"produced NO parseable selection and fell back to the\n"
                  f"         rank-1 candidate. Those are Top-1, not OptiSet.")
        if stats["refine_kept_selection"]:
            print(f"  note : {stats['refine_kept_selection']} queries had no "
                  f"parseable refinement and kept the\n         selection "
                  f"stage's set unchanged.")
        if stats["query_errors"]:
            print(f"  [WARN] {stats['query_errors']} queries raised and are "
                  f"ABSENT from the output file.\n         An evaluator that "
                  f"averages over the file's rows will silently score a "
                  f"smaller\n         query set than the other baselines.")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
