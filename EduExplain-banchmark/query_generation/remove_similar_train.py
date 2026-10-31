#!/usr/bin/env python3

"""
Detect semantically similar queries between an EduExplain train set
and test set.

Pipeline
--------
1. Load train and test JSON.
2. Reproduce the concept merging/canonicalization logic used for the
   EduExplain concept merger.
3. Extract concepts from:
      - train records: record["concepts"]
      - test records: union of concepts from all gold_sets/resources_used
4. Candidate generation:
      overlap coefficient >= 50%
5. For candidates, ask Llama-3.1-70B-Instruct whether the two queries
   express the same/highly similar learner information need.
6. Remove similar train queries.
7. Save:
      - cleaned train set
      - train/test similar-query mapping
      - optional candidate diagnostics

The TEST SET IS NEVER MODIFIED.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "~/models/Llama-3.1-70B-Instruct"

DEFAULT_CONCEPTS = (
    "./data/"
    "extended_concepts.json"
)

DEFAULT_THRESHOLD = 0.50


# ============================================================
# CONCEPT CANONICALIZATION
# EXACTLY THE SAME LOGIC AS YOUR MERGER SCRIPT
# ============================================================

LEADING_ARTICLES = {
    "a",
    "an",
    "the",
}


def canonicalize(text):
    """
    Normalize a concept name/ID for duplicate detection.

    This reproduces the logic from the supplied concept merger:

      - leading section labels: (a), (b), (c), (d), ...
      - numeric section labels: (1), (2), ...
      - leading articles: a, an, the
      - possessives: Newton's -> Newtons
      - punctuation
      - whitespace
      - underscores/hyphens
      - conversion to snake_case concept ID
    """

    text = str(text).strip().lower()

    # --------------------------------------------------------
    # Normalize apostrophe variants.
    # --------------------------------------------------------

    text = text.replace("’", "'")
    text = text.replace("`", "'")

    # --------------------------------------------------------
    # Remove leading parenthetical/letter labels.
    #
    # Examples:
    #   (a) Reproductive Health
    #   (d) Reproductive Health
    #   (iv) Some Topic
    # --------------------------------------------------------

    text = re.sub(
        r"^\s*\(\s*[a-z]+\s*\)\s*",
        "",
        text,
    )

    # --------------------------------------------------------
    # Numeric labels.
    #
    # Examples:
    #   (1) Topic
    #   (2) Topic
    # --------------------------------------------------------

    text = re.sub(
        r"^\s*\(\s*\d+\s*\)\s*",
        "",
        text,
    )

    # --------------------------------------------------------
    # Normalize possessives.
    #
    # Newton's -> Newtons
    # Kohlrausch's -> Kohlrauschs
    # --------------------------------------------------------

    text = re.sub(r"'s\b", "s", text)

    # Remove remaining apostrophes.
    text = text.replace("'", "")

    # --------------------------------------------------------
    # Replace punctuation/separators with spaces.
    # --------------------------------------------------------

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    # --------------------------------------------------------
    # Collapse whitespace.
    # --------------------------------------------------------

    words = text.split()

    # --------------------------------------------------------
    # Remove leading articles.
    #
    # "The Male Reproductive System"
    # -> "Male Reproductive System"
    # --------------------------------------------------------

    while words and words[0] in LEADING_ARTICLES:
        words.pop(0)

    # --------------------------------------------------------
    # Convert to concept ID.
    # --------------------------------------------------------

    return "_".join(words)


def concept_id_from_display_name(display_name):
    """
    Convert a display name into the canonical concept ID.
    """
    return canonicalize(display_name)


# ============================================================
# CHOOSE CANONICAL CONCEPT
# EXACTLY THE SAME LOGIC AS YOUR MERGER SCRIPT
# ============================================================

def choose_canonical(concept_ids, concepts):
    """
    Select the concept ID that survives when multiple IDs refer
    to the same normalized concept.

    Preference order:

        1. Structural source
        2. Highest frequency
        3. Shorter ID
        4. Lexicographically smaller ID
    """

    def score(cid):
        info = concepts[cid]

        source = str(info.get("source", "")).lower()

        structural_priority = 0 if source == "structural" else 1

        frequency = info.get("frequency", 0)

        try:
            frequency = int(frequency)
        except (TypeError, ValueError):
            frequency = 0

        return (
            structural_priority,
            -frequency,
            len(cid),
            cid,
        )

    return sorted(concept_ids, key=score)[0]


# ============================================================
# BUILD DUPLICATE GROUPS
# ============================================================

def find_duplicate_groups(concepts):
    """
    Group concept IDs by their canonicalized display name.

    This is the same behavior as the supplied merger script.
    """

    groups = defaultdict(list)

    for concept_id, info in concepts.items():

        display_name = info.get(
            "display_name",
            concept_id,
        )

        key_from_id = canonicalize(concept_id)
        key_from_name = canonicalize(display_name)

        # Same as original merger:
        # display name is treated as more semantically meaningful.
        key = key_from_name

        groups[key].append(concept_id)

    duplicates = {
        key: ids
        for key, ids in groups.items()
        if len(ids) > 1
    }

    return duplicates


# ============================================================
# BUILD REMAP
# ============================================================

def build_remap(concepts, duplicate_groups):
    """
    Create:

        duplicate_id -> canonical_id
    """

    remap = {}

    for key, ids in duplicate_groups.items():

        canonical_id = choose_canonical(
            ids,
            concepts,
        )

        for concept_id in ids:

            if concept_id != canonical_id:
                remap[concept_id] = canonical_id

    return remap


# ============================================================
# LOAD / WRITE JSON
# ============================================================

def load_json(path):
    path = Path(path)

    print(f"Loading: {path}")

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Wrote: {path}")


# ============================================================
# APPLY CONCEPT REMAP
# ============================================================

def remap_concepts(concepts, remap):
    """
    Apply the exact duplicate-ID remapping.

    Also removes duplicates created by the merge.
    """

    if not isinstance(concepts, list):
        return []

    result = []

    for concept_id in concepts:

        canonical_id = remap.get(
            concept_id,
            concept_id,
        )

        result.append(canonical_id)

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(result))


# ============================================================
# EXTRACT TRAIN CONCEPTS
# ============================================================

def get_train_concepts(record, remap):
    """
    Train records already have a 'concepts' field.

    Example:

        {
            "query": "...",
            "concepts": [
                "half_life_of_a_reaction"
            ]
        }
    """

    concepts = record.get("concepts", [])

    if not isinstance(concepts, list):
        return set()

    concepts = remap_concepts(
        concepts,
        remap,
    )

    return {
        c
        for c in concepts
        if isinstance(c, str) and c.strip()
    }


# ============================================================
# EXTRACT TEST CONCEPTS FROM GOLD SET
# ============================================================

def get_test_concepts(record, remap):
    """
    Test records do not necessarily have a top-level concepts
    field.

    Instead, concepts are obtained from:

        gold_sets
            -> resources_used
                -> concepts

    All concepts appearing in the test query's gold resources
    are unioned together.
    """

    concepts = []

    gold_sets = record.get(
        "gold_sets",
        [],
    )

    if not isinstance(gold_sets, list):
        return set()

    for gold_set in gold_sets:

        if not isinstance(gold_set, dict):
            continue

        resources_used = gold_set.get(
            "resources_used",
            [],
        )

        if not isinstance(resources_used, list):
            continue

        for resource in resources_used:

            if not isinstance(resource, dict):
                continue

            resource_concepts = resource.get(
                "concepts",
                [],
            )

            if not isinstance(resource_concepts, list):
                continue

            concepts.extend(resource_concepts)

    concepts = remap_concepts(
        concepts,
        remap,
    )

    return {
        c
        for c in concepts
        if isinstance(c, str) and c.strip()
    }


# ============================================================
# CONCEPT OVERLAP
# ============================================================

def concept_overlap(train_concepts, test_concepts):
    """
    Calculate overlap coefficient:

        |intersection| / min(|train|, |test|)

    Example:

        train = {A, B}
        test  = {A, B, C}

        intersection = {A, B}

        overlap = 2 / min(2, 3)
                = 1.0

    This therefore counts as >= 50% overlap.
    """

    if not train_concepts or not test_concepts:
        return 0.0

    intersection = (
        train_concepts
        & test_concepts
    )

    denominator = min(
        len(train_concepts),
        len(test_concepts),
    )

    if denominator == 0:
        return 0.0

    return len(intersection) / denominator


# ============================================================
# LOAD LLAMA
# ============================================================

def load_llama(model_name):
    """
    Load Llama-3.1-70B-Instruct locally.

    device_map='auto' allows Transformers to distribute the model
    over the available GPUs/CPU according to available memory.
    """

    print()
    print("=" * 70)
    print("LOADING LLAMA")
    print("=" * 70)
    print(f"Model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prefer bfloat16 when supported.
    if torch.cuda.is_available():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()

    print("Model loaded.")

    return tokenizer, model


# ============================================================
# LLM PROMPT
# ============================================================

def build_similarity_prompt(
    train_query,
    test_query,
    train_concepts,
    test_concepts,
):
    """
    Ask the LLM whether the two queries express the same learner
    information need.

    The LLM receives concepts as supporting context but the final
    judgment is primarily semantic/query-level.
    """

    shared_concepts = sorted(
        train_concepts & test_concepts
    )

    return f"""
You are evaluating whether two educational benchmark queries are
semantically redundant.

Your task is to determine whether the TRAIN query and TEST query
ask for essentially the SAME learner information need.

A pair should be marked SIMILAR if:

- they ask for the same concept or knowledge,
- they seek essentially the same explanation,
- one is a paraphrase of the other,
- one asks the same thing using different wording,
- or the difference is only superficial wording/context.

A pair should NOT be marked SIMILAR if:

- they concern different aspects of the concepts,
- one asks for an example while the other asks for a definition,
- one asks for a comparison while the other asks for an unrelated fact,
- they merely share a topic but require substantially different information,
- or the concepts overlap but the actual learner information need is different.

The fact that concepts overlap is only a candidate-generation signal.
You must judge the meaning of the queries themselves.

TRAIN QUERY:
{train_query}

TEST QUERY:
{test_query}

TRAIN CONCEPTS:
{json.dumps(sorted(train_concepts), ensure_ascii=False)}

TEST CONCEPTS:
{json.dumps(sorted(test_concepts), ensure_ascii=False)}

SHARED CONCEPTS:
{json.dumps(shared_concepts, ensure_ascii=False)}

Return ONLY valid JSON in exactly this format:

{{
  "similar": true,
  "reason": "brief explanation"
}}

or

{{
  "similar": false,
  "reason": "brief explanation"
}}

Do not return markdown.
Do not return code fences.
Do not return any text outside the JSON object.
""".strip()


# ============================================================
# PARSE LLM OUTPUT
# ============================================================

def parse_llm_output(raw):
    """
    Robustly parse the LLM's JSON response.
    """

    if raw is None:
        return None

    text = raw.strip()

    # Remove code fences if model ignores instruction.
    text = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace(
        "```",
        "",
    ).strip()

    # Try direct JSON first.
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # Fallback: extract first JSON object.
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if not match:
        return None

    try:
        parsed = json.loads(
            match.group(0)
        )

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        return None

    return None


# ============================================================
# LLM SIMILARITY JUDGMENT
# ============================================================

@torch.inference_mode()
def llm_similarity(
    train_query,
    test_query,
    train_concepts,
    test_concepts,
    tokenizer,
    model,
):
    """
    Run one semantic comparison through Llama.
    """

    prompt = build_similarity_prompt(
        train_query=train_query,
        test_query=test_query,
        train_concepts=train_concepts,
        test_concepts=test_concepts,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict semantic similarity evaluator "
                "for educational queries."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for models/tokenizers without chat template.
        formatted = (
            "System: You are a strict semantic similarity "
            "evaluator for educational queries.\n\n"
            "User:\n"
            + prompt
            + "\n\nAssistant:"
        )

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )

    # Put inputs on the same device as the model's first device.
    if hasattr(model, "device"):
        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
        }
    else:
        inputs = {
            k: v.to("cuda")
            for k, v in inputs.items()
        }

    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = output_ids[
        0,
        inputs["input_ids"].shape[1]:,
    ]

    raw_output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    parsed = parse_llm_output(
        raw_output
    )

    if parsed is None:
        print()
        print("WARNING: Could not parse LLM output.")
        print("Raw output:")
        print(raw_output)
        return False, "unparseable LLM response"

    similar = parsed.get(
        "similar",
        False,
    )

    reason = parsed.get(
        "reason",
        "",
    )

    # Be strict about accepted values.
    if isinstance(similar, str):
        similar = (
            similar.lower().strip()
            in {
                "true",
                "yes",
                "similar",
            }
        )
    else:
        similar = bool(similar)

    return similar, str(reason)


# ============================================================
# MAIN SIMILARITY PIPELINE
# ============================================================

def find_similar_queries(
    train_data,
    test_data,
    remap,
    tokenizer,
    model,
    threshold=DEFAULT_THRESHOLD,
):
    """
    Find train/test query pairs that:

        1. Have >= threshold concept overlap.
        2. Are judged semantically similar by Llama.
    """

    # --------------------------------------------------------
    # Precompute train concepts.
    # --------------------------------------------------------

    train_items = []

    for train_index, record in enumerate(train_data):

        query = record.get(
            "query",
            "",
        )

        if not isinstance(query, str):
            continue

        query = query.strip()

        if not query:
            continue

        concepts = get_train_concepts(
            record,
            remap,
        )

        train_items.append(
            {
                "index": train_index,
                "query": query,
                "concepts": concepts,
                "query_id": record.get(
                    "query_id"
                ),
                "bundle_id": record.get(
                    "bundle_id"
                ),
            }
        )

    # --------------------------------------------------------
    # Precompute test concepts.
    # --------------------------------------------------------

    test_items = []

    for test_index, record in enumerate(test_data):

        query = record.get(
            "query",
            "",
        )

        if not isinstance(query, str):
            continue

        query = query.strip()

        if not query:
            continue

        concepts = get_test_concepts(
            record,
            remap,
        )

        test_items.append(
            {
                "index": test_index,
                "query": query,
                "concepts": concepts,
                "query_id": record.get(
                    "query_id"
                ),
            }
        )

    print()
    print("=" * 70)
    print("CONCEPT STATISTICS")
    print("=" * 70)

    print(
        f"Train queries with concepts : "
        f"{len(train_items):,}"
    )

    print(
        f"Test queries with concepts  : "
        f"{len(test_items):,}"
    )

    # --------------------------------------------------------
    # Candidate generation.
    # --------------------------------------------------------

    candidates = []

    for train_item in train_items:

        if not train_item["concepts"]:
            continue

        for test_item in test_items:

            if not test_item["concepts"]:
                continue

            overlap = concept_overlap(
                train_item["concepts"],
                test_item["concepts"],
            )

            if overlap >= threshold:

                candidates.append(
                    {
                        "train_index": train_item[
                            "index"
                        ],
                        "test_index": test_item[
                            "index"
                        ],
                        "train_query": train_item[
                            "query"
                        ],
                        "test_query": test_item[
                            "query"
                        ],
                        "train_query_id": train_item[
                            "query_id"
                        ],
                        "test_query_id": test_item[
                            "query_id"
                        ],
                        "train_concepts": sorted(
                            train_item[
                                "concepts"
                            ]
                        ),
                        "test_concepts": sorted(
                            test_item[
                                "concepts"
                            ]
                        ),
                        "shared_concepts": sorted(
                            train_item[
                                "concepts"
                            ]
                            & test_item[
                                "concepts"
                            ]
                        ),
                        "concept_overlap": overlap,
                    }
                )

    print()
    print("=" * 70)
    print("CONCEPT CANDIDATES")
    print("=" * 70)

    print(
        f"Candidates with >= "
        f"{threshold:.0%} concept overlap: "
        f"{len(candidates):,}"
    )

    # --------------------------------------------------------
    # LLM semantic filtering.
    # --------------------------------------------------------

    similar_pairs = []

    for i, candidate in enumerate(
        candidates,
        start=1,
    ):

        print()
        print("-" * 70)
        print(
            f"LLM COMPARISON "
            f"{i:,}/{len(candidates):,}"
        )
        print("-" * 70)

        print(
            f"Train ID : "
            f"{candidate['train_query_id']}"
        )

        print(
            f"Test ID  : "
            f"{candidate['test_query_id']}"
        )

        print(
            f"Concept overlap : "
            f"{candidate['concept_overlap']:.2%}"
        )

        print()
        print(
            f"TRAIN: {candidate['train_query']}"
        )

        print(
            f"TEST : {candidate['test_query']}"
        )

        similar, reason = llm_similarity(
            train_query=candidate[
                "train_query"
            ],
            test_query=candidate[
                "test_query"
            ],
            train_concepts=set(
                candidate[
                    "train_concepts"
                ]
            ),
            test_concepts=set(
                candidate[
                    "test_concepts"
                ]
            ),
            tokenizer=tokenizer,
            model=model,
        )

        candidate[
            "llm_similar"
        ] = similar

        candidate[
            "llm_reason"
        ] = reason

        print()
        print(
            f"LLM decision: "
            f"{'SIMILAR' if similar else 'NOT SIMILAR'}"
        )

        if reason:
            print(
                f"Reason: {reason}"
            )

        if similar:
            similar_pairs.append(
                candidate
            )

    return similar_pairs, candidates


# ============================================================
# BUILD OUTPUTS
# ============================================================

def build_clean_train(
    train_data,
    similar_pairs,
):
    """
    Remove every train query that was judged similar to at least
    one test query.
    """

    removed_indices = {
        pair["train_index"]
        for pair in similar_pairs
    }

    cleaned_train = [
        record
        for index, record in enumerate(
            train_data
        )
        if index not in removed_indices
    ]

    return cleaned_train, removed_indices


def build_mapping(similar_pairs):
    """
    Mapping contains ONLY query texts, as requested.

    Example:

    [
      {
        "train_query": "...",
        "test_query": "..."
      }
    ]
    """

    mapping = []

    seen = set()

    for pair in similar_pairs:

        key = (
            pair["train_query"],
            pair["test_query"],
        )

        if key in seen:
            continue

        seen.add(key)

        mapping.append(
            {
                "train_query": pair[
                    "train_query"
                ],
                "test_query": pair[
                    "test_query"
                ],
            }
        )

    return mapping


# ============================================================
# DIAGNOSTIC OUTPUT
# ============================================================

def build_diagnostics(candidates):
    """
    Save concept-overlap candidates together with the LLM decision.

    This is useful for checking whether the 50% filtering step is
    behaving correctly.
    """

    diagnostics = []

    for candidate in candidates:

        diagnostics.append(
            {
                "train_query": candidate[
                    "train_query"
                ],
                "test_query": candidate[
                    "test_query"
                ],
                "train_query_id": candidate[
                    "train_query_id"
                ],
                "test_query_id": candidate[
                    "test_query_id"
                ],
                "train_concepts": candidate[
                    "train_concepts"
                ],
                "test_concepts": candidate[
                    "test_concepts"
                ],
                "shared_concepts": candidate[
                    "shared_concepts"
                ],
                "concept_overlap": candidate[
                    "concept_overlap"
                ],
                "llm_similar": candidate.get(
                    "llm_similar"
                ),
                "llm_reason": candidate.get(
                    "llm_reason"
                ),
            }
        )

    return diagnostics


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Remove train queries that are semantically "
            "similar to queries in the test set."
        )
    )

    parser.add_argument(
        "--train",
        required=True,
        help="Path to train JSON.",
    )

    parser.add_argument(
        "--test",
        required=True,
        help="Path to test JSON.",
    )

    parser.add_argument(
        "--concepts",
        default=DEFAULT_CONCEPTS,
        help=(
            "Path to extended_concepts.json used for "
            "concept merging."
        ),
    )

    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Path to local Llama model.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Minimum concept overlap coefficient. "
            "Default: 0.50"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="similarity_outputs",
        help="Directory for output JSON files.",
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDATE
    # ========================================================

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(
            "--threshold must be between 0 and 1."
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    train_data = load_json(
        args.train
    )

    test_data = load_json(
        args.test
    )

    concept_data = load_json(
        args.concepts
    )

    if not isinstance(train_data, list):
        raise TypeError(
            "Train JSON must contain a top-level list."
        )

    if not isinstance(test_data, list):
        raise TypeError(
            "Test JSON must contain a top-level list."
        )

    if not isinstance(concept_data, dict):
        raise TypeError(
            "Concept JSON must contain a top-level "
            "dictionary keyed by concept ID."
        )

    print()
    print("=" * 70)
    print("DATASET SIZES")
    print("=" * 70)

    print(
        f"Train records : {len(train_data):,}"
    )

    print(
        f"Test records  : {len(test_data):,}"
    )

    print(
        f"Concepts      : {len(concept_data):,}"
    )

    # ========================================================
    # CONCEPT MERGE
    # ========================================================

    print()
    print("=" * 70)
    print("BUILDING CONCEPT REMAP")
    print("=" * 70)

    duplicate_groups = find_duplicate_groups(
        concept_data
    )

    remap = build_remap(
        concept_data,
        duplicate_groups,
    )

    print(
        f"Duplicate groups : "
        f"{len(duplicate_groups):,}"
    )

    print(
        f"IDs remapped     : "
        f"{len(remap):,}"
    )

    # ========================================================
    # SHOW SOME REMAPS
    # ========================================================

    if remap:

        print()
        print("Example concept remappings:")

        for duplicate_id, canonical_id in list(
            remap.items()
        )[:20]:

            print(
                f"  {duplicate_id} "
                f"-> {canonical_id}"
            )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    tokenizer, model = load_llama(
        args.model
    )

    # ========================================================
    # FIND SIMILAR
    # ========================================================

    similar_pairs, candidates = find_similar_queries(
        train_data=train_data,
        test_data=test_data,
        remap=remap,
        tokenizer=tokenizer,
        model=model,
        threshold=args.threshold,
    )

    # ========================================================
    # CLEAN TRAIN
    # ========================================================

    cleaned_train, removed_indices = (
        build_clean_train(
            train_data,
            similar_pairs,
        )
    )

    # ========================================================
    # MAPPING
    # ========================================================

    mapping = build_mapping(
        similar_pairs
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    diagnostics = build_diagnostics(
        candidates
    )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_train_path = (
        output_dir
        / "train_without_test_similar_new.json"
    )

    mapping_path = (
        output_dir
        / "similar_train_test_queries_new.json"
    )

    candidates_path = (
        output_dir
        / "similarity_candidates_diagnostics_new.json"
    )

    # ========================================================
    # WRITE
    # ========================================================

    write_json(
        cleaned_train_path,
        cleaned_train,
    )

    write_json(
        mapping_path,
        mapping,
    )

    write_json(
        candidates_path,
        diagnostics,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Original train queries : "
        f"{len(train_data):,}"
    )

    print(
        f"Concept-overlap pairs  : "
        f"{len(candidates):,}"
    )

    print(
        f"Semantically similar   : "
        f"{len(similar_pairs):,}"
    )

    print(
        f"Train queries removed  : "
        f"{len(removed_indices):,}"
    )

    print(
        f"Final train queries    : "
        f"{len(cleaned_train):,}"
    )

    print()
    print(
        f"Clean train : "
        f"{cleaned_train_path}"
    )

    print(
        f"Mapping     : "
        f"{mapping_path}"
    )

    print(
        f"Diagnostics : "
        f"{candidates_path}"
    )

    print()
    print("DONE")


if __name__ == "__main__":
    main()
