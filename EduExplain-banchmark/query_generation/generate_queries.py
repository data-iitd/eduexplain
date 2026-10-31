"""
generate_queries.py

Stage 8: Query Generation

Reads:
    data/test_bundles.json

Writes:
    outputs/generated_queries_test.json

Uses:
    prompts/system_prompt.txt
    prompts/single_prompt.txt
    prompts/pair_prompt.txt
    prompts/triple_prompt.txt
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from config import *


# ==========================================================
# Prompt Loading
# ==========================================================

def load_prompt(path: Path):

    return path.read_text(
        encoding="utf-8"
    )


SYSTEM_PROMPT = load_prompt(
    SYSTEM_PROMPT
)

PROMPTS = {}

for bundle_type, path in PROMPT_FILES.items():

    PROMPTS[bundle_type] = load_prompt(path)


# ==========================================================
# Model
# ==========================================================

def load_model(model_name):

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(

        model_name,

        trust_remote_code=True

    )

    model = AutoModelForCausalLM.from_pretrained(

        model_name,

        torch_dtype=(
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        ),

        device_map="auto",

        trust_remote_code=True,

    )

    model.eval()

    print("Model loaded.")

    return tokenizer, model


# ==========================================================
# Generation
# ==========================================================

def generate(

    tokenizer,

    model,

    system_prompt,

    user_prompt,

    max_new_tokens=2048,

):

    messages = [

        {

            "role": "system",

            "content": system_prompt,

        },

        {

            "role": "user",

            "content": user_prompt,

        }

    ]

    text = tokenizer.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=True,

    )

    inputs = tokenizer(

        [text],

        return_tensors="pt",

    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=max_new_tokens,

            do_sample=True,

            temperature=TEMPERATURE,

            top_p=TOP_P,

            repetition_penalty=REPETITION_PENALTY,

            pad_token_id=tokenizer.eos_token_id,

        )

    generated = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(

        generated,

        skip_special_tokens=True

    ).strip()


# ==========================================================
# Helper Functions
# ==========================================================

# def majority_grade(bundle):

#     return int(

#         max(

#             bundle["class_distribution"].items(),

#             key=lambda x: x[1]

#         )[0]

#     )
def majority_grade(bundle):
    distribution = bundle.get("class_distribution", {})

    if not distribution:
        return None

    return int(
        max(
            distribution.items(),
            key=lambda x: x[1]
        )[0]
    )


# def subject(bundle):

#     return bundle["resources"][0]["subject"]

def subject(bundle):

    resources = bundle.get("resources") or []

    if not resources:
        return None

    return resources[0].get("subject")


def concepts(bundle):

    return [

        c["concept_id"]

        for c in bundle["concepts"]

    ]


def concept_names(bundle):

    return ", ".join(

        c["display_name"]

        for c in bundle["concepts"]

    )


def resource_text(bundle):

    blocks = []

    for idx, resource in enumerate(

        bundle["resources"],

        start=1,

    ):

        summary = (

            resource.get("description")

            or

            resource.get("summary")

            or

            ""

        )

        block = f"""
RESOURCE {idx}

Type:
{resource["resource_type"]}

Chapter:
{resource["chapter_title"]}

Section:
{resource.get("section")}

Summary:
{summary}
"""

        blocks.append(block)

    return "\n".join(blocks)


# ==========================================================
# Prompt Builder
# ==========================================================

def build_prompt(bundle):

    prompt = PROMPTS[
        bundle["bundle_type"]
    ]

    replacements = {

        "{{PRIMARY_CONCEPT}}":

            bundle["primary_concept"]["display_name"],

        "{{CONCEPTS}}":

            concept_names(bundle),

        "{{GRADE}}":

            str(

                majority_grade(bundle)

            ),

        "{{SUBJECT}}":

            subject(bundle),

        "{{RESOURCES}}":

            resource_text(bundle),

    }

    for key, value in replacements.items():

        prompt = prompt.replace(

            key,

            value

        )

    return prompt


# ==========================================================
# Output Parsing
# ==========================================================

def parse_output(raw_output):

    text = re.sub(

        r"```(?:json)?",

        "",

        raw_output,

    )

    text = text.replace(

        "```",

        ""

    ).strip()

    match = re.search(

        r"\[.*\]",

        text,

        re.DOTALL,

    )

    if match is None:

        return None

    try:

        parsed = json.loads(

            match.group()

        )

    except Exception:

        return None

    if not isinstance(

        parsed,

        list,

    ):

        return None

    return parsed
# ==========================================================
# Validation
# ==========================================================

VALID_QUERY_TYPES = {
    "explanation",
    "why/how",
    "misconception",
    "visual",    
    "example",
    "revision",
    "practice",
}

VALID_VARIANT_STYLES = {
    "search",
    "chat",
    "learning_context",
}


# def validate_queries(queries):

#     if queries is None:
#         return False

#     if not isinstance(queries, list):
#         return False

#     if len(queries) != 5:
#         return False

#     for q in queries:

#         if not isinstance(q, dict):
#             return False

#         if "query_type" not in q:
#             return False

#         if "query_variants" not in q:
#             return False

#         if q["query_type"] not in VALID_QUERY_TYPES:
#             return False

#         variants = q["query_variants"]

#         if not isinstance(variants, list):
#             return False

#         if len(variants) != 3:
#             return False

#         seen_styles = set()

#         for variant in variants:

#             if not isinstance(variant, dict):
#                 return False

#             if "style" not in variant:
#                 return False

#             if "query" not in variant:
#                 return False

#             if variant["style"] not in VALID_VARIANT_STYLES:
#                 return False

#             if variant["style"] in seen_styles:
#                 return False

#             seen_styles.add(variant["style"])

#             if not isinstance(variant["query"], str):
#                 return False

#             if len(variant["query"].strip()) == 0:
#                 return False

#     return True

def validate_queries(queries):

    if queries is None:
        return False, "parsed output is None"

    if not isinstance(queries, list):
        return False, f"expected list, got {type(queries).__name__}"

    # if len(queries) != 5:
    #     return False, f"expected 5 queries, got {len(queries)}"

    for i, q in enumerate(queries):

        if not isinstance(q, dict):
            return False, (
                f"query {i}: expected dict, "
                f"got {type(q).__name__}"
            )

        if "query_type" not in q:
            return False, f"query {i}: missing query_type"

        if "query_variants" not in q:
            return False, f"query {i}: missing query_variants"

        if q["query_type"] not in VALID_QUERY_TYPES:
            return False, (
                f"query {i}: invalid query_type "
                f"{q['query_type']!r}"
            )

        variants = q["query_variants"]

        if not isinstance(variants, list):
            return False, (
                f"query {i}: query_variants is "
                f"{type(variants).__name__}"
            )

        if len(variants) != 3:
            return False, (
                f"query {i} ({q['query_type']}): "
                f"expected 3 variants, got {len(variants)}"
            )

        seen_styles = set()

        for j, variant in enumerate(variants):

            if not isinstance(variant, dict):
                return False, (
                    f"query {i}, variant {j}: "
                    f"expected dict"
                )

            if "style" not in variant:
                return False, (
                    f"query {i}, variant {j}: "
                    f"missing style"
                )

            if "query" not in variant:
                return False, (
                    f"query {i}, variant {j}: "
                    f"missing query"
                )

            style = variant["style"]

            if style not in VALID_VARIANT_STYLES:
                return False, (
                    f"query {i}, variant {j}: "
                    f"invalid style {style!r}"
                )

            if style in seen_styles:
                return False, (
                    f"query {i}: duplicate style "
                    f"{style!r}"
                )

            seen_styles.add(style)

            if not isinstance(variant["query"], str):
                return False, (
                    f"query {i}, variant {j}: "
                    f"query is not a string"
                )

            if not variant["query"].strip():
                return False, (
                    f"query {i}, variant {j}: "
                    f"query is empty"
                )

    return True, "valid"


# ==========================================================
# Metadata Injection
# ==========================================================

def inject_metadata(bundle, queries, start_id):

    final_queries = []

    grade = majority_grade(bundle)

    subj = subject(bundle)

    concept_ids = concepts(bundle)

    bundle_type = bundle["bundle_type"]

    for i, q in enumerate(queries):

        final_queries.append({

            "query_id":
                f"Q{start_id+i:06d}",

            "bundle_id":
                bundle["bundle_id"],

            "query_type":
                q["query_type"],

            "query_variants":
                q["query_variants"],

            "grade":
                grade,

            "subject":
                subj,

            "concepts":
                concept_ids,

            "bundle_type":
                bundle_type,

            "gold_sets":
                []

        })

    return final_queries


# ==========================================================
# Resume Previous Run
# ==========================================================

def load_previous_output():

    if not OUTPUT_JSON.exists():

        return []

    with open(

        OUTPUT_JSON,

        "r",

        encoding="utf-8"

    ) as f:

        return json.load(f)

# ==========================================================
# Empty Resource Tracking
# ==========================================================

EMPTY_RESOURCE_LOG = (
    OUTPUT_JSON.parent / "bundles_without_resources.json"
)


def load_empty_resource_log():
    if not EMPTY_RESOURCE_LOG.exists():
        return []

    try:
        with open(
            EMPTY_RESOURCE_LOG,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

    except Exception as e:
        print(
            f"WARNING: Could not load empty-resource log: {e}"
        )

    return []


empty_resource_bundles = load_empty_resource_log()


def save_empty_resource_log():

    EMPTY_RESOURCE_LOG.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        EMPTY_RESOURCE_LOG,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            empty_resource_bundles,
            f,
            indent=2,
            ensure_ascii=False
        )


def track_empty_resource_bundle(bundle):

    bundle_id = bundle.get("bundle_id")

    # Avoid duplicates if the script is resumed/re-run.
    existing_ids = {
        item.get("bundle_id")
        for item in empty_resource_bundles
    }

    if bundle_id in existing_ids:
        return

    primary = bundle.get(
        "primary_concept",
        {}
    )

    tracked = {
        "bundle_id": bundle_id,

        "bundle_type": bundle.get(
            "bundle_type"
        ),

        "primary_concept": {
            "concept_id": primary.get(
                "concept_id"
            ),
            "display_name": primary.get(
                "display_name"
            )
        },

        "concepts": [
            {
                "concept_id": c.get(
                    "concept_id"
                ),
                "display_name": c.get(
                    "display_name"
                )
            }
            for c in bundle.get(
                "concepts",
                []
            )
        ],

        "class_distribution": bundle.get(
            "class_distribution",
            {}
        ),

        "resource_count": len(
            bundle.get(
                "resources",
                []
            ) or []
        ),

        "reason": "bundle has no resources"
    }

    empty_resource_bundles.append(tracked)


generated_queries = load_previous_output()

generated_bundle_ids = {

    q["bundle_id"]

    for q in generated_queries

}

next_query_id = len(generated_queries) + 1


print()

print("="*60)

print(

    f"Already generated bundles : "

    f"{len(generated_bundle_ids)}"

)

print(

    f"Existing queries          : "

    f"{len(generated_queries)}"

)

print("="*60)


# ==========================================================
# Save
# ==========================================================

def save_output(all_queries):

    OUTPUT_JSON.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    with open(

        OUTPUT_JSON,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            all_queries,

            f,

            indent=2,

            ensure_ascii=False,

        )


# ==========================================================
# Bundle Generation
# ==========================================================

def generate_bundle_queries(

    tokenizer,

    model,

    bundle,

):

    prompt = build_prompt(bundle)

    for attempt in range(MAX_RETRIES):

        print(

            f"Attempt "

            f"{attempt+1}/{MAX_RETRIES}"

        )

        raw_output = generate(

            tokenizer,

            model,

            SYSTEM_PROMPT,

            prompt,

        )

        parsed = parse_output(

            raw_output

        )

        # if validate_queries(

        #     parsed

        # ):

        #     return parsed

        # print(

        #     "Invalid output. Retrying..."

        # )
        valid, reason = validate_queries(parsed)

        if valid:
            return parsed

        print(
            f"Invalid output: {reason}"
        )
        print("\n--- RAW MODEL OUTPUT ---")
        print(raw_output)
        print("--- PARSED OUTPUT ---")
        print(
            json.dumps(
                parsed,
                indent=2,
                ensure_ascii=False
            )
            if parsed is not None
            else "None"
        )
        print("--- VALIDATION ERROR ---")
        print(reason)

        print("Retrying...")

    return None
# ==========================================================
# Main
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--input",

        type=str,

        default=str(INPUT_JSON),

    )

    args = parser.parse_args()

    input_path = Path(args.input)

    with open(

        input_path,

        "r",

        encoding="utf-8",

    ) as f:

        bundles = json.load(f)

    print(f"Loaded {len(bundles)} bundles.")

    tokenizer, model = load_model(MODEL_NAME)

    global next_query_id

    total_processed = 0
    total_empty_resources = 0

    total_skipped = 0

    total_failed = 0

    start_time = datetime.now()

    # for bundle in bundles:

    #     bundle_id = bundle["bundle_id"]

    #     if bundle_id in generated_bundle_ids:

    #         total_skipped += 1

    #         print(f"[SKIP] {bundle_id}")

    #         continue

    #     print()

    #     print("=" * 80)

    #     print(f"Bundle : {bundle_id}")

    #     print(

    #         f"Type   : {bundle['bundle_type']}"

    #     )

    #     print(

    #         f"Concept: "

    #         f"{bundle['primary_concept']['display_name']}"

    #     )

    #     print("=" * 80)
    for bundle in bundles:

        bundle_id = bundle.get("bundle_id")

        if bundle_id in generated_bundle_ids:
            total_skipped += 1
            print(f"[SKIP] {bundle_id}")
            continue

        # --------------------------------------------------
        # EMPTY RESOURCE CHECK
        # --------------------------------------------------

        resources = bundle.get("resources") or []

        if len(resources) == 0:

            primary = bundle.get(
                "primary_concept",
                {}
            )
            total_empty_resources += 1

            print()
            print("=" * 80)
            print(f"Bundle : {bundle_id}")
            print(
                "Concept: "
                f"{primary.get('display_name', 'UNKNOWN')}"
            )
            print("=" * 80)
            print(
                "[NO RESOURCES] "
                "Tracking concept and skipping generation."
            )

            track_empty_resource_bundle(
                bundle
            )

            total_skipped += 1

            # Save immediately so information isn't lost
            # if the model crashes later.
            save_empty_resource_log()

            continue

        # --------------------------------------------------
        # NORMAL BUNDLE
        # --------------------------------------------------

        print()
        print("=" * 80)

        print(
            f"Bundle : {bundle_id}"
        )

        print(
            f"Type   : "
            f"{bundle.get('bundle_type')}"
        )

        print(
            f"Concept: "
            f"{bundle.get('primary_concept', {}).get('display_name')}"
        )

        print("=" * 80)

        queries = generate_bundle_queries(

            tokenizer,

            model,

            bundle,

        )

        if queries is None:

            print(

                "FAILED after retries."

            )

            total_failed += 1

            continue

        final_queries = inject_metadata(

            bundle,

            queries,

            next_query_id,

        )

        next_query_id += len(final_queries)

        generated_queries.extend(

            final_queries

        )

        generated_bundle_ids.add(

            bundle_id

        )

        save_output(

            generated_queries

        )

        total_processed += 1

        print(

            f"Generated "

            f"{len(final_queries)} queries."

        )

    duration = datetime.now() - start_time

    print()

    print("=" * 80)

    print("QUERY GENERATION COMPLETE")

    print("=" * 80)

    print(

        f"Processed : {total_processed}"

    )

    print(
        f"Skipped   : "
        f"{total_skipped}"
    )

    print(
        f"No resources: "
        f"{len(empty_resource_bundles)}"
    )

    print(

        f"Failed    : {total_failed}"

    )

    print(

        f"Queries   : {len(generated_queries)}"

    )

    print(

        f"Duration  : {duration}"

    )

        # Save one final time.
    save_empty_resource_log()

    print()
    print("=" * 80)
    print("EMPTY RESOURCE REPORT")
    print("=" * 80)

    print(
        f"Bundles with no resources : "
        f"{len(empty_resource_bundles)}"
    )

    print(
        f"Saved to                   : "
        f"{EMPTY_RESOURCE_LOG}"
    )

    if empty_resource_bundles:

        print()
        print("Concepts without resources:")

        for item in empty_resource_bundles:

            primary = item.get(
                "primary_concept",
                {}
            )

            print(
                f"  {item.get('bundle_id')} : "
                f"{primary.get('concept_id')} : "
                f"{primary.get('display_name')}"
            )

    print("=" * 80)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()
