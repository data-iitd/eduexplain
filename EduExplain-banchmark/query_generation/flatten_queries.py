import json
from pathlib import Path
SCRIPT_DIR=Path(__file__).resolve().parent

INPUT_PATH = SCRIPT_DIR/"generated_queries_hierarchial_new.json"
OUTPUT_PATH = SCRIPT_DIR/"generated_queries_new.json"


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    hierarchical_queries = json.load(f)


flattened_queries = []

for item in hierarchical_queries:

    parent_id = item["query_id"]

    for idx, variant in enumerate(item["query_variants"], start=1):

        flattened_queries.append({

            "query_id":
                f"{parent_id}_V{idx}",

            "parent_query_id":
                parent_id,

            "variant_style":
                variant["style"],

            "query":
                variant["query"],

            "grade":
                item["grade"],

            "subject":
                item["subject"],

            "concepts":
                item["concepts"],

            "bundle_type":
                item["bundle_type"],

            "query_type":
                item["query_type"],

            "gold_sets":
                item["gold_sets"]

        })


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    json.dump(

        flattened_queries,

        f,

        indent=2,

        ensure_ascii=False

    )

print(f"Flattened {len(hierarchical_queries)} parent queries.")
print(f"Generated {len(flattened_queries)} benchmark queries.")
print(f"Saved to {OUTPUT_PATH}")
