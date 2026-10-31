import json
import random
from pathlib import Path
from collections import defaultdict

# ==========================================================
# Configuration
# ==========================================================
SCRIPT_DIR=Path(__file__).resolve().parent
INPUT_JSON = SCRIPT_DIR/"generated_queries_hierarchial_new.json"
MAIN_OUTPUT = SCRIPT_DIR/"main_queries_new.json"

ROBUSTNESS_OUTPUT = SCRIPT_DIR/"robustness_queries_new.json"

RANDOM_SEED = 42

TARGET_COUNTS = {

    "single": 300,

    "pair": 300,

    "triple": 300,

}

QUERY_TYPES = [

    "explanation",

    "why/how",

    "example",

    "revision",

    "practice",

    "visual", 
    
    "misconception"

]

VARIANT_STYLES = [

    "search",

    "chat",

    "learning_context",

]

random.seed(RANDOM_SEED)

# ==========================================================
# Load Queries
# ==========================================================

with open(
    INPUT_JSON,
    "r",
    encoding="utf-8"
) as f:

    queries = json.load(f)

print(f"Loaded {len(queries)} parent queries.")

# ==========================================================
# Bucket Queries
# ==========================================================

buckets = defaultdict(list)

for query in queries:

    key = (

        query["bundle_type"],

        query["query_type"]

    )

    buckets[key].append(query)

# ==========================================================
# Sample Queries
# ==========================================================

selected_queries = []

print("\nSelecting benchmark queries...\n")

for bundle_type, target in TARGET_COUNTS.items():

    required = target // len(QUERY_TYPES)

    for query_type in QUERY_TYPES:

        key = (

            bundle_type,

            query_type

        )

        bucket = buckets.get(key, [])

        random.shuffle(bucket)

        if len(bucket) < required:

            print(

                f"[WARNING] "

                f"{bundle_type} / {query_type}: "

                f"Needed {required}, "

                f"Found {len(bucket)}."

            )

            selected = bucket

        else:

            selected = bucket[:required]

        selected_queries.extend(selected)

print()

print(

    f"Selected "

    f"{len(selected_queries)} "

    f"parent queries."

)

# ==========================================================
# Balanced Variant Assignment
# ==========================================================

style_schedule = []

for style in VARIANT_STYLES:

    style_schedule.extend(

        [style] *

        (len(selected_queries) // len(VARIANT_STYLES))

    )

remaining = (

    len(selected_queries)

    -

    len(style_schedule)

)

for i in range(remaining):

    style_schedule.append(

        VARIANT_STYLES[i]

    )

random.shuffle(style_schedule)

# ==========================================================
# Helper Functions
# ==========================================================

def variant_by_style(

    variants,

    style,

):

    for variant in variants:

        if variant["style"] == style:

            return variant

    return None


def remaining_variants(

    variants,

    chosen_style,

):

    remaining = []

    for variant in variants:

        if variant["style"] != chosen_style:

            remaining.append(variant)

    return remaining

# ==========================================================
# Build Benchmark Files
# ==========================================================

main_queries = []

robustness_queries = []

style_counter = defaultdict(int)

for idx, query in enumerate(selected_queries):

    chosen_style = style_schedule[idx]

    main_variant = variant_by_style(

        query["query_variants"],

        chosen_style,

    )

    if main_variant is None:

        print(

            f"[WARNING] "

            f"{query['query_id']} "

            f"does not contain "

            f"{chosen_style} variant."

        )

        continue

    style_counter[chosen_style] += 1

    # -------------------------------
    # Main benchmark
    # -------------------------------

    main_queries.append(

        {

            "query_id":

                query["query_id"],

            "bundle_id":

                query["bundle_id"],

            "query":

                main_variant["query"],

            "variant_style":

                main_variant["style"],

            "query_type":

                query["query_type"],

            "grade":

                query["grade"],

            "subject":

                query["subject"],

            "concepts":

                query["concepts"],

            "bundle_type":

                query["bundle_type"],

            "gold_sets":

                []

        }

    )

    # -------------------------------
    # Robustness benchmark
    # -------------------------------

    remaining = remaining_variants(

        query["query_variants"],

        chosen_style,

    )

    for r_idx, variant in enumerate(remaining, start=1):

        robustness_queries.append(

            {

                "query_id":

                    f"{query['query_id']}_R{r_idx}",

                "parent_query_id":

                    query["query_id"],

                "bundle_id":

                    query["bundle_id"],

                "query":

                    variant["query"],

                "variant_style":

                    variant["style"],

                "query_type":

                    query["query_type"],

                "grade":

                    query["grade"],

                "subject":

                    query["subject"],

                "concepts":

                    query["concepts"],

                "bundle_type":

                    query["bundle_type"],

                "gold_sets":

                    []

            }

        )

# ==========================================================
# Save Outputs
# ==========================================================

MAIN_OUTPUT.parent.mkdir(

    parents=True,

    exist_ok=True,

)

with open(

    MAIN_OUTPUT,

    "w",

    encoding="utf-8",

) as f:

    json.dump(

        main_queries,

        f,

        indent=2,

        ensure_ascii=False,

    )

with open(

    ROBUSTNESS_OUTPUT,

    "w",

    encoding="utf-8",

) as f:

    json.dump(

        robustness_queries,

        f,

        indent=2,

        ensure_ascii=False,

    )

# ==========================================================
# Statistics
# ==========================================================

print()

print("=" * 60)

print("Benchmark Construction Complete")

print("=" * 60)

print(

    f"Main Benchmark Queries      : {len(main_queries)}"

)

print(

    f"Robustness Queries          : {len(robustness_queries)}"

)

print()

print("Main Variant Distribution:")

for style in VARIANT_STYLES:

    print(

        f"  {style:<18}"

        f"{style_counter[style]}"

    )

print()

print(f"Saved main benchmark to       : {MAIN_OUTPUT}")

print(f"Saved robustness benchmark to : {ROBUSTNESS_OUTPUT}")

print("=" * 60)
