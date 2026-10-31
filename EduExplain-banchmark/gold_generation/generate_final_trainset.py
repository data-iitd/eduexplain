
#!/usr/bin/env python3

import json
from pathlib import Path


# ============================================================
# INPUT FILES
# ============================================================

TRAIN_VALIDATION_FILE = Path(
    "./LLM_validation/train_output.json"
)

GOLD_SETS_FILE = Path(
    "./gold_generation/gold_sets_corrected.json"
)

SIMILARITY_FILTERED_TRAIN_FILE = Path(
    "./query_generation/similarity_outputs/"
    "train_without_test_similar.json"
)

RESOURCES_FILE = Path(
    "./data/"
    "all_resources_unique_ids_clean1_merged.json"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_FILE = Path(
    "./LLM_validation/final_training_set.json"
)

REMOVED_GOLD_SETS_FILE = Path(
    "./LLM_validation/removed_gold_sets.json"
)

REMOVED_QUERIES_FILE = Path(
    "./LLM_validation/removed_queries.json"
)

MISSING_RESOURCES_FILE = Path(
    "./LLM_validation/missing_gold_resources.json"
)


# ============================================================
# RESOURCE FIELDS TO ADD
# ============================================================

RESOURCE_FIELDS = [
    "resource_type",
    "subject",
    "chapter_title",
    "summary",
    "description",
    "image_caption",
    "text",
]


# ============================================================
# VALIDATION CATEGORIES
# ============================================================

RATING_CATEGORIES = [
    "query_naturalness_educational_relevance",
    "query_category_correctness",
    "selected_resource_relevance",
    "explanation_completeness",
    "resource_redundancy",
    "grade_level_consistency",
]


# ============================================================
# LOAD / SAVE
# ============================================================

def load_json(path):
    print(f"Loading: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Wrote: {path}")


# ============================================================
# GOLD SET VALIDATION
# ============================================================

def count_zero_ratings(gold_set):
    """
    Count how many validation categories have rating == 0.
    """

    scores = gold_set.get(
        "scores",
        {}
    )

    zero_categories = []

    for category in RATING_CATEGORIES:

        value = scores.get(category)

        if value == 0:
            zero_categories.append(
                category
            )

    return zero_categories


def gold_set_survives(gold_set):
    """
    Keep gold set if it has fewer than two zero ratings.

        0 zeros -> KEEP
        1 zero  -> KEEP
        2+ zeros -> REMOVE
    """

    zero_categories = count_zero_ratings(
        gold_set
    )

    return len(zero_categories) < 2


# ============================================================
# PROCESS VALIDATION FILE
# ============================================================

def process_validation(train_validation):
    """
    Determine which gold_set_index values survive.

    Returns:

        surviving_gold_indices
        removed_gold_sets
        removed_queries
    """

    surviving_gold_indices = {}

    removed_gold_sets = []

    removed_queries = []

    for query_record in train_validation:

        query_id = query_record.get(
            "query_id"
        )

        gold_sets = query_record.get(
            "gold_sets",
            []
        )

        if not isinstance(
            gold_sets,
            list
        ):
            gold_sets = []

        surviving = set()

        for gold_set in gold_sets:

            gold_set_index = gold_set.get(
                "gold_set_index"
            )

            zero_categories = count_zero_ratings(
                gold_set
            )

            if len(zero_categories) >= 2:

                removed_gold_sets.append(
                    {
                        "query_id": query_id,
                        "gold_set_index": gold_set_index,
                        "query": query_record.get(
                            "query"
                        ),
                        "zero_count": len(
                            zero_categories
                        ),
                        "zero_categories": (
                            zero_categories
                        ),
                        "scores": gold_set.get(
                            "scores",
                            {}
                        ),
                    }
                )

            else:

                surviving.add(
                    gold_set_index
                )

        surviving_gold_indices[
            query_id
        ] = surviving

        if not surviving:

            removed_queries.append(
                {
                    "query_id": query_id,
                    "query": query_record.get(
                        "query"
                    ),
                    "reason": (
                        "No gold set survived "
                        "the two-zero rating filter."
                    ),
                    "original_gold_set_count": len(
                        gold_sets
                    ),
                }
            )

    return (
        surviving_gold_indices,
        removed_gold_sets,
        removed_queries,
    )


# ============================================================
# INDEX CORRECTED GOLD SETS
# ============================================================

def index_gold_sets(gold_sets_data):
    """
    Index corrected gold sets by:

        query_id
            -> gold_set_index
                -> gold_set

    gold_sets_corrected.json does not necessarily contain
    gold_set_index, so the list position is used.
    """

    indexed = {}

    for query_record in gold_sets_data:

        query_id = query_record.get(
            "query_id"
        )

        if query_id is None:
            continue

        indexed[query_id] = {}

        gold_sets = query_record.get(
            "gold_sets",
            []
        )

        if not isinstance(
            gold_sets,
            list
        ):
            continue

        for gold_set_index, gold_set in enumerate(
            gold_sets
        ):

            indexed[query_id][
                gold_set_index
            ] = gold_set

    return indexed


# ============================================================
# INDEX RESOURCES
# ============================================================

def index_resources(resources):
    """
    Create:

        resource_id -> resource

    This makes resource lookup O(1).
    """

    resource_index = {}

    for resource in resources:

        if not isinstance(
            resource,
            dict
        ):
            continue

        resource_id = resource.get(
            "resource_id"
        )

        if resource_id is None:
            continue

        resource_index[
            resource_id
        ] = resource

    return resource_index


# ============================================================
# ENRICH GOLD SET RESOURCES
# ============================================================

def enrich_gold_set(
    gold_set,
    resource_index,
    missing_resources,
    query_id,
    gold_set_index,
):
    """
    For every resource in resources_used, add:

        resource_type
        subject
        chapter_title
        summary
        description
        image_caption
        text

    using resource_id as the key.

    Existing fields are retained.
    """

    enriched_gold_set = dict(
        gold_set
    )

    resources_used = gold_set.get(
        "resources_used",
        []
    )

    if not isinstance(
        resources_used,
        list
    ):
        enriched_gold_set[
            "resources_used"
        ] = []

        return enriched_gold_set

    enriched_resources = []

    for resource in resources_used:

        if not isinstance(
            resource,
            dict
        ):
            continue

        # Make a copy so the original loaded data
        # is not modified.
        enriched_resource = dict(
            resource
        )

        resource_id = resource.get(
            "resource_id"
        )

        if resource_id is None:

            missing_resources.append(
                {
                    "query_id": query_id,
                    "gold_set_index": (
                        gold_set_index
                    ),
                    "resource_id": None,
                    "reason": (
                        "Resource entry has no "
                        "resource_id."
                    ),
                }
            )

            enriched_resources.append(
                enriched_resource
            )

            continue

        source_resource = resource_index.get(
            resource_id
        )

        if source_resource is None:

            missing_resources.append(
                {
                    "query_id": query_id,
                    "gold_set_index": (
                        gold_set_index
                    ),
                    "resource_id": resource_id,
                    "reason": (
                        "resource_id not found in "
                        "all_resources_unique_ids_clean1_merged.json."
                    ),
                }
            )

            enriched_resources.append(
                enriched_resource
            )

            continue

        # ----------------------------------------------------
        # Add the requested fields.
        # ----------------------------------------------------

        for field in RESOURCE_FIELDS:

            enriched_resource[
                field
            ] = source_resource.get(
                field
            )

        enriched_resources.append(
            enriched_resource
        )

    enriched_gold_set[
        "resources_used"
    ] = enriched_resources

    return enriched_gold_set


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BUILD FINAL TRAINING SET")
    print("=" * 80)

    # ========================================================
    # LOAD ALL FOUR FILES
    # ========================================================

    train_validation = load_json(
        TRAIN_VALIDATION_FILE
    )

    corrected_gold_sets = load_json(
        GOLD_SETS_FILE
    )

    similarity_filtered_train = load_json(
        SIMILARITY_FILTERED_TRAIN_FILE
    )

    resources = load_json(
        RESOURCES_FILE
    )

    # ========================================================
    # VALIDATE TOP-LEVEL STRUCTURES
    # ========================================================

    if not isinstance(
        train_validation,
        list
    ):
        raise TypeError(
            "train_output.json must contain a list."
        )

    if not isinstance(
        corrected_gold_sets,
        list
    ):
        raise TypeError(
            "gold_sets_corrected.json must contain a list."
        )

    if not isinstance(
        similarity_filtered_train,
        list
    ):
        raise TypeError(
            "train_without_test_similar.json "
            "must contain a list."
        )

    if not isinstance(
        resources,
        list
    ):
        raise TypeError(
            "all_resources_unique_ids_clean1_merged.json "
            "must contain a list."
        )

    print()
    print("Input sizes:")
    print(
        f"  Validation queries        : "
        f"{len(train_validation):,}"
    )
    print(
        f"  Corrected-gold queries    : "
        f"{len(corrected_gold_sets):,}"
    )
    print(
        f"  Similarity-filtered train : "
        f"{len(similarity_filtered_train):,}"
    )
    print(
        f"  Resources                 : "
        f"{len(resources):,}"
    )

    # ========================================================
    # STEP 1
    # VALIDATE GOLD SETS
    # ========================================================

    (
        surviving_gold_indices,
        removed_gold_sets,
        removed_queries_validation,
    ) = process_validation(
        train_validation
    )

    print()
    print("=" * 80)
    print("STEP 1: GOLD-SET VALIDATION FILTER")
    print("=" * 80)

    total_gold_sets = sum(
        len(
            record.get(
                "gold_sets",
                []
            )
        )
        for record in train_validation
    )

    surviving_gold_set_count = sum(
        len(indices)
        for indices in surviving_gold_indices.values()
    )

    print(
        f"Original gold sets       : "
        f"{total_gold_sets:,}"
    )

    print(
        f"Removed gold sets        : "
        f"{len(removed_gold_sets):,}"
    )

    print(
        f"Surviving gold sets      : "
        f"{surviving_gold_set_count:,}"
    )

    print(
        f"Queries with no survivors: "
        f"{len(removed_queries_validation):,}"
    )

    # ========================================================
    # STEP 2
    # INDEX CORRECTED GOLD SETS
    # ========================================================

    corrected_gold_index = index_gold_sets(
        corrected_gold_sets
    )

    # ========================================================
    # STEP 3
    # INDEX RESOURCES
    # ========================================================

    print()
    print("=" * 80)
    print("STEP 2: INDEXING RESOURCES")
    print("=" * 80)

    resource_index = index_resources(
        resources
    )

    print(
        f"Indexed resources: "
        f"{len(resource_index):,}"
    )

    # ========================================================
    # STEP 4
    # INDEX SIMILARITY-FILTERED TRAIN
    # ========================================================

    similarity_train_by_id = {}

    for record in similarity_filtered_train:

        query_id = record.get(
            "query_id"
        )

        if query_id is None:
            continue

        similarity_train_by_id[
            query_id
        ] = record

    # ========================================================
    # STEP 5
    # BUILD FINAL TRAINING SET
    # ========================================================

    final_training_set = []

    removed_no_gold = []
    missing_corrected_gold = []
    missing_resources = []

    for query_id, train_record in (
        similarity_train_by_id.items()
    ):

        # ----------------------------------------------------
        # Query must have at least one surviving gold set.
        # ----------------------------------------------------

        surviving_indices = (
            surviving_gold_indices.get(
                query_id,
                set()
            )
        )

        if not surviving_indices:

            removed_no_gold.append(
                {
                    "query_id": query_id,
                    "query": train_record.get(
                        "query"
                    ),
                    "reason": (
                        "Query has no surviving "
                        "gold sets after validation."
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Query must exist in corrected gold sets.
        # ----------------------------------------------------

        available_gold_sets = (
            corrected_gold_index.get(
                query_id
            )
        )

        if available_gold_sets is None:

            missing_corrected_gold.append(
                {
                    "query_id": query_id,
                    "query": train_record.get(
                        "query"
                    ),
                    "reason": (
                        "Query exists in similarity-filtered "
                        "train but not in gold_sets_corrected.json."
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Retrieve and enrich surviving gold sets.
        # ----------------------------------------------------

        final_gold_sets = []

        for gold_set_index in sorted(
            surviving_indices
        ):

            if (
                gold_set_index
                not in available_gold_sets
            ):

                missing_corrected_gold.append(
                    {
                        "query_id": query_id,
                        "query": train_record.get(
                            "query"
                        ),
                        "gold_set_index": (
                            gold_set_index
                        ),
                        "reason": (
                            "Surviving gold_set_index "
                            "was not found in "
                            "gold_sets_corrected.json."
                        ),
                    }
                )

                continue

            gold_set = available_gold_sets[
                gold_set_index
            ]

            enriched_gold_set = enrich_gold_set(
                gold_set=gold_set,
                resource_index=resource_index,
                missing_resources=missing_resources,
                query_id=query_id,
                gold_set_index=gold_set_index,
            )

            final_gold_sets.append(
                enriched_gold_set
            )

        # ----------------------------------------------------
        # If nothing could be recovered, remove query.
        # ----------------------------------------------------

        if not final_gold_sets:

            removed_no_gold.append(
                {
                    "query_id": query_id,
                    "query": train_record.get(
                        "query"
                    ),
                    "reason": (
                        "No surviving gold set could "
                        "be matched in "
                        "gold_sets_corrected.json."
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Construct final record.
        # ----------------------------------------------------

        final_record = dict(
            train_record
        )

        final_record[
            "gold_sets"
        ] = final_gold_sets

        final_training_set.append(
            final_record
        )

    # ========================================================
    # STEP 6
    # SAVE OUTPUTS
    # ========================================================

    save_json(
        OUTPUT_FILE,
        final_training_set
    )

    save_json(
        REMOVED_GOLD_SETS_FILE,
        removed_gold_sets
    )

    all_removed_queries = (
        removed_queries_validation
        + removed_no_gold
        + missing_corrected_gold
    )

    save_json(
        REMOVED_QUERIES_FILE,
        all_removed_queries
    )

    save_json(
        MISSING_RESOURCES_FILE,
        missing_resources
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_final_gold_sets = sum(
        len(
            record.get(
                "gold_sets",
                []
            )
        )
        for record in final_training_set
    )

    total_final_resources = 0

    for record in final_training_set:

        for gold_set in record.get(
            "gold_sets",
            []
        ):

            total_final_resources += len(
                gold_set.get(
                    "resources_used",
                    []
                )
            )

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"Original validation queries       : "
        f"{len(train_validation):,}"
    )

    print(
        f"Similarity-filtered train queries: "
        f"{len(similarity_filtered_train):,}"
    )

    print(
        f"Final training queries            : "
        f"{len(final_training_set):,}"
    )

    print(
        f"Final gold sets                   : "
        f"{total_final_gold_sets:,}"
    )

    print(
        f"Final resource references         : "
        f"{total_final_resources:,}"
    )

    print(
        f"Gold sets removed (>=2 zeroes)    : "
        f"{len(removed_gold_sets):,}"
    )

    print(
        f"Queries removed/no gold           : "
        f"{len(removed_no_gold):,}"
    )

    print(
        f"Missing corrected gold entries    : "
        f"{len(missing_corrected_gold):,}"
    )

    print(
        f"Missing resources                 : "
        f"{len(missing_resources):,}"
    )

    print()
    print("OUTPUT:")
    print(
        f"  Final train: {OUTPUT_FILE}"
    )
    print(
        f"  Removed gold sets: "
        f"{REMOVED_GOLD_SETS_FILE}"
    )
    print(
        f"  Removed queries: "
        f"{REMOVED_QUERIES_FILE}"
    )
    print(
        f"  Missing resources: "
        f"{MISSING_RESOURCES_FILE}"
    )

    print()
    print("DONE")


if __name__ == "__main__":
    main()
