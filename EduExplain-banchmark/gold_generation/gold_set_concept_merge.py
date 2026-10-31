#!/usr/bin/env python3

import json
import re
from collections import defaultdict
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    "./data"
)

EXTENDED_CONCEPTS = (
    BASE_DIR / "extended_concepts.json"
)

GOLD_SETS = Path(
    "./data/gold_sets_final_validated_new.json"
)

OUTPUT_GOLD_SETS = Path(
    "./data/gold_sets_final_validated_new_merged.json"
)

OUTPUT_REMAP = Path(
    "./gold_generation/gold_set_concept_merge_remap_new.json"
)


# ============================================================
# CANONICALIZATION
# ============================================================

LEADING_ARTICLES = {
    "a",
    "an",
    "the",
}


def canonicalize(text):
    """
    EXACT SAME canonicalization logic as the resource
    concept-merging script.

    Handles:
      - leading section labels: (a), (b), (c), ...
      - leading numeric labels: (1), (2), ...
      - leading articles: a, an, the
      - possessives: Newton's -> Newtons
      - punctuation
      - whitespace
      - underscores / hyphens
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
    # Remove leading numeric labels.
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

    text = re.sub(
        r"'s\b",
        "s",
        text,
    )

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
    #
    # "A Male Reproductive System"
    # -> "Male Reproductive System"
    # --------------------------------------------------------

    while words and words[0] in {
        "a",
        "an",
        "the",
    }:
        words.pop(0)

    # --------------------------------------------------------
    # Convert to concept-ID style.
    # --------------------------------------------------------

    return "_".join(words)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):
    print(f"Loading: {path}")

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# WRITE JSON
# ============================================================

def write_json(path, data):

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
# CHOOSE CANONICAL CONCEPT
# ============================================================

def choose_canonical(
    concept_ids,
    concepts,
):
    """
    EXACT SAME canonical-selection logic as the
    resource concept-merging script.

    Preference order:

    1. Structural source
    2. Highest frequency
    3. Shorter ID
    4. Lexicographically smaller ID
    """

    def score(cid):

        info = concepts[cid]

        source = str(
            info.get(
                "source",
                "",
            )
        ).lower()

        structural_priority = (
            0
            if source == "structural"
            else 1
        )

        frequency = info.get(
            "frequency",
            0,
        )

        try:
            frequency = int(
                frequency
            )
        except (
            TypeError,
            ValueError,
        ):
            frequency = 0

        return (
            structural_priority,
            -frequency,
            len(cid),
            cid,
        )

    return sorted(
        concept_ids,
        key=score,
    )[0]


# ============================================================
# FIND DUPLICATE GROUPS
# ============================================================

def find_duplicate_groups(concepts):
    """
    EXACT SAME duplicate detection used by the
    resource concept-merging script.
    """

    groups = defaultdict(list)

    for concept_id, info in concepts.items():

        display_name = info.get(
            "display_name",
            concept_id,
        )

        # Same as original script:
        # the display name is used for the
        # canonical comparison key.

        key_from_id = canonicalize(
            concept_id
        )

        key_from_name = canonicalize(
            display_name
        )

        # Keep this variable because this is
        # exactly how your original script works.
        #
        # The display name is considered more
        # semantically meaningful.
        key = key_from_name

        groups[key].append(
            concept_id
        )

    duplicates = {
        key: ids
        for key, ids in groups.items()
        if len(ids) > 1
    }

    return duplicates


# ============================================================
# BUILD REMAP
# ============================================================

def build_remap(
    concepts,
    duplicate_groups,
):
    """
    Create:

        duplicate_id -> canonical_id

    using EXACTLY the same mapping logic
    as your resource merger.
    """

    remap = {}

    for key, ids in duplicate_groups.items():

        canonical_id = choose_canonical(
            ids,
            concepts,
        )

        for concept_id in ids:

            if concept_id != canonical_id:

                remap[
                    concept_id
                ] = canonical_id

    return remap


# ============================================================
# REWRITE GOLD-SET RESOURCES
# ============================================================

def rewrite_gold_sets(
    gold_sets_data,
    remap,
):
    """
    Rewrite every resource concept list inside:

        gold_sets
            -> resources_used
                -> concepts

    using the same remap used for the
    main resource dataset.

    Also removes duplicate concept IDs created
    by merging.
    """

    changed_gold_sets = 0
    changed_resources = 0
    changed_concepts = 0
    deduplicated_concepts = 0

    for query_record in gold_sets_data:

        query_changed = False

        gold_sets = query_record.get(
            "gold_sets",
            [],
        )

        if not isinstance(
            gold_sets,
            list,
        ):
            continue

        for gold_set in gold_sets:

            resources_used = (
                gold_set.get(
                    "resources_used",
                    [],
                )
            )

            if not isinstance(
                resources_used,
                list,
            ):
                continue

            gold_set_changed = False

            for resource in resources_used:

                if not isinstance(
                    resource,
                    dict,
                ):
                    continue

                concepts = resource.get(
                    "concepts"
                )

                if not isinstance(
                    concepts,
                    list,
                ):
                    continue

                new_concepts = []

                resource_changed = False

                for concept_id in concepts:

                    canonical_id = remap.get(
                        concept_id,
                        concept_id,
                    )

                    if (
                        canonical_id
                        != concept_id
                    ):

                        changed_concepts += 1
                        resource_changed = True

                    new_concepts.append(
                        canonical_id
                    )

                # ------------------------------------------------
                # Remove duplicates introduced by merging.
                # ------------------------------------------------

                deduplicated = list(
                    dict.fromkeys(
                        new_concepts
                    )
                )

                if len(deduplicated) != len(
                    new_concepts
                ):

                    deduplicated_concepts += (
                        len(new_concepts)
                        - len(deduplicated)
                    )

                    resource_changed = True

                if resource_changed:

                    resource[
                        "concepts"
                    ] = deduplicated

                    changed_resources += 1
                    gold_set_changed = True
                    query_changed = True

            if gold_set_changed:
                changed_gold_sets += 1

        if query_changed:
            pass

    return (
        changed_gold_sets,
        changed_resources,
        changed_concepts,
        deduplicated_concepts,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_gold_sets(
    gold_sets_data,
    concepts,
):
    """
    Check all concept IDs used inside gold sets
    against the concept vocabulary.

    Returns:

        used
        missing
    """

    used = defaultdict(int)

    for query_record in gold_sets_data:

        gold_sets = query_record.get(
            "gold_sets",
            [],
        )

        if not isinstance(
            gold_sets,
            list,
        ):
            continue

        for gold_set in gold_sets:

            resources_used = (
                gold_set.get(
                    "resources_used",
                    [],
                )
            )

            if not isinstance(
                resources_used,
                list,
            ):
                continue

            for resource in resources_used:

                concepts_list = (
                    resource.get(
                        "concepts",
                        [],
                    )
                )

                if not isinstance(
                    concepts_list,
                    list,
                ):
                    continue

                for concept_id in concepts_list:

                    used[
                        concept_id
                    ] += 1

    missing = {
        concept_id: count
        for concept_id, count
        in used.items()
        if concept_id not in concepts
    }

    return used, missing


# ============================================================
# PRINT DUPLICATE GROUPS
# ============================================================

def print_duplicate_groups(
    duplicate_groups,
    concepts,
):
    """
    Print the same duplicate-group information
    used by your original merger.
    """

    if not duplicate_groups:
        print()
        print(
            "No duplicate concept groups found."
        )
        return

    print()
    print("Duplicate groups:")
    print("-" * 70)

    for key, ids in sorted(
        duplicate_groups.items()
    ):

        canonical_id = choose_canonical(
            ids,
            concepts,
        )

        print()
        print(
            f"Canonical key : {key}"
        )

        print(
            f"Canonical ID  : {canonical_id}"
        )

        for concept_id in ids:

            info = concepts[
                concept_id
            ]

            print(
                f"  - {concept_id}"
                f" | display_name="
                f"{info.get('display_name')!r}"
                f" | frequency="
                f"{info.get('frequency')}"
                f" | source="
                f"{info.get('source')}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CONCEPT MERGING FOR GOLD SETS")
    print("=" * 80)

    print()
    print(
        "This script uses the SAME concept remapping"
    )
    print(
        "logic as the main EduExplain resource merger."
    )

    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    concepts = load_json(
        EXTENDED_CONCEPTS
    )

    gold_sets_data = load_json(
        GOLD_SETS
    )

    if not isinstance(
        concepts,
        dict,
    ):
        raise TypeError(
            "extended_concepts.json must contain "
            "a top-level dictionary."
        )

    if not isinstance(
        gold_sets_data,
        list,
    ):
        raise TypeError(
            "gold_sets_corrected.json must contain "
            "a top-level list."
        )

    print()
    print(
        f"Loaded concepts : "
        f"{len(concepts):,}"
    )

    print(
        f"Loaded queries  : "
        f"{len(gold_sets_data):,}"
    )

    # --------------------------------------------------------
    # Find the SAME duplicate groups
    # --------------------------------------------------------

    duplicate_groups = find_duplicate_groups(
        concepts
    )

    print()
    print(
        f"Duplicate groups found : "
        f"{len(duplicate_groups):,}"
    )

    print_duplicate_groups(
        duplicate_groups,
        concepts,
    )

    # --------------------------------------------------------
    # Build EXACT SAME remap
    # --------------------------------------------------------

    remap = build_remap(
        concepts,
        duplicate_groups,
    )

    print()
    print(
        f"Concept IDs to remap : "
        f"{len(remap):,}"
    )

    # --------------------------------------------------------
    # Show remappings
    # --------------------------------------------------------

    if remap:

        print()
        print("Concept remappings:")
        print("-" * 70)

        for duplicate_id, canonical_id in sorted(
            remap.items()
        ):

            print(
                f"  {duplicate_id}"
                f" -> "
                f"{canonical_id}"
            )

    # --------------------------------------------------------
    # Rewrite gold sets
    # --------------------------------------------------------

    (
        changed_gold_sets,
        changed_resources,
        changed_concepts,
        deduplicated_concepts,
    ) = rewrite_gold_sets(
        gold_sets_data,
        remap,
    )

    print()
    print("=" * 80)
    print("GOLD SET REWRITE")
    print("=" * 80)

    print(
        f"Gold sets changed           : "
        f"{changed_gold_sets:,}"
    )

    print(
        f"Resources changed           : "
        f"{changed_resources:,}"
    )

    print(
        f"Concept references remapped : "
        f"{changed_concepts:,}"
    )

    print(
        f"Duplicate concepts removed  : "
        f"{deduplicated_concepts:,}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    used, missing = validate_gold_sets(
        gold_sets_data,
        concepts,
    )

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    print(
        f"Unique concept IDs used by gold sets : "
        f"{len(used):,}"
    )

    print(
        f"Missing from vocabulary              : "
        f"{len(missing):,}"
    )

    if missing:

        print()
        print(
            "WARNING: These concept IDs are used "
            "by gold sets but are not present "
            "in extended_concepts.json."
        )

        print()
        print("Top missing concept IDs:")

        for concept_id, count in sorted(
            missing.items(),
            key=lambda x: -x[1],
        )[:50]:

            print(
                f"  {concept_id}: "
                f"{count:,} occurrences"
            )

    else:

        print()
        print(
            "SUCCESS: Every concept ID used by "
            "the gold sets exists in the vocabulary."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    write_json(
        OUTPUT_GOLD_SETS,
        gold_sets_data,
    )

    write_json(
        OUTPUT_REMAP,
        remap,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print()
    print(
        f"Merged gold sets:"
        f"\n  {OUTPUT_GOLD_SETS}"
    )

    print()
    print(
        f"Concept remap:"
        f"\n  {OUTPUT_REMAP}"
    )


if __name__ == "__main__":
    main()
