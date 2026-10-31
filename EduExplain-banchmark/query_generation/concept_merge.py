#!/usr/bin/env python3

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("./data")

EXTENDED_CONCEPTS = BASE_DIR / "extended_concepts.json"
RESOURCES = BASE_DIR / "all_resources_unique_ids_clean1.json"

OUTPUT_CONCEPTS = BASE_DIR / "extended_concepts_merged.json"
OUTPUT_RESOURCES = BASE_DIR / "all_resources_unique_ids_clean1_merged.json"
OUTPUT_REMAP = BASE_DIR / "concept_merge_remap.json"


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
    Normalize a concept name/ID for duplicate detection.

    Handles:
      - leading section labels: (a), (b), (c), (d), ...
      - leading articles: a, an, the
      - possessives: Newton's -> Newtons
      - punctuation
      - whitespace
      - underscores/hyphens
    """

    text = str(text).strip().lower()

    # Normalize apostrophe variants.
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

    # Also handle numeric labels if they occur:
    #
    # (1) Topic
    # (2) Topic

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
    # Convert to concept ID.
    # --------------------------------------------------------

    return "_".join(words)


# ============================================================
# DISPLAY NAME -> ID
# ============================================================

def concept_id_from_display_name(display_name):
    """
    Convert a display name into the canonical concept ID.
    """

    return canonicalize(display_name)


# ============================================================
# CHOOSE CANONICAL CONCEPT
# ============================================================

def choose_canonical(concept_ids, concepts):
    """
    Select the concept ID that should survive when multiple
    IDs refer to the same normalized concept.

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
# LOAD
# ============================================================

def load_json(path):
    print(f"Loading: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# BUILD DUPLICATE GROUPS
# ============================================================

def find_duplicate_groups(concepts):
    """
    Group concept IDs by their canonicalized representation.
    """

    groups = defaultdict(list)

    for concept_id, info in concepts.items():

        display_name = info.get("display_name", concept_id)

        # Use both the ID and display name.

        key_from_id = canonicalize(concept_id)
        key_from_name = canonicalize(display_name)

        # Normally these should agree. If they don't, the
        # display name is more semantically meaningful.
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

    for every duplicate group.
    """

    remap = {}

    for key, ids in duplicate_groups.items():

        canonical_id = choose_canonical(ids, concepts)

        for concept_id in ids:

            if concept_id != canonical_id:
                remap[concept_id] = canonical_id

    return remap


# ============================================================
# MERGE CONCEPT VOCABULARY
# ============================================================

def merge_concepts(concepts, remap):
    """
    Create merged concept dictionary.

    The canonical concept survives.

    Frequencies from duplicate concepts are added together.
    """

    merged = {}

    # First create all surviving concepts.
    for concept_id, info in concepts.items():

        if concept_id in remap:
            continue

        merged[concept_id] = dict(info)

    # Merge duplicate metadata into canonical concepts.
    for duplicate_id, canonical_id in remap.items():

        duplicate_info = concepts[duplicate_id]
        canonical_info = merged[canonical_id]

        # Add frequency.
        duplicate_frequency = duplicate_info.get("frequency", 0)

        try:
            duplicate_frequency = int(duplicate_frequency)
        except (TypeError, ValueError):
            duplicate_frequency = 0

        canonical_frequency = canonical_info.get("frequency", 0)

        try:
            canonical_frequency = int(canonical_frequency)
        except (TypeError, ValueError):
            canonical_frequency = 0

        canonical_info["frequency"] = (
            canonical_frequency + duplicate_frequency
        )

    return merged


# ============================================================
# REWRITE RESOURCES
# ============================================================

def rewrite_resources(resources, remap):
    """
    Replace duplicate concept IDs inside each resource.
    """

    changed_resources = 0
    changed_concepts = 0

    for resource in resources:

        concepts = resource.get("concepts")

        if not isinstance(concepts, list):
            continue

        new_concepts = []
        resource_changed = False

        for concept_id in concepts:

            canonical_id = remap.get(concept_id, concept_id)

            if canonical_id != concept_id:
                changed_concepts += 1
                resource_changed = True

            new_concepts.append(canonical_id)

        # Remove duplicates created by merging.
        deduplicated = list(dict.fromkeys(new_concepts))

        if len(deduplicated) != len(new_concepts):
            resource_changed = True

        resource["concepts"] = deduplicated

        if resource_changed:
            changed_resources += 1

    return changed_resources, changed_concepts


# ============================================================
# VALIDATION
# ============================================================

def validate_resources(resources, concepts):
    """
    Check how many resource concept IDs are not represented
    in the merged vocabulary.
    """

    used = defaultdict(int)

    for resource in resources:

        for concept_id in resource.get("concepts", []):
            used[concept_id] += 1

    missing = {
        concept_id: count
        for concept_id, count in used.items()
        if concept_id not in concepts
    }

    return used, missing


# ============================================================
# WRITE JSON
# ============================================================

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Wrote: {path}")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Merge duplicate concept IDs in EduExplain data."
    )

    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the original JSON files.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("EDUEXPLAIN CONCEPT DUPLICATE MERGER")
    print("=" * 70)

    print()
    print(f"Extended concepts : {EXTENDED_CONCEPTS}")
    print(f"Resources         : {RESOURCES}")
    print(f"In place          : {args.in_place}")
    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    concepts = load_json(EXTENDED_CONCEPTS)
    resources = load_json(RESOURCES)

    if not isinstance(concepts, dict):
        raise TypeError(
            "extended_concepts.json must contain a top-level "
            "dictionary keyed by concept ID."
        )

    if not isinstance(resources, list):
        raise TypeError(
            "all_resources_unique_ids_clean1.json must contain "
            "a top-level list of resources."
        )

    print()
    print(f"Loaded concepts  : {len(concepts):,}")
    print(f"Loaded resources : {len(resources):,}")

    # --------------------------------------------------------
    # Duplicate detection
    # --------------------------------------------------------

    duplicate_groups = find_duplicate_groups(concepts)

    print()
    print(f"Duplicate groups found : {len(duplicate_groups):,}")

    # Show groups.
    if duplicate_groups:

        print()
        print("Duplicate groups:")
        print("-" * 70)

        for key, ids in sorted(duplicate_groups.items()):

            canonical_id = choose_canonical(ids, concepts)

            print()
            print(f"Canonical key : {key}")
            print(f"Canonical ID  : {canonical_id}")

            for concept_id in ids:

                info = concepts[concept_id]

                print(
                    f"  - {concept_id}"
                    f" | display_name={info.get('display_name')!r}"
                    f" | frequency={info.get('frequency')}"
                    f" | source={info.get('source')}"
                )

    # --------------------------------------------------------
    # Build remap
    # --------------------------------------------------------

    remap = build_remap(
        concepts,
        duplicate_groups,
    )

    print()
    print(f"Concept IDs to remap : {len(remap):,}")

    # --------------------------------------------------------
    # Merge concepts
    # --------------------------------------------------------

    merged_concepts = merge_concepts(
        concepts,
        remap,
    )

    # --------------------------------------------------------
    # Rewrite resources
    # --------------------------------------------------------

    changed_resources, changed_concepts = rewrite_resources(
        resources,
        remap,
    )

    print()
    print(f"Resources changed        : {changed_resources:,}")
    print(f"Concept references changed: {changed_concepts:,}")
    print(f"Concept count before     : {len(concepts):,}")
    print(f"Concept count after      : {len(merged_concepts):,}")

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    used, missing = validate_resources(
        resources,
        merged_concepts,
    )

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(f"Unique concept IDs used by resources : {len(used):,}")
    print(f"Missing from merged vocabulary       : {len(missing):,}")

    if missing:

        print()
        print("WARNING: Some resource concept IDs are not present")
        print("in extended_concepts.json.")
        print()
        print("Top missing concept IDs:")

        for concept_id, count in sorted(
            missing.items(),
            key=lambda x: -x[1],
        )[:30]:

            print(
                f"  {concept_id}: "
                f"{count:,} resources"
            )

    else:
        print()
        print("SUCCESS: Every resource concept ID exists")
        print("in the merged vocabulary.")

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    if args.in_place:

        output_concepts = EXTENDED_CONCEPTS
        output_resources = RESOURCES

    else:

        output_concepts = OUTPUT_CONCEPTS
        output_resources = OUTPUT_RESOURCES

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    write_json(
        output_concepts,
        merged_concepts,
    )

    write_json(
        output_resources,
        resources,
    )

    write_json(
        OUTPUT_REMAP,
        remap,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print(f"Concept vocabulary : {output_concepts}")
    print(f"Resources          : {output_resources}")
    print(f"Remap              : {OUTPUT_REMAP}")

    print()
    print("Example remappings:")

    for duplicate_id, canonical_id in list(remap.items())[:20]:
        print(
            f"  {duplicate_id}"
            f" -> "
            f"{canonical_id}"
        )


if __name__ == "__main__":
    main()
