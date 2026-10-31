import json
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR=Path(__file__).resolve().parent
QUERY_INPUT_PATH = SCRIPT_DIR/"query_generation_input.json"
PMI_GRAPH_PATH = SCRIPT_DIR/"pmi_graph.json"

OUTPUT_PATH = SCRIPT_DIR/"query_generation_bundles.json"

TRIPLE_PMI_THRESHOLD = 0.0


# ============================================================
# Load data
# ============================================================

with open(QUERY_INPUT_PATH, "r", encoding="utf-8") as f:
    query_generation_input = json.load(f)

with open(PMI_GRAPH_PATH, "r", encoding="utf-8") as f:
    pmi_graph = json.load(f)


# ============================================================
# O(1) Concept Lookup
# ============================================================

concept_map = {
    concept["concept_id"]: concept
    for concept in query_generation_input
}


# ============================================================
# Helper Functions
# ============================================================

def deduplicate_resources(resources):
    """
    Remove duplicate resources while preserving order.

    Deduplication is based on resource_id.
    """

    seen = set()
    unique_resources = []

    for resource in resources:

        resource_id = resource["resource_id"]

        if resource_id in seen:
            continue

        seen.add(resource_id)
        unique_resources.append(resource)

    return unique_resources


def merge_resources(concept_ids):
    """
    Merge representative resources from multiple concepts.

    Parameters
    ----------
    concept_ids : list[str]

    Returns
    -------
    list
        Deduplicated list of representative resources.
    """

    merged = []

    for concept_id in concept_ids:

        concept = concept_map[concept_id]

        merged.extend(
            concept["representative_resources"]
        )

    return deduplicate_resources(merged)


def merge_class_distribution(concept_ids):
    """
    Merge class distributions.
    """

    merged = {}

    for concept_id in concept_ids:

        dist = concept_map[concept_id].get(
            "class_distribution",
            {}
        )

        for cls, count in dist.items():

            merged[cls] = merged.get(cls, 0) + count

    return merged


def merge_resource_type_distribution(concept_ids):
    """
    Merge resource type distributions.
    """

    merged = {}

    for concept_id in concept_ids:

        dist = concept_map[concept_id].get(
            "resource_type_distribution",
            {}
        )

        for rtype, count in dist.items():

            merged[rtype] = merged.get(rtype, 0) + count

    return merged


def concept_object(concept_id):
    """
    Return minimal concept representation.
    """

    concept = concept_map[concept_id]

    return {

        "concept_id": concept_id,

        "display_name":
            concept["display_name"],

        "frequency":
            concept["frequency"]
    }


# ============================================================
# Bundle Builders
# ============================================================

def build_single_bundle(concept_id):

    return {

        "bundle_id":
            f"single_{concept_id}",

        "bundle_type":
            "single",

        "primary_concept":
            concept_object(concept_id),

        "concepts":
            [
                concept_object(concept_id)
            ],

        "class_distribution":
            merge_class_distribution(
                [concept_id]
            ),

        "resource_type_distribution":
            merge_resource_type_distribution(
                [concept_id]
            ),

        "resources":
            merge_resources(
                [concept_id]
            )
    }


def build_pair_bundle(
    concept_id,
    neighbor_id
):

    concept_ids = [
        concept_id,
        neighbor_id
    ]
    pair_key="_".join(sorted([concept_id, neighbor_id]))

    return {        

        "bundle_id":
            f"pair_{pair_key}",


        "bundle_type":
            "pair",

        "primary_concept":
            concept_object(concept_id),

        "concepts":
            [
                concept_object(cid)
                for cid in concept_ids
            ],

        "class_distribution":
            merge_class_distribution(
                concept_ids
            ),

        "resource_type_distribution":
            merge_resource_type_distribution(
                concept_ids
            ),

        "resources":
            merge_resources(
                concept_ids
            )
    }


def build_triple_bundle(
    concept_id,
    neighbor1,
    neighbor2
):

    concept_ids = [
        concept_id,
        neighbor1,
        neighbor2
    ]

    triple_key = "_".join(
    sorted([concept_id, neighbor1, neighbor2])
    )

    return {

        "bundle_id":
            f"triple_{triple_key}",

        "bundle_type":
            "triple",

        "primary_concept":
            concept_object(concept_id),

        "concepts":
            [
                concept_object(cid)
                for cid in concept_ids
            ],

        "class_distribution":
            merge_class_distribution(
                concept_ids
            ),

        "resource_type_distribution":
            merge_resource_type_distribution(
                concept_ids
            ),

        "resources":
            merge_resources(
                concept_ids
            )
    }
# ============================================================
# Bundle Generation
# ============================================================

bundles = []

seen_pairs = set()
seen_triples = set()

single_count = 0
pair_count = 0
triple_count = 0


for concept in query_generation_input:

    concept_id = concept["concept_id"]

    # --------------------------------------------------------
    # Single Bundle
    # --------------------------------------------------------

    bundles.append(
        build_single_bundle(concept_id)
    )

    single_count += 1

    # --------------------------------------------------------
    # Get Top PMI Neighbors
    # --------------------------------------------------------
    #afg
    neighbors = [
    n
    for n in concept.get("related_concepts", [])
    if n["concept_id"] in concept_map
]

    if len(neighbors) == 0:
        continue

    # --------------------------------------------------------
    # Pair Bundles
    # --------------------------------------------------------

    for neighbor in neighbors[:2]:

        neighbor_id = neighbor["concept_id"]

        pair_key = tuple(
            sorted([
                concept_id,
                neighbor_id
            ])
        )
        # this is for ensuring that we don't create duplicate pair bundles for the same concept pair
        #so if there is[A,B], we don't want to create [B,A] as a separate bundle, we only want one bundle for the pair

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)

        bundles.append(
            build_pair_bundle(
                concept_id,
                neighbor_id
            )
        )

        pair_count += 1

    # --------------------------------------------------------
    # Triple Bundle
    # --------------------------------------------------------

    if len(neighbors) < 2:
        continue

    neighbor1 = neighbors[0]["concept_id"]
    neighbor2 = neighbors[1]["concept_id"]

    triple_key = tuple(
        sorted([
            concept_id,
            neighbor1,
            neighbor2
        ])
    )

    if triple_key in seen_triples:
        continue

    # --------------------------------------------------------
    # Pairwise PMI Validation
    # --------------------------------------------------------

    pairwise_valid = False

    if neighbor1 in pmi_graph:

        neighbor1_neighbors = pmi_graph[
            neighbor1
        ]

        if neighbor2 in neighbor1_neighbors:

            pmi = neighbor1_neighbors[
                neighbor2
            ].get(
                "pmi",
                0.0
            )

            if pmi >= TRIPLE_PMI_THRESHOLD:
                pairwise_valid = True

    if not pairwise_valid:
        continue

    seen_triples.add(
        triple_key
    )

    bundles.append(
        build_triple_bundle(
            concept_id,
            neighbor1,
            neighbor2
        )
    )

    triple_count += 1


# ============================================================
# Sort Bundles
# ============================================================

bundle_order = {
    "single": 0,
    "pair": 1,
    "triple": 2
}

bundles.sort(
    key=lambda x: (
        bundle_order[x["bundle_type"]],
        x["primary_concept"]["display_name"]
    )
)


# ============================================================
# Save
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        bundles,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# Statistics
# ============================================================

print()

print("=" * 60)

print("Bundle Generation Complete")

print("=" * 60)

print(f"Single Bundles : {single_count}")

print(f"Pair Bundles   : {pair_count}")

print(f"Triple Bundles : {triple_count}")

print(f"Total Bundles  : {len(bundles)}")

print()

print(f"Saved to: {OUTPUT_PATH}")

print("=" * 60)


# ============================================================
# Sanity Checks
# ============================================================

print()

print("Sample Bundles")

print()

for bundle in bundles[:5]:

    print("--------------------------------")

    print(
        bundle["bundle_type"].upper()
    )

    print()

    print(
        "Concepts:"
    )

    for concept in bundle["concepts"]:

        print(
            f"  - {concept['display_name']}"
        )

    print()

    print(
        f"Resources: {len(bundle['resources'])}"
    )

print("--------------------------------")

