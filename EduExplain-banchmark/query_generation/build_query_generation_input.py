import json
from collections import defaultdict, deque
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
SCRIPT_DIR=Path(__file__).resolve().parent
DATA_DIR=SCRIPT_DIR/"data"
CONCEPT_INDEX_PATH = SCRIPT_DIR/"concept_index.json"
PMI_GRAPH_PATH = SCRIPT_DIR/"pmi_graph.json"
EXTENDED_CONCEPTS_PATH = "./data/extended_concepts_merged.json"

OUTPUT_PATH = SCRIPT_DIR/"query_generation_input.json"

MAX_REPRESENTATIVE_RESOURCES = 15
MAX_RELATED_CONCEPTS = 10

# --------------------------------------------------
# Load files
# --------------------------------------------------

with open(CONCEPT_INDEX_PATH, "r", encoding="utf-8") as f:
    concept_index = json.load(f)

with open(PMI_GRAPH_PATH, "r", encoding="utf-8") as f:
    pmi_graph = json.load(f)

with open(EXTENDED_CONCEPTS_PATH, "r", encoding="utf-8") as f:
    extended_concepts = json.load(f)

print(f"Loaded {len(concept_index)} concepts")
print(f"Loaded PMI graph for {len(pmi_graph)} concepts")

# --------------------------------------------------
# Representative resource selection
# --------------------------------------------------

def select_representative_resources(
    resources,
    max_resources=15
):
    """
    Select diverse representative resources using
    round-robin across resource types.

    Example:
        explanation
        activity
        example
        exercise
        chapter_summary
        ...

    repeated until max_resources reached.
    """

    grouped = defaultdict(list)

    for resource in resources:
        resource_type = (
            resource.get("resource_type")
            or "unknown"
        )

        grouped[resource_type].append(resource)

    queues = {
        resource_type: deque(items)
        for resource_type, items in grouped.items()
    }

    selected = []

    while (
        len(selected) < max_resources
        and any(len(q) > 0 for q in queues.values())
    ):

        for resource_type in list(queues.keys()):

            if len(selected) >= max_resources:
                break

            if queues[resource_type]:
                selected.append(
                    queues[resource_type].popleft()
                )

    return selected


# --------------------------------------------------
# Build query generation input
# --------------------------------------------------

query_generation_input = []

for concept_id, concept_data in concept_index.items():

    resources = concept_data.get(
        "resources",
        []
    )

    # ----------------------------------------------
    # Resource type distribution
    # ----------------------------------------------

    resource_type_distribution = defaultdict(int)

    for resource in resources:

        resource_type = (
            resource.get("resource_type")
            or "unknown"
        )

        resource_type_distribution[
            resource_type
        ] += 1

    # ----------------------------------------------
    # Representative resources
    # ----------------------------------------------

    representative_resources = (
        select_representative_resources(
            resources,
            MAX_REPRESENTATIVE_RESOURCES
        )
    )

    # ----------------------------------------------
    # Related concepts
    # ----------------------------------------------

    related_concepts = []

    neighbors = pmi_graph.get(
        concept_id,
        {}
    )

    for neighbor_id, neighbor_data in list(
        neighbors.items()
    )[:MAX_RELATED_CONCEPTS]:

        related_concepts.append({

            "concept_id":
                neighbor_id,

            "display_name":
                extended_concepts.get(
                    neighbor_id,
                    {}
                ).get(
                    "display_name",
                    neighbor_id
                ),

            "count":
                neighbor_data.get(
                    "count",
                    0
                ),

            "pmi":
                neighbor_data.get(
                    "pmi",
                    0.0
                )
        })

    # ----------------------------------------------
    # Build final entry
    # ----------------------------------------------

    entry = {

        "concept_id":
            concept_id,

        "display_name":
            concept_data.get(
                "display_name",
                concept_id
            ),

        "frequency":
            concept_data.get(
                "frequency",
                0
            ),

        "class_distribution":
            concept_data.get(
                "class_distribution",
                {}
            ),

        "resource_type_distribution":
            dict(
                resource_type_distribution
            ),

        "representative_resources":
            representative_resources,

        "related_concepts":
            related_concepts
    }

    query_generation_input.append(entry)

# --------------------------------------------------
# Sort by frequency
# --------------------------------------------------

query_generation_input.sort(
    key=lambda x: x["frequency"],
    reverse=True
)

# --------------------------------------------------
# Save
# --------------------------------------------------

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
        query_generation_input,
        f,
        indent=2,
        ensure_ascii=False
    )

print(
    f"Saved {len(query_generation_input)} concepts "
    f"to {OUTPUT_PATH}"
)
