import json
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
SCRIPT_DIR=Path(__file__).resolve().parent
DATA_DIR="./data"
ALL_RESOURCES_PATH = "./data/all_resources_unique_ids_clean1_merged.json"
EXTENDED_CONCEPTS_PATH ="./data/extended_concepts_merged.json"
OUTPUT_PATH = SCRIPT_DIR/"concept_index.json"

PEDAGOGICAL_TERMS = {
    "example",
    "examples",
    "exercise",
    "exercises",
    "activity",
    "activities",
    "summary",
    "summaries",
    "introduction",
    "conclusion",
    "figure",
    "figures",
    "table",
    "tables",
    "diagram",
    "diagrams",
    "observation",
    "observations",
    "experiment",
    "experiments",
    "question",
    "questions",
    "answer",
    "answers",
    "product",
    "products",
    "property",
    "properties",
    "development",
    "opposite",
    "importance"
}


# --------------------------------------------------
# Load files
# --------------------------------------------------

with open(ALL_RESOURCES_PATH, "r", encoding="utf-8") as f:
    resources = json.load(f)

with open(EXTENDED_CONCEPTS_PATH, "r", encoding="utf-8") as f:
    extended_concepts = json.load(f)

print(f"Loaded {len(resources)} resources")
print(f"Loaded {len(extended_concepts)} concepts")


# --------------------------------------------------
# Initialize index from canonical vocabulary
# --------------------------------------------------
def is_valid_concept(concept_id):
    return concept_id not in PEDAGOGICAL_TERMS

concept_index = {}

for concept_id, concept_data in extended_concepts.items():
    if not is_valid_concept(concept_id):
        continue
    concept_index[concept_id] = {
        "display_name": concept_data.get(
            "display_name",
            concept_id
        ),

        "frequency": 0,

        "class_distribution": {},

        "resources": []
    }


# --------------------------------------------------
# Populate index
# --------------------------------------------------

for resource in resources:

    concepts = resource.get("concepts", [])

    if not concepts:
        continue

    for concept_id in concepts:

        if concept_id not in concept_index:
            continue

        entry = concept_index[concept_id]

        entry["frequency"] += 1

        class_no = str(
            resource.get("class_no", "unknown")
        )

        class_dist = entry["class_distribution"]

        class_dist[class_no] = (
            class_dist.get(class_no, 0) + 1
        )

        entry["resources"].append({

            "resource_id":
                resource.get("resource_id"),

            "resource_type":
                resource.get("resource_type"),

            "class_no":
                resource.get("class_no"),

            "subject":
                resource.get("subject"),

            "chapter_no":
                resource.get("chapter_no"),

            "chapter_title":
                resource.get("chapter_title"),

            "section":
                resource.get("section"),

            "subsection":
                resource.get("subsection"),

            "summary":
                resource.get("summary"),

            "description":
                resource.get("description")
        })


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
        concept_index,
        f,
        indent=2,
        ensure_ascii=False
    )

print(
    f"Saved concept index to {OUTPUT_PATH}"
)
