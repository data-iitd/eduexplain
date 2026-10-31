import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
SCRIPT_DIR=Path(__file__).resolve().parent
DATA_DIR=SCRIPT_DIR.parent/"data"
ALL_RESOURCES_PATH = "./data/all_resources_unique_ids_clean1_merged.json"
OUTPUT_PATH = SCRIPT_DIR/"pmi_graph.json"

MIN_COOCCURRENCE = 2

# --------------------------------------------------
# Load resources
# --------------------------------------------------

with open(ALL_RESOURCES_PATH, "r", encoding="utf-8") as f:
    resources = json.load(f)

print(f"Loaded {len(resources)} resources")

# --------------------------------------------------
# Frequency counts
# --------------------------------------------------

concept_frequency = defaultdict(int)

for resource in resources:

    concepts = resource.get("concepts", [])

    # remove duplicates within resource
    concepts = list(set(concepts))

    for concept in concepts:
        concept_frequency[concept] += 1

total_resources = len(resources)

print(f"Unique concepts: {len(concept_frequency)}")

# --------------------------------------------------
# Co-occurrence counts
# --------------------------------------------------

pair_counts = defaultdict(int)

for resource in resources:

    concepts = resource.get("concepts", [])

    concepts = sorted(set(concepts))

    if len(concepts) < 2:
        continue

    for c1, c2 in combinations(concepts, 2):

        pair_counts[(c1, c2)] += 1

print(f"Unique pairs: {len(pair_counts)}")

# --------------------------------------------------
# Compute PMI
# --------------------------------------------------

pmi_graph = defaultdict(dict)

for (c1, c2), count in pair_counts.items():

    if count < MIN_COOCCURRENCE:
        continue

    freq1 = concept_frequency[c1]
    freq2 = concept_frequency[c2]

    p_a = freq1 / total_resources
    p_b = freq2 / total_resources
    p_ab = count / total_resources

    # PMI
    pmi = math.log(
        p_ab / (p_a * p_b)
    )

    pmi = round(pmi, 4)

    pmi_graph[c1][c2] = {
        "count": count,
        "pmi": pmi
    }

    pmi_graph[c2][c1] = {
        "count": count,
        "pmi": pmi
    }

# --------------------------------------------------
# Sort neighbors by PMI
# --------------------------------------------------

final_graph = {}

for concept, neighbors in pmi_graph.items():

    sorted_neighbors = sorted(
        neighbors.items(),
        key=lambda x: (
            x[1]["pmi"],
            x[1]["count"]
        ),
        reverse=True
    )

    final_graph[concept] = {
        neighbor: values
        for neighbor, values in sorted_neighbors
    }

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
        final_graph,
        f,
        indent=2,
        ensure_ascii=False
    )

print(
    f"Saved PMI graph to {OUTPUT_PATH}"
)

print(
    f"Concepts in graph: {len(final_graph)}"
)
