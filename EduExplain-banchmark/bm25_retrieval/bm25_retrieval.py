import json
import re
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from pathlib import Path

SCRIPT_DIR=Path(__file__).resolve().parent
DATA_DIR=SCRIPT_DIR.parent/"data"
json_resource=DATA_DIR/"all_resources_unique_ids_clean.json"
json_queries=DATA_DIR/"main_queries_final.json"
json_output=SCRIPT_DIR/"bm25_top200_candidates.json"

# --------------------------------------------------
# Simple tokenizer
# --------------------------------------------------

def tokenize(text):
    return re.findall(r"\w+", text.lower())

# --------------------------------------------------
# Load resources
# --------------------------------------------------

with open(json_resource, "r", encoding="utf-8") as f:
    resources = json.load(f)

corpus = []
resource_ids = []

for r in resources:
    text = (r.get("summary") or "") + " " + (r.get("description") or "")
    corpus.append(tokenize(text))
    resource_ids.append(r["resource_id"])

# --------------------------------------------------
# Build BM25 Index
# --------------------------------------------------

bm25 = BM25Okapi(corpus)

# --------------------------------------------------
# Resource lookup
# --------------------------------------------------

resource_lookup = {
    r["resource_id"]: r
    for r in resources
}

# --------------------------------------------------
# Load queries
# --------------------------------------------------

with open(json_queries, "r", encoding="utf-8") as f:
    queries = json.load(f)

TOP_K = 200

results = []

# --------------------------------------------------
# Retrieval
# --------------------------------------------------

for q in tqdm(queries):

    query_tokens = tokenize(q["query"])

    scores = bm25.get_scores(query_tokens)

    top_idx = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:TOP_K]

    candidates = []

    for idx in top_idx:

        resource = resource_lookup[resource_ids[idx]]

        candidates.append({
            "resource_id": resource["resource_id"],
            "score": float(scores[idx]),
            "resource_type": resource["resource_type"],
            "summary": resource.get("summary"),
            "description": resource.get("description"),
            "concepts": resource.get("concepts", []),
            
        })

    results.append({
        "query_id": q["query_id"],
        "query_type": q["query_type"],
        "variant_style":q["variant_style"],
        "bundle_type": q["bundle_type"],
        "candidates": candidates
    })

# --------------------------------------------------
# Save
# --------------------------------------------------

with open(json_output, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Done!")
