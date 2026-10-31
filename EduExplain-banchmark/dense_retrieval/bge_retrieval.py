import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

SCRIPT_DIR=Path(__file__).resolve().parent
DATA_DIR=SCRIPT_DIR.parent/"data"
json_resource_embeddings=SCRIPT_DIR/"resource_embeddings.npy"
json_query_embeddings=SCRIPT_DIR/"query_embeddings.npy"
json_resource_ids=SCRIPT_DIR/"resource_ids.json"
json_query_ids=SCRIPT_DIR/"query_ids.json"
json_resources="./data/all_resources_unique_ids_clean1_merged.json"
json_queries="./query_generation/main_queries.json"
json_output=SCRIPT_DIR/"bge_top200_candidates.json"


# --------------------------------------------------
# INPUTS
# --------------------------------------------------

resource_embeddings = np.load(json_resource_embeddings)    # shape: (num_resources, dim)
query_embeddings    = np.load(json_query_embeddings)       # shape: (num_queries, dim)

with open(json_resource_ids, "r", encoding="utf-8") as f:
    resource_ids = json.load(f)                             # list of resource_id strings

with open(json_query_ids, "r", encoding="utf-8") as f:
    query_ids = json.load(f)                                # list of query_id strings

with open(json_resources, "r", encoding="utf-8") as f:
    all_resources = json.load(f)

with open(json_queries, "r", encoding="utf-8") as f:
    all_queries = json.load(f)

# --------------------------------------------------
# BUILD LOOKUP DICTS
# --------------------------------------------------

resource_lookup = {r["resource_id"]: r for r in all_resources}
query_lookup    = {q["query_id"]: q   for q in all_queries}

print(f"Resources : {len(resource_lookup)}")
print(f"Queries   : {len(query_lookup)}")

# --------------------------------------------------
# GENERATE TOP-200 BGE CANDIDATES
# (cosine similarity using precomputed embeddings)
# --------------------------------------------------

TOP_K = 200
results = []

for i in tqdm(range(len(query_ids))):

    qid = query_ids[i]

    # Cosine similarity between this query and all resources
    # BGE embeddings are L2-normalized so dot product = cosine similarity
    scores = query_embeddings[i] @ resource_embeddings.T    # shape: (num_resources,)

    # Get top-200 indices, sorted by score descending
    top_idx = np.argpartition(scores, -TOP_K)[-TOP_K:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    # Build candidate list
    candidates = []
    for idx in top_idx:
        rid      = resource_ids[idx]
        resource = resource_lookup[rid]
        candidates.append({
            "resource_id":    resource["resource_id"],
            "embedding_index": int(idx),               # needed by MMR selectors
            "score":          float(scores[idx]),
            "resource_type":  resource["resource_type"],
            "summary":        resource.get("summary"),
            "description":    resource.get("description"),
            "concepts":       resource.get("concepts", []),
        })

    # Attach query metadata so selectors and evaluator can slice by these fields
    query_meta = query_lookup.get(qid, {})
    results.append({
        "query_id":     qid,
        "query_type":   query_meta.get("query_type"),   # Explanation / Why/How / etc.
        "variant_style": query_meta.get("variant_style"), # search / chat / learning_context
        "bundle_type":  query_meta.get("bundle_type"),  # single / pair / triple
        "candidates":   candidates,
    })

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

with open(json_output, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved {len(results)} queries → bge_top200_candidates.json")
