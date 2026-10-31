import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ==========================================================
# Configuration
# ==========================================================
SCRIPT_DIR=Path(__file__).resolve().parent
DATA_DIR=SCRIPT_DIR/"data"
MAIN_QUERY_PATH = "./query_generation/main_queries_new.json"

RESOURCE_PATH = "./data/all_resources_unique_ids_clean1_merged.json"

OUTPUT_PATH = SCRIPT_DIR/"query_candidate_pool_2_new.json"
BM25_K = 10

DENSE_K = 10

MAX_CANDIDATES = 30

# EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_MODEL = "~/models/bge-base-en-v1.5"

CONCEPT_LIMIT=20


# ==========================================================
# Loading
# ==========================================================

print("Loading resources...")

with open(
    RESOURCE_PATH,
    "r",
    encoding="utf-8"
) as f:

    all_resources = json.load(f)

print(f"Loaded {len(all_resources)} resources.")

print("Loading benchmark queries...")

with open(
    MAIN_QUERY_PATH,
    "r",
    encoding="utf-8"
) as f:

    queries = json.load(f)

print(f"Loaded {len(queries)} queries.")


# ==========================================================
# Build Concept Index
# ==========================================================

print("Building concept index...")

concept_index = defaultdict(list)

for resource in all_resources:

    for concept in resource.get("concepts", []):

        concept_index[concept].append(resource)

print(
    f"Indexed {len(concept_index)} concepts."
)


# ==========================================================
# BM25
# ==========================================================

print("Building BM25 index...")


def build_document(resource):

    summary = resource.get("summary") or ""

    description = resource.get("description") or ""

    return (
        summary + " " + description
    ).strip()


bm25_documents = [

    build_document(r)

    for r in all_resources

]

bm25_tokens = [

    doc.lower().split()

    for doc in bm25_documents

]

bm25 = BM25Okapi(
    bm25_tokens
)

print("BM25 ready.")


# ==========================================================
# Dense Retriever
# ==========================================================

print("Loading embedding model...")

embedder = SentenceTransformer(
    EMBEDDING_MODEL,
    device="cuda"
)

print("Generating embeddings...")

resource_embeddings = embedder.encode(

    bm25_documents,

    convert_to_numpy=True,

    show_progress_bar=True,

    normalize_embeddings=True,
    
    batch_size=64

)

resource_embeddings = np.asarray(

    resource_embeddings,

    dtype=np.float32

)

faiss_index = faiss.IndexFlatIP(

    resource_embeddings.shape[1]

)

faiss_index.add(

    resource_embeddings

)

print("Dense retriever ready.")


# ==========================================================
# Retrieval Functions
# ==========================================================

def retrieve_concept_resources(

    concepts,

):

    resources = []

    for concept in concepts:

        resources.extend(

            concept_index.get(

                concept,

                []

            )

        )

    return resources


def retrieve_bm25(

    query,

    k,

):

    scores = bm25.get_scores(

        query.lower().split()

    )

    ranked = np.argsort(scores)[::-1]

    results = []

    for idx in ranked[:k]:

        results.append(

            all_resources[idx]

        )

    return results


def retrieve_dense(

    query,

    k,

):

    embedding = embedder.encode(

        [query],

        normalize_embeddings=True,

        convert_to_numpy=True

    ).astype(np.float32)

    _, indices = faiss_index.search(

        embedding,

        k

    )

    results = []

    for idx in indices[0]:

        results.append(

            all_resources[idx]

        )

    return results


# ==========================================================
# Candidate Serialization
# ==========================================================

def serialize_resource(resource):

    return {

        "resource_id":

            resource["resource_id"],

        "resource_type":

            resource["resource_type"],

        "class_no":

            resource["class_no"],

        "subject":

            resource["subject"],

        "chapter_no":

            resource["chapter_no"],

        "chapter_title":

            resource["chapter_title"],

        "section":

            resource["section"],

        "subsection":

            resource["subsection"],

        "summary":

            resource["summary"],

        "description":

            resource["description"]

    }
# ==========================================================
# Candidate Pool Construction
# ==========================================================

candidate_pool = []

print()

print("=" * 70)
print("Building Candidate Pools")
print("=" * 70)

for idx, query in enumerate(queries, start=1):

    print(

        f"[{idx}/{len(queries)}] "

        f"{query['query_id']}"

    )

    # ------------------------------------------------------
    # Concept Retrieval
    # ------------------------------------------------------

    concept_candidates = retrieve_concept_resources(

        query["concepts"]

    )

    # ------------------------------------------------------
    # BM25
    # ------------------------------------------------------

    bm25_candidates = retrieve_bm25(

        query["query"],

        BM25_K

    )

    # ------------------------------------------------------
    # Dense Retrieval
    # ------------------------------------------------------

    dense_candidates = retrieve_dense(

        query["query"],

        DENSE_K

    )

    # ------------------------------------------------------
    # Merge + Deduplicate
    # ------------------------------------------------------

    merged = {}

    # Concept resources first

    # Concept resources first (up to CONCEPT_LIMIT)

    for resource in concept_candidates[:CONCEPT_LIMIT]:

        merged[
            resource["resource_id"]
        ] = resource

    # BM25

    if len(merged) < MAX_CANDIDATES:

        for resource in bm25_candidates:

            merged.setdefault(

                resource["resource_id"],

                resource

            )

            if len(merged) >= MAX_CANDIDATES:

                break

    # Dense

    if len(merged) < MAX_CANDIDATES:

        for resource in dense_candidates:

            merged.setdefault(

                resource["resource_id"],

                resource

            )

            if len(merged) >= MAX_CANDIDATES:

                break

    # ------------------------------------------------------
    # Serialize
    # ------------------------------------------------------

    serialized_resources = [

        serialize_resource(r)

        for r in merged.values()

    ]

    candidate_pool.append(

        {

            "query_id":

                query["query_id"],

            "query":

                query["query"],

            "query_type":

                query["query_type"],

            "bundle_type":

                query["bundle_type"],

            "grade":

                query["grade"],

            "subject":

                query["subject"],

            "concepts":

                query["concepts"],

            "candidate_resources":

                serialized_resources

        }

    )

# ==========================================================
# Save
# ==========================================================

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

        candidate_pool,

        f,

        indent=2,

        ensure_ascii=False

    )

# ==========================================================
# Statistics
# ==========================================================

candidate_sizes = [

    len(

        q["candidate_resources"]

    )

    for q in candidate_pool

]

print()

print("=" * 70)
print("Candidate Pool Statistics")
print("=" * 70)

print(

    f"Queries                 : {len(candidate_pool)}"

)

print(

    f"Average Candidates      : "

    f"{np.mean(candidate_sizes):.2f}"

)

print(

    f"Minimum Candidates      : "

    f"{np.min(candidate_sizes)}"

)

print(

    f"Maximum Candidates      : "

    f"{np.max(candidate_sizes)}"

)

print()

print(

    f"Saved candidate pool to\n"

    f"{OUTPUT_PATH}"

)

print("=" * 70)
