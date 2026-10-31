
import json
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from pathlib import Path
from multiprocessing import freeze_support


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

json_resources = (
    "./data/all_resources_unique_ids_clean1_merged.json"
)

json_resource_ids = SCRIPT_DIR / "resource_ids.json"
json_output = SCRIPT_DIR / "resource_embeddings.npy"


def main():
    # -------------------------
    # Load Model
    # -------------------------
    model = BGEM3FlagModel(
        "~/models/bge-m3",
        use_fp16=True
    )

    # -------------------------
    # Load Resources
    # -------------------------
    with open(json_resources, "r", encoding="utf-8") as f:
        resources = json.load(f)

    resource_ids = []
    resource_texts = []

    for r in resources:
        summary = r.get("summary") or ""
        description = r.get("description") or ""

        text = summary + " " + description

        resource_ids.append(r["resource_id"])
        resource_texts.append(text)

    print(f"Loaded {len(resource_texts)} resources.")

    # -------------------------
    # Encode Resources
    # -------------------------
    resource_embeddings = model.encode(
        resource_texts,
        batch_size=64,
        max_length=512
    )["dense_vecs"]

    # -------------------------
    # Save Embeddings
    # -------------------------
    np.save(json_output, resource_embeddings)

    with open(json_resource_ids, "w", encoding="utf-8") as f:
        json.dump(resource_ids, f, indent=2)

    print(f"Saved embeddings to: {json_output}")
    print(f"Saved resource IDs to: {json_resource_ids}")
    print(f"Embedding shape: {resource_embeddings.shape}")


if __name__ == "__main__":
    freeze_support()
    main()
