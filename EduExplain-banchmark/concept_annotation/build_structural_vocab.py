import json
import re
from pathlib import Path
SCRIPT_DIR=Path(__file__).resolve().parent
DATA_DIR=SCRIPT_DIR.parent/"data"
HELPER_DIR=SCRIPT_DIR.parent/"build_resources"

RESOURCE_DIR = HELPER_DIR/"data_resources_summaries"          """ this should refer to the path of resources after parsing +summary phase"""
OUTPUT_DIR = SCRIPT_DIR/"concepts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def canonicalize(title: str) -> str:
    """
    Convert title to stable concept id.

    Example:
        "Centre of Mass" -> "centre_of_mass"
        "Newton's First Law" -> "newton_s_first_law"
    """

    title = title.strip().lower()

    title = re.sub(r"[^a-z0-9]+", "_", title)

    title = re.sub(r"_+", "_", title)

    title = title.strip("_")

    return title


def add_concept(vocab, title):

    if not title:
        return

    concept_id = canonicalize(title)

    if not concept_id:
        return

    if concept_id not in vocab:
        vocab[concept_id] = {
            "display_name": title.strip(),
            "frequency": 0
        }

    vocab[concept_id]["frequency"] += 1


def main():

    vocab = {}

    resource_files = list(
        RESOURCE_DIR.rglob("*.json")
    )

    print(f"Found {len(resource_files)} resource files")

    total_resources = 0

    for file_path in resource_files:

        try:
            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:
                resources = json.load(f)

        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            continue

        total_resources += len(resources)

        for resource in resources:

            add_concept(
                vocab,
                resource.get("chapter_title")
            )

            add_concept(
                vocab,
                resource.get("section")
            )

            add_concept(
                vocab,
                resource.get("subsection")
            )

    sorted_vocab = dict(
        sorted(
            vocab.items(),
            key=lambda x: (
                -x[1]["frequency"],
                x[0]
            )
        )
    )

    output_path = (
        OUTPUT_DIR /
        "structural_concepts.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            sorted_vocab,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("Done.")
    print(f"Resources scanned : {total_resources}")
    print(f"Concepts found    : {len(sorted_vocab)}")
    print(f"Saved to          : {output_path}")


if __name__ == "__main__":
    main()
