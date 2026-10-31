"""
filter_structural_vocab.py
==========================

Input:
    vocab/structural_concepts.json

Output:
    vocab/structural_concepts_filtered.json

Removes only obvious non-concept administrative headings.
No frequency filtering.
No length filtering.
No review buckets.

Everything else is retained as part of the structural vocabulary.
"""

import json
from pathlib import Path


INPUT_FILE = Path("data/concepts/structural_concepts.json")

OUTPUT_FILE = Path(
    "data/concepts/structural_concepts_filtered.json"
)


BAD_EXACT = {
    "introduction",
    "summary",
    "exercise",
    "exercises",
    "activity",
    "activities",
    "appendix",
    "what you have learnt",
    "points to ponder",
    "chapter summary",
    "review questions",
    "learning",
    "understanding",
    "moving",
    "basic",
    "important"
}


def should_keep(display_name: str) -> bool:

    text = display_name.lower().strip()

    if text in BAD_EXACT:
        return False

    return True


def main():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        vocab = json.load(f)

    filtered_vocab = {}

    removed = 0

    for concept_id, info in vocab.items():

        display_name = info["display_name"]

        if should_keep(display_name):

            filtered_vocab[concept_id] = info

        else:

            removed += 1

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            filtered_vocab,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("STRUCTURAL VOCAB FILTER COMPLETE")
    print("=" * 60)
    print(f"Original concepts : {len(vocab)}")
    print(f"Removed concepts  : {removed}")
    print(f"Final concepts    : {len(filtered_vocab)}")
    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()