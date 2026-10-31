"""
build_resources.py
==================
CLI script: reads parsed chunks + raw extraction, emits resource JSON files.

Usage:
    python build_resources.py \
        --parsed_dir  parsed_extraction/ \
        --raw_dir     raw_marker_extraction/ \
        --output_dir  resources/ \
        --manifest    manifest.json

Manifest format (same manifest used by your existing parser):
    [
        {
            "document_id": "NCERT_SCI_11",
            "parsed_file": "class_11_chem/ch01.json",
            "raw_file":    "class_11_chem/ch01.json"
        },
        ...
    ]

If no manifest is provided, the script walks parsed_dir recursively
and matches raw files by relative path.
"""

import argparse
import json
import os
from pathlib import Path

from src.resources.resource_builder import ResourceBuilder


def load_json(path: str) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_one(
    document_id: str,
    parsed_path: str,
    raw_path: str | None,
    output_path: str,
) -> int:
    chunks = load_json(parsed_path)
    if not isinstance(chunks, list):
        print(f"  [SKIP] {parsed_path}: expected list of chunks")
        return 0

    raw_extraction = load_json(raw_path) if raw_path and os.path.exists(raw_path) else {}

    builder = ResourceBuilder(
        document_id=document_id,
        raw_extraction=raw_extraction,
    )
    resources = builder.build_from_chunks(chunks)
    save_json([r.to_dict() for r in resources], output_path)
    return len(resources)


def main():
    parser = argparse.ArgumentParser(
        description="Convert parsed NCERT chunks to benchmark resource JSON."
    )
    parser.add_argument("--parsed_dir",  required=True)
    parser.add_argument("--raw_dir",     required=True)
    parser.add_argument("--output_dir",  required=True)
    #parser.add_argument("--manifest",    default=None)
    args = parser.parse_args()

    
    
    # Auto-discover: walk parsed_dir, match raw by relative path
    entries = []
    for parsed_path in sorted(Path(args.parsed_dir).rglob("*.json")):
        rel = parsed_path.relative_to(args.parsed_dir)
        raw_path = Path(args.raw_dir) / rel
        out_path = Path(args.output_dir) / rel
        # Use filename stem as document_id fallback
        rel = parsed_path.relative_to(args.parsed_dir)

        subject_folder = rel.parts[0].upper()
        chapter_name = parsed_path.stem.upper()

        document_id = f"{subject_folder}_{chapter_name}"
        entries.append((document_id, str(parsed_path), str(raw_path), str(out_path)))

    total = 0
    for document_id, parsed_path, raw_path, output_path in entries:
        print(f"\n{document_id}")
        print(f"  parsed : {parsed_path}")
        print(f"  raw    : {raw_path}")
        n = process_one(document_id, parsed_path, raw_path, output_path)
        print(f"  → {n} resources written to {output_path}")
        total += n

    print(f"\nDone. Total resources: {total}")


if __name__ == "__main__":
    main()
