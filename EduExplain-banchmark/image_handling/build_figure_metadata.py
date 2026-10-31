import os
import json
import re

# =========================
# CONFIG
# =========================
FIGURE_DIR = "./figures"
OUTPUT_PATH = "./outputs/figure_resources.json"

RESOURCE_TYPE = "figure"

# subject mapping
SUBJECT_MAP = {
    "SCI": "Science",
    "BIO": "Biology",
    "CHEM": "Chemistry",
    "PHY": "Physics"
}

# =========================
# BUILD RESOURCES
# =========================
resources = []

for folder in sorted(os.listdir(FIGURE_DIR)):

    folder_path = os.path.join(FIGURE_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    print(f"Processing folder: {folder}")

    # =========================
    # EXTRACT METADATA FROM FOLDER NAME
    # =========================
    # Example: CLASS_10_SCI_CH_01

    class_match = re.search(r"CLASS_(\d+)", folder)
    chapter_match = re.search(r"CH_(\d+)", folder)
    subject_match = re.search(r"CLASS_\d+_([A-Z]+)_CH", folder)

    if not (class_match and chapter_match and subject_match):
        print(f"⚠️ Skipping invalid folder name: {folder}")
        continue

    class_no = int(class_match.group(1))
    chapter_no = int(chapter_match.group(1))
    subject_code = subject_match.group(1)

    subject = SUBJECT_MAP.get(subject_code, subject_code)

    document_id = folder  # PERFECT now

    # =========================
    # PROCESS IMAGES IN FOLDER
    # =========================
    for idx, file in enumerate(sorted(os.listdir(folder_path))):

        if not file.lower().endswith(".png"):
            continue

        image_path = os.path.join(folder_path, file)

        resource = {
            "document_id": document_id,

            "resource_id": f"{document_id}_figure_{idx}",

            "resource_type": RESOURCE_TYPE,

            "class_no": class_no,
            "subject": subject,

            "chapter_no": chapter_no,
            "chapter_title": None,  # optional (can fill later)

            "section": None,
            "subsection": None,

            "page_start": None,
            "page_end": None,

            "text": None,

            "image_path": image_path,
            "image_caption": None,

            # to be filled later
            "description": None,
            "summary": None,
            "concepts": [],

            "has_equation": False
        }

        resources.append(resource)

# =========================
# SAVE JSON
# =========================
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(resources, f, indent=4, ensure_ascii=False)

print("\n✅ Done!")
print(f"Total figures: {len(resources)}")
print(f"Saved to: {OUTPUT_PATH}")
