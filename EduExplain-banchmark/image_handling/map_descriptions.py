import json
import re
import os

# STEP 1: Load JSON file

with open("outputs/figure_resources.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# STEP 2: Load descriptions.txt

with open("description.txt", "r", encoding="utf-8") as f:
    text = f.read()

# STEP 3: Extract descriptions
desc_map = {}
matches = re.findall(
r"Processing:\s*(figure_\d+.png).*?Description:\s*(.*?)(?=\nProcessing:|\Z)",
text,
re.DOTALL
)

for image_name, description in matches:
    desc_map[image_name.strip()] = description.strip()


print(f"Extracted {len(desc_map)} descriptions")


# STEP 4: Map into JSON

matched = 0

for item in data:
    image_path = item.get("image_path", "")
    file_name = os.path.basename(image_path)

    if file_name in desc_map:
        item["description"] = desc_map[file_name]
        matched += 1

print(f"Matched {matched} descriptions")

# STEP 5: Save updated JSON

with open("outputs/updated_figure_resources.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("Done! File saved.")
