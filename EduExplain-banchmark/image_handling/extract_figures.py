from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download

import cv2
import os

# =========================
# PATHS
# =========================
PAGE_ROOT = "./page_images"
FIGURE_ROOT = "./figures"

os.makedirs(FIGURE_ROOT, exist_ok=True)

print("Loading model...")

# =========================
# LOAD MODEL
# =========================
model_path = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
)

model = YOLOv10(model_path)

print("Classes:")
print(model.names)

# =========================
# FIND FIGURE CLASS
# =========================
figure_class = None

for k, v in model.names.items():
    if str(v).lower() == "figure":
        figure_class = k
        break

if figure_class is None:
    raise ValueError("❌ Figure class not found in model!")

print("✅ Figure class =", figure_class)

# =========================
# EXTRACTION LOOP
# =========================
total_count = 0

for book in sorted(os.listdir(PAGE_ROOT)):

    book_folder = os.path.join(PAGE_ROOT, book)

    if not os.path.isdir(book_folder):
        continue

    print(f"\n📂 Processing: {book}")

    # ✅ Create same folder inside figures
    save_folder = os.path.join(FIGURE_ROOT, book)
    os.makedirs(save_folder, exist_ok=True)

    local_count = 0  # reset per chapter

    for image_file in sorted(os.listdir(book_folder)):

        if not image_file.endswith(".png"):
            continue

        image_path = os.path.join(book_folder, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"⚠️ Skipping unreadable file: {image_path}")
            continue

        # =========================
        # RUN MODEL
        # =========================
        results = model(image_path, conf=0.30)
        boxes = results[0].boxes

        if boxes is None:
            continue

        for box in boxes:

            cls = int(box.cls[0])

            if cls != figure_class:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # safety check
            h, w, _ = image.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # =========================
            # SAVE WITH PROPER NAME
            # =========================
            figure_name = f"{book}_figure_{local_count}.png"

            save_path = os.path.join(save_folder, figure_name)

            cv2.imwrite(save_path, crop)

            print(f"✅ Saved: {save_path}")

            local_count += 1
            total_count += 1

print("\n🎉 Total Figures Extracted:", total_count)
