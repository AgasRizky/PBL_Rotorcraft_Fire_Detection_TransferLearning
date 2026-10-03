import os
import csv

# Lokasi dataset raw
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_raw")
OUTPUT_FILE = os.path.join(BASE_DIR, "metadata.csv")

# Class yang digunakan
classes = ["fire", "no_fire"]

# Ekstensi gambar yang diterima
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

rows = []

for class_name in classes:
    class_dir = os.path.join(DATASET_DIR, class_name)

    if not os.path.exists(class_dir):
        print(f"WARNING: Folder tidak ditemukan: {class_dir}")
        continue

    for filename in sorted(os.listdir(class_dir)):
        if filename.lower().endswith(image_extensions):
            rows.append({
                "filename": filename,
                "class": class_name,
                "source": "dataset_raw"
            })

# Tulis metadata.csv
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["filename", "class", "source"]
    )

    writer.writeheader()
    writer.writerows(rows)

print("=" * 60)
print("METADATA DATASET")
print("=" * 60)
print(f"Output : {OUTPUT_FILE}")
print(f"Total  : {len(rows)} gambar")
print()

for class_name in classes:
    count = sum(1 for row in rows if row["class"] == class_name)
    print(f"{class_name:10s}: {count} gambar")

print("=" * 60)
