import os
import random
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "dataset_raw")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

CLASSES = ["fire", "no_fire"]

# Jumlah data per kelas
TRAIN_COUNT = 18
VAL_COUNT = 5
TEST_COUNT = 2

# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)

# ============================================================
# SPLIT DATASET
# ============================================================

for class_name in CLASSES:

    source_dir = os.path.join(RAW_DIR, class_name)

    train_dir = os.path.join(DATASET_DIR, "train", class_name)
    val_dir = os.path.join(DATASET_DIR, "val", class_name)
    test_dir = os.path.join(DATASET_DIR, "test", class_name)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Ambil semua file gambar
    images = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Urutkan terlebih dahulu agar reproducible
    images.sort()

    # Pastikan jumlah sesuai
    required = TRAIN_COUNT + VAL_COUNT + TEST_COUNT

    if len(images) < required:
        raise ValueError(
            f"Kelas {class_name} hanya memiliki {len(images)} gambar. "
            f"Dibutuhkan minimal {required}."
        )

    # Random shuffle
    random.shuffle(images)

    # Split
    train_images = images[:TRAIN_COUNT]
    val_images = images[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]
    test_images = images[
        TRAIN_COUNT + VAL_COUNT:
        TRAIN_COUNT + VAL_COUNT + TEST_COUNT
    ]

    # Copy file
    for filename in train_images:
        shutil.copy2(
            os.path.join(source_dir, filename),
            os.path.join(train_dir, filename)
        )

    for filename in val_images:
        shutil.copy2(
            os.path.join(source_dir, filename),
            os.path.join(val_dir, filename)
        )

    for filename in test_images:
        shutil.copy2(
            os.path.join(source_dir, filename),
            os.path.join(test_dir, filename)
        )

    print(f"\nClass: {class_name}")
    print(f"  Train      : {len(train_images)}")
    print(f"  Validation : {len(val_images)}")
    print(f"  Test       : {len(test_images)}")

print("\n===================================")
print("DATASET SPLIT SELESAI")
print("===================================")
print(f"Train      : {TRAIN_COUNT * len(CLASSES)}")
print(f"Validation : {VAL_COUNT * len(CLASSES)}")
print(f"Test       : {TEST_COUNT * len(CLASSES)}")
print(f"Total      : {(TRAIN_COUNT + VAL_COUNT + TEST_COUNT) * len(CLASSES)}")
print("===================================")
