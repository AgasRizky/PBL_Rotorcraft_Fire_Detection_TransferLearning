import os
import json
import time
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "dataset")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
NUM_CLASSES = 2

CLASS_NAMES = ["fire", "no_fire"]


# ============================================================
# TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("=" * 60)
print("TEST EVALUATION - FIRE CLASSIFICATION")
print("=" * 60)

print(f"Device       : {DEVICE}")
print(f"Test images  : {len(test_dataset)}")
print(f"Class mapping: {test_dataset.class_to_idx}")
print("=" * 60)


# ============================================================
# MODEL FUNCTIONS
# ============================================================

def create_resnet18():
    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model


def create_resnet50():
    model = models.resnet50(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model


def create_mobilenet():
    model = models.mobilenet_v3_small(
        weights=None
    )

    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# MODEL CONFIGURATION
# ============================================================

models_config = {
    "resnet18": (
        create_resnet18,
        os.path.join(MODEL_DIR, "resnet18_best.pth")
    ),

    "resnet50": (
        create_resnet50,
        os.path.join(MODEL_DIR, "resnet50_best.pth")
    ),

    "mobilenet": (
        create_mobilenet,
        os.path.join(MODEL_DIR, "mobilenet_best.pth")
    )
}


# ============================================================
# EVALUATION
# ============================================================

all_results = {}


for model_name, (model_creator, model_path) in models_config.items():

    print()
    print("=" * 60)
    print(f"TESTING: {model_name.upper()}")
    print("=" * 60)

    model = model_creator()

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()

    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predictions = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = correct / total

    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Correct       : {correct}/{total}")

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1]
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    dummy_input = torch.randn(
        1,
        3,
        224,
        224
    ).to(DEVICE)

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    # Measure
    repetitions = 20

    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(repetitions):
            _ = model(dummy_input)

    end_time = time.perf_counter()

    total_time = end_time - start_time

    latency_ms = (
        total_time / repetitions
    ) * 1000

    fps = 1000 / latency_ms

    print(f"\nInference Latency : {latency_ms:.2f} ms/image")
    print(f"Approx FPS        : {fps:.2f}")

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    model_result_dir = os.path.join(
        RESULT_DIR,
        model_name
    )

    os.makedirs(
        model_result_dir,
        exist_ok=True
    )

    plt.figure(figsize=(5, 4))

    plt.imshow(cm)

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.colorbar()

    plt.xticks(
        [0, 1],
        CLASS_NAMES
    )

    plt.yticks(
        [0, 1],
        CLASS_NAMES
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    cm_path = os.path.join(
        model_result_dir,
        "confusion_matrix.png"
    )

    plt.savefig(
        cm_path,
        dpi=200
    )

    plt.close()

    print(f"Confusion matrix saved: {cm_path}")

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    all_results[model_name] = {
        "test_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "latency_ms": latency_ms,
        "fps": fps,
        "confusion_matrix": cm.tolist()
    }


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULT_DIR,
    "test_summary.json"
)

with open(
    summary_path,
    "w"
) as f:

    json.dump(
        all_results,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("TEST EVALUATION SELESAI")
print("=" * 60)

for model_name, result in all_results.items():

    print(
        f"{model_name:10s} | "
        f"Test Acc: {result['test_accuracy']:.2%} | "
        f"Latency: {result['latency_ms']:.2f} ms | "
        f"FPS: {result['fps']:.2f}"
    )

print("=" * 60)

print(
    f"Summary saved: {summary_path}"
)
