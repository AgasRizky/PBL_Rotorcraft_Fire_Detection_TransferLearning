import os
import time
import copy
import json

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

NUM_CLASSES = 2
BATCH_SIZE = 8
NUM_EPOCHS = 10

# Learning rate untuk classifier/head
LEARNING_RATE = 0.001

SEED = 42

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("TRANSFER LEARNING - FIRE CLASSIFICATION")
print("=" * 60)
print(f"Device       : {DEVICE}")
print(f"Epoch        : {NUM_EPOCHS}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print("=" * 60)


# ============================================================
# TRANSFORM
# ============================================================

# Training:
# - Resize/crop
# - Random horizontal flip
# - Random rotation
# - ImageNet normalization
#
# Validation:
# - Tidak menggunakan augmentasi random

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
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

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=val_transform
)

print("\nClass mapping:")
print(train_dataset.class_to_idx)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

dataloaders = {
    "train": train_loader,
    "val": val_loader
}

dataset_sizes = {
    "train": len(train_dataset),
    "val": len(val_dataset)
}

print(f"\nTraining images   : {dataset_sizes['train']}")
print(f"Validation images: {dataset_sizes['val']}")


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(model_name):

    if model_name == "resnet18":

        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False

        # Replace classifier
        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    elif model_name == "resnet50":

        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False

        # Replace classifier
        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    elif model_name == "mobilenet":

        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        # Freeze backbone
        for param in model.features.parameters():
            param.requires_grad = False

        # Replace classifier
        input_features = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            input_features,
            NUM_CLASSES
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_model(model, model_name):

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Hanya parameter classifier/head yang trainable
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    start_time = time.time()

    print("\n")
    print("=" * 60)
    print(f"TRAINING: {model_name.upper()}")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
        )

        print("-" * 40)

        for phase in ["train", "val"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(
                    phase == "train"
                ):

                    outputs = model(inputs)

                    loss = criterion(
                        outputs,
                        labels
                    )

                    _, predictions = torch.max(
                        outputs,
                        1
                    )

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += (
                    loss.item()
                    * inputs.size(0)
                )

                running_corrects += torch.sum(
                    predictions == labels.data
                )

            epoch_loss = (
                running_loss /
                dataset_sizes[phase]
            )

            epoch_acc = (
                running_corrects.double() /
                dataset_sizes[phase]
            ).item()

            history[f"{phase}_loss"].append(
                epoch_loss
            )

            history[f"{phase}_acc"].append(
                epoch_acc
            )

            print(
                f"{phase.capitalize():5s} "
                f"Loss: {epoch_loss:.4f} "
                f"Acc: {epoch_acc:.4f}"
            )

            # Simpan model terbaik berdasarkan validation accuracy
            if (
                phase == "val"
                and epoch_acc > best_val_acc
            ):
                best_val_acc = epoch_acc
                best_model_weights = copy.deepcopy(
                    model.state_dict()
                )

    elapsed_time = time.time() - start_time

    print("\nTraining selesai.")
    print(
        f"Best validation accuracy: "
        f"{best_val_acc:.4f}"
    )

    print(
        f"Training time: "
        f"{elapsed_time:.2f} seconds"
    )

    # Load best model
    model.load_state_dict(best_model_weights)

    return model, history, elapsed_time, best_val_acc


# ============================================================
# PLOT HISTORY
# ============================================================

def save_graphs(history, model_name):

    result_dir = os.path.join(
        RESULTS_DIR,
        model_name
    )

    os.makedirs(
        result_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Accuracy graph
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    epochs = range(
        1,
        len(history["train_acc"]) + 1
    )

    plt.plot(
        epochs,
        history["train_acc"],
        marker="o",
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        history["val_acc"],
        marker="o",
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        f"{model_name.upper()} - Accuracy"
    )

    plt.legend()
    plt.grid(True)

    accuracy_path = os.path.join(
        result_dir,
        "accuracy.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Loss graph
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"{model_name.upper()} - Loss"
    )

    plt.legend()
    plt.grid(True)

    loss_path = os.path.join(
        result_dir,
        "loss.png"
    )

    plt.savefig(
        loss_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Accuracy graph: {accuracy_path}")
    print(f"Loss graph    : {loss_path}")


# ============================================================
# SAVE HISTORY
# ============================================================

def save_history(history, model_name):

    result_dir = os.path.join(
        RESULTS_DIR,
        model_name
    )

    path = os.path.join(
        result_dir,
        "history.json"
    )

    with open(path, "w") as f:
        json.dump(
            history,
            f,
            indent=4
        )

    print(f"History saved: {path}")


# ============================================================
# MAIN
# ============================================================

models_to_train = [
    "resnet18",
    "resnet50",
    "mobilenet"
]

summary = []

for model_name in models_to_train:

    model = create_model(model_name)

    model, history, training_time, best_val_acc = (
        train_model(
            model,
            model_name
        )
    )

    # Save model
    model_path = os.path.join(
        MODELS_DIR,
        f"{model_name}_best.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print(
        f"Model saved: {model_path}"
    )

    # Save graphs
    save_graphs(
        history,
        model_name
    )

    # Save history
    save_history(
        history,
        model_name
    )

    summary.append({
        "model": model_name,
        "best_val_accuracy": best_val_acc,
        "training_time_seconds": training_time
    })


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULTS_DIR,
    "training_summary.json"
)

with open(summary_path, "w") as f:
    json.dump(
        summary,
        f,
        indent=4
    )

print("\n")
print("=" * 60)
print("SEMUA MODEL SELESAI DILATIH")
print("=" * 60)

for result in summary:

    print(
        f"{result['model']:10s} | "
        f"Val Acc: "
        f"{result['best_val_accuracy']:.4f} | "
        f"Time: "
        f"{result['training_time_seconds']:.2f}s"
    )

print("=" * 60)
print(f"Summary: {summary_path}")
