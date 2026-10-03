import os
import json
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULT_DIR = os.path.join(BASE_DIR, "results")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")

os.makedirs(GRAPH_DIR, exist_ok=True)


# ============================================================
# LOAD TRAINING RESULTS
# ============================================================

with open(
    os.path.join(RESULT_DIR, "training_summary.json"),
    "r"
) as f:
    training_data = json.load(f)


# Convert training list -> dictionary
training_results = {
    item["model"]: item
    for item in training_data
}


# ============================================================
# LOAD TEST RESULTS
# ============================================================

with open(
    os.path.join(RESULT_DIR, "test_summary.json"),
    "r"
) as f:
    test_results = json.load(f)


# ============================================================
# MODEL NAMES
# ============================================================

models = [
    "resnet18",
    "resnet50",
    "mobilenet"
]

display_names = [
    "ResNet18",
    "ResNet50",
    "MobileNetV3-Small"
]


# ============================================================
# COLLECT DATA
# ============================================================

val_accuracy = [
    training_results[m]["best_val_accuracy"]
    for m in models
]

training_time = [
    training_results[m]["training_time_seconds"]
    for m in models
]

test_accuracy = [
    test_results[m]["test_accuracy"]
    for m in models
]

latency = [
    test_results[m]["latency_ms"]
    for m in models
]

fps = [
    test_results[m]["fps"]
    for m in models
]


# ============================================================
# PRINT TABLE
# ============================================================

print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    f"{'Model':20s}"
    f"{'Val Acc':>12s}"
    f"{'Test Acc':>12s}"
    f"{'Train Time':>15s}"
    f"{'Latency':>12s}"
    f"{'FPS':>10s}"
)

print("-" * 80)

for i in range(len(models)):

    print(
        f"{display_names[i]:20s}"
        f"{val_accuracy[i] * 100:>11.2f}%"
        f"{test_accuracy[i] * 100:>11.2f}%"
        f"{training_time[i]:>14.2f}s"
        f"{latency[i]:>11.2f}ms"
        f"{fps[i]:>10.2f}"
    )

print("=" * 80)


# ============================================================
# GRAPH 1 - VALIDATION vs TEST ACCURACY
# ============================================================

x = range(len(display_names))

plt.figure(figsize=(9, 6))

plt.plot(
    x,
    [v * 100 for v in val_accuracy],
    marker="o",
    linewidth=2,
    label="Best Validation Accuracy"
)

plt.plot(
    x,
    [v * 100 for v in test_accuracy],
    marker="s",
    linewidth=2,
    label="Test Accuracy"
)

plt.xticks(x, display_names)
plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 110)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_DIR, "accuracy_comparison.png"),
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 2 - TRAINING TIME
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    display_names,
    training_time
)

plt.ylabel("Training Time (seconds)")
plt.xlabel("Model")
plt.title("Training Time Comparison")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_DIR, "training_time_comparison.png"),
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 3 - LATENCY
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    display_names,
    latency
)

plt.ylabel("Latency (ms)")
plt.xlabel("Model")
plt.title("Inference Latency Comparison")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_DIR, "latency_comparison.png"),
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 4 - FPS
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    display_names,
    fps
)

plt.ylabel("FPS")
plt.xlabel("Model")
plt.title("Inference FPS Comparison")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_DIR, "fps_comparison.png"),
    dpi=300
)

plt.close()


print()
print("Graphs saved to:")
print(GRAPH_DIR)

print()
print("Generated files:")
print("- accuracy_comparison.png")
print("- training_time_comparison.png")
print("- latency_comparison.png")
print("- fps_comparison.png")
