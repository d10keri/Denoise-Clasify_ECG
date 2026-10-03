import torch
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

from src.models.cnn1d import CNN1DBaseline
from src.data.ecg_dataset import make_dataloader

CHECKPOINT  = "checkpoints/cnn1d_best.pt"
CONFIG_PATH = "data/processed"          # processed_dir
RESULTS_DIR = Path("results/cnn1d")
CLASS_NAMES = ["N", "S", "V", "F", "Q"]
BATCH_SIZE  = 64


def evaluate():
    ckpt   = torch.load(CHECKPOINT, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = ckpt.get("model_config", {"num_classes": 5})
    model = CNN1DBaseline(num_classes=model_cfg["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    test_loader = make_dataloader(
        CONFIG_PATH, "test",
        batch_size=BATCH_SIZE,
        add_channel_dim=True,
        shuffle=False,
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            logits = model(X.to(device))
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.tolist())

    # ── Metrics ────────────────────────────────────────────────────────────
    acc = accuracy_score(all_labels, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    report = classification_report(
        all_labels, all_preds, output_dict=True, zero_division=0
    )
    per_class = {
        name: {
            "precision": report[str(i)]["precision"],
            "recall":    report[str(i)]["recall"],
            "f1":        report[str(i)]["f1-score"],
            "support":   report[str(i)]["support"],
        }
        for i, name in enumerate(CLASS_NAMES)
    }

    metrics = {
        "model":           "CNN1DBaseline",
        "checkpoint":      CHECKPOINT,
        "best_epoch":      ckpt["epoch"],
        "test_accuracy":   acc,
        "macro_precision": p_macro,
        "macro_recall":    r_macro,
        "macro_f1":        f1_macro,
        "per_class":       per_class,
    }

    # ── Save outputs ───────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(RESULTS_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # CSV
    with open(RESULTS_DIR / "test_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value", "Precision", "Recall", "F1", "Support"])
        writer.writerow(["Overall Accuracy", f"{acc:.4f}", "", "", "", ""])
        writer.writerow(["Macro-Average", "", f"{p_macro:.4f}", f"{r_macro:.4f}", f"{f1_macro:.4f}", ""])
        writer.writerow([])
        writer.writerow(["Class", "Precision", "Recall", "F1", "Support"])
        for name, m in per_class.items():
            writer.writerow([name, f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                             f"{m['f1']:.4f}", m["support"]])

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)
    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES))); ax.set_yticklabels(CLASS_NAMES)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f"{z:.2f}", ha="center", va="center",
                color="white" if z > 0.5 else "black")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.title("CNN1D — Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    return metrics


if __name__ == "__main__":
    metrics = evaluate()
    print(json.dumps(metrics, indent=2))
