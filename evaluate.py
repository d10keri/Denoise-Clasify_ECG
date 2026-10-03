import torch
import yaml
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from src.models.mlp import MLPBaseline
from src.data.ecg_dataset import make_dataloader

def evaluate(config_path: str, checkpoint_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPBaseline(
        input_size=cfg["model"]["input_size"],
        hidden1=cfg["model"]["hidden1"],
        hidden2=cfg["model"]["hidden2"],
        num_classes=cfg["model"]["num_classes"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    test_loader = make_dataloader(
        data_dir=cfg["paths"]["processed_dir"],
        split="test",
        batch_size=cfg["training"]["batch_size"],
        shuffle=False
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.to(device))
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_batch.tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    
    class_names = ["N", "S", "V", "F", "Q"]
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": report[str(i)]["precision"],
            "recall": report[str(i)]["recall"],
            "f1": report[str(i)]["f1-score"],
            "support": report[str(i)]["support"]
        }

    metrics = {
        "model": "MLPBaseline",
        "checkpoint": checkpoint_path,
        "best_epoch": ckpt["epoch"],
        "test_accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "per_class": per_class
    }
    
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    import csv
    
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    csv_path = results_dir / "test_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value", "Precision", "Recall", "F1", "Support"])
        writer.writerow(["Overall Accuracy", f"{acc:.4f}", "", "", "", ""])
        writer.writerow(["Macro-Average", "", f"{prec_macro:.4f}", f"{rec_macro:.4f}", f"{f1_macro:.4f}", ""])
        writer.writerow([])
        writer.writerow(["Class", "Precision", "Recall", "F1", "Support"])
        for name, m in per_class.items():
            writer.writerow([name, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}", m['support']])

    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)
    
    # Set labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Annotate cells
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
        
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(results_dir / "confusion_matrix.png")
    
    return metrics

if __name__ == "__main__":
    metrics = evaluate("configs/mlp_config.yaml", "checkpoints/mlp_best.pt")
    print(json.dumps(metrics, indent=2))
