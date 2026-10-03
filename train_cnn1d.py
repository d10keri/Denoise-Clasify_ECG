import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.models.cnn1d import CNN1DBaseline
from src.data.ecg_dataset import make_dataloader

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    "model": {"name": "CNN1DBaseline", "num_classes": 5},
    "training": {
        "seed": 42, "epochs": 30, "batch_size": 64,
        "learning_rate": 1e-3, "optimizer": "adam",
    },
    "paths": {
        "processed_dir": "data/processed",
        "metadata_dir":  "data/metadata",
        "checkpoint":    "checkpoints/cnn1d_best.pt",
        "results_dir":   "results/cnn1d",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"accuracy": acc, "macro_precision": p,
            "macro_recall": r, "macro_f1": f1}


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    return {"loss": avg_loss, **compute_metrics(all_labels, all_preds)}


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    return {"loss": avg_loss, **compute_metrics(all_labels, all_preds)}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    set_seed(CFG["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # DataLoaders — CNN needs channel dim
    train_loader = make_dataloader(
        CFG["paths"]["processed_dir"], "train",
        batch_size=CFG["training"]["batch_size"],
        add_channel_dim=True, shuffle=True,
    )
    val_loader = make_dataloader(
        CFG["paths"]["processed_dir"], "val",
        batch_size=CFG["training"]["batch_size"],
        add_channel_dim=True, shuffle=False,
    )

    # Model
    model = CNN1DBaseline(num_classes=CFG["model"]["num_classes"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Class weights from train metadata
    with open(Path(CFG["paths"]["metadata_dir"]) / "dataset_info.json") as f:
        info = json.load(f)
    class_weights = torch.tensor(info["class_weights"], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=CFG["training"]["learning_rate"])

    # Training loop
    best_val_f1 = -1.0
    best_epoch  = -1
    history = {
        "epochs": [], "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "train_macro_f1": [], "val_macro_f1": [],
        "best_epoch": -1,
    }

    Path(CFG["paths"]["checkpoint"]).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for epoch in range(1, CFG["training"]["epochs"] + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl = validate_one_epoch(model, val_loader, criterion, device)

        history["epochs"].append(epoch)
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(vl["loss"])
        history["train_accuracy"].append(tr["accuracy"])
        history["val_accuracy"].append(vl["accuracy"])
        history["train_macro_f1"].append(tr["macro_f1"])
        history["val_macro_f1"].append(vl["macro_f1"])

        print(
            f"Epoch {epoch:2d}/{CFG['training']['epochs']} | "
            f"Train Loss: {tr['loss']:.4f} Acc: {tr['accuracy']:.4f} F1: {tr['macro_f1']:.4f} | "
            f"Val Loss: {vl['loss']:.4f} Acc: {vl['accuracy']:.4f} F1: {vl['macro_f1']:.4f}"
        )

        if vl["macro_f1"] > best_val_f1:
            best_val_f1 = vl["macro_f1"]
            best_epoch  = epoch
            history["best_epoch"] = best_epoch
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1":         best_val_f1,
                "model_config":         CFG["model"],
                "training_config":      CFG["training"],
            }, CFG["paths"]["checkpoint"])
            print(f"  --> New best Val Macro-F1: {best_val_f1:.4f} (saved checkpoint)")

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s | Best epoch: {best_epoch} | Best Val Macro-F1: {best_val_f1:.4f}")

    # Save history
    results_dir = Path(CFG["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"History saved to {results_dir / 'history.json'}")


if __name__ == "__main__":
    main()
