import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import json
import random
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.models.mlp import MLPBaseline
from src.data.ecg_dataset import make_dataloader

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_cls, rec_cls, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return {
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "f1_per_class": f1_cls.tolist()
    }

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y_batch.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return {"loss": avg_loss, **metrics}

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return {"loss": avg_loss, **metrics}

def main():
    with open("configs/mlp_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPBaseline(
        input_size=cfg["model"]["input_size"],
        hidden1=cfg["model"]["hidden1"],
        hidden2=cfg["model"]["hidden2"],
        num_classes=cfg["model"]["num_classes"]
    ).to(device)

    train_loader = make_dataloader(
        data_dir=cfg["paths"]["processed_dir"],
        split="train",
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )
    val_loader = make_dataloader(
        data_dir=cfg["paths"]["processed_dir"],
        split="val",
        batch_size=cfg["training"]["batch_size"],
        shuffle=False
    )

    with open(Path(cfg["paths"]["metadata_dir"]) / "dataset_info.json", "r") as f:
        info = json.load(f)
    class_weights = torch.tensor(info["class_weights"], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))

    best_val_f1 = -1.0
    best_epoch = -1
    
    history = {
        "epochs": [],
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "train_macro_f1": [], "val_macro_f1": [],
        "best_epoch": -1
    }

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        history["epochs"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["train_macro_f1"].append(train_metrics["macro_f1"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        print(f"Epoch {epoch:2d}/{cfg['training']['epochs']} | Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            history["best_epoch"] = best_epoch
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": best_val_f1,
                "config": cfg,
            }
            torch.save(checkpoint, Path(cfg["paths"]["checkpoint_dir"]) / "mlp_best.pt")
            print(f"  --> New best Val Macro-F1: {best_val_f1:.4f} (saved checkpoint)")

    with open(Path(cfg["paths"]["results_dir"]) / "history.json", "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
