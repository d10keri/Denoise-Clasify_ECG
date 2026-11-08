"""
E2. Huấn luyện mô hình Depthwise Separable 1D-CNN (CNN1DBinaryDS) qua 3 seeds (42, 43, 44).
Hỗ trợ --variant B (mặc định, ưu tiên độ chính xác), A (đối chứng), C (ưu tiên chi phí).

Giữ nguyên toàn bộ file và checkpoint của M1, M1-Reg trước đó làm backup.
Lưu:
- checkpoints/cnn1d_binary_ds{V}_best_seed{S}.pt
- results/binary/m2_ds_fp32/{V}/history_seed{S}.json
"""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from src.models.cnn1d_binary_ds import CNN1DBinaryDS, DS_VARIANTS, count_bn_folded_params
from src.data.binary_dataset import make_binary_dataloader, ECGBinaryDataset

CFG = {
    "training": {
        "seeds": [42, 43, 44],
        "epochs": 40,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "min_lr": 1e-5,
        "weight_decay": 1e-3,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "augmentation": {
            "max_shift_samples": 3,
            "scale_range": [0.88, 1.12],
            "jitter_std": 0.015,
        },
        "checkpoint_metric": "val_auroc",
    },
    "paths": {
        "processed_dir": "data/processed",
        "checkpoint_dir": "checkpoints",
        "results_base_dir": "results/binary/m2_ds_fp32",
    },
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def augment_ecg_batch(x: torch.Tensor) -> torch.Tensor:
    b, c, length = x.shape
    out = x.clone()

    max_shift = CFG["training"]["augmentation"]["max_shift_samples"]
    if max_shift > 0:
        shifts = torch.randint(-max_shift, max_shift + 1, (b,), device=x.device)
        for i in range(b):
            s = int(shifts[i].item())
            if s > 0:
                out[i, :, s:] = x[i, :, : length - s]
                out[i, :, :s] = x[i, :, :1]
            elif s < 0:
                s_abs = -s
                out[i, :, : length - s_abs] = x[i, :, s_abs:]
                out[i, :, length - s_abs :] = x[i, :, -1:]

    low, high = CFG["training"]["augmentation"]["scale_range"]
    scales = torch.empty((b, 1, 1), device=x.device).uniform_(low, high)
    out = out * scales

    std = CFG["training"]["augmentation"]["jitter_std"]
    if std > 0:
        out = out + torch.randn_like(out) * std

    return out


def compute_binary_metrics(y_true: np.ndarray, probs_pos: np.ndarray, threshold: float = 0.5):
    y_pred = (probs_pos >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp

    acc = (tn + tp) / max(total, 1)
    recall_pos = tp / max(tp + fn, 1)
    spec_neg = tn / max(tn + fp, 1)
    prec_pos = tp / max(tp + fp, 1)
    npv_neg = tn / max(tn + fn, 1)

    f1_pos = 2.0 * prec_pos * recall_pos / max(prec_pos + recall_pos, 1e-12)
    f1_neg = 2.0 * npv_neg * spec_neg / max(npv_neg + spec_neg, 1e-12)
    macro_f1 = 0.5 * (f1_pos + f1_neg)
    auroc = float(roc_auc_score(y_true, probs_pos))
    pr_auc = float(average_precision_score(y_true, probs_pos))

    return {
        "accuracy": float(acc),
        "recall_non_N": float(recall_pos),
        "specificity_N": float(spec_neg),
        "precision_non_N": float(prec_pos),
        "f1_non_N": float(f1_pos),
        "macro_f1": float(macro_f1),
        "youden_j": float(recall_pos + spec_neg - 1.0),
        "auroc": auroc,
        "pr_auc": pr_auc,
    }


def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_probs_pos = []
    all_labels = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if is_train:
                X = augment_ecg_batch(X)
                optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
            all_probs_pos.append(probs)
            all_labels.append(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_true = np.concatenate(all_labels, axis=0)
    probs_pos = np.concatenate(all_probs_pos, axis=0)
    metrics = compute_binary_metrics(y_true, probs_pos, threshold=0.5)
    return {"loss": float(avg_loss), **metrics}


def train_single_seed(variant: str, seed: int, device: torch.device):
    set_seed(seed)
    var_cfg = DS_VARIANTS[variant]
    epochs = CFG["training"]["epochs"]

    print(f"\n{'='*78}")
    print(f"TRAINING CNN1DBinaryDS (Variant {variant}) | SEED = {seed} | DEVICE = {device}")
    print(f"Description: {var_cfg['description']}")
    print(f"{'='*78}")

    train_loader = make_binary_dataloader(
        CFG["paths"]["processed_dir"],
        "train",
        batch_size=CFG["training"]["batch_size"],
        add_channel_dim=True,
        shuffle=True,
    )
    val_loader = make_binary_dataloader(
        CFG["paths"]["processed_dir"],
        "val",
        batch_size=256,
        add_channel_dim=True,
        shuffle=False,
    )

    train_ds: ECGBinaryDataset = train_loader.dataset
    class_weights = train_ds.get_binary_class_weights().to(device)

    model = CNN1DBinaryDS(variant=variant, num_classes=2).to(device)
    raw_params = sum(p.numel() for p in model.parameters())
    fused_params = count_bn_folded_params(model)
    if seed == CFG["training"]["seeds"][0]:
        print(
            f"Trainable Params: {raw_params:,} | MCU Fused Params: {fused_params:,} | "
            f"MACCs: {var_cfg['expected_maccs']:,}"
        )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["training"]["learning_rate"],
        weight_decay=CFG["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=CFG["training"]["min_lr"],
    )

    best_val_auroc = -1.0
    best_epoch = -1

    history = {
        "variant": variant,
        "seed": seed,
        "pytorch_trainable_params": raw_params,
        "mcu_fused_params": fused_params,
        "expected_maccs": var_cfg["expected_maccs"],
        "class_weights": class_weights.cpu().tolist(),
        "epochs": [],
        "lr": [],
        "train_loss": [],
        "val_loss": [],
        "train_auroc": [],
        "val_auroc": [],
        "val_pr_auc": [],
        "val_recall_non_N": [],
        "val_specificity_N": [],
        "val_f1_non_N": [],
        "val_macro_f1": [],
        "best_epoch": -1,
        "best_val_auroc": -1.0,
    }

    ckpt_dir = Path(CFG["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"cnn1d_binary_ds{variant}_best_seed{seed}.pt"

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        tr = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        vl = run_epoch(model, val_loader, criterion, optimizer, device, is_train=False)
        scheduler.step()

        history["epochs"].append(epoch)
        history["lr"].append(current_lr)
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(vl["loss"])
        history["train_auroc"].append(tr["auroc"])
        history["val_auroc"].append(vl["auroc"])
        history["val_pr_auc"].append(vl["pr_auc"])
        history["val_recall_non_N"].append(vl["recall_non_N"])
        history["val_specificity_N"].append(vl["specificity_N"])
        history["val_f1_non_N"].append(vl["f1_non_N"])
        history["val_macro_f1"].append(vl["macro_f1"])

        marker = ""
        if vl["auroc"] > best_val_auroc:
            best_val_auroc = vl["auroc"]
            best_epoch = epoch
            history["best_epoch"] = best_epoch
            history["best_val_auroc"] = best_val_auroc
            torch.save(
                {
                    "variant": variant,
                    "seed": seed,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auroc": best_val_auroc,
                    "val_pr_auc": vl["pr_auc"],
                    "val_macro_f1": vl["macro_f1"],
                    "val_f1_non_N": vl["f1_non_N"],
                    "val_recall_non_N": vl["recall_non_N"],
                    "val_specificity_N": vl["specificity_N"],
                    "model_config": {"name": "CNN1DBinaryDS", "variant": variant, "num_classes": 2},
                    "training_config": CFG["training"],
                    "class_weights": class_weights.cpu().tolist(),
                },
                ckpt_path,
            )
            marker = " <-- [BEST VAL AUROC]"

        print(
            f"Epoch {epoch:2d}/{epochs} (lr={current_lr:.5f}) | "
            f"TrLoss: {tr['loss']:.4f} TrAUC: {tr['auroc']:.4f} | "
            f"VlLoss: {vl['loss']:.4f} VlAUC: {vl['auroc']:.4f} VlPR: {vl['pr_auc']:.4f} "
            f"Rec(1): {vl['recall_non_N']:.4f} Spec(0): {vl['specificity_N']:.4f} "
            f"F1(1): {vl['f1_non_N']:.4f}{marker}"
        )

    elapsed = time.time() - t0
    history["elapsed_seconds"] = round(elapsed, 2)

    results_dir = Path(CFG["paths"]["results_base_dir"]) / variant
    results_dir.mkdir(parents=True, exist_ok=True)
    hist_path = results_dir / f"history_seed{seed}.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(
        f"Variant {variant} | Seed {seed} finished in {elapsed:.1f}s | "
        f"Best Epoch: {best_epoch} (Val AUROC={best_val_auroc:.4f}) | "
        f"Saved: {ckpt_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="B", choices=["A", "B", "C"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant = args.variant.upper()
    for seed in CFG["training"]["seeds"]:
        train_single_seed(variant, seed, device)


if __name__ == "__main__":
    main()
