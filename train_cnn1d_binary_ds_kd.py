"""
E4. Huấn luyện Knowledge Distillation (Chưng cất tri thức) cho CNN1DBinaryDS (Student)
từ tổ hợp 3 mô hình Teacher M1-Reg (CNN1DBinaryRegularized seeds 42, 43, 44).

- Teacher: Ensemble trung bình xác suất mềm (soft targets ở nhiệt độ T) của 3 checkpoints
  M1-Reg (`checkpoints/cnn1d_binary_reg_best_seed{42,43,44}.pt`), chỉ chạy lúc train,
  chi phí trên STM32 lúc suy luận = 0.
- Student: `CNN1DBinaryDS` (mặc định `--variant B`, 1,468,544 MACCs, 29,058 fused params).
- Hàm mất mát:
    L = alpha * CE_weighted(logits_S, y) + (1 - alpha) * T^2 * KLDiv(log_softmax(logits_S / T), prob_T(x / T))
- Bảo toàn 100% các checkpoints trước đó, lưu mới tại:
  - `checkpoints/cnn1d_binary_ds{V}_kd_best_seed{S}.pt`
  - `results/binary/m2_ds_kd_fp32/{V}/history_seed{S}.json`
"""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from src.models.cnn1d_binary_reg import CNN1DBinaryRegularized
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
        "kd": {
            "alpha": 0.5,
            "temperature": 3.0,
            "teacher_seeds": [42, 43, 44],
        },
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
        "results_base_dir": "results/binary/m2_ds_kd_fp32",
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
    """
    Augmentation nhịp ECG trực tiếp trên GPU (đã vector hóa theo giá trị dịch chuyển s):
    1. Dịch thời gian ngẫu nhiên +-3 mẫu (~+-8.3 ms ở 360 Hz).
    2. Co giãn biên độ ngẫu nhiên U(0.88, 1.12).
    3. Nhiễu trắng Gaussian N(0, 0.015^2).
    """
    b, c, length = x.shape
    out = x.clone()

    max_shift = CFG["training"]["augmentation"]["max_shift_samples"]
    if max_shift > 0:
        shifts = torch.randint(-max_shift, max_shift + 1, (b,), device=x.device)
        for s in range(-max_shift, max_shift + 1):
            if s == 0:
                continue
            mask = shifts == s
            if not mask.any():
                continue
            if s > 0:
                out[mask, :, s:] = x[mask, :, : length - s]
                out[mask, :, :s] = x[mask, :, :1]
            else:
                s_abs = -s
                out[mask, :, : length - s_abs] = x[mask, :, s_abs:]
                out[mask, :, length - s_abs :] = x[mask, :, -1:]

    low, high = CFG["training"]["augmentation"]["scale_range"]
    scales = torch.empty((b, 1, 1), device=x.device).uniform_(low, high)
    out = out * scales

    std = CFG["training"]["augmentation"]["jitter_std"]
    if std > 0:
        out = out + torch.randn_like(out) * std

    return out


def load_teacher_ensemble(device: torch.device) -> list[nn.Module]:
    """Nạp 3 checkpoints M1-Reg làm Teacher Ensemble (đóng băng trọng số, chế độ eval)."""
    ckpt_dir = Path(CFG["paths"]["checkpoint_dir"])
    teachers = []
    for t_seed in CFG["training"]["kd"]["teacher_seeds"]:
        t_path = ckpt_dir / f"cnn1d_binary_reg_best_seed{t_seed}.pt"
        if not t_path.exists():
            raise FileNotFoundError(f"Không tìm thấy Teacher checkpoint: {t_path}")
        teacher = CNN1DBinaryRegularized(num_classes=2).to(device)
        ckpt = torch.load(t_path, map_location=device, weights_only=False)
        teacher.load_state_dict(ckpt["model_state_dict"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teachers.append(teacher)
    return teachers


@torch.no_grad()
def get_teacher_soft_probs(teachers: list[nn.Module], x: torch.Tensor, temperature: float) -> torch.Tensor:
    """Tính trung bình xác suất mềm (soft targets ở nhiệt độ T) từ tổ hợp Teacher M1-Reg."""
    probs_sum = None
    for teacher in teachers:
        logits_t = teacher(x)
        probs_t = F.softmax(logits_t / temperature, dim=1)
        if probs_sum is None:
            probs_sum = probs_t
        else:
            probs_sum = probs_sum + probs_t
    return probs_sum / float(len(teachers))


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


def run_epoch_kd(
    student: nn.Module,
    teachers: list[nn.Module],
    loader,
    ce_criterion: nn.Module,
    optimizer,
    device: torch.device,
    alpha: float,
    temperature: float,
    is_train: bool,
):
    if is_train:
        student.train()
    else:
        student.eval()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_kd_loss = 0.0
    all_probs_pos = []
    all_labels = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if is_train:
                X = augment_ecg_batch(X)
                optimizer.zero_grad()

            logits_s = student(X)
            ce_loss = ce_criterion(logits_s, y)

            # Soft targets từ Teacher Ensemble ở nhiệt độ T
            probs_t = get_teacher_soft_probs(teachers, X, temperature)
            log_probs_s = F.log_softmax(logits_s / temperature, dim=1)
            kd_loss = F.kl_div(log_probs_s, probs_t, reduction="batchmean") * (temperature ** 2)

            loss = alpha * ce_loss + (1.0 - alpha) * kd_loss

            if is_train:
                loss.backward()
                optimizer.step()

            batch_sz = len(y)
            total_loss += loss.item() * batch_sz
            total_ce_loss += ce_loss.item() * batch_sz
            total_kd_loss += kd_loss.item() * batch_sz

            probs = torch.softmax(logits_s.detach(), dim=1)[:, 1].cpu().numpy()
            all_probs_pos.append(probs)
            all_labels.append(y.cpu().numpy())

    n_samples = len(loader.dataset)
    avg_loss = total_loss / n_samples
    avg_ce = total_ce_loss / n_samples
    avg_kd = total_kd_loss / n_samples

    y_true = np.concatenate(all_labels, axis=0)
    probs_pos = np.concatenate(all_probs_pos, axis=0)
    metrics = compute_binary_metrics(y_true, probs_pos, threshold=0.5)
    return {
        "loss": float(avg_loss),
        "ce_loss": float(avg_ce),
        "kd_loss": float(avg_kd),
        **metrics,
    }


def train_single_seed_kd(
    variant: str,
    seed: int,
    alpha: float,
    temperature: float,
    teachers: list[nn.Module],
    device: torch.device,
):
    set_seed(seed)
    var_cfg = DS_VARIANTS[variant]
    epochs = CFG["training"]["epochs"]

    print(f"\n{'='*82}")
    print(
        f"TRAINING KD Student CNN1DBinaryDS (Variant {variant}) | SEED = {seed} | "
        f"alpha={alpha}, T={temperature} | DEVICE = {device}"
    )
    print(f"Teacher: Ensemble of 3 M1-Reg checkpoints (seeds {CFG['training']['kd']['teacher_seeds']})")
    print(f"{'='*82}")

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

    student = CNN1DBinaryDS(variant=variant, num_classes=2).to(device)
    raw_params = sum(p.numel() for p in student.parameters())
    fused_params = count_bn_folded_params(student)
    if seed == CFG["training"]["seeds"][0]:
        print(
            f"Student Params: {raw_params:,} | MCU Fused Params: {fused_params:,} | "
            f"MACCs: {var_cfg['expected_maccs']:,}"
        )

    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        student.parameters(),
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
        "kd_alpha": alpha,
        "kd_temperature": temperature,
        "teacher_seeds": CFG["training"]["kd"]["teacher_seeds"],
        "pytorch_trainable_params": raw_params,
        "mcu_fused_params": fused_params,
        "expected_maccs": var_cfg["expected_maccs"],
        "class_weights": class_weights.cpu().tolist(),
        "epochs": [],
        "lr": [],
        "train_loss": [],
        "train_ce_loss": [],
        "train_kd_loss": [],
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
    ckpt_path = ckpt_dir / f"cnn1d_binary_ds{variant}_kd_best_seed{seed}.pt"

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        tr = run_epoch_kd(
            student, teachers, train_loader, ce_criterion, optimizer, device, alpha, temperature, is_train=True
        )
        vl = run_epoch_kd(
            student, teachers, val_loader, ce_criterion, optimizer, device, alpha, temperature, is_train=False
        )
        scheduler.step()

        history["epochs"].append(epoch)
        history["lr"].append(current_lr)
        history["train_loss"].append(tr["loss"])
        history["train_ce_loss"].append(tr["ce_loss"])
        history["train_kd_loss"].append(tr["kd_loss"])
        history["val_loss"].append(vl["loss"])
        history["train_auroc"].append(tr["auroc"])
        history["val_auroc"].append(vl["auroc"])
        history["val_pr_auc"].append(vl["pr_auc"])
        history["val_recall_non_N"].append(vl["recall_non_N"])
        history["val_specificity_N"].append(vl["specificity_N"])
        history["val_f1_non_N"].append(vl["f1_non_N"])
        history["val_macro_f1"].append(vl["macro_f1"])

        is_best = vl["auroc"] > best_val_auroc
        if is_best:
            best_val_auroc = vl["auroc"]
            best_epoch = epoch
            history["best_epoch"] = best_epoch
            history["best_val_auroc"] = best_val_auroc

            torch.save(
                {
                    "epoch": epoch,
                    "seed": seed,
                    "variant": variant,
                    "kd_alpha": alpha,
                    "kd_temperature": temperature,
                    "model_name": f"CNN1DBinaryDS_{variant}_KD",
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auroc": best_val_auroc,
                    "val_metrics_at_0_5": vl,
                    "class_weights": class_weights.cpu().tolist(),
                    "config": CFG,
                    "variant_config": var_cfg,
                },
                ckpt_path,
            )

        mark = " <-- [BEST VAL AUROC]" if is_best else ""
        print(
            f"Epoch {epoch:2d}/{epochs} (lr={current_lr:.5f}) | "
            f"TrLoss: {tr['loss']:.4f} (CE:{tr['ce_loss']:.3f}/KD:{tr['kd_loss']:.3f}) TrAUC: {tr['auroc']:.4f} | "
            f"VlLoss: {vl['loss']:.4f} VlAUC: {vl['auroc']:.4f} VlPR: {vl['pr_auc']:.4f} "
            f"Rec(1): {vl['recall_non_N']:.4f} Spec(0): {vl['specificity_N']:.4f} "
            f"F1(1): {vl['f1_non_N']:.4f}{mark}"
        )

    elapsed = time.time() - t0
    results_dir = Path(CFG["paths"]["results_base_dir"]) / variant
    results_dir.mkdir(parents=True, exist_ok=True)
    hist_path = results_dir / f"history_seed{seed}.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(
        f"Variant {variant} (KD) | Seed {seed} finished in {elapsed:.1f}s | "
        f"Best Epoch: {best_epoch} (Val AUROC={best_val_auroc:.4f}) | Saved: {ckpt_path}"
    )
    return history


def main():
    parser = argparse.ArgumentParser(description="Train CNN1DBinaryDS with Knowledge Distillation (Step E4)")
    parser.add_argument(
        "--variant",
        type=str,
        default="B",
        choices=["A", "B", "C"],
        help="DS-CNN architectural variant (default: B)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=CFG["training"]["kd"]["alpha"],
        help="Weight for hard CE loss vs KD loss (default: 0.5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=CFG["training"]["kd"]["temperature"],
        help="Distillation temperature T (default: 3.0)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=CFG["training"]["seeds"],
        help="Random seeds to run (default: 42 43 44)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teachers = load_teacher_ensemble(device)

    summaries = []
    for seed in args.seeds:
        hist = train_single_seed_kd(
            args.variant,
            seed,
            alpha=args.alpha,
            temperature=args.temperature,
            teachers=teachers,
            device=device,
        )
        summaries.append(
            {
                "seed": seed,
                "best_epoch": hist["best_epoch"],
                "best_val_auroc": hist["best_val_auroc"],
            }
        )

    print(f"\n{'='*82}")
    print(f"SUMMARY ACROSS SEEDS (CNN1DBinaryDS Variant {args.variant} + KD alpha={args.alpha}, T={args.temperature}):")
    for s in summaries:
        print(f"  Seed {s['seed']}: Best Epoch = {s['best_epoch']:2d} | Best Val AUROC = {s['best_val_auroc']:.4f}")
    mean_auc = float(np.mean([s["best_val_auroc"] for s in summaries]))
    std_auc = float(np.std([s["best_val_auroc"] for s in summaries]))
    print(f"  Mean Val AUROC = {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"{'='*82}")


if __name__ == "__main__":
    main()
