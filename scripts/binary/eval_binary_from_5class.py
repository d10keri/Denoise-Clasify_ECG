"""
A2. Đánh giá đường chuẩn (Baseline "miễn phí") phân loại 2 lớp (N vs non-N)
trực tiếp từ checkpoint 5 lớp hiện có: checkpoints/cnn1d_best.pt.

Phương pháp:
- Tính xác suất P(non-N) = 1 - P(N) = sum_{c in {S,V,F,Q}} softmax(logits)[c].
- Quét ngưỡng quyết định trên tập VAL (không dùng TEST để chọn ngưỡng).
- Đánh giá trên VAL và TEST tại:
  1. Quyết định gốc 5 lớp gộp lại (argmax(logits) > 0)
  2. Ngưỡng mặc định P(non-N) >= 0.50
  3. Ngưỡng tối ưu F1(non-N) trên VAL
  4. Ngưỡng đạt Recall(non-N) >= 0.85 có Specificity cao nhất trên VAL
  5. Ngưỡng cân bằng Youden's J (Recall + Specificity - 1) tối đa trên VAL
- Báo cáo kèm tỷ lệ phát hiện (Recall) riêng cho từng lớp bệnh lý gốc (S, V, F, Q).
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# Đảm bảo import được từ thư mục gốc project
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn1d import CNN1DBaseline
from src.data.ecg_dataset import make_dataloader

CHECKPOINT = PROJECT_ROOT / "checkpoints/cnn1d_best.pt"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_PATH = PROJECT_ROOT / "results/binary/baseline_from_5class.json"
CLASS_NAMES_5 = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}


@torch.no_grad()
def get_predictions(model, split: str, device: torch.device):
    loader = make_dataloader(
        str(PROCESSED_DIR), split, batch_size=256, add_channel_dim=True, shuffle=False
    )
    all_probs_5 = []
    all_y5 = []
    for X, y in loader:
        logits = model(X.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs_5.append(probs)
        all_y5.append(y.numpy())

    probs_5 = np.concatenate(all_probs_5, axis=0)
    y5 = np.concatenate(all_all_y5 := all_y5, axis=0).astype(np.int64)
    y2 = (y5 > 0).astype(np.int64)
    # Tổng xác suất các lớp 1..4 (S, V, F, Q) chính xác về số học hơn 1 - P(N)
    prob_non_n = probs_5[:, 1:].sum(axis=1)
    pred_5_mapped = (probs_5.argmax(axis=1) > 0).astype(np.int64)
    return y5, y2, probs_5, prob_non_n, pred_5_mapped


def evaluate_at_predictions(y5: np.ndarray, y2: np.ndarray, y_pred: np.ndarray, threshold=None) -> dict:
    cm = confusion_matrix(y2, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    total = tn + fp + fn + tp

    acc = (tn + tp) / max(total, 1)
    recall_non_n = tp / max(tp + fn, 1)          # Sensitivity (Recall lớp 1 = non-N)
    specificity_n = tn / max(tn + fp, 1)         # Specificity (Recall lớp 0 = N)
    precision_non_n = tp / max(tp + fp, 1)       # PPV cho non-N
    npv_n = tn / max(tn + fn, 1)                 # NPV cho N
    f1_non_n = (
        2.0 * precision_non_n * recall_non_n / max(precision_non_n + recall_non_n, 1e-12)
    )
    f1_n = 2.0 * npv_n * specificity_n / max(npv_n + specificity_n, 1e-12)
    macro_f1 = 0.5 * (f1_n + f1_non_n)

    # Recall theo từng lớp gốc S, V, F, Q (bao nhiêu % nhịp S/V/F/Q được báo là non-N)
    per_subclass_recall = {}
    for c in range(1, 5):
        mask = y5 == c
        n_c = int(np.sum(mask))
        detected_c = int(np.sum(y_pred[mask] == 1)) if n_c > 0 else 0
        per_subclass_recall[CLASS_NAMES_5[c]] = {
            "support": n_c,
            "detected_as_non_N": detected_c,
            "recall": round(detected_c / n_c, 6) if n_c > 0 else None,
        }

    res = {
        "threshold": round(float(threshold), 4) if threshold is not None else "argmax_5class > 0",
        "accuracy": round(acc, 6),
        "recall_non_N_sensitivity": round(recall_non_n, 6),
        "specificity_N": round(specificity_n, 6),
        "precision_non_N": round(precision_non_n, 6),
        "f1_non_N": round(f1_non_n, 6),
        "f1_N": round(f1_n, 6),
        "macro_f1_binary": round(macro_f1, 6),
        "youden_j": round(recall_non_n + specificity_n - 1.0, 6),
        "passes_gate_85_85": bool(recall_non_n >= 0.85 and specificity_n >= 0.85),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "subclass_detection_recall": per_subclass_recall,
    }
    return res


def find_optimal_thresholds_on_val(y5_val: np.ndarray, y2_val: np.ndarray, prob_val: np.ndarray) -> dict:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1_th = 0.5
    best_f1_val = -1.0

    best_j_th = 0.5
    best_j_val = -2.0

    best_gate85_th = None
    best_gate85_spec = -1.0

    for th in thresholds:
        pred = (prob_val >= th).astype(np.int64)
        cm = confusion_matrix(y2_val, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rec = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2.0 * prec * rec / max(prec + rec, 1e-12)
        j = rec + spec - 1.0

        if f1 > best_f1_val:
            best_f1_val = f1
            best_f1_th = float(th)

        if j > best_j_val:
            best_j_val = j
            best_j_th = float(th)

        if rec >= 0.85 and spec > best_gate85_spec:
            best_gate85_spec = spec
            best_gate85_th = float(th)

    return {
        "best_val_f1_threshold": round(best_f1_th, 4),
        "best_val_youden_j_threshold": round(best_j_th, 4),
        "val_recall_ge_85_threshold": round(best_gate85_th, 4) if best_gate85_th is not None else 0.5,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    model_cfg = ckpt.get("model_config", {"num_classes": 5})

    model = CNN1DBaseline(num_classes=model_cfg["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    y5_val, y2_val, _, prob_val, pred5_val = get_predictions(model, "val", device)
    y5_test, y2_test, _, prob_test, pred5_test = get_predictions(model, "test", device)

    val_auroc = float(roc_auc_score(y2_val, prob_val))
    val_prauc = float(average_precision_score(y2_val, prob_val))
    test_auroc = float(roc_auc_score(y2_test, prob_test))
    test_prauc = float(average_precision_score(y2_test, prob_test))

    opt_ths = find_optimal_thresholds_on_val(y5_val, y2_val, prob_val)
    th_f1 = opt_ths["best_val_f1_threshold"]
    th_j = opt_ths["best_val_youden_j_threshold"]
    th_r85 = opt_ths["val_recall_ge_85_threshold"]

    modes = {
        "1_argmax_5class_mapped": (pred5_val, pred5_test, None),
        "2_default_threshold_0.50": (
            (prob_val >= 0.50).astype(np.int64),
            (prob_test >= 0.50).astype(np.int64),
            0.50,
        ),
        "3_val_best_f1_threshold": (
            (prob_val >= th_f1).astype(np.int64),
            (prob_test >= th_f1).astype(np.int64),
            th_f1,
        ),
        "4_val_recall_ge_0.85_threshold": (
            (prob_val >= th_r85).astype(np.int64),
            (prob_test >= th_r85).astype(np.int64),
            th_r85,
        ),
        "5_val_best_youden_j_threshold": (
            (prob_val >= th_j).astype(np.int64),
            (prob_test >= th_j).astype(np.int64),
            th_j,
        ),
    }

    val_evals = {}
    test_evals = {}
    for name, (p_val, p_test, th) in modes.items():
        val_evals[name] = evaluate_at_predictions(y5_val, y2_val, p_val, th)
        test_evals[name] = evaluate_at_predictions(y5_test, y2_test, p_test, th)

    output = {
        "task": "A2 - Zero-Retrain Binary Baseline from 5-Class CNN1DBaseline Checkpoint",
        "source_checkpoint": "checkpoints/cnn1d_best.pt",
        "source_best_epoch": int(ckpt["epoch"]),
        "threshold_free_metrics": {
            "val": {"auroc": round(val_auroc, 6), "pr_auc": round(val_prauc, 6)},
            "test": {"auroc": round(test_auroc, 6), "pr_auc": round(test_prauc, 6)},
        },
        "thresholds_selected_on_val": opt_ths,
        "val_operating_points": val_evals,
        "test_operating_points": test_evals,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=== KẾT QUẢ BASELINE 2 LỚP TỪ CHECKPOINT 5 LỚP (A2) ===")
    print(f"Checkpoint: checkpoints/cnn1d_best.pt (Epoch {ckpt['epoch']})")
    print(f"Threshold-free | VAL  -> AUROC: {val_auroc:.4f} | PR-AUC: {val_prauc:.4f}")
    print(f"Threshold-free | TEST -> AUROC: {test_auroc:.4f} | PR-AUC: {test_prauc:.4f}\n")

    print("--- Kết quả trên tập VAL ---")
    for name, m in val_evals.items():
        sub = m["subclass_detection_recall"]
        print(
            f"  {name:32s} (th={str(m['threshold']):>17s}) | "
            f"Recall(non-N): {m['recall_non_N_sensitivity']:.4f} | "
            f"Spec(N): {m['specificity_N']:.4f} | "
            f"F1(non-N): {m['f1_non_N']:.4f} | "
            f"Gate85: {str(m['passes_gate_85_85']):5s} | "
            f"S:{sub['S']['recall']:.2f} V:{sub['V']['recall']:.2f} F:{sub['F']['recall']:.2f}"
        )

    print("\n--- Kết quả trên tập TEST (Áp ngưỡng cố định từ VAL) ---")
    for name, m in test_evals.items():
        sub = m["subclass_detection_recall"]
        print(
            f"  {name:32s} (th={str(m['threshold']):>17s}) | "
            f"Recall(non-N): {m['recall_non_N_sensitivity']:.4f} | "
            f"Spec(N): {m['specificity_N']:.4f} | "
            f"F1(non-N): {m['f1_non_N']:.4f} | "
            f"Gate85: {str(m['passes_gate_85_85']):5s} | "
            f"S:{sub['S']['recall']:.2f} V:{sub['V']['recall']:.2f} F:{sub['F']['recall']:.2f} Q:{sub['Q']['recall']:.2f}"
        )
    print(f"\nĐã lưu kết quả chi tiết tại: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
