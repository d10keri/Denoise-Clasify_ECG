"""
B3. Đánh giá mô hình 1D-CNN 2 lớp (CNN1DBinary) trên 3 seeds (42, 43, 44).
Quy tắc nghiêm ngặt:
- Mọi ngưỡng quyết định được chọn HOÀN TOÀN trên tập VAL của từng seed.
- Áp ngưỡng cố định từ VAL sang tập TEST để báo cáo trung bình ± độ lệch chuẩn (mean ± std).
- Kiểm tra Gate B: Recall(non-N) >= 0.85 VÀ Specificity(N) >= 0.85.
- Lưu kết quả vào:
  + results/binary/m1_fp32/test_metrics.json
  + results/binary/m1_fp32/confusion_matrix.png
"""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)

from src.models.cnn1d_binary import CNN1DBinary
from src.data.binary_dataset import make_binary_dataloader, ECGBinaryDataset

SEEDS = [42, 43, 44]
PROCESSED_DIR = "data/processed"
CKPT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results/binary/m1_fp32")
CLASS_NAMES_5 = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}


@torch.no_grad()
def predict_split(model, split: str, device: torch.device):
    loader = make_binary_dataloader(
        PROCESSED_DIR, split, batch_size=256, add_channel_dim=True, shuffle=False
    )
    ds: ECGBinaryDataset = loader.dataset
    probs_list = []
    for X, _ in loader:
        logits = model(X.to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs_list.append(probs)
    probs_pos = np.concatenate(probs_list, axis=0)
    return ds.y5.copy(), ds.y_binary.copy(), probs_pos


def select_thresholds_on_val(y2_val: np.ndarray, prob_val: np.ndarray) -> dict:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1_th, best_f1 = 0.5, -1.0
    best_j_th, best_j = 0.5, -2.0
    best_r85_th, best_r85_spec = None, -1.0
    best_eer_th, min_diff = 0.5, 10.0

    for th in thresholds:
        pred = (prob_val >= th).astype(np.int64)
        cm = confusion_matrix(y2_val, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rec = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2.0 * prec * rec / max(prec + rec, 1e-12)
        j = rec + spec - 1.0
        diff = abs(rec - spec)

        if f1 > best_f1:
            best_f1 = f1
            best_f1_th = float(th)
        if j > best_j:
            best_j = j
            best_j_th = float(th)
        if rec >= 0.85 and spec > best_r85_spec:
            best_r85_spec = spec
            best_r85_th = float(th)
        if diff < min_diff:
            min_diff = diff
            best_eer_th = float(th)

    return {
        "1_default_0.50": 0.50,
        "2_val_best_f1": round(best_f1_th, 4),
        "3_val_best_youden_j": round(best_j_th, 4),
        "4_val_recall_ge_0.85": round(best_r85_th, 4) if best_r85_th is not None else 0.50,
        "5_val_equal_rec_spec": round(best_eer_th, 4),
    }


def eval_at_threshold(y5: np.ndarray, y2: np.ndarray, prob: np.ndarray, th: float) -> dict:
    pred = (prob >= th).astype(np.int64)
    cm = confusion_matrix(y2, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    total = tn + fp + fn + tp

    acc = (tn + tp) / max(total, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1_pos = 2.0 * prec * rec / max(prec + rec, 1e-12)
    f1_neg = 2.0 * npv * spec / max(npv + spec, 1e-12)
    macro_f1 = 0.5 * (f1_pos + f1_neg)

    sub_recalls = {}
    for c in range(1, 5):
        mask = y5 == c
        n_c = int(np.sum(mask))
        det_c = int(np.sum(pred[mask] == 1)) if n_c > 0 else 0
        sub_recalls[CLASS_NAMES_5[c]] = round(det_c / n_c, 6) if n_c > 0 else 0.0

    return {
        "threshold": round(float(th), 4),
        "accuracy": round(acc, 6),
        "recall_non_N": round(rec, 6),
        "specificity_N": round(spec, 6),
        "precision_non_N": round(prec, 6),
        "f1_non_N": round(f1_pos, 6),
        "macro_f1": round(macro_f1, 6),
        "youden_j": round(rec + spec - 1.0, 6),
        "passes_gate_85_85": bool(rec >= 0.85 and spec >= 0.85),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "subclass_recall": sub_recalls,
    }


def summarize_across_seeds(per_seed_dict: dict, op_names: list) -> dict:
    metric_keys = [
        "threshold",
        "accuracy",
        "recall_non_N",
        "specificity_N",
        "precision_non_N",
        "f1_non_N",
        "macro_f1",
        "youden_j",
    ]
    summary = {}
    for op in op_names:
        op_sum = {}
        for k in metric_keys:
            vals = [per_seed_dict[s][op][k] for s in SEEDS]
            op_sum[k] = {
                "mean": round(float(np.mean(vals)), 6),
                "std": round(float(np.std(vals)), 6),
            }
        sub_sum = {}
        for c_name in ["S", "V", "F", "Q"]:
            vals = [per_seed_dict[s][op]["subclass_recall"][c_name] for s in SEEDS]
            sub_sum[c_name] = {
                "mean": round(float(np.mean(vals)), 6),
                "std": round(float(np.std(vals)), 6),
            }
        op_sum["subclass_recall"] = sub_sum
        op_sum["passes_gate_85_85_count"] = int(
            sum(1 for s in SEEDS if per_seed_dict[s][op]["passes_gate_85_85"])
        )
        summary[op] = op_sum
    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_seed_val = {}
    per_seed_test = {}
    per_seed_meta = {}
    val_probs_all = []
    test_probs_all = []

    y5_val_ref = y2_val_ref = y5_test_ref = y2_test_ref = None

    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"cnn1d_binary_best_seed{seed}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")

        model = CNN1DBinary(num_classes=2)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        y5_val, y2_val, prob_val = predict_split(model, "val", device)
        y5_test, y2_test, prob_test = predict_split(model, "test", device)

        y5_val_ref, y2_val_ref = y5_val, y2_val
        y5_test_ref, y2_test_ref = y5_test, y2_test
        val_probs_all.append(prob_val)
        test_probs_all.append(prob_test)

        val_auroc = float(roc_auc_score(y2_val, prob_val))
        val_prauc = float(average_precision_score(y2_val, prob_val))
        test_auroc = float(roc_auc_score(y2_test, prob_test))
        test_prauc = float(average_precision_score(y2_test, prob_test))

        ths = select_thresholds_on_val(y2_val, prob_val)

        per_seed_meta[seed] = {
            "best_epoch": int(ckpt["epoch"]),
            "val_auroc": round(val_auroc, 6),
            "val_pr_auc": round(val_prauc, 6),
            "test_auroc": round(test_auroc, 6),
            "test_pr_auc": round(test_prauc, 6),
            "selected_thresholds_on_val": ths,
        }

        per_seed_val[seed] = {
            op: eval_at_threshold(y5_val, y2_val, prob_val, th)
            for op, th in ths.items()
        }
        per_seed_test[seed] = {
            op: eval_at_threshold(y5_test, y2_test, prob_test, th)
            for op, th in ths.items()
        }

    op_names = list(per_seed_val[SEEDS[0]].keys())
    val_summary = summarize_across_seeds(per_seed_val, op_names)
    test_summary = summarize_across_seeds(per_seed_test, op_names)

    # Thống kê AUROC và PR-AUC trung bình ± std qua 3 seeds
    threshold_free_summary = {
        "val_auroc": {
            "mean": round(float(np.mean([per_seed_meta[s]["val_auroc"] for s in SEEDS])), 6),
            "std": round(float(np.std([per_seed_meta[s]["val_auroc"] for s in SEEDS])), 6),
        },
        "val_pr_auc": {
            "mean": round(float(np.mean([per_seed_meta[s]["val_pr_auc"] for s in SEEDS])), 6),
            "std": round(float(np.std([per_seed_meta[s]["val_pr_auc"] for s in SEEDS])), 6),
        },
        "test_auroc": {
            "mean": round(float(np.mean([per_seed_meta[s]["test_auroc"] for s in SEEDS])), 6),
            "std": round(float(np.std([per_seed_meta[s]["test_auroc"] for s in SEEDS])), 6),
        },
        "test_pr_auc": {
            "mean": round(float(np.mean([per_seed_meta[s]["test_pr_auc"] for s in SEEDS])), 6),
            "std": round(float(np.std([per_seed_meta[s]["test_pr_auc"] for s in SEEDS])), 6),
        },
    }

    # Vẽ hình Confusion Matrix (seed 42 tại ngưỡng Youden's J) + ROC curve 3 seeds trên Test
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Normalized Confusion Matrix (Seed 42 - Best Youden's J)
    th_j_42 = per_seed_meta[42]["selected_thresholds_on_val"]["3_val_best_youden_j"]
    pred_test_42 = (test_probs_all[0] >= th_j_42).astype(np.int64)
    cm_norm = confusion_matrix(y2_test_ref, pred_test_42, labels=[0, 1], normalize="true")
    cax = axes[0].matshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(cax, ax=axes[0])
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["N (0)", "non-N (1)"])
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["N (0)", "non-N (1)"])
    for (i, j), z in np.ndenumerate(cm_norm):
        axes[0].text(
            j, i, f"{z:.3f}", ha="center", va="center",
            color="white" if z > 0.5 else "black", fontsize=12, fontweight="bold"
        )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"Test Confusion Matrix (Seed 42, th={th_j_42:.2f})")

    # Subplot 2: ROC Curves trên tập Test của 3 seeds
    for idx, seed in enumerate(SEEDS):
        fpr, tpr, _ = roc_curve(y2_test_ref, test_probs_all[idx])
        auc_s = per_seed_meta[seed]["test_auroc"]
        axes[1].plot(fpr, tpr, label=f"Seed {seed} (AUROC = {auc_s:.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    # Đánh dấu vùng Gate B (Recall >= 0.85, Specificity >= 0.85 -> FPR <= 0.15)
    axes[1].axvline(0.15, color="red", linestyle=":", alpha=0.6, label="Gate B (Spec>=0.85, Rec>=0.85)")
    axes[1].axhline(0.85, color="red", linestyle=":", alpha=0.6)
    axes[1].set_xlabel("1 - Specificity (FPR)")
    axes[1].set_ylabel("Recall non-N (TPR)")
    axes[1].set_title("Test ROC Curves (3 Seeds) vs Gate B Target")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    output = {
        "task": "B3 - Evaluation of 2-Class CNN1DBinary (M1 FP32) across 3 Seeds",
        "seeds": SEEDS,
        "threshold_free_summary": threshold_free_summary,
        "val_summary_mean_std": val_summary,
        "test_summary_mean_std": test_summary,
        "per_seed_metadata": per_seed_meta,
        "per_seed_val": per_seed_val,
        "per_seed_test": per_seed_test,
    }

    out_json = RESULTS_DIR / "test_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n=== TỔNG HỢP KẾT QUẢ M1 FP32 (CNN1DBinary - 3 SEEDS: 42, 43, 44) ===")
    print(
        f"VAL  AUROC: {threshold_free_summary['val_auroc']['mean']:.4f} ± {threshold_free_summary['val_auroc']['std']:.4f} | "
        f"PR-AUC: {threshold_free_summary['val_pr_auc']['mean']:.4f} ± {threshold_free_summary['val_pr_auc']['std']:.4f}"
    )
    print(
        f"TEST AUROC: {threshold_free_summary['test_auroc']['mean']:.4f} ± {threshold_free_summary['test_auroc']['std']:.4f} | "
        f"PR-AUC: {threshold_free_summary['test_pr_auc']['mean']:.4f} ± {threshold_free_summary['test_pr_auc']['std']:.4f}"
    )

    print("\n--- KẾT QUẢ TRÊN TẬP VAL (Mean ± Std qua 3 seeds) ---")
    for op, m in val_summary.items():
        sub = m["subclass_recall"]
        print(
            f"  {op:24s} (th={m['threshold']['mean']:.2f}±{m['threshold']['std']:.2f}) | "
            f"Rec(non-N): {m['recall_non_N']['mean']:.4f}±{m['recall_non_N']['std']:.4f} | "
            f"Spec(N): {m['specificity_N']['mean']:.4f}±{m['specificity_N']['std']:.4f} | "
            f"F1(1): {m['f1_non_N']['mean']:.4f}±{m['f1_non_N']['std']:.4f} | "
            f"Gate85: {m['passes_gate_85_85_count']}/3"
        )

    print("\n--- KẾT QUẢ TRÊN TẬP TEST (Áp ngưỡng cố định từ VAL - Mean ± Std qua 3 seeds) ---")
    for op, m in test_summary.items():
        sub = m["subclass_recall"]
        print(
            f"  {op:24s} (th={m['threshold']['mean']:.2f}±{m['threshold']['std']:.2f}) | "
            f"Rec(non-N): {m['recall_non_N']['mean']:.4f}±{m['recall_non_N']['std']:.4f} | "
            f"Spec(N): {m['specificity_N']['mean']:.4f}±{m['specificity_N']['std']:.4f} | "
            f"F1(1): {m['f1_non_N']['mean']:.4f}±{m['f1_non_N']['std']:.4f} | "
            f"Gate85: {m['passes_gate_85_85_count']}/3 | "
            f"S:{sub['S']['mean']:.2f} V:{sub['V']['mean']:.2f} F:{sub['F']['mean']:.2f} Q:{sub['Q']['mean']:.2f}"
        )
    print(f"\nĐã lưu báo cáo tại: {out_json}")
    print(f"Đã lưu biểu đồ tại: {RESULTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
