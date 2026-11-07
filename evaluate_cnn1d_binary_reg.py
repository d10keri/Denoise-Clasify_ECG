"""
Đánh giá mô hình chống Overfitting CNN1DBinaryRegularized (M1-Reg FP32)
trên 3 seeds (42, 43, 44) và so sánh trực tiếp với CNN1DBinary gốc (backup M1 FP32).

Lưu kết quả vào:
- results/binary/m1_reg_fp32/test_metrics.json
- results/binary/m1_reg_fp32/confusion_matrix.png
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

from src.models.cnn1d_binary_reg import CNN1DBinaryRegularized
from src.data.binary_dataset import make_binary_dataloader, ECGBinaryDataset
from evaluate_cnn1d_binary import (
    predict_split,
    select_thresholds_on_val,
    eval_at_threshold,
    summarize_across_seeds,
)

SEEDS = [42, 43, 44]
CKPT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results/binary/m1_reg_fp32")
M1_BASE_JSON = Path("results/binary/m1_fp32/test_metrics.json")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_seed_val = {}
    per_seed_test = {}
    per_seed_meta = {}
    test_probs_all = []
    y2_test_ref = None

    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"cnn1d_binary_reg_best_seed{seed}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        m_cfg = ckpt.get("model_config", {})

        model = CNN1DBinaryRegularized(
            num_classes=m_cfg.get("num_classes", 2),
            dropout_conv=m_cfg.get("dropout_conv", 0.15),
            dropout_fc=m_cfg.get("dropout_fc", 0.30),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        y5_val, y2_val, prob_val = predict_split(model, "val", device)
        y5_test, y2_test, prob_test = predict_split(model, "test", device)

        y2_test_ref = y2_test
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

    # Vẽ Confusion Matrix & ROC Curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
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
    axes[0].set_title(f"M1-Reg Test CM (Seed 42, th={th_j_42:.2f})")

    for idx, seed in enumerate(SEEDS):
        fpr, tpr, _ = roc_curve(y2_test_ref, test_probs_all[idx])
        auc_s = per_seed_meta[seed]["test_auroc"]
        axes[1].plot(fpr, tpr, label=f"Seed {seed} (AUROC = {auc_s:.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].axvline(0.15, color="red", linestyle=":", alpha=0.6, label="Gate B (Spec>=0.85, Rec>=0.85)")
    axes[1].axhline(0.85, color="red", linestyle=":", alpha=0.6)
    axes[1].set_xlabel("1 - Specificity (FPR)")
    axes[1].set_ylabel("Recall non-N (TPR)")
    axes[1].set_title("M1-Reg Test ROC Curves (3 Seeds)")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    output = {
        "task": "Evaluation of Regularized 2-Class CNN1DBinaryRegularized (M1-Reg FP32) across 3 Seeds",
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

    print("\n=== TỔNG HỢP KẾT QUẢ M1-REG FP32 (CNN1DBinaryRegularized - 3 SEEDS) ===")
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


if __name__ == "__main__":
    main()
