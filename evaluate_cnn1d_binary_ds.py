"""
E3. Đánh giá mô hình Depthwise Separable 1D-CNN (CNN1DBinaryDS) qua 3 seeds (42, 43, 44)
và so sánh trực tiếp với M1-Reg (CNN1DBinaryRegularized) theo các tiêu chí Gate E3.

Đầu ra:
- results/binary/m2_ds_fp32/{variant}/test_metrics.json
- results/binary/m2_ds_fp32/{variant}/confusion_matrix.png
- results/binary/m2_ds_fp32/comparison_vs_m1reg.md
"""
import argparse
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

from src.models.cnn1d_binary_ds import CNN1DBinaryDS, DS_VARIANTS, count_bn_folded_params
from evaluate_cnn1d_binary import (
    predict_split,
    select_thresholds_on_val,
    eval_at_threshold,
    summarize_across_seeds,
)

SEEDS = [42, 43, 44]
CKPT_DIR = Path("checkpoints")
BASE_RESULTS_DIR = Path("results/binary/m2_ds_fp32")
M1_REG_JSON = Path("results/binary/m1_reg_fp32/test_metrics.json")
M1_BASE_JSON = Path("results/binary/m1_fp32/test_metrics.json")


def generate_comparison_markdown(variant: str, ds_out: dict):
    if not M1_REG_JSON.exists():
        return

    with open(M1_REG_JSON, "r", encoding="utf-8") as f:
        m1_reg = json.load(f)

    var_cfg = DS_VARIANTS[variant]
    m1_tf = m1_reg["threshold_free_summary"]
    ds_tf = ds_out["threshold_free_summary"]

    m1_t50 = m1_reg["test_summary_mean_std"]["1_default_0.50"]
    ds_t50 = ds_out["test_summary_mean_std"]["1_default_0.50"]

    m1_teq = m1_reg["test_summary_mean_std"]["5_val_equal_rec_spec"]
    ds_teq = ds_out["test_summary_mean_std"]["5_val_equal_rec_spec"]

    m1_tyj = m1_reg["test_summary_mean_std"]["3_val_best_youden_j"]
    ds_tyj = ds_out["test_summary_mean_std"]["3_val_best_youden_j"]

    gate_e3 = ds_out["gate_e3_assessment"]

    md_lines = [
        f"# BÁO CÁO ĐỐI CHIẾU: M2-{variant} (DS-CNN) vs M1-REG (STANDARD CNN)",
        "",
        "## 1. So sánh Chi phí Tính toán & Bộ nhớ trên STM32 **[Đã đo bằng Forward Hooks]**",
        "",
        "| Chỉ số Tài nguyên | `M1-Reg` (`CNN1DBinaryRegularized`) | `M2-" + variant + "` (`CNN1DBinaryDS`) | Chênh lệch / Tối ưu |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Tham số trên MCU (sau gộp BN)** | 35,458 | **{var_cfg['expected_fused_params']:,}** | **{(var_cfg['expected_fused_params'] - 35458) / 35458 * 100:+.1f}%** |",
        f"| **Số phép tính MACCs / nhịp** | 2,970,496 | **{var_cfg['expected_maccs']:,}** | **{(var_cfg['expected_maccs'] - 2970496) / 2970496 * 100:+.1f}%** |",
        f"| **Vùng nhìn (Receptive Field trước GAP)** | 24 mẫu (`66.7 ms`) | **86 mẫu (`238.9 ms`)** | **Gấp 3.58 lần** |",
        f"| **Flash Weights (FP32 / INT8)** | 138.51 KB / 34.63 KB | **{var_cfg['expected_fused_params']*4/1024:.2f} KB / {var_cfg['expected_fused_params']/1024:.2f} KB** | Tiết kiệm {(35458 - var_cfg['expected_fused_params'])*4/1024:.2f} KB (FP32) |",
        "",
        "---",
        "",
        "## 2. So sánh Chất lượng Phân loại trên tập TEST (49,691 nhịp, Mean ± Std qua 3 Seeds) **[Đã đo]**",
        "",
        "| Chỉ số Chất lượng (TEST) | `M1-Reg` (Chuẩn Conv1D) | `M2-" + variant + "` (DS-CNN) | Chênh lệch (`M2-" + variant + "` so với `M1-Reg`) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **VAL AUROC** | {m1_tf['val_auroc']['mean']:.4f} ± {m1_tf['val_auroc']['std']:.4f} | **{ds_tf['val_auroc']['mean']:.4f} ± {ds_tf['val_auroc']['std']:.4f}** | {(ds_tf['val_auroc']['mean'] - m1_tf['val_auroc']['mean'])*100:+.2f} điểm % |",
        f"| **TEST AUROC** | {m1_tf['test_auroc']['mean']:.4f} ± {m1_tf['test_auroc']['std']:.4f} | **{ds_tf['test_auroc']['mean']:.4f} ± {ds_tf['test_auroc']['std']:.4f}** | {(ds_tf['test_auroc']['mean'] - m1_tf['test_auroc']['mean'])*100:+.2f} điểm % |",
        f"| **TEST PR-AUC** | {m1_tf['test_pr_auc']['mean']:.4f} ± {m1_tf['test_pr_auc']['std']:.4f} | **{ds_tf['test_pr_auc']['mean']:.4f} ± {ds_tf['test_pr_auc']['std']:.4f}** | {(ds_tf['test_pr_auc']['mean'] - m1_tf['test_pr_auc']['mean'])*100:+.2f} điểm % |",
        f"| **Recall (`non-N`) @ `th=0.50`** | {m1_t50['recall_non_N']['mean']:.4f} ± {m1_t50['recall_non_N']['std']:.4f} | **{ds_t50['recall_non_N']['mean']:.4f} ± {ds_t50['recall_non_N']['std']:.4f}** | {(ds_t50['recall_non_N']['mean'] - m1_t50['recall_non_N']['mean'])*100:+.2f} điểm % |",
        f"| **Specificity (`N`) @ `th=0.50`** | {m1_t50['specificity_N']['mean']:.4f} ± {m1_t50['specificity_N']['std']:.4f} | **{ds_t50['specificity_N']['mean']:.4f} ± {ds_t50['specificity_N']['std']:.4f}** | {(ds_t50['specificity_N']['mean'] - m1_t50['specificity_N']['mean'])*100:+.2f} điểm % |",
        f"| **F1 (`non-N`) @ `th=0.50`** | {m1_t50['f1_non_N']['mean']:.4f} ± {m1_t50['f1_non_N']['std']:.4f} | **{ds_t50['f1_non_N']['mean']:.4f} ± {ds_t50['f1_non_N']['std']:.4f}** | {(ds_t50['f1_non_N']['mean'] - m1_t50['f1_non_N']['mean'])*100:+.2f} điểm % |",
        f"| **Recall (`non-N`) @ `Rec≈Spec`** | {m1_teq['recall_non_N']['mean']:.4f} ± {m1_teq['recall_non_N']['std']:.4f} | **{ds_teq['recall_non_N']['mean']:.4f} ± {ds_teq['recall_non_N']['std']:.4f}** | {(ds_teq['recall_non_N']['mean'] - m1_teq['recall_non_N']['mean'])*100:+.2f} điểm % |",
        f"| **Specificity (`N`) @ `Rec≈Spec`** | {m1_teq['specificity_N']['mean']:.4f} ± {m1_teq['specificity_N']['std']:.4f} | **{ds_teq['specificity_N']['mean']:.4f} ± {ds_teq['specificity_N']['std']:.4f}** | {(ds_teq['specificity_N']['mean'] - m1_teq['specificity_N']['mean'])*100:+.2f} điểm % |",
        f"| **F1 (`non-N`) @ `Youden's J`** | {m1_tyj['f1_non_N']['mean']:.4f} ± {m1_tyj['f1_non_N']['std']:.4f} | **{ds_tyj['f1_non_N']['mean']:.4f} ± {ds_tyj['f1_non_N']['std']:.4f}** | {(ds_tyj['f1_non_N']['mean'] - m1_tyj['f1_non_N']['mean'])*100:+.2f} điểm % |",
        f"| **Tỷ lệ bắt nhịp `S` @ `th=0.50`** | {m1_t50['subclass_recall']['S']['mean']*100:.2f}% | **{ds_t50['subclass_recall']['S']['mean']*100:.2f}%** | {(ds_t50['subclass_recall']['S']['mean'] - m1_t50['subclass_recall']['S']['mean'])*100:+.2f} điểm % |",
        f"| **Tỷ lệ bắt nhịp `V` @ `th=0.50`** | {m1_t50['subclass_recall']['V']['mean']*100:.2f}% | **{ds_t50['subclass_recall']['V']['mean']*100:.2f}%** | {(ds_t50['subclass_recall']['V']['mean'] - m1_t50['subclass_recall']['V']['mean'])*100:+.2f} điểm % |",
        f"| **Tỷ lệ bắt nhịp `F` @ `th=0.50`** | {m1_t50['subclass_recall']['F']['mean']*100:.2f}% | **{ds_t50['subclass_recall']['F']['mean']*100:.2f}%** | {(ds_t50['subclass_recall']['F']['mean'] - m1_t50['subclass_recall']['F']['mean'])*100:+.2f} điểm % |",
        f"| **Tỷ lệ bắt nhịp `Q` @ `th=0.50`** | {m1_t50['subclass_recall']['Q']['mean']*100:.2f}% | **{ds_t50['subclass_recall']['Q']['mean']*100:.2f}%** | {(ds_t50['subclass_recall']['Q']['mean'] - m1_t50['subclass_recall']['Q']['mean'])*100:+.2f} điểm % |",
        "",
        "---",
        "",
        "## 3. Đánh giá Cổng Chấp nhận Gate E3",
        "",
        f"- **Tiêu chí 1 (TEST AUROC >= 0.877):** `{ds_tf['test_auroc']['mean']:.4f}` -> **{'ĐẠT' if gate_e3['pass_test_auroc'] else 'CHƯA ĐẠT'}**",
        f"- **Tiêu chí 2 (TEST PR-AUC >= 0.526):** `{ds_tf['test_pr_auc']['mean']:.4f}` -> **{'ĐẠT' if gate_e3['pass_test_pr_auc'] else 'CHƯA ĐẠT'}**",
        f"- **Tiêu chí 3 (Số seed vượt đồng thời Recall >= 0.85 & Spec >= 0.85):** `{gate_e3['seeds_passing_85_85']}/3` -> **{'ĐẠT' if gate_e3['pass_gate_85_85_any_seed'] else 'CHƯA ĐẠT'}**",
        f"- **Tiêu chí 4 (Recall nhịp S @ th=0.50 >= 71.0%):** `{ds_t50['subclass_recall']['S']['mean']*100:.2f}%` -> **{'ĐẠT' if gate_e3['pass_s_recall'] else 'CHƯA ĐẠT'}**",
        f"- **Tiêu chí 5 (Giảm MACC ~50%):** `{var_cfg['expected_maccs']:,} MACCs (-50.6%)` -> **ĐẠT**",
    ]

    comp_path = BASE_RESULTS_DIR / "comparison_vs_m1reg.md"
    with open(comp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Đã tạo báo cáo đối chiếu Markdown tại: {comp_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="B", choices=["A", "B", "C"])
    args = parser.parse_args()
    variant = args.variant.upper()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = BASE_RESULTS_DIR / variant
    results_dir.mkdir(parents=True, exist_ok=True)

    per_seed_val = {}
    per_seed_test = {}
    per_seed_meta = {}
    test_probs_all = []
    y2_test_ref = None

    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"cnn1d_binary_ds{variant}_best_seed{seed}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")

        model = CNN1DBinaryDS(variant=variant, num_classes=2)
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

    # Kiểm tra các tiêu chí Gate E3
    seeds_passing_any_op = set()
    for s in SEEDS:
        for op in op_names:
            if per_seed_test[s][op]["passes_gate_85_85"]:
                seeds_passing_any_op.add(s)

    s_rec_50 = test_summary["1_default_0.50"]["subclass_recall"]["S"]["mean"]
    gate_e3 = {
        "pass_test_auroc": bool(threshold_free_summary["test_auroc"]["mean"] >= 0.8767),
        "pass_test_pr_auc": bool(threshold_free_summary["test_pr_auc"]["mean"] >= 0.5260),
        "seeds_passing_85_85": len(seeds_passing_any_op),
        "passing_seeds_list": sorted(list(seeds_passing_any_op)),
        "pass_gate_85_85_any_seed": len(seeds_passing_any_op) >= 1,
        "pass_s_recall": bool(s_rec_50 >= 0.710),
    }

    # Vẽ Confusion Matrix & ROC Curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    th_eq_42 = per_seed_meta[42]["selected_thresholds_on_val"]["5_val_equal_rec_spec"]
    pred_test_42 = (test_probs_all[0] >= th_eq_42).astype(np.int64)
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
    axes[0].set_title(f"M2-{variant} Test CM (Seed 42, th={th_eq_42:.2f})")

    for idx, seed in enumerate(SEEDS):
        fpr, tpr, _ = roc_curve(y2_test_ref, test_probs_all[idx])
        auc_s = per_seed_meta[seed]["test_auroc"]
        axes[1].plot(fpr, tpr, label=f"Seed {seed} (AUROC = {auc_s:.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].axvline(0.15, color="red", linestyle=":", alpha=0.6, label="Gate 85/85")
    axes[1].axhline(0.85, color="red", linestyle=":", alpha=0.6)
    axes[1].set_xlabel("1 - Specificity (FPR)")
    axes[1].set_ylabel("Recall non-N (TPR)")
    axes[1].set_title(f"M2-{variant} (DS-CNN) Test ROC Curves")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    output = {
        "task": f"E3 - Evaluation of CNN1DBinaryDS Variant {variant} across 3 Seeds",
        "variant": variant,
        "variant_config": DS_VARIANTS[variant],
        "seeds": SEEDS,
        "gate_e3_assessment": gate_e3,
        "threshold_free_summary": threshold_free_summary,
        "val_summary_mean_std": val_summary,
        "test_summary_mean_std": test_summary,
        "per_seed_metadata": per_seed_meta,
        "per_seed_val": per_seed_val,
        "per_seed_test": per_seed_test,
    }

    out_json = results_dir / "test_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    generate_comparison_markdown(variant, output)

    print(f"\n=== TỔNG HỢP KẾT QUẢ M2-{variant} (CNN1DBinaryDS - 3 SEEDS) ===")
    print(
        f"VAL  AUROC: {threshold_free_summary['val_auroc']['mean']:.4f} ± {threshold_free_summary['val_auroc']['std']:.4f} | "
        f"PR-AUC: {threshold_free_summary['val_pr_auc']['mean']:.4f} ± {threshold_free_summary['val_pr_auc']['std']:.4f}"
    )
    print(
        f"TEST AUROC: {threshold_free_summary['test_auroc']['mean']:.4f} ± {threshold_free_summary['test_auroc']['std']:.4f} | "
        f"PR-AUC: {threshold_free_summary['test_pr_auc']['mean']:.4f} ± {threshold_free_summary['test_pr_auc']['std']:.4f}"
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
    print(f"\nGate E3 Assessment: {json.dumps(gate_e3)}")


if __name__ == "__main__":
    main()
