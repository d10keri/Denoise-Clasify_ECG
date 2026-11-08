"""
E4. Đánh giá mô hình CNN1DBinaryDS sau khi huấn luyện bằng Knowledge Distillation (KD)
trên tập VAL và TEST qua 3 seeds (42, 43, 44).

Giao thức:
- Tuyệt đối không chọn ngưỡng trên TEST.
- Dùng cùng 5 điểm vận hành chọn trên VAL (`select_thresholds_on_val`) và áp cố định sang TEST.
- Lưu kết quả tại:
  - `results/binary/m2_ds_kd_fp32/{V}/test_metrics.json`
  - `results/binary/m2_ds_kd_fp32/{V}/confusion_matrix.png`
  - Cập nhật bảng tổng hợp `results/binary/m2_ds_fp32/comparison_vs_m1reg.md`
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

from src.models.cnn1d_binary_ds import CNN1DBinaryDS, DS_VARIANTS
from evaluate_cnn1d_binary import (
    predict_split,
    select_thresholds_on_val,
    eval_at_threshold,
    summarize_across_seeds,
)

SEEDS = [42, 43, 44]
CKPT_DIR = Path("checkpoints")
RESULTS_BASE_DIR = Path("results/binary/m2_ds_kd_fp32")
M1_REG_METRICS_PATH = Path("results/binary/m1_reg_fp32/test_metrics.json")
M2_B_METRICS_PATH = Path("results/binary/m2_ds_fp32/B/test_metrics.json")
M2_A_METRICS_PATH = Path("results/binary/m2_ds_fp32/A/test_metrics.json")


def update_master_comparison_markdown(kd_variant: str, kd_data: dict):
    """Cập nhật bảng đối chiếu tổng hợp M1-Reg vs M2-A vs M2-B vs M2-B-KD."""
    if not M1_REG_METRICS_PATH.exists() or not M2_B_METRICS_PATH.exists():
        return

    with open(M1_REG_METRICS_PATH, "r", encoding="utf-8") as f:
        m1 = json.load(f)
    with open(M2_B_METRICS_PATH, "r", encoding="utf-8") as f:
        m2b = json.load(f)
    m2a = None
    if M2_A_METRICS_PATH.exists():
        with open(M2_A_METRICS_PATH, "r", encoding="utf-8") as f:
            m2a = json.load(f)

    def fmt_ms(d: dict, key1: str, key2: str = None):
        sub = d[key1] if key2 is None else d[key1][key2]
        return f"`{sub['mean']:.4f} ± {sub['std']:.4f}`"

    def fmt_pct(d: dict, cls_char: str):
        sub = d["subclass_recall"][cls_char]
        return f"`{sub['mean']*100:.2f}% ± {sub['std']*100:.2f}%`"

    m1_tf = m1["threshold_free_summary"]
    m1_t05 = m1["test_summary_mean_std"]["1_default_0.50"]
    m1_f1 = m1["test_summary_mean_std"]["2_val_best_f1"]
    m1_r85 = m1["test_summary_mean_std"]["4_val_recall_ge_0.85"]

    m2b_tf = m2b["threshold_free_summary"]
    m2b_t05 = m2b["test_summary_mean_std"]["1_default_0.50"]
    m2b_f1 = m2b["test_summary_mean_std"]["2_val_best_f1"]
    m2b_r85 = m2b["test_summary_mean_std"]["4_val_recall_ge_0.85"]

    kd_tf = kd_data["threshold_free_summary"]
    kd_t05 = kd_data["test_summary_mean_std"]["1_default_0.50"]
    kd_f1 = kd_data["test_summary_mean_std"]["2_val_best_f1"]
    kd_r85 = kd_data["test_summary_mean_std"]["4_val_recall_ge_0.85"]

    lines = [
        "# BÁO CÁO ĐỐI CHIẾU THỰC NGHIỆM (E2 + E3 + E4): `M1-REG` vs `M2-A` vs `M2-B` vs `M2-B-KD`",
        "",
        "## 1. So sánh Chi phí Tính toán & Bộ nhớ trên STM32 **[Đã đo bằng Forward Hooks & Fused BN]**",
        "",
        "| Chỉ số Tài nguyên | `M1-Reg` (Chuẩn Conv1D) | `M2-A` (DS-Conv 3 tầng) | `M2-B` & `M2-B-KD` (DS-CNN 4 tầng) |",
        "| :--- | :---: | :---: | :---: |",
        "| **Tham số trên MCU (sau gộp BN)** | 35,458 (100%) | 11,330 (-68.0%) | **29,058 (-18.1%)** |",
        "| **Số phép tính MACCs / nhịp** | 2,970,496 (100%) | 873,856 (-70.6%) | **1,468,544 (-50.6%)** |",
        "| **Vùng nhìn (Receptive Field trước GAP)** | 24 mẫu (`66.7 ms`) | 24 mẫu (`66.7 ms`) | **86 mẫu (`238.9 ms`)** |",
        "| **Flash Weights (FP32 / INT8)** | 138.51 KB / 34.63 KB | 44.26 KB / 11.06 KB | **113.51 KB / 28.38 KB** |",
        "",
        "---",
        "",
        "## 2. So sánh Chất lượng Phân loại trên tập TEST (`49,691` nhịp, `Mean ± Std` qua 3 Seeds) **[Đã đo]**",
        "",
        "| Chỉ số Chất lượng (TEST) | `M1-Reg` (Teacher) | `M2-A` (3 tầng, RF=24) | `M2-B` (4 tầng, RF=86) | **`M2-B-KD` (E4: Chưng cất từ `M1-Reg`)** |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **VAL AUROC** | {fmt_ms(m1_tf, 'val_auroc')} | {fmt_ms(m2a['threshold_free_summary'], 'val_auroc') if m2a else 'N/A'} | {fmt_ms(m2b_tf, 'val_auroc')} | **{fmt_ms(kd_tf, 'val_auroc')}** |",
        f"| **TEST AUROC** | {fmt_ms(m1_tf, 'test_auroc')} | {fmt_ms(m2a['threshold_free_summary'], 'test_auroc') if m2a else 'N/A'} | {fmt_ms(m2b_tf, 'test_auroc')} | **{fmt_ms(kd_tf, 'test_auroc')}** |",
        f"| **TEST PR-AUC** | {fmt_ms(m1_tf, 'test_pr_auc')} | {fmt_ms(m2a['threshold_free_summary'], 'test_pr_auc') if m2a else 'N/A'} | {fmt_ms(m2b_tf, 'test_pr_auc')} | **{fmt_ms(kd_tf, 'test_pr_auc')}** |",
        f"| **Recall (`non-N`) @ `th=0.50`** | {fmt_ms(m1_t05, 'recall_non_N')} | {fmt_ms(m2a['test_summary_mean_std']['1_default_0.50'], 'recall_non_N') if m2a else 'N/A'} | {fmt_ms(m2b_t05, 'recall_non_N')} | **{fmt_ms(kd_t05, 'recall_non_N')}** |",
        f"| **Specificity (`N`) @ `th=0.50`** | {fmt_ms(m1_t05, 'specificity_N')} | {fmt_ms(m2a['test_summary_mean_std']['1_default_0.50'], 'specificity_N') if m2a else 'N/A'} | {fmt_ms(m2b_t05, 'specificity_N')} | **{fmt_ms(kd_t05, 'specificity_N')}** |",
        f"| **F1 (`non-N`) @ `th=0.50`** | {fmt_ms(m1_t05, 'f1_non_N')} | {fmt_ms(m2a['test_summary_mean_std']['1_default_0.50'], 'f1_non_N') if m2a else 'N/A'} | {fmt_ms(m2b_t05, 'f1_non_N')} | **{fmt_ms(kd_t05, 'f1_non_N')}** |",
        f"| **Recall (`non-N`) @ `4_val_recall_ge_0.85`** | {fmt_ms(m1_r85, 'recall_non_N')} | {fmt_ms(m2a['test_summary_mean_std']['4_val_recall_ge_0.85'], 'recall_non_N') if m2a else 'N/A'} | {fmt_ms(m2b_r85, 'recall_non_N')} | **{fmt_ms(kd_r85, 'recall_non_N')}** |",
        f"| **Specificity (`N`) @ `4_val_recall_ge_0.85`** | {fmt_ms(m1_r85, 'specificity_N')} | {fmt_ms(m2a['test_summary_mean_std']['4_val_recall_ge_0.85'], 'specificity_N') if m2a else 'N/A'} | {fmt_ms(m2b_r85, 'specificity_N')} | **{fmt_ms(kd_r85, 'specificity_N')}** |",
        f"| **Accuracy @ `2_val_best_f1`** | {fmt_ms(m1_f1, 'accuracy')} | {fmt_ms(m2a['test_summary_mean_std']['2_val_best_f1'], 'accuracy') if m2a else 'N/A'} | {fmt_ms(m2b_f1, 'accuracy')} | **{fmt_ms(kd_f1, 'accuracy')}** |",
        f"| **F1 (`non-N`) @ `2_val_best_f1`** | {fmt_ms(m1_f1, 'f1_non_N')} | {fmt_ms(m2a['test_summary_mean_std']['2_val_best_f1'], 'f1_non_N') if m2a else 'N/A'} | {fmt_ms(m2b_f1, 'f1_non_N')} | **{fmt_ms(kd_f1, 'f1_non_N')}** |",
        f"| **Tỷ lệ bắt nhịp `S` @ `th=0.50`** | {fmt_pct(m1_t05, 'S')} | {fmt_pct(m2a['test_summary_mean_std']['1_default_0.50'], 'S') if m2a else 'N/A'} | {fmt_pct(m2b_t05, 'S')} | **{fmt_pct(kd_t05, 'S')}** |",
        f"| **Tỷ lệ bắt nhịp `V` @ `th=0.50`** | {fmt_pct(m1_t05, 'V')} | {fmt_pct(m2a['test_summary_mean_std']['1_default_0.50'], 'V') if m2a else 'N/A'} | {fmt_pct(m2b_t05, 'V')} | **{fmt_pct(kd_t05, 'V')}** |",
        f"| **Tỷ lệ bắt nhịp `F` @ `th=0.50`** | {fmt_pct(m1_t05, 'F')} | {fmt_pct(m2a['test_summary_mean_std']['1_default_0.50'], 'F') if m2a else 'N/A'} | {fmt_pct(m2b_t05, 'F')} | **{fmt_pct(kd_t05, 'F')}** |",
        f"| **Tỷ lệ bắt nhịp `Q` @ `th=0.50`** | {fmt_pct(m1_t05, 'Q')} | {fmt_pct(m2a['test_summary_mean_std']['1_default_0.50'], 'Q') if m2a else 'N/A'} | {fmt_pct(m2b_t05, 'Q')} | **{fmt_pct(kd_t05, 'Q')}** |",
        "",
        "---",
        "",
        f"## 3. Đánh giá Gate E3/E4 cho `M2-{kd_variant}-KD`",
        "",
        f"- **TEST AUROC >= 0.877:** `{kd_tf['test_auroc']['mean']:.4f}` -> **{'ĐẠT' if kd_data['gate_e3_assessment']['pass_test_auroc'] else 'CHƯA ĐẠT'}**",
        f"- **TEST PR-AUC >= 0.526:** `{kd_tf['test_pr_auc']['mean']:.4f}` -> **{'ĐẠT' if kd_data['gate_e3_assessment']['pass_test_pr_auc'] else 'CHƯA ĐẠT'}**",
        f"- **Số seed vượt đồng thời `Recall >= 0.85 & Spec >= 0.85` (ngưỡng từ VAL):** `{kd_data['gate_e3_assessment']['seeds_passing_85_85']}/3` (Seeds: `{kd_data['gate_e3_assessment']['passing_seeds_list']}`) -> **{'ĐẠT' if kd_data['gate_e3_assessment']['pass_gate_85_85_any_seed'] else 'CHƯA ĐẠT'}**",
        f"- **Recall nhịp `S` @ `th=0.50` >= 71.0%:** `{kd_t05['subclass_recall']['S']['mean']*100:.2f}%` -> **{'ĐẠT' if kd_data['gate_e3_assessment']['pass_s_recall'] else 'CHƯA ĐẠT'}**",
        f"- **Giảm chi phí MACC ~50%:** `1,468,544 MACCs (-50.6%)` -> **ĐẠT**",
    ]

    md_path = Path("results/binary/m2_ds_fp32/comparison_vs_m1reg.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Đã cập nhật báo cáo đối chiếu tổng hợp tại: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN1DBinaryDS KD across 3 seeds (Step E4)")
    parser.add_argument(
        "--variant",
        type=str,
        default="B",
        choices=["A", "B", "C"],
        help="DS-CNN architectural variant to evaluate (default: B)",
    )
    args = parser.parse_args()
    variant = args.variant.upper()
    var_cfg = DS_VARIANTS[variant]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = RESULTS_BASE_DIR / variant
    results_dir.mkdir(parents=True, exist_ok=True)

    per_seed_val = {}
    per_seed_test = {}
    per_seed_meta = {}
    test_probs_all = []
    y2_test_ref = None

    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"cnn1d_binary_ds{variant}_kd_best_seed{seed}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
            "kd_alpha": float(ckpt.get("kd_alpha", 0.5)),
            "kd_temperature": float(ckpt.get("kd_temperature", 3.0)),
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
    axes[0].set_title(f"M2-{variant}-KD Test CM (Seed 42, th={th_eq_42:.2f})")

    for idx, seed in enumerate(SEEDS):
        fpr, tpr, _ = roc_curve(y2_test_ref, test_probs_all[idx])
        auc_s = per_seed_meta[seed]["test_auroc"]
        axes[1].plot(fpr, tpr, label=f"Seed {seed} (AUROC = {auc_s:.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].axvline(0.15, color="red", linestyle=":", alpha=0.6, label="Gate 85/85")
    axes[1].axhline(0.85, color="red", linestyle=":", alpha=0.6)
    axes[1].set_xlabel("1 - Specificity (FPR)")
    axes[1].set_ylabel("Recall non-N (TPR)")
    axes[1].set_title(f"M2-{variant}-KD (DS-CNN + KD) Test ROC Curves")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    output = {
        "task": f"E4 - Evaluation of CNN1DBinaryDS Variant {variant} + Knowledge Distillation across 3 Seeds",
        "variant": variant,
        "variant_config": var_cfg,
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

    update_master_comparison_markdown(variant, output)

    print(f"\n=== TỔNG HỢP KẾT QUẢ M2-{variant}-KD (CNN1DBinaryDS + Knowledge Distillation - 3 SEEDS) ===")
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
    print(f"\nGate E3/E4 Assessment: {json.dumps(gate_e3)}")


if __name__ == "__main__":
    main()
