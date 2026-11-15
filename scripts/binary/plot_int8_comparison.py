"""
Vẽ biểu đồ đối chiếu toàn diện FP32 vs INT8 (Giai đoạn C, D, F) cho báo cáo:
1. Subclass Recall (S, V, F, Q) và Overall Recall/Specificity giữa FP32 và INT8.
2. So sánh tài nguyên trên NUCLEO-F446RE (Flash KB, Peak RAM KB, Latency ms @ 180 MHz).
Lưu tại: results/binary/int8_ptq/fp32_vs_int8_comparison.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INT8_JSON = PROJECT_ROOT / "results/binary/int8_ptq/test_metrics_int8.json"
SIM_JSON = PROJECT_ROOT / "results/binary/stm32_cubeai_simulation_report.json"
M1_FP32_JSON = PROJECT_ROOT / "results/binary/m1_reg_fp32/test_metrics.json"
M2_FP32_JSON = PROJECT_ROOT / "results/binary/m2_ds_kd_fp32/B/test_metrics.json"
OUT_PNG = PROJECT_ROOT / "results/binary/int8_ptq/fp32_vs_int8_comparison.png"


def main():
    with open(INT8_JSON, "r", encoding="utf-8") as f:
        int8_data = json.load(f)
    with open(SIM_JSON, "r", encoding="utf-8") as f:
        sim_data = json.load(f)
    with open(M1_FP32_JSON, "r", encoding="utf-8") as f:
        m1_fp32 = json.load(f)
    with open(M2_FP32_JSON, "r", encoding="utf-8") as f:
        m2_fp32 = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Subplot 1: Chất lượng phân loại (Mean 3 Seeds trên TEST 49,691 nhịp) ---
    metrics_labels = ["AUROC", "PR-AUC", "Recall\n(non-N)", "Specificity\n(N)", "Recall S\n(@th=0.5)", "Recall V\n(@th=0.5)"]

    def get_vals(fp32_Or_int8_dict, is_int8: bool):
        tf = fp32_Or_int8_dict["threshold_free_int8_summary" if is_int8 else "threshold_free_summary"]
        t_sum = fp32_Or_int8_dict["test_summary_fixed_fp32_th" if is_int8 else "test_summary_mean_std"]
        eq = t_sum["5_val_equal_rec_spec"]
        d50 = t_sum["1_default_0.50"]
        return [
            tf["test_auroc"]["mean"] * 100,
            tf["test_pr_auc"]["mean"] * 100,
            eq["recall_non_N"]["mean"] * 100,
            eq["specificity_N"]["mean"] * 100,
            d50["subclass_recall"]["S"]["mean"] * 100,
            d50["subclass_recall"]["V"]["mean"] * 100,
        ]

    v_m1_fp32 = get_vals(m1_fp32, False)
    v_m1_int8 = get_vals(int8_data["M1-Reg"], True)
    v_m2_fp32 = get_vals(m2_fp32, False)
    v_m2_int8 = get_vals(int8_data["M2-B-KD"], True)

    x = np.arange(len(metrics_labels))
    w = 0.20
    axes[0].bar(x - 1.5 * w, v_m1_fp32, width=w, label="M1-Reg FP32", color="#4C72B0", alpha=0.9)
    axes[0].bar(x - 0.5 * w, v_m1_int8, width=w, label="M1-Reg INT8", color="#64B5CD", alpha=0.9)
    axes[0].bar(x + 0.5 * w, v_m2_fp32, width=w, label="M2-B-KD FP32", color="#DD8452", alpha=0.9)
    axes[0].bar(x + 1.5 * w, v_m2_int8, width=w, label="M2-B-KD INT8", color="#55A868", alpha=0.9)

    axes[0].axhline(85.0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Gate 85% Target")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_labels, fontsize=9.5)
    axes[0].set_ylim(45, 102)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("TEST Classification Quality: FP32 vs INT8 (49,691 Beats, 3-Seed Mean)")
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.3)

    # --- Subplot 2: Tài nguyên & Độ trễ trên NUCLEO-F446RE (180 MHz) ---
    res_labels = [
        "Weights Flash\n(KB)",
        "Total Flash\nwith Runtime (KB)",
        "Peak RAM\nwith Runtime (KB)",
        "Est. Latency\n@180MHz (ms)",
    ]

    def get_hw_vals(model_key: str, prec: str):
        m_sim = sim_data[model_key]
        mem = m_sim["memory_summary"][prec]
        b = m_sim["board_evaluations"]["NUCLEO-F446RE"][prec]
        w_kb = mem["weights_ro_kb"] if prec == "FP32" else mem["weights_and_scales_ro_kb"]
        return [w_kb, b["total_flash_kb"], b["total_ram_kb"], b["estimated_latency_ms"]]

    hw_m1_fp32 = get_hw_vals("M1-Reg", "FP32")
    hw_m1_int8 = get_hw_vals("M1-Reg", "INT8")
    hw_m2_fp32 = get_hw_vals("M2-B-KD", "FP32")
    hw_m2_int8 = get_hw_vals("M2-B-KD", "INT8")

    x2 = np.arange(len(res_labels))
    b1 = axes[1].bar(x2 - 1.5 * w, hw_m1_fp32, width=w, label="M1-Reg FP32", color="#4C72B0")
    b2 = axes[1].bar(x2 - 0.5 * w, hw_m1_int8, width=w, label="M1-Reg INT8", color="#64B5CD")
    b3 = axes[1].bar(x2 + 0.5 * w, hw_m2_fp32, width=w, label="M2-B-KD FP32", color="#DD8452")
    b4 = axes[1].bar(x2 + 1.5 * w, hw_m2_int8, width=w, label="M2-B-KD INT8", color="#55A868")

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 2.0,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=0,
            )

    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(res_labels, fontsize=9.5)
    axes[1].set_ylim(0, 175)
    axes[1].set_ylabel("KB / ms")
    axes[1].set_title("NUCLEO-F446RE (180 MHz, 512KB Flash, 128KB RAM) Footprint")
    axes[1].legend(loc="upper right", fontsize=8.5)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"Đã lưu biểu đồ đối chiếu FP32 vs INT8 tại: {OUT_PNG}")


if __name__ == "__main__":
    main()
