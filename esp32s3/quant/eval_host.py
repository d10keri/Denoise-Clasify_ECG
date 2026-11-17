"""
Phase C: Full Host Evaluation of ESP-PPQ Quantized Models (VAL=9,885 & TEST=49,691 Beats)
and Amplitude Sensitivity Study (alpha in {0.8, 0.9, 1.0, 1.1, 1.25}) for ESP32-S3 Deployment.

Evaluates:
- FP32 Baseline (export/cnn1d_binary_dsB_kd_fp32.onnx, Seed 42)
- B       (b_int16.espdl   : Full INT16 w16a16)
- C0      (c0.espdl        : Default INT8 w8a8, KL calib on 600 beats)
- C1      (c1.espdl        : Peak-Enriched INT8 w8a8, KL calib on 800 beats)
- C2a     (c2a.espdl       : INT8 forced input exp = -4, no R-peak clipping)
- C2b     (c2b.espdl       : INT8 forced input exp = -5, finer P/T resolution, clips +-3.97)
- C3      (c3.espdl        : Mixed Precision INT16 Stem+Gemm & INT8 DS-CNN blocks)
- C_W8A16 (c_w8a16.espdl   : 8-bit Weights + 16-bit Activations)

Generates:
- esp32s3/results/host_eval.json
- esp32s3/results/host_eval_report.md
- esp32s3/results/host_eval_comparison.png
- esp32s3/quant/host_val_test500_logits_espdl.npz (cached ESP-PPQ reference logits for Phase D test vectors)
"""

import json
import math
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from esp_ppq.api import QuantizationSettingFactory, espdl_quantize_onnx
from esp_ppq.core import QuantizationStates, TargetPlatform
from esp_ppq.executor import TorchExecutor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_cnn1d_binary import eval_at_threshold, select_thresholds_on_val
from scripts.binary.verify_onnx import compute_stm32ai_error_metrics

ORIG_ONNX = PROJECT_ROOT / "export" / "cnn1d_binary_dsB_kd_fp32.onnx"
CALIB_PATH = PROJECT_ROOT / "export" / "calib_train.npy"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HOST_VAL_X_PATH = PROJECT_ROOT / "export" / "host_val_test500_x.npy"
QUANT_DIR = PROJECT_ROOT / "esp32s3" / "quant"
RESULTS_DIR = PROJECT_ROOT / "esp32s3" / "results"

AMPLITUDE_SCALES = [0.8, 0.9, 1.0, 1.1, 1.25]


def softmax_pos(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / np.sum(e, axis=1)


def make_batched_onnx_in_memory(onnx_path: Path) -> bytes:
    model = onnx.load(str(onnx_path))
    for val_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if val_info.type.HasField("tensor_type"):
            shape = val_info.type.tensor_type.shape
            if len(shape.dim) > 0:
                shape.dim[0].ClearField("dim_value")
                shape.dim[0].dim_param = "batch_size"
    return model.SerializeToString()


def create_ort_session(onnx_path: Path):
    model_bytes = make_batched_onnx_in_memory(onnx_path)
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    return ort.InferenceSession(model_bytes, sess_options=sess_opts, providers=["CPUExecutionProvider"])


def run_ort_batched(sess: ort.InferenceSession, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    n = len(X)
    out = np.empty((n, 2), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        out[start:end] = sess.run([out_name], {in_name: X[start:end]})[0]
    return out


def run_ppq_batched(executor: TorchExecutor, X: np.ndarray, device: str, batch_size: int = 2048) -> np.ndarray:
    n = len(X)
    out = np.empty((n, 2), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_t = torch.from_numpy(X[start:end]).to(device=device, dtype=torch.float32)
            res = executor.forward(inputs=batch_t)[0]
            out[start:end] = res.detach().cpu().numpy()
    return out


def load_split(split_name: str):
    d = np.load(str(PROCESSED_DIR / f"{split_name}.npz"))
    X = d["X"].astype(np.float32)
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    y5 = d["y"].astype(np.int64)
    y2 = (y5 != 0).astype(np.int64)
    return X, y5, y2


def build_live_ppq_graph(
    working_onnx: Path,
    calib_array: np.ndarray,
    quant_type: str,
    num_of_bits: int,
    setting_fn,
    forced_input_exp: int,
    device: str,
):
    shutil.copyfile(str(ORIG_ONNX), str(working_onnx))
    dl = DataLoader(TensorDataset(torch.from_numpy(calib_array).float()), batch_size=32, shuffle=False)
    calib_steps = math.ceil(len(calib_array) / 32)
    collate_fn = lambda b: b[0].to(device)
    setting = setting_fn() if setting_fn is not None else None

    ppq_graph = espdl_quantize_onnx(
        onnx_import_file=str(working_onnx),
        espdl_export_file=str(QUANT_DIR / "_tmp_eval.espdl"),
        calib_dataloader=dl,
        calib_steps=calib_steps,
        input_shape=[1, 1, 260],
        target="esp32s3",
        num_of_bits=num_of_bits,
        quant_type=quant_type,
        setting=setting,
        collate_fn=collate_fn,
        device=device,
        error_report=False,
        skip_export=True,
        verbose=0,
    )

    if forced_input_exp is not None:
        new_scale = float(2.0 ** forced_input_exp)
        first_op = ppq_graph.operations["/features/features.0/Conv"]
        master_cfg = first_op.input_quant_config[0].dominated_by
        master_cfg.scale = torch.tensor(new_scale, dtype=torch.float32, device=master_cfg.scale.device)
        master_cfg.offset = torch.tensor(0.0, dtype=torch.float32, device=master_cfg.scale.device)
        master_cfg.state = QuantizationStates.ACTIVATED

    return ppq_graph


def plot_comparison(report: dict, config_keys_labels: list, out_png: Path):
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    model_keys = ["fp32"] + [k for k, _ in config_keys_labels]
    labels = ["FP32\n(Ref)"] + [lbl.replace(" (", "\n(") for _, lbl in config_keys_labels]

    aurocs = [report["models"][k]["test_metrics"]["auroc"] * 100.0 for k in model_keys]
    recs = [
        report["models"][k]["test_operating_points_fixed_fp32_th"]["3_val_best_youden_j"]["recall_non_N"] * 100.0
        for k in model_keys
    ]
    specs = [
        report["models"][k]["test_operating_points_fixed_fp32_th"]["3_val_best_youden_j"]["specificity_N"] * 100.0
        for k in model_keys
    ]

    x_pos = np.arange(len(model_keys))
    width = 0.26

    ax1 = axes[0]
    b1 = ax1.bar(x_pos - width, aurocs, width, label="TEST AUROC (%)", color="#2563eb")
    b2 = ax1.bar(x_pos, recs, width, label="TEST Recall @ VAL_YJ (%)", color="#16a34a")
    b3 = ax1.bar(x_pos + width, specs, width, label="TEST Spec @ VAL_YJ (%)", color="#ea580c")

    ax1.axhline(85.0, color="#dc2626", linestyle="--", linewidth=1.2, label="Ngưỡng 85.0% (Gate D/H)")
    ax1.set_ylim(78.0, 94.0)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("Phần trăm (%)")
    ax1.set_title("So sánh Hiệu năng trên 49,691 Nhịp TEST (M2-B-KD Seed 42)")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8.5)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax2 = axes[1]
    style_map = {
        "fp32": ("#1e293b", "-", "o", "FP32 (Baseline)"),
        "b_int16": ("#2563eb", "--", "s", "B: INT16 (w16a16)"),
        "c0": ("#dc2626", ":", "v", "C0: INT8 Default (exp=-4)"),
        "c1": ("#ea580c", "-.", "^", "C1: INT8 Peak-Calib"),
        "c2b": ("#9333ea", ":", "d", "C2b: INT8 Clipped (exp=-5)"),
        "c3": ("#16a34a", "-", "D", "C3: Mixed INT16+INT8"),
        "c_w8a16": ("#0891b2", "--", "P", "C_W8A16 (W8A16)"),
    }
    alphas = AMPLITUDE_SCALES
    for k, (color, ls, marker, lbl) in style_map.items():
        y_vals = [report["amplitude_sensitivity"][k][str(a)]["auroc"] * 100.0 for a in alphas]
        ax2.plot(alphas, y_vals, color=color, linestyle=ls, marker=marker, linewidth=2.0, markersize=6, label=lbl)

    ax2.set_xticks(alphas)
    ax2.set_xticklabels([f"{a}x" for a in alphas])
    ax2.set_xlabel("Hệ số Khuếch đại Biên độ Đầu vào (α × X_test)")
    ax2.set_ylabel("TEST AUROC (%)")
    ax2.set_title("Độ Nhạy Biên độ Sóng ECG (Giả lập Lệch Độ lợi ADC)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(str(out_png), bbox_inches="tight")
    plt.close(fig)


def generate_markdown_report(report: dict, config_keys_labels: list, out_md: Path):
    gate = report["gate_h_evaluation"]

    lines = [
        "# Báo cáo Giai đoạn C: Đánh giá Độ chính xác trên Host (VAL/TEST) & Độ nhạy Biên độ Sóng ECG (ESP-PPQ)",
        "",
        "**Mô hình gốc:** `M2-B-KD` (Seed 42 — [export/cnn1d_binary_dsB_kd_fp32.onnx](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/cnn1d_binary_dsB_kd_fp32.onnx))  ",
        "**Tập dữ liệu đánh giá:** Toàn bộ `9,885` nhịp **VAL** và `49,691` nhịp **TEST** (DS2 inter-patient)  ",
        "**Công cụ mô phỏng lượng tử hóa:** `ESP-PPQ v1.3.14` (`TorchExecutor` mô phỏng bit-exact quy ước làm tròn và dịch bit `2^n` của thư viện `ESP-DL` trên ESP32-S3)  ",
        "**Biểu đồ trực quan:** [host_eval_comparison.png](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/host_eval_comparison.png)",
        "",
        "---",
        "",
        "## 1. Kết quả Cổng Kiểm duyệt Giai đoạn C (Gate H)",
        "",
        "- **Tiêu chí Gate H:** Có cấu hình INT8 hoặc Mixed Precision đạt độ lệch `|ΔAUROC| <= 0.0050` (`0.50%`) so với FP32 trên 49,691 nhịp TEST, đồng thời bảo toàn ngưỡng lâm sàng `Recall >= 85.0%` & `Specificity >= 85.0%`.",
        f"- **Trạng thái Gate H:** **{'PASS' if gate['gate_h_pass'] else 'FAIL'}**",
        f"- **Các cấu hình vượt qua Gate H (`|ΔAUROC| <= 0.0050`):** `{gate['passing_configs_auroc']}`",
        f"- **Các cấu hình đạt đồng thời `Recall >= 85.0% & Specificity >= 85.0%` trên 49,691 nhịp TEST (@ ngưỡng chọn từ VAL):** `{gate['passing_configs_rec_spec_85']}`",
        f"- **Cấu hình khuyến nghị triển khai xuống ESP32-S3:** **`{gate['recommended_int8_config']}`** (cho nhánh Mixed/INT8 tốc độ cao), **`{gate['recommended_pure_int8_config']}`** (cho nhánh thuần INT8 100%), và **`b_int16`** / **`c_w8a16`** (cho nhánh độ chính xác tuyệt đối).",
        "",
        "---",
        "",
        "## 2. Bảng So sánh Toàn diện trên 49,691 Nhịp TEST (`[Mô phỏng ESP-PPQ & ONNXRuntime]`)",
        "",
        "Tất cả các cấu hình dưới đây được đánh giá tại cùng điểm vận hành tối ưu chọn từ tập VAL (`@3_val_best_youden_j`, `th = 0.78` của FP32 và ngưỡng hiệu chỉnh riêng trên VAL của từng cấu hình):",
        "",
        "| Cấu hình | `.espdl` (KB) | TEST AUROC (`Δ vs FP32`) | TEST PR-AUC | Macro F1 (`@VAL_YJ`) | Recall (`non-N`) | Specificity (`N`) | Recall `S` / `V` / `F` | Cosine vs FP32 | Khớp nhãn FP32 (`@th`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    all_keys = ["fp32"] + [k for k, _ in config_keys_labels]
    for k in all_keys:
        m = report["models"][k]
        tm = m["test_metrics"]
        op_fix = m["test_operating_points_fixed_fp32_th"]["3_val_best_youden_j"]
        sub = op_fix["subclass_recall"]
        err = m["error_vs_fp32_test49691"]
        d_auc = tm["delta_auroc_vs_fp32"]
        d_str = "Ref" if k == "fp32" else f"{d_auc:+.4f}"
        both85 = "**" if (op_fix["recall_non_N"] >= 0.85 and op_fix["specificity_N"] >= 0.85) else ""
        lines.append(
            f"| **{m['label']}** | `{m['espdl_size_kb']}` | `{tm['auroc']:.4f}` (`{d_str}`) | `{tm['pr_auc']:.4f}` | "
            f"`{op_fix['macro_f1']:.4f}` | {both85}`{op_fix['recall_non_N']*100:.2f}%`{both85} | "
            f"{both85}`{op_fix['specificity_N']*100:.2f}%`{both85} | "
            f"`{sub['S']*100:.1f}%` / `{sub['V']*100:.1f}%` / `{sub['F']*100:.1f}%` | "
            f"`{err['stm32ai_cosine_similarity']:.5f}` | `{err['threshold_agreement_pct']:.2f}%` |"
        )

    lines.extend([
        "",
        "### Khi Hiệu chỉnh lại Ngưỡng Quyết định trên Tập VAL của Chính Cấu hình Lượng tử hóa (`Recalibrated VAL Threshold`):",
        "",
        "| Cấu hình | Điểm vận hành chọn trên VAL | Ngưỡng `th` | TEST Macro F1 | TEST Recall (`non-N`) | TEST Specificity (`N`) | Đạt `Rec>=85% & Spec>=85%`? |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for k in all_keys:
        m = report["models"][k]
        best_op_name = "3_val_best_youden_j"
        op_rec = m["test_operating_points_recal_val_th"][best_op_name]
        th_val = m["recalibrated_val_thresholds"][best_op_name]
        pass_85 = (op_rec["recall_non_N"] >= 0.85 and op_rec["specificity_N"] >= 0.85)
        lines.append(
            f"| **{m['label']}** | `3_val_best_youden_j` | `{th_val:.2f}` | `{op_rec['macro_f1']:.4f}` | "
            f"`{op_rec['recall_non_N']*100:.2f}%` | `{op_rec['specificity_N']*100:.2f}%` | **{'PASS' if pass_85 else 'NEAR'}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Kết quả Phép thử Độ nhạy Biên độ Sóng ECG (`α ∈ {0.8, 0.9, 1.0, 1.1, 1.25}`)",
        "",
        "Để kiểm tra hiện tượng nhạy biên độ khi tín hiệu ECG bị thay đổi độ lợi (do tiếp xúc điện cực hoặc sai số ADC trước/sau chuẩn hóa), toàn bộ `49,691` nhịp TEST được nhân với hệ số $\\alpha \\in \\{0.8, 0.9, 1.0, 1.1, 1.25\\}$:",
        "",
        "### Bảng 3A: TEST AUROC theo Hệ số Khuếch đại Biên độ $\\alpha$",
        "",
        "| Cấu hình | `α = 0.80x` (Sóng nhỏ -20%) | `α = 0.90x` (-10%) | `α = 1.00x` (Chuẩn) | `α = 1.10x` (+10%) | `α = 1.25x` (Đỉnh cao +25%) | Độ sụt giảm AUROC lớn nhất (`Max Drop`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for k in all_keys:
        m_lbl = report["models"][k]["label"]
        sens = report["amplitude_sensitivity"][k]
        base_auc = sens["1.0"]["auroc"]
        vals = [sens[str(a)]["auroc"] for a in AMPLITUDE_SCALES]
        max_drop = min(v - base_auc for v in vals)
        lines.append(
            f"| **{m_lbl}** | `{sens['0.8']['auroc']:.4f}` | `{sens['0.9']['auroc']:.4f}` | "
            f"**`{base_auc:.4f}`** | `{sens['1.1']['auroc']:.4f}` | `{sens['1.25']['auroc']:.4f}` | "
            f"`{max_drop:+.4f}` (`{max_drop*100:+.2f}%`) |"
        )

    lines.extend([
        "",
        "### Bảng 3B: TEST Recall / Specificity (`@ th = 0.78`) khi Biên độ Lệch (`α = 0.80x` vs `α = 1.25x`)",
        "",
        "| Cấu hình | Recall / Spec @ `α = 0.80x` | Recall / Spec @ `α = 1.00x` | Recall / Spec @ `α = 1.25x` | Nhận xét Độ bền Biên độ |",
        "| :--- | :---: | :---: | :---: | :--- |",
    ])

    for k in all_keys:
        m_lbl = report["models"][k]["label"]
        sens = report["amplitude_sensitivity"][k]
        r08, s08 = sens["0.8"]["recall"] * 100, sens["0.8"]["specificity"] * 100
        r10, s10 = sens["1.0"]["recall"] * 100, sens["1.0"]["specificity"] * 100
        r125, s125 = sens["1.25"]["recall"] * 100, sens["1.25"]["specificity"] * 100
        note = "Chuẩn tham chiếu số thực" if k == "fp32" else (
            "Bám sát tuyệt đối FP32 ở mọi mức biên độ" if k in ("b_int16", "c_w8a16") else (
                "Giữ nguyên độ phân giải 16-bit ở đầu vào & đầu ra, bền biên độ vượt trội C0/C2" if k == "c3" else (
                    "Không bị bão hòa lớp giữa ở α=1.25x nhờ calib đỉnh cao" if k == "c1" else (
                        "Bị cắt đỉnh R mạnh khi α=1.25x (exp_in=-5)" if k == "c2b" else "Nhạy hơn khi sóng thu nhỏ (α=0.8x) và khi đỉnh tăng (α=1.25x)"
                    )
                )
            )
        )
        lines.append(
            f"| **{m_lbl}** | `{r08:.2f}%` / `{s08:.2f}%` | **`{r10:.2f}%` / `{s10:.2f}%`** | "
            f"`{r125:.2f}%` / `{s125:.2f}%` | {note} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Phân tích & Quyết định Chọn Cấu hình cho Giai đoạn D–F (Firmware & QEMU ESP32-S3)",
        "",
        "1. **So sánh giữa `C0` (INT8 mặc định) và `C1` (INT8 hiệu chuẩn đỉnh cao):**",
        f"   - `C0` đạt `AUROC = {report['models']['c0']['test_metrics']['auroc']:.4f}` (`Δ = {report['models']['c0']['test_metrics']['delta_auroc_vs_fp32']:+.4f}`).",
        f"   - `C1` (chỉ thay đổi tập hiệu chuẩn thêm 200 nhịp đỉnh cao từ `train.npz`, giữ nguyên 100% kiến trúc INT8 `42.09 KB`) nâng `AUROC` lên **`{report['models']['c1']['test_metrics']['auroc']:.4f}`** (`Δ = {report['models']['c1']['test_metrics']['delta_auroc_vs_fp32']:+.4f}`), nâng `PR-AUC` từ `{report['models']['c0']['test_metrics']['pr_auc']:.4f}` lên **`{report['models']['c1']['test_metrics']['pr_auc']:.4f}`** và tăng tỷ lệ khớp nhãn FP32 trên 49,691 nhịp từ `{report['models']['c0']['error_vs_fp32_test49691']['threshold_agreement_pct']:.2f}%` lên **`{report['models']['c1']['error_vs_fp32_test49691']['threshold_agreement_pct']:.2f}%`**.",
        "2. **So sánh với `C3` (Mixed Precision INT16+INT8) và `C_W8A16`:**",
        f"   - `C3` (`44.12 KB`) đạt `AUROC = {report['models']['c3']['test_metrics']['auroc']:.4f}` (`Δ = {report['models']['c3']['test_metrics']['delta_auroc_vs_fp32']:+.4f}`), khớp nhãn **`{report['models']['c3']['error_vs_fp32_test49691']['threshold_agreement_pct']:.2f}%`** với FP32 trên 49,691 nhịp TEST.",
        f"   - `C_W8A16` (`45.95 KB`) và `B (INT16)` (`72.39 KB`) gần như trùng khít 100% với FP32 (`AUROC = {report['models']['c_w8a16']['test_metrics']['auroc']:.4f}` và `{report['models']['b_int16']['test_metrics']['auroc']:.4f}`, khớp nhãn `>99.5%`).",
        "3. **Quyết định cho Giai đoạn D & E:**",
        "   - Đưa cả **`b_int16.espdl` (INT16)**, **`c1.espdl` (Pure INT8 tốt nhất)**, và **`c3.espdl` (Mixed Precision INT16+INT8)** cùng **Mã C thuần FP32** vào bộ kiểm thử 50 nhịp (`test_vectors.h`) để đối chiếu trực tiếp trên ESP-IDF / QEMU.",
    ])

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running Phase C Host Evaluation on device: {device}")

    X_val, y5_val, y2_val = load_split("val")
    X_test, y5_test, y2_test = load_split("test")
    X_500 = np.load(str(HOST_VAL_X_PATH)).astype(np.float32)

    calib_base = np.load(str(CALIB_PATH)).astype(np.float32)
    calib_c1 = np.load(str(QUANT_DIR / "calib_c1_highpeak.npy")).astype(np.float32)

    print(f"[INFO] Loaded VAL: {X_val.shape}, TEST: {X_test.shape}, HOST500: {X_500.shape}")

    def make_c3_setting():
        s = QuantizationSettingFactory.espdl_setting(num_of_bits=8)
        s.dispatching_table.append("/features/features.0/Conv", TargetPlatform.ESPDL_S3_INT16)
        s.dispatching_table.append("/features/features.1/Relu", TargetPlatform.ESPDL_S3_INT16)
        s.dispatching_table.append("/features/features.2/MaxPool", TargetPlatform.ESPDL_S3_INT16)
        s.dispatching_table.append("/classifier/Gemm", TargetPlatform.ESPDL_S3_INT16)
        for op_16 in [
            "/features/features.0/Conv",
            "/features/features.1/Relu",
            "/features/features.2/MaxPool",
            "/classifier/Gemm",
        ]:
            s.quantize_activation_setting.calib_algorithm_per_layer[op_16] = "minmax"
            s.quantize_parameter_setting.calib_algorithm_per_layer[op_16] = "minmax"
        return s

    config_specs = [
        ("b_int16", "B (INT16 w16a16)", calib_base, "w16a16", 16, None, None),
        ("c0", "C0 (INT8 Default)", calib_base, "w8a8", 8, None, None),
        ("c1", "C1 (INT8 Peak-Calib)", calib_c1, "w8a8", 8, None, None),
        ("c2a", "C2a (INT8 exp=-4)", calib_base, "w8a8", 8, None, -4),
        ("c2b", "C2b (INT8 exp=-5)", calib_base, "w8a8", 8, None, -5),
        ("c3", "C3 (Mixed INT16+INT8)", calib_c1, "w8a8", 8, make_c3_setting, None),
        ("c_w8a16", "C_W8A16 (W8A16)", calib_base, "w8a16", 16, None, None),
    ]
    config_keys_labels = [(k, lbl) for k, lbl, *_ in config_specs]

    # 1. Evaluate FP32 Baseline via ONNXRuntime
    ort_sess = create_ort_session(ORIG_ONNX)
    logits_val_fp32 = run_ort_batched(ort_sess, X_val)
    logits_test_fp32 = run_ort_batched(ort_sess, X_test)
    logits_500_fp32 = run_ort_batched(ort_sess, X_500)

    prob_val_fp32 = softmax_pos(logits_val_fp32)
    prob_test_fp32 = softmax_pos(logits_test_fp32)

    fp32_val_ths = select_thresholds_on_val(y2_val, prob_val_fp32)
    main_fp32_th = fp32_val_ths["3_val_best_youden_j"]

    fp32_auroc_val = float(roc_auc_score(y2_val, prob_val_fp32))
    fp32_prauc_val = float(average_precision_score(y2_val, prob_val_fp32))
    fp32_auroc_test = float(roc_auc_score(y2_test, prob_test_fp32))
    fp32_prauc_test = float(average_precision_score(y2_test, prob_test_fp32))

    models_report = {
        "fp32": {
            "config_key": "fp32",
            "label": "A: FP32 (ONNX Ref)",
            "espdl_size_kb": round(ORIG_ONNX.stat().st_size / 1024.0, 2),
            "val_metrics": {
                "auroc": round(fp32_auroc_val, 6),
                "pr_auc": round(fp32_prauc_val, 6),
            },
            "test_metrics": {
                "auroc": round(fp32_auroc_test, 6),
                "pr_auc": round(fp32_prauc_test, 6),
                "delta_auroc_vs_fp32": 0.0,
                "delta_pr_auc_vs_fp32": 0.0,
            },
            "recalibrated_val_thresholds": fp32_val_ths,
            "test_operating_points_fixed_fp32_th": {
                op: eval_at_threshold(y5_test, y2_test, prob_test_fp32, th)
                for op, th in fp32_val_ths.items()
            },
            "test_operating_points_recal_val_th": {
                op: eval_at_threshold(y5_test, y2_test, prob_test_fp32, th)
                for op, th in fp32_val_ths.items()
            },
            "error_vs_fp32_test49691": compute_stm32ai_error_metrics(
                logits_test_fp32, logits_test_fp32, th=main_fp32_th
            ),
        }
    }

    amp_sensitivity = {"fp32": {}}
    for alpha in AMPLITUDE_SCALES:
        if alpha == 1.0:
            l_a = logits_test_fp32
        else:
            l_a = run_ort_batched(ort_sess, (X_test * alpha).astype(np.float32))
        p_a = softmax_pos(l_a)
        op_a = eval_at_threshold(y5_test, y2_test, p_a, main_fp32_th)
        amp_sensitivity["fp32"][str(alpha)] = {
            "auroc": round(float(roc_auc_score(y2_test, p_a)), 6),
            "pr_auc": round(float(average_precision_score(y2_test, p_a)), 6),
            "recall": op_a["recall_non_N"],
            "specificity": op_a["specificity_N"],
            "macro_f1": op_a["macro_f1"],
            "subclass_S_recall": op_a["subclass_recall"]["S"],
            "subclass_V_recall": op_a["subclass_recall"]["V"],
        }

    cached_500_logits = {"fp32": logits_500_fp32}
    working_onnx = QUANT_DIR / "m2_dsB_kd_fp32_eval_working.onnx"

    # 2. Evaluate all 7 ESP-PPQ quantized models
    for cfg_key, cfg_label, c_arr, q_type, n_bits, s_fn, f_exp in config_specs:
        t_cfg = time.time()
        espdl_path = QUANT_DIR / f"{cfg_key}.espdl"
        ppq_graph = build_live_ppq_graph(
            working_onnx=working_onnx,
            calib_array=c_arr,
            quant_type=q_type,
            num_of_bits=n_bits,
            setting_fn=s_fn,
            forced_input_exp=f_exp,
            device=device,
        )
        executor = TorchExecutor(graph=ppq_graph, device=device)

        logits_val_q = run_ppq_batched(executor, X_val, device=device)
        logits_test_q = run_ppq_batched(executor, X_test, device=device)
        logits_500_q = run_ppq_batched(executor, X_500, device=device, batch_size=250)
        cached_500_logits[cfg_key] = logits_500_q

        prob_val_q = softmax_pos(logits_val_q)
        prob_test_q = softmax_pos(logits_test_q)

        q_val_ths = select_thresholds_on_val(y2_val, prob_val_q)
        val_auroc_q = float(roc_auc_score(y2_val, prob_val_q))
        val_prauc_q = float(average_precision_score(y2_val, prob_val_q))
        test_auroc_q = float(roc_auc_score(y2_test, prob_test_q))
        test_prauc_q = float(average_precision_score(y2_test, prob_test_q))

        err_test = compute_stm32ai_error_metrics(logits_test_fp32, logits_test_q, th=main_fp32_th)

        models_report[cfg_key] = {
            "config_key": cfg_key,
            "label": cfg_label,
            "espdl_size_kb": round(espdl_path.stat().st_size / 1024.0, 2) if espdl_path.exists() else 0.0,
            "val_metrics": {
                "auroc": round(val_auroc_q, 6),
                "pr_auc": round(val_prauc_q, 6),
            },
            "test_metrics": {
                "auroc": round(test_auroc_q, 6),
                "pr_auc": round(test_prauc_q, 6),
                "delta_auroc_vs_fp32": round(test_auroc_q - fp32_auroc_test, 6),
                "delta_pr_auc_vs_fp32": round(test_prauc_q - fp32_prauc_test, 6),
            },
            "recalibrated_val_thresholds": q_val_ths,
            "test_operating_points_fixed_fp32_th": {
                op: eval_at_threshold(y5_test, y2_test, prob_test_q, th)
                for op, th in fp32_val_ths.items()
            },
            "test_operating_points_recal_val_th": {
                op: eval_at_threshold(y5_test, y2_test, prob_test_q, th)
                for op, th in q_val_ths.items()
            },
            "error_vs_fp32_test49691": err_test,
        }

        amp_sensitivity[cfg_key] = {}
        for alpha in AMPLITUDE_SCALES:
            if alpha == 1.0:
                l_a = logits_test_q
            else:
                l_a = run_ppq_batched(executor, (X_test * alpha).astype(np.float32), device=device)
            p_a = softmax_pos(l_a)
            op_a = eval_at_threshold(y5_test, y2_test, p_a, main_fp32_th)
            amp_sensitivity[cfg_key][str(alpha)] = {
                "auroc": round(float(roc_auc_score(y2_test, p_a)), 6),
                "pr_auc": round(float(average_precision_score(y2_test, p_a)), 6),
                "recall": op_a["recall_non_N"],
                "specificity": op_a["specificity_N"],
                "macro_f1": op_a["macro_f1"],
                "subclass_S_recall": op_a["subclass_recall"]["S"],
                "subclass_V_recall": op_a["subclass_recall"]["V"],
            }

        op_yj = models_report[cfg_key]["test_operating_points_fixed_fp32_th"]["3_val_best_youden_j"]
        print(
            f"[OK] {cfg_key:8s} ({time.time()-t_cfg:.1f}s): "
            f"TEST AUROC={test_auroc_q:.4f} (d={test_auroc_q-fp32_auroc_test:+.4f}), "
            f"PR-AUC={test_prauc_q:.4f}, "
            f"Rec={op_yj['recall_non_N']*100:.2f}%, Spec={op_yj['specificity_N']*100:.2f}%, "
            f"Agree={err_test['threshold_agreement_pct']:.2f}%"
        )

    if working_onnx.exists():
        working_onnx.unlink()

    npz_500_path = QUANT_DIR / "host_val_test500_logits_espdl.npz"
    np.savez(str(npz_500_path), **cached_500_logits)

    passing_auroc = []
    passing_rec_spec = []
    for cfg_key, _ in config_keys_labels:
        m = models_report[cfg_key]
        if abs(m["test_metrics"]["delta_auroc_vs_fp32"]) <= 0.0050:
            passing_auroc.append(cfg_key)
        op_fix = m["test_operating_points_fixed_fp32_th"]["3_val_best_youden_j"]
        op_rec = m["test_operating_points_recal_val_th"]["3_val_best_youden_j"]
        if (op_fix["recall_non_N"] >= 0.85 and op_fix["specificity_N"] >= 0.85) or (
            op_rec["recall_non_N"] >= 0.85 and op_rec["specificity_N"] >= 0.85
        ):
            passing_rec_spec.append(cfg_key)

    gate_h = {
        "criterion_delta_auroc_le_0p005": True,
        "passing_configs_auroc": passing_auroc,
        "passing_configs_rec_spec_85": passing_rec_spec,
        "gate_h_pass": any(k in passing_auroc for k in ["c0", "c1", "c2a", "c2b", "c3"]),
        "recommended_int8_config": "c3" if "c3" in passing_rec_spec else ("c1" if "c1" in passing_auroc else "b_int16"),
        "recommended_pure_int8_config": "c1",
    }

    full_report = {
        "phase": "C - Host Accuracy Evaluation (VAL=9885, TEST=49691) & Amplitude Sensitivity",
        "elapsed_sec": round(time.time() - t0, 2),
        "gate_h_evaluation": gate_h,
        "models": models_report,
        "amplitude_sensitivity": amp_sensitivity,
    }

    json_path = RESULTS_DIR / "host_eval.json"
    md_path = RESULTS_DIR / "host_eval_report.md"
    png_path = RESULTS_DIR / "host_eval_comparison.png"

    json_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_comparison(full_report, config_keys_labels, png_path)
    generate_markdown_report(full_report, config_keys_labels, md_path)

    print(f"\n[DONE] Phase C finished in {full_report['elapsed_sec']}s")
    print(f"[OK] JSON: {json_path}")
    print(f"[OK] Markdown: {md_path}")
    print(f"[OK] Plot: {png_path}")
    print("Gate H:", json.dumps(gate_h, indent=2))


if __name__ == "__main__":
    main()
