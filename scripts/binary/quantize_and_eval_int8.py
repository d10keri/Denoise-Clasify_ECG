"""
D2 & D3. Lượng tử hóa Post-Training Static Quantization (PTQ INT8 - định dạng QDQ)
bằng `onnxruntime.quantization` với tập hiệu chuẩn phân tầng `export/calib_train.npy` (600 nhịp từ TRAIN),
sau đó đánh giá toàn diện INT8 vs FP32 trên toàn bộ tập VAL (9,885 nhịp) và TEST (49,691 nhịp)
cho cả 2 kiến trúc:
1. `M1-Reg` (`CNN1DBinaryRegularized` - 3 seeds: 42, 43, 44)
2. `M2-B-KD` (`CNN1DBinaryDS Variant B + KD` - 3 seeds: 42, 43, 44)

Báo cáo đầy đủ (theo Bước D3 trong plan/edge_ai_int8_binary_plan.md):
- Sai số tín hiệu đầu ra (Host Validation INT8 vs FP32): L2r, Cosine Similarity (COS), Max/Mean Logit Diff,
  Tỷ lệ dự đoán trùng khớp với FP32 (% Agreement).
- Chỉ số phân loại trên TEST (3 seeds Mean ± Std và từng Seed):
  + Cùng ngưỡng cố định chọn từ FP32 VAL
  + Thử hiệu chỉnh lại ngưỡng trên INT8 VAL
  + Recall(non-N), Specificity(N), F1(non-N), Macro-F1, AUROC, PR-AUC
  + Recall riêng theo từng lớp gốc (S, V, F, Q) để kiểm tra lớp hiếm nào bị ảnh hưởng bởi làm tròn 8-bit.
- Kiểm tra cổng Gate D: Recall >= 0.85, Specificity >= 0.85, và độ sụt giảm so với FP32 <= 1-2 điểm %.
"""
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_cnn1d_binary import (
    eval_at_threshold,
    select_thresholds_on_val,
    summarize_across_seeds,
)
from scripts.binary.verify_onnx import compute_stm32ai_error_metrics

EXPORT_DIR = PROJECT_ROOT / "export"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
RESULTS_INT8_DIR = PROJECT_ROOT / "results/binary/int8_ptq"
SEEDS = [42, 43, 44]


class ECGCalibrationDataReader(CalibrationDataReader):
    """Bộ đọc dữ liệu hiệu chuẩn 600 nhịp (1, 1, 260) cho onnxruntime.quantization."""
    def __init__(self, calib_npy_path: Path, input_name: str = "ecg_input"):
        self.data = np.load(calib_npy_path).astype(np.float32)
        self.input_name = input_name
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: self.data[i : i + 1]} for i in range(len(self.data))]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def quantize_onnx_model(fp32_onnx_path: Path, int8_onnx_path: Path, calib_npy_path: Path) -> dict:
    reader = ECGCalibrationDataReader(calib_npy_path, input_name="ecg_input")
    int8_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # Sử dụng định dạng QDQ chuẩn (tương thích ST Edge AI Core & CMSIS-NN),
    # per-channel symmetric INT8 cho weights (QInt8) và asymmetric INT8/UINT8 cho activations.
    quantize_static(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
        },
    )

    q_model = onnx.load(str(int8_onnx_path))
    ops = [n.op_type for n in q_model.graph.node]
    q_count = sum(1 for op in ops if op == "QuantizeLinear")
    dq_count = sum(1 for op in ops if op == "DequantizeLinear")

    return {
        "fp32_onnx": str(fp32_onnx_path.relative_to(PROJECT_ROOT)),
        "int8_onnx": str(int8_onnx_path.relative_to(PROJECT_ROOT)),
        "fp32_file_kb": round(fp32_onnx_path.stat().st_size / 1024.0, 2),
        "int8_file_kb": round(int8_onnx_path.stat().st_size / 1024.0, 2),
        "quant_format": "QDQ (Per-Channel Symmetric QInt8 Weights, Asymmetric QInt8 Activations)",
        "calibration_method": "MinMax on 600 stratified train beats",
        "num_quantize_nodes": q_count,
        "num_dequantize_nodes": dq_count,
    }


def make_batched_onnx_in_memory(onnx_path: Path) -> bytes:
    """
    Chuyển trục batch (dim 0) của bản copy ONNX trong bộ nhớ thành động ('N')
    để ONNXRuntime có thể suy luận theo lô (batch=512) trên 49,691 nhịp test trong <1 giây,
    trong khi kết quả toán học từng mẫu trùng khít 100% với chạy từng nhịp (1, 1, 260).
    File ONNX trên đĩa (`export/*.onnx`) vẫn giữ nguyên tĩnh 100% `(1, 1, 260)` cho STM32.
    """
    model = onnx.load(str(onnx_path))
    for val_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if val_info.type.HasField("tensor_type"):
            shape = val_info.type.tensor_type.shape
            if len(shape.dim) > 0:
                shape.dim[0].ClearField("dim_value")
                shape.dim[0].dim_param = "batch_size"
    return model.SerializeToString()


def run_ort_fast_batched(onnx_path: Path, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model_bytes = make_batched_onnx_in_memory(onnx_path)
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(model_bytes, sess_options=sess_opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    n = len(X)
    out = np.empty((n, 2), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        out[start:end] = sess.run([out_name], {in_name: X[start:end]})[0]
    return out


def softmax_pos(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / np.sum(e, axis=1)


def evaluate_model_family_int8(
    family_name: str,
    fp32_onnx_pattern: str,
    int8_onnx_pattern: str,
    fp32_metrics_json_path: Path,
    X_val: np.ndarray,
    y5_val: np.ndarray,
    y2_val: np.ndarray,
    X_test: np.ndarray,
    y5_test: np.ndarray,
    y2_test: np.ndarray,
    calib_path: Path,
) -> dict:
    with open(fp32_metrics_json_path, "r", encoding="utf-8") as f:
        fp32_ref_data = json.load(f)

    quant_meta = {}
    host_val_errors = {}
    per_seed_meta = {}
    per_seed_val_fixed_th = {}
    per_seed_test_fixed_th = {}
    per_seed_test_recal_th = {}
    test_probs_int8_all = []
    test_probs_fp32_all = []

    for seed in SEEDS:
        fp32_path = EXPORT_DIR / fp32_onnx_pattern.format(seed=seed)
        int8_path = EXPORT_DIR / int8_onnx_pattern.format(seed=seed)

        q_info = quantize_onnx_model(fp32_path, int8_path, calib_path)
        quant_meta[str(seed)] = q_info

        # Chạy suy luận FP32 và INT8 trên toàn bộ VAL (9,885 nhịp) và TEST (49,691 nhịp)
        logits_val_fp32 = run_ort_fast_batched(fp32_path, X_val)
        logits_val_int8 = run_ort_fast_batched(int8_path, X_val)

        logits_test_fp32 = run_ort_fast_batched(fp32_path, X_test)
        logits_test_int8 = run_ort_fast_batched(int8_path, X_test)

        prob_val_fp32 = softmax_pos(logits_val_fp32)
        prob_val_int8 = softmax_pos(logits_val_int8)
        prob_test_fp32 = softmax_pos(logits_test_fp32)
        prob_test_int8 = softmax_pos(logits_test_int8)

        test_probs_fp32_all.append(prob_test_fp32)
        test_probs_int8_all.append(prob_test_int8)

        # Ngưỡng cố định đã chọn từ FP32 VAL
        fp32_ths = fp32_ref_data["per_seed_metadata"][str(seed)]["selected_thresholds_on_val"]
        # Ngưỡng hiệu chỉnh lại trên INT8 VAL (theo Bước D3)
        int8_recal_ths = select_thresholds_on_val(y2_val, prob_val_int8)

        # Sai số Host Validation (INT8 so với FP32 trên tập TEST 49,691 nhịp)
        main_th = fp32_ths["5_val_equal_rec_spec"]
        err_val = compute_stm32ai_error_metrics(logits_val_fp32, logits_val_int8, th=main_th)
        err_test = compute_stm32ai_error_metrics(logits_test_fp32, logits_test_int8, th=main_th)
        host_val_errors[str(seed)] = {
            "val_9885_beats": err_val,
            "test_49691_beats": err_test,
        }

        val_auroc_int8 = float(roc_auc_score(y2_val, prob_val_int8))
        val_prauc_int8 = float(average_precision_score(y2_val, prob_val_int8))
        test_auroc_int8 = float(roc_auc_score(y2_test, prob_test_int8))
        test_prauc_int8 = float(average_precision_score(y2_test, prob_test_int8))

        fp32_seed_meta = fp32_ref_data["per_seed_metadata"][str(seed)]
        per_seed_meta[seed] = {
            "val_auroc_int8": round(val_auroc_int8, 6),
            "val_pr_auc_int8": round(val_prauc_int8, 6),
            "test_auroc_int8": round(test_auroc_int8, 6),
            "test_pr_auc_int8": round(test_prauc_int8, 6),
            "test_auroc_fp32": fp32_seed_meta["test_auroc"],
            "test_pr_auc_fp32": fp32_seed_meta["test_pr_auc"],
            "delta_test_auroc_pct_pt": round(100.0 * (test_auroc_int8 - fp32_seed_meta["test_auroc"]), 4),
            "delta_test_pr_auc_pct_pt": round(100.0 * (test_prauc_int8 - fp32_seed_meta["test_pr_auc"]), 4),
            "fixed_fp32_thresholds": fp32_ths,
            "recalibrated_int8_thresholds": int8_recal_ths,
        }

        per_seed_val_fixed_th[seed] = {
            op: eval_at_threshold(y5_val, y2_val, prob_val_int8, th)
            for op, th in fp32_ths.items()
        }
        per_seed_test_fixed_th[seed] = {
            op: eval_at_threshold(y5_test, y2_test, prob_test_int8, th)
            for op, th in fp32_ths.items()
        }
        per_seed_test_recal_th[seed] = {
            op: eval_at_threshold(y5_test, y2_test, prob_test_int8, th)
            for op, th in int8_recal_ths.items()
        }

    op_names = list(per_seed_val_fixed_th[SEEDS[0]].keys())
    val_summary_fixed = summarize_across_seeds(per_seed_val_fixed_th, op_names)
    test_summary_fixed = summarize_across_seeds(per_seed_test_fixed_th, op_names)
    test_summary_recal = summarize_across_seeds(per_seed_test_recal_th, op_names)

    threshold_free_int8 = {
        "val_auroc": {
            "mean": round(float(np.mean([per_seed_meta[s]["val_auroc_int8"] for s in SEEDS])), 6),
            "std": round(float(np.std([per_seed_meta[s]["val_auroc_int8"] for s in SEEDS])), 6),
        },
        "val_pr_auc": {
            "mean": round(float(np.mean([per_seed_meta[s]["val_pr_auc_int8"] for s in SEEDS])), 6),
            "std": round(float(np.std([per_seed_meta[s]["val_pr_auc_int8"] for s in SEEDS])), 6),
        },
        "test_auroc": {
            "mean": round(float(np.mean([per_seed_meta[s]["test_auroc_int8"] for s in SEEDS])), 6),
            "std": round(float(np.std([per_seed_meta[s]["test_auroc_int8"] for s in SEEDS])), 6),
        },
        "test_pr_auc": {
            "mean": round(float(np.mean([per_seed_meta[s]["test_pr_auc_int8"] for s in SEEDS])), 6),
            "std": round(float(np.std([per_seed_meta[s]["test_pr_auc_int8"] for s in SEEDS])), 6),
        },
    }

    # Tính độ chênh lệch trực tiếp INT8 vs FP32 trên tập TEST
    fp32_test_sum = fp32_ref_data["test_summary_mean_std"]
    deltas_vs_fp32 = {}
    for op in op_names:
        rec_fp32 = fp32_test_sum[op]["recall_non_N"]["mean"]
        spec_fp32 = fp32_test_sum[op]["specificity_N"]["mean"]
        f1_fp32 = fp32_test_sum[op]["f1_non_N"]["mean"]

        rec_int8 = test_summary_fixed[op]["recall_non_N"]["mean"]
        spec_int8 = test_summary_fixed[op]["specificity_N"]["mean"]
        f1_int8 = test_summary_fixed[op]["f1_non_N"]["mean"]

        sub_deltas = {}
        for c_char in ["S", "V", "F", "Q"]:
            s_fp32 = fp32_test_sum[op]["subclass_recall"][c_char]["mean"]
            s_int8 = test_summary_fixed[op]["subclass_recall"][c_char]["mean"]
            sub_deltas[c_char] = round(100.0 * (s_int8 - s_fp32), 4)

        deltas_vs_fp32[op] = {
            "delta_recall_pct_pt": round(100.0 * (rec_int8 - rec_fp32), 4),
            "delta_specificity_pct_pt": round(100.0 * (spec_int8 - spec_fp32), 4),
            "delta_f1_pct_pt": round(100.0 * (f1_int8 - f1_fp32), 4),
            "delta_subclass_recall_pct_pt": sub_deltas,
        }

    # Kiểm tra Gate D trên Seed 42 (ứng viên chính) và trung bình 3 seeds
    s42_passing_ops = []
    for op in op_names:
        m_int8 = per_seed_test_fixed_th[42][op]
        m_fp32 = fp32_ref_data["per_seed_test"]["42"][op]
        rec_drop = 100.0 * (m_fp32["recall_non_N"] - m_int8["recall_non_N"])
        spec_drop = 100.0 * (m_fp32["specificity_N"] - m_int8["specificity_N"])
        if m_int8["passes_gate_85_85"] and rec_drop <= 2.0 and spec_drop <= 2.0:
            s42_passing_ops.append(
                {
                    "operating_point": op,
                    "threshold": m_int8["threshold"],
                    "recall_int8": m_int8["recall_non_N"],
                    "specificity_int8": m_int8["specificity_N"],
                    "recall_drop_pct_pt": round(rec_drop, 4),
                    "specificity_drop_pct_pt": round(spec_drop, 4),
                }
            )

    return {
        "family_name": family_name,
        "quantization_metadata": quant_meta,
        "host_validation_signal_errors": host_val_errors,
        "threshold_free_int8_summary": threshold_free_int8,
        "threshold_free_fp32_summary": fp32_ref_data["threshold_free_summary"],
        "val_summary_fixed_fp32_th": val_summary_fixed,
        "test_summary_fixed_fp32_th": test_summary_fixed,
        "test_summary_recalibrated_int8_th": test_summary_recal,
        "deltas_int8_minus_fp32_fixed_th": deltas_vs_fp32,
        "per_seed_metadata": {str(k): v for k, v in per_seed_meta.items()},
        "per_seed_test_fixed_th": {str(k): v for k, v in per_seed_test_fixed_th.items()},
        "per_seed_test_recal_th": {str(k): v for k, v in per_seed_test_recal_th.items()},
        "gate_d_seed42_passing_points": s42_passing_ops,
        "passes_gate_d": len(s42_passing_ops) > 0,
    }


def main():
    t0 = time.time()
    RESULTS_INT8_DIR.mkdir(parents=True, exist_ok=True)
    calib_path = EXPORT_DIR / "calib_train.npy"

    val_npz = np.load(PROCESSED_DIR / "val.npz", allow_pickle=False)
    X_val = np.expand_dims(val_npz["X"].astype(np.float32), axis=1)
    y5_val = val_npz["y"].astype(np.int64)
    y2_val = (y5_val > 0).astype(np.int64)

    test_npz = np.load(PROCESSED_DIR / "test.npz", allow_pickle=False)
    X_test = np.expand_dims(test_npz["X"].astype(np.float32), axis=1)
    y5_test = test_npz["y"].astype(np.int64)
    y2_test = (y5_test > 0).astype(np.int64)

    # 1. Đánh giá M1-Reg INT8
    m1_reg_res = evaluate_model_family_int8(
        family_name="M1-Reg (CNN1DBinaryRegularized)",
        fp32_onnx_pattern="cnn1d_binary_reg_seed{seed}_fp32.onnx",
        int8_onnx_pattern="cnn1d_binary_reg_seed{seed}_int8.onnx",
        fp32_metrics_json_path=PROJECT_ROOT / "results/binary/m1_reg_fp32/test_metrics.json",
        X_val=X_val,
        y5_val=y5_val,
        y2_val=y2_val,
        X_test=X_test,
        y5_test=y5_test,
        y2_test=y2_test,
        calib_path=calib_path,
    )
    shutil.copy2(
        EXPORT_DIR / "cnn1d_binary_reg_seed42_int8.onnx",
        EXPORT_DIR / "cnn1d_binary_reg_int8.onnx",
    )
    shutil.copy2(
        EXPORT_DIR / "cnn1d_binary_reg_seed42_int8.onnx",
        EXPORT_DIR / "cnn1d_binary_int8.onnx",
    )

    # 2. Đánh giá M2-B-KD INT8
    m2_kd_res = evaluate_model_family_int8(
        family_name="M2-B-KD (CNN1DBinaryDS Variant B + KD)",
        fp32_onnx_pattern="cnn1d_binary_dsB_kd_seed{seed}_fp32.onnx",
        int8_onnx_pattern="cnn1d_binary_dsB_kd_seed{seed}_int8.onnx",
        fp32_metrics_json_path=PROJECT_ROOT / "results/binary/m2_ds_kd_fp32/B/test_metrics.json",
        X_val=X_val,
        y5_val=y5_val,
        y2_val=y2_val,
        X_test=X_test,
        y5_test=y5_test,
        y2_test=y2_test,
        calib_path=calib_path,
    )
    shutil.copy2(
        EXPORT_DIR / "cnn1d_binary_dsB_kd_seed42_int8.onnx",
        EXPORT_DIR / "cnn1d_binary_dsB_kd_int8.onnx",
    )

    master_out = {
        "task": "Phase D - Post-Training Static INT8 Quantization & Evaluation (M1-Reg & M2-B-KD)",
        "elapsed_seconds": round(time.time() - t0, 2),
        "M1-Reg": m1_reg_res,
        "M2-B-KD": m2_kd_res,
    }

    out_json = RESULTS_INT8_DIR / "test_metrics_int8.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(master_out, f, indent=2, ensure_ascii=False)

    print("=== HOÀN TẤT BƯỚC D2 & D3: LƯỢNG TỬ HÓA INT8 & ĐÁNH GIÁ TRÊN 49,691 NHỊP TEST ===")
    for k, res in [("M1-Reg", m1_reg_res), ("M2-B-KD", m2_kd_res)]:
        tf_fp32 = res["threshold_free_fp32_summary"]
        tf_int8 = res["threshold_free_int8_summary"]
        err42 = res["host_validation_signal_errors"]["42"]["test_49691_beats"]
        print(f"\n[{k}]")
        print(
            f"  - Host Val (Seed 42, 49,691 Test Beats): "
            f"L2r = {err42['stm32ai_l2r']:.4f} | COS = {err42['stm32ai_cosine_similarity']:.6f} | "
            f"Pred Agreement (th) = {err42['threshold_agreement_pct']:.2f}% | "
            f"Pred Agreement (argmax) = {err42['argmax_agreement_pct']:.2f}%"
        )
        print(
            f"  - TEST AUROC (3 seeds): FP32 = {tf_fp32['test_auroc']['mean']:.4f} -> "
            f"INT8 = {tf_int8['test_auroc']['mean']:.4f} ± {tf_int8['test_auroc']['std']:.4f}"
        )
        print(
            f"  - TEST PR-AUC (3 seeds): FP32 = {tf_fp32['test_pr_auc']['mean']:.4f} -> "
            f"INT8 = {tf_int8['test_pr_auc']['mean']:.4f} ± {tf_int8['test_pr_auc']['std']:.4f}"
        )
        print(f"  - Gate D Pass (Seed 42): {res['passes_gate_d']} -> {res['gate_d_seed42_passing_points']}")

    print(f"\nĐã lưu toàn bộ kết quả INT8 tại: {out_json}")


if __name__ == "__main__":
    main()
