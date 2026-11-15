"""
C1 & C3. Kiểm chứng độ tương đương toán học giữa PyTorch gốc (chưa gộp BN)
và đồ thị ONNX tĩnh (đã gộp BN + AvgPool1d tĩnh) chạy trên ONNXRuntime CPU.

Tiêu chí qua cổng (theo plan/edge_ai_int8_binary_plan.md):
- Gate C1: Chạy trên >= 1,000 nhịp test (ở đây kiểm tra trên 2,000 nhịp test phân tầng
  và tập Host Validation 500 nhịp):
  + Sai khác logit tuyệt đối lớn nhất (max |logit_pt - logit_onnx|) <= 1e-4
  + Tỷ lệ trùng nhãn dự đoán = 100.0%
- Gate C3 (Host Validation FP32):
  + Sai số tương đối L2r = ||Y_onnx - Y_pt||_2 / ||Y_pt||_2 << 1e-5
  + Độ tương đồng Cosine (COS) >= 0.999999
  + Lưu đầu ra tham chiếu FP32 (`export/host_val_test500_y_*.npy`) để dùng trong STM32Cube AI Studio.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn1d_binary import CNN1DBinary
from src.models.cnn1d_binary_reg import CNN1DBinaryRegularized
from src.models.cnn1d_binary_ds import CNN1DBinaryDS

CKPT_DIR = PROJECT_ROOT / "checkpoints"
EXPORT_DIR = PROJECT_ROOT / "export"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
RESULTS_DIR = PROJECT_ROOT / "results/binary"


def compute_stm32ai_error_metrics(y_ref: np.ndarray, y_pred: np.ndarray, th: float = 0.50) -> dict:
    diff = y_pred - y_ref
    max_abs_diff = float(np.max(np.abs(diff)))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # L2 relative error (L2r) theo công thức của STM32Cube.AI: ||y_pred - y_ref||_2 / ||y_ref||_2
    l2_num = float(np.linalg.norm(diff.ravel(), ord=2))
    l2_den = float(np.linalg.norm(y_ref.ravel(), ord=2))
    l2r = l2_num / max(l2_den, 1e-12)

    # Cosine similarity (COS)
    dot = float(np.sum(y_ref * y_pred))
    norm_ref = float(np.linalg.norm(y_ref.ravel(), ord=2))
    norm_pred = float(np.linalg.norm(y_pred.ravel(), ord=2))
    cos_sim = dot / max(norm_ref * norm_pred, 1e-12)

    # Xác suất lớp 1 (softmax)
    def _softmax_pos(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits, axis=1, keepdims=True)
        e = np.exp(z)
        return e[:, 1] / np.sum(e, axis=1)

    p_ref = _softmax_pos(y_ref)
    p_pred = _softmax_pos(y_pred)
    max_prob_diff = float(np.max(np.abs(p_pred - p_ref)))

    cls_ref_argmax = np.argmax(y_ref, axis=1)
    cls_pred_argmax = np.argmax(y_pred, axis=1)
    match_argmax_pct = 100.0 * float(np.mean(cls_ref_argmax == cls_pred_argmax))

    cls_ref_th = (p_ref >= th).astype(np.int64)
    cls_pred_th = (p_pred >= th).astype(np.int64)
    match_th_pct = 100.0 * float(np.mean(cls_ref_th == cls_pred_th))

    return {
        "num_samples": int(len(y_ref)),
        "max_abs_logit_diff": max_abs_diff,
        "mean_abs_logit_diff": mae,
        "rmse_logit": rmse,
        "stm32ai_l2r": l2r,
        "stm32ai_cosine_similarity": cos_sim,
        "max_abs_prob_diff": max_prob_diff,
        "argmax_agreement_pct": round(match_argmax_pct, 4),
        "threshold_agreement_pct": round(match_th_pct, 4),
        "passes_gate_c1": bool(max_abs_diff <= 1e-4 and match_argmax_pct == 100.0),
    }


def run_ort_single_beat_loop(onnx_path: Path, X: np.ndarray) -> np.ndarray:
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    outputs = np.empty((len(X), 2), dtype=np.float32)
    for i in range(len(X)):
        sample = X[i : i + 1]  # (1, 1, 260)
        outputs[i] = sess.run([out_name], {in_name: sample})[0][0]
    return outputs


def main():
    test_npz = np.load(PROCESSED_DIR / "test.npz", allow_pickle=False)
    X_test_all = test_npz["X"].astype(np.float32)
    if X_test_all.ndim == 2:
        X_test_all = np.expand_dims(X_test_all, axis=1)
    y5_test_all = test_npz["y"].astype(np.int64)

    # Chọn 2,000 nhịp test phân tầng (1,000 N + 1,000 non-N) để kiểm chứng C1 (>= 1,000 nhịp)
    rng = np.random.default_rng(42)
    idx_n = rng.choice(np.where(y5_test_all == 0)[0], size=1000, replace=False)
    idx_abn = rng.choice(np.where(y5_test_all > 0)[0], size=1000, replace=False)
    idx_2000 = np.concatenate([idx_n, idx_abn])
    rng.shuffle(idx_2000)
    X_verify = X_test_all[idx_2000]

    # Nạp tập Host Validation 500 nhịp (C3)
    X_hval = np.load(EXPORT_DIR / "host_val_test500_x.npy")

    models_to_verify = []
    for s in [42, 43, 44]:
        ckpt = torch.load(CKPT_DIR / f"cnn1d_binary_reg_best_seed{s}.pt", map_location="cpu", weights_only=False)
        m = CNN1DBinaryRegularized(num_classes=2)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models_to_verify.append(
            (f"M1-Reg_seed{s}", m, EXPORT_DIR / f"cnn1d_binary_reg_seed{s}_fp32.onnx", 0.49 if s == 42 else 0.50)
        )

    for s in [42, 43, 44]:
        ckpt = torch.load(CKPT_DIR / f"cnn1d_binary_dsB_kd_best_seed{s}.pt", map_location="cpu", weights_only=False)
        m = CNN1DBinaryDS(variant="B", num_classes=2)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models_to_verify.append(
            (f"M2-B-KD_seed{s}", m, EXPORT_DIR / f"cnn1d_binary_dsB_kd_seed{s}_fp32.onnx", 0.78 if s == 42 else 0.50)
        )

    ckpt_m1 = torch.load(CKPT_DIR / "cnn1d_binary_best_seed42.pt", map_location="cpu", weights_only=False)
    m_base = CNN1DBinary(num_classes=2)
    m_base.load_state_dict(ckpt_m1["model_state_dict"])
    m_base.eval()
    models_to_verify.append(
        ("M1-Base_seed42", m_base, EXPORT_DIR / "cnn1d_binary_m1_base_fp32.onnx", 0.50)
    )

    report = {}
    print("=== KẾT QUẢ KIỂM CHỨNG ONNX FP32 vs PYTORCH GỐC (BƯỚC C1 & C3) ===")
    for name, pt_model, onnx_path, th in models_to_verify:
        with torch.no_grad():
            y_pt_2000 = pt_model(torch.from_numpy(X_verify)).numpy()
            y_pt_hval = pt_model(torch.from_numpy(X_hval)).numpy()

        y_ort_2000 = run_ort_single_beat_loop(onnx_path, X_verify)
        y_ort_hval = run_ort_single_beat_loop(onnx_path, X_hval)

        m_2000 = compute_stm32ai_error_metrics(y_pt_2000, y_ort_2000, th=th)
        m_hval = compute_stm32ai_error_metrics(y_pt_hval, y_ort_hval, th=th)

        if name == "M1-Reg_seed42":
            np.save(EXPORT_DIR / "host_val_test500_y_m1reg.npy", y_ort_hval)
        elif name == "M2-B-KD_seed42":
            np.save(EXPORT_DIR / "host_val_test500_y_m2bkd.npy", y_ort_hval)

        report[name] = {
            "onnx_file": str(onnx_path.relative_to(PROJECT_ROOT)),
            "verify_2000_test_beats": m_2000,
            "host_val_500_beats": m_hval,
        }
        print(
            f"[{name:16s}] 2,000 Test Beats | "
            f"MaxLogitDiff: {m_2000['max_abs_logit_diff']:.2e} | "
            f"L2r: {m_2000['stm32ai_l2r']:.2e} | "
            f"COS: {m_2000['stm32ai_cosine_similarity']:.8f} | "
            f"Match: {m_2000['argmax_agreement_pct']:.2f}% | "
            f"Gate C1: {'PASS' if m_2000['passes_gate_c1'] else 'FAIL'}"
        )

    out_path = RESULTS_DIR / "onnx_verification_c1_c3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu báo cáo kiểm chứng C1 & C3 tại: {out_path}")


if __name__ == "__main__":
    main()
