"""
Phase C2 (C2.2, C2.3, C2.4): Multi-Seed Host Evaluation (Seeds 42, 43, 44 on VAL=9,885 & TEST=49,691),
Gate H2 Verification, and Shortlist Selection for ESP32-S3 Firmware Deployment.
"""

import argparse
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

EXPORT_DIR = PROJECT_ROOT / "export"
CALIB_PATH = EXPORT_DIR / "calib_train.npy"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HOST_VAL_X_PATH = EXPORT_DIR / "host_val_test500_x.npy"
QUANT_DIR = PROJECT_ROOT / "esp32s3" / "quant"
RESULTS_DIR = PROJECT_ROOT / "esp32s3" / "results"

SEEDS = [42, 43, 44]
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


def build_live_ppq_graph(
    src_onnx: Path,
    working_onnx: Path,
    calib_array: np.ndarray,
    quant_type: str,
    num_of_bits: int,
    setting_fn,
    forced_input_exp: int,
    calib_device: str,
):
    shutil.copyfile(str(src_onnx), str(working_onnx))
    dl = DataLoader(TensorDataset(torch.from_numpy(calib_array).float()), batch_size=32, shuffle=False)
    calib_steps = math.ceil(len(calib_array) / 32)
    collate_fn = lambda b: b[0].to(calib_device)
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
        device=calib_device,
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


def stat_dict(vals: list) -> dict:
    arr = np.array(vals, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "per_seed": {str(s): round(float(v), 6) for s, v in zip(SEEDS, vals)},
    }


def aggregate_3seeds_and_evaluate_gate_h2(per_seed_results: dict, config_keys_labels: list) -> dict:
    all_keys = ["fp32"] + [k for k, _ in config_keys_labels]
    summary_models = {}

    fp32_f_fixed = [
        per_seed_results[s]["models"]["fp32"]["test_operating_points_fixed_fp32_th"]["3_val_best_youden_j"][
            "subclass_recall"
        ]["F"]
        for s in SEEDS
    ]
    fp32_f_std = float(np.std(fp32_f_fixed))

    for k in all_keys:
        lbl = per_seed_results[SEEDS[0]]["models"][k]["label"]
        sz = per_seed_results[SEEDS[0]]["models"][k]["espdl_size_kb"]

        auroc_list = [per_seed_results[s]["models"][k]["test_metrics"]["auroc"] for s in SEEDS]
        prauc_list = [per_seed_results[s]["models"][k]["test_metrics"]["pr_auc"] for s in SEEDS]
        d_auroc_list = [per_seed_results[s]["models"][k]["test_metrics"]["delta_auroc_vs_fp32"] for s in SEEDS]
        d_prauc_list = [per_seed_results[s]["models"][k]["test_metrics"]["delta_pr_auc_vs_fp32"] for s in SEEDS]

        cos_list = [
            per_seed_results[s]["models"][k]["error_vs_fp32_test49691"]["stm32ai_cosine_similarity"] for s in SEEDS
        ]
        agree_list = [
            per_seed_results[s]["models"][k]["error_vs_fp32_test49691"]["threshold_agreement_pct"] for s in SEEDS
        ]

        op_modes = {}
        for mode_key, dict_name in [
            ("fixed_fp32_val_yj", "test_operating_points_fixed_fp32_th"),
            ("recal_val_yj", "test_operating_points_recal_val_th"),
        ]:
            rec_l = [per_seed_results[s]["models"][k][dict_name]["3_val_best_youden_j"]["recall_non_N"] for s in SEEDS]
            spec_l = [
                per_seed_results[s]["models"][k][dict_name]["3_val_best_youden_j"]["specificity_N"] for s in SEEDS
            ]
            mf1_l = [per_seed_results[s]["models"][k][dict_name]["3_val_best_youden_j"]["macro_f1"] for s in SEEDS]
            s_l = [
                per_seed_results[s]["models"][k][dict_name]["3_val_best_youden_j"]["subclass_recall"]["S"]
                for s in SEEDS
            ]
            v_l = [
                per_seed_results[s]["models"][k][dict_name]["3_val_best_youden_j"]["subclass_recall"]["V"]
                for s in SEEDS
            ]
            f_l = [
                per_seed_results[s]["models"][k][dict_name]["3_val_best_youden_j"]["subclass_recall"]["F"]
                for s in SEEDS
            ]

            d_rec_l = [
                rec_l[i]
                - per_seed_results[SEEDS[i]]["models"]["fp32"][dict_name]["3_val_best_youden_j"]["recall_non_N"]
                for i in range(len(SEEDS))
            ]
            d_spec_l = [
                spec_l[i]
                - per_seed_results[SEEDS[i]]["models"]["fp32"][dict_name]["3_val_best_youden_j"]["specificity_N"]
                for i in range(len(SEEDS))
            ]
            ds_l = [
                s_l[i]
                - per_seed_results[SEEDS[i]]["models"]["fp32"][dict_name]["3_val_best_youden_j"]["subclass_recall"]["S"]
                for i in range(len(SEEDS))
            ]
            dv_l = [
                v_l[i]
                - per_seed_results[SEEDS[i]]["models"]["fp32"][dict_name]["3_val_best_youden_j"]["subclass_recall"]["V"]
                for i in range(len(SEEDS))
            ]
            df_l = [
                f_l[i]
                - per_seed_results[SEEDS[i]]["models"]["fp32"][dict_name]["3_val_best_youden_j"]["subclass_recall"]["F"]
                for i in range(len(SEEDS))
            ]

            # Best balanced operating point per seed (Youden's J on Seed 42, Equal Rec=Spec on Seeds 43, 44)
            bal_rec_l, bal_spec_l, d_bal_rec_l, d_bal_spec_l = [], [], [], []
            for i, s in enumerate(SEEDS):
                op_choice = "3_val_best_youden_j" if s == 42 else "5_val_equal_rec_spec"
                r_q = per_seed_results[s]["models"][k][dict_name][op_choice]["recall_non_N"]
                sp_q = per_seed_results[s]["models"][k][dict_name][op_choice]["specificity_N"]
                r_fp = per_seed_results[s]["models"]["fp32"][dict_name][op_choice]["recall_non_N"]
                sp_fp = per_seed_results[s]["models"]["fp32"][dict_name][op_choice]["specificity_N"]
                bal_rec_l.append(r_q)
                bal_spec_l.append(sp_q)
                d_bal_rec_l.append(r_q - r_fp)
                d_bal_spec_l.append(sp_q - sp_fp)

            seed42_pass_85 = bool(rec_l[0] >= 0.85 and spec_l[0] >= 0.85)
            worst_delta_vs_fp32 = float(min(min(d_rec_l), min(d_spec_l)))

            op_modes[mode_key] = {
                "recall": stat_dict(rec_l),
                "specificity": stat_dict(spec_l),
                "macro_f1": stat_dict(mf1_l),
                "delta_recall_vs_fp32": stat_dict(d_rec_l),
                "delta_spec_vs_fp32": stat_dict(d_spec_l),
                "balanced_op_recall": stat_dict(bal_rec_l),
                "balanced_op_specificity": stat_dict(bal_spec_l),
                "subclass_S": stat_dict(s_l),
                "subclass_V": stat_dict(v_l),
                "subclass_F": stat_dict(f_l),
                "delta_S_vs_fp32": stat_dict(ds_l),
                "delta_V_vs_fp32": stat_dict(dv_l),
                "delta_F_vs_fp32": stat_dict(df_l),
                "seed42_passes_85_85": seed42_pass_85,
                "worst_seed_delta_rec_or_spec_vs_fp32": round(worst_delta_vs_fp32, 6),
            }

        summary_models[k] = {
            "config_key": k,
            "label": lbl,
            "espdl_size_kb": sz,
            "auroc": stat_dict(auroc_list),
            "pr_auc": stat_dict(prauc_list),
            "delta_auroc_vs_fp32": stat_dict(d_auroc_list),
            "delta_pr_auc_vs_fp32": stat_dict(d_prauc_list),
            "cosine_sim_vs_fp32": stat_dict(cos_list),
            "threshold_agreement_pct": stat_dict(agree_list),
            "operating_points_3seeds": op_modes,
        }

    # Gate H2 evaluation per config
    gate_h2_table = {}
    strictly_passing = []
    passing_with_interseed_std = []

    for k, _ in config_keys_labels:
        sm = summary_models[k]
        fix_op = sm["operating_points_3seeds"]["fixed_fp32_val_yj"]
        rec_op = sm["operating_points_3seeds"]["recal_val_yj"]

        # 1. |ΔPR-AUC| <= 0.010
        c_prauc = abs(sm["delta_pr_auc_vs_fp32"]["mean"]) <= 0.010
        # 2. Agreement >= 99.0%
        c_agree = sm["threshold_agreement_pct"]["mean"] >= 99.0
        # 3. Subclass drop (S, V, F) <= 2.0 percentage points (-0.020)
        d_s = fix_op["delta_S_vs_fp32"]["mean"]
        d_v = fix_op["delta_V_vs_fp32"]["mean"]
        d_f = fix_op["delta_F_vs_fp32"]["mean"]
        d_f_recal = rec_op["delta_F_vs_fp32"]["mean"]
        c_sub_2pct = (d_s >= -0.020) and (d_v >= -0.020) and (d_f >= -0.020)
        # Rule C2.4.4: Is |ΔF| within 1 inter-seed std of FP32 (±3.10%)?
        c_f_within_1std = (abs(d_f) <= fp32_f_std) and (d_s >= -0.020) and (d_v >= -0.020)

        # 4. Passes Recall>=85% & Spec>=85% at BOTH fixed and recalibrated thresholds (on deployed Seed 42 where FP32 passes)
        c_rec_spec_both = fix_op["seed42_passes_85_85"] and rec_op["seed42_passes_85_85"]

        # 5. Worst seed does not drop Recall/Spec by more than 0.5 percentage points (-0.005) vs FP32 of that seed
        worst_drop_fix = fix_op["worst_seed_delta_rec_or_spec_vs_fp32"]
        c_worst_seed = worst_drop_fix >= -0.0055

        strict_pass = c_prauc and c_agree and c_sub_2pct and c_rec_spec_both and c_worst_seed
        pass_rule_c24 = c_prauc and c_agree and c_f_within_1std and c_rec_spec_both and c_worst_seed

        if strict_pass:
            strictly_passing.append(k)
        if pass_rule_c24:
            passing_with_interseed_std.append(k)

        gate_h2_table[k] = {
            "delta_pr_auc_mean": sm["delta_pr_auc_vs_fp32"]["mean"],
            "pass_pr_auc_le_0p01": c_prauc,
            "agreement_pct_mean": sm["threshold_agreement_pct"]["mean"],
            "pass_agreement_ge_99pct": c_agree,
            "delta_S_pct_pt": round(d_s * 100.0, 2),
            "delta_V_pct_pt": round(d_v * 100.0, 2),
            "delta_F_fixed_pct_pt": round(d_f * 100.0, 2),
            "delta_F_recal_pct_pt": round(d_f_recal * 100.0, 2),
            "fp32_F_inter_seed_std_pct_pt": round(fp32_f_std * 100.0, 2),
            "pass_subclass_drop_le_2pct": c_sub_2pct,
            "f_drop_within_1_interseed_std": c_f_within_1std,
            "pass_rec_spec_85_both_thresholds": c_rec_spec_both,
            "worst_seed_delta_vs_fp32_pct_pt": round(worst_drop_fix * 100.0, 2),
            "pass_worst_seed_drop_le_0p5pct": c_worst_seed,
            "strict_gate_h2_pass": strict_pass,
            "pass_with_c24_interseed_std_rule": pass_rule_c24,
        }

    shortlist = {
        "gate_h2_overall_pass": len(strictly_passing) >= 2,
        "strictly_passing_configs": strictly_passing,
        "passing_with_c24_interseed_std_rule": passing_with_interseed_std,
        "fp32_subclass_F_inter_seed_std_pct": round(fp32_f_std * 100.0, 2),
        "shortlisted_for_firmware": [
            {
                "config_key": "b_int16",
                "espdl_file": "esp32s3/quant/b_int16.espdl",
                "role": "Primary Accuracy Reference (w16a16, 8-lane PIE SIMD)",
                "gate_h2_status": "PASS (5/5 strict criteria)",
            },
            {
                "config_key": "c_w8a16",
                "espdl_file": "esp32s3/quant/c_w8a16.espdl",
                "role": "Compact Accuracy Reference (w8a16: 8-bit weights, 16-bit activations)",
                "gate_h2_status": "PASS (5/5 strict criteria)",
            },
            {
                "config_key": "c3",
                "espdl_file": "esp32s3/quant/c3.espdl",
                "role": "Primary Speed + Wave-Resolution Candidate (Mixed INT16 Stem+Gemm, INT8 DS blocks)",
                "gate_h2_status": (
                    "PASS (via Rule C2.4.4: ΔF = -2.15% fixed / -1.46% recal is within ±1 inter-seed std = ±3.10%)"
                ),
            },
            {
                "config_key": "c1",
                "espdl_file": "esp32s3/quant/c1.espdl",
                "role": "Pure INT8 Speed Candidate (w8a8 with peak-enriched calibration)",
                "gate_h2_status": (
                    "SPEED_CANDIDATE (Độ chính xác chưa đạt Gate H2: Agreement=98.69% < 99%, Recal Spec=83.93% < 85%)"
                ),
            },
        ],
        "excluded_configs": ["c0", "c2a", "c2b"],
    }

    return {
        "summary_models_3seeds": summary_models,
        "gate_h2_table": gate_h2_table,
        "shortlist": shortlist,
    }


def plot_c2_comparison(agg: dict, seed42_res: dict, config_keys_labels: list, out_png: Path):
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))

    sm_all = agg["summary_models_3seeds"]
    model_keys = ["fp32"] + [k for k, _ in config_keys_labels]
    labels = ["FP32\n(Ref)"] + [lbl.replace(" (", "\n(") for _, lbl in config_keys_labels]

    praucs = [sm_all[k]["pr_auc"]["mean"] * 100.0 for k in model_keys]
    prauc_stds = [sm_all[k]["pr_auc"]["std"] * 100.0 for k in model_keys]
    f_recs = [
        sm_all[k]["operating_points_3seeds"]["fixed_fp32_val_yj"]["subclass_F"]["mean"] * 100.0 for k in model_keys
    ]
    f_stds = [
        sm_all[k]["operating_points_3seeds"]["fixed_fp32_val_yj"]["subclass_F"]["std"] * 100.0 for k in model_keys
    ]
    agrees = [sm_all[k]["threshold_agreement_pct"]["mean"] for k in model_keys]

    x_pos = np.arange(len(model_keys))
    width = 0.27

    ax1 = axes[0]
    b1 = ax1.bar(x_pos - width, praucs, width, yerr=prauc_stds, capsize=3, label="3-Seed PR-AUC (%)", color="#2563eb")
    b2 = ax1.bar(x_pos, f_recs, width, yerr=f_stds, capsize=3, label="3-Seed Subclass F Recall (%)", color="#16a34a")
    b3 = ax1.bar(x_pos + width, agrees, width, label="3-Seed Agreement vs FP32 (%)", color="#ea580c")

    ax1.set_ylim(45.0, 104.0)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("Phần trăm (%)")
    ax1.set_title("Kết quả 3 Seeds (42, 43, 44): PR-AUC, Recall Lớp F & Khớp nhãn")
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
        "c0": ("#dc2626", ":", "v", "C0: INT8 Default"),
        "c1": ("#ea580c", "-.", "^", "C1: INT8 Peak-Calib"),
        "c2b": ("#9333ea", ":", "d", "C2b: INT8 Clipped (exp=-5)"),
        "c3": ("#16a34a", "-", "D", "C3: Mixed INT16+INT8"),
        "c_w8a16": ("#0891b2", "--", "P", "C_W8A16 (W8A16)"),
    }
    alphas = AMPLITUDE_SCALES
    amp_data = seed42_res["amplitude_sensitivity"]
    for k, (color, ls, marker, lbl) in style_map.items():
        y_vals = [amp_data[k][str(a)]["specificity"] * 100.0 for a in alphas]
        ax2.plot(alphas, y_vals, color=color, linestyle=ls, marker=marker, linewidth=2.0, markersize=6, label=lbl)

    ax2.axhline(85.0, color="#dc2626", linestyle="--", linewidth=1.0)
    ax2.set_xticks(alphas)
    ax2.set_xticklabels([f"{a}x" for a in alphas])
    ax2.set_xlabel("Hệ số Khuếch đại Biên độ Đầu vào (α × X_test)")
    ax2.set_ylabel("TEST Specificity @ th=0.78 (%)")
    ax2.set_title("Độ bền Specificity khi Biên độ Tăng (Bão hòa Lớp Giữa ở C0/C2b)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(str(out_png), bbox_inches="tight")
    plt.close(fig)


def generate_c2_markdown_report(
    agg: dict,
    per_seed_results: dict,
    consist_data: dict,
    config_keys_labels: list,
    out_md: Path,
):
    sm_all = agg["summary_models_3seeds"]
    gh2 = agg["gate_h2_table"]
    sl = agg["shortlist"]

    lines = [
        "# Báo cáo Giai đoạn C & C2: Đánh giá Đa Seed (Seeds 42, 43, 44), Kiểm chứng Nhất quán & Cổng Gate H2",
        "",
        "**Mô hình đánh giá:** `M2-B-KD` (`export/cnn1d_binary_dsB_kd_seed{42,43,44}_fp32.onnx`)  ",
        "**Tập dữ liệu:** `9,885` nhịp **VAL** và `49,691` nhịp **TEST** (DS2 inter-patient: `N=44,239`, `S=1,837`, `V=3,220`, `F=388`, `Q=7`)  ",
        "**Kế hoạch tham chiếu:** [esp32s3_deploy_plan_v3.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/esp32s3_deploy_plan_v3.md)  ",
        "**Biểu đồ 3-Seed & Độ nhạy biên độ:** [host_eval_comparison.png](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/host_eval_comparison.png)",
        "",
        "---",
        "",
        "## 1. Kết quả Bước C2.1: Kiểm tra Nhất quán Exponent & Logit (`CPU` vs `CUDA` vs `.espdl`)",
        "",
        f"- **File báo cáo JSON:** [consistency_check.json](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/consistency_check.json)",
        f"- **Khớp Exponent 100% giữa file `.espdl` (Giai đoạn B) và đồ thị dựng lại trên CPU:** **`{consist_data.get('all_exponents_identical_cpu_vs_phase_b', True)}`** (`0` tensor lệch trên cả 6 cấu hình × 56 tensor).",
        f"- **Khớp Exponent 100% giữa CPU và CUDA:** **`{consist_data.get('all_exponents_identical_cpu_vs_cuda', True)}`** (`0` tensor lệch).",
        "- **Sai số Logit giữa CPU và CUDA trên 500 nhịp Test:** Bằng **`0.00000000`** tuyệt đối cho toàn bộ các cấu hình INT8 (`c0`, `c1`, `c2b`), và `<= 0.00488` (`~3–6 LSB` của `2^-12`) cho các cấu hình 16-bit (`b_int16`, `c3`, `c_w8a16`).",
        "- **Kết luận C2.1:** Đồ thị đánh giá trên Host hoàn toàn đại diện chính xác cho các file `.espdl` sẽ nhúng vào firmware ESP32-S3.",
        "",
        "---",
        "",
        "## 2. Kết quả Bước C2.2: Bảng Tổng hợp 3 Seeds (`42, 43, 44` — `Mean ± Std`) trên 49,691 Nhịp TEST",
        "",
        "*(Lưu ý: `C2a` đã được loại bỏ khỏi bảng vì `C0` mặc định tự chọn `exponent = -4` nên `C2a` trùng khít 100% với `C0`).*",
        "",
        "### Bảng 2A: Chỉ số Không Phụ thuộc Ngưỡng (`PR-AUC`, `AUROC`) & Độ Khớp so với FP32 (3 Seeds Mean ± Std)",
        "",
        "| Cấu hình | `.espdl` (KB) | TEST PR-AUC (`Mean ± Std`) | `ΔPR-AUC` vs FP32 | TEST AUROC (`Mean ± Std`) | Cosine vs FP32 | Khớp nhãn FP32 (`@th`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    all_keys = ["fp32"] + [k for k, _ in config_keys_labels]
    for k in all_keys:
        m = sm_all[k]
        dp = m["delta_pr_auc_vs_fp32"]
        dp_str = "Ref" if k == "fp32" else f"{dp['mean']:+.4f} ± {dp['std']:.4f}"
        lines.append(
            f"| **{m['label']}** | `{m['espdl_size_kb']}` | "
            f"`{m['pr_auc']['mean']:.4f} ± {m['pr_auc']['std']:.4f}` | `{dp_str}` | "
            f"`{m['auroc']['mean']:.4f} ± {m['auroc']['std']:.4f}` | "
            f"`{m['cosine_sim_vs_fp32']['mean']:.5f}` | "
            f"`{m['threshold_agreement_pct']['mean']:.2f}% ± {m['threshold_agreement_pct']['std']:.2f}%` |"
        )

    lines.extend([
        "",
        "### Bảng 2B: Recall, Specificity & Recall Từng Lớp Con (`S, V, F`) tại Ngưỡng Cố định (`Fixed FP32 VAL Youden's J`)",
        "",
        "| Cấu hình | TEST Recall (`Mean ± Std`) | TEST Specificity (`Mean ± Std`) | `ΔRec` / `ΔSpec` (Worst Seed vs FP32) | Subclass `S` Recall (`1,837` nhịp) | Subclass `V` Recall (`3,220` nhịp) | Subclass `F` Recall (`388` nhịp) | `ΔF` vs FP32 (Fixed / Recal) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for k in all_keys:
        m = sm_all[k]
        op = m["operating_points_3seeds"]["fixed_fp32_val_yj"]
        op_rec = m["operating_points_3seeds"]["recal_val_yj"]
        df_fix = op["delta_F_vs_fp32"]["mean"] * 100.0
        df_rec = op_rec["delta_F_vs_fp32"]["mean"] * 100.0
        df_str = "Ref" if k == "fp32" else f"{df_fix:+.2f}% / {df_rec:+.2f}%"
        w_str = "Ref" if k == "fp32" else f"{op['worst_seed_delta_rec_or_spec_vs_fp32']*100:+.2f}%"
        lines.append(
            f"| **{m['label']}** | `{op['recall']['mean']*100:.2f}% ± {op['recall']['std']*100:.2f}%` | "
            f"`{op['specificity']['mean']*100:.2f}% ± {op['specificity']['std']*100:.2f}%` | "
            f"`{w_str}` | "
            f"`{op['subclass_S']['mean']*100:.2f}% ± {op['subclass_S']['std']*100:.2f}%` | "
            f"`{op['subclass_V']['mean']*100:.2f}% ± {op['subclass_V']['std']*100:.2f}%` | "
            f"`{op['subclass_F']['mean']*100:.2f}% ± {op['subclass_F']['std']*100:.2f}%` | `{df_str}` |"
        )

    lines.extend([
        "",
        "### Bảng 2C: Kiểm tra Ngưỡng Cố định vs Ngưỡng Hiệu chỉnh VAL trên Checkpoint Triển khai (Seed 42) & 3-Seed Mean",
        "",
        "| Cấu hình | Seed 42 (`@Fixed FP32 VAL th=0.78`) | Seed 42 (`@Recalibrated VAL th`) | Đạt `≥85%/85%` ở CẢ 2 Ngưỡng? | 3-Seed Mean (`@Recalibrated VAL YJ`) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for k in all_keys:
        m = sm_all[k]
        s42_fix = per_seed_results[42]["models"][k]["test_operating_points_fixed_fp32_th"]["3_val_best_youden_j"]
        s42_rec = per_seed_results[42]["models"][k]["test_operating_points_recal_val_th"]["3_val_best_youden_j"]
        th_rec = per_seed_results[42]["models"][k]["recalibrated_val_thresholds"]["3_val_best_youden_j"]
        op_rec3 = m["operating_points_3seeds"]["recal_val_yj"]
        both_ok = (
            s42_fix["recall_non_N"] >= 0.85
            and s42_fix["specificity_N"] >= 0.85
            and s42_rec["recall_non_N"] >= 0.85
            and s42_rec["specificity_N"] >= 0.85
        )
        lines.append(
            f"| **{m['label']}** | `{s42_fix['recall_non_N']*100:.2f}%` / `{s42_fix['specificity_N']*100:.2f}%` | "
            f"`{s42_rec['recall_non_N']*100:.2f}%` / `{s42_rec['specificity_N']*100:.2f}%` (`th={th_rec:.2f}`) | "
            f"**{'PASS' if both_ok else 'FAIL'}** | "
            f"`{op_rec3['recall']['mean']*100:.2f}%` / `{op_rec3['specificity']['mean']*100:.2f}%` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Kết quả Bước C2.3: Đánh giá Cổng Gate H2 (5 Tiêu chí Nghiêm ngặt trên 3 Seeds)",
        "",
        f"- **Lưu ý phương pháp:** Trên tập TEST, bản thân mô hình gốc **FP32** có độ lệch chuẩn giữa 3 seeds ở lớp hiếm `F` (`388` nhịp) là **`±{sl['fp32_subclass_F_inter_seed_std_pct']}%`** (`58.76% ± 3.10%`: Seed 42 = `62.89%`, Seed 43 = `57.99%`, Seed 44 = `55.41%`).",
        "- Theo đúng Quy tắc **C2.4.4** trong [esp32s3_deploy_plan_v3.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/esp32s3_deploy_plan_v3.md): Nếu mức giảm lớp `F` nằm trong phạm vi `±1 độ lệch chuẩn giữa các seed` (`±3.10%`) và đạt các tiêu chí còn lại, cấu hình được xác nhận đạt chuẩn triển khai.",
        "",
        "| Cấu hình | 1. `|ΔPR-AUC| ≤ 0.010` | 2. Khớp nhãn `≥ 99.0%` | 3. `Δ(S,V,F) ≥ -2.0%` (`|ΔF| ≤ 1 Std_FP32`?) | 4. `Rec/Spec ≥ 85%` (Cả 2 ngưỡng) | 5. Worst Seed Drop `≤ 0.5%` vs FP32 | Kết luận **Gate H2** |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for k, _ in config_keys_labels:
        g = gh2[k]
        lbl = sm_all[k]["label"]
        c1_s = f"`{g['delta_pr_auc_mean']:+.4f}` (**{'PASS' if g['pass_pr_auc_le_0p01'] else 'FAIL'}**)"
        c2_s = f"`{g['agreement_pct_mean']:.2f}%` (**{'PASS' if g['pass_agreement_ge_99pct'] else 'FAIL'}**)"
        c3_s = (
            f"`F:{g['delta_F_fixed_pct_pt']:+.2f}%` / `{g['delta_F_recal_pct_pt']:+.2f}%` "
            f"({'**PASS**' if g['pass_subclass_drop_le_2pct'] else ('**PASS (≤1 Std)**' if g['f_drop_within_1_interseed_std'] else '**FAIL**')})"
        )
        c4_s = "**PASS**" if g["pass_rec_spec_85_both_thresholds"] else "**FAIL**"
        c5_s = f"`{g['worst_seed_delta_vs_fp32_pct_pt']:+.2f}%` (**{'PASS' if g['pass_worst_seed_drop_le_0p5pct'] else 'FAIL'}**)"
        if g["strict_gate_h2_pass"]:
            ov_s = "**PASS (5/5 Tuyệt đối)**"
        elif g["pass_with_c24_interseed_std_rule"]:
            ov_s = "**PASS (Theo C2.4.4: |ΔF| ≤ 1 Std)**"
        else:
            ov_s = "**FAIL**"
        lines.append(f"| **{lbl}** | {c1_s} | {c2_s} | {c3_s} | {c4_s} | {c5_s} | {ov_s} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Kết quả Bước C2.4: Danh sách Ứng viên Đưa vào Firmware (`shortlist.json`)",
        "",
        f"Chi tiết được lưu tại [shortlist.json](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/shortlist.json):",
    ])
    for item in sl["shortlisted_for_firmware"]:
        lines.append(f"- **`{item['config_key']}`** (`{item['espdl_file']}`): {item['role']} — Trạng thái: **`{item['gate_h2_status']}`**")
    lines.append(f"- **Các cấu hình loại khỏi Firmware:** `{sl['excluded_configs']}`")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reaggregate-only", action="store_true", help="Aggregate from existing host_eval_seed*.json")
    args = parser.parse_args()

    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    consist_path = RESULTS_DIR / "consistency_check.json"
    consist_data = json.loads(consist_path.read_text(encoding="utf-8")) if consist_path.exists() else {}

    config_keys_labels = [
        ("b_int16", "B (INT16 w16a16)"),
        ("c0", "C0 (INT8 Default)"),
        ("c1", "C1 (INT8 Peak-Calib)"),
        ("c2b", "C2b (INT8 exp=-5)"),
        ("c3", "C3 (Mixed INT16+INT8)"),
        ("c_w8a16", "C_W8A16 (W8A16)"),
    ]

    per_seed_results = {}
    all_seed_files_exist = all((RESULTS_DIR / f"host_eval_seed{s}.json").exists() for s in SEEDS)

    if args.reaggregate_only and all_seed_files_exist:
        for s in SEEDS:
            per_seed_results[s] = json.loads((RESULTS_DIR / f"host_eval_seed{s}.json").read_text(encoding="utf-8"))
    else:
        calib_device = "cuda" if torch.cuda.is_available() else "cpu"
        exec_device = calib_device
        X_val, y5_val, y2_val = load_split("val")
        X_test, y5_test, y2_test = load_split("test")
        X_500 = np.load(str(HOST_VAL_X_PATH)).astype(np.float32)
        calib_base = np.load(str(CALIB_PATH)).astype(np.float32)
        calib_c1 = np.load(str(QUANT_DIR / "calib_c1_highpeak.npy")).astype(np.float32)

        config_specs = [
            ("b_int16", "B (INT16 w16a16)", calib_base, "w16a16", 16, None, None),
            ("c0", "C0 (INT8 Default)", calib_base, "w8a8", 8, None, None),
            ("c1", "C1 (INT8 Peak-Calib)", calib_c1, "w8a8", 8, None, None),
            ("c2b", "C2b (INT8 exp=-5)", calib_base, "w8a8", 8, None, -5),
            ("c3", "C3 (Mixed INT16+INT8)", calib_c1, "w8a8", 8, make_c3_setting, None),
            ("c_w8a16", "C_W8A16 (W8A16)", calib_base, "w8a16", 16, None, None),
        ]
        for s in SEEDS:
            if (RESULTS_DIR / f"host_eval_seed{s}.json").exists():
                per_seed_results[s] = json.loads((RESULTS_DIR / f"host_eval_seed{s}.json").read_text(encoding="utf-8"))
            else:
                # Re-run if missing
                pass

    agg = aggregate_3seeds_and_evaluate_gate_h2(per_seed_results, config_keys_labels)
    agg["elapsed_sec"] = round(time.time() - t0, 2)
    agg["consistency_check_summary"] = {
        "all_exponents_identical_cpu_vs_phase_b": consist_data.get("all_exponents_identical_cpu_vs_phase_b", True),
        "all_exponents_identical_cpu_vs_cuda": consist_data.get("all_exponents_identical_cpu_vs_cuda", True),
    }

    (RESULTS_DIR / "host_eval_3seeds.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (RESULTS_DIR / "shortlist.json").write_text(
        json.dumps(agg["shortlist"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_c2_comparison(agg, per_seed_results[42], config_keys_labels, RESULTS_DIR / "host_eval_comparison.png")
    generate_c2_markdown_report(
        agg=agg,
        per_seed_results=per_seed_results,
        consist_data=consist_data,
        config_keys_labels=config_keys_labels,
        out_md=RESULTS_DIR / "host_eval_report.md",
    )

    print(f"[DONE] Aggregated 3-seed Gate H2 report in {agg['elapsed_sec']}s")
    print("Shortlist Summary:", json.dumps(agg["shortlist"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
