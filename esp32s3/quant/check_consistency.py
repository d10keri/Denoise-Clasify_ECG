"""
Phase C2.1: Consistency Check between exported .espdl (CPU Phase B) and live PPQ evaluation graphs (CPU vs CUDA).
Checks every tensor's quantization exponent and compares 500-beat logits between CPU and CUDA.
Saves: esp32s3/results/consistency_check.json
"""

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from esp_ppq.api import QuantizationSettingFactory, espdl_quantize_onnx
from esp_ppq.core import QuantizationStates, TargetPlatform
from esp_ppq.executor import TorchExecutor
from esp_ppq.IR.quantize import QuantableOperation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORIG_ONNX = PROJECT_ROOT / "export" / "cnn1d_binary_dsB_kd_fp32.onnx"
CALIB_PATH = PROJECT_ROOT / "export" / "calib_train.npy"
HOST_VAL_X_PATH = PROJECT_ROOT / "export" / "host_val_test500_x.npy"
QUANT_DIR = PROJECT_ROOT / "esp32s3" / "quant"
RESULTS_DIR = PROJECT_ROOT / "esp32s3" / "results"


def extract_tensor_exponents(ppq_graph) -> dict:
    exps = {}
    for op_name, op in ppq_graph.operations.items():
        if not isinstance(op, QuantableOperation):
            continue
        for cfg, var in zip(op.input_quant_config + op.output_quant_config, op.inputs + op.outputs):
            s = cfg.scale.detach().cpu().numpy().tolist() if isinstance(cfg.scale, torch.Tensor) else cfg.scale
            if isinstance(s, float) and s > 0:
                exps[f"{op_name}:{var.name}"] = int(round(math.log2(s)))
            elif isinstance(s, list) and len(s) > 0:
                exps[f"{op_name}:{var.name}"] = [int(round(math.log2(x))) for x in s if x > 0]
    return exps


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


def build_graph_on_device(working_onnx: Path, calib_arr: np.ndarray, q_type: str, n_bits: int, s_fn, f_exp, device: str):
    shutil.copyfile(str(ORIG_ONNX), str(working_onnx))
    dl = DataLoader(TensorDataset(torch.from_numpy(calib_arr).float()), batch_size=32, shuffle=False)
    calib_steps = math.ceil(len(calib_arr) / 32)
    setting = s_fn() if s_fn is not None else None
    g = espdl_quantize_onnx(
        onnx_import_file=str(working_onnx),
        espdl_export_file=str(QUANT_DIR / "_tmp_consist.espdl"),
        calib_dataloader=dl,
        calib_steps=calib_steps,
        input_shape=[1, 1, 260],
        target="esp32s3",
        num_of_bits=n_bits,
        quant_type=q_type,
        setting=setting,
        collate_fn=lambda b: b[0].to(device),
        device=device,
        error_report=False,
        skip_export=True,
        verbose=0,
    )
    if f_exp is not None:
        new_scale = float(2.0 ** f_exp)
        first_op = g.operations["/features/features.0/Conv"]
        master_cfg = first_op.input_quant_config[0].dominated_by
        master_cfg.scale = torch.tensor(new_scale, dtype=torch.float32, device=master_cfg.scale.device)
        master_cfg.offset = torch.tensor(0.0, dtype=torch.float32, device=master_cfg.scale.device)
        master_cfg.state = QuantizationStates.ACTIVATED
    return g


def eval_500(ppq_graph, x_500: np.ndarray, device: str) -> np.ndarray:
    ex = TorchExecutor(graph=ppq_graph, device=device)
    out = np.empty((len(x_500), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(x_500), 250):
            bt = torch.from_numpy(x_500[i : i + 250]).to(device=device, dtype=torch.float32)
            out[i : i + 250] = ex.forward(inputs=bt)[0].detach().cpu().numpy()
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    calib_base = np.load(str(CALIB_PATH)).astype(np.float32)
    calib_c1 = np.load(str(QUANT_DIR / "calib_c1_highpeak.npy")).astype(np.float32)
    x_500 = np.load(str(HOST_VAL_X_PATH)).astype(np.float32)
    phase_b_summary = json.loads((QUANT_DIR / "quant_summary.json").read_text(encoding="utf-8"))

    configs = [
        ("b_int16", calib_base, "w16a16", 16, None, None),
        ("c0", calib_base, "w8a8", 8, None, None),
        ("c1", calib_c1, "w8a8", 8, None, None),
        ("c2b", calib_base, "w8a8", 8, None, -5),
        ("c3", calib_c1, "w8a8", 8, make_c3_setting, None),
        ("c_w8a16", calib_base, "w8a16", 16, None, None),
    ]

    working_onnx = QUANT_DIR / "_tmp_consist_working.onnx"
    has_cuda = torch.cuda.is_available()
    results = {
        "cuda_available": has_cuda,
        "all_exponents_identical_cpu_vs_phase_b": True,
        "all_exponents_identical_cpu_vs_cuda": True,
        "all_logits_identical_cpu_vs_cuda": True,
        "configs": {},
    }

    for cfg_key, c_arr, q_type, n_bits, s_fn, f_exp in configs:
        g_cpu = build_graph_on_device(working_onnx, c_arr, q_type, n_bits, s_fn, f_exp, "cpu")
        exp_cpu = extract_tensor_exponents(g_cpu)
        logits_cpu = eval_500(g_cpu, x_500, "cpu")

        # Compare against Phase B quant_summary.json operations
        b_ops = phase_b_summary[cfg_key]["operations"]
        b_mismatches = []
        for op_entry in b_ops:
            op_name = op_entry["name"]
            for io_item in op_entry["inputs"] + op_entry["outputs"]:
                k = f"{op_name}:{io_item['var']}"
                v_cpu = exp_cpu.get(k)
                if isinstance(v_cpu, list):
                    v_cpu_Cmp = [min(v_cpu), max(v_cpu)] if len(v_cpu) > 1 else v_cpu[0]
                else:
                    v_cpu_Cmp = v_cpu
                if v_cpu_Cmp != io_item["exponent"]:
                    b_mismatches.append({"tensor": k, "phase_b": io_item["exponent"], "cpu_rebuild": v_cpu_Cmp})

        cuda_mismatches = []
        max_logit_diff_cpu_cuda = 0.0
        if has_cuda:
            g_cuda = build_graph_on_device(working_onnx, c_arr, q_type, n_bits, s_fn, f_exp, "cuda")
            exp_cuda = extract_tensor_exponents(g_cuda)
            logits_cuda = eval_500(g_cuda, x_500, "cuda")
            for k, v_cpu in exp_cpu.items():
                v_cuda = exp_cuda.get(k)
                if v_cpu != v_cuda:
                    cuda_mismatches.append({"tensor": k, "cpu": v_cpu, "cuda": v_cuda})
            max_logit_diff_cpu_cuda = float(np.max(np.abs(logits_cpu - logits_cuda)))

        if b_mismatches:
            results["all_exponents_identical_cpu_vs_phase_b"] = False
        if cuda_mismatches:
            results["all_exponents_identical_cpu_vs_cuda"] = False
        if max_logit_diff_cpu_cuda > 1e-5:
            results["all_logits_identical_cpu_vs_cuda"] = False

        results["configs"][cfg_key] = {
            "num_tensors_checked": len(exp_cpu),
            "phase_b_vs_cpu_mismatches": b_mismatches,
            "cpu_vs_cuda_exponent_mismatches": cuda_mismatches,
            "max_abs_logit_diff_cpu_vs_cuda_500beats": round(max_logit_diff_cpu_cuda, 8),
        }
        print(
            f"[C2.1] {cfg_key:8s}: tensors={len(exp_cpu)}, "
            f"phaseB_diff={len(b_mismatches)}, cuda_exp_diff={len(cuda_mismatches)}, "
            f"max_logit_diff_cpu_cuda={max_logit_diff_cpu_cuda:.8f}"
        )

    if working_onnx.exists():
        working_onnx.unlink()

    out_path = RESULTS_DIR / "consistency_check.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Saved {out_path}")


if __name__ == "__main__":
    main()
