"""
Phase B: Quantization of M2-B-KD using ESP-PPQ (v1.3.14) for ESP32-S3 (N16R8)
Generates .espdl models, .json quantization configs, .info graph dumps, and layerwise SNR logs for:
- B     (b_int16.espdl) : Full INT16 (w16a16, TargetPlatform.ESPDL_S3_INT16)
- C0    (c0.espdl)      : Default INT8 (w8a8, TargetPlatform.ESPDL_S3_INT8, KL calibration on calib_train.npy)
- C1    (c1.espdl)      : Peak-aware INT8 (calibrated on calib_train.npy + 200 high-amplitude V/S/F beats from train.npz)
- C2a   (c2a.espdl)     : Forced Input Exponent -4 (scale = 0.0625, no R-peak clipping, covers +-7.94)
- C2b   (c2b.espdl)     : Forced Input Exponent -5 (scale = 0.03125, clips at +-3.97, 2x finer P/T wave resolution)
- C3    (c3.espdl)      : Mixed Precision (First Conv block + Final Gemm in INT16, middle DS-CNN blocks in INT8)
- CW8A16(c_w8a16.espdl) : 8-bit Weights + 16-bit Activations (ESPDL_S3_W8A16)
"""

import copy
import io
import json
import math
import os
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import onnx
import torch
from torch.utils.data import DataLoader, TensorDataset

import esp_ppq.lib as PFL
from esp_ppq.api import (
    QuantizationSettingFactory,
    espdl_quantize_onnx,
    get_target_platform,
)
from esp_ppq.api.espdl_interface import generate_test_value, get_random_inputs
from esp_ppq.core import QuantizationStates, TargetPlatform
from esp_ppq.executor import TorchExecutor
from esp_ppq.IR.quantize import QuantableOperation
from esp_ppq.quantization.analyse import graphwise_error_analyse, layerwise_error_analyse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORIG_ONNX = PROJECT_ROOT / "export" / "cnn1d_binary_dsB_kd_fp32.onnx"
CALIB_PATH = PROJECT_ROOT / "export" / "calib_train.npy"
TRAIN_NPZ_PATH = PROJECT_ROOT / "data" / "processed" / "train.npz"
HOST_VAL_X_PATH = PROJECT_ROOT / "export" / "host_val_test500_x.npy"
HOST_VAL_Y_PATH = PROJECT_ROOT / "export" / "host_val_test500_y_m2bkd.npy"

QUANT_DIR = PROJECT_ROOT / "esp32s3" / "quant"
RESULTS_DIR = PROJECT_ROOT / "esp32s3" / "results"


def build_c1_highpeak_calibration(calib_base: np.ndarray, train_npz_path: Path, out_path: Path) -> np.ndarray:
    """
    Create C1 calibration set: 600 beats from calib_train.npy + 200 high-absmax non-N (V/S/F) beats
    strictly from train.npz (never touching VAL or TEST).
    """
    if out_path.exists():
        arr = np.load(str(out_path))
        if arr.shape == (800, 1, 260):
            return arr

    data = np.load(str(train_npz_path))
    x_train = data["X"].astype(np.float32)
    y_train = data["y"]
    if x_train.ndim == 2:
        x_train = x_train[:, np.newaxis, :]

    # Non-N classes: y != 0 (S=1, V=2, F=3, Q=4)
    non_n_mask = y_train != 0
    x_non_n = x_train[non_n_mask]
    abs_max_per_beat = np.max(np.abs(x_non_n).reshape(len(x_non_n), -1), axis=1)

    # Sort descending by peak amplitude and pick 200 well-spaced high-peak beats (top 2000 pool -> 200)
    sorted_idx = np.argsort(-abs_max_per_beat)
    top_pool = sorted_idx[:2000]
    rng = np.random.default_rng(42)
    chosen_idx = rng.choice(top_pool, size=200, replace=False)
    high_peak_beats = x_non_n[chosen_idx]

    c1_calib = np.concatenate([calib_base, high_peak_beats], axis=0).astype(np.float32)
    np.save(str(out_path), c1_calib)
    return c1_calib


def extract_graph_quant_details(ppq_graph) -> dict:
    """Extract input/output scales, exponents, bitwidths, and platform assignments per operation."""
    ops_info = []
    input_exponents = {}
    output_exponents = {}

    for op_name, op in ppq_graph.operations.items():
        if not isinstance(op, QuantableOperation):
            continue
        in_cfgs = []
        for cfg, var in zip(op. input_quant_config, op.inputs):
            scale_val = cfg.scale.detach().cpu().numpy().tolist() if isinstance(cfg.scale, torch.Tensor) else cfg.scale
            exp_val = None
            if isinstance(scale_val, float) and scale_val > 0:
                exp_val = int(round(math.log2(scale_val)))
            elif isinstance(scale_val, list) and len(scale_val) > 0:
                exps = [int(round(math.log2(s))) for s in scale_val if s > 0]
                exp_val = [min(exps), max(exps)] if len(exps) > 1 else (exps[0] if exps else None)
            in_cfgs.append({
                "var": var.name,
                "is_param": bool(var.is_parameter),
                "bits": cfg.num_of_bits,
                "state": cfg.state.name,
                "exponent": exp_val,
            })
            if var.name in ppq_graph.inputs and exp_val is not None:
                input_exponents[var.name] = exp_val

        out_cfgs = []
        for cfg, var in zip(op.output_quant_config, op.outputs):
            scale_val = cfg.scale.detach().cpu().numpy().tolist() if isinstance(cfg.scale, torch.Tensor) else cfg.scale
            exp_val = None
            if isinstance(scale_val, float) and scale_val > 0:
                exp_val = int(round(math.log2(scale_val)))
            out_cfgs.append({
                "var": var.name,
                "bits": cfg.num_of_bits,
                "state": cfg.state.name,
                "exponent": exp_val,
            })
            if var.name in ppq_graph.outputs and exp_val is not None:
                output_exponents[var.name] = exp_val

        ops_info.append({
            "name": op_name,
            "type": op.type,
            "platform": op.platform.name,
            "inputs": in_cfgs,
            "outputs": out_cfgs,
        })

    return {
        "graph_input_exponents": input_exponents,
        "graph_output_exponents": output_exponents,
        "operations": ops_info,
    }


def force_input_exponent_and_reexport(
    ppq_graph,
    forced_exponent: int,
    espdl_export_file: Path,
    target_platform: TargetPlatform,
    input_shape: list,
    device: str = "cpu",
):
    """
    Force the quantization scale of the graph input tensor (and its dominated configs)
    to 2^forced_exponent, then export to .espdl.
    """
    new_scale = float(2.0 ** forced_exponent)
    first_op = ppq_graph.operations["/features/features.0/Conv"]
    in_cfg = first_op.input_quant_config[0]
    master_cfg = in_cfg.dominated_by
    master_cfg.scale = torch.tensor(new_scale, dtype=torch.float32, device=master_cfg.scale.device)
    master_cfg.offset = torch.tensor(0.0, dtype=torch.float32, device=master_cfg.scale.device)
    master_cfg.state = QuantizationStates.ACTIVATED

    dummy_inputs = get_random_inputs(input_shape, torch.float32, device)
    values_for_test = generate_test_value(ppq_graph, device, dummy_inputs, None)

    PFL.Exporter(platform=target_platform).export(
        file_path=str(espdl_export_file),
        graph=ppq_graph,
        values_for_test=values_for_test,
        export_config=True,
    )


def quick_sanity_eval_500(ppq_graph, x_500: np.ndarray, y_ref_500: np.ndarray, device: str = "cpu") -> dict:
    """Run PPQ TorchExecutor on the 500 host validation beats to check sanity right in Phase B."""
    executor = TorchExecutor(graph=ppq_graph, device=device)
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(x_500), 50):
            batch = torch.from_numpy(x_500[i : i + 50]).float().to(device)
            out = executor.forward(inputs=batch)[0].cpu().numpy()
            preds_list.append(out)
    y_q = np.concatenate(preds_list, axis=0)

    diff = np.abs(y_q - y_ref_500)
    max_abs_diff = float(np.max(diff))
    mean_abs_diff = float(np.mean(diff))
    cos_sim = float(
        np.sum(y_q * y_ref_500) / (np.linalg.norm(y_q) * np.linalg.norm(y_ref_500) + 1e-12)
    )
    pred_q = np.argmax(y_q, axis=1)
    pred_ref = np.argmax(y_ref_500, axis=1)
    agree_pct = float(np.mean(pred_q == pred_ref) * 100.0)
    return {
        "host500_agree_vs_fp32_pct": round(agree_pct, 2),
        "host500_cosine_sim": round(cos_sim, 6),
        "host500_max_abs_logit_diff": round(max_abs_diff, 4),
        "host500_mean_abs_logit_diff": round(mean_abs_diff, 4),
    }


def run_single_config(
    cfg_key: str,
    description: str,
    working_onnx: Path,
    calib_array: np.ndarray,
    quant_type: str,
    num_of_bits: int,
    setting_fn,
    dispatching_override: dict,
    forced_input_exp: int,
    x_500: np.ndarray,
    y_ref_500: np.ndarray,
    log_buffer: io.StringIO,
):
    print(f"\n=== Running Config {cfg_key}: {description} ===")
    log_buffer.write(f"\n{'='*80}\nCONFIG {cfg_key}: {description}\n{'='*80}\n")

    # Always refresh working ONNX copy so esp-ppq never mutates original export/*.onnx
    shutil.copyfile(str(ORIG_ONNX), str(working_onnx))

    dl = DataLoader(TensorDataset(torch.from_numpy(calib_array).float()), batch_size=32, shuffle=False)
    calib_steps = math.ceil(len(calib_array) / 32)
    collate_fn = lambda b: b[0].to("cpu")

    espdl_file = QUANT_DIR / f"{cfg_key}.espdl"
    setting = setting_fn() if setting_fn is not None else None

    f_capture = io.StringIO()
    with redirect_stdout(f_capture):
        ppq_graph = espdl_quantize_onnx(
            onnx_import_file=str(working_onnx),
            espdl_export_file=str(espdl_file),
            calib_dataloader=dl,
            calib_steps=calib_steps,
            input_shape=[1, 1, 260],
            target="esp32s3",
            num_of_bits=num_of_bits,
            quant_type=quant_type,
            setting=setting,
            dispatching_override=dispatching_override,
            collate_fn=collate_fn,
            device="cpu",
            error_report=True,
            skip_export=(forced_input_exp is not None),
            export_config=True,
            export_test_values=True,
            verbose=0,
        )

        if forced_input_exp is not None:
            target_platform = get_target_platform("esp32s3", quant_type=quant_type)
            force_input_exponent_and_reexport(
                ppq_graph=ppq_graph,
                forced_exponent=forced_input_exp,
                espdl_export_file=espdl_file,
                target_platform=target_platform,
                input_shape=[1, 1, 260],
                device="cpu",
            )
            # Run error analysis after forcing input exponent
            print(f"\n--- Error Report After Forcing Input Exponent = {forced_input_exp} ---")
            graphwise_error_analyse(
                graph=ppq_graph,
                running_device="cpu",
                collate_fn=collate_fn,
                dataloader=dl,
            )
            layerwise_error_analyse(
                graph=ppq_graph,
                running_device="cpu",
                collate_fn=collate_fn,
                dataloader=dl,
            )

    captured_text = f_capture.getvalue()
    log_buffer.write(captured_text + "\n")

    # Save quantized PPQ graph for fast evaluation in Phase C
    ppq_pt_file = QUANT_DIR / f"{cfg_key}.ppq.pt"
    try:
        torch.save(ppq_graph, str(ppq_pt_file))
    except Exception as e:
        print(f"[Warn] Could not pickle ppq_graph for {cfg_key}: {e}")

    quant_details = extract_graph_quant_details(ppq_graph)
    sanity500 = quick_sanity_eval_500(ppq_graph, x_500, y_ref_500, device="cpu")
    size_bytes = espdl_file.stat().st_size if espdl_file.exists() else 0

    summary = {
        "config_key": cfg_key,
        "description": description,
        "espdl_file": str(espdl_file.relative_to(PROJECT_ROOT)),
        "espdl_size_bytes": size_bytes,
        "espdl_size_kb": round(size_bytes / 1024.0, 2),
        "calib_samples": len(calib_array),
        "quant_type": quant_type,
        "forced_input_exponent": forced_input_exp,
        "graph_input_exponents": quant_details["graph_input_exponents"],
        "graph_output_exponents": quant_details["graph_output_exponents"],
        "sanity_host500": sanity500,
        "operations": quant_details["operations"],
    }
    print(
        f"[OK] {cfg_key}: size={summary['espdl_size_kb']} KB, "
        f"in_exp={summary['graph_input_exponents']}, "
        f"out_exp={summary['graph_output_exponents']}, "
        f"host500_agree={sanity500['host500_agree_vs_fp32_pct']}%, "
        f"cos={sanity500['host500_cosine_sim']}"
    )
    return summary


def main():
    QUANT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    calib_base = np.load(str(CALIB_PATH)).astype(np.float32)
    calib_c1_path = QUANT_DIR / "calib_c1_highpeak.npy"
    calib_c1 = build_c1_highpeak_calibration(calib_base, TRAIN_NPZ_PATH, calib_c1_path)

    x_500 = np.load(str(HOST_VAL_X_PATH)).astype(np.float32)
    y_ref_500 = np.load(str(HOST_VAL_Y_PATH)).astype(np.float32)

    working_onnx = QUANT_DIR / "m2_dsB_kd_fp32_working.onnx"
    log_buffer = io.StringIO()

    # Define settings for C3 Mixed Precision:
    # First Conv block (/features/features.0/Conv, /features/features.1/Relu, /features/features.2/MaxPool)
    # and Final classifier (/classifier/Gemm) run in 16-bit (TargetPlatform.ESPDL_S3_INT16)
    def make_c3_setting():
        s = QuantizationSettingFactory.espdl_setting(num_of_bits=8)
        s.dispatching_table.append("/features/features.0/Conv", TargetPlatform.ESPDL_S3_INT16)
        s.dispatching_table.append("/features/features.1/Relu", TargetPlatform.ESPDL_S3_INT16)
        s.dispatching_table.append("/features/features.2/MaxPool", TargetPlatform.ESPDL_S3_INT16)
        s.dispatching_table.append("/classifier/Gemm", TargetPlatform.ESPDL_S3_INT16)
        # Use minmax on the 16-bit layers so 16-bit dynamic range is not clipped by 8-bit KL observer
        for op_16 in [
            "/features/features.0/Conv",
            "/features/features.1/Relu",
            "/features/features.2/MaxPool",
            "/classifier/Gemm",
        ]:
            s.quantize_activation_setting.calib_algorithm_per_layer[op_16] = "minmax"
            s.quantize_parameter_setting.calib_algorithm_per_layer[op_16] = "minmax"
        return s

    configs = [
        (
            "b_int16",
            "B: Full INT16 (w16a16, ESPDL_S3_INT16, minmax calib on 600 beats)",
            calib_base,
            "w16a16",
            16,
            None,
            None,
            None,
        ),
        (
            "c0",
            "C0: Default INT8 (w8a8, ESPDL_S3_INT8, KL calib on 600 beats)",
            calib_base,
            "w8a8",
            8,
            None,
            None,
            None,
        ),
        (
            "c1",
            "C1: Peak-Enriched INT8 (w8a8, KL calib on 600 base + 200 high-peak V/S/F train beats)",
            calib_c1,
            "w8a8",
            8,
            None,
            None,
            None,
        ),
        (
            "c2a",
            "C2a: INT8 with Forced Input Exponent -4 (scale=0.0625, range +-7.94, 0% R-peak clipping)",
            calib_base,
            "w8a8",
            8,
            None,
            None,
            -4,
        ),
        (
            "c2b",
            "C2b: INT8 with Forced Input Exponent -5 (scale=0.03125, range +-3.97, 2x finer P/T resolution)",
            calib_base,
            "w8a8",
            8,
            None,
            None,
            -5,
        ),
        (
            "c3",
            "C3: Mixed Precision INT16+INT8 (Stem Conv1+MaxPool1 & Final Gemm in INT16, middle DS blocks in INT8, peak-enriched calib)",
            calib_c1,
            "w8a8",
            8,
            make_c3_setting,
            None,
            None,
        ),
        (
            "c_w8a16",
            "C_W8A16: 8-bit Weights + 16-bit Activations (ESPDL_S3_W8A16, full 16-bit wave dynamic range)",
            calib_base,
            "w8a16",
            16,
            None,
            None,
            None,
        ),
    ]

    all_summaries = {}
    for (
        cfg_key,
        desc,
        c_arr,
        q_type,
        n_bits,
        s_fn,
        d_override,
        f_exp,
    ) in configs:
        try:
            res = run_single_config(
                cfg_key=cfg_key,
                description=desc,
                working_onnx=working_onnx,
                calib_array=c_arr,
                quant_type=q_type,
                num_of_bits=n_bits,
                setting_fn=s_fn,
                dispatching_override=d_override,
                forced_input_exp=f_exp,
                x_500=x_500,
                y_ref_500=y_ref_500,
                log_buffer=log_buffer,
            )
            all_summaries[cfg_key] = res
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[ERROR] {cfg_key} failed: {e}\n{tb}")
            log_buffer.write(f"\n[ERROR] {cfg_key} failed:\n{tb}\n")
            all_summaries[cfg_key] = {"config_key": cfg_key, "error": str(e)}

    # Clean up temporary working ONNX
    if working_onnx.exists():
        working_onnx.unlink()

    log_path = QUANT_DIR / "quant_log.txt"
    log_path.write_text(log_buffer.getvalue(), encoding="utf-8")

    summary_path = QUANT_DIR / "quant_summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[DONE] Saved log to {log_path}")
    print(f"[DONE] Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
