"""
E0. Script đo và kiểm chứng chi phí tính toán (Parameters, MACCs, Receptive Field,
Flash Weights, Peak Activation RAM) bằng PyTorch forward hooks cho:
- M1-Reg (CNN1DBinaryRegularized - 35,458 params)
- M2-A   (CNN1DBinaryDS variant A - 11,330 params)
- M2-B   (CNN1DBinaryDS variant B - 29,058 params - Đề xuất chính)
- M2-C   (CNN1DBinaryDS variant C - 16,802 params)

Đầu ra: results/binary/m2_ds_fp32/cost_report.json
"""
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn1d_binary_reg import CNN1DBinaryRegularized
from src.models.cnn1d_binary_ds import CNN1DBinaryDS, DS_VARIANTS, count_bn_folded_params

OUTPUT_PATH = PROJECT_ROOT / "results/binary/m2_ds_fp32/cost_report.json"
FS_HZ = 360.0


def profile_model(model: nn.Module, name: str, input_length: int = 260) -> dict:
    model.eval()
    layers_info = []
    rf = 1
    stride_cum = 1

    def make_hook(layer_name: str):
        def hook(module, inp, out):
            nonlocal rf, stride_cum
            in_tensor = inp[0]
            out_tensor = out

            if isinstance(module, nn.Conv1d):
                k = module.kernel_size[0]
                s = module.stride[0]
                d = module.dilation[0]
                g = module.groups
                c_in = module.in_channels
                c_out = module.out_channels
                l_in = int(in_tensor.shape[-1])
                l_out = int(out_tensor.shape[-1])

                rf = rf + (k - 1) * d * stride_cum
                stride_cum = stride_cum * s

                # Fused MCU params (weight + 1 bias per output channel after BN folding)
                fused_params = (c_out * (c_in // g) * k) + c_out
                maccs = l_out * c_out * (c_in // g) * k
                in_elems = int(in_tensor.numel())
                out_elems = int(out_tensor.numel())

                layers_info.append(
                    {
                        "layer": layer_name,
                        "type": "DepthwiseConv1d" if (g == c_in and c_in > 1) else (
                            "PointwiseConv1d" if k == 1 else "StandardConv1d"
                        ),
                        "config": f"{c_in}->{c_out}, k={k}, groups={g}",
                        "input_shape": list(in_tensor.shape),
                        "output_shape": list(out_tensor.shape),
                        "fused_params": int(fused_params),
                        "maccs": int(maccs),
                        "receptive_field_samples": int(rf),
                        "in_out_ram_fp32_bytes": int((in_elems + out_elems) * 4),
                        "in_out_ram_int8_bytes": int((in_elems + out_elems) * 1),
                        "output_ram_fp32_bytes": int(out_elems * 4),
                    }
                )
            elif isinstance(module, nn.MaxPool1d):
                k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
                s = module.stride if isinstance(module.stride, int) else module.stride[0]
                rf = rf + (k - 1) * stride_cum
                stride_cum = stride_cum * s
                in_elems = int(in_tensor.numel())
                out_elems = int(out_tensor.numel())
                layers_info.append(
                    {
                        "layer": layer_name,
                        "type": "MaxPool1d",
                        "config": f"k={k}, s={s}",
                        "input_shape": list(in_tensor.shape),
                        "output_shape": list(out_tensor.shape),
                        "fused_params": 0,
                        "maccs": 0,
                        "receptive_field_samples": int(rf),
                        "in_out_ram_fp32_bytes": int((in_elems + out_elems) * 4),
                        "in_out_ram_int8_bytes": int((in_elems + out_elems) * 1),
                        "output_ram_fp32_bytes": int(out_elems * 4),
                    }
                )
            elif isinstance(module, nn.Linear):
                c_in = module.in_features
                c_out = module.out_features
                fused_params = c_in * c_out + (c_out if module.bias is not None else 0)
                maccs = c_in * c_out
                in_elems = int(in_tensor.numel())
                out_elems = int(out_tensor.numel())
                layers_info.append(
                    {
                        "layer": layer_name,
                        "type": "Linear",
                        "config": f"{c_in}->{c_out}",
                        "input_shape": list(in_tensor.shape),
                        "output_shape": list(out_tensor.shape),
                        "fused_params": int(fused_params),
                        "maccs": int(maccs),
                        "receptive_field_samples": int(rf),
                        "in_out_ram_fp32_bytes": int((in_elems + out_elems) * 4),
                        "in_out_ram_int8_bytes": int((in_elems + out_elems) * 1),
                        "output_ram_fp32_bytes": int(out_elems * 4),
                    }
                )

        return hook

    hooks = []
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.MaxPool1d, nn.Linear)):
            hooks.append(m.register_forward_hook(make_hook(n)))

    with torch.no_grad():
        dummy = torch.zeros(1, 1, input_length)
        _ = model(dummy)

    for h in hooks:
        h.remove()

    raw_trainable_params = sum(p.numel() for p in model.parameters())
    total_fused_params = count_bn_folded_params(model)
    total_maccs = sum(item["maccs"] for item in layers_info)
    peak_inout_ram_fp32_bytes = max(item["in_out_ram_fp32_bytes"] for item in layers_info)
    peak_inout_ram_int8_bytes = max(item["in_out_ram_int8_bytes"] for item in layers_info)
    max_single_tensor_fp32_bytes = max(item["output_ram_fp32_bytes"] for item in layers_info)

    return {
        "model_name": name,
        "pytorch_trainable_params": int(raw_trainable_params),
        "mcu_fused_params": int(total_fused_params),
        "total_maccs": int(total_maccs),
        "receptive_field_before_gap_samples": int(rf),
        "receptive_field_before_gap_ms": round(1000.0 * rf / FS_HZ, 2),
        "flash_weights_fp32_kb": round(total_fused_params * 4 / 1024.0, 2),
        "flash_weights_int8_kb": round(total_fused_params * 1 / 1024.0, 2),
        "peak_pingpong_ram_fp32_kb": round(peak_inout_ram_fp32_bytes / 1024.0, 2),
        "peak_pingpong_ram_int8_kb": round(peak_inout_ram_int8_bytes / 1024.0, 2),
        "max_single_tensor_fp32_kb": round(max_single_tensor_fp32_bytes / 1024.0, 2),
        "layers": layers_info,
    }


def main():
    models = {
        "M1-Reg (Baseline 2-Class Regularized)": CNN1DBinaryRegularized(num_classes=2),
        "M2-A (Direct DS Replacement)": CNN1DBinaryDS(variant="A", num_classes=2),
        "M2-B (Proposed Accuracy-Prioritized DS-CNN)": CNN1DBinaryDS(variant="B", num_classes=2),
        "M2-C (Compact Cost-Prioritized DS-CNN)": CNN1DBinaryDS(variant="C", num_classes=2),
    }

    profiles = {}
    for label, mod in models.items():
        profiles[label] = profile_model(mod, label)

    base_maccs = profiles["M1-Reg (Baseline 2-Class Regularized)"]["total_maccs"]
    base_params = profiles["M1-Reg (Baseline 2-Class Regularized)"]["mcu_fused_params"]

    for label, p in profiles.items():
        p["maccs_vs_m1_reg_pct"] = round(100.0 * (p["total_maccs"] - base_maccs) / base_maccs, 2)
        p["params_vs_m1_reg_pct"] = round(
            100.0 * (p["mcu_fused_params"] - base_params) / base_params, 2
        )

    # Kiểm chứng khớp tuyệt đối với bảng thiết kế trong plan/ds_cnn_binary_plan.md
    assert profiles["M1-Reg (Baseline 2-Class Regularized)"]["mcu_fused_params"] == 35458
    assert profiles["M1-Reg (Baseline 2-Class Regularized)"]["total_maccs"] == 2970496
    assert profiles["M1-Reg (Baseline 2-Class Regularized)"]["receptive_field_before_gap_samples"] == 24

    for v_key, label in [
        ("A", "M2-A (Direct DS Replacement)"),
        ("B", "M2-B (Proposed Accuracy-Prioritized DS-CNN)"),
        ("C", "M2-C (Compact Cost-Prioritized DS-CNN)"),
    ]:
        assert profiles[label]["mcu_fused_params"] == DS_VARIANTS[v_key]["expected_fused_params"]
        assert profiles[label]["total_maccs"] == DS_VARIANTS[v_key]["expected_maccs"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

    print("=== BÁO CÁO ĐO ĐẠC TÀI NGUYÊN & TRƯỜNG TIẾP NHẬN (E0 + E1) ===")
    header = (
        f"{'Model Variant':44s} | {'MCU Params':>10s} | {'MACCs':>10s} | "
        f"{'Δ MACC':>8s} | {'RF (mẫu/ms)':>14s} | {'Flash FP32/INT8':>16s} | {'Peak RAM FP32/INT8':>18s}"
    )
    print(header)
    print("-" * len(header))
    for label, p in profiles.items():
        rf_str = f"{p['receptive_field_before_gap_samples']:2d} ({p['receptive_field_before_gap_ms']:5.1f}ms)"
        fl_str = f"{p['flash_weights_fp32_kb']:6.2f}/{p['flash_weights_int8_kb']:5.2f} KB"
        ram_str = f"{p['peak_pingpong_ram_fp32_kb']:5.2f}/{p['peak_pingpong_ram_int8_kb']:5.2f} KB"
        print(
            f"{label:44s} | {p['mcu_fused_params']:10,d} | {p['total_maccs']:10,d} | "
            f"{p['maccs_vs_m1_reg_pct']:+7.1f}% | {rf_str:>14s} | {fl_str:>16s} | {ram_str:>18s}"
        )

    print("\n--- Chi tiết từng tầng của M2-B (Biến thể đề xuất chính) ---")
    for lyr in profiles["M2-B (Proposed Accuracy-Prioritized DS-CNN)"]["layers"]:
        print(
            f"  {lyr['layer']:16s} ({lyr['type']:15s} {lyr['config']:18s}) | "
            f"Out: {str(lyr['output_shape']):14s} | "
            f"Params: {lyr['fused_params']:6,d} | MACCs: {lyr['maccs']:7,d} | "
            f"RF: {lyr['receptive_field_samples']:2d}"
        )
    print(f"\n[GATE E0 + E1 PASS] Đã lưu báo cáo chi tiết tại: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
