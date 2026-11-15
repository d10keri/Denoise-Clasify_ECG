"""
C1. Xuất các mô hình phân loại nhị phân ECG (FP32) sang định dạng ONNX tĩnh
tối ưu cho STM32Cube.AI / ST Edge AI Core (input cố định `(1, 1, 260)`, không dùng dynamic_axes).

Đặc điểm kỹ thuật (theo plan/edge_ai_int8_binary_plan.md - Bước C1):
1. Không sửa bất kỳ file model gốc nào (`cnn1d_binary.py`, `cnn1d_binary_reg.py`, `cnn1d_binary_ds.py`).
2. Dùng wrapper xuất tĩnh (`MCUStaticONNXWrapper`):
   - Tự động gộp toán học (fold/fuse) mọi tầng `BatchNorm1d` vào tầng `Conv1d` liền trước
     bằng `torch.nn.utils.fuse_conv_bn_eval`, loại bỏ hoàn toàn các tầng `Dropout` / `Identity`.
   - Thay `AdaptiveAvgPool1d(1)` + `squeeze(-1)` bằng `AvgPool1d(kernel_size=L, stride=L)` + `Flatten(1)`
     (với L=65 cho M1/M1-Reg và L=32 cho M2-B-KD) để đồ thị ONNX hoàn toàn tĩnh, sạch 100%
     cho ST Edge AI Core và ONNXRuntime PTQ.
3. Xuất đầy đủ:
   - `export/cnn1d_binary_fp32.onnx` (từ `M1-Reg Seed 42` - ứng viên chính thức của Giai đoạn B/C)
   - `export/cnn1d_binary_reg_seed{42,43,44}_fp32.onnx` (cả 3 seed của `M1-Reg`)
   - `export/cnn1d_binary_dsB_kd_fp32.onnx` & `export/cnn1d_binary_dsB_kd_seed{42,43,44}_fp32.onnx`
     (cả 3 seed của `M2-B-KD` - ứng viên DS-CNN từ Giai đoạn E)
   - `export/cnn1d_binary_m1_base_fp32.onnx` (bản M1 gốc không BN để đối chiếu)
"""
import copy
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import onnx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn1d_binary import CNN1DBinary
from src.models.cnn1d_binary_reg import CNN1DBinaryRegularized
from src.models.cnn1d_binary_ds import CNN1DBinaryDS

CKPT_DIR = PROJECT_ROOT / "checkpoints"
EXPORT_DIR = PROJECT_ROOT / "export"
SEEDS = [42, 43, 44]
OPSET_VERSION = 17


class MCUStaticONNXWrapper(nn.Module):
    """
    Wrapper đóng gói mô hình thành chuỗi tuần tự tĩnh hoàn toàn cho MCU:
    - Gộp (fuse) Conv1d + BatchNorm1d -> Conv1d duy nhất (có bias).
    - Bỏ toàn bộ Dropout / Dropout1d.
    - Thay AdaptiveAvgPool1d(1) bằng AvgPool1d(kernel_size=spatial_len, stride=spatial_len)
      theo đúng khuyến nghị tại Bước C1 trong plan/edge_ai_int8_binary_plan.md.
    """
    def __init__(self, fused_feature_layers: nn.ModuleList, spatial_len: int, classifier: nn.Linear):
        super().__init__()
        self.features = nn.Sequential(*fused_feature_layers)
        self.avg_pool = nn.AvgPool1d(kernel_size=spatial_len, stride=spatial_len)
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = copy.deepcopy(classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.classifier(x)


def extract_and_fuse_layers(module: nn.Module) -> list:
    """
    Duyệt đệ quy qua các tầng con của feature extractor, gộp mọi cặp (Conv1d, BatchNorm1d)
    liền kề và chỉ giữ lại Conv1d, ReLU, MaxPool1d.
    """
    raw_leaf_modules = []

    def _collect(m: nn.Module):
        children = list(m.children())
        if len(children) == 0:
            raw_leaf_modules.append(m)
        else:
            for c in children:
                _collect(c)

    _collect(module)

    fused_list = []
    i = 0
    while i < len(raw_leaf_modules):
        curr = raw_leaf_modules[i]
        if isinstance(curr, nn.Conv1d):
            if i + 1 < len(raw_leaf_modules) and isinstance(raw_leaf_modules[i + 1], nn.BatchNorm1d):
                bn = raw_leaf_modules[i + 1]
                fused_conv = torch.nn.utils.fuse_conv_bn_eval(curr.eval(), bn.eval())
                fused_list.append(fused_conv)
                i += 2
                continue
            else:
                fused_list.append(copy.deepcopy(curr).eval())
        elif isinstance(curr, (nn.ReLU, nn.MaxPool1d)):
            fused_list.append(copy.deepcopy(curr).eval())
        elif isinstance(curr, (nn.Dropout, nn.Dropout1d, nn.AdaptiveAvgPool1d, nn.Identity)):
            pass
        else:
            raise ValueError(f"Toán tử chưa xử lý trong wrapper: {type(curr)}")
        i += 1

    return fused_list


def build_mcu_wrapper(orig_model: nn.Module, input_length: int = 260) -> MCUStaticONNXWrapper:
    orig_model.eval()
    if hasattr(orig_model, "features"):
        feat_layers = extract_and_fuse_layers(orig_model.features)
    elif hasattr(orig_model, "stem") and hasattr(orig_model, "ds_blocks"):
        feat_layers = extract_and_fuse_layers(orig_model.stem) + extract_and_fuse_layers(orig_model.ds_blocks)
    else:
        raise ValueError(f"Cấu trúc model không nhận diện được: {type(orig_model)}")

    temp_seq = nn.Sequential(*feat_layers).eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 1, input_length)
        feat_out = temp_seq(dummy)
        spatial_len = int(feat_out.shape[-1])

    wrapper = MCUStaticONNXWrapper(
        fused_feature_layers=nn.ModuleList(feat_layers),
        spatial_len=spatial_len,
        classifier=orig_model.classifier,
    )
    wrapper.eval()

    # Kiểm tra độ khớp tuyệt đối giữa model gốc và wrapper đã gộp BN + AvgPool1d tĩnh
    with torch.no_grad():
        test_x = torch.randn(8, 1, input_length)
        y_orig = orig_model(test_x)
        y_wrap = wrapper(test_x)
        max_diff = float((y_orig - y_wrap).abs().max().item())
        assert max_diff < 1e-5, f"Sai lệch sau khi gộp BN/Pool quá lớn: {max_diff}"

    return wrapper


def export_single_onnx(model: nn.Module, out_path: Path, input_length: int = 260) -> dict:
    wrapper = build_mcu_wrapper(model, input_length=input_length)
    dummy_input = torch.randn(1, 1, input_length, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Sử dụng trình xuất ONNX chuẩn với hằng số gấp gọn (constant folding) và shape tĩnh
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=["ecg_input"],
        output_names=["logits"],
        dynamo=False,
    )

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    ops = [node.op_type for node in onnx_model.graph.node]
    n_params = sum(p.numel() for p in wrapper.parameters())
    file_size_bytes = out_path.stat().st_size

    return {
        "onnx_path": str(out_path.relative_to(PROJECT_ROOT)),
        "file_size_bytes": int(file_size_bytes),
        "file_size_kb": round(file_size_bytes / 1024.0, 2),
        "fused_mcu_params": int(n_params),
        "opset_version": OPSET_VERSION,
        "input_shape": [1, 1, input_length],
        "output_shape": [1, 2],
        "onnx_operators_in_order": ops,
    }


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    export_summary = {}

    # 1. Xuất M1-Reg (CNN1DBinaryRegularized) cho cả 3 seeds (42, 43, 44)
    export_summary["M1-Reg"] = {}
    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"cnn1d_binary_reg_best_seed{seed}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m_cfg = ckpt.get("model_config", {})
        model = CNN1DBinaryRegularized(
            num_classes=m_cfg.get("num_classes", 2),
            dropout_conv=m_cfg.get("dropout_conv", 0.15),
            dropout_fc=m_cfg.get("dropout_fc", 0.30),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        out_path = EXPORT_DIR / f"cnn1d_binary_reg_seed{seed}_fp32.onnx"
        info = export_single_onnx(model, out_path)
        export_summary["M1-Reg"][str(seed)] = info

        if seed == 42:
            # Tạo bản tên chuẩn theo đúng kế hoạch edge_ai_int8_binary_plan.md
            main_reg_path = EXPORT_DIR / "cnn1d_binary_reg_fp32.onnx"
            main_plan_path = EXPORT_DIR / "cnn1d_binary_fp32.onnx"
            shutil.copy2(out_path, main_reg_path)
            shutil.copy2(out_path, main_plan_path)

    # 2. Xuất M2-B-KD (CNN1DBinaryDS Variant B + Knowledge Distillation) cho cả 3 seeds (42, 43, 44)
    export_summary["M2-B-KD"] = {}
    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"cnn1d_binary_dsB_kd_best_seed{seed}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = CNN1DBinaryDS(variant="B", num_classes=2)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        out_path = EXPORT_DIR / f"cnn1d_binary_dsB_kd_seed{seed}_fp32.onnx"
        info = export_single_onnx(model, out_path)
        export_summary["M2-B-KD"][str(seed)] = info

        if seed == 42:
            main_ds_path = EXPORT_DIR / "cnn1d_binary_dsB_kd_fp32.onnx"
            shutil.copy2(out_path, main_ds_path)

    # 3. Xuất M1 gốc (Backup không có BN - Seed 42) để đối chiếu
    ckpt_m1 = torch.load(CKPT_DIR / "cnn1d_binary_best_seed42.pt", map_location="cpu", weights_only=False)
    model_m1 = CNN1DBinary(num_classes=2)
    model_m1.load_state_dict(ckpt_m1["model_state_dict"])
    model_m1.eval()
    export_summary["M1-Base-Seed42"] = export_single_onnx(
        model_m1, EXPORT_DIR / "cnn1d_binary_m1_base_fp32.onnx"
    )

    summary_path = EXPORT_DIR / "onnx_export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(export_summary, f, indent=2, ensure_ascii=False)

    print("=== HOÀN TẤT BƯỚC C1: XUẤT FILE ONNX TĨNH CHO STM32 ===")
    print(f"1. M1-Reg (Seed 42 -> cnn1d_binary_fp32.onnx):")
    print(f"   - Params (Fused): {export_summary['M1-Reg']['42']['fused_mcu_params']:,}")
    print(f"   - Kích thước file: {export_summary['M1-Reg']['42']['file_size_kb']} KB")
    print(f"   - Chuỗi toán tử ONNX: {export_summary['M1-Reg']['42']['onnx_operators_in_order']}")
    print(f"2. M2-B-KD (Seed 42 -> cnn1d_binary_dsB_kd_fp32.onnx):")
    print(f"   - Params (Fused): {export_summary['M2-B-KD']['42']['fused_mcu_params']:,}")
    print(f"   - Kích thước file: {export_summary['M2-B-KD']['42']['file_size_kb']} KB")
    print(f"   - Chuỗi toán tử ONNX: {export_summary['M2-B-KD']['42']['onnx_operators_in_order']}")


if __name__ == "__main__":
    main()
