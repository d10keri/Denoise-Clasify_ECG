"""
E1. Định nghĩa kiến trúc Depthwise Separable 1D-CNN cho bài toán phân loại ECG 2 lớp
(0 = N, 1 = non-N [S, V, F, Q]).
Hoàn toàn độc lập với các model trước (cnn1d.py, cnn1d_binary.py, cnn1d_binary_reg.py).

Hỗ trợ 3 biến thể cấu hình qua словарь DS_VARIANTS:
- Variant "A" (M2-A): Thay thế trực tiếp Conv2, Conv3 bằng DS-Conv (k=5, 3), giữ 3 tầng.
- Variant "B" (M2-B - Đề xuất chính): Stem k=7 + 3 khối DS (k=9, 7, 5), vùng nhìn 86 mẫu (~239 ms).
- Variant "C" (M2-C - Ưu tiên chi phí): Giống B nhưng giảm độ rộng kênh (24 -> 48 -> 96 -> 96).
"""
from typing import Dict, Any
import torch
import torch.nn as nn


DS_VARIANTS: Dict[str, Dict[str, Any]] = {
    "A": {
        "description": "M2-A: Direct DS replacement (3 stages, RF=24 samples)",
        "stem": {"out_ch": 32, "k": 5, "pool": 2},
        "blocks": [
            {"in_ch": 32, "out_ch": 64, "k_dw": 5, "pool": 2, "dropout": 0.15},
            {"in_ch": 64, "out_ch": 128, "k_dw": 3, "pool": None, "dropout": 0.0},
        ],
        "head_dropout": 0.30,
        "expected_fused_params": 11330,
        "expected_maccs": 873856,
    },
    "B": {
        "description": "M2-B: Accuracy-prioritized Wide-RF 4-stage DS-CNN (RF=86 samples)",
        "stem": {"out_ch": 32, "k": 7, "pool": 2},
        "blocks": [
            {"in_ch": 32, "out_ch": 64, "k_dw": 9, "pool": 2, "dropout": 0.10},
            {"in_ch": 64, "out_ch": 128, "k_dw": 7, "pool": 2, "dropout": 0.10},
            {"in_ch": 128, "out_ch": 128, "k_dw": 5, "pool": None, "dropout": 0.0},
        ],
        "head_dropout": 0.20,
        "expected_fused_params": 29058,
        "expected_maccs": 1468544,
    },
    "C": {
        "description": "M2-C: Compact Wide-RF 4-stage DS-CNN (24->48->96->96, RF=86 samples)",
        "stem": {"out_ch": 24, "k": 7, "pool": 2},
        "blocks": [
            {"in_ch": 24, "out_ch": 48, "k_dw": 9, "pool": 2, "dropout": 0.10},
            {"in_ch": 48, "out_ch": 96, "k_dw": 7, "pool": 2, "dropout": 0.10},
            {"in_ch": 96, "out_ch": 96, "k_dw": 5, "pool": None, "dropout": 0.0},
        ],
        "head_dropout": 0.20,
        "expected_fused_params": 16802,
        "expected_maccs": 853344,
    },
}


class DSConvBlock(nn.Module):
    """
    Khối Depthwise Separable Convolution 1D chuẩn cho Edge AI:
      Depthwise Conv1d (groups = in_channels, bias=False) -> BatchNorm1d -> ReLU
      Pointwise Conv1d (kernel_size = 1, bias=False)      -> BatchNorm1d -> ReLU
      [Optional MaxPool1d(2)] -> [Optional Dropout1d(p)]

    Lưu ý triển khai MCU:
    - Đặt bias=False ở Conv1d vì BatchNorm1d liền sau đã có tham số dịch (beta).
    - Khi gộp (fold) BatchNorm1d vào Conv1d để xuất xuống STM32, mỗi tầng Conv1d
      sẽ có đúng (weight + 1 bias/channel), hoàn toàn không phát sinh chi phí phụ.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        ]
        if pool_size is not None and pool_size > 1:
            layers.append(nn.MaxPool1d(pool_size))
        if dropout is not None and dropout > 0.0:
            layers.append(nn.Dropout1d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN1DBinaryDS(nn.Module):
    """
    Mô hình 1D Depthwise Separable CNN cho phân loại nhị phân ECG (0 = N, 1 = non-N).
    Input:  (batch, 1, 260)
    Output: (batch, 2) logits
    """
    def __init__(self, variant: str = "B", num_classes: int = 2):
        super().__init__()
        variant = variant.upper()
        if variant not in DS_VARIANTS:
            raise ValueError(f"Variant phải thuộc {list(DS_VARIANTS.keys())}, nhận được: {variant}")

        self.variant = variant
        cfg = DS_VARIANTS[variant]
        stem_cfg = cfg["stem"]

        self.stem = nn.Sequential(
            nn.Conv1d(
                1,
                stem_cfg["out_ch"],
                kernel_size=stem_cfg["k"],
                padding=stem_cfg["k"] // 2,
                bias=False,
            ),
            nn.BatchNorm1d(stem_cfg["out_ch"]),
            nn.ReLU(),
            nn.MaxPool1d(stem_cfg["pool"]),
        )

        blocks = []
        last_ch = stem_cfg["out_ch"]
        for b_cfg in cfg["blocks"]:
            blocks.append(
                DSConvBlock(
                    in_channels=b_cfg["in_ch"],
                    out_channels=b_cfg["out_ch"],
                    kernel_size=b_cfg["k_dw"],
                    pool_size=b_cfg["pool"],
                    dropout=b_cfg["dropout"],
                )
            )
            last_ch = b_cfg["out_ch"]

        self.ds_blocks = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head_dropout = nn.Dropout(cfg["head_dropout"])
        self.classifier = nn.Linear(last_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.ds_blocks(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.head_dropout(x)
        return self.classifier(x)


def count_bn_folded_params(model: nn.Module) -> int:
    """
    Tính chính xác số lượng tham số tĩnh trên MCU sau khi gộp (fold) BatchNorm1d vào Conv1d:
    - Mỗi tầng Conv1d sẽ có: weight.numel() + out_channels (1 bias cho mỗi kênh ra).
    - Mỗi tầng Linear sẽ có: weight.numel() + bias.numel().
    """
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            total += m.weight.numel() + m.out_channels
        elif isinstance(m, nn.Linear):
            total += m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
    return total


if __name__ == "__main__":
    dummy_in = torch.zeros(1, 1, 260)
    print("=== KIỂM TRA KHỞI TẠO CÁC BIẾN THỂ CNN1DBinaryDS (E1) ===")
    for var_name, cfg in DS_VARIANTS.items():
        model = CNN1DBinaryDS(variant=var_name, num_classes=2)
        model.eval()
        out = model(dummy_in)
        raw_params = sum(p.numel() for p in model.parameters())
        fused_params = count_bn_folded_params(model)
        print(
            f"[Variant {var_name}] {cfg['description']}\n"
            f"  -> Output shape: {tuple(out.shape)} | "
            f"PyTorch Trainable Params: {raw_params:,} | "
            f"MCU Fused Params: {fused_params:,} (Expected: {cfg['expected_fused_params']:,})"
        )
        assert out.shape == (1, 2), f"Shape mismatch for variant {var_name}"
        assert (
            fused_params == cfg["expected_fused_params"]
        ), f"Fused params mismatch for {var_name}: {fused_params} != {cfg['expected_fused_params']}"
    print("Tất cả các biến thể A, B, C đều vượt qua kiểm tra khởi tạo & số lượng tham số!")
