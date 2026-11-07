import torch
import torch.nn as nn


class CNN1DBinaryRegularized(nn.Module):
    """
    1D-CNN Binary Classifier (Regularized) cho ECG (0 = N, 1 = non-N [S, V, F, Q]).
    Giữ nguyên cấu trúc 3 tầng Conv1d + Linear(128, 2) của CNN1DBinary (file backup vẫn giữ nguyên),
    bổ sung:
    - BatchNorm1d sau mỗi Conv1d: ổn định phân phối đặc trưng liên bệnh nhân (inter-patient),
      và đặc biệt KHI XUẤT ONNX / STM32 sẽ được gộp (fuse/fold) hoàn toàn vào trọng số Conv1d
      (0 MACC tăng thêm, 0 byte Flash/RAM tăng thêm trên MCU).
    - Dropout1d / Dropout: chống đồng thích nghi (co-adaptation) trên 18 bệnh nhân tập Train,
      tự động bị loại bỏ khi chạy inference (model.eval()).

    Input:  (batch, 1, 260)
    Output: (batch, 2) logits
    Trainable Parameters: 35,906 (Sau khi fuse BatchNorm cho MCU: đúng 35,458 params như M1 gốc)
    """
    def __init__(
        self,
        num_classes: int = 2,
        dropout_conv: float = 0.15,
        dropout_fc: float = 0.30,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(dropout_conv),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(dropout_conv),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )
        self.head_dropout = nn.Dropout(dropout_fc)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)         # (batch, 128, 1)
        x = x.squeeze(-1)            # (batch, 128)
        x = self.head_dropout(x)
        return self.classifier(x)    # (batch, 2)


if __name__ == "__main__":
    model = CNN1DBinaryRegularized(num_classes=2)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    bn_params = sum(
        p.numel() for m in model.modules() if isinstance(m, nn.BatchNorm1d) for p in m.parameters()
    )
    fused_params = n_params - bn_params
    dummy_in = torch.zeros(1, 1, 260)
    dummy_out = model(dummy_in)
    print("CNN1DBinaryRegularized initialized successfully.")
    print(f"Trainable parameters (with BN): {n_params:,} (BN params: {bn_params})")
    print(f"Effective MCU parameters after BN-folding: {fused_params:,} (Matches CNN1DBinary: 35,458)")
    print(f"Input shape:  {tuple(dummy_in.shape)}")
    print(f"Output shape: {tuple(dummy_out.shape)}")
    assert fused_params == 35458, f"Expected 35,458 fused params, got {fused_params}"
    assert dummy_out.shape == (1, 2)
