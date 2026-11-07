import torch
import torch.nn as nn


class CNN1DBinary(nn.Module):
    """
    1D-CNN Binary Classifier for ECG (0 = N, 1 = non-N [S, V, F, Q]).
    Cấu trúc feature extractor giữ nguyên 100% như CNN1DBaseline,
    chỉ thay tầng phân loại cuối từ Linear(128, 5) sang Linear(128, 2).
    Input:  (batch, 1, 260)
    Output: (batch, 2) logits
    Parameters: 35,458
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # (batch, 128, 1)
        x = x.squeeze(-1)          # (batch, 128)
        return self.classifier(x)  # (batch, 2)


if __name__ == "__main__":
    model = CNN1DBinary(num_classes=2)
    n_params = sum(p.numel() for p in model.parameters())
    dummy_in = torch.zeros(1, 1, 260)
    dummy_out = model(dummy_in)
    print(f"CNN1DBinary initialized successfully.")
    print(f"Total parameters: {n_params:,} (Expected: 35,458)")
    print(f"Input shape:  {tuple(dummy_in.shape)}")
    print(f"Output shape: {tuple(dummy_out.shape)}")
    assert n_params == 35458, f"Parameter count mismatch: {n_params} != 35458"
    assert dummy_out.shape == (1, 2), f"Output shape mismatch: {dummy_out.shape}"
