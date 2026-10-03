import torch.nn as nn

class CNN1DBaseline(nn.Module):
    """
    1D-CNN Baseline for ECG Classification.
    Input:  (batch, 1, 260)
    Output: (batch, 5) logits
    """
    def __init__(self, num_classes: int = 5):
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

    def forward(self, x):
        x = self.features(x)       # (batch, 128, 1)
        x = x.squeeze(-1)          # (batch, 128)
        return self.classifier(x)  # (batch, 5)
