import torch.nn as nn

class MLPBaseline(nn.Module):
    def __init__(self, input_size: int, hidden1: int, hidden2: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)
