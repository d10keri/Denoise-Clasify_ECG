"""
binary_dataset.py
Dataset wrapper cho bài toán phân loại ECG nhị phân:
- Lớp 0: N (Normal, nhãn gốc == 0)
- Lớp 1: non-N (Abnormal: S, V, F, Q, nhãn gốc > 0)
Không sửa đổi file src/data/ecg_dataset.py hay dữ liệu gốc.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.ecg_dataset import ECGClassificationDataset


class ECGBinaryDataset(Dataset):
    BINARY_CLASS_NAMES = ["N", "non-N"]
    ORIG_CLASS_NAMES = ["N", "S", "V", "F", "Q"]

    def __init__(self, data_dir: str, split: str, add_channel_dim: bool = True):
        self.base_dataset = ECGClassificationDataset(
            data_dir=data_dir, split=split, add_channel_dim=add_channel_dim
        )
        self.split = split
        self.add_channel_dim = add_channel_dim
        # Nhãn 5 lớp gốc (int32) và nhãn nhị phân (0 = N, 1 = non-N)
        self.y5 = self.base_dataset.y.astype(np.int64)
        self.y_binary = (self.y5 > 0).astype(np.int64)

    def __len__(self) -> int:
        return len(self.y_binary)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.base_dataset.X[idx], dtype=torch.float32)
        if self.add_channel_dim:
            x = x.unsqueeze(0)  # (1, 260)
        y_bin = torch.tensor(self.y_binary[idx], dtype=torch.long)
        return x, y_bin

    def get_binary_class_weights(self) -> torch.Tensor:
        """
        Tính trọng số lớp cân bằng cho bài toán 2 lớp trên tập train:
        w_c = Total / (2 * N_c)
        """
        total = len(self.y_binary)
        n0 = int(np.sum(self.y_binary == 0))
        n1 = int(np.sum(self.y_binary == 1))
        w0 = total / (2.0 * max(n0, 1))
        w1 = total / (2.0 * max(n1, 1))
        return torch.tensor([w0, w1], dtype=torch.float32)


def make_binary_dataloader(
    data_dir: str,
    split: str,
    batch_size: int = 64,
    add_channel_dim: bool = True,
    shuffle: bool = None,
    num_workers: int = 0,
) -> DataLoader:
    if shuffle is None:
        shuffle = split == "train"

    dataset = ECGBinaryDataset(
        data_dir=data_dir, split=split, add_channel_dim=add_channel_dim
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
