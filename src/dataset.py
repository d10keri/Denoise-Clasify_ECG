from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDenoisingDataset(Dataset):
    """
    Dataset cho bài toán:
        noisy ECG -> clean ECG

    Dữ liệu:
        X_clean: (num_records, signal_length)
        X_noisy: (num_records, num_noises, signal_length)

    Mỗi sample trả về:
        noisy_window: (1, window_size)
        clean_window: (1, window_size)
    """

    def __init__(
        self,
        data_dir,
        split="train",
        window_size=1024,
        stride=1024,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.window_size = window_size
        self.stride = stride

        clean_path = self.data_dir / f"X_clean_{split}.npy"
        noisy_path = self.data_dir / f"X_noisy_{split}.npy"

        if not clean_path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {clean_path}")

        if not noisy_path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {noisy_path}")

        # Không load toàn bộ dữ liệu vào RAM
        self.clean = np.load(clean_path, mmap_mode="r")
        self.noisy = np.load(noisy_path, mmap_mode="r")

        if self.clean.shape[0] != self.noisy.shape[0]:
            raise ValueError("Số record clean và noisy không giống nhau.")

        if self.clean.shape[1] != self.noisy.shape[2]:
            raise ValueError("Độ dài tín hiệu clean/noisy không giống nhau.")

        self.num_records = self.clean.shape[0]
        self.num_noises = self.noisy.shape[1]
        self.signal_length = self.clean.shape[1]

        # Tạo danh sách vị trí window
        self.samples = []

        for record_idx in range(self.num_records):
            for noise_idx in range(self.num_noises):
                start = 0

                while start + self.window_size <= self.signal_length:
                    self.samples.append(
                        (record_idx, noise_idx, start)
                    )
                    start += self.stride

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record_idx, noise_idx, start = self.samples[idx]

        end = start + self.window_size

        noisy_window = self.noisy[
            record_idx, noise_idx, start:end
        ]

        clean_window = self.clean[
            record_idx, start:end
        ]

        # float64 -> float32
        noisy_window = np.asarray(
            noisy_window,
            dtype=np.float32
        )

        clean_window = np.asarray(
            clean_window,
            dtype=np.float32
        )

        # PyTorch 1D CNN thường dùng:
        # (channel, samples)
        noisy_window = torch.from_numpy(
            noisy_window
        ).unsqueeze(0)

        clean_window = torch.from_numpy(
            clean_window
        ).unsqueeze(0)

        return noisy_window, clean_window