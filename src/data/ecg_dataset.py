"""
ecg_dataset.py
PyTorch Dataset and DataLoader cho ECG Classification.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class ECGClassificationDataset(Dataset):
    CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']

    def __init__(self, data_dir: str, split: str, add_channel_dim: bool = False):
        """
        Args:
            data_dir: Đường dẫn đến thư mục chứa file NPZ (vd: 'data/processed')
            split: 'train', 'val', hoặc 'test'
            add_channel_dim: True cho 1D-CNN (shape: 1, 260), False cho MLP (shape: 260)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.add_channel_dim = add_channel_dim
        
        file_path = self.data_dir / f"{split}.npz"
        if not file_path.exists():
            raise FileNotFoundError(f"Khong tim thay file {file_path}")
            
        data = np.load(file_path, allow_pickle=False)
        self.X = data['X']
        self.y = data['y']
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        
        if self.add_channel_dim:
            x = x.unsqueeze(0)                # (1, 260) float32
        
        return x, y

    def get_class_weights(self) -> torch.Tensor:
        """
        Tính class weights từ training set cho CrossEntropyLoss.
        Chỉ gọi trên split='train'.
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.arange(5)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=self.y
        )
        return torch.tensor(weights, dtype=torch.float32)

    def summary(self) -> str:
        """In tóm tắt dataset."""
        from collections import Counter
        counts = Counter(self.y.tolist())
        lines  = [f"ECGClassificationDataset [{self.split}]",
                  f"  Total beats: {len(self):,}",
                  f"  add_channel_dim: {self.add_channel_dim}"]
        for cls_id in range(5):
            n = counts.get(cls_id, 0)
            lines.append(f"  Class {cls_id} ({self.CLASS_NAMES[cls_id]}): {n:,}")
        return "\n".join(lines)


def make_dataloader(
    data_dir: str,
    split: str,
    batch_size: int = 64,
    add_channel_dim: bool = False,
    shuffle: bool = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Factory function tạo DataLoader cho 1 split.
    shuffle mặc định: True cho train, False cho val/test.
    """
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = ECGClassificationDataset(data_dir, split, add_channel_dim)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader
