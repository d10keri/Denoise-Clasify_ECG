"""
test_dataset.py
Kiểm tra PyTorch Dataset và DataLoader cho Phase 1.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.data.ecg_dataset import ECGClassificationDataset, make_dataloader

def run_tests():
    print("=== T9 PyTorch Dataset & DataLoader Tests ===\n")
    
    # Test MLP dataset
    ds_mlp = ECGClassificationDataset('data/processed', split='train', add_channel_dim=False)
    print(ds_mlp.summary())
    x, y = ds_mlp[0]
    assert x.shape == (260,),    f"MLP x.shape={x.shape}, phải là (260,)"
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    print("  [PASS]: MLP sample shape and dtype OK")
    
    # Test CNN dataset
    ds_cnn = ECGClassificationDataset('data/processed', split='train', add_channel_dim=True)
    x, y = ds_cnn[0]
    assert x.shape == (1, 260),  f"CNN x.shape={x.shape}, phải là (1, 260)"
    assert x.dtype == torch.float32
    print("  [PASS]: CNN sample shape and dtype OK")
    
    # Test DataLoader
    loader = make_dataloader('data/processed', 'train', batch_size=64, add_channel_dim=False)
    X_batch, y_batch = next(iter(loader))
    assert X_batch.shape == (64, 260),  f"MLP batch shape={X_batch.shape}"
    print(f"  [PASS]: MLP batch shape OK: {X_batch.shape}")
    
    loader_cnn = make_dataloader('data/processed', 'train', batch_size=64, add_channel_dim=True)
    X_batch, y_batch = next(iter(loader_cnn))
    assert X_batch.shape == (64, 1, 260), f"CNN batch shape={X_batch.shape}"
    print(f"  [PASS]: CNN batch shape OK: {X_batch.shape}")
    
    # Test class weights
    weights = ds_mlp.get_class_weights()
    assert weights.shape == (5,)
    assert not torch.any(torch.isnan(weights))
    print(f"  [PASS]: Class weights computed OK: {weights.numpy()}")
    
    # Test val và test
    for split in ['val', 'test']:
        ds = ECGClassificationDataset('data/processed', split)
        x, y = ds[0]
        assert x.shape == (260,)
        assert y.dtype == torch.int64
    print("  [PASS]: val and test splits load OK")
    
    print("\n[PASS] Tat ca T9 PyTorch Dataset/DataLoader tests passed")

if __name__ == "__main__":
    run_tests()
