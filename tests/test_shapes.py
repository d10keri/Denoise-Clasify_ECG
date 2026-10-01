"""
test_shapes.py
Kiểm tra toàn diện shape, dtype, và value range của dataset.
"""
import numpy as np
import torch
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
WINDOW_SIZE   = 260
N_CLASSES     = 5


def load_split(split_name: str) -> dict:
    fp = PROCESSED_DIR / f"{split_name}.npz"
    assert fp.exists(), f"File khong ton tai: {fp}"
    return dict(np.load(fp, allow_pickle=False))


# ── Shape tests ──────────────────────────────────────────────────────────────

def test_X_shape_all_splits():
    """X phải là 2D, dimension 1 = 260."""
    for split in ['train', 'val', 'test']:
        data = load_split(split)
        X    = data['X']
        assert X.ndim == 2, f"[{split}] X.ndim={X.ndim}, phai la 2"
        assert X.shape[1] == WINDOW_SIZE, \
            f"[{split}] X.shape[1]={X.shape[1]}, phai la {WINDOW_SIZE}"
        print(f"  - {split}: X.shape = {X.shape}")


def test_y_shape_all_splits():
    """y phải là 1D, length == X.shape[0]."""
    for split in ['train', 'val', 'test']:
        data = load_split(split)
        X, y = data['X'], data['y']
        assert y.ndim == 1, f"[{split}] y.ndim={y.ndim}, phai la 1"
        assert X.shape[0] == y.shape[0], \
            f"[{split}] X.shape[0]={X.shape[0]} != y.shape[0]={y.shape[0]}"
        print(f"  - {split}: y.shape = {y.shape}")


def test_dtype_all_splits():
    """X phải float32, y phải int32 hoặc int64."""
    for split in ['train', 'val', 'test']:
        data = load_split(split)
        assert data['X'].dtype == np.float32, \
            f"[{split}] X.dtype={data['X'].dtype}, phai la float32"
        assert data['y'].dtype in (np.int32, np.int64), \
            f"[{split}] y.dtype={data['y'].dtype}, phai la int32/int64"
        print(f"  - {split}: X.dtype={data['X'].dtype}, y.dtype={data['y'].dtype}")


def test_y_values_valid_classes():
    """y chỉ chứa giá trị {0, 1, 2, 3, 4}."""
    for split in ['train', 'val', 'test']:
        data = load_split(split)
        unique = set(data['y'].tolist())
        assert unique.issubset({0, 1, 2, 3, 4}), \
            f"[{split}] y chua gia tri khong hop le: {unique - {0,1,2,3,4}}"
        print(f"  - {split}: unique classes = {sorted(unique)}")


def test_X_no_nan_no_inf():
    """X không chứa NaN hay Inf."""
    for split in ['train', 'val', 'test']:
        data = load_split(split)
        X    = data['X']
        assert not np.any(np.isnan(X)), f"[{split}] X chua NaN!"
        assert not np.any(np.isinf(X)), f"[{split}] X chua Inf!"
        print(f"  - {split}: X khong co NaN/Inf")


def test_X_normalized_per_beat():
    """Mỗi beat (row) phải có mean ≈ 0 và std ≈ 1."""
    for split in ['train', 'val', 'test']:
        data    = load_split(split)
        X       = data['X']
        means   = np.mean(X, axis=1)  # (N,)
        stds    = np.std(X, axis=1)   # (N,)
        
        # Cho phép tolerance nhỏ (float32 precision)
        max_mean_err = np.max(np.abs(means))
        max_std_err  = np.max(np.abs(stds - 1.0))
        
        assert max_mean_err < 0.01, \
            f"[{split}] Max |mean| = {max_mean_err:.6f} > 0.01"
        assert max_std_err < 0.1, \
            f"[{split}] Max |std-1| = {max_std_err:.6f} > 0.1"
        print(f"  - {split}: per-beat normalization OK "
              f"(max|mean|={max_mean_err:.4f}, max|std-1|={max_std_err:.4f})")


def test_meta_arrays_consistency():
    """record_ids và beat_indices phải có length = N_beats."""
    for split in ['train', 'val', 'test']:
        data = load_split(split)
        N    = data['X'].shape[0]
        assert data['record_ids'].shape  == (N,), \
            f"[{split}] record_ids.shape={data['record_ids'].shape}, phai la ({N},)"
        assert data['beat_indices'].shape == (N,), \
            f"[{split}] beat_indices.shape={data['beat_indices'].shape}, phai la ({N},)"
        print(f"  - {split}: meta arrays shape = ({N},)")


# ── PyTorch compatibility tests ───────────────────────────────────────────────

def test_mlp_input_shape():
    """
    Kiểm tra X tương thích với MLP input.
    MLP nhận: (batch_size, 260)
    """
    data        = load_split('train')
    X_np        = data['X'][:32]                    # lấy 1 batch
    X_tensor    = torch.from_numpy(X_np)            # (32, 260)
    
    assert X_tensor.shape == (32, WINDOW_SIZE), \
        f"MLP input shape sai: {X_tensor.shape}, phai la (32, {WINDOW_SIZE})"
    assert X_tensor.dtype == torch.float32
    print(f"  - MLP input: {X_tensor.shape} {X_tensor.dtype}")


def test_cnn_input_shape():
    """
    Kiểm tra X tương thích với 1D-CNN input.
    CNN nhận: (batch_size, 1, 260) — thêm channel dim bằng unsqueeze(1)
    """
    data        = load_split('train')
    X_np        = data['X'][:32]                    # (32, 260)
    X_tensor    = torch.from_numpy(X_np)            # (32, 260)
    X_cnn       = X_tensor.unsqueeze(1)             # (32, 1, 260)
    
    assert X_cnn.shape == (32, 1, WINDOW_SIZE), \
        f"CNN input shape sai: {X_cnn.shape}, phai la (32, 1, {WINDOW_SIZE})"
    assert X_cnn.dtype == torch.float32
    print(f"  - 1D-CNN input: {X_cnn.shape} {X_cnn.dtype}")


def test_label_tensor_for_crossentropy():
    """
    y phải convert được sang torch.long để dùng với CrossEntropyLoss.
    """
    data   = load_split('train')
    y_np   = data['y'][:32]
    y_tens = torch.from_numpy(y_np).long()
    
    assert y_tens.dtype == torch.int64, f"y dtype phai la int64, got {y_tens.dtype}"
    assert y_tens.min() >= 0 and y_tens.max() < N_CLASSES, \
        f"y values phai trong [0, {N_CLASSES-1}], got [{y_tens.min()}, {y_tens.max()}]"
    print(f"  - Label tensor: {y_tens.shape} {y_tens.dtype} thuoc [{y_tens.min()}, {y_tens.max()}]")


if __name__ == "__main__":
    print("=== Shape & Compatibility Tests ===\n")
    tests = [
        test_X_shape_all_splits,
        test_y_shape_all_splits,
        test_dtype_all_splits,
        test_y_values_valid_classes,
        test_X_no_nan_no_inf,
        test_X_normalized_per_beat,
        test_meta_arrays_consistency,
        test_mlp_input_shape,
        test_cnn_input_shape,
        test_label_tensor_for_crossentropy,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS]: {t.__name__}\n")
            passed += 1
        except (AssertionError, FileNotFoundError) as e:
            print(f"  [FAIL]: {t.__name__}\n    {e}\n")
    print(f"{passed}/{len(tests)} tests passed")
