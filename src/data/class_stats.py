"""
class_stats.py
Tính và báo cáo phân phối lớp, class weights.
"""
import numpy as np
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
N_CLASSES   = 5
MIN_SAMPLES_WARNING = 50


def print_distribution(y: np.ndarray, split_name: str) -> dict:
    """In phân phối lớp và trả về dict counts."""
    counts = Counter(y.tolist())
    total  = len(y)
    
    print(f"\n{'='*50}")
    print(f"Class Distribution - {split_name.upper()} ({total} beats)")
    print(f"{'='*50}")
    print(f"{'Class':<8} {'Name':<8} {'Count':>8} {'Ratio':>8}")
    print(f"{'-'*36}")
    
    for cls_id in range(N_CLASSES):
        n     = counts.get(cls_id, 0)
        ratio = n / total * 100 if total > 0 else 0
        warn  = " [!] " if n < MIN_SAMPLES_WARNING else ""
        print(f"  {cls_id:<6} {CLASS_NAMES[cls_id]:<8} {n:>8,} {ratio:>7.2f}%{warn}")
    
    return dict(counts)


def compute_weights(y_train: np.ndarray) -> np.ndarray:
    """
    Tính class weights cho CrossEntropyLoss.
    Dùng sklearn 'balanced' formula:
        weight[c] = N_total / (N_classes * N_c)
    """
    classes = np.arange(N_CLASSES)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    print("\nClass Weights (cho CrossEntropyLoss):")
    for cls_id, w in enumerate(weights):
        print(f"  Class {cls_id} ({CLASS_NAMES[cls_id]}): {w:.4f}")
    return weights.astype(np.float32)


def check_minimum_samples(y: np.ndarray, split_name: str,
                          min_samples: int = MIN_SAMPLES_WARNING):
    """Cảnh báo nếu bất kỳ lớp nào quá ít samples."""
    counts = Counter(y.tolist())
    issues = []
    for cls_id in range(N_CLASSES):
        n = counts.get(cls_id, 0)
        if n < min_samples:
            issues.append(f"Class {cls_id} ({CLASS_NAMES[cls_id]}): {n} samples < {min_samples}")
    
    if issues:
        print(f"\n[WARNING] [{split_name}]: Cac lop co qua it samples:")
        for msg in issues:
            print(f"   {msg}")
    return issues

if __name__ == "__main__":
    # Test internal logic
    print("=== Testing class_stats.py ===")
    dummy_y = np.array([0]*900 + [1]*50 + [2]*40 + [3]*5 + [4]*5)
    print_distribution(dummy_y, 'test_split')
    check_minimum_samples(dummy_y, 'test_split')
    weights = compute_weights(dummy_y)
    print("\n[PASS] Testing completed successfully!")
