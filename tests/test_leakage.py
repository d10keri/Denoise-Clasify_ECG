"""
test_leakage.py
Kiểm tra toàn diện data leakage cho Phase 1 MITDB dataset. (Level 1 & 2)
"""
import sys
import numpy as np
from pathlib import Path

# Đảm bảo có thể import src module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splits import get_split_records, EXCLUDED

PROCESSED_DIR = Path("data/processed")

# ── Level 1: Record-level split integrity ──────────────────────────────────

def test_no_record_overlap_in_split_definition():
    """Định nghĩa split trong code không có overlap."""
    splits = get_split_records()
    train  = set(splits['train'])
    val    = set(splits['val'])
    test   = set(splits['test'])
    
    assert train & val  == set(), f"Train-Val overlap: {train & val}"
    assert train & test == set(), f"Train-Test overlap: {train & test}"
    assert val   & test == set(), f"Val-Test overlap: {val & test}"


def test_excluded_records_not_in_any_split():
    """Paced records (102,104,107,217) không được xuất hiện trong bất kỳ split."""
    splits    = get_split_records()
    used      = set(splits['train'] + splits['val'] + splits['test'])
    excluded  = set(EXCLUDED)
    
    assert excluded & used == set(), \
        f"Paced records xuất hiện trong split: {excluded & used}"


def test_total_records_count():
    """Tổng số records sau khi split phải là 44."""
    splits = get_split_records()
    total  = len(splits['train']) + len(splits['val']) + len(splits['test'])
    assert total == 44, f"Tổng phải là 44, got {total}"


# ── Level 2: NPZ file record_id integrity ──────────────────────────────────

def test_npz_record_ids_no_overlap():
    """
    Sau khi build dataset (T7), mỗi record_id chỉ xuất hiện trong
    đúng 1 split. Kiểm tra 3 file NPZ.
    """
    files = {
        'train': PROCESSED_DIR / 'train.npz',
        'val':   PROCESSED_DIR / 'val.npz',
        'test':  PROCESSED_DIR / 'test.npz',
    }
    
    for split_name, fp in files.items():
        if not fp.exists():
            print(f"  SKIP {split_name}: {fp} chua ton tai")
            return
    
    train_ids = set(np.load(files['train'])['record_ids'].tolist())
    val_ids   = set(np.load(files['val'])  ['record_ids'].tolist())
    test_ids  = set(np.load(files['test']) ['record_ids'].tolist())
    
    assert train_ids & val_ids  == set(), \
        f"LEAKAGE! Record IDs xuat hien o ca Train va Val: {train_ids & val_ids}"
    assert train_ids & test_ids == set(), \
        f"LEAKAGE! Record IDs xuat hien o ca Train va Test: {train_ids & test_ids}"
    assert val_ids   & test_ids == set(), \
        f"LEAKAGE! Record IDs xuat hien o ca Val va Test: {val_ids & test_ids}"
    
    print(f"  - NPZ record IDs: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")


def test_npz_beat_indices_no_cross_patient_leak():
    """
    Mỗi (record_id, beat_index) pair chỉ xuất hiện trong 1 split.
    """
    files = {
        'train': PROCESSED_DIR / 'train.npz',
        'val':   PROCESSED_DIR / 'val.npz',
        'test':  PROCESSED_DIR / 'test.npz',
    }
    
    for fp in files.values():
        if not fp.exists():
            print(f"  SKIP: NPZ files chua ton tai")
            return
    
    def get_pairs(fp):
        data = np.load(fp)
        return set(zip(data['record_ids'].tolist(), data['beat_indices'].tolist()))
    
    train_pairs = get_pairs(files['train'])
    val_pairs   = get_pairs(files['val'])
    test_pairs  = get_pairs(files['test'])
    
    tv = train_pairs & val_pairs
    tt = train_pairs & test_pairs
    vt = val_pairs   & test_pairs
    
    assert len(tv) == 0, f"LEAKAGE! {len(tv)} (record,beat) pairs trung Train-Val"
    assert len(tt) == 0, f"LEAKAGE! {len(tt)} (record,beat) pairs trung Train-Test"
    assert len(vt) == 0, f"LEAKAGE! {len(vt)} (record,beat) pairs trung Val-Test"
    
    print(f"  - No (record, beat_index) pairs leaked between splits")


def test_expected_record_ids_per_split():
    """
    Record IDs trong NPZ phải khớp CHÍNH XÁC với định nghĩa trong splits.py.
    """
    files = {
        'train': PROCESSED_DIR / 'train.npz',
        'val':   PROCESSED_DIR / 'val.npz',
        'test':  PROCESSED_DIR / 'test.npz',
    }
    splits = get_split_records()
    
    for split_name, fp in files.items():
        if not fp.exists():
            print(f"  SKIP {split_name}: {fp} chua ton tai")
            return
        
        data             = np.load(fp)
        actual_ids       = set(data['record_ids'].tolist())
        expected_ids     = set(splits[split_name])
        
        assert actual_ids == expected_ids, \
            f"Split '{split_name}': expected {expected_ids}, got {actual_ids}\n" \
            f"  Missing: {expected_ids - actual_ids}\n" \
            f"  Extra: {actual_ids - expected_ids}"
    
    print("  - Record IDs trong NPZ khop hoan toan voi splits.py")


if __name__ == "__main__":
    print("=== Data Leakage Tests ===\n")
    tests = [
        test_no_record_overlap_in_split_definition,
        test_excluded_records_not_in_any_split,
        test_total_records_count,
        test_npz_record_ids_no_overlap,
        test_npz_beat_indices_no_cross_patient_leak,
        test_expected_record_ids_per_split,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS]: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL]: {t.__name__}\n    {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
