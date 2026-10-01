"""
splits.py
Inter-patient split theo De Chazal 2004 (DS1/DS2 protocol).
"""
from typing import Dict, List

# DS1 = 22 records — dùng để train + val
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
       122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]

# DS2 = 22 records — chỉ dùng để test cuối
DS2 = [100, 103, 105, 111, 113, 117, 121, 123,
       200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# 4 records loại ra (paced beats)
EXCLUDED = [102, 104, 107, 217]

# Trong DS1, tách 4 records làm validation
VAL_RECORDS   = [203, 220, 223, 230]
TRAIN_RECORDS = [r for r in DS1 if r not in VAL_RECORDS]


def get_split_records() -> Dict[str, List[int]]:
    """
    Trả về danh sách record IDs cho từng split.
    Đây là phân chia cố định — không random.
    """
    # Sanity checks
    assert set(TRAIN_RECORDS) & set(VAL_RECORDS) == set(), \
        "Train và Val records bị overlap!"
    assert set(DS1) & set(DS2) == set(), \
        "DS1 và DS2 bị overlap!"
    assert set(EXCLUDED) & set(DS1 + DS2) == set(), \
        "Excluded records vẫn còn trong DS1/DS2!"
    
    return {
        'train': sorted(TRAIN_RECORDS),
        'val':   sorted(VAL_RECORDS),
        'test':  sorted(DS2),
    }


def get_all_44_records() -> List[int]:
    """44 records hợp lệ (loại 4 paced records)."""
    return sorted(DS1 + DS2)
