"""
mitdb_loader.py
Tải signal và annotation từ MIT-BIH Arrhythmia Database.
"""
import wfdb
import numpy as np
from pathlib import Path
from typing import Generator, Tuple

# 48 record IDs của MITDB
ALL_RECORDS = [
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
    122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
    209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
    222, 223, 228, 230, 231, 232, 233, 234
]

def load_record(record_id: int, channel_idx: int = 0,
                source: str = "local",
                local_path: str = r"C:\d2l-en\data\ECG_Project_Data\mitdb"
                ) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load 1 record: signal (channel_idx) + annotation.

    Args:
        record_id:   int, ví dụ 100
        channel_idx: 0 = MLII (default)
        source:      'local' (default) hoặc 'physionet'
        local_path:  Path đến thư mục chứa .hea/.dat/.atr
    Returns:
        signal:      np.ndarray float64, shape (N,)
        ann_samples: np.ndarray int64,   R-peak positions
        ann_symbols: list[str],          annotation symbols
    """
    rec_str = str(record_id)

    if source == "physionet":
        record = wfdb.rdrecord(rec_str, pn_dir='mitdb')
        ann    = wfdb.rdann  (rec_str, 'atr', pn_dir='mitdb')
    else:
        # Local source — ✓ path đã xác nhận
        rec_path = Path(local_path) / rec_str
        record   = wfdb.rdrecord(str(rec_path))
        ann      = wfdb.rdann  (str(rec_path), 'atr')
    
    signal      = record.p_signal[:, channel_idx]  # MLII
    ann_samples = ann.sample
    ann_symbols = ann.symbol
    
    return signal, ann_samples, ann_symbols


def iter_records(record_ids: list, channel_idx: int = 0,
                 source: str = "local",
                 local_path: str = r"C:\d2l-en\data\ECG_Project_Data\mitdb") -> Generator:
    """Iterate qua danh sách records, yield từng record."""
    for rec_id in record_ids:
        signal, ann_samples, ann_symbols = load_record(
            rec_id, channel_idx, source, local_path
        )
        yield rec_id, signal, ann_samples, ann_symbols
