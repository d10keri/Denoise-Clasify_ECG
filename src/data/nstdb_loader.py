"""
nstdb_loader.py
Load MIT-BIH Noise Stress Test Database (NSTDB) noise recordings.

Noise recordings: bw (baseline wander), ma (muscle artifact), em (electrode motion)
Each recording: 650,000 samples at 360 Hz, 2 channels
"""

import wfdb
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


NOISE_TYPES = ["bw", "ma", "em"]


def load_noise_record(
    noise_type: str,
    channel_idx: int = 0,
    nstdb_path: str = "C:/d2l-en/data/ECG_Project_Data/nstdb",
) -> np.ndarray:
    """
    Load mot noise recording tu NSTDB.

    Args:
        noise_type:  'bw', 'ma', hoac 'em'
        channel_idx: 0 = noise1 (default), 1 = noise2
        nstdb_path:  Path den thu muc NSTDB

    Returns:
        noise_signal: 1D float64 array, (N_samples,)
                      da convert sang physical units (mV) qua WFDB
    """
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"noise_type phai la {NOISE_TYPES}, got '{noise_type}'")

    rec_path = Path(nstdb_path) / noise_type
    record = wfdb.rdrecord(str(rec_path))
    signal = record.p_signal[:, channel_idx]  # (N_samples,) float64 in mV
    return signal


def get_noise_blocks(
    noise_signal: np.ndarray,
    block_size: int = 100_000,
    guard_samples: int = 500,
) -> List[Tuple[int, int]]:
    """
    Chia noise_signal thanh cac blocks khong overlap.

    Args:
        noise_signal:  1D noise array
        block_size:    samples per block (default 100,000 = ~4.6 min @ 360Hz)
        guard_samples: so samples bo di o moi ranh gioi block (tranh filter edge effects)

    Returns:
        blocks: list of (start_idx, end_idx) pairs
                Moi block la [start + guard, start + block_size - guard)
                Blocks khong chong nhau va khong cross ranh gioi
    """
    n = len(noise_signal)
    blocks = []
    offset = 0
    while offset + block_size <= n:
        start = offset + guard_samples
        end = offset + block_size - guard_samples
        if end > start:
            blocks.append((start, end))
        offset += block_size
    return blocks


def assign_blocks_to_splits(
    blocks: List[Tuple[int, int]],
    train_block_indices: List[int] = None,
    val_block_indices: List[int] = None,
    test_block_indices: List[int] = None,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Gan cac noise blocks vao train/val/test splits.

    Default allocation (6 blocks tu 650,000 samples / 100,000):
        Blocks 0-3 → train
        Block 4    → val
        Block 5    → test

    Args:
        blocks:              List of (start, end) tuples tu get_noise_blocks()
        train_block_indices: Indices cua blocks danh cho train (default: [0,1,2,3])
        val_block_indices:   Indices cua blocks danh cho val   (default: [4])
        test_block_indices:  Indices cua blocks danh cho test  (default: [5])

    Returns:
        dict with keys 'train', 'val', 'test', moi key la list of (start, end)
    """
    n_blocks = len(blocks)
    if train_block_indices is None:
        # Default: blocks 0-3 = train, 4 = val, 5 = test
        train_block_indices = list(range(min(4, n_blocks)))
    if val_block_indices is None:
        val_block_indices = [4] if n_blocks > 4 else []
    if test_block_indices is None:
        test_block_indices = [5] if n_blocks > 5 else []

    def select(indices):
        return [blocks[i] for i in indices if i < n_blocks]

    assignment = {
        "train": select(train_block_indices),
        "val":   select(val_block_indices),
        "test":  select(test_block_indices),
    }

    # Sanity check: no overlap between splits
    all_assigned = (
        set(train_block_indices) | set(val_block_indices) | set(test_block_indices)
    )
    tv = set(train_block_indices) & set(val_block_indices)
    tt = set(train_block_indices) & set(test_block_indices)
    vt = set(val_block_indices) & set(test_block_indices)
    assert tv == set(), f"Train-Val noise block overlap: {tv}"
    assert tt == set(), f"Train-Test noise block overlap: {tt}"
    assert vt == set(), f"Val-Test noise block overlap: {vt}"

    return assignment


def sample_noise_window(
    noise_signal: np.ndarray,
    block_start: int,
    block_end: int,
    window_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    """
    Lay mot window ngau nhien tu mot noise block.

    Args:
        noise_signal: Full noise signal array
        block_start:  Start index of available block
        block_end:    End index of available block
        window_size:  Number of samples to extract
        rng:          numpy Generator for reproducible random sampling

    Returns:
        (noise_window, noise_start_idx):
            noise_window:     1D float64 array, shape (window_size,)
            noise_start_idx:  Start index trong noise_signal
    """
    available = block_end - block_start - window_size
    if available < 0:
        raise ValueError(
            f"Block [{block_start}, {block_end}] ({block_end - block_start} samples) "
            f"nho hon window_size ({window_size}). "
            f"Tang block_size hoac giam guard_samples."
        )
    offset = int(rng.integers(0, available + 1))
    start = block_start + offset
    end = start + window_size
    return noise_signal[start:end].copy(), start


def sample_noise_window_deterministic(
    noise_signal: np.ndarray,
    block_start: int,
    block_end: int,
    window_size: int,
    window_idx: int = 0,
    step: int = 512,
) -> Tuple[np.ndarray, int]:
    """
    Lay mot window tu mot noise block theo vi tri xac dinh (cho val/test deterministic).

    Args:
        noise_signal: Full noise signal array
        block_start:  Start index of available block
        block_end:    End index of available block
        window_size:  Number of samples to extract
        window_idx:   Index of window within block (0 = first, 1 = second, ...)
        step:         Step size between consecutive deterministic windows

    Returns:
        (noise_window, noise_start_idx)
    """
    start = block_start + window_idx * step
    end = start + window_size
    if end > block_end:
        # Wrap around: dung cuoi block
        start = block_end - window_size
        end = block_end
    if start < block_start:
        start = block_start
        end = start + window_size
    return noise_signal[start:end].copy(), start


def describe_noise_record(noise_type: str, nstdb_path: str) -> dict:
    """
    In thong tin NSTDB record de kiem tra.
    """
    rec_path = Path(nstdb_path) / noise_type
    record = wfdb.rdrecord(str(rec_path))
    return {
        "noise_type": noise_type,
        "n_samples": record.sig_len,
        "n_channels": record.n_sig,
        "fs": record.fs,
        "sig_name": record.sig_name,
        "units": record.units,
        "duration_sec": record.sig_len / record.fs,
    }
