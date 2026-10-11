"""
denoising_data.py
Level 2 Denoising Dataset:
  - Signal preprocessing (filter + normalize) cho clean reference
  - Window extraction (sliding window, no R-peak centering)
  - SNR calculation va noise injection
  - PyTorch Dataset cho training (on-the-fly) va val/test (deterministic)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict, Optional

# Reuse filter function tu Phase 1 (khong thay doi)
from src.data.preprocessing import bandpass_butterworth


# ─── 1. Signal preprocessing ──────────────────────────────────────────────────

def filter_clean_reference(
    signal: np.ndarray,
    lowcut: float = 0.05,
    highcut: float = 60.0,
    fs: float = 360.0,
    order: int = 4,
) -> np.ndarray:
    """
    Ap dung bandpass Butterworth filter cho ECG clean reference.

    Lowcut = 0.05 Hz (KHAC Phase 1 la 0.5 Hz) de giu BW content cho model hoc.
    Highcut = 60.0 Hz de loai powerline noise.

    Args:
        signal:  Raw ECG signal, 1D float array
        lowcut:  Lower cutoff Hz (default 0.05 — giu BW content)
        highcut: Upper cutoff Hz (default 60.0)
        fs:      Sampling rate (default 360.0 Hz)
        order:   Butterworth order (default 4)

    Returns:
        filtered: 1D float64 array, cung shape voi signal
    """
    return bandpass_butterworth(signal, lowcut, highcut, fs, order)


def compute_norm_factor(signal: np.ndarray, eps: float = 1e-8) -> float:
    """
    Tinh normalization factor: max absolute value cua toan bo signal.

    Dung cho record-level normalization de bao ton quan he biên do clean/noisy.

    Args:
        signal: 1D float array (full filtered record)
        eps:    Small value de tranh chia cho 0

    Returns:
        factor: float, max abs value
    """
    factor = float(np.max(np.abs(signal)))
    return max(factor, eps)


# ─── 2. Windowing ─────────────────────────────────────────────────────────────

def extract_window_positions(
    signal_length: int,
    window_size: int = 1024,
    stride: int = 512,
) -> List[Tuple[int, int]]:
    """
    Tinh cac vi tri (start, end) cua cac sliding windows.

    Khong dung zero-padding. Window cuoi bi loai neu khong du do dai.
    Khong co R-peak centering — day la waveform-level denoising.

    Args:
        signal_length: Tong so samples trong signal
        window_size:   Do dai moi window (default 1024)
        stride:        Buoc nhay (default 512 = 50% overlap)

    Returns:
        positions: List of (start, end) pairs
    """
    positions = []
    start = 0
    while start + window_size <= signal_length:
        positions.append((start, start + window_size))
        start += stride
    return positions


# ─── 3. SNR calculation va noise injection ────────────────────────────────────

def compute_snr_db(clean_window: np.ndarray, noise_window: np.ndarray) -> float:
    """
    Tinh SNR theo dinh nghia:
        SNR_dB = 10 * log10(P_clean / P_noise)
    voi P = mean(x^2) (mean square power, bao gom DC).

    Luu y: Clean reference da duoc high-pass filter (lowcut 0.05 Hz),
    nen DC component da duoc loai truoc.

    Args:
        clean_window: 1D array (window_size,)
        noise_window: 1D array (window_size,)

    Returns:
        snr_db: float
    """
    p_clean = float(np.mean(clean_window ** 2))
    p_noise = float(np.mean(noise_window ** 2))
    if p_noise < 1e-12:
        return float('inf')  # Noise co power gan bang 0
    if p_clean < 1e-12:
        return float('-inf')  # Clean signal co power gan bang 0
    return 10.0 * np.log10(p_clean / p_noise)


def compute_noise_alpha(
    clean_window: np.ndarray,
    noise_window: np.ndarray,
    target_snr_db: float,
) -> float:
    """
    Tinh he so scaling alpha de dat target SNR.

    SNR_dB = 10 * log10(P_clean / (alpha^2 * P_noise))
    => alpha = sqrt(P_clean / (P_noise * 10^(SNR_dB/10)))

    Args:
        clean_window:  1D float array
        noise_window:  1D float array (chua scaled)
        target_snr_db: Target SNR in dB

    Returns:
        alpha: float >= 0
    """
    p_clean = float(np.mean(clean_window ** 2))
    p_noise = float(np.mean(noise_window ** 2))
    if p_noise < 1e-12:
        return 0.0  # Noise bang 0, alpha vo nghia
    if p_clean < 1e-12:
        return 0.0  # Clean signal bang 0
    target_ratio = 10.0 ** (target_snr_db / 10.0)
    alpha = float(np.sqrt(p_clean / (p_noise * target_ratio)))
    return alpha


def inject_noise(
    clean_window: np.ndarray,
    noise_window: np.ndarray,
    target_snr_db: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Inject scaled noise vao clean_window de dat target SNR.

    Args:
        clean_window:  1D float array, normalized clean ECG
        noise_window:  1D float array, noise segment (chua scaled)
        target_snr_db: Target SNR in dB

    Returns:
        noisy_window:    clean_window + alpha * noise_window
        alpha:           Scaling factor
        measured_snr_db: SNR thuc te sau khi inject
    """
    if len(clean_window) != len(noise_window):
        raise ValueError(
            f"clean_window ({len(clean_window)}) va noise_window ({len(noise_window)}) "
            f"phai co cung chieu dai."
        )
    alpha = compute_noise_alpha(clean_window, noise_window, target_snr_db)
    scaled_noise = alpha * noise_window
    noisy_window = clean_window + scaled_noise
    measured_snr_db = compute_snr_db(clean_window, scaled_noise)
    return noisy_window.astype(np.float32), alpha, measured_snr_db


# ─── 4. PyTorch Dataset ──────────────────────────────────────────────────────

class DenoisingDataset(Dataset):
    """
    PyTorch Dataset cho Level 2 ECG Denoising.

    Modes:
        'train': On-the-fly random noise type va SNR selection.
                 Moi __getitem__ call tra ve 1 noisy+clean pair voi noise/SNR ngau nhien.

        'val' / 'test': Load pre-generated deterministic noisy samples tu NPZ.
                        Dam bao reproducibility cho evaluation.

    Returns:
        (noisy_window, clean_window): Tuple of float32 tensors, shape (1, window_size)
    """

    def __init__(
        self,
        clean_npz_path: str,
        split: str,
        noise_signals: Optional[Dict[str, np.ndarray]] = None,
        noise_blocks: Optional[Dict[str, Dict[str, List[Tuple[int, int]]]]] = None,
        snr_levels_db: List[float] = None,
        noisy_npz_path: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Args:
            clean_npz_path:  Path den file NPZ chua clean windows
                             Keys: 'windows' (N, W), 'record_ids', 'start_indices'
            split:           'train', 'val', hoac 'test'
            noise_signals:   Dict {'bw': array, 'ma': array, 'em': array}
                             Can thiet cho mode='train'
            noise_blocks:    Dict {'bw': {'train': [(s,e), ...], 'val': ..., 'test': ...}, ...}
                             Can thiet cho mode='train'
            snr_levels_db:   List of SNR values (default: [-6, 0, 6, 12, 18, 24])
            noisy_npz_path:  Path den pre-generated noisy NPZ cho val/test
                             Keys: 'noisy' (N*n_snr*n_noise, W), 'clean' (N*n_snr*n_noise, W)
            seed:            Base random seed cho training augmentation
        """
        self.split = split
        self.snr_levels_db = snr_levels_db or [-6, 0, 6, 12, 18, 24]
        self.seed = seed
        self.noise_types = ["bw", "ma", "em"]

        # Load clean windows
        data = np.load(clean_npz_path)
        self.clean_windows = data["windows"].astype(np.float32)  # (N, W)
        self.record_ids = data["record_ids"]
        self.start_indices = data["start_indices"]
        self.n_windows = len(self.clean_windows)
        self.window_size = self.clean_windows.shape[1]

        if split == "train":
            # On-the-fly mode: luu noise signals va blocks
            if noise_signals is None or noise_blocks is None:
                raise ValueError("'train' mode can noise_signals va noise_blocks")
            self.noise_signals = noise_signals
            self.noise_blocks = noise_blocks
        else:
            # Val/test mode: load pre-generated noisy samples
            if noisy_npz_path is None:
                raise ValueError(f"'{split}' mode can noisy_npz_path")
            noisy_data = np.load(noisy_npz_path)
            self.noisy_windows = noisy_data["noisy"].astype(np.float32)   # (N_total, W)
            self.paired_clean = noisy_data["clean"].astype(np.float32)   # (N_total, W)
            self.n_windows = len(self.noisy_windows)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.split == "train":
            return self._get_train_item(idx)
        else:
            return self._get_eval_item(idx)

    def _get_train_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """On-the-fly noise injection cho training."""
        clean = self.clean_windows[idx]  # (W,) float32

        # Random seed per sample = base_seed XOR idx (reproducible per idx)
        rng = np.random.default_rng(self.seed ^ idx)

        # Random noise type va SNR
        noise_type = self.noise_types[int(rng.integers(0, len(self.noise_types)))]
        target_snr = float(rng.choice(self.snr_levels_db))

        # Random noise block from train blocks
        train_blocks = self.noise_blocks[noise_type]["train"]
        block = train_blocks[int(rng.integers(0, len(train_blocks)))]
        block_start, block_end = block

        # Sample noise window
        from src.data.nstdb_loader import sample_noise_window
        noise_sig = self.noise_signals[noise_type]
        noise_window, _ = sample_noise_window(
            noise_sig, block_start, block_end, self.window_size, rng
        )

        # Inject noise
        noisy, _, _ = inject_noise(clean.astype(np.float64),
                                   noise_window, target_snr)

        return (
            torch.tensor(noisy, dtype=torch.float32).unsqueeze(0),    # (1, W)
            torch.tensor(clean, dtype=torch.float32).unsqueeze(0),    # (1, W)
        )

    def _get_eval_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load pre-generated sample cho val/test."""
        noisy = self.noisy_windows[idx]
        clean = self.paired_clean[idx]
        return (
            torch.tensor(noisy, dtype=torch.float32).unsqueeze(0),
            torch.tensor(clean, dtype=torch.float32).unsqueeze(0),
        )


def make_denoising_dataloader(
    dataset: DenoisingDataset,
    batch_size: int = 32,
    shuffle: bool = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Factory function tao DataLoader cho DenoisingDataset.
    """
    if shuffle is None:
        shuffle = (dataset.split == "train")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
