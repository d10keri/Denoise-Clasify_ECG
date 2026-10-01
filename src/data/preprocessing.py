"""
preprocessing.py
Preprocessing Phase 1: Bandpass filter [0.5–60 Hz] + per-beat Z-score.
Python 3.9.17 compatible (scipy >= 1.7).
"""
import numpy as np
from scipy.signal import butter, filtfilt, firwin, lfilter
from typing import Literal


def bandpass_butterworth(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 60.0,
    fs: float = 360.0,
    order: int = 4,
) -> np.ndarray:
    """
    Bandpass Butterworth IIR filter [lowcut, highcut] Hz.
    Dùng filtfilt (zero-phase, forward-backward) để không gây phase shift.

    Lưu ý: order=4 với bandpass tạo ra bộ lọc 8-pole thực tế.
    Butterworth rất phổ biến trong ECG processing vì maximally flat passband.

    Args:
        signal:  1D float array, ECG signal thô
        lowcut:  lower cutoff Hz  (default 0.5 — loại baseline wander)
        highcut: upper cutoff Hz  (default 60.0 — loại powerline noise)
        fs:      sampling rate Hz (MITDB = 360)
        order:   filter order     (default 4)
    Returns:
        filtered: 1D float64 array, cùng shape với signal
    """
    nyq  = fs / 2.0
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return filtfilt(b, a, signal).astype(np.float64)


def bandpass_fir(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 60.0,
    fs: float = 360.0,
    numtaps: int = 101,
) -> np.ndarray:
    """
    Bandpass FIR filter [lowcut, highcut] Hz dùng windowed-sinc (Hamming).
    FIR có linear phase tự nhiên — không cần filtfilt.
    numtaps phải lẻ để filter có độ trễ nguyên mẫu (symmetric).

    Args:
        signal:  1D float array
        lowcut:  lower cutoff Hz
        highcut: upper cutoff Hz
        fs:      sampling rate Hz
        numtaps: số taps (phải lẻ, default 101)
    Returns:
        filtered: 1D float64 array, có độ trễ (numtaps-1)//2 mẫu
                  (không ảnh hưởng vì ta chỉ dùng R-peak positions từ annotation)
    """
    assert numtaps % 2 == 1, "numtaps phải là số lẻ"
    nyq  = fs / 2.0
    b    = firwin(numtaps, [lowcut / nyq, highcut / nyq], pass_zero=False)
    return lfilter(b, 1.0, signal).astype(np.float64)


def preprocess_signal(
    signal: np.ndarray,
    filter_type: Literal['butterworth', 'fir'] = 'butterworth',
    lowcut: float = 0.5,
    highcut: float = 60.0,
    fs: float = 360.0,
    butterworth_order: int = 4,
    fir_numtaps: int = 101,
) -> np.ndarray:
    """
    Entry point duy nhất cho preprocessing signal.
    Áp dụng bandpass filter [lowcut, highcut] Hz.

    Args:
        signal:           Raw ECG signal, 1D array
        filter_type:      'butterworth' (default) hoặc 'fir'
        lowcut:           0.5 Hz
        highcut:          60.0 Hz
        fs:               360.0 Hz
        butterworth_order: 4
        fir_numtaps:      101
    Returns:
        filtered signal, 1D float64
    """
    if filter_type == 'butterworth':
        return bandpass_butterworth(signal, lowcut, highcut, fs, butterworth_order)
    elif filter_type == 'fir':
        return bandpass_fir(signal, lowcut, highcut, fs, fir_numtaps)
    else:
        raise ValueError(f"filter_type phải là 'butterworth' hoặc 'fir', got '{filter_type}'")


def zscore_normalize(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-beat Z-score normalization.
    window: 1D array (260,)  →  mean=0, std=1
    """
    mean = np.mean(window)
    std  = np.std(window)
    return (window - mean) / (std + eps)
