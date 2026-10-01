"""
segmentation.py
R-peak centered beat segmentation + AAMI mapping + per-beat normalization.
"""
import numpy as np
from typing import Tuple, Dict

# AAMI EC57 mapping — nguồn: de Chazal 2004
AAMI_MAP: Dict[str, int] = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,            # SVEB
    'V': 2, 'E': 2,                              # VEB
    'F': 3,                                       # Fusion
    '/': 4, 'f': 4, 'Q': 4,                      # Unknown
}

CLASS_NAMES = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}

BEFORE_R = 99
AFTER_R  = 161   # total window = 260


def segment_record(signal: np.ndarray,
                   ann_samples: np.ndarray,
                   ann_symbols: list,
                   record_id: int,
                   before_r: int = BEFORE_R,
                   after_r: int = AFTER_R,
                   eps: float = 1e-8
                   ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Segment một record thành beats.

    Returns:
        X:    (N_beats, window_size) float32
        y:    (N_beats,)             int32
        meta: {record_ids, beat_indices, original_symbols}
    """
    beats_X     = []
    beats_y     = []
    beat_idx    = []
    orig_syms   = []
    
    sig_len      = len(signal)
    window_size  = before_r + after_r
    
    for r_pos, sym in zip(ann_samples, ann_symbols):
        # Bỏ qua annotation không phải beat (rhythm markers, etc.)
        if sym not in AAMI_MAP:
            continue
        
        start = int(r_pos) - before_r
        end   = int(r_pos) + after_r
        
        # Boundary check — bỏ beats gần biên
        if start < 0 or end > sig_len:
            continue
        
        window = signal[start:end].copy()        # (260,)
        
        # Per-beat Z-score normalization
        mean = np.mean(window)
        std  = np.std(window)
        window = (window - mean) / (std + eps)
        
        beats_X.append(window.astype(np.float32))
        beats_y.append(AAMI_MAP[sym])
        beat_idx.append(int(r_pos))
        orig_syms.append(sym)
    
    if len(beats_X) == 0:
        X   = np.empty((0, window_size), dtype=np.float32)
        y   = np.empty((0,),             dtype=np.int32)
    else:
        X   = np.stack(beats_X, axis=0).astype(np.float32)   # (N, 260)
        y   = np.array(beats_y,  dtype=np.int32)
    
    meta = {
        'record_ids':       np.full(len(beats_y), record_id, dtype=np.int32),
        'beat_indices':     np.array(beat_idx, dtype=np.int64),
        'original_symbols': orig_syms,
    }
    
    return X, y, meta
