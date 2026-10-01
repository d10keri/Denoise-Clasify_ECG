"""
build_dataset.py
Entry point để build MITDB classification dataset cho Phase 1.
"""
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys

import yaml

# Cần fix import path vì script được chạy từ thư mục gốc
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mitdb_loader   import iter_records
from src.data.preprocessing  import preprocess_signal
from src.data.segmentation   import segment_record
from src.data.splits         import get_split_records
from src.data.class_stats    import print_distribution, compute_weights


def build_split(record_ids, cfg, split_name):
    """Build X, y, meta cho 1 split."""
    X_all, y_all = [], []
    rec_ids_all, beat_idx_all, orig_syms_all = [], [], []
    
    source     = cfg['database']['source']
    local_path = cfg['database'].get('local_path')
    ch_idx     = cfg['signal']['channel_index']
    pp_cfg     = cfg['preprocessing']

    for rec_id, signal, ann_samples, ann_symbols in tqdm(
        iter_records(record_ids, ch_idx, source, local_path),
        total=len(record_ids), desc=f"Processing {split_name}"
    ):
        filtered = preprocess_signal(
            signal,
            filter_type='butterworth',  # Chỉ dùng butterworth theo correction
            lowcut=pp_cfg.get('lowcut_hz', 0.5),
            highcut=pp_cfg.get('highcut_hz', 60.0),
            butterworth_order=pp_cfg.get('butterworth_order', 4),
        )
        
        X, y, meta = segment_record(
            filtered, ann_samples, ann_symbols,
            record_id=rec_id,
            before_r=cfg['window']['before_r'],
            after_r=cfg['window']['after_r'],
        )
        
        X_all.append(X)
        y_all.append(y)
        rec_ids_all.append(meta['record_ids'])
        beat_idx_all.append(meta['beat_indices'])
        orig_syms_all.extend(meta['original_symbols'])
    
    if len(X_all) > 0:
        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)
        meta = {
            'record_ids':       np.concatenate(rec_ids_all),
            'beat_indices':     np.concatenate(beat_idx_all),
            'original_symbols': orig_syms_all,
        }
    else:
        X_all = np.empty((0, cfg['window']['size']), dtype=np.float32)
        y_all = np.empty((0,), dtype=np.int32)
        meta = {
            'record_ids':       np.empty((0,), dtype=np.int32),
            'beat_indices':     np.empty((0,), dtype=np.int64),
            'original_symbols': [],
        }
    return X_all, y_all, meta


def save_split(out_dir, split_name, X, y, meta):
    """Lưu 1 split thành NPZ."""
    out_path = Path(out_dir) / f"{split_name}.npz"
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        record_ids=meta['record_ids'],
        beat_indices=meta['beat_indices'],
    )
    print(f"  Saved {split_name}.npz: X={X.shape}, y={y.shape} -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/dataset_config.yaml')
    parser.add_argument('--seed',   type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    splits      = get_split_records()
    processed   = Path(cfg['paths']['processed_dir'])
    metadata    = Path(cfg['paths']['metadata_dir'])
    processed.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    
    # Build từng split
    results = {}
    for split_name in ['train', 'val', 'test']:
        X, y, meta = build_split(splits[split_name], cfg, split_name)
        results[split_name] = (X, y, meta)
    
    # No oversampling in Phase 1 (as requested)
    X_tr, y_tr, meta_tr = results['train']
    
    # Save NPZ
    print("\nSaving NPZ files...")
    save_split(processed, 'train', X_tr, y_tr, meta_tr)
    save_split(processed, 'val',   *results['val'])
    save_split(processed, 'test',  *results['test'])
    
    # Print distributions & compute weights
    weights = {}
    for split_name in ['train', 'val', 'test']:
        _, y, _ = results[split_name]
        print_distribution(y, split_name)
        if split_name == 'train':
            weights['class_weights'] = compute_weights(y).tolist()
    
    # Save metadata JSON
    pp_cfg = cfg['preprocessing']
    info = {
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "python_version": sys.version,
        "source_database": "MIT-BIH Arrhythmia Database",
        "mitdb_local_path": cfg['database'].get('local_path'),
        "excluded_records": [102, 104, 107, 217],
        "train_records": splits['train'],
        "val_records":   splits['val'],
        "test_records":  splits['test'],
        "window_size":   cfg['window']['size'],
        "before_r":      cfg['window']['before_r'],
        "after_r":       cfg['window']['after_r'],
        "sampling_rate": cfg['signal']['sampling_rate'],
        "lead":          cfg['signal']['lead'],
        "channel_index": cfg['signal']['channel_index'],
        "normalization": pp_cfg.get('normalization', 'per_beat_zscore'),
        "preprocessing": {
            "filter_type":        "butterworth",
            "lowcut_hz":          pp_cfg.get('lowcut_hz', 0.5),
            "highcut_hz":         pp_cfg.get('highcut_hz', 60.0),
            "butterworth_order":  pp_cfg.get('butterworth_order', 4),
        },
        "class_names":   ['N', 'S', 'V', 'F', 'Q'],
        "split_protocol": "De Chazal 2004 inter-patient DS1/DS2",
        "n_train_beats": int(len(y_tr)),
        "n_val_beats":   int(len(results['val'][1])),
        "n_test_beats":  int(len(results['test'][1])),
        **weights,
    }
    
    with open(metadata / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    print(f"\n[PASS] Metadata saved -> {metadata / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
