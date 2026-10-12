"""
build_denoising_dataset.py
Entry point: xay dung Level 2 Denoising Dataset tu MITDB + NSTDB.

Pipeline:
  1. Load config
  2. Load va filter clean ECG records (MITDB)
  3. Extract sliding windows, normalize per-record
  4. Save clean windows NPZ + manifests CSV
  5. Load NSTDB noise recordings
  6. Assign noise blocks to splits, save noise_blocks.json
  7. Generate deterministic val + test noisy samples
  8. Save dataset_info.json va build_log.txt

Khong ghi de data/processed/ cua Phase 1.
Output: C:/d2l-en/data/ECG_Project_Data/Processed/denoising_v1/
"""

import argparse
import json
import csv
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pyrefly: ignore [missing-import]
from src.data.mitdb_loader import load_record
from src.data.nstdb_loader import (
    load_noise_record,
    get_noise_blocks,
    assign_blocks_to_splits,
    sample_noise_window_deterministic,
    describe_noise_record,
    NOISE_TYPES,
)
from src.data.denoising_data import (
    filter_clean_reference,
    compute_norm_factor,
    extract_window_positions,
    inject_noise,
    compute_snr_db,
)


# ─── Config helpers ──────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_output_dirs(output_dir: Path) -> dict:
    dirs = {
        "clean": output_dir / "clean_windows",
        "manifests": output_dir / "manifests",
        "noisy_eval": output_dir / "noisy_eval",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ─── Stage 1: Build clean windows ────────────────────────────────────────────

def build_clean_windows_for_split(
    record_ids: list,
    split_name: str,
    cfg: dict,
    log: list,
) -> tuple:
    """
    Load, filter, normalize, va extract windows cho tat ca records trong 1 split.

    Returns:
        windows:     (N, W) float32
        record_ids_arr: (N,) int32
        start_indices:  (N,) int64
        norm_factors:   (N,) float64  -- mot gia tri per window (record-level)
        manifest_rows:  list of dicts
    """
    mitdb_path = cfg["database"]["mitdb_path"]
    ch_idx = cfg["signal"]["channel_index"]
    fs = cfg["signal"]["sampling_rate"]
    lowcut = cfg["preprocessing"]["lowcut_hz"]
    highcut = cfg["preprocessing"]["highcut_hz"]
    order = cfg["preprocessing"]["butterworth_order"]
    win_size = cfg["window"]["size"]
    stride = cfg["window"]["stride"]

    all_windows, all_record_ids, all_starts, all_norms = [], [], [], []
    manifest_rows = []

    for rec_id in tqdm(record_ids, desc=f"Clean windows [{split_name}]"):
        try:
            raw_signal, _, _ = load_record(rec_id, ch_idx, "local", mitdb_path)
        except Exception as e:
            msg = f"[WARN] Record {rec_id}: load failed — {e}"
            print(msg); log.append(msg)
            continue

        # Step 1: Filter (0.05–60 Hz Butterworth, zero-phase)
        filtered = filter_clean_reference(raw_signal, lowcut, highcut, fs, order)

        # Step 2: Record-level normalization
        norm_factor = compute_norm_factor(filtered)
        normalized = (filtered / norm_factor).astype(np.float32)

        # Step 3: Extract windows
        positions = extract_window_positions(len(normalized), win_size, stride)
        if len(positions) == 0:
            msg = f"[WARN] Record {rec_id}: no windows extracted (signal too short?)"
            print(msg); log.append(msg)
            continue

        for (start, end) in positions:
            window = normalized[start:end]
            all_windows.append(window)
            all_record_ids.append(rec_id)
            all_starts.append(start)
            all_norms.append(norm_factor)
            manifest_rows.append({
                "record_id": rec_id,
                "lead": cfg["signal"]["lead"],
                "start_index": start,
                "end_index": end,
                "split": split_name,
                "sampling_rate": fs,
                "window_size": win_size,
                "stride": stride,
                "normalization_method": cfg["preprocessing"]["normalization"],
                "norm_factor": float(norm_factor),
            })

    if len(all_windows) == 0:
        return (
            np.empty((0, win_size), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
            manifest_rows,
        )

    return (
        np.stack(all_windows, axis=0).astype(np.float32),
        np.array(all_record_ids, dtype=np.int32),
        np.array(all_starts, dtype=np.int64),
        np.array(all_norms, dtype=np.float64),
        manifest_rows,
    )


# ─── Stage 2: Noise block assignment ─────────────────────────────────────────

def build_noise_block_assignments(cfg: dict, log: list) -> dict:
    """
    Load NSTDB noise recordings, tao blocks, assign to splits.
    Returns: {noise_type: {split: [(start, end), ...], 'signal': array}}
    """
    nstdb_path = cfg["database"]["nstdb_path"]
    block_size = cfg["noise"]["block_size"]
    guard = cfg["noise"]["guard_samples"]

    noise_assignment = {}
    for nt in NOISE_TYPES:
        info = describe_noise_record(nt, nstdb_path)
        log.append(f"Noise '{nt}': {info['n_samples']} samples, {info['fs']} Hz, "
                   f"{info['n_channels']} channels, {info['duration_sec']:.1f}s")

        signal = load_noise_record(nt, cfg["signal"]["noise_channel"], nstdb_path)
        blocks = get_noise_blocks(signal, block_size, guard)
        split_blocks = assign_blocks_to_splits(blocks)

        log.append(f"  Blocks: total={len(blocks)}, "
                   f"train={len(split_blocks['train'])}, "
                   f"val={len(split_blocks['val'])}, "
                   f"test={len(split_blocks['test'])}")

        noise_assignment[nt] = {
            "signal": signal,
            "train": split_blocks["train"],
            "val":   split_blocks["val"],
            "test":  split_blocks["test"],
        }
    return noise_assignment


# ─── Stage 3: Generate deterministic noisy samples ───────────────────────────

def generate_eval_noisy_samples(
    clean_windows: np.ndarray,        # (N, W) float32
    manifest_rows: list,
    noise_assignment: dict,
    cfg: dict,
    split_name: str,
    log: list,
) -> tuple:
    """
    Generate deterministic noisy samples cho val hoac test split.

    Moi clean window × N noise types × M SNR levels = N×M samples.
    Vi tri noise window duoc chon theo window_idx de deterministic.

    Returns:
        noisy_arr: (N * n_noise * n_snr, W) float32
        clean_arr: (N * n_noise * n_snr, W) float32
        meta_rows: list of dicts
    """
    snr_levels = cfg["noise"]["snr_levels_db"]
    win_size = cfg["window"]["size"]
    noise_types = cfg["noise"]["types"]
    step = cfg["window"]["stride"]  # step for deterministic noise window selection

    n_clean = len(clean_windows)
    n_combos = n_clean * len(noise_types) * len(snr_levels)

    noisy_list, clean_list, meta_rows = [], [], []

    for win_idx, (clean_w, mrow) in enumerate(
        tqdm(zip(clean_windows, manifest_rows),
             total=n_clean, desc=f"Generating noisy [{split_name}]")
    ):
        for noise_type in noise_types:
            blocks = noise_assignment[noise_type][split_name]
            if len(blocks) == 0:
                msg = f"[WARN] No {noise_type} blocks for split {split_name}"
                log.append(msg)
                continue
            # Use first block for deterministic generation
            block_start, block_end = blocks[0]

            for snr_idx, target_snr in enumerate(snr_levels):
                # Deterministic noise position: composite index
                det_idx = win_idx * len(snr_levels) + snr_idx
                noise_w, noise_start = sample_noise_window_deterministic(
                    noise_assignment[noise_type]["signal"],
                    block_start, block_end,
                    win_size,
                    window_idx=det_idx,
                    step=step,
                )
                noisy_w, alpha, measured_snr = inject_noise(
                    clean_w.astype(np.float64), noise_w, target_snr
                )
                noisy_list.append(noisy_w.astype(np.float32))
                clean_list.append(clean_w.astype(np.float32))
                meta_rows.append({
                    **mrow,
                    "noise_type": noise_type,
                    "noise_channel": cfg["signal"]["noise_channel"],
                    "noise_start_index": noise_start,
                    "noise_end_index": noise_start + win_size,
                    "target_snr_db": float(target_snr),
                    "measured_snr_db": float(measured_snr),
                    "noise_scale_alpha": float(alpha),
                    "random_seed": "deterministic",
                    "sample_id": (
                        f"{mrow['record_id']}_{mrow['start_index']}"
                        f"_{noise_type}_{int(target_snr):+d}dB"
                    ),
                })

    if len(noisy_list) == 0:
        return (
            np.empty((0, win_size), dtype=np.float32),
            np.empty((0, win_size), dtype=np.float32),
            [],
        )
    return (
        np.stack(noisy_list, axis=0),
        np.stack(clean_list, axis=0),
        meta_rows,
    )


# ─── Stage 4: Save functions ─────────────────────────────────────────────────

def save_clean_npz(path: Path, windows, record_ids, start_indices, norm_factors):
    np.savez_compressed(
        path,
        windows=windows,
        record_ids=record_ids,
        start_indices=start_indices,
        norm_factors=norm_factors,
    )
    size_mb = path.stat().st_size / 1e6
    print(f"  Saved {path.name}: shape={windows.shape}, size={size_mb:.1f} MB")


def save_noisy_npz(path: Path, noisy, clean):
    np.savez_compressed(path, noisy=noisy, clean=clean)
    size_mb = path.stat().st_size / 1e6
    print(f"  Saved {path.name}: noisy={noisy.shape}, clean={clean.shape}, "
          f"size={size_mb:.1f} MB")


def save_manifest_csv(path: Path, rows: list):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {path.name}: {len(rows)} rows")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build Level 2 ECG Denoising Dataset"
    )
    parser.add_argument(
        "--config", default="configs/denoising_dataset.yaml",
        help="Path to denoising_dataset.yaml"
    )
    args = parser.parse_args()

    t_start = time.time()
    log = [f"Build started: {datetime.now().isoformat()}"]
    print("=" * 60)
    print("Level 2 Denoising Dataset Builder")
    print("=" * 60)

    # 1. Load config
    cfg = load_config(args.config)
    output_dir = Path(cfg["paths"]["output_dir"])
    dirs = make_output_dirs(output_dir)
    print(f"Output dir: {output_dir}")

    seed = cfg["reproducibility"]["seed"]
    np.random.seed(seed)

    split_defs = {
        "train": cfg["splits"]["train_records"],
        "val":   cfg["splits"]["val_records"],
        "test":  cfg["splits"]["test_records"],
    }

    # Sanity check split definitions
    excluded = set(cfg["splits"]["excluded_records"])
    all_used = set(split_defs["train"] + split_defs["val"] + split_defs["test"])
    assert not (excluded & all_used), \
        f"Excluded records in split: {excluded & all_used}"
    assert not (set(split_defs["train"]) & set(split_defs["val"])), "Train-Val overlap"
    assert not (set(split_defs["train"]) & set(split_defs["test"])), "Train-Test overlap"
    assert not (set(split_defs["val"]) & set(split_defs["test"])), "Val-Test overlap"
    log.append(f"Split sanity check: PASS (train={len(split_defs['train'])}, "
               f"val={len(split_defs['val'])}, test={len(split_defs['test'])})")
    print("[PASS] Split sanity check")

    # 2. Build clean windows for all splits
    print("\n--- Stage 1: Building clean windows ---")
    clean_data = {}
    for split_name, record_ids in split_defs.items():
        windows, rec_ids_arr, starts, norms, manifest_rows = \
            build_clean_windows_for_split(record_ids, split_name, cfg, log)
        clean_data[split_name] = {
            "windows": windows,
            "record_ids": rec_ids_arr,
            "start_indices": starts,
            "norm_factors": norms,
            "manifest_rows": manifest_rows,
        }
        print(f"  {split_name}: {len(windows)} windows from {len(record_ids)} records")
        log.append(f"{split_name}: {len(windows)} clean windows from {len(record_ids)} records")

        # Save clean NPZ
        save_clean_npz(
            dirs["clean"] / f"{split_name}_clean.npz",
            windows, rec_ids_arr, starts, norms
        )
        # Save clean manifest
        save_manifest_csv(
            dirs["manifests"] / f"{split_name}_manifest.csv",
            manifest_rows
        )

    # 3. Noise block assignment
    print("\n--- Stage 2: Noise block assignment ---")
    noise_assignment = build_noise_block_assignments(cfg, log)

    # Save noise_blocks.json (without 'signal' arrays)
    noise_blocks_json = {}
    for nt, data in noise_assignment.items():
        noise_blocks_json[nt] = {
            "train": data["train"],
            "val":   data["val"],
            "test":  data["test"],
        }
    blocks_path = dirs["manifests"] / "noise_blocks.json"
    with open(blocks_path, "w", encoding="utf-8") as f:
        json.dump(noise_blocks_json, f, indent=2)
    print(f"  Saved noise_blocks.json")

    # 4. Generate deterministic val + test noisy samples
    print("\n--- Stage 3: Generating eval noisy samples (val + test) ---")
    for split_name in ["val", "test"]:
        d = clean_data[split_name]
        if len(d["windows"]) == 0:
            log.append(f"[WARN] {split_name}: no clean windows, skipping noisy gen")
            continue
        noisy_arr, clean_arr, meta_rows = generate_eval_noisy_samples(
            d["windows"], d["manifest_rows"], noise_assignment, cfg, split_name, log
        )
        save_noisy_npz(
            dirs["noisy_eval"] / f"{split_name}_noisy.npz",
            noisy_arr, clean_arr
        )
        save_manifest_csv(
            dirs["manifests"] / f"{split_name}_noisy_manifest.csv",
            meta_rows
        )
        n_snr = len(cfg["noise"]["snr_levels_db"])
        n_noise = len(cfg["noise"]["types"])
        log.append(f"{split_name} noisy: {len(noisy_arr)} samples "
                   f"({len(d['windows'])} clean × {n_noise} noise × {n_snr} SNR)")
        print(f"  {split_name}: {len(noisy_arr)} noisy samples generated")

    # 5. Save dataset_info.json
    print("\n--- Stage 4: Saving dataset info ---")
    snr_err_stats = {}
    for split_name in ["val", "test"]:
        mpath = dirs["manifests"] / f"{split_name}_noisy_manifest.csv"
        if mpath.exists():
            import csv as _csv
            errors = []
            with open(mpath, newline="", encoding="utf-8") as mf:
                reader = _csv.DictReader(mf)
                for row in reader:
                    try:
                        err = abs(float(row["measured_snr_db"]) - float(row["target_snr_db"]))
                        errors.append(err)
                    except (ValueError, KeyError):
                        pass
            if errors:
                snr_err_stats[split_name] = {
                    "mean_abs_error_db": float(np.mean(errors)),
                    "max_abs_error_db": float(np.max(errors)),
                    "n_samples": len(errors),
                }

    info = {
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "python_version": sys.version,
        "source_databases": {
            "clean_ecg": "MIT-BIH Arrhythmia Database (MITDB)",
            "noise": "MIT-BIH Noise Stress Test Database (NSTDB)",
        },
        "mitdb_path": cfg["database"]["mitdb_path"],
        "nstdb_path": cfg["database"]["nstdb_path"],
        "split_protocol": "Level 2 patient-grouped (201+202 both in train)",
        "excluded_records": cfg["splits"]["excluded_records"],
        "train_records": cfg["splits"]["train_records"],
        "val_records": cfg["splits"]["val_records"],
        "test_records": cfg["splits"]["test_records"],
        "preprocessing": {
            "filter": "Butterworth IIR bandpass, zero-phase (filtfilt)",
            "lowcut_hz": cfg["preprocessing"]["lowcut_hz"],
            "highcut_hz": cfg["preprocessing"]["highcut_hz"],
            "order": cfg["preprocessing"]["butterworth_order"],
            "normalization": cfg["preprocessing"]["normalization"],
        },
        "window": cfg["window"],
        "noise": {
            "types": cfg["noise"]["types"],
            "snr_levels_db": cfg["noise"]["snr_levels_db"],
            "block_size": cfg["noise"]["block_size"],
            "guard_samples": cfg["noise"]["guard_samples"],
            "channel": cfg["signal"]["noise_channel"],
        },
        "window_counts": {
            sp: int(len(clean_data[sp]["windows"])) for sp in split_defs
        },
        "snr_error_statistics": snr_err_stats,
        "seed": seed,
        "build_duration_sec": round(time.time() - t_start, 1),
        "build_log_summary": log[-10:],  # Last 10 log entries
    }
    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"  Saved dataset_info.json")

    # 6. Save build log
    log.append(f"Build completed: {datetime.now().isoformat()}")
    log.append(f"Total time: {time.time() - t_start:.1f}s")
    log_path = output_dir / "build_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log))

    # 7. Summary
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    for split_name in split_defs:
        n = len(clean_data[split_name]["windows"])
        recs = len(split_defs[split_name])
        print(f"  {split_name:5s}: {n:6,} clean windows from {recs} records")
    if snr_err_stats:
        for sp, stats in snr_err_stats.items():
            print(f"  {sp} SNR error: mean={stats['mean_abs_error_db']:.4f} dB, "
                  f"max={stats['max_abs_error_db']:.4f} dB")
    print(f"\nOutput: {output_dir}")
    print(f"Time: {time.time() - t_start:.1f}s")
    print(f"\nRun validation:")
    print(f"  conda run -n ecg python scripts/validate_denoising_dataset.py")


if __name__ == "__main__":
    main()
