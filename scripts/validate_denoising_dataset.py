"""
validate_denoising_dataset.py
Validation suite cho Level 2 Denoising Dataset.

Kiem tra:
  1. Data integrity (shapes, dtypes, NaN/Inf)
  2. Signal processing (SNR accuracy, normalization)
  3. Leakage (records, noise blocks, windows)
  4. Reproducibility (manifests match NPZ)
  5. Visual diagnostics (diagnostic plots)

Chay sau khi build_denoising_dataset.py hoan thanh.
"""

import sys
import csv
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.denoising_data import compute_snr_db, inject_noise


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


class ValidationReport:
    def __init__(self):
        self.results = []

    def record(self, name: str, passed: bool, detail: str = ""):
        status = PASS if passed else FAIL
        self.results.append({"name": name, "passed": passed, "detail": detail})
        print(f"  {status} {name}" + (f" — {detail}" if detail else ""))

    def summary(self) -> tuple:
        n_pass = sum(1 for r in self.results if r["passed"])
        n_fail = sum(1 for r in self.results if not r["passed"])
        return n_pass, n_fail

    def save(self, path: Path):
        lines = []
        for r in self.results:
            status = PASS if r["passed"] else FAIL
            lines.append(f"{status} {r['name']}" + (f" — {r['detail']}" if r["detail"] else ""))
        path.write_text("\n".join(lines), encoding="utf-8")


def main():
    t_start = time.time()
    parser_args = sys.argv[1:]
    config_path = "configs/denoising_dataset.yaml"
    for i, arg in enumerate(parser_args):
        if arg == "--config" and i + 1 < len(parser_args):
            config_path = parser_args[i + 1]

    cfg = load_config(config_path)
    output_dir = Path(cfg["paths"]["output_dir"])
    clean_dir = output_dir / "clean_windows"
    manifests_dir = output_dir / "manifests"
    noisy_eval_dir = output_dir / "noisy_eval"

    report = ValidationReport()
    split_defs = {
        "train": cfg["splits"]["train_records"],
        "val":   cfg["splits"]["val_records"],
        "test":  cfg["splits"]["test_records"],
    }
    win_size = cfg["window"]["size"]
    snr_levels = cfg["noise"]["snr_levels_db"]
    noise_types = cfg["noise"]["types"]
    excluded = set(cfg["splits"]["excluded_records"])

    # ── 1. File existence ──────────────────────────────────────────────────────
    section("1. File Existence")
    required_files = (
        [clean_dir / f"{sp}_clean.npz" for sp in split_defs]
        + [manifests_dir / f"{sp}_manifest.csv" for sp in split_defs]
        + [manifests_dir / "noise_blocks.json"]
        + [noisy_eval_dir / f"{sp}_noisy.npz" for sp in ["val", "test"]]
        + [output_dir / "dataset_info.json"]
    )
    for fpath in required_files:
        report.record(f"File exists: {fpath.name}", fpath.exists(), str(fpath))

    # ── 2. Clean window shape and dtype ───────────────────────────────────────
    section("2. Clean Window Shape & Dtype")
    clean_data = {}
    for sp in split_defs:
        fpath = clean_dir / f"{sp}_clean.npz"
        if not fpath.exists():
            report.record(f"{sp}_clean.npz shape", False, "File missing")
            continue
        data = np.load(fpath)
        windows = data["windows"]
        rec_ids = data["record_ids"]
        starts = data["start_indices"]

        shape_ok = windows.ndim == 2 and windows.shape[1] == win_size
        report.record(f"{sp} windows shape", shape_ok,
                      f"{windows.shape} (expected (N, {win_size}))")

        dtype_ok = windows.dtype == np.float32
        report.record(f"{sp} windows dtype float32", dtype_ok, str(windows.dtype))

        nan_ok = not np.any(np.isnan(windows)) and not np.any(np.isinf(windows))
        report.record(f"{sp} no NaN/Inf in clean windows", nan_ok)

        count_ok = len(windows) > 0
        report.record(f"{sp} has windows", count_ok, f"{len(windows)} windows")

        # Record IDs match split definition
        expected_ids = set(split_defs[sp])
        actual_ids = set(rec_ids.tolist())
        ids_ok = expected_ids == actual_ids
        report.record(f"{sp} record IDs match config",
                      ids_ok,
                      f"expected={sorted(expected_ids)}, got={sorted(actual_ids)}" if not ids_ok else "")

        # Excluded records not present
        exc_ok = len(excluded & actual_ids) == 0
        report.record(f"{sp} no excluded records",
                      exc_ok,
                      f"Found: {excluded & actual_ids}" if not exc_ok else "")

        clean_data[sp] = {"windows": windows, "record_ids": rec_ids, "start_indices": starts}

    # ── 3. Record leakage ─────────────────────────────────────────────────────
    section("3. Record Leakage")
    if all(sp in clean_data for sp in split_defs):
        for sp_a, sp_b in [("train", "val"), ("train", "test"), ("val", "test")]:
            ids_a = set(clean_data[sp_a]["record_ids"].tolist())
            ids_b = set(clean_data[sp_b]["record_ids"].tolist())
            overlap = ids_a & ids_b
            report.record(f"No record overlap {sp_a}-{sp_b}",
                          len(overlap) == 0,
                          f"Overlapping records: {overlap}" if overlap else "")

    # 201 and 202 both in train (Level 2 patient-grouped)
    if "train" in clean_data:
        train_ids = set(clean_data["train"]["record_ids"].tolist())
        both_in_train = (201 in train_ids) and (202 in train_ids)
        report.record("Records 201 and 202 both in train (patient-grouped)",
                      both_in_train,
                      f"train_ids contains 201: {201 in train_ids}, 202: {202 in train_ids}")

    # ── 4. Noise block leakage ─────────────────────────────────────────────────
    section("4. Noise Block Leakage")
    blocks_path = manifests_dir / "noise_blocks.json"
    if blocks_path.exists():
        with open(blocks_path, "r", encoding="utf-8") as f:
            noise_blocks = json.load(f)

        for nt in noise_types:
            if nt not in noise_blocks:
                report.record(f"Noise type '{nt}' in blocks.json", False)
                continue
            bdata = noise_blocks[nt]
            # Check no overlap between splits
            for sp_a, sp_b in [("train", "val"), ("train", "test"), ("val", "test")]:
                a_intervals = set(map(tuple, bdata.get(sp_a, [])))
                b_intervals = set(map(tuple, bdata.get(sp_b, [])))
                overlap = a_intervals & b_intervals
                report.record(f"{nt}: no block overlap {sp_a}-{sp_b}",
                              len(overlap) == 0,
                              f"Overlapping blocks: {overlap}" if overlap else "")

    # ── 5. Val/test noisy sample checks ───────────────────────────────────────
    section("5. Val/Test Noisy Sample Checks")
    for sp in ["val", "test"]:
        fpath = noisy_eval_dir / f"{sp}_noisy.npz"
        if not fpath.exists():
            report.record(f"{sp} noisy.npz exists", False)
            continue
        ndata = np.load(fpath)
        noisy = ndata["noisy"]
        clean = ndata["clean"]

        # Shape consistency
        shape_ok = (noisy.shape == clean.shape and noisy.ndim == 2
                    and noisy.shape[1] == win_size)
        report.record(f"{sp} noisy/clean shape consistent",
                      shape_ok, f"noisy={noisy.shape}, clean={clean.shape}")

        # No NaN/Inf
        nan_ok = (not np.any(np.isnan(noisy)) and not np.any(np.isinf(noisy))
                  and not np.any(np.isnan(clean)) and not np.any(np.isinf(clean)))
        report.record(f"{sp} no NaN/Inf in noisy samples", nan_ok)

        # Count: should be n_clean × n_noise × n_snr (or close to it)
        if sp in clean_data:
            n_clean_expected = len(clean_data[sp]["windows"])
            n_expected = n_clean_expected * len(noise_types) * len(snr_levels)
            count_ok = len(noisy) == n_expected
            report.record(f"{sp} expected sample count",
                          count_ok,
                          f"{len(noisy)} (expected {n_expected})")

    # ── 6. SNR accuracy from manifest ─────────────────────────────────────────
    section("6. SNR Accuracy")
    snr_tolerance_db = 0.5
    for sp in ["val", "test"]:
        mpath = manifests_dir / f"{sp}_noisy_manifest.csv"
        if not mpath.exists():
            report.record(f"{sp} SNR manifest exists", False)
            continue
        errors = []
        with open(mpath, newline="", encoding="utf-8") as mf:
            reader = csv.DictReader(mf)
            for row in reader:
                try:
                    target = float(row["target_snr_db"])
                    measured = float(row["measured_snr_db"])
                    errors.append(abs(measured - target))
                except (ValueError, KeyError):
                    pass
        if errors:
            mean_err = float(np.mean(errors))
            max_err = float(np.max(errors))
            all_ok = max_err <= snr_tolerance_db
            report.record(f"{sp} SNR error within ±{snr_tolerance_db} dB",
                          all_ok,
                          f"mean={mean_err:.4f} dB, max={max_err:.4f} dB, n={len(errors)}")

    # ── 7. Manifest row count matches NPZ ─────────────────────────────────────
    section("7. Manifest Row Count vs NPZ")
    for sp in split_defs:
        mpath = manifests_dir / f"{sp}_manifest.csv"
        npath = clean_dir / f"{sp}_clean.npz"
        if mpath.exists() and npath.exists():
            with open(mpath, newline="", encoding="utf-8") as mf:
                n_rows = sum(1 for _ in csv.DictReader(mf))
            n_windows = len(np.load(npath)["windows"])
            ok = n_rows == n_windows
            report.record(f"{sp} manifest rows == NPZ windows",
                          ok, f"manifest={n_rows}, npz={n_windows}")

    # ── 8. Normalization check ─────────────────────────────────────────────────
    section("8. Normalization Check")
    for sp in ["train", "val"]:
        if sp not in clean_data:
            continue
        windows = clean_data[sp]["windows"]
        if len(windows) == 0:
            continue
        # All windows should have max abs <= 1.0 (normalized by record max)
        max_abs = float(np.max(np.abs(windows)))
        norm_ok = max_abs <= 1.0 + 1e-5
        report.record(f"{sp} clean windows max_abs <= 1.0",
                      norm_ok, f"max_abs={max_abs:.6f}")

        # Windows should not be all zero
        nonzero_ok = float(np.mean(np.abs(windows))) > 1e-4
        report.record(f"{sp} clean windows not all zero", nonzero_ok)

    # ── 9. Visual diagnostics ─────────────────────────────────────────────────
    section("9. Visual Diagnostics")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Plot: clean window example + noisy versions
    if "val" in clean_data and (noisy_eval_dir / "val_noisy.npz").exists():
        try:
            windows = clean_data["val"]["windows"]
            rec_ids = clean_data["val"]["record_ids"]
            ndata = np.load(noisy_eval_dir / "val_noisy.npz")
            noisy_all = ndata["noisy"]
            clean_all = ndata["clean"]

            n_snr = len(snr_levels)
            n_noise_t = len(noise_types)
            n_per_clean = n_noise_t * n_snr

            # Pick first clean window and its noisy versions
            sample_clean = windows[0]
            # noisy_all rows are ordered: win_idx × noise_type × snr_level
            sample_noisy_rows = noisy_all[0:n_per_clean]
            sample_labels = [
                f"{nt} {snr:+d}dB"
                for nt in noise_types
                for snr in snr_levels
            ]

            fs = cfg["signal"]["sampling_rate"]
            t = np.arange(win_size) / fs

            n_plots = 1 + min(n_per_clean, 6)  # clean + up to 6 noisy
            fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots), dpi=80)
            if n_plots == 1:
                axes = [axes]

            axes[0].plot(t, sample_clean, "b-", linewidth=0.8, label="Clean reference")
            axes[0].set_title(
                f"Clean reference — Record {rec_ids[0]}, "
                f"val split (filter: [{cfg['preprocessing']['lowcut_hz']}–"
                f"{cfg['preprocessing']['highcut_hz']}] Hz)"
            )
            axes[0].set_ylabel("Normalized amplitude")
            axes[0].legend(loc="upper right")
            axes[0].grid(True, alpha=0.3)

            for i, (noisy_row, label) in enumerate(
                zip(sample_noisy_rows[:6], sample_labels[:6])
            ):
                axes[i + 1].plot(t, noisy_row, "r-", linewidth=0.5, alpha=0.8, label=f"Noisy: {label}")
                axes[i + 1].plot(t, clean_all[i], "b-", linewidth=0.8, alpha=0.5, label="Clean")
                axes[i + 1].set_title(f"Noisy input: {label}")
                axes[i + 1].set_ylabel("Normalized amplitude")
                axes[i + 1].legend(loc="upper right", fontsize=7)
                axes[i + 1].grid(True, alpha=0.3)

            axes[-1].set_xlabel("Time (s)")
            plt.tight_layout()
            plot_path = plots_dir / "sample_clean_noisy_pairs.png"
            plt.savefig(plot_path, dpi=80, bbox_inches="tight")
            plt.close()
            report.record("Diagnostic plot saved", True, str(plot_path))
            print(f"  Plot saved: {plot_path}")
        except Exception as e:
            report.record("Diagnostic plot", False, str(e))
    else:
        report.record("Diagnostic plot (skipped, data missing)", True)

    # ── Final summary ──────────────────────────────────────────────────────────
    section("VALIDATION SUMMARY")
    n_pass, n_fail = report.summary()
    total = n_pass + n_fail
    print(f"\n  Passed: {n_pass}/{total}")
    print(f"  Failed: {n_fail}/{total}")
    print(f"  Time:   {time.time() - t_start:.1f}s")

    # Save validation report
    val_report_path = output_dir / "validation_report.txt"
    report.save(val_report_path)
    print(f"\n  Validation report saved: {val_report_path}")

    if n_fail > 0:
        print(f"\n  {FAIL} {n_fail} checks failed. Review above output.")
        sys.exit(1)
    else:
        print(f"\n  {PASS} All {n_pass} checks passed.")


if __name__ == "__main__":
    main()
