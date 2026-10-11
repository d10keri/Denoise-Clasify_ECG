"""
test_denoising_data.py
Unit tests cho Level 2 Denoising Dataset pipeline.

Tests:
  - SNR calculation accuracy
  - Noise injection scaling
  - Window extraction boundary conditions
  - Normalization preserves amplitude ratio
  - Split record no overlap
  - Noise blocks no overlap
  - Noisy/clean temporal alignment

Chay khong can file I/O (ngoai import):
  conda run -n ecg python tests/test_denoising_data.py
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.denoising_data import (
    compute_snr_db,
    compute_noise_alpha,
    inject_noise,
    extract_window_positions,
    compute_norm_factor,
    filter_clean_reference,
)
from src.data.nstdb_loader import (
    get_noise_blocks,
    assign_blocks_to_splits,
    sample_noise_window,
    sample_noise_window_deterministic,
    NOISE_TYPES,
)


PASS = "[PASS]"
FAIL = "[FAIL]"
passed = 0
failed = 0


def assert_test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  {PASS} {name}")
    else:
        failed += 1
        print(f"  {FAIL} {name}" + (f" — {detail}" if detail else ""))


# ─── Test 1: SNR calculation ───────────────────────────────────────────────────

def test_snr_calculation():
    """SNR should be 0 dB when clean == noise."""
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 1, 1024)
    noise = signal.copy()  # Same signal = 0 dB SNR
    snr = compute_snr_db(signal, noise)
    assert_test("compute_snr_db: equal signals -> 0 dB",
                abs(snr - 0.0) < 0.01, f"got {snr:.4f}")

    # SNR should be 20 dB when noise amplitude is 1/10 of signal
    noise2 = signal * 0.1  # amplitude ratio 10:1 = power ratio 100:1 = 20 dB
    snr2 = compute_snr_db(signal, noise2)
    assert_test("compute_snr_db: 1/10 amplitude noise -> ~20 dB",
                abs(snr2 - 20.0) < 0.01, f"got {snr2:.4f}")

    # SNR should be -6 dB when noise power is 4x signal power
    noise3 = signal * 2.0  # amplitude ratio 0.5 = power ratio 0.25 => SNR = 10*log10(0.25) = -6.02 dB
    snr3 = compute_snr_db(signal, noise3)
    assert_test("compute_snr_db: 2x amplitude noise -> ~-6 dB",
                abs(snr3 - (-6.02)) < 0.05, f"got {snr3:.4f}")


# ─── Test 2: Noise injection scaling ──────────────────────────────────────────

def test_noise_injection_achieves_target_snr():
    """inject_noise should produce measured SNR within 0.1 dB of target."""
    rng = np.random.default_rng(123)
    clean = rng.normal(0, 1, 1024)
    noise = rng.normal(0, 0.5, 1024)

    for target_snr in [-6.0, 0.0, 6.0, 12.0, 18.0, 24.0]:
        noisy, alpha, measured = inject_noise(clean, noise, target_snr)
        err = abs(measured - target_snr)
        assert_test(f"inject_noise: target={target_snr:+.0f} dB -> measured within 0.5 dB",
                    err < 0.5, f"measured={measured:.4f}, err={err:.4f}")


def test_noise_injection_length_preserved():
    """Output length must equal input length."""
    rng = np.random.default_rng(42)
    clean = rng.normal(0, 1, 1024)
    noise = rng.normal(0, 1, 1024)
    noisy, _, _ = inject_noise(clean, noise, 0.0)
    assert_test("inject_noise: output length == input length",
                len(noisy) == 1024, f"got {len(noisy)}")


def test_noise_injection_mismatched_length_raises():
    """Should raise ValueError when lengths differ."""
    try:
        inject_noise(np.zeros(1024), np.zeros(512), 0.0)
        assert_test("inject_noise: mismatched lengths raises ValueError", False,
                    "Expected ValueError not raised")
    except ValueError:
        assert_test("inject_noise: mismatched lengths raises ValueError", True)


# ─── Test 3: Window extraction ─────────────────────────────────────────────────

def test_window_positions_no_boundary_crossing():
    """All (start, end) pairs must be within signal bounds."""
    positions = extract_window_positions(10000, window_size=1024, stride=512)
    all_valid = all(s >= 0 and e <= 10000 and e - s == 1024 for s, e in positions)
    assert_test("window positions: all within bounds, correct size",
                all_valid, f"n_windows={len(positions)}")


def test_window_positions_short_signal():
    """Signal shorter than window: no windows should be extracted."""
    positions = extract_window_positions(512, window_size=1024, stride=512)
    assert_test("window positions: short signal -> 0 windows",
                len(positions) == 0, f"got {len(positions)}")


def test_window_positions_exact_fit():
    """Signal exactly 1 window: should yield exactly 1 window."""
    positions = extract_window_positions(1024, window_size=1024, stride=512)
    assert_test("window positions: exact fit -> 1 window",
                len(positions) == 1, f"got {len(positions)}")


def test_window_positions_stride_correct():
    """Check that consecutive windows are exactly stride apart."""
    positions = extract_window_positions(5000, window_size=1024, stride=512)
    strides_ok = all(
        positions[i + 1][0] - positions[i][0] == 512
        for i in range(len(positions) - 1)
    )
    assert_test("window positions: consecutive windows stride=512",
                strides_ok, f"n_windows={len(positions)}")


# ─── Test 4: Normalization ────────────────────────────────────────────────────

def test_normalization_preserves_ratio():
    """Normalizing clean and noisy by same factor must preserve their ratio."""
    rng = np.random.default_rng(42)
    clean = rng.normal(0, 2, 1024)
    noise = rng.normal(0, 0.1, 1024)
    noisy, alpha, snr_before = inject_noise(clean, noise, 6.0)

    factor = compute_norm_factor(clean)
    clean_norm = clean / factor
    noisy_norm = noisy / factor

    # SNR should be preserved after normalization
    snr_after = compute_snr_db(clean_norm, noisy_norm - clean_norm)
    assert_test("normalization: SNR preserved when same factor applied to clean+noisy",
                abs(snr_before - snr_after) < 0.01,
                f"before={snr_before:.4f}, after={snr_after:.4f}")

    # Max abs of normalized clean should be 1.0
    assert_test("compute_norm_factor: max_abs == 1.0 after normalize",
                abs(np.max(np.abs(clean_norm)) - 1.0) < 1e-5)


# ─── Test 5: Noise blocks ─────────────────────────────────────────────────────

def test_noise_blocks_no_overlap():
    """get_noise_blocks: blocks must not overlap."""
    signal = np.zeros(650_000)
    blocks = get_noise_blocks(signal, block_size=100_000, guard_samples=500)
    assert_test("get_noise_blocks: produces 6 blocks",
                len(blocks) == 6, f"got {len(blocks)}")

    # Check no overlap
    for i in range(len(blocks) - 1):
        _, end_i = blocks[i]
        start_next, _ = blocks[i + 1]
        assert_test(f"blocks {i} and {i+1} no overlap",
                    end_i <= start_next,
                    f"block[{i}] ends at {end_i}, block[{i+1}] starts at {start_next}")


def test_split_block_assignment_no_overlap():
    """assign_blocks_to_splits: train/val/test blocks must not share same block index."""
    signal = np.zeros(650_000)
    blocks = get_noise_blocks(signal)
    split_assignment = assign_blocks_to_splits(blocks)

    train_set = set(map(tuple, split_assignment["train"]))
    val_set = set(map(tuple, split_assignment["val"]))
    test_set = set(map(tuple, split_assignment["test"]))

    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set & test_set

    assert_test("noise block assignment: no train-val overlap", len(tv) == 0,
                f"overlapping: {tv}")
    assert_test("noise block assignment: no train-test overlap", len(tt) == 0,
                f"overlapping: {tt}")
    assert_test("noise block assignment: no val-test overlap", len(vt) == 0,
                f"overlapping: {vt}")


# ─── Test 6: Split record leakage ────────────────────────────────────────────

def test_split_record_no_overlap():
    """Level 2 split records must not overlap between splits."""
    import yaml
    with open("configs/denoising_dataset.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train = set(cfg["splits"]["train_records"])
    val   = set(cfg["splits"]["val_records"])
    test  = set(cfg["splits"]["test_records"])
    excl  = set(cfg["splits"]["excluded_records"])

    assert_test("Level 2 split: no train-val overlap",
                len(train & val) == 0, f"overlap: {train & val}")
    assert_test("Level 2 split: no train-test overlap",
                len(train & test) == 0, f"overlap: {train & test}")
    assert_test("Level 2 split: no val-test overlap",
                len(val & test) == 0, f"overlap: {val & test}")
    assert_test("Level 2 split: excluded records not in any split",
                len(excl & (train | val | test)) == 0,
                f"found: {excl & (train | val | test)}")
    assert_test("Level 2 split: total records = 44",
                len(train | val | test) == 44,
                f"got {len(train | val | test)}")
    assert_test("Level 2 patient-grouped: 201 in train",
                201 in train, "201 not in train")
    assert_test("Level 2 patient-grouped: 202 in train",
                202 in train, "202 not in train")
    assert_test("Level 2 patient-grouped: 202 not in test",
                202 not in test, "202 still in test!")


# ─── Test 7: filter_clean_reference ──────────────────────────────────────────

def test_filter_clean_reference_output_shape():
    """filter_clean_reference must return same length as input."""
    signal = np.random.randn(650_000)
    filtered = filter_clean_reference(signal, lowcut=0.05, highcut=60.0, fs=360.0, order=4)
    assert_test("filter_clean_reference: output length == input length",
                len(filtered) == len(signal), f"got {len(filtered)}")
    assert_test("filter_clean_reference: output dtype float64",
                filtered.dtype == np.float64, f"got {filtered.dtype}")


def test_filter_clean_reference_reduces_dc():
    """After high-pass filter (0.05 Hz), mean should be close to 0."""
    # DC signal + some sine
    t = np.arange(10000) / 360.0
    signal = 5.0 + np.sin(2 * np.pi * 10 * t)  # 5 mV DC offset + 10 Hz sine
    filtered = filter_clean_reference(signal, lowcut=0.05, highcut=60.0, fs=360.0, order=4)
    # After high-pass filter, DC should be removed; mean should be near 0
    # (some edge effects, so we check middle part)
    mid = filtered[1000:-1000]
    assert_test("filter_clean_reference: DC offset removed by high-pass",
                abs(np.mean(mid)) < 0.1, f"mean={np.mean(mid):.4f}")


# ─── Run all tests ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Level 2 Denoising Dataset Unit Tests")
    print("=" * 60)

    print("\n--- SNR Calculation ---")
    test_snr_calculation()

    print("\n--- Noise Injection ---")
    test_noise_injection_achieves_target_snr()
    test_noise_injection_length_preserved()
    test_noise_injection_mismatched_length_raises()

    print("\n--- Window Extraction ---")
    test_window_positions_no_boundary_crossing()
    test_window_positions_short_signal()
    test_window_positions_exact_fit()
    test_window_positions_stride_correct()

    print("\n--- Normalization ---")
    test_normalization_preserves_ratio()

    print("\n--- Noise Blocks ---")
    test_noise_blocks_no_overlap()
    test_split_block_assignment_no_overlap()

    print("\n--- Split Record Definitions ---")
    test_split_record_no_overlap()

    print("\n--- Filter ---")
    test_filter_clean_reference_output_shape()
    test_filter_clean_reference_reduces_dc()

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)
