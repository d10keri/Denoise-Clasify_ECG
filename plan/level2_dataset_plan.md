# LEVEL 2 DENOISING DATASET — IMPLEMENTATION PLAN

**Version:** 1.0 | **Date:** 2026-10-11

---

## 1. NEW FILES TO CREATE

```
Denoise-Clasify_ECG/
├── configs/
│   └── denoising_dataset.yaml          [NEW] Full Level 2 configuration
├── src/
│   └── data/
│       ├── nstdb_loader.py             [NEW] Load NSTDB noise recordings
│       └── denoising_data.py           [NEW] Windowing, SNR, paired generation, PyTorch Dataset
├── scripts/
│   ├── build_denoising_dataset.py      [NEW] Entry point: build clean windows + manifests
│   └── validate_denoising_dataset.py   [NEW] Full validation suite
├── tests/
│   └── test_denoising_data.py          [NEW] Unit tests
├── plan/
│   └── level2_dataset_plan.md          THIS FILE
└── Processed/denoising_v1/             [NEW OUTPUT DIR — separate from Phase 1]
    ├── clean_windows/                   Pre-filtered, normalized clean windows
    ├── manifests/                       CSV manifests + noise block JSON
    ├── noisy_eval/                      Deterministic val/test noisy samples
    └── dataset_info.json                Full build metadata
```

---

## 2. CONFIGURATION (denoising_dataset.yaml)

```yaml
database:
  mitdb_path: "C:/d2l-en/data/ECG_Project_Data/mitdb"
  nstdb_path: "C:/d2l-en/data/ECG_Project_Data/nstdb"

signal:
  sampling_rate: 360
  lead: "MLII"
  channel_index: 0
  noise_channel: 0     # noise1 channel in NSTDB

preprocessing:
  filter_type: "butterworth"
  lowcut_hz: 0.05      # Lower than Phase 1 (0.5) — preserves BW content
  highcut_hz: 60.0
  butterworth_order: 4
  normalization: "record_max_abs"

window:
  size: 1024
  stride: 512

noise:
  types: ["bw", "ma", "em"]
  snr_levels_db: [-6, 0, 6, 12, 18, 24]
  block_size: 100000   # samples per noise block
  guard_samples: 500   # guard interval at block boundaries
  # Blocks 1-4 = train, 5 = val, 6 = test

splits:
  # Level 2 patient-grouped split
  # Records 201+202 both in train (same patient)
  train_records: [101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
                  122, 124, 201, 202, 205, 207, 208, 209, 215]
  val_records:   [203, 220, 223, 230]
  test_records:  [100, 103, 105, 111, 113, 117, 121, 123,
                  200, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
  excluded_records: [102, 104, 107, 217]

paths:
  output_dir: "C:/d2l-en/data/ECG_Project_Data/Processed/denoising_v1"

reproducibility:
  seed: 42
```

---

## 3. IMPLEMENTATION STEPS

### Step 1: `configs/denoising_dataset.yaml`
- Create the YAML config above
- All hyperparameters configurable

### Step 2: `src/data/nstdb_loader.py`
- `load_noise_record(noise_type, channel_idx, nstdb_path)` → `np.ndarray`
- `get_noise_blocks(noise_signal, block_size, guard_samples)` → list of (start, end) tuples
- `assign_blocks_to_splits(blocks)` → dict {train: [...], val: [...], test: [...]}
- `sample_noise_window(noise_signal, start, end, window_size)` → `np.ndarray`
- Raises ValueError if window_size > available block length

### Step 3: `src/data/denoising_data.py`
**Module 1: Signal operations**
- `filter_signal(signal, lowcut, highcut, fs, order)` — reuse from Phase 1 preprocessing.py
- `compute_norm_factor(signal)` → float (max abs value)
- `normalize(signal, factor)` → signal / factor
- `compute_snr_db(clean, noise)` → float
- `compute_alpha(clean_window, noise_segment, target_snr_db)` → float
- `inject_noise(clean_window, noise_segment, alpha)` → noisy_window

**Module 2: Windowing**
- `extract_windows(signal, window_size, stride)` → list of (start, end) tuples
- Returns non-crossing windows, drops incomplete at end

**Module 3: Dataset class (PyTorch)**
- `DenoisingDataset(clean_npz, manifest_csv, mode, noise_loader, snr_levels, seed)`
  - `mode='train'`: on-the-fly random noise/SNR selection
  - `mode='val'` or `'test'`: load pre-generated noisy_eval NPZ
  - Returns `(noisy_window, clean_window)` as float32 tensors

### Step 4: `scripts/build_denoising_dataset.py`
Entry point pipeline:
1. Load config
2. For each split and each record: load MITDB → filter → normalize → extract windows → save to NPZ
3. Generate noise block assignments → save to `noise_blocks.json`
4. Generate deterministic val/test noisy samples → save to `noisy_eval/`
5. Save manifests (CSV per split)
6. Save dataset_info.json
7. Print statistics

### Step 5: `scripts/validate_denoising_dataset.py`
Run all validation checks (see Stage H specification).

### Step 6: `tests/test_denoising_data.py`
Unit tests (importable, fast, no file I/O where possible):
- test_snr_calculation_accuracy
- test_noise_scaling_correctness
- test_window_extraction_no_boundary_crossing
- test_normalization_preserves_ratio
- test_split_record_no_overlap
- test_noise_blocks_no_overlap
- test_noisy_clean_aligned
- test_metadata_fields_present

---

## 4. VALIDATION CHECKLIST

After `build_denoising_dataset.py` runs:

### Data integrity
- [ ] Expected records present in each split
- [ ] No unexpected records
- [ ] All windows = 1024 samples
- [ ] No NaN/Inf values
- [ ] Metadata rows = sample count
- [ ] Clean/noisy pairs aligned

### Signal processing
- [ ] Sampling rate 360 Hz consistent
- [ ] Lead MLII consistent
- [ ] SNR measurement within ±0.5 dB of target
- [ ] Normalization preserves clean/noisy ratio
- [ ] Filter [0.05–60 Hz] applied consistently

### Leakage
- [ ] Clean records not shared between splits
- [ ] Records 201 and 202 both in train
- [ ] Noise blocks not overlapping across splits
- [ ] Windows not crossing noise block boundaries
- [ ] Synthetic copies stay in same split as clean source

### Reproducibility
- [ ] Config, seed, record lists saved
- [ ] Build command documented
- [ ] Re-running produces same manifests
- [ ] Val/test noisy samples deterministic

---

## 5. ESTIMATED DATASET SIZE

**Clean windows estimate:**
- ~30 minutes of ECG per record
- 44 records × 650,000 samples / 512 stride ≈ 44 × 1,270 windows ≈ 55,880 windows
- (Some records shorter or with boundary drops; actual will be ~50,000–60,000 windows)

**Storage (clean windows only):**
- ~55,000 windows × 1024 samples × 4 bytes (float32) = ~225 MB

**Noisy eval (val + test, 6 SNR × 3 noise types):**
- ~25 records × 1,270 × 18 = ~571,500 noisy samples × 1024 × 4 bytes ≈ 2.3 GB
- (This is large — may subset to 1 noise type × 6 SNR for pre-generated eval)

**Recommendation:** Pre-generate val noisy for offline inspection; generate test on-the-fly or store selectively.

---

## 6. DEPENDENCY NOTES

- Reuse `src/data/mitdb_loader.py::load_record` — no changes
- Reuse `src/data/preprocessing.py::bandpass_butterworth` — no changes
- New `src/data/nstdb_loader.py` — new, no overlap with Phase 1
- New `src/data/denoising_data.py` — new, no overlap
- `src/data/splits.py` — read TRAIN/VAL/TEST lists for reference; do NOT import DS1/DS2 directly (Level 2 has different split)
- All new output goes to `C:/d2l-en/data/ECG_Project_Data/Processed/denoising_v1/` — never overwrites Phase 1 `data/processed/`

---

*Plan version 1.0 — all decisions documented in `reports/level2/level2_dataset_research_report.md`.*
