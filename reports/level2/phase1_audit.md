# BÁO CÁO KIỂM TRA PHASE 1 (AUDIT) — Chuẩn bị cho Level 2 Denoising Dataset

**Ngày kiểm tra:** 2026-10-11
**Người thực hiện:** Antigravity AI
**Phiên bản kiểm tra:** 1.0
**Mục đích:** Kiểm tra toàn bộ codebase Phase 1 trước khi thiết kế pipeline Level 2 denoising. Không sửa đổi bất kỳ file Phase 1 nào.

---

## 1. CẤU TRÚC REPOSITORY HIỆN TẠI

```
Denoise-Clasify_ECG/
├── configs/
│   ├── config.yaml              (1 line: data_root path)
│   ├── dataset_config.yaml      (52 lines — Phase 1 full config)
│   └── mlp_config.yaml          (29 lines — MLP model config)
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mitdb_loader.py      (63 lines)
│   │   ├── preprocessing.py     (109 lines)
│   │   ├── segmentation.py      (85 lines)
│   │   ├── splits.py            (46 lines)
│   │   ├── class_stats.py
│   │   └── ecg_dataset.py       (96 lines — PyTorch Dataset)
│   ├── models/
│   │   ├── mlp.py
│   │   ├── cnn1d.py
│   │   ├── denoise_net.py       (EMPTY — 0 bytes, placeholder)
│   │   └── classifier.py        (empty)
│   └── dataset.py
├── scripts/
│   └── build_dataset.py         (172 lines)
├── tests/
│   ├── test_dataset.py          (60 lines)
│   ├── test_leakage.py          (160 lines)
│   └── test_shapes.py
├── data/
│   ├── processed/
│   │   ├── train.npz            (39.6 MB)
│   │   ├── val.npz              (9.6 MB)
│   │   └── test.npz             (48.1 MB)
│   ├── metadata/
│   └── raw/                     (empty — data at external path)
├── reports/
│   └── phase1_phase2_report.md
├── plans/
│   ├── dataset_plan.md
│   └── phase_2_mlp.md
├── train.py, train_cnn1d.py, evaluate.py, evaluate_cnn1d.py
└── requirements.txt
```

---

## 2. XAC NHAN THONG SO THUC SU DUOC IMPLEMENT

### 2.1 Data paths (verified from source code)

| Resource | Actual path |
|---|---|
| MITDB local | `C:\d2l-en\data\ECG_Project_Data\mitdb` |
| NSTDB local | `C:\d2l-en\data\ECG_Project_Data\nstdb` |
| Processed (Phase 1) | `data/processed/` (relative to project root) |
| Metadata | `data/metadata/` |
| Main config | `configs/dataset_config.yaml` |

**NSTDB is already available at `C:\d2l-en\data\ECG_Project_Data\nstdb`** — no download needed.

### 2.2 Preprocessing parameters actually implemented

Verified from `src/data/preprocessing.py` and `scripts/build_dataset.py`:

| Parameter | Actual value | Matches report? |
|---|---|---|
| Filter type | Butterworth IIR (filtfilt, zero-phase) | YES |
| Lowcut | 0.5 Hz | YES |
| Highcut | 60.0 Hz | YES |
| Order | 4 (bandpass = 8-pole effective) | YES |
| Window size | 260 samples | YES |
| Before R-peak | 99 samples | YES |
| After R-peak | 161 samples | YES |
| Normalization | Per-beat Z-score (mean=0, std=1) | YES |
| Sampling rate | 360 Hz | YES |
| Lead | MLII (channel_index=0) | YES |
| Output dtype | float32 | YES |

**Conclusion:** Report and implementation are fully synchronized.

**IMPORTANT NOTE FOR LEVEL 2:** Z-score normalization in Phase 1 is applied INSIDE `segmentation.py` (lines 63-64), not as a separate step. The clean signal is normalized before saving to NPZ. This behavior CANNOT be reused for denoising because it destroys the amplitude relationship between clean and noisy signals.

### 2.3 Split definition (verified from `src/data/splits.py`)

| Split | Records | Count | Beats (from report) |
|---|---|---|---|
| Train | 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 205, 207, 208, 209, 215 | 18 | 41,116 |
| Val | 203, 220, 223, 230 | 4 | 9,885 |
| Test | 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234 | 22 | 49,691 |
| **Total** | | **44** | **100,692** |

**Excluded:** 102, 104, 107, 217 (paced rhythms)

Split protocol: **De Chazal 2004 inter-patient (DS1/DS2)**:
- DS1 (22 records) → Train (18) + Val (4)
- DS2 (22 records) → Test (all 22)

---

## 3. COMPONENT ANALYSIS

### 3.1 Components safe to reuse in Level 2

| Component | File | Reuse | Reason |
|---|---|---|---|
| MITDB loader | `src/data/mitdb_loader.py` | YES — fully | Loads raw signal without normalization |
| Butterworth filter | `src/data/preprocessing.py::bandpass_butterworth` | YES — directly | Needed for clean-reference construction |
| FIR filter | `src/data/preprocessing.py::bandpass_fir` | YES — as option | Alternative filter |
| Split record lists | `src/data/splits.py` | YES — read only | Reference record IDs only |

### 3.2 Classification-specific components — DO NOT reuse

| Component | Reason not to reuse |
|---|---|
| `segmentation.py` | R-peak centering + AAMI mapping + per-beat Z-score — all classification-specific |
| `ecg_dataset.py` | Returns `(X, y)` for classification |
| `build_dataset.py` | Beat extraction pipeline, not waveform windowing |
| `tests/test_dataset.py` | Tests classification NPZ format |
| Per-beat Z-score | Destroys amplitude relationship between clean/noisy pairs |

---

## 4. INVESTIGATION: EXCLUDED RECORDS

### 4.1 Why are 102, 104, 107, 217 excluded from Phase 1?

**Actual reason from literature:**
- These four records contain **paced beats** — heartbeats generated by a cardiac pacemaker.
- In the De Chazal 2004 protocol, they are excluded because the pacemaker artifact was not faithfully recorded in the original analog tape, making automated detection unreliable.
- The paced beat symbols (`/`, `f`) map to AAMI class Q (Unknown) — rare and unreliable.

**Is exclusion necessary for Level 2 (denoising)?**

| Factor | Assessment |
|---|---|
| Paced beats affect classification accuracy | Yes |
| Paced beats affect denoising task directly | NO — the ECG waveform itself has BW/MA/EM noise regardless of beat type |
| Signal quality of the 4 records | ECG waveform is valid; pacemaker artifact is an additional overlay |
| Should they be included for denoising? | Justified scientifically, but excluded for downstream compatibility |

**Decision for Level 2:** Exclude records 102, 104, 107, 217 to maintain compatibility with Phase 1 for downstream classification evaluation. This is a conservative choice that reduces 4 records from potential training data but ensures a coherent clean→denoise→classify pipeline.

The exclusion is NOT scientifically required for the denoising task itself. Future work could include these records in a denoising-only experiment.

### 4.2 Records 201 and 202 — Same patient, different splits!

**Critical finding:**

From `src/data/splits.py`:
- Record **201** → **TRAIN split** (DS1)
- Record **202** → **TEST split** (DS2)

Records 201 and 202 are confirmed to be from the **same male subject** in MITDB (48 recordings from 47 subjects — one person recorded twice).

**Why does this matter for strict patient-independent evaluation?**

| Split level | Implication |
|---|---|
| Record-level (Phase 1 current) | 201 and 202 in different splits — no window overlap |
| Patient-level (strict) | Same person — ECG morphology is similar — model may implicitly "recognize" the patient |

**For Phase 1 (classification):** The De Chazal 2004 protocol itself places 201 in DS1 and 202 in DS2. This is the standard benchmark protocol used by the field for comparison. **Do NOT change Phase 1 split.**

**For Level 2 (denoising):**
- If Phase 1 split is kept: patient-morphology leakage exists in test
- If strict patient-grouped split is used: 201 and 202 must be in the same partition

---

## 5. FILES MISSING FOR LEVEL 2

| File to create | Purpose |
|---|---|
| `configs/denoising_dataset.yaml` | Full Level 2 pipeline configuration |
| `src/data/nstdb_loader.py` | Load and inspect NSTDB noise recordings |
| `src/data/denoising_data.py` | Windowing, SNR scaling, paired data generation |
| `scripts/build_denoising_dataset.py` | Entry point to build denoising dataset |
| `scripts/validate_denoising_dataset.py` | Validation suite after build |
| `tests/test_denoising_data.py` | Unit tests for denoising pipeline |
| `reports/level2/level2_dataset_research_report.md` | Research report |
| `reports/level2/dataset_build_report.md` | Build report |
| `plan/level2_dataset_plan.md` | Implementation plan |

---

## 6. NSTDB DATA CONFIRMED

| Resource | Path | Status |
|---|---|---|
| MITDB | `C:\d2l-en\data\ECG_Project_Data\mitdb\` | EXISTS — 48 records |
| NSTDB bw.dat | `C:\d2l-en\data\ECG_Project_Data\nstdb\bw.dat` | EXISTS — 1,950,000 bytes |
| NSTDB ma.dat | `C:\d2l-en\data\ECG_Project_Data\nstdb\ma.dat` | EXISTS — 1,950,000 bytes |
| NSTDB em.dat | `C:\d2l-en\data\ECG_Project_Data\nstdb\em.dat` | EXISTS — 1,950,000 bytes |
| NSTDB preconstructed | `118e00`...`119e_6` (12 records) | EXISTS — reference only |

**NSTDB technical specs confirmed from header files:**
- 3 noise recordings (bw, ma, em): each has **650,000 samples, 360 Hz, 2 channels**
- Duration: ~30 minutes (650,000 / 360 = 1,805 seconds = 30.1 minutes)
- Preconstructed noisy records use records 118 and 119 + em noise at SNR: -6, 0, 6, 12, 18, 24 dB

---

## 7. LEAKAGE RISKS IN PHASE 1

| Leakage type | Phase 1 status |
|---|---|
| Record-level split | NO overlap (verified by tests) |
| Window-level overlap | NO overlap (R-peak centered, non-overlapping) |
| Patient-level leakage (201/202) | EXISTS by design (De Chazal protocol — documented limitation) |
| Normalization leakage | NONE (per-beat, no cross-sample fitting) |
| Test contamination | NONE (test never used for checkpointing) |

---

## 8. TWO SPLIT PROTOCOL OPTIONS FOR LEVEL 2

### Option A: Preserve Phase 1 Split (Record-level)

**Advantages:**
- Full compatibility with Level 1
- Easy downstream comparison for denoise → classify pipeline

**Disadvantages:**
- Records 201 (train) and 202 (test) from same patient — potential morphology leakage
- Not a true patient-independent evaluation for denoising

**When to use:** Baseline experiment for Phase 1 downstream comparison.

### Option B: Strict Patient-Grouped Split

**Advantages:**
- Scientifically correct patient-independent evaluation
- Denoising metrics reflect true generalization

**Disadvantages:**
- Not compatible with Phase 1 split
- Moving record 202 from test to train changes test composition for Level 2

**When to use:** Primary scientific evaluation protocol for denoising.

### Recommendation

**Primary Protocol for Level 2:** Option B — Strict patient-grouped split.
- Both 201 and 202 placed in TRAIN partition.
- Level 2 test set = DS2 records minus 202 plus no paced records.
- Phase 1 split is NOT changed.

**Secondary Protocol (Comparability Experiment):** Option A — same as Phase 1.
- Explicitly document the patient-leakage limitation.

> **IMPORTANT:** Option B applies ONLY to Level 2 dataset. Phase 1 split remains intact.

---

## 9. AUDIT SUMMARY

| Item | Result |
|---|---|
| Report vs. Code synchronized | YES — fully synchronized |
| Preprocessing parameters accurate | YES — verified from source |
| Data paths exist | YES — MITDB + NSTDB both present |
| Phase 1 split unchanged | YES — will not be modified |
| NSTDB specs confirmed | YES — 650K samples, 360 Hz, 2ch, 3 noise types |
| Phase 1 leakage tests pass | YES — per report |
| Placeholder files | denoise_net.py, classifier.py (empty) |
| Missing files for Level 2 | 9 new files to create |
| 201/202 same-patient issue | DOCUMENTED — will handle in Level 2 split |
| 102/104/107/217 exclusion | DECIDED — exclude for compatibility |

---

*Audit complete. No Phase 1 files were modified.*
