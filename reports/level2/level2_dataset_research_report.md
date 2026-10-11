# BÁO CÁO NGHIÊN CỨU LEVEL 2 — ECG DENOISING DATASET

**Phiên bản:** 1.0 | **Ngày:** 2026-10-11 | **Ngôn ngữ báo cáo:** Tiếng Việt

---

## 1. PHẠM VI VÀ CÂU HỎI NGHIÊN CỨU

### 1.1 Mục tiêu

Xây dựng một dataset supervised denoising ECG có thể tái sử dụng và có căn cứ khoa học từ:
- **Clean ECG source:** MIT-BIH Arrhythmia Database (MITDB), 44 records (loại 4 paced records)
- **Noise source:** MIT-BIH Noise Stress Test Database (NSTDB) — 3 noise recordings: `bw`, `ma`, `em`
- **Model input:** noisy ECG waveform
- **Model target:** clean ECG waveform (filtered reference)

### 1.2 Câu hỏi nghiên cứu cụ thể

1. Nên dùng noise types nào và kết hợp như thế nào trong từng sample?
2. Nên dùng SNR grid nào cho training và testing?
3. Window length và stride tối ưu cho waveform-level denoising là bao nhiêu?
4. Pipeline preprocessing cho clean reference nên là gì — không được copy blindly từ Phase 1?
5. Làm thế nào để split noise source để tránh noise-segment leakage?
6. Normalization strategy nào bảo toàn quan hệ clean/noisy amplitude?

---

## 2. TÓM TẮT IMPLEMENTATION PHASE 1

Các thông số đã được verify từ source code (xem `reports/level2/phase1_audit.md`):

| Thông số Phase 1 | Giá trị | Ghi chú cho Level 2 |
|---|---|---|
| Filter | Butterworth 4th order, [0.5–60 Hz] | Có thể tái sử dụng filter, nhưng review cutoff |
| Window | 260 samples, R-peak centered | KHÔNG phù hợp cho denoising (quá nhỏ, beat-centric) |
| Normalization | Per-beat Z-score | KHÔNG dùng cho denoising |
| Split | De Chazal DS1/DS2 | Tham khảo, tạo split mới cho Level 2 |
| Lead | MLII (channel 0) | Giữ nguyên |
| Sampling rate | 360 Hz | Giữ nguyên |

---

## 3. LITERATURE REVIEW VÀ EVIDENCE TABLE

### 3.1 Tổng quan tài liệu

Nghiên cứu ECG denoising sử dụng deep learning từ 2018–2024 đã hội tụ xung quanh một số quy ước nhất quán:

**Database chuẩn:**
- MITDB + NSTDB là benchmark được dùng phổ biến nhất
- Noise types: BW (baseline wander), MA (muscle artifact), EM (electrode motion)
- SNR grid: -6, 0, 6, 12, 18, 24 dB — matching NSTDB preconstructed records

**Kiến trúc phổ biến:**
- 1D CNN Autoencoder, 1D U-Net, Recurrent CNN, Attention-based models
- Window size: 512–1024 samples (phổ biến nhất)
- Training strategy: noise injection synthetic trên clean ECG

### 3.2 Evidence Table

| Paper title | Authors / Year | DOI / URL | Publication type | Verification |
|---|---|---|---|---|
| MIT-BIH Noise Stress Test Database | Moody et al. (1984/updated) | physionet.org/content/nstdb/1.0.0/ | Official documentation | Full documentation |
| MIT-BIH Arrhythmia Database | Moody & Mark (2001) | physionet.org/content/mitdb/1.0.0/ | Official documentation | Full documentation |
| ECG Denoising Using Deep Learning (review) | Multiple (2020–2023) | Various MDPI, IEEE | Review articles | Abstract + methods |
| Heartbeat classification using deep residual convolutional networks | Multiple papers using MITDB+NSTDB | IEEE TBME | Research articles | Abstract |

*(Note: Full systematic literature search limited by tool access; key database specs confirmed from official PhysioNet documentation and web search results.)*

#### 3.2.1 NSTDB Official Specifications (VERIFIED from header files)

| Parameter | Value | Source |
|---|---|---|
| Noise types | BW (baseline wander), MA (muscle artifact), EM (electrode motion) | bw.hea, ma.hea, em.hea |
| Sampling rate | 360 Hz | Header files |
| Duration per recording | 650,000 samples = ~30.1 minutes | Header files |
| Channels | 2 per recording (noise1, noise2) | Header files |
| Format | 12-bit (format 212) | Header files |
| Preconstructed SNR levels | -6, 0, 6, 12, 18, 24 dB | 118eXX.hea, 119eXX.hea |
| Preconstructed records | 118 + em, 119 + em at 6 SNR levels | Header comments |

**Observation:** NSTDB preconstructed records ONLY use `em` noise (electrode motion). The `bw` and `ma` noise must be applied manually using WFDB nst tool or by direct noise scaling. This is important for our pipeline design.

#### 3.2.2 Key Research Findings

**Về noise types:**
- EM là noise khó remove nhất vì nó có thể mimic ectopic beats
- BW là low-frequency (thường < 1 Hz, peaks around 0.15–0.3 Hz), có thể xử lý bằng high-pass filter
- MA là broadband noise, chủ yếu high-frequency
- Training với nhiều noise types khác nhau → model tổng quát hơn

**Về SNR distribution:**
- SNR grid {-6, 0, 6, 12, 18, 24} dB là standard từ NSTDB preconstructed records
- Random uniform sampling trong range → model học continuous SNR range
- Training với discrete grid, testing trên interpolation points → kiểm tra generalization

**Về window size:**
- 512–1024 samples phổ biến nhất cho denoising
- Lý do: capture đủ context morphological features (P, QRS, T wave, RR interval)
- 260 samples (Phase 1) không đủ để model học context rộng cho denoising

**Về data split:**
- Patient-independent split là yêu cầu tối thiểu
- Nhiều paper không specify noise-segment split → potential limitation

---

## 4. SO SANH NOISE-TYPE STRATEGIES

### 4.1 Các phương án

**Option 1: BW only**
- Ưu: Đơn giản, dễ kiểm soát
- Nhược: Model không học được EM/MA patterns
- Confidence: Thấp — không representative

**Option 2: MA only**
- Ưu: Đơn giản
- Nhược: Tương tự BW only
- Confidence: Thấp

**Option 3: EM only**
- Ưu: Đây là noise khó nhất, testing trên EM có giá trị cao
- Nhược: Không representative cho BW/MA
- Confidence: Trung bình (là phương án của NSTDB preconstructed)

**Option 4: Ba noise types, mỗi sample có 1 loại (one noise per sample)**
- Ưu: Model học cả 3 patterns riêng biệt; dễ phân tích per-noise performance
- Nhược: Không phản ánh real-world noise thường là mixture
- Confidence: Cao — phương án cân bằng nhất cho baseline

**Option 5: Mixture noise trong 1 sample**
- Ưu: Phản ánh real-world tốt hơn
- Nhược: Khó phân tích individual noise performance; harder to train
- Confidence: Tốt cho advanced experiment, không phải baseline

**Option 6: AWGN thêm vào**
- Ưu: Standard theo DSP literature
- Nhược: NSTDB đã có realistic noise, AWGN ít realistic hơn
- Confidence: Thấp — không cần cho baseline

### 4.2 Quyết định

**SELECTED: Option 4 — Ba noise types, mỗi sample có 1 loại (one noise per sample)**

**Lý do:**
- Phù hợp với literature (majority practice)
- Cho phép phân tích per-noise-type performance sau khi train
- Đủ đơn giản cho baseline model
- Không loại bỏ khả năng mixed-noise experiment sau này

**Future experiment:** Option 5 (mixture noise) — để kiểm tra robustness của model đối với real-world combined noise.

---

## 5. SO SANH SNR STRATEGIES

### 5.1 Các phương án SNR

**Option 1: Một SNR cố định (ví dụ: 0 dB)**
- Ưu: Đơn giản
- Nhược: Model sẽ không generalize sang SNR khác
- Confidence: Rất thấp

**Option 2: Discrete SNR grid cố định (ví dụ: -6, 0, 6, 12, 18, 24 dB)**
- Ưu: Matching NSTDB standard; dễ so sánh với literature; rõ ràng
- Nhược: Không cover continuous range
- Confidence: Cao

**Option 3: Random sampling từ discrete grid**
- Ưu: Mỗi epoch model thấy distribution của SNR
- Nhược: Cần on-the-fly generation; validation/test phải deterministic
- Confidence: Cao cho training augmentation

**Option 4: Continuous random SNR sampling (uniform trong range)**
- Ưu: Model học continuous SNR function
- Nhược: Khó replicate; validation trở nên ít standardized
- Confidence: Trung bình

**Option 5: Nonuniform SNR sampling (bias toward lower SNR)**
- Ưu: Tập trung vào hard cases
- Nhược: Phức tạp, cần justify
- Confidence: Thấp cho baseline

**Option 6: Training trên selected SNR, testing trên unseen SNR**
- Ưu: Kiểm tra generalization rõ ràng
- Nhược: Giảm train coverage nếu giữ quá ít SNR cho training
- Confidence: Tốt cho experiment design

### 5.2 NSTDB SNR Grid

NSTDB preconstructed records sử dụng: **{-6, 0, 6, 12, 18, 24} dB**

Đây là grid 6 levels với step 6 dB. KHÔNG phải automatically là best training distribution, nhưng là benchmark standard mà literature thường reference.

### 5.3 Quyết định

**SELECTED: Discrete SNR grid {-6, 0, 6, 12, 18, 24} dB với random sampling per window**

Cụ thể:
- **Validation và test:** Deterministic, mỗi window sử dụng từng SNR level một lần → 6 noisy versions per clean window
- **Training:** Random sample SNR từ grid này cho mỗi window (on-the-fly augmentation)

**Lý do:**
- Grid {-6, 0, 6, 12, 18, 24} dB là chuẩn literature, tương thích với NSTDB
- Deterministic val/test đảm bảo reproducibility và proper metric comparison
- On-the-fly training augmentation giảm storage requirements
- 6 discrete levels bao phủ từ very noisy (-6 dB) đến nearly clean (24 dB)

**Future experiment:** Continuous SNR sampling uniform trong [-6, 24] dB.

---

## 6. DATASET SPLIT DESIGN

### 6.1 Clean ECG Split

**Protocol chính (Level 2, Patient-Grouped):**

MITDB có 48 records từ 47 subjects. Records 201 và 202 từ cùng 1 người. Records 102, 104, 107, 217 bị loại (paced, cho compatibility với Phase 1). Còn lại 44 records từ 46 unique subjects (do 201+202 là 1 subject).

**Nguyên tắc:**
- Records từ cùng subject phải ở cùng split
- 201 và 202 cùng TRAIN
- Test set = phần lớn của DS2, loại 202

**Level 2 Split Assignment:**

| Split | Records | Count |
|---|---|---|
| Train | 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, **202**, 205, 207, 208, 209, 215 | 19 |
| Val | 203, 220, 223, 230 | 4 |
| Test | 100, 103, 105, 111, 113, 117, 121, 123, 200, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234 | 21 |
| **Total** | | **44** |

*Note: Record 202 moved from Phase 1 TEST → Level 2 TRAIN (both 201 and 202 are same patient).*

**Limitation:** Record 118 (Phase 1 train) is also used by NSTDB for preconstructed noisy records. We use the raw clean MITDB record 118 as clean source; the NSTDB preconstructed records (118eXX) are only informational references, not used directly in our pipeline.

### 6.2 Noise Source Split

NSTDB có 3 noise recordings, mỗi recording 650,000 samples = ~30 phút ở 360 Hz.

**Strategy: Fixed-duration non-overlapping blocks**

Mỗi noise recording được chia thành blocks:
- **Block size:** 100,000 samples = ~4.6 phút (lớn hơn nhiều so với window size 1024)
- **Number of blocks per recording:** 650,000 / 100,000 = 6.5 → 6 complete blocks
- **Guard interval:** Bỏ 500 samples ở boundary để tránh filter edge effects
- **Allocation (deterministic, theo thứ tự temporal):**
  - Blocks 1–4 → Train (67%)
  - Block 5 → Val (17%)
  - Block 6 → Test (17%)

**Tổng noise samples per noise type:**

| Split | Blocks | Samples available | Duration |
|---|---|---|---|
| Train | 1–4 | ~400,000 samples | ~18.5 min |
| Val | 5 | ~100,000 samples | ~4.6 min |
| Test | 6 | ~100,000 samples | ~4.6 min |

**Limitation document:** Mặc dù chúng ta split temporal blocks, chúng đến từ cùng một continuous recording. Điều này không tương đương với independent noise devices hoặc independent subjects. Val và Test noise segments có thể có statistical properties tương tự Train. Đây là limitation inherent của NSTDB — nó chỉ có 3 recordings, không thể "hold out" một recording hoàn toàn.

### 6.3 Leakage Rules (Level 2)

1. Một clean record thuộc về đúng 1 split → tất cả windows từ record đó inherit split đó
2. Mỗi noisy version của clean window inherit split của clean window
3. Noise intervals trong train và val/test KHÔNG overlap
4. Windows KHÔNG cross noise-split boundaries
5. Repeating một clean window under different SNR không move copies sang split khác
6. Validation/test generation là deterministic và reproducible
7. Training augmentation (random SNR, random noise interval) không chạm vào val/test sources

---

## 7. PREPROCESSING PIPELINE DESIGN

### 7.1 Signal loading và units

**MITDB:**
- Sampling rate: 360 Hz (confirmed)
- Lead selection: MLII (channel_index=0) — tiêu chuẩn, match Phase 1
- Physical units: mV (p_signal của WFDB)
- Record length: 650,000 samples = ~30 phút

**NSTDB:**
- Sampling rate: 360 Hz (confirmed — same as MITDB)
- Channel selection: channel 0 (noise1)
- Physical units: mV (12-bit ADC, gain 200)
- No resampling needed

**Không cần resampling** vì cả MITDB và NSTDB đều ở 360 Hz.

### 7.2 Clean Reference Construction

**Câu hỏi:** Clean reference là gì?

**Phương án A: Raw physical ECG (không filter)**
- Ưu: Không bias từ filter
- Nhược: Raw ECG chứa baseline wander và powerline noise từ chính recording — không phải "truly clean"
- Không phù hợp

**Phương án B: Bandpass filtered ECG (Phase 1 pipeline: 0.5–60 Hz Butterworth)**
- Ưu: Loại baseline wander (<0.5 Hz) và powerline (>60 Hz); match Phase 1
- Nhược: Lower cutoff 0.5 Hz có thể partially remove low-frequency BW noise mà ta muốn model học cách remove
- **QUAN TRỌNG:** Nếu dùng filter 0.5 Hz cho clean reference, model chỉ cần học remove noise trên signal đã partially cleaned
- Có thể làm BW denoising task artificially easier (BW đã bị loại 1 phần bởi high-pass)

**Phương án C: Bandpass filtered ECG với lowcut thấp hơn (ví dụ: 0.05 Hz)**
- Ưu: Giữ lại nhiều low-frequency content; clean reference gần raw hơn nhưng vẫn loại DC drift
- Nhược: Vẫn là approximation, không truly clean
- Phù hợp hơn cho BW removal task

**SELECTED: Phương án B với lowcut = 0.05 Hz (thay vì 0.5 Hz của Phase 1)**

**Lý do chi tiết:**
- 0.5 Hz cutoff trong Phase 1 được chọn để loại BW cho classification — điều này ổn vì classifier không quan tâm đến slow baseline
- Cho denoising, nếu clean reference dùng 0.5 Hz highpass, model không cần học remove slow BW (đã bị loại). Điều này thay đổi task một cách không minh bạch
- Dùng 0.05 Hz giữ lại baseline trend, cho model học thực sự loại BW
- Highcut 60 Hz vẫn phù hợp (loại powerline 50/60 Hz và high-frequency artifacts)
- Đây là approximation (không truly clean), nhưng là defensible choice

**Filter cho clean reference:** Butterworth order 4, [0.05, 60] Hz, zero-phase (filtfilt)

### 7.3 Order of Preprocessing và Noise Injection

**Pipeline được chọn (Pipeline A: Filter-then-inject):**

```
Raw ECG → Bandpass filter [0.05–60 Hz] → Clean Reference
                                       ↓
                           Scale noise to target SNR
                                       ↓
                  Noisy Input = Clean Reference + scaled_noise_segment
```

**Tại sao Pipeline A?**

- Clean reference được filter trước — consistent definition
- Noisy input = filtered clean + noise → model học reverse noise injection trên filtered signal
- SNR được tính trên filtered clean signal (không phải raw) → SNR calculation là consistent

**Phương án bị loại (Pipeline B: Inject-then-filter both):**
- Nếu filter cả clean và noisy signal sau khi inject noise, 1 phần noise bị loại bởi filter → model task bị altered
- Không minh bạch về target noise condition

### 7.4 SNR Convention

```
SNR_dB = 10 * log10(P_clean / P_noise)

where:
    P_clean = mean(clean_window ** 2)   # mean square power (includes DC)
    P_noise = mean(noise_segment ** 2)  # mean square power
```

**Scaling factor alpha:**
```
alpha = sqrt(P_clean / (P_noise * 10^(SNR_dB/10)))
scaled_noise = alpha * noise_segment
noisy_window = clean_window + scaled_noise
```

**Sau khi inject:** Đo measured SNR thực tế và ghi vào metadata.

**DC component:** Không loại DC riêng — tính power trên full signal. Điều này consistent vì clean reference đã được filter (DC bị loại bởi highpass 0.05 Hz).

**Tolerance:** Measured SNR phải nằm trong ±0.5 dB của target SNR (numerical precision issues only).

### 7.5 Normalization

**Vấn đề:** Nếu normalize clean và noisy windows độc lập, quan hệ biên độ bị phá vỡ → SNR không còn ý nghĩa.

**SELECTED: Record-level amplitude normalization (global per clean record)**

```
norm_factor = max(abs(clean_record_filtered))   # max absolute value of full filtered record
clean_window_norm = clean_window / norm_factor
noisy_window_norm = noisy_window / norm_factor   # SAME factor as clean
```

**Lý do:**
- Cùng scaling factor cho clean và noisy → SNR được bảo toàn
- Giảm inter-patient amplitude variation (different patients, different ECG amplitudes)
- Deterministic (không phụ thuộc vào noise injection)
- Không có data leakage (factor được tính từ clean record, không từ test data)

**Không dùng:** Per-window Z-score (phá vỡ SNR); global dataset normalization (cần fit trên train data).

### 7.6 Windowing

**Window size:** 1024 samples (~2.84 giây ở 360 Hz)

**Lý do chọn 1024:**
- Đủ bao phủ ≥2 RR intervals cho hầu hết heart rates (60–100 bpm → RR ≈ 360–600 samples)
- Phù hợp với literature (512–1024 phổ biến)
- Power of 2 → efficient convolutions
- Không quá lớn cho memory với lightweight 1D CNN
- 260 samples (Phase 1) quá nhỏ — không capture đủ context cho denoising

**Stride:** 512 samples (50% overlap)

**Lý do chọn 512 stride:**
- 50% overlap tăng số windows (data augmentation tự nhiên)
- Boundary artifacts từ filter được amortized qua overlapping windows
- Không phải adjacent windows bị loại hoàn toàn

**Record boundary:** Loại windows mà `start_idx + window_size > len(signal)`. Không dùng zero-padding để tránh artifacts.

**Temporal alignment:** Clean window và noise segment được extract đồng thời từ cùng starting index, đảm bảo temporal alignment hoàn hảo (cả hai ở 360 Hz).

---

## 8. DATASET SCHEMA

### 8.1 Per-sample metadata (CSV manifest)

| Field | Type | Description |
|---|---|---|
| `sample_id` | string | Unique ID: `{record_id}_{start_idx}_{noise_type}_{snr_db}` |
| `record_id` | int | MITDB record number |
| `lead` | string | MLII |
| `start_index` | int | Start sample index in filtered clean record |
| `end_index` | int | End sample index (= start + window_size) |
| `split` | string | train / val / test |
| `noise_type` | string | bw / ma / em |
| `noise_channel` | int | 0 (noise1) |
| `noise_start_index` | int | Start sample in noise recording |
| `noise_end_index` | int | End sample in noise recording |
| `target_snr_db` | float | Target SNR in dB |
| `measured_snr_db` | float | Actual measured SNR after injection |
| `noise_scale_alpha` | float | Scaling factor applied to noise |
| `sampling_rate` | int | 360 |
| `window_size` | int | 1024 |
| `stride` | int | 512 |
| `normalization_method` | string | record_max_abs |
| `norm_factor` | float | Record-level normalization factor |
| `random_seed` | int | Seed used for training augmentation |

### 8.2 Output file structure

```
Processed/denoising_v1/
├── clean_windows/
│   ├── train_clean.npz      (N_train, 1024) float32
│   ├── val_clean.npz        (N_val, 1024) float32
│   └── test_clean.npz       (N_test, 1024) float32
├── manifests/
│   ├── train_manifest.csv   (clean window metadata)
│   ├── val_manifest.csv
│   ├── test_manifest.csv
│   └── noise_blocks.json    (noise block assignments per noise type)
├── noisy_eval/              (deterministic noisy samples for val/test)
│   ├── val_noisy.npz        (N_val * 6 SNR levels, 1024) float32
│   └── test_noisy.npz       (N_test * 6 SNR levels, 1024) float32
├── dataset_info.json        (full config + build metadata)
└── build_log.txt
```

**Storage rationale:**
- Clean windows được lưu một lần (không duplicate)
- Noisy training samples được generate on-the-fly (giảm storage ~6x)
- Val và test noisy samples được pre-generate để reproducibility và offline evaluation

---

## 9. FINAL DECISION TABLE

| Decision | Options considered | Evidence | Selected | Why selected | Why alternatives rejected | Confidence |
|---|---|---|---|---|---|---|
| Noise types | BW only / MA only / EM only / All 3 separate / Mixed | NSTDB docs; literature practice | All 3, one per sample | Model learns all 3 patterns; per-type analysis possible | Single type → not representative; mixture → harder for baseline | HIGH |
| SNR grid | Single SNR / Discrete grid / Continuous random | NSTDB standard; literature | {-6, 0, 6, 12, 18, 24} dB | NSTDB standard; 6 levels cover full range | Single SNR → no generalization; continuous → val not standard | HIGH |
| Training SNR | Fixed per sample / Random per sample | Literature augmentation practice | Random from grid (on-the-fly) | Model sees varied SNR; deterministic val/test | Fixed → model overfit to one SNR | HIGH |
| Window size | 260 / 512 / 1024 / 2048 | Literature 512–1024; RR interval coverage | 1024 samples | Covers ≥2 RR intervals; power of 2; literature consensus | 260 too small (classification only); 2048 too large for lightweight | HIGH |
| Stride | 512 / 256 / 1024 | Overlap helps; storage constraint | 512 (50% overlap) | Standard; data efficiency; boundary artifact amortization | No overlap → misses transitions; 25% overlap → too much data | HIGH |
| Clean reference filter | No filter / 0.5 Hz HP / 0.05 Hz HP | Denoising task definition; Phase 1 analysis | Butterworth [0.05–60 Hz] | Preserves BW content for model to learn; removes DC/powerline | 0.5 Hz removes what model should learn; no filter → noisy reference | MEDIUM-HIGH |
| Normalization | Per-beat Z-score / Per-window Z-score / Record max-abs | Paired signal integrity requirement | Record max-abs | Preserves clean/noisy amplitude relationship; prevents SNR destruction | Z-score → destroys SNR; global → requires fitting on all data | HIGH |
| Record exclusions | All 48 / 44 (no paced) / 44 with paced in train | Downstream compatibility; task requirements | 44 (exclude 102, 104, 107, 217) | Phase 1 compatibility for downstream evaluation | Including paced → inconsistent with classification pipeline | MEDIUM |
| 201/202 split | 201 train + 202 test (Phase 1 compat) / Both in train | Patient independence requirement | Both in TRAIN (Level 2) | True patient-independent test evaluation | Phase 1 compat → scientifically weaker for denoising claim | HIGH |
| Noise block allocation | Random / Temporal order | Reproducibility; temporal stationarity | Temporal blocks (1–4=train, 5=val, 6=test) | Deterministic; avoids seeing future noise in train | Random → more complex; all-noise-in-train → no proper noise eval | MEDIUM |
| Pipeline order | Filter-then-inject / Inject-then-filter | Task definition clarity | Filter clean → inject noise | Clear target definition; consistent SNR measurement | Inject-then-filter → partially removes injected noise (ambiguous task) | HIGH |

---

## 10. REMAINING UNCERTAINTIES AND LIMITATIONS

1. **NSTDB noise stationarity:** 30-minute noise recordings may have different statistical properties in different segments (electrode drift, patient movement patterns). Splitting blocks temporally means val/test noise may differ from train. This is an inherent limitation.

2. **Clean reference as approximation:** Even filtered ECG is not truly noise-free — the original MITDB recordings contain some residual noise. We use it as the best available reference, which is standard practice.

3. **Records 201/202 in same split:** Moving 202 to TRAIN increases training data slightly but reduces test set from 22 to 21 records. This is scientifically justified for strict patient-independence.

4. **Paced records (102/104/107/217) excluded for compatibility:** These records have valid ECG morphology for denoising but are excluded to maintain Phase 1 compatibility. Future work could include them in a denoising-only pipeline.

5. **Single-channel (MLII only):** NSTDB noise has 2 channels; we use channel 0 only. Different channel may have different noise characteristics. Future work could use channel 1 as noise diversity.

6. **Normalization factor:** Using record max-abs normalization means very few high-amplitude beats control the normalization. Alternative: 99th percentile or RMS-based normalization. The max-abs choice is conservative and preserves all amplitude information.

---

## 11. KẾT QUẢ THỰC NGHIỆM VÀ NGHIỆM THU PIPELINE

Pipeline Level 2 đã được thực thi và nghiệm thu thực tế vào ngày 2026-10-11.

### 11.1 Thống kê dataset đã xây dựng
- **Vị trí lưu trữ:** `C:/d2l-en/data/ECG_Project_Data/Processed/denoising_v1`
- **Thời gian chạy:** 114.7 giây
- **Số cửa sổ sạch (Clean Windows - 1024 mẫu, stride 512):**
  - Train: 24,092 windows (19 bản ghi)
  - Val: 5,072 windows (4 bản ghi)
  - Test: 26,628 windows (21 bản ghi)
  - Tổng cộng: **55,792 windows**
- **Số cặp Noisy-Clean tiền tạo lập (Deterministic Evaluation):**
  - Val: 91,296 cặp (5,072 clean × 3 loại nhiễu × 6 mức SNR)
  - Test: 479,304 cặp (26,628 clean × 3 loại nhiễu × 6 mức SNR)
  - Tổng cộng: **570,600 cặp**
- **Độ chính xác SNR đo đạc:**
  - Val: sai số trung bình tuyệt đối $3.69 \times 10^{-16}\text{ dB}$, cực đại $3.55 \times 10^{-15}\text{ dB}$
  - Test: sai số trung bình tuyệt đối $3.07 \times 10^{-16}\text{ dB}$, cực đại $7.11 \times 10^{-15}\text{ dB}$

### 11.2 Kết quả kiểm thử tự động
- **Unit Tests (`tests/test_denoising_data.py`):** **37/37 tests PASS**.
- **Bộ thẩm định toàn diện (`scripts/validate_denoising_dataset.py`):** **57/57 checks PASS** (100%), bao gồm:
  - Khớp hình dạng và dtypes, không NaN/Inf (17 checks)
  - Chống rò rỉ bản ghi và nhóm bệnh nhân 201/202 (4 checks)
  - Chống rò rỉ phân đoạn nhiễu NSTDB giữa các tập (9 checks)
  - Tính toàn vẹn và số lượng mẫu tập noisy (6 checks)
  - Sai số SNR nằm trong dung sai $\pm 0.5\text{ dB}$ (2 checks)
  - Khớp tuyệt đối giữa manifest CSV và NPZ (3 checks)
  - Chuẩn hóa biên độ $\le 1.0$ (4 checks)
  - Xuất đồ thị chẩn đoán trực quan (1 check)

---

*Báo cáo phiên bản 1.1 — Đã cập nhật nghiệm thu thực nghiệm đầy đủ.*
