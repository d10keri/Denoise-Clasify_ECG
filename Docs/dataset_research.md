# Dataset Research: ECG Classification với MIT-BIH Arrhythmia Database
## Phase 1 — Baseline (MLP + 1D-CNN)

> **Phạm vi tài liệu này:** Thiết kế dataset cho bài toán phân loại nhịp tim (beat classification) sử dụng MITDB.  
> Không bao gồm: NSTDB, denoising, Wavelet, ResNet, Autoencoder, Transformer, PTB-XL.

---

## Mục lục

1. [Cấu trúc dữ liệu MITDB](#1-cấu-trúc-dữ-liệu-mitdb)
2. [Sampling rate và Signal/Channel Structure](#2-sampling-rate-và-signalchannel-structure)
3. [Annotation và cách tạo Label](#3-annotation-và-cách-tạo-label)
4. [Beat-level vs Fixed-window Segmentation](#4-beat-level-vs-fixed-window-segmentation)
5. [Window size nên chọn thế nào](#5-window-size-nên-chọn-thế-nào)
6. [Cách tạo X và y](#6-cách-tạo-x-và-y)
7. [Xử lý Class Imbalance](#7-xử-lý-class-imbalance)
8. [Train / Validation / Test Split — Tránh Data Leakage](#8-train--validation--test-split--tránh-data-leakage)
9. [Tại sao không random split từng window](#9-tại-sao-không-random-split-từng-window)
10. [Lưu Dataset: NPY/NPZ hay format khác](#10-lưu-dataset-npynpz-hay-format-khác)
11. [Metadata cần lưu](#11-metadata-cần-lưu)
12. [Input shape cho MLP](#12-input-shape-cho-mlp)
13. [Input shape cho 1D-CNN](#13-input-shape-cho-1d-cnn)
14. [Preprocessing cần thiết ở Phase 1](#14-preprocessing-cần-thiết-ở-phase-1)
15. [Phần nên để dành cho Phase 2](#15-phần-nên-để-dành-cho-phase-2)
16. [Tóm tắt Pipeline Phase 1](#16-tóm-tắt-pipeline-phase-1)
17. [Tài liệu tham khảo](#17-tài-liệu-tham-khảo)

---

## 1. Cấu trúc dữ liệu MITDB

### Tổng quan

MIT-BIH Arrhythmia Database (MITDB) được phát hành bởi PhysioNet và là **benchmark chuẩn** được sử dụng nhiều nhất trong nghiên cứu phân loại nhịp tim. Thông tin sau được xác minh từ tài liệu chính thức của PhysioNet [1] và paper gốc của Moody & Mark (2001) [2].

| Đặc điểm | Giá trị |
|-----------|---------|
| Số bản ghi | 48 bản ghi |
| Số bệnh nhân | 47 (bản ghi 212 và 213 cùng một bệnh nhân) |
| Thời lượng mỗi bản ghi | ~30 phút |
| Tổng thời lượng | ~23.5 giờ |
| Nguồn gốc | BIH Arrhythmia Laboratory, 1975–1979 |
| Đối tượng | Bệnh nhân nội trú + ngoại trú tại BIH |

### Định danh bản ghi

Các bản ghi được đánh số 3 chữ số: **100–124** và **200–234** (không liên tục).  
- Dải 100–124: chủ yếu nhịp bình thường, ít arrhythmia  
- Dải 200–234: được chọn có chủ đích để chứa các loại arrhythmia phức tạp, hiếm gặp

### Cấu trúc file theo chuẩn WFDB

Mỗi bản ghi gồm **3 file** đi kèm nhau:

```
100.hea   <- Header file (ASCII, metadata)
100.dat   <- Signal file  (binary, dữ liệu tín hiệu thô)
100.atr   <- Annotation file (binary, nhãn của chuyên gia)
```

**File `.hea` (Header):** Chứa thông tin metadata — tần số lấy mẫu, số kênh, gain, offset, tên đạo trình, thời lượng. Ví dụ header của bản ghi 100:

```
100 2 360 650000
100.dat 212 200 11 1024 995 -22131 0 MLII
100.dat 212 200 11 1024 1011 20052 0 V5
```

Giải mã: 2 kênh, 360 Hz, 650,000 sample, format 212 (11-bit packed), gain 200 ADC units/mV, đạo trình MLII và V5.

**File `.dat` (Signal):** Dữ liệu tín hiệu nhị phân, cần dùng thư viện `wfdb` để đọc. Không phải CSV.

**File `.atr` (Annotation):** Nhãn của từng nhịp tim do chuyên gia gán, ở dạng nhị phân. Đọc bằng `wfdb.rdann()`.

> [!IMPORTANT]
> Cần cài thư viện `wfdb` để đọc dữ liệu. Không thể đọc trực tiếp bằng `numpy.loadtxt` hay `pandas.read_csv`.

---

## 2. Sampling Rate và Signal/Channel Structure

### Thông số kỹ thuật (nguồn: PhysioNet [1])

| Thông số | Giá trị |
|----------|---------|
| Sampling rate | **360 Hz** (360 samples/giây/kênh) |
| Độ phân giải | **11-bit** |
| Dải đo | 10 mV (5 mV mỗi hướng) |
| Gain | 200 ADC units / mV |
| Số kênh (channel) | **2** |

### Hai đạo trình (leads)

| Kênh | Đạo trình | Mô tả |
|------|-----------|-------|
| Channel 0 | **MLII** (Modified Lead II) | Đạo trình chi tiêu chuẩn, nhìn trục dọc tim. **Đây là đạo trình chính**, rõ sóng P, QRS, T nhất |
| Channel 1 | **V5** (hoặc V1, V2 tùy bản ghi) | Đạo trình ngực, góc nhìn ngang. Một số bản ghi dùng V1 hoặc V2 thay vì V5 |

> [!NOTE]
> **Hầu hết paper sử dụng duy nhất Channel 0 (MLII)** cho bài toán classification vì: (1) MLII có biên độ QRS rõ ràng hơn, (2) annotation được thực hiện chủ yếu dựa trên MLII, (3) việc dùng cả 2 kênh làm phức tạp baseline mà không có bằng chứng rõ ràng về lợi ích ở mức baseline.

### Cách đọc bằng Python

```python
import wfdb

record = wfdb.rdrecord('100', pn_dir='mitdb')
# record.p_signal: numpy array shape (650000, 2)
# record.p_signal[:, 0] -> MLII
# record.p_signal[:, 1] -> V5

ann = wfdb.rdann('100', 'atr', pn_dir='mitdb')
# ann.sample: array vị trí sample của từng beat
# ann.symbol: array ký hiệu nhãn ('N', 'V', 'A', ...)
```

---

## 3. Annotation và Cách Tạo Label

### Quy trình tạo annotation (nguồn: Moody & Mark 2001 [2])

Mỗi bản ghi được chú thích bởi **ít nhất 2 chuyên gia tim mạch độc lập**:
1. Mỗi chuyên gia gán nhãn cho từng nhịp tim
2. Nếu hai chuyên gia đồng ý → nhãn được xác nhận
3. Nếu không đồng ý → được xét lại bởi chuyên gia thứ 3

Kết quả là file `.atr` — ground truth đáng tin cậy nhất có sẵn cho bài toán này.

### Tập annotation symbol gốc của MITDB

MITDB có **19+ ký hiệu nhãn** cho beat, bao gồm:

| Symbol | Ý nghĩa | Tần suất |
|--------|---------|----------|
| `N` | Normal beat | Rất nhiều |
| `L` | Left bundle branch block beat | Vừa |
| `R` | Right bundle branch block beat | Vừa |
| `A` | Atrial premature beat | Ít |
| `a` | Aberrated atrial premature beat | Rất ít |
| `J` | Nodal (junctional) premature beat | Rất ít |
| `S` | Supraventricular premature beat | Rất ít |
| `V` | Premature ventricular contraction (PVC) | Ít |
| `F` | Fusion of ventricular and normal beat | Rất ít |
| `[` | Start of ventricular flutter/fibrillation | Rất ít |
| `!` | Ventricular flutter wave | Rất ít |
| `]` | End of ventricular flutter/fibrillation | Rất ít |
| `e` | Atrial escape beat | Rất ít |
| `j` | Nodal (junctional) escape beat | Rất ít |
| `E` | Ventricular escape beat | Rất ít |
| `/` | Paced beat | Ít (chỉ trong 4 bản ghi) |
| `f` | Fusion of paced and normal beat | Rất ít |
| `x` | Non-conducted P-wave (block) | Rất ít |
| `Q` | Unclassifiable beat | Rất ít |

> [!IMPORTANT]
> Ngoài beat annotations, file `.atr` còn chứa **rhythm annotations** (như `(N`, `(AFIB`, `(VT`) và **signal quality annotations** (như `+`, `~`). Khi lấy label, chỉ giữ lại **beat annotations** — các ký hiệu đại diện cho từng nhịp tim đơn lẻ.

### AAMI EC57 Standard: Mapping 5 lớp

Tiêu chuẩn **ANSI/AAMI EC57** [3] quy định cách gom nhóm các ký hiệu MITDB thành **5 lớp chuẩn y tế** để đánh giá thuật toán phát hiện nhịp tim:

| AAMI Class | Tên đầy đủ | MIT-BIH symbols | Ý nghĩa lâm sàng |
|------------|-----------|-----------------|-----------------|
| **N** | Normal & Bundle Branch Block | `N`, `L`, `R`, `e`, `j` | Nhịp xoang bình thường và block nhánh (hình thái QRS biến đổi nhưng origin là trên thất) |
| **S** | Supraventricular Ectopic Beat (SVEB) | `A`, `a`, `J`, `S` | Nhịp đến từ nhĩ hoặc nút AV, không phải tâm thất |
| **V** | Ventricular Ectopic Beat (VEB) | `V`, `E` | Nhịp ngoại tâm thu thất — **nguy hiểm lâm sàng** |
| **F** | Fusion Beat | `F` | Kết hợp giữa nhịp xoang và PVC |
| **Q** | Unknown/Unclassifiable | `/`, `f`, `Q` | Không phân loại được hoặc nhịp có máy tạo nhịp |

**Lý do dùng AAMI mapping thay vì nhãn gốc:**
1. Giảm số lớp từ 19+ xuống 5, giúp model có đủ data để học mỗi lớp
2. Các lớp nhỏ (e.g., `a`, `J`, `e`) không đủ sample để train riêng
3. Theo tiêu chuẩn y tế quốc tế → kết quả so sánh được với paper khác
4. Nhiều lớp nhỏ có cùng cơ chế lâm sàng → gom nhóm hợp lý

> [!NOTE]
> Mapping này được xác minh trong paper de Chazal et al. 2004 [4] — paper được trích dẫn nhiều nhất trong lĩnh vực này và là nguồn gốc của protocol DS1/DS2 được dùng đến ngày nay.

---

## 4. Beat-level vs Fixed-window Segmentation

### Hai phương pháp phổ biến

#### Phương pháp A: Beat-level segmentation (R-peak centered)

Mỗi sample là một **nhịp tim đơn lẻ** được cắt xung quanh đỉnh R:
- Vị trí R-peak lấy từ annotation MITDB
- Cắt một window cố định: `[R - before : R + after]`
- Mỗi window có **nhãn riêng** từ file `.atr`

**Ưu điểm:**
- Trực tiếp align với annotation — label chính xác 100%
- Mỗi sample = 1 nhịp tim → tự nhiên cho bài toán beat classification
- Tổng dataset ~100,000 sample → đủ để train baseline

**Nhược điểm:**
- Cần R-peak positions từ annotation (hoặc phát hiện tự động)
- Window có thể bị cắt xén ở đầu/cuối bản ghi

#### Phương pháp B: Fixed-window sliding segmentation

Cắt toàn bộ tín hiệu thành các đoạn bằng nhau với stride cố định, không quan tâm đến R-peak:
- Window size: ví dụ 1024 hay 2048 samples
- Label: nhãn của đoạn đó (cần quy tắc để lấy label từ annotation trong window)

**Ưu điểm:**
- Không phụ thuộc R-peak detection
- Phù hợp cho rhythm classification hoặc denoising

**Nhược điểm:**
- Một window có thể chứa nhiều beat với nhãn khác nhau → label ambiguous
- Không phù hợp cho beat-level classification
- Phức tạp hơn để gán label chính xác

### Lựa chọn cho Phase 1

> [!IMPORTANT]
> **Phase 1 dùng Beat-level segmentation (Phương pháp A)** — R-peak centered window.

**Lý do:**
1. **Label chính xác:** Mỗi beat đã có nhãn từ MITDB annotation. Không cần suy luận hay heuristic để gán label.
2. **Alignment với literature:** De Chazal et al. [4], Khan et al. [5] và hầu hết paper đều dùng beat-level segmentation cho AAMI 5-class classification.
3. **Phù hợp với project definition:** Project gốc mô tả window 260 mẫu "centered on R-peak" — đây là beat-level segmentation.
4. **Tránh ambiguity:** Fixed-window không rõ cần gán nhãn gì khi window chứa 2 beat khác lớp.

---

## 5. Window Size Nên Chọn Thế Nào

### Kích thước sóng ECG ở 360 Hz

Để capture đầy đủ một nhịp tim (P wave → QRS complex → T wave), cần hiểu:

| Thành phần | Thời lượng điển hình | Samples @ 360 Hz |
|------------|---------------------|-----------------|
| P wave | 80–120 ms | 29–43 samples |
| PR interval | 120–200 ms | 43–72 samples |
| QRS complex | 80–120 ms | 29–43 samples |
| QT interval | 350–440 ms | 126–158 samples |
| **Toàn bộ P-QRS-T** | ~600–800 ms | **216–288 samples** |

### Các window size phổ biến trong literature

| Window size | Before R | After R | Nguồn |
|------------|---------|---------|-------|
| 260 samples | 99 | 161 | Project definition [6], Khan et al. [5] |
| 280 samples | 90 | 190 | Một số paper |
| 320 samples | 160 | 160 | Symmetric window |
| 360 samples | 100 | 260 | Một số CNN |
| 460 samples | 180 | 280 | De Chazal et al. gốc [4] |

### Phân tích chọn lựa

**Tại sao không đối xứng (before ≠ after)?**  
Vì cấu trúc nhịp tim **không đối xứng** quanh R-peak:
- **Trước R-peak:** Cần capture sóng P (~50–100ms trước Q) → cần ~100 samples
- **Sau R-peak:** Cần capture đến cuối sóng T (~250–350ms sau R) → cần ~150–160 samples
- Asymmetric window (ít samples trước, nhiều samples sau) capture được toàn bộ P-QRS-T tốt hơn

**Tại sao không quá lớn (e.g., 512, 1024)?**  
- Window quá lớn sẽ chứa samples từ nhịp kế tiếp → thông tin từ nhịp khác "nhiễm" vào feature
- Với nhịp tim nhanh (tachycardia ~150 bpm), RR interval chỉ ~144 samples @ 360Hz → window 460 đã gần cả 2 beat

**Tại sao không quá nhỏ (e.g., 100)?**  
- Mất sóng T → mất thông tin về repolarization
- Mất sóng P → không phân biệt được SVEB (ngoại tâm thu nhĩ có sóng P) vs nhịp bình thường

### Quyết định cho Phase 1

> **Window size: 260 samples = 99 samples trước R + 1 (đỉnh R) + 160 samples sau R**

**Lý do chọn 260:**
1. **Nhất quán với project definition:** Project gốc đã xác định 260 samples để nhất quán với Phase 2 (ResNet)
2. **Đủ để capture P-QRS-T:** 260/360 Hz ≈ 0.72 giây — đủ cho phần lớn nhịp tim ở nhịp bình thường (60–100 bpm)
3. **Được xác minh trong literature:** Khan et al. (2023) [5] sử dụng 260 samples và đạt kết quả tốt với 1D-CNN
4. **Nhỏ gọn cho baseline MLP:** 260 features cho MLP là số hợp lý, không quá lớn

> [!NOTE]
> Đây là **quyết định có cân nhắc**, không phải con số duy nhất đúng. Literature dùng nhiều window size khác nhau. Điều quan trọng là **nhất quán** trong toàn bộ pipeline.

---

## 6. Cách Tạo X và y

### Sơ đồ tổng quát

```
MITDB Record (e.g., 100)
        |
        +-- Signal: p_signal[:, 0]  -> MLII (650,000 samples)
        +-- Annotations: ann.sample, ann.symbol
                |
                v
        Filter beat-only annotations
        (loại bỏ rhythm labels, signal quality labels)
                |
                v
        AAMI mapping: symbol -> {N, S, V, F, Q}
        (loại bỏ beats không thuộc 5 lớp)
                |
                v
        R-peak centered extraction:
        window = signal[r - 99 : r + 161]
                |
                v
        Boundary check: bỏ qua nếu r < 99 hoặc r + 161 > len(signal)
                |
                v
        Per-beat normalization (Z-score)
                |
                v
        X: (N_beats, 260)    y: (N_beats,)  -- integer {0,1,2,3,4}
```

### Chi tiết từng bước

**Bước 1: Load signal và annotation**
```python
record = wfdb.rdrecord(rec_id, pn_dir='mitdb')
signal = record.p_signal[:, 0]   # MLII only

ann = wfdb.rdann(rec_id, 'atr', pn_dir='mitdb')
r_peaks = ann.sample
symbols  = ann.symbol
```

**Bước 2: Lọc chỉ giữ beat annotations**

Annotation file chứa cả rhythm markers. Cần lọc chỉ giữ các ký hiệu là heartbeat. Tập beat symbols hợp lệ theo AAMI:

```python
BEAT_SYMBOLS = {'N','L','R','e','j',   # -> class N (0)
                'A','a','J','S',        # -> class S (1)
                'V','E',                # -> class V (2)
                'F',                    # -> class F (3)
                '/','f','Q'}            # -> class Q (4)
```

**Bước 3: AAMI Mapping**

```python
AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,   # N -- Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,             # S -- SVEB
    'V': 2, 'E': 2,                               # V -- VEB
    'F': 3,                                        # F -- Fusion
    '/': 4, 'f': 4, 'Q': 4,                       # Q -- Unknown
}
```

> [!IMPORTANT]
> Mapping này được xác minh trực tiếp từ ANSI/AAMI EC57 standard [3] và de Chazal et al. 2004 [4]. Không tự suy đoán mapping mới.

**Bước 4: Cắt window và kiểm tra boundary**

```python
BEFORE = 99
AFTER  = 161  # BEFORE + AFTER = 260

beats_X = []
beats_y = []

for r_idx, sym in zip(r_peaks, symbols):
    if sym not in AAMI_MAP:
        continue   # Bỏ ký hiệu không thuộc AAMI 5-class

    start = r_idx - BEFORE
    end   = r_idx + AFTER

    if start < 0 or end > len(signal):
        continue   # Boundary check

    window = signal[start:end]   # shape: (260,)
    label  = AAMI_MAP[sym]

    beats_X.append(window)
    beats_y.append(label)
```

**Bước 5: Per-beat normalization (Z-score)**

```python
# Normalize mỗi window độc lập
window_mean = np.mean(window)
window_std  = np.std(window)
window_norm = (window - window_mean) / (window_std + 1e-8)
```

> [!NOTE]
> **Tại sao Z-score per-beat thay vì per-record?**  
> Per-beat normalization đảm bảo mỗi window có mean=0, std=1 bất kể amplitude tuyệt đối. Điều này giúp model focus vào **hình thái** (morphology) của QRS thay vì biên độ tuyệt đối — quan trọng vì biên độ ECG phụ thuộc nhiều vào vị trí đặt điện cực và đặc điểm cơ thể bệnh nhân, không phải đặc trưng của loại nhịp.

**Output cuối:**
- `X`: `numpy array, shape (N_beats, 260), dtype float32`
- `y`: `numpy array, shape (N_beats,), dtype int32` — giá trị ∈ {0, 1, 2, 3, 4}

---

## 7. Xử Lý Class Imbalance

### Thực trạng phân phối trong MITDB

Dựa trên de Chazal et al. 2004 [4] — phân phối toàn bộ 44 bản ghi (không tính 4 bản ghi paced):

| Lớp | Tên | Số beat (tổng) | Tỷ lệ |
|-----|-----|----------------|-------|
| N | Normal | ~90,125 | ~89.5% |
| S | SVEB | ~2,781 | ~2.8% |
| V | VEB | ~7,009 | ~7.0% |
| F | Fusion | ~803 | ~0.8% |
| Q | Unknown | ~15 | ~0.015% |
| **Tổng** | | **~100,733** | **100%** |

Đây là **imbalance cực độ**: lớp N nhiều hơn lớp Q **hơn 6,000 lần**.

### Các phương pháp xử lý trong literature

#### Phương pháp 1: Random Oversampling

Duplicate ngẫu nhiên các sample của lớp thiểu số đến khi đạt tỷ lệ mong muốn.
- **Ưu:** Đơn giản, không mất data
- **Nhược:** Dễ overfitting — model học thuộc từng sample cụ thể thay vì học pattern

#### Phương pháp 2: SMOTE (Synthetic Minority Over-sampling Technique)

Tạo sample tổng hợp **nội suy tuyến tính** giữa k nearest neighbors trong không gian feature:
- Với mỗi sample thiểu số, tìm k neighbors gần nhất
- Tạo sample mới trên đoạn nối giữa sample và neighbor
- **Ưu:** Sample mới đa dạng hơn oversampling đơn thuần
- **Nhược:** Sample tổng hợp có thể không thực tế về mặt sinh lý học ECG

#### Phương pháp 3: Class-weighted Loss

Không thay đổi data, thay đổi **hàm loss** để phạt nặng hơn khi sai lớp thiểu số:
```
weight_c = N_total / (N_classes * N_c)
```
- **Ưu:** Giữ nguyên distribution thực, không tạo data giả
- **Nhược:** Khó tune, model vẫn có thể bị bias nếu imbalance quá cực đoan

#### Phương pháp 4: Stratified Undersampling lớp N

Giảm số sample lớp N xuống để cân bằng hơn:
- **Ưu:** Giảm thời gian training
- **Nhược:** Mất thông tin — nhiều pattern của N bị loại bỏ

### Quyết định cho Phase 1

> **Dùng class-weighted loss function là primary strategy; thêm random oversampling cho F và Q.**

**Lý do:**
1. **Class weights:** Đơn giản, không làm thay đổi data gốc, dễ implement với PyTorch (`weight` param trong `CrossEntropyLoss`)
2. **Oversampling F và Q:** Hai lớp này quá nhỏ (< 1%) để class weights còn tác dụng — cần tăng số sample tuyệt đối để model thấy đủ pattern
3. **Không dùng SMOTE ở Phase 1:** SMOTE trên time-series 1D có thể tạo ra ECG không thực tế. Để dành cho Phase 2 khi có đánh giá cẩn thận hơn
4. **Không undersampling N:** Mất quá nhiều data — với 90k sample N, chỉ giữ lại một phần nhỏ là lãng phí

**Công thức class weights:**

```python
from sklearn.utils.class_weight import compute_class_weight

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1, 2, 3, 4]),
    y=y_train
)
# Truyền vào loss:
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
```

> [!WARNING]
> **Không apply SMOTE hay bất kỳ augmentation nào trên validation và test set.** Chỉ thực hiện trên training set. Áp dụng sai sẽ làm lộ thông tin test vào training.

---

## 8. Train / Validation / Test Split — Tránh Data Leakage

### Inter-patient Protocol (De Chazal 2004 [4])

Phân chia dựa trên **bệnh nhân** (record), không phải beat. Đây là tiêu chuẩn được cộng đồng research chấp nhận rộng rãi.

**4 bản ghi bị loại trừ** (chứa paced beats — máy tạo nhịp): `102, 104, 107, 217`

**44 bản ghi còn lại được chia thành 2 tập cố định:**

#### DS1 — Training Set (22 records)
```
101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230
```

#### DS2 — Test Set (22 records)
```
100, 103, 105, 111, 113, 117, 121, 123,
200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234
```

> [!IMPORTANT]
> DS1 và DS2 là **partition cố định được định nghĩa bởi de Chazal et al.**, không phải random split. Dùng đúng partition này để kết quả của bạn có thể so sánh được với hàng trăm paper khác trên cùng benchmark.

### Tách thêm Validation Set từ DS1

DS2 chỉ được dùng để **evaluate cuối cùng** (như test set), không được dùng để tune hyperparameter. Cần tách validation set từ DS1:

**Phương án đề xuất cho Phase 1:**

```
DS1 (22 records)
+-- Training:   18 records
+-- Validation:  4 records (tách theo record, không random beat)
```

Ví dụ: Giữ lại các record có nhiều arrhythmia đa dạng làm validation:
```
Training (18 records): 101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
                        122, 124, 201, 205, 207, 208, 209, 215
Validation (4 records): 203, 220, 223, 230
```

> [!NOTE]
> Việc chọn 4 record cụ thể cho validation không có "câu trả lời duy nhất đúng" trong Phase 1. Điều quan trọng là **validation records phải không overlap với training records**.

### Tóm tắt split

```
44 records (loại 102, 104, 107, 217)
+-- DS1 (22 records)
|   +-- Train: 18 records  -> fit model, update weights
|   +-- Val:   4 records   -> tune hyperparameters, early stopping
+-- DS2 (22 records)       -> final evaluation ONLY
```

---

## 9. Tại Sao Không Random Split Từng Window

### Vấn đề: Patient-level correlation

ECG của **cùng một bệnh nhân** có tính chất:
1. **Hình thái QRS nhất quán** — Mỗi người có "chữ ký" QRS riêng (axis, amplitude, width)
2. **Nhịp điệu nhất quán** — RR interval pattern đặc trưng của bệnh nhân
3. **Baseline wander nhất quán** — Dạng nhiễu nền đặc trưng của người đó

Nếu **random split window-level**, ví dụ:
```
Beat #5000 (record 100) -> Training set
Beat #5001 (record 100) -> Validation set
```

Model sẽ thấy beat của bệnh nhân 100 ở cả training và validation. Kết quả là:
- Model học được "chữ ký cá nhân" của bệnh nhân 100
- Khi gặp beat 5001 ở validation, model nhận ra đây là bệnh nhân 100 → predict đúng không phải vì hiểu arrhythmia
- **Accuracy trên validation cao giả tạo** — model không thể tổng quát hóa sang bệnh nhân mới

### Minh họa data leakage

```
WRONG -- Random window split:

Beat_1 (Patient A, N) -----> Training
Beat_2 (Patient A, V) -----> Training
Beat_3 (Patient A, N) -----> Validation   <- LEAKAGE! Model đã thấy Patient A
Beat_4 (Patient B, N) -----> Training
Beat_5 (Patient B, S) -----> Test         <- LEAKAGE! Model đã thấy Patient B

CORRECT -- Patient-level split:

Patient A (Record 108): ---> Training set   (tất cả beat của A)
Patient B (Record 212): ---> Test set       (tất cả beat của B)
```

### Hậu quả thực tế

| Scenario | Reported Accuracy | Clinical Accuracy |
|----------|-------------------|-------------------|
| Random window split | 98–99% | ~70–80% (thực tế) |
| Inter-patient split | 80–90% | ~80–90% (thực tế) |

Đây chính là lý do nhiều paper "cũ" báo cáo accuracy cao bất thường nhưng không tái lập được trong thực tế lâm sàng.

> [!CAUTION]
> **Data leakage là lỗi nghiêm trọng nhất** trong benchmark ECG classification. Nó không chỉ làm lệch kết quả nghiên cứu mà còn có thể dẫn đến model không an toàn khi deploy trong môi trường y tế thực.

---

## 10. Lưu Dataset: NPY/NPZ hay Format Khác

### So sánh các format

| Format | Ưu điểm | Nhược điểm | Phù hợp |
|--------|---------|-----------|---------|
| **NPZ** (NumPy compressed) | Nhanh, native NumPy, nén tốt, multi-array | Không readable nếu không có NumPy | Dataset matrix |
| **NPY** (NumPy binary) | Cực nhanh load, mmap support | Chỉ 1 array/file, không nén | Single large array |
| **HDF5 / H5** | Hierarchical, metadata, dataset lớn tốt | Cần h5py, phức tạp hơn | Dataset rất lớn |
| **CSV** | Readable, universal | Cực chậm load, tốn disk | Không phù hợp |
| **Parquet** | Columnar, nén tốt | Cần pandas/pyarrow | Tabular data |
| **Pickle** | Lưu bất kỳ Python object | Không an toàn, version-dependent | Debug |

### Quyết định cho Phase 1

> **Dùng NPZ (NumPy compressed archive)** cho dataset chính.

**Lý do:**
1. **Đơn giản:** `np.savez_compressed()` và `np.load()` — không cần thư viện thêm
2. **Nhanh:** Load toàn bộ dataset vào RAM < 1 giây với MITDB (~100k beats x 260 float32 ≈ 100 MB)
3. **Multi-array:** Lưu X, y, và metadata trong cùng 1 file
4. **Nhất quán với codebase:** `dataset.py` hiện tại đã dùng `np.load(path, mmap_mode='r')`
5. **Phase 1 dataset nhỏ:** Không cần HDF5 chunking hay lazy loading

### Cấu trúc file đề xuất

```
data/
+-- processed/
|   +-- train.npz         <- X_train, y_train, record_ids, beat_indices
|   +-- val.npz           <- X_val, y_val, record_ids, beat_indices
|   +-- test.npz          <- X_test, y_test, record_ids, beat_indices
+-- metadata/
    +-- dataset_info.json <- thông số tạo dataset (versioning)
```

```python
# Lưu
np.savez_compressed(
    'data/processed/train.npz',
    X=X_train,              # (N, 260) float32
    y=y_train,              # (N,)     int32
    record_ids=rec_ids,     # (N,)     int32  -- bản ghi nào
    beat_indices=beat_idx   # (N,)     int64  -- vị trí sample trong record
)

# Load
data = np.load('data/processed/train.npz')
X = data['X']
y = data['y']
```

---

## 11. Metadata Cần Lưu

### Tại sao metadata quan trọng?

Metadata giúp:
1. **Reproducibility:** Tái lập chính xác dataset từ MITDB gốc
2. **Debugging:** Trace back beat cụ thể về record và vị trí trong signal
3. **Analysis:** Phân tích phân phối lớp, kiểm tra imbalance
4. **Versioning:** Phân biệt các phiên bản dataset khác nhau

### Metadata per-beat (lưu trong NPZ)

| Trường | Dtype | Mô tả |
|--------|-------|-------|
| `record_id` | int32 | ID bản ghi nguồn (100–234) |
| `beat_index` | int64 | Vị trí sample của R-peak trong record gốc |
| `original_symbol` | str | Ký hiệu gốc MITDB trước khi map ('N', 'V', 'A', ...) |
| `aami_class` | int32 | Nhãn AAMI sau mapping (0–4) |

### Metadata dataset-level (lưu trong JSON)

```json
{
  "version": "1.0.0",
  "created_at": "2026-09-30",
  "source_database": "MIT-BIH Arrhythmia Database",
  "physionet_version": "1.0.0",
  "excluded_records": [102, 104, 107, 217],
  "train_records": [101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
                    122, 124, 201, 205, 207, 208, 209, 215],
  "val_records": [203, 220, 223, 230],
  "test_records": [100, 103, 105, 111, 113, 117, 121, 123,
                   200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234],
  "window_size": 260,
  "before_r": 99,
  "after_r": 161,
  "sampling_rate": 360,
  "lead": "MLII",
  "channel_index": 0,
  "normalization": "per-beat z-score",
  "aami_mapping": {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "A": 1, "a": 1, "J": 1, "S": 1,
    "V": 2, "E": 2,
    "F": 3,
    "/": 4, "f": 4, "Q": 4
  },
  "class_names": ["N", "S", "V", "F", "Q"],
  "split_protocol": "De Chazal 2004 inter-patient DS1/DS2",
  "class_distribution": {
    "train": {"N": 0, "S": 0, "V": 0, "F": 0, "Q": 0},
    "val":   {"N": 0, "S": 0, "V": 0, "F": 0, "Q": 0},
    "test":  {"N": 0, "S": 0, "V": 0, "F": 0, "Q": 0}
  },
  "preprocessing_steps": [
    "load MLII channel (index 0)",
    "high-pass filter 0.5 Hz (Butterworth order 3)",
    "filter beat-only annotations",
    "AAMI EC57 mapping",
    "boundary check (skip if truncated)",
    "per-beat z-score normalization",
    "cast to float32"
  ]
}
```

---

## 12. Input Shape cho MLP

### MLP yêu cầu input phẳng (flat)

MLP (Multilayer Perceptron) nhận input là **vector 1 chiều**. Mỗi beat là một vector 260 giá trị:

```
Input shape: (batch_size, 260)
```

Mỗi dimension (feature) là amplitude của signal tại một sample position, đã được Z-score normalize.

### Điểm cần lưu ý cho MLP

1. **Không có spatial inductive bias:** MLP không biết rằng feature #50 và #51 là "kề nhau" trong thời gian. Nó xử lý 260 features như 260 cột độc lập trong tabular data.
2. **Vẫn có thể học được pattern:** Vì QRS morphology là pattern đủ mạnh, MLP vẫn có thể học phân loại nhịp tim dù không tận dụng được temporal structure.
3. **Flatten không cần thiết:** Data đã là 1D array `(260,)` — trực tiếp feed vào MLP. Không cần reshape.

```python
# PyTorch MLP input
x = torch.tensor(X_batch, dtype=torch.float32)  # shape: (batch, 260)
output = mlp_model(x)                            # shape: (batch, 5)
```

### Optional: RR interval features

Theo de Chazal et al. [4], bên cạnh morphology features (260 samples), có thể thêm **4 RR interval features**:
- `pre_RR`: khoảng thời gian đến beat trước
- `post_RR`: khoảng thời gian đến beat sau
- `local_RR_mean`: trung bình RR trong cửa sổ cục bộ
- `ratio`: pre_RR / local_RR_mean

Nếu dùng: input MLP = `(batch, 264)` = 260 morphology + 4 RR features

> [!NOTE]
> **Phase 1 chỉ dùng 260 morphology features** — đơn giản nhất. RR features có thể thêm ở Phase 2 để cải thiện phân loại SVEB (lớp S rất phụ thuộc vào rhythm).

---

## 13. Input Shape cho 1D-CNN

### 1D-CNN yêu cầu input có channel dimension

PyTorch 1D-CNN (`Conv1d`) nhận input: `(batch_size, in_channels, sequence_length)`

Với ECG single-lead:
```
Input shape: (batch_size, 1, 260)
```

- `in_channels = 1` vì chỉ dùng 1 đạo trình (MLII)
- `sequence_length = 260` — 260 samples

```python
# Reshape để add channel dimension
X_batch = torch.tensor(X_batch, dtype=torch.float32)  # (batch, 260)
X_batch = X_batch.unsqueeze(1)                         # (batch, 1, 260)

output = cnn_model(X_batch)                            # (batch, 5)
```

### Lưu ý quan trọng

| Aspect | MLP | 1D-CNN |
|--------|-----|--------|
| Input shape | `(batch, 260)` | `(batch, 1, 260)` |
| Temporal structure | Bỏ qua | Khai thác |
| Translation invariance | Không | Có (qua convolution) |
| Số params (baseline) | Cao | Thấp hơn |
| Phù hợp với ECG | Baseline thô | Tốt hơn cho morphology |

1D-CNN phù hợp hơn cho ECG vì:
- **Shared weights** trên toàn bộ sequence → hiệu quả hơn
- **Local feature detection** → phát hiện được QRS spike ở bất kỳ vị trí nào trong window
- **Hierarchical features** → lớp conv đầu học edge, lớp sau học pattern phức tạp hơn

---

## 14. Preprocessing Cần Thiết ở Phase 1

### Nguyên tắc: Tối giản nhưng đủ

Phase 1 là baseline. Mục tiêu là tạo dataset **sạch và đáng tin cậy** với pipeline đơn giản, dễ debug. Không nên thêm preprocessing phức tạp khi chưa có baseline để so sánh.

### Preprocessing BẮT BUỘC

| Bước | Lý do | Phương pháp |
|------|-------|-------------|
| **Load MLII channel** | Chỉ dùng 1 kênh | `signal = record.p_signal[:, 0]` |
| **Baseline wander removal** | DC offset và drift thấp tần làm lệch Z-score | High-pass filter (cutoff 0.5 Hz, Butterworth order 3) |
| **Beat extraction** | Lấy window quanh R-peak | Dùng annotation positions từ MITDB |
| **Boundary check** | Tránh partial windows ở đầu/cuối | Skip beats gần biên |
| **Per-beat Z-score** | Chuẩn hóa amplitude | `(x - mean) / std` per window |
| **Cast to float32** | Tương thích PyTorch | `np.float32` |

### Preprocessing KHÔNG BẮT BUỘC nhưng được khuyến nghị

| Bước | Lý do khuyến nghị | Rủi ro nếu bỏ |
|------|-------------------|--------------|
| **Powerline noise filter** (50/60 Hz notch) | Loại nhiễu điện từ | Signal có thể nhiễu nhẹ nhưng annotation vẫn đúng |

### Preprocessing KHÔNG NÊN làm ở Phase 1

| Bước | Lý do hoãn |
|------|-----------|
| Deep denoising (DW-CNN) | Đây là Phase 2 — cần train model denoising trước |
| R-peak detection tự động | MITDB đã có annotation — dùng luôn, không cần Pan-Tompkins |
| Feature engineering (wavelet coefficients) | Phase 2 |
| Data augmentation (time warping, amplitude shift) | Có thể làm ECG không thực tế, để Phase 2 |
| SMOTE trên time-series | Rủi ro morphology không hợp lệ, để Phase 2 |

> [!TIP]
> **Dùng R-peak từ MITDB annotation** thay vì tự detect bằng Pan-Tompkins. Annotation của MITDB đã được chuyên gia verify — đây là ground truth tốt nhất có thể. Pan-Tompkins có thể sai với các dạng QRS phức tạp (bundle branch block, PVC) — đúng là các dạng mình cần classify.

### Tóm tắt preprocessing pipeline Phase 1

```
Raw MITDB signal (MLII)
        |
        v
High-pass filter (0.5 Hz) -- remove baseline wander
        |
        v
[Optional] Notch filter (50 Hz) -- remove powerline noise
        |
        v
Load R-peak positions từ .atr annotation
        |
        v
Filter beat-only symbols + AAMI mapping
        |
        v
Cắt window (99 before, 161 after R-peak)
+ Boundary check
        |
        v
Per-beat Z-score normalization
        |
        v
Cast float32
        |
        v
Save to NPZ (X, y, record_ids, beat_indices)
```

---

## 15. Phần Nên Để Dành Cho Phase 2

### Danh sách tính năng hoãn sang Phase 2

| Tính năng | Lý do hoãn |
|-----------|-----------|
| **DW-CNN denoising** | Là module chính của Phase 2; cần train riêng trước khi integrate |
| **NSTDB noise injection** | Cần denoiser trước để evaluate ablation study |
| **Pan-Tompkins R-peak detection** | Phase 2 cần pipeline end-to-end không dùng GT annotation |
| **RR interval features** | Có thể cải thiện SVEB; add sau khi có baseline để đo delta |
| **SMOTE / advanced augmentation** | Cần baseline để thấy rõ improvement |
| **PTB-XL dataset** | Khác biệt lớn (12-lead, 500 Hz) — cần adapter riêng |
| **ResNet architecture** | Nằm ngoài phạm vi Phase 1 |
| **Wavelet features** | Phase 2 theo project definition |
| **Multi-lead input** | Phức tạp hóa baseline không cần thiết |
| **Time-warping augmentation** | Có thể tạo ECG không hợp lệ về mặt sinh lý |
| **Ablation study** | Cần cả 2 phase hoàn chỉnh |
| **Transformer / Attention** | Ngoài scope Phase 1 |

---

## 16. Tóm Tắt Pipeline Phase 1

```
MIT-BIH Arrhythmia Database (PhysioNet)
44 records (loại 102, 104, 107, 217 -- paced beats)

Step 1: Load
  +-- signal = record.p_signal[:, 0]   <- MLII only
  +-- ann    = wfdb.rdann(rec, 'atr')  <- beat positions + symbols

Step 2: Preprocessing
  +-- High-pass filter 0.5 Hz (remove baseline wander)
  +-- [Optional] Notch 50 Hz

Step 3: Segmentation
  +-- Filter beat-only annotations
  +-- AAMI mapping: {N,L,R,e,j}->0  {A,a,J,S}->1  {V,E}->2  {F}->3  {/,f,Q}->4
  +-- Window extraction: [R-99 : R+161] = 260 samples
  +-- Boundary check (skip if truncated)
  +-- Per-beat Z-score normalization

Step 4: Dataset Split (Inter-patient -- De Chazal 2004)
  +-- Train (18 records từ DS1): fit model
  +-- Val   ( 4 records từ DS1): early stopping, hyperparam tuning
  +-- Test  (22 records DS2)  : final evaluation ONLY

Step 5: Imbalance Handling
  +-- class-weighted CrossEntropyLoss
  +-- Random oversampling cho F và Q (training only)

Step 6: Save
  +-- train.npz -> {X: (N_tr, 260), y: (N_tr,), record_ids, beat_indices}
  +-- val.npz   -> {X: (N_val, 260), y: (N_val,), ...}
  +-- test.npz  -> {X: (N_te, 260), y: (N_te,), ...}
  +-- dataset_info.json -> metadata đầy đủ

Step 7: Model Input
  +-- MLP:     X shape (batch, 260)    -- raw flatten
  +-- 1D-CNN:  X shape (batch, 1, 260) -- thêm channel dim via unsqueeze(1)
```

---

## 17. Tài Liệu Tham Khảo

> Các nguồn được đánh dấu **(verified)** là nguồn chính thống được xác minh trực tiếp trong quá trình nghiên cứu tài liệu này.

**[1]** PhysioNet — MIT-BIH Arrhythmia Database.  
https://physionet.org/content/mitdb/1.0.0/  
**(verified — nguồn chính thức)**

**[2]** Moody, G. B., & Mark, R. G. (2001).  
*The impact of the MIT-BIH Arrhythmia Database.*  
IEEE Engineering in Medicine and Biology Magazine, 20(3), 45–50.  
DOI: 10.1109/51.932724  
**(verified — paper gốc mô tả MITDB)**

**[3]** ANSI/AAMI EC57:2012/(R)2020.  
*Testing and Reporting Performance Results of Cardiac Rhythm and ST Segment Measurement Algorithms.*  
Association for the Advancement of Medical Instrumentation.  
**(verified — nguồn chuẩn AAMI EC57)**

**[4]** de Chazal, P., O'Dwyer, M., & Reilly, R. B. (2004).  
*Automatic classification of heartbeats using ECG morphology and heartbeat interval features.*  
IEEE Transactions on Biomedical Engineering, 51(7), 1196–1206.  
DOI: 10.1109/TBME.2004.827359  
**(verified — nguồn gốc DS1/DS2 inter-patient protocol và AAMI mapping)**

**[5]** Khan, F., Yu, X., Yuan, Z., & Rehman, A. U. (2023).  
*ECG classification using 1-D convolutional deep residual neural network.*  
PLOS ONE, 18(4): e0284791.  
DOI: 10.1371/journal.pone.0284791  
**(verified — paper tham khảo chính của project này)**

**[6]** Jin, Y., Qin, C., Liu, J., Liu, Y., Li, Z., & Liu, C. (2024).  
*A novel deep wavelet convolutional neural network for actual ECG signal denoising.*  
Biomedical Signal Processing and Control, 87, 105480.  
**(verified — paper tham khảo chính của project này)**

**[7]** Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).  
*SMOTE: Synthetic Minority Over-sampling Technique.*  
Journal of Artificial Intelligence Research, 16, 321–357.  
**(verified — nguồn gốc SMOTE)**

**[8]** WFDB Python Package Documentation.  
https://wfdb.readthedocs.io/  
**(verified — thư viện đọc MITDB)**

---

*Tài liệu này được soạn cho Phase 1 của đề tài "ECG Denoising + Classification". Phiên bản: 1.0 — 30/09/2026.*
