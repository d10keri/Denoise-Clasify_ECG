# BÁO CÁO XÂY DỰNG VÀ NGHIỆM THU DATASET LEVEL 2 (DATASET BUILD REPORT)
## Supervised ECG Denoising Dataset (MITDB + NSTDB)

**Phiên bản:** 1.0.0  
**Ngày thực hiện:** 2026-10-11  
**Tác giả:** Antigravity AI Engine  
**Môi trường thực thi:** Conda environment `ecg` (Python 3.11.16, PyTorch 2.12.1+cu130, WFDB 4.3.1, NumPy 2.4.6)  
**Hệ điều hành:** Windows 10/11 x64  

---

## 1. TỔNG QUAN XÂY DỰNG DỮ LIỆU

Dataset Level 2 được thiết kế và sinh tự động cho bài toán Supervised ECG Denoising (khử nhiễu tín hiệu điện tâm đồ có giám sát):
- **Tín hiệu sạch tham chiếu ($X_{\text{clean}}$):** MIT-BIH Arrhythmia Database (MITDB), chuyển đạo MLII, tần số lấy mẫu 360 Hz.
- **Nguồn nhiễu sinh lý học ($N$):** MIT-BIH Noise Stress Test Database (NSTDB) với 3 loại nhiễu đặc trưng:
  - `bw`: Baseline Wander (nhiễu trôi đường đẳng điện, tần số thấp do hô hấp).
  - `ma`: Muscle Artifact (nhiễu cơ, băng rộng, tần số cao do vận động).
  - `em`: Electrode Motion (nhiễu dịch chuyển điện cực, khó khử nhất, dễ nhầm với ngoại tâm thu).
- **Tín hiệu đầu vào mô hình ($X_{\text{noisy}}$):** Được tổng hợp theo kiểm soát SNR chính xác:
  $$X_{\text{noisy}} = X_{\text{clean}} + \alpha \cdot N$$
  với $\alpha = \sqrt{\frac{P_{\text{clean}}}{P_{\text{noise}} \cdot 10^{\text{SNR}_{\text{target}} / 10}}}$.

---

## 2. CÁC LỆNH TÁI LẬP (REPRODUCIBILITY COMMANDS)

Để tái lập toàn bộ quy trình từ đầu hoặc thẩm định lại tính toàn vẹn của dataset, thực thi các lệnh sau tại thư mục gốc repository (`c:\d2l-en\Denoise_Classify ECG\Denoise-Clasify_ECG`):

### 2.1 Kiểm tra đơn vị pipeline (Unit Tests)
```powershell
conda run -n ecg python tests/test_denoising_data.py
```
*Thời gian chạy: ~3 giây. Kết quả kỳ vọng: 37/37 tests PASS.*

### 2.2 Xây dựng toàn bộ dataset (Dataset Build)
```powershell
conda run -n ecg python scripts/build_denoising_dataset.py --config configs/denoising_dataset.yaml
```
*Thời gian chạy: 114.7 giây. Dữ liệu được xuất tại `C:/d2l-en/data/ECG_Project_Data/Processed/denoising_v1`.*

### 2.3 Thẩm định tính toàn vẹn và chống rò rỉ (Validation & Leakage Suite)
```powershell
conda run -n ecg python scripts/validate_denoising_dataset.py --config configs/denoising_dataset.yaml
```
*Thời gian chạy: 45.0 giây. Kết quả kỳ vọng: 57/57 checks PASS, lưu đồ thị tại `plots/sample_clean_noisy_pairs.png`.*

---

## 3. CẤU TRÚC VÀ DUNG LƯỢNG THƯ MỤC ARTIFACTS ĐÃ TẠO

Toàn bộ dữ liệu Level 2 được lưu trữ tách biệt hoàn toàn khỏi dữ liệu Phase 1, đặt tại:  
`C:\d2l-en\data\ECG_Project_Data\Processed\denoising_v1`

```
Processed/denoising_v1/
├── clean_windows/
│   ├── train_clean.npz            46,223,671 bytes  (44.1 MB)  [24,092 windows × 1024]
│   ├── val_clean.npz               9,757,603 bytes   (9.3 MB)  [ 5,072 windows × 1024]
│   └── test_clean.npz             51,218,017 bytes  (48.8 MB)  [26,628 windows × 1024]
├── manifests/
│   ├── train_manifest.csv          1,833,038 bytes   (1.7 MB)  [24,092 rows metadata]
│   ├── val_manifest.csv              377,514 bytes   (0.4 MB)  [ 5,072 rows metadata]
│   ├── test_manifest.csv           2,003,354 bytes   (1.9 MB)  [26,628 rows metadata]
│   ├── val_noisy_manifest.csv     14,621,980 bytes  (13.9 MB)  [91,296 rows metadata]
│   ├── test_noisy_manifest.csv    76,801,606 bytes  (73.2 MB)  [479,304 rows metadata]
│   └── noise_blocks.json               1,173 bytes             [Phân chia block nhiễu]
├── noisy_eval/
│   ├── val_noisy.npz             359,909,838 bytes (343.2 MB)  [91,296 noisy/clean pairs]
│   └── test_noisy.npz          1,874,912,560 bytes   (1.75 GB) [479,304 noisy/clean pairs]
├── plots/
│   └── sample_clean_noisy_pairs.png  271,621 bytes             [Đồ thị chẩn đoán trực quan]
├── dataset_info.json                   2,712 bytes             [Metadata tổng hợp cấu hình & kết quả]
├── build_log.txt                         707 bytes             [Log thời gian build]
└── validation_report.txt               3,892 bytes             [Log 57 checks PASS]
```

**Tổng dung lượng lưu trữ:** ~2.3 GB (chủ yếu là tập test evaluation tiền tạo lập để đảm bảo benchmark bất biến).

---

## 4. CHI TIẾT PHÂN PHỐI DỮ LIỆU (DATA STATISTICS)

### 4.1 Thống kê cửa sổ sạch (Clean Windows)
- Kích thước cửa sổ: **1024 mẫu** (~2.84 giây ở 360 Hz).
- Bước trượt (Stride): **512 mẫu** (50% chồng lấn).
- Chuẩn hóa: `record_max_abs` (tối đa biên độ tuyệt đối trong từng bản ghi $\le 1.0$).
- Kiểu dữ liệu: `float32`.

| Tập dữ liệu | Số bản ghi MITDB | Danh sách bản ghi | Số cửa sổ sạch (Clean Windows) | Tỉ lệ (%) |
|---|---|---|---|---|
| **Train** | 19 | 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 202, 205, 207, 208, 209, 215 | 24,092 | 43.18% |
| **Validation** | 4 | 203, 220, 223, 230 | 5,072 | 9.09% |
| **Test** | 21 | 100, 103, 105, 111, 113, 117, 121, 123, 200, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234 | 26,628 | 47.73% |
| **Tổng cộng** | **44** | *(Loại 4 bản ghi paced: 102, 104, 107, 217)* | **55,792** | **100%** |

*Ghi chú quan trọng:* Bản ghi 201 và 202 xuất phát từ cùng một bệnh nhân. Ở Phase 1 (De Chazal 2004), 201 thuộc DS1 và 202 thuộc DS2. Ở Level 2, cả 201 và 202 được gom trọn vẹn vào tập **Train** nhằm triệt tiêu rò rỉ danh tính bệnh nhân sang tập Test.

### 4.2 Thống kê phân chia khối nhiễu NSTDB (Noise Disjoint Blocks)
Mỗi bản ghi nhiễu dài 650.000 mẫu (~1805.6 giây = 30.1 phút). Được chia thành 6 khối cố định (100.000 mẫu/khối, chừa 500 mẫu guard band tại biên):
- **Khối 0, 1, 2, 3 (samples 0 - 400.000):** Dành riêng cho tập **Train** (66.7% dữ liệu nhiễu). Dùng để tiêm nhiễu ngẫu nhiên on-the-fly khi huấn luyện mô hình.
- **Khối 4 (samples 400.000 - 500.000):** Dành riêng cho tập **Validation** (16.7% dữ liệu nhiễu).
- **Khối 5 (samples 500.000 - 600.000):** Dành riêng cho tập **Test** (16.7% dữ liệu nhiễu).

*Bảo đảm toán học:* Tập giao $\text{Blocks}(\text{Train}) \cap \text{Blocks}(\text{Val}) \cap \text{Blocks}(\text{Test}) = \emptyset$. Không có bất kỳ đoạn nhiễu nào bị tái sử dụng giữa các tập.

### 4.3 Thống kê tập Noisy Evaluation (Val & Test)
Các tập Val Noisy và Test Noisy được sinh trước cố định (deterministic, seed=42) trên toàn bộ tổ hợp lưới SNR chuẩn NSTDB `[-6, 0, 6, 12, 18, 24] dB` và 3 loại nhiễu (`bw`, `ma`, `em`):

| Tập đánh giá | Số Clean Windows | Số loại nhiễu | Số mức SNR | Tổng số cặp Noisy-Clean | Dung lượng NPZ |
|---|---|---|---|---|---|
| **Val Noisy** | 5,072 | 3 | 6 | **91,296** | 359.9 MB |
| **Test Noisy** | 26,628 | 3 | 6 | **479,304** | 1.87 GB |
| **Tổng cộng** | — | — | — | **570,600** | **~2.23 GB** |

---

## 5. ĐỘ CHÍNH XÁC SNR THỰC NGHIỆM

Độ chính xác của việc tiêm nhiễu được kiểm định trên toàn bộ 570.600 mẫu:
- **Tập Validation (91.296 mẫu):**
  - Sai số tuyệt đối trung bình: $\mu_{|\text{error}|} = 3.692 \times 10^{-16}\text{ dB}$
  - Sai số tuyệt đối cực đại: $\max_{|\text{error}|} = 3.553 \times 10^{-15}\text{ dB}$
- **Tập Test (479.304 mẫu):**
  - Sai số tuyệt đối trung bình: $\mu_{|\text{error}|} = 3.071 \times 10^{-16}\text{ dB}$
  - Sai số tuyệt đối cực đại: $\max_{|\text{error}|} = 7.105 \times 10^{-15}\text{ dB}$

*Kết luận:* Mức độ sai lệch giữa SNR mong muốn và SNR thực tế sau khi tổng hợp tín hiệu đạt xấp xỉ mức sai số số học dấu phẩy động 64-bit của máy tính ($10^{-15}$ đến $10^{-16}\text{ dB}$), bỏ xa dung sai cho phép ($\le 0.5\text{ dB}$).

---

## 6. KẾT QUẢ THẨM ĐỊNH TOÀN DIỆN (VALIDATION SUITE AUDIT)

Bộ kiểm thử `validate_denoising_dataset.py` đã thực hiện 57 bài test nghiêm ngặt với kết quả **57/57 PASS (100%)**:

1. **Kiểm tra sự tồn tại của tệp (10/10 PASS):**
   - Đầy đủ 3 file `.npz` clean, 3 file `.csv` clean manifest, file `noise_blocks.json`, 2 file `.npz` noisy eval, 1 file `dataset_info.json`.
2. **Kích thước hình dạng và Kiểu dữ liệu (17/17 PASS):**
   - Train shape: `(24092, 1024)`, float32, no NaN/Inf.
   - Val shape: `(5072, 1024)`, float32, no NaN/Inf.
   - Test shape: `(26628, 1024)`, float32, no NaN/Inf.
   - Không chứa bất kỳ bản ghi nào trong 4 bản ghi loại trừ (102, 104, 107, 217).
3. **Chống rò rỉ dữ liệu bản ghi (4/4 PASS):**
   - Giao giữa Train-Val: $\emptyset$.
   - Giao giữa Train-Test: $\emptyset$.
   - Giao giữa Val-Test: $\emptyset$.
   - Bản ghi 201 và 202 nằm trọn vẹn trong tập Train.
4. **Chống rò rỉ phân đoạn nhiễu (9/9 PASS):**
   - Không có khoảng lấy mẫu nhiễu nào trùng lặp giữa Train-Val, Train-Test, Val-Test cho cả 3 loại nhiễu `bw`, `ma`, `em`.
5. **Tính toàn vẹn của tập Noisy (6/6 PASS):**
   - Cặp `noisy` và `clean` khớp kích thước chính xác `(91296, 1024)` cho Val và `(479304, 1024)` cho Test.
   - Tuyệt đối không chứa NaN hoặc Inf.
6. **Độ chính xác SNR (2/2 PASS):**
   - 100% mẫu trong tập Val và Test đều có sai số SNR $\le 0.5\text{ dB}$.
7. **Tính nhất quán giữa Manifest và NPZ (3/3 PASS):**
   - Train: 24.092 dòng manifest = 24.092 arrays.
   - Val: 5.072 dòng manifest = 5.072 arrays.
   - Test: 26.628 dòng manifest = 26.628 arrays.
8. **Kiểm tra chuẩn hóa (4/4 PASS):**
   - Cực đại biên độ tuyệt đối trên các cửa sổ sạch $\le 1.000000$.
   - Không có cửa sổ nào toàn số 0.
9. **Chẩn đoán hình ảnh trực quan (1/1 PASS):**
   - Đã sinh và kiểm tra đồ thị đối chiếu giữa Clean Reference và 6 biến thể Noisy tương ứng tại các mức SNR khác nhau.

---

## 7. ĐỒ THỊ CHẨN ĐOÁN TRỰC QUAN

Đồ thị đã được lưu tại:  
`C:\d2l-en\data\ECG_Project_Data\Processed\denoising_v1\plots\sample_clean_noisy_pairs.png` (271.6 KB).

Đồ thị hiển thị:
- **Subplot 1:** Tín hiệu Clean reference từ bản ghi 203 (thuộc tập Val), đã qua lọc Butterworth [0.05 - 60 Hz], bảo toàn trọn vẹn hình thái phức bộ QRS, sóng P và sóng T.
- **Subplots 2 - 7:** Các biến thể noisy đầu vào tương ứng với các tổ hợp nhiễu và SNR khác nhau (`bw -6dB`, `bw 0dB`, `ma 6dB`, `em 12dB`, v.v.), vẽ đè cùng tín hiệu gốc trên cùng trục thời gian để minh họa mức độ biến dạng sóng.

---

## 8. HẠN CHẾ VÀ ĐỀ XUẤT CHO BƯỚC HUẤN LUYỆN TIẾP THEO

### 8.1 Các hạn chế cố hữu
1. **Dung lượng tập Test Noisy:** Tập `test_noisy.npz` chiếm 1.87 GB trên đĩa cứng do chứa đầy đủ 479.304 cặp tín hiệu được sinh sẵn phục vụ đánh giá ngoại tuyến toàn diện. Khi nạp vào RAM máy tính để tính toán, khuyến nghị sử dụng `mmap_mode='r'` hoặc nạp theo từng batch/loại nhiễu để tránh tràn bộ nhớ.
2. **Tính chất trôi của nhiễu NSTDB:** Do chia khối theo thời gian (temporal blocks), đặc tính thống kê của các khối nhiễu sau (ở tập Val và Test) có thể có chút biến động so với khối đầu (ở tập Train). Đây là kịch bản thử thách thực tế phản ánh chính xác khả năng tổng quát hóa của mô hình học sâu.

### 8.2 Đề xuất kỹ thuật cho bước tiếp theo (Training Step)
1. **Lớp dữ liệu PyTorch DataLoader:** Đã sẵn sàng lớp `DenoisingDataset` trong `src/data/denoising_data.py`. Lớp này nạp mảng `clean_windows` vào bộ nhớ và thực hiện tiêm nhiễu ngẫu nhiên on-the-fly trong quá trình huấn luyện, giúp tiết kiệm bộ nhớ và đóng vai trò như một cơ chế Data Augmentation tự nhiên.
2. **Kiến trúc mô hình mục tiêu:** Xây dựng một mạng 1D Fully-Convolutional Denoising Autoencoder hoặc 1D U-Net nhỏ gọn, nhận đầu vào `(Batch, 1, 1024)` và dự đoán đầu ra `(Batch, 1, 1024)` với hàm mất mát MSE hoặc kết hợp MSE + SNR loss.
3. **Chỉ số đánh giá (Evaluation Metrics):**
   - $\text{SNR}_{\text{improvement}} = \text{SNR}_{\text{out}} - \text{SNR}_{\text{in}}$ (dB)
   - Root Mean Square Error (RMSE)
   - Percent Root Mean Square Difference (PRD).
