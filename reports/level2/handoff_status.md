# BÁO CÁO BÀN GIAO TIẾN ĐỘ LEVEL 2 (HANDOFF STATUS) — ECG DENOISING DATASET

**Ngày cập nhật:** 2026-10-11  
**Mục tiêu:** Tổng hợp tình trạng thực tế của dự án sau khi tiếp nhận từ Agent trước, phân định rõ ràng giữa sự thật đã kiểm chứng bằng thực nghiệm và các giả định/kế hoạch tiếp theo.

---

## 1. TỔNG QUAN TÌNH TRẠNG HIỆN TẠI

| Hạng mục | Tình trạng | Minh chứng thực tế (Evidence) |
|---|---|---|
| **Môi trường Python (Conda `ecg`)** | **ĐÃ XÁC NHẬN (VERIFIED)** | Python 3.11.16, PyTorch 2.12.1+cu130, WFDB 4.3.1, NumPy 2.4.6 tại `C:\Users\vietc\.conda\envs\ecg\python.exe` |
| **Phase 1 Codebase & Data** | **NGUYÊN VẸN (INTACT)** | Không có file Phase 1 nào bị thay đổi hay ghi đè (`git status` xác nhận sạch) |
| **Báo cáo nghiên cứu & thiết kế** | **HOÀN THÀNH (COMPLETED)** | `reports/level2/phase1_audit.md` (11.8 KB), `reports/level2/level2_dataset_research_report.md` (25.1 KB), `plan/level2_dataset_plan.md` (7.2 KB) |
| **Mã nguồn Dataset & Pipeline** | **HOÀN THÀNH (COMPLETED)** | `configs/denoising_dataset.yaml`, `src/data/nstdb_loader.py`, `src/data/denoising_data.py`, `scripts/build_denoising_dataset.py`, `scripts/validate_denoising_dataset.py`, `tests/test_denoising_data.py` |
| **Unit Tests Pipeline** | **ĐÃ CHẠY & PASS 100%** | `tests/test_denoising_data.py`: **37/37 tests PASS** |
| **Full Dataset Build** | **ĐÃ CHẠY HOÀN TẤT** | Sinh thành công tại `C:\d2l-en\data\ECG_Project_Data\Processed\denoising_v1` trong 114.7s |
| **Validation & Leakage Suite** | **ĐÃ CHẠY & PASS 100%** | `scripts/validate_denoising_dataset.py`: **57/57 checks PASS** trong 45.0s |
| **Diagnostic Plots** | **ĐÃ XUẤT THÀNH CÔNG** | `Processed/denoising_v1/plots/sample_clean_noisy_pairs.png` (271 KB) |
| **Huấn luyện mô hình Denoising** | **CHƯA THỰC HIỆN (Theo chỉ thị)** | Yêu cầu dừng lại sau khi dataset & validation hoàn tất; chờ người dùng chỉ thị bước tiếp theo |

---

## 2. CHI TIẾT CÁC HẠNG MỤC ĐÃ HOÀN THÀNH VÀ KIỂM CHỨNG

### 2.1 Môi trường thực thi
- **Hệ điều hành:** Windows 10/11 (`win32`).
- **Thư mục làm việc (CWD):** `C:\d2l-en\Denoise_Classify ECG\Denoise-Clasify_ECG`.
- **Conda Environment:** `ecg`.
- **Thư viện chính:**
  - `python`: 3.11.16
  - `torch`: 2.12.1+cu130 (hỗ trợ CUDA)
  - `wfdb`: 4.3.1
  - `numpy`: 2.4.6

### 2.2 Các quyết định thiết kế khoa học đã thực hiện
1. **Clean Reference Preprocessing:**
   - Sử dụng bộ lọc Butterworth IIR bậc 4 bandpass hai chiều (`filtfilt`, zero-phase).
   - Dải tần: **[0.05 Hz, 60.0 Hz]**. Cutoff thấp 0.05 Hz thay vì 0.5 Hz (như Phase 1) để giữ lại thành phần Baseline Wander (BW) cho mô hình denoising học cách khử nhiễu thực tế.
2. **Kích thước Window:**
   - 1024 samples (~2.84 giây ở 360 Hz, bao quát tối thiểu 2 chu kỳ RR).
   - Stride 512 samples (50% overlap).
3. **Chuẩn hóa (Normalization):**
   - Áp dụng `record_max_abs` trên toàn bản ghi ECG trước khi chia window.
   - Tuyệt đối **không** dùng per-beat Z-score (vì Z-score sẽ phá hủy tỉ lệ biên độ và công suất giữa tín hiệu sạch và tín hiệu nhiễu).
4. **Chia tập dữ liệu (Split Protocol):**
   - Loại bỏ 4 bản ghi nhịp tạo nhịp (paced): 102, 104, 107, 217 (đảm bảo đồng bộ với Phase 1).
   - Bản ghi 201 và 202 (cùng một bệnh nhân) đều được đưa vào tập **Train** (19 bản ghi), loại bỏ hoàn toàn rò rỉ bệnh nhân giữa Train và Test.
   - Tập Val: 4 bản ghi (203, 220, 223, 230).
   - Tập Test: 21 bản ghi.
5. **Phân chia khối nhiễu NSTDB (Noise Block Allocation):**
   - 3 bản ghi nhiễu NSTDB: `bw`, `ma`, `em` (mỗi bản ghi 650.000 samples, 2 channels).
   - Chia mỗi bản ghi thành 6 khối rời rạc không gối nhau (100.000 samples/khối):
     - Blocks 0, 1, 2, 3: dành cho Train (~67%).
     - Block 4: dành cho Val (~17%).
     - Block 5: dành cho Test (~17%).
   - Hoàn toàn độc lập về mặt thời gian giữa các tập, chống data leakage về nhiễu.
6. **Mức SNR kiểm định:**
   - Lưới 6 mức chuẩn NSTDB: `[-6, 0, 6, 12, 18, 24] dB`.
   - Công thức tính tỉ lệ nhiễu: $\alpha = \sqrt{\frac{P_{\text{clean}}}{P_{\text{noise}} \cdot 10^{\text{SNR}_{\text{target}} / 10}}}$.
   - Tín hiệu nhiễu: $x_{\text{noisy}} = x_{\text{clean}} + \alpha \cdot \text{noise}$.

---

## 3. KẾT QUẢ XÂY DỰNG DATASET (FULL DATASET BUILD)

Lệnh đã thực thi:
```bash
conda run -n ecg python scripts/build_denoising_dataset.py --config configs/denoising_dataset.yaml
```

**Kết quả xác nhận:**
- Thời gian thực thi: **114.7 giây**.
- **Cửa sổ sạch (Clean Windows):**
  - Train: 24.092 windows (từ 19 bản ghi). File: `clean_windows/train_clean.npz` (46.2 MB).
  - Val: 5.072 windows (từ 4 bản ghi). File: `clean_windows/val_clean.npz` (9.8 MB).
  - Test: 26.628 windows (từ 21 bản ghi). File: `clean_windows/test_clean.npz` (51.2 MB).
  - **Tổng cộng: 55.792 clean windows**.
- **Tập Noisy Đánh giá (Deterministic Evaluation):**
  - Val Noisy: 5.072 clean × 3 loại nhiễu × 6 mức SNR = **91.296 samples**. File: `noisy_eval/val_noisy.npz` (359.9 MB).
  - Test Noisy: 26.628 clean × 3 loại nhiễu × 6 mức SNR = **479.304 samples**. File: `noisy_eval/test_noisy.npz` (1.87 GB).
  - **Tổng cộng: 570.600 paired noisy-clean evaluation samples**.
- **Sai số SNR đo đạc thực tế so với mục tiêu:**
  - Val: sai số trung bình tuyệt đối = $3.69 \times 10^{-16}\text{ dB}$, sai số lớn nhất = $3.55 \times 10^{-15}\text{ dB}$.
  - Test: sai số trung bình tuyệt đối = $3.07 \times 10^{-16}\text{ dB}$, sai số lớn nhất = $7.11 \times 10^{-15}\text{ dB}$.
  - Đạt độ chính xác xấp xỉ mức độ chính xác số học của máy tính (machine epsilon).

---

## 4. KẾT QUẢ KIỂM THỬ VÀ VALIDATION TỰ ĐỘNG

### 4.1 Unit Tests (`tests/test_denoising_data.py`)
- Lệnh: `conda run -n ecg python tests/test_denoising_data.py`
- Kết quả: **37/37 tests PASS**.
- Nội dung kiểm tra: Độ chính xác công thức SNR, bảo toàn pha và biên độ của bộ lọc, trích xuất window tại biên tín hiệu, tính bất biến tỉ lệ biên độ sau chuẩn hóa, tính không chồng lấn của các khối nhiễu NSTDB.

### 4.2 Bộ kiểm định toàn diện (`scripts/validate_denoising_dataset.py`)
- Lệnh: `conda run -n ecg python scripts/validate_denoising_dataset.py --config configs/denoising_dataset.yaml`
- Kết quả: **57/57 checks PASS**, 0 thất bại. Thời gian kiểm định: **45.0 giây**.
- Tóm tắt các nhóm kiểm định:
  1. **File Existence (10/10 PASS):** Toàn bộ file npz, csv, json, log tồn tại đầy đủ.
  2. **Clean Window Shape & Dtype (17/17 PASS):** Đúng chiều `(N, 1024)`, kiểu dữ liệu `float32`, 0 giá trị NaN/Inf, đúng danh sách bản ghi, không chứa 4 bản ghi paced.
  3. **Record Leakage (4/4 PASS):** Không có bản ghi nào trùng lặp giữa Train-Val, Train-Test, Val-Test. Bản ghi 201 và 202 cùng nằm trọn vẹn trong tập Train.
  4. **Noise Block Leakage (9/9 PASS):** Không có khoảng thời gian nhiễu nào trùng lặp giữa Train-Val, Train-Test, Val-Test cho cả 3 loại nhiễu (`bw`, `ma`, `em`).
  5. **Val/Test Noisy Sample Integrity (6/6 PASS):** Cặp tín hiệu khớp kích thước `(N, 1024)`, không có NaN/Inf, số lượng mẫu đúng tuyệt đối 91.296 và 479.304.
  6. **SNR Accuracy (2/2 PASS):** Tất cả các mẫu đều đạt sai số SNR $\le 0.5\text{ dB}$ (sai số thực tế $\approx 0.0000\text{ dB}$).
  7. **Manifest Consistency (3/3 PASS):** Số dòng trong CSV manifest khớp 100% với số mảng trong NPZ.
  8. **Normalization Check (4/4 PASS):** Giá trị cực đại biên độ $\le 1.0$, không có window nào toàn 0.
  9. **Visual Diagnostics (1/1 PASS):** Tạo và lưu đồ thị chẩn đoán trực quan thành công tại `plots/sample_clean_noisy_pairs.png`.

---

## 5. NHỮNG GÌ CHƯA HOÀN THÀNH HOẶC CÒN THIẾU

1. **Huấn luyện mô hình Denoising (Level 2 Training):**
   - Chưa thực hiện theo đúng chỉ thị trong mục tiêu của Phase 2: *"Do not train the denoising model yet unless the dataset implementation and validation have been completed and the user explicitly requests the training stage."*
   - Cần thiết kế kiến trúc mô hình (ví dụ 1D Fully-Convolutional Autoencoder hoặc U-Net 1D) và pipeline training.
2. **Kịch bản huấn luyện & đánh giá định lượng SNR sau khử nhiễu (Denoising Evaluation Metrics):**
   - Cần triển khai các hàm đánh giá: $\text{SNR}_{\text{improvement}}$, Root Mean Square Error (RMSE), Percent Root Mean Square Difference (PRD).

---

## 6. HÀNH ĐỘNG TIẾP THEO CHÍNH XÁC (PRECISE NEXT ACTION)

1. Lưu báo cáo xây dựng dataset chi tiết tại `reports/level2/dataset_build_report.md`.
2. Bổ sung kết quả nghiệm thu thực tế vào `reports/level2/level2_dataset_research_report.md`.
3. Báo cáo tổng thể cho người dùng (bằng tiếng Việt) để kết thúc giai đoạn xây dựng & thẩm định dataset Level 2, sẵn sàng chuyển sang bước huấn luyện mô hình khi người dùng yêu cầu.
