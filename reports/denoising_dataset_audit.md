# BÁO CÁO THẨM ĐỊNH DATASET ECG DENOISING (DATASET AUDIT REPORT)

> **Dự án:** ECG Denoising & Classification (Level 2 — Phase 2)  
> **Ngày thực hiện thẩm định:** 13/10/2026  
> **Môi trường thực thi quy chuẩn:** Conda environment `ecg` (`C:\Users\vietc\.conda\envs\ecg\python.exe`)  
> **Trạng thái thẩm định:** Đã đối soát toàn diện mã nguồn, cấu hình và tệp dữ liệu thực tế.

---

## 1. TỔNG QUAN HỆ THỐNG VÀ ĐỊA CHỈ DỮ LIỆU

### 1.1 Phân định rõ ràng giữa dữ liệu Phase 1 và Level 2
Trong repository, hiện đang có **hai hệ thống dữ liệu hoàn toàn khác nhau** phục vụ hai bài toán độc lập:

1. **Dữ liệu Phase 1 (ECG Beat Classification — Phân loại nhịp tim):**
   - **Đường dẫn trong repo:** `data/processed/train.npz`, `val.npz`, `test.npz`.
   - **Đặc điểm:** Tín hiệu được cắt từng nhịp đơn lẻ căn chỉnh theo đỉnh R (R-peak centered), chiều dài chỉ **260 mẫu** (~0.72 giây), chuẩn hóa **Per-beat Z-score**, kèm nhãn phân loại AAMI ($N, S, V, F, Q$).
   - **Đánh giá:** Tập dữ liệu này **HOÀN TOÀN KHÔNG DÙNG ĐƯỢC** cho bài toán ECG Denoising vì chiều dài quá ngắn (chưa được 1 chu kỳ tim hoàn chỉnh), Z-score từng nhịp đã phá vỡ tỉ lệ biên độ và công suất tự nhiên giữa ECG và nhiễu.

2. **Dữ liệu Level 2 (ECG Denoising — Khử nhiễu dạng sóng có giám sát):**
   - **Tệp cấu hình:** [configs/denoising_dataset.yaml](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/configs/denoising_dataset.yaml)
   - **Đường dẫn lưu trữ thực tế trên đĩa cứng:** `C:/d2l-en/data/ECG_Project_Data/Processed/denoising_v1`
   - **Mã nguồn loader & pipeline:**
     - [src/data/nstdb_loader.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/src/data/nstdb_loader.py)
     - [src/data/denoising_data.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/src/data/denoising_data.py)
     - [scripts/build_denoising_dataset.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/scripts/build_denoising_dataset.py)
     - [scripts/validate_denoising_dataset.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/scripts/validate_denoising_dataset.py)

---

## 2. CẤU TRÚC VÀ ĐẶC ĐIỂM KỸ THUẬT CỦA DATASET LEVEL 2 (ĐÃ XÁC MINH)

### 2.1 Cấu trúc thư mục dữ liệu `denoising_v1`
Dữ liệu Level 2 đã được biên dịch hoàn tất và thẩm định tự động với 57/57 tiêu chí PASS. Thư mục bao gồm:

```
C:/d2l-en/data/ECG_Project_Data/Processed/denoising_v1/
├── clean_windows/                     # Mảng tín hiệu sạch tham chiếu (Clean Reference)
│   ├── train_clean.npz               # 24,092 cửa sổ (44.1 MB)
│   ├── val_clean.npz                 #  5,072 cửa sổ ( 9.3 MB)
│   └── test_clean.npz                # 26,628 cửa sổ (48.8 MB)
├── noisy_eval/                        # Dữ liệu kiểm định nhiễu tiền tạo lập (Deterministic)
│   ├── val_noisy.npz                 # 91,296 cặp Noisy/Clean (343.2 MB)
│   └── test_noisy.npz                # 479,304 cặp Noisy/Clean (1.75 GB)
├── manifests/                         # Metadata chi tiết từng cửa sổ
│   ├── train_manifest.csv            # 24,092 dòng
│   ├── val_manifest.csv              #  5,072 dòng
│   ├── test_manifest.csv             # 26,628 dòng
│   ├── val_noisy_manifest.csv        # 91,296 dòng
│   ├── test_noisy_manifest.csv       # 479,304 dòng
│   └── noise_blocks.json             # Bản đồ phân bổ phân đoạn thời gian của các loại nhiễu
├── plots/                             # Đồ thị đối chứng trực quan
│   └── sample_clean_noisy_pairs.png  # Đồ thị waveform sạch đối chiếu 6 mức SNR
├── dataset_info.json                  # Metadata tổng thể của toàn bộ dataset
├── build_log.txt                      # Nhật ký thời gian xây dựng (114.7 giây)
└── validation_report.txt              # Nhật ký kiểm thử toàn vẹn (57 checks PASS)
```

### 2.2 Thông số tín hiệu và hình dạng tensor (Tensor Shapes)
- **Tần số lấy mẫu ($f_s$):** **360 Hz** (chuẩn của cả MITDB và NSTDB).
- **Chuyển đạo (Lead):** **MLII** (Modified Limb Lead II), kênh 0 trong bản ghi MITDB.
- **Kích thước cửa sổ (Window Size):** **1024 mẫu** ($\approx 2.84$ giây tại 360 Hz). Chiều dài này đủ dài để bao quát từ 2 đến 4 chu kỳ tim hoàn chỉnh (RR intervals), cung cấp đầy đủ ngữ cảnh thời gian (P-QRS-T) cho mô hình học.
- **Bước trượt (Stride):** **512 mẫu** (50% chồng lấn). Không dùng zero-padding; bỏ phần thừa ở cuối bản ghi nếu không đủ 1024 mẫu.
- **Kiểu dữ liệu (Data Type):** `float32`.
- **Kích thước tensor đầu vào/đầu ra:**
  - Đơn lẻ: `(1, 1024)` (kênh tín hiệu 1D).
  - Theo mẻ (Batch): `(Batch_Size, 1, 1024)`.

---

## 3. PHÂN CHIA TẬP DỮ LIỆU VÀ CHỐNG RÒ RỈ THÔNG TIN (DATA LEAKAGE)

### 3.1 Phân chia theo bệnh nhân (Patient-wise Split)
Dataset Level 2 sử dụng 44 bản ghi sạch từ MITDB (đã loại bỏ 4 bản ghi nhịp tạo nhịp có dị tật kim nhọn nhân tạo: `102, 104, 107, 217`).

| Tập dữ liệu | Số bản ghi MITDB | Danh sách bản ghi | Số cửa sổ sạch ($X_{\text{clean}}$) | Tỉ lệ (%) |
|---|---|---|---|---|
| **Train** | 19 | 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, **201, 202**, 205, 207, 208, 209, 215 | 24,092 | 43.18% |
| **Validation** | 4 | 203, 220, 223, 230 | 5,072 | 9.09% |
| **Test** | 21 | 100, 103, 105, 111, 113, 117, 121, 123, 200, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234 | 26,628 | 47.73% |

> [!IMPORTANT]
> **Khắc phục triệt để lỗi rò rỉ danh tính bệnh nhân của Phase 1:**  
> Ở Phase 1 (theo phân chia De Chazal 2004), bản ghi 201 nằm ở DS1 (Train) và bản ghi 202 nằm ở DS2 (Test), mặc dù hai bản ghi này thuộc về **cùng một bệnh nhân**. Tại Level 2, cả hai bản ghi 201 và 202 được gom trọn vẹn vào tập **Train**. Kiểm định giao tập cho thấy:
> $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$

### 3.2 Phân chia khối nhiễu NSTDB chống rò rỉ thời gian (Noise Block Allocation)
Nguồn nhiễu được lấy từ cơ sở dữ liệu MIT-BIH Noise Stress Test Database (NSTDB) gồm 3 loại:
1. `bw`: Baseline Wander (nhiễu trôi đường đẳng điện do hô hấp, tần số thấp $< 1\text{ Hz}$).
2. `ma`: Muscle Artifact (nhiễu co cơ vân, dải rộng tần số cao).
3. `em`: Electrode Motion (nhiễu dịch chuyển điện cực, xung đột biến dạng lớn, dễ nhầm với ngoại tâm thu thất).

Mỗi bản ghi nhiễu NSTDB dài 650,000 mẫu (~30.1 phút). Pipeline chia thành 6 khối độc lập (100,000 mẫu/khối, bỏ 500 mẫu guard band tại biên):
- **Khối 0, 1, 2, 3 (mẫu 0 – 400,000):** Dành riêng cho tập **Train** (chiếm ~67%).
- **Khối 4 (mẫu 400,000 – 500,000):** Dành riêng cho tập **Validation** (chiếm ~17%).
- **Khối 5 (mẫu 500,000 – 600,000):** Dành riêng cho tập **Test** (chiếm ~17%).

**Kết quả kiểm tra:** Không có sự trùng lặp phân đoạn nhiễu hay ranh giới thời gian giữa Train, Val và Test.

---

## 4. TIỀN XỬ LÝ VÀ CHUẨN HÓA TÍN HIỆU

### 4.1 Bộ lọc tín hiệu sạch tham chiếu (Clean Reference Filtering)
- **Phương pháp:** Bộ lọc số Butterworth IIR bậc 4, dải thông $[0.05\text{ Hz}, 60.0\text{ Hz}]$, lọc 2 chiều qua hàm `scipy.signal.filtfilt` (Zero-phase filtering, không làm lệch pha thời gian).
- **Lý do chọn ngưỡng cắt thấp $0.05\text{ Hz}$ thay vì $0.5\text{ Hz}$ của Phase 1:**
  - Ở Phase 1, mục tiêu là phân loại nhịp tim nên cần triệt tiêu tối đa trôi đường đẳng điện ($0.5\text{ Hz}$).
  - Ở Level 2, mục tiêu là **mô hình học sâu tự học cách khử trôi đường đẳng điện**. Nếu lọc $0.5\text{ Hz}$, ta đã vô tình làm mất thông tin tần số thấp tự nhiên của ECG. Mức $0.05\text{ Hz}$ (chuẩn AHA/IEC cho chẩn đoán tim mạch) chỉ loại bỏ thành phần trôi thuần DC ($0\text{ Hz}$), giữ nguyên dạng sóng để mô hình denoising xử lý.
- **Ngưỡng cắt cao $60.0\text{ Hz}$:** Triệt tiêu nhiễu lưới điện xoay chiều (60 Hz powerline noise).

### 4.2 Phương pháp chuẩn hóa (Normalization)
- **Phương pháp áp dụng:** `record_max_abs`  
  $$x_{\text{norm}}(t) = \frac{x(t)}{\max |x_{\text{record}}|}$$
- **Ý nghĩa:** Chia toàn bộ tín hiệu trong một bản ghi cho giá trị tuyệt đối lớn nhất của bản ghi đó. Biên độ sau chuẩn hóa nằm trọn vẹn trong khoảng $[-1.0, 1.0]$.
- **Tại sao không dùng Z-score?**
  Nếu chuẩn hóa Z-score ($\frac{x - \mu}{\sigma}$) trên từng cửa sổ nhỏ, tỉ lệ công suất giữa sóng R và nhiễu sẽ bị thay đổi ngẫu nhiên, phá vỡ công thức tính toán tỉ lệ công suất SNR vật lý.

---

## 5. TỔNG HỢP NHIỄU VÀ ĐỘ CHÍNH XÁC SNR

### 5.1 Công thức toán học ghép cặp tín hiệu
Tín hiệu nhiễu đầu vào ($x_{\text{noisy}}$) được tổng hợp có kiểm soát từ tín hiệu sạch ($x_{\text{clean}}$) và tín hiệu nhiễu thô ($x_{\text{noise}}$):
$$P_{\text{clean}} = \frac{1}{W} \sum_{i=1}^{W} x_{\text{clean}}[i]^2, \quad P_{\text{noise}} = \frac{1}{W} \sum_{i=1}^{W} x_{\text{noise}}[i]^2$$
$$\alpha = \sqrt{\frac{P_{\text{clean}}}{P_{\text{noise}} \cdot 10^{\text{SNR}_{\text{target}} / 10}}}$$
$$x_{\text{noisy}} = x_{\text{clean}} + \alpha \cdot x_{\text{noise}}$$

### 5.2 Lưới mức SNR chuẩn hóa
Áp dụng 6 mức SNR chuẩn quốc tế của NSTDB:  
$$\text{SNR} \in \{-6\text{ dB}, 0\text{ dB}, 6\text{ dB}, 12\text{ dB}, 18\text{ dB}, 24\text{ dB}\}$$
- **$-6\text{ dB}, 0\text{ dB}$:** Mức nhiễu cực nặng (công suất nhiễu gấp 4 lần hoặc bằng công suất tim), sóng ECG gần như bị chìm hoàn toàn.
- **$6\text{ dB}, 12\text{ dB}$:** Mức nhiễu trung bình (tương đương khi bệnh nhân vận động, thở mạnh).
- **$18\text{ dB}, 24\text{ dB}$:** Mức nhiễu nhẹ (tín hiệu ECG rõ, có gợn lăn tăn).

### 5.3 Độ chính xác thực tế đo được
- Sai số SNR tuyệt đối trung bình: $\approx 3 \times 10^{-16}\text{ dB}$ (đạt mức giới hạn chính xác số học của máy tính).
- Tính ghép cặp (Paired Alignment): Cực kỳ chính xác, sample $i$ của $x_{\text{noisy}}$ tương ứng $100\%$ về thời gian với sample $i$ của $x_{\text{clean}}$.

---

## 6. ĐÁNH GIÁ ĐIỀU KIỆN HUẤN LUYỆN VÀ NHỮNG ĐIỂM CẦN LƯU Ý

### 6.1 Dataset đã đủ điều kiện để huấn luyện Denoising hay chưa?
**KẾT LUẬN: ĐÃ ĐỦ ĐIỀU KIỆN 100% VỀ MẶT DỮ LIỆU.**  
Cả ba cấu phần thiết yếu cho Supervised Learning đều đã sẵn sàng:
1. Tập tín hiệu sạch chuẩn hóa (`train_clean.npz`, `val_clean.npz`, `test_clean.npz`).
2. Tập đánh giá nhiễu cố định chuẩn mực cho benchmark ngoại tuyến (`val_noisy.npz`, `test_noisy.npz`).
3. Lớp nạp dữ liệu PyTorch `DenoisingDataset` hỗ trợ tiêm nhiễu ngẫu nhiên on-the-fly cho huấn luyện.

### 6.2 Những lưu ý kỹ thuật quan trọng cho phase huấn luyện tiếp theo (Chỉ đề xuất giải pháp)
1. **Lưu ý về bộ sinh số ngẫu nhiên on-the-fly trong `DenoisingDataset`:**
   - Trong `src/data/denoising_data.py` (dòng 264), hạt giống ngẫu nhiên đang được tính bằng `rng = np.random.default_rng(self.seed ^ idx)`.
   - *Hạn chế:* Công thức này chỉ phụ thuộc vào chỉ số mẫu `idx`. Do đó, qua từng epoch, mẫu số `idx` sẽ luôn nhận cùng một loại nhiễu và cùng mức SNR.
   - *Giải pháp đề xuất (cho bước sau):* Khi triển khai huấn luyện, cần truyền thêm tham số `epoch` vào hoặc sử dụng `np.random.default_rng()` động để mỗi epoch mô hình được học một biến thể nhiễu mới (Data Augmentation tối đa).
2. **Dung lượng lớn của tập Test Noisy:**
   - File `test_noisy.npz` có dung lượng $1.75\text{ GB}$ (479,304 mẫu).
   - *Giải pháp đề xuất:* Khi chạy script đánh giá sau này, nên sử dụng cờ `np.load(..., mmap_mode='r')` để đọc trực tiếp từ đĩa cứng theo từng batch, tránh tải toàn bộ vào RAM gây tràn bộ nhớ.
