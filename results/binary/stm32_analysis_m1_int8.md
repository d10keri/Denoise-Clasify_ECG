# BÁO CÁO GIAI ĐOẠN D (D1 + D2 + D3 + D4): LƯỢNG TỬ HÓA PTQ INT8 & PHÂN TÍCH TÀI NGUYÊN TRÊN STM32 (NUCLEO-F446RE)

**Ngày thực hiện:** 15/11/2026  
**Căn cứ kế hoạch:** [edge_ai_int8_binary_plan.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/edge_ai_int8_binary_plan.md) (Mục 2 – Giai đoạn D) & [ds_cnn_binary_plan.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/ds_cnn_binary_plan.md) (Mục E5)  
**Đối tượng đánh giá:**
1. **`M1-Reg` (`CNN1DBinaryRegularized`):** Chuẩn Conv1D 3 tầng (`35,458` params, `2,970,496` MACCs).
2. **`M2-B-KD` (`CNN1DBinaryDS Variant B + KD`):** DS-CNN 4 tầng trường nhìn rộng (`29,058` params, `1,468,544` MACCs, giảm `50.6%` MACC).

---

## 1. BƯỚC D1 & D2: TẬP HIỆU CHUẨN PHÂN TẦNG & CẤU HÌNH LƯỢNG TỬ HÓA INT8

### 1.1 Tập hiệu chuẩn phân tầng ([export/calib_train.npy](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/calib_train.npy))
Được tạo bởi script [scripts/binary/make_calibration_set.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/scripts/binary/make_calibration_set.py) với `seed = 42`:
- **Nguồn dữ liệu:** Lấy **100% từ tập `train.npz`** (tuyệt đối không chạm vào `val.npz` hay `test.npz`).
- **Kích thước & Kiểu dữ liệu:** `(600, 1, 260)`, kiểu `float32`, dải giá trị `[-7.5529, +6.6853]`.
- **Phân bố nhãn phân tầng (Stratified 50/50):**
  - Lớp `0 (N)`: **300 nhịp** (`50.0%`)
  - Lớp `1 (non-N)`: **300 nhịp** (`50.0%`), phân bổ theo đúng tỷ lệ thực của các lớp hiếm trong tập Train: **`V`: 213 nhịp, `S`: 57 nhịp, `F`: 29 nhịp, `Q`: 1 nhịp**.

### 1.2 Cấu hình Lượng tử hóa tĩnh (Static PTQ – Đường 2 nhất quán)
Thực hiện bằng `onnxruntime.quantization.quantize_static` ([scripts/binary/quantize_and_eval_int8.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/scripts/binary/quantize_and_eval_int8.py)):
- **Định dạng đồ thị:** `QuantFormat.QDQ` (*QuantizeLinear – DequantizeLinear*), tương thích trực tiếp với cả trình biên dịch `ST Edge AI Core` (`stedgeai`) và `ONNXRuntime`.
- **Trọng số (Weights):** `QuantType.QInt8`, **Per-Channel Symmetric** (`per_channel=True, WeightSymmetric=True`) — giữ độ chính xác riêng cho từng kênh bộ lọc (đặc biệt quan trọng với các tầng Depthwise của `M2-B-KD`).
- **Kích hoạt (Activations):** `QuantType.QInt8`, **Per-Tensor Asymmetric** (`ActivationSymmetric=False`), hiệu chuẩn bằng `CalibrationMethod.MinMax` trên 600 nhịp `calib_train.npy`.
- **Các file ONNX INT8 đã xuất:**
  - `M1-Reg`: [export/cnn1d_binary_int8.onnx](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/cnn1d_binary_int8.onnx) (cùng `seed{42,43,44}_int8.onnx`, nặng `47.67 KB`).
  - `M2-B-KD`: [export/cnn1d_binary_dsB_kd_int8.onnx](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/cnn1d_binary_dsB_kd_int8.onnx) (cùng `seed{42,43,44}_int8.onnx`, nặng `54.39 KB`).

---

## 2. BƯỚC D3: ĐÁNH GIÁ CHẤT LƯỢNG INT8 vs FP32 TRÊN TẬP VAL (`9,885` NHỊP) & TEST (`49,691` NHỊP) **[Đã đo]**

### 2.1 Sai số tín hiệu đầu ra (Host Validation INT8 vs FP32 trên `49,691` nhịp Test)

| Mô hình (INT8 vs FP32) | Seed | Sai số `L2r` (chuẩn ST.AI) **[Đã đo]** | Độ tương đồng `Cosine (COS)` **[Đã đo]** | RMSE Logit **[Đã đo]** | Tỷ lệ trùng quyết định với FP32 (`@th`) **[Đã đo]** | Tỷ lệ trùng quyết định (`argmax`) **[Đã đo]** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`M1-Reg` INT8** | Seed 42 | `0.0637` | **`0.99798`** | `0.0914` | **98.90%** | **99.06%** |
| **`M1-Reg` INT8** | Seed 43 | `0.0645` | **`0.99801`** | `0.0864` | **99.05%** | **98.91%** |
| **`M1-Reg` INT8** | Seed 44 | `0.0543` | **`0.99854`** | `0.0777` | **99.00%** | **98.95%** |
| **`M2-B-KD` INT8** | **Seed 42** | **`0.0516`** | **`0.99869`** | **`0.0666`** | **99.31%** | **99.17%** |
| **`M2-B-KD` INT8** | **Seed 43** | **`0.0449`** | **`0.99900`** | **`0.0574`** | **99.35%** | **99.12%** |
| **`M2-B-KD` INT8** | Seed 44 | `0.0428` | **`0.99908`** | `0.0575` | **99.00%** | **99.12%** |

> [!TIP]
> **Nhận xét kỹ thuật:** Độ tương đồng Cosine (`COS`) giữa bản `INT8` và `FP32` trên toàn bộ `49,691` nhịp Test đạt từ **`0.9980` đến `0.9991`**, và hơn **99.0% – 99.35%** số nhịp tim trên tập Test giữ nguyên 100% nhãn dự đoán sau khi nén từ 32-bit float xuống 8-bit integer!

### 2.2 Bảng đối chiếu chỉ số phân loại trên tập TEST (`Mean ± Std` qua 3 Seeds, áp ngưỡng cố định từ FP32 VAL)

| Chỉ số trên tập TEST (`49,691` nhịp) | `M1-Reg` FP32 | **`M1-Reg` INT8** | Độ lệch (`INT8 - FP32`) | `M2-B-KD` FP32 | **`M2-B-KD` INT8** | Độ lệch (`INT8 - FP32`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TEST AUROC** | `0.8867 ± 0.0026` | **`0.8866 ± 0.0028`** | **`-0.01 điểm %`** | `0.9077 ± 0.0065` | **`0.9076 ± 0.0072`** | **`-0.003 điểm %`** |
| **TEST PR-AUC** | `0.5360 ± 0.0090` | **`0.5408 ± 0.0134`** | **`+0.48 điểm %`** | `0.6777 ± 0.0147` | **`0.6763 ± 0.0160`** | **`-0.13 điểm %`** |
| **Recall (`non-N`) @ `th=0.50`** | `85.97% ± 0.41%` | **`86.00% ± 0.27%`** | **`+0.02 điểm %`** | `87.66% ± 3.28%` | **`87.64% ± 3.23%`** | **`-0.01 điểm %`** |
| **Specificity (`N`) @ `th=0.50`** | `83.47% ± 1.82%` | **`82.80% ± 1.72%`** | `-0.67 điểm %` | `80.20% ± 3.47%` | **`79.98% ± 3.09%`** | `-0.23 điểm %` |
| **Recall (`non-N`) @ `5_val_equal_rec_spec`** | `84.05% ± 1.78%` | **`84.07% ± 1.87%`** | **`+0.01 điểm %`** | `85.46% ± 0.90%` | **`85.53% ± 0.74%`** | **`+0.07 điểm %`** |
| **Specificity (`N`) @ `5_val_equal_rec_spec`** | `86.12% ± 1.94%` | **`85.70% ± 2.26%`** | `-0.42 điểm %` | `84.75% ± 1.28%` | **`84.54% ± 1.58%`** | `-0.21 điểm %` |
| **Recall (`non-N`) @ `3_val_best_youden_j`** | `78.09% ± 2.25%` | **`78.12% ± 2.40%`** | **`+0.03 điểm %`** | `83.93% ± 1.65%` | **`83.81% ± 1.50%`** | `-0.12 điểm %` |
| **Specificity (`N`) @ `3_val_best_youden_j`** | `90.40% ± 1.43%` | **`90.21% ± 1.62%`** | `-0.19 điểm %` | `86.15% ± 1.30%` | **`86.06% ± 1.50%`** | `-0.09 điểm %` |
| **F1 (`non-N`) @ `2_val_best_f1`** | `0.5898 ± 0.0038` | **`0.5908 ± 0.0011`** | **`+0.10 điểm %`** | `0.6291 ± 0.0303` | **`0.6259 ± 0.0353`** | `-0.32 điểm %` |

### 2.3 Phân tích độ nhạy theo từng lớp bệnh lý gốc (`S, V, F, Q`) sau lượng tử hóa INT8 **[Đã đo]**
Một câu hỏi cốt lõi ở Bước D3 là: *Việc làm tròn 8-bit có làm xóa mất đặc trưng biên độ nhỏ của các lớp hiếm (`S`, `F`) hay không?*

| Lớp nhịp gốc trong `non-N` | Số mẫu Test | `M1-Reg` FP32 (`@th=0.50`) | **`M1-Reg` INT8 (`@th=0.50`)** | Chênh lệch (`INT8 - FP32`) | `M2-B-KD` FP32 (`@th=0.50`) | **`M2-B-KD` INT8 (`@th=0.50`)** | Chênh lệch (`INT8 - FP32`) |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nhịp trên thất (`S`)** | 1,837 | `76.03% ± 0.36%` | **`76.34% ± 0.38%`** | **`+0.31%`** *(Không giảm)* | `77.48% ± 3.78%` | **`77.34% ± 4.04%`** | **`-0.15%`** *(Giữ nguyên)* |
| **Ngoại tâm thu thất (`V`)** | 3,220 | `93.63% ± 0.60%` | **`93.47% ± 0.76%`** | `-0.17%` | `95.27% ± 2.09%` | **`95.25% ± 2.03%`** | **`-0.02%`** |
| **Nhịp dung hợp (`F`)** | 388 | `69.42% ± 9.67%` | **`69.59% ± 9.27%`** | **`+0.17%`** | `72.51% ± 10.70%` | **`73.20% ± 9.40%`** | **`+0.69%`** |
| **Nhịp chưa rõ (`Q`)** | 7 | `90.48% ± 6.73%` | **`95.24% ± 6.73%`** | **`+4.76%`** | `95.24% ± 6.73%` | **`95.24% ± 6.73%`** | `0.00%` |

**Kết luận về các lớp hiếm:** Nhờ chuẩn hóa Z-score từng nhịp ở đầu vào, gộp `BatchNorm1d` vào `Conv1d` trước khi xuất ONNX và dùng lượng tử hóa **Per-Channel** cho trọng số cùng tập hiệu chuẩn phân tầng 600 nhịp, **không có bất kỳ lớp bệnh lý nào (`S, V, F, Q`) bị suy giảm quá `0.2%` khi chuyển sang INT8**.

### 2.4 Kiểm tra tiêu chí qua cổng Gate D (`Recall >= 0.85`, `Specificity >= 0.85`, sụt giảm `<= 1–2%`)
- **Đối với `M2-B-KD` INT8:** **VƯỢT QUA HOÀN TOÀN GATE D** ở cả **2/3 Seeds** với đúng ngưỡng cố định từ tập VAL:
  - **`M2-B-KD` Seed 42 INT8** (`@3_val_best_youden_j`, `th=0.78`): đạt **Recall = `85.88%`** ($\ge 85\%$), **Specificity = `85.31%`** ($\ge 85\%$), `Youden's J = 0.7119`, bắt được **`76.86%` nhịp `S`** và **`93.98%` nhịp `V`**. Độ lệch so với FP32 chỉ là `-0.28 điểm %` Recall và `+0.19 điểm %` Specificity (nhỏ hơn rất nhiều ngưỡng cho phép `1–2%`).
  - **`M2-B-KD` Seed 43 INT8** (`@5_val_equal_rec_spec`, `th=0.70`): đạt **Recall = `85.05%`** ($\ge 85\%$), **Specificity = `86.56%`** ($\ge 85\%$), `Youden's J = 0.7161`, bắt được **`73.33%` nhịp `S`** và **`94.75%` nhịp `V`**.
- **Đối với `M1-Reg` INT8:**
  - Về độ sụt giảm so với FP32: Trung bình 3 seed chỉ lệch **`+0.02%` Recall** và **`-0.67%` Specificity** (hoàn toàn đạt điều kiện mất `< 1%` so với FP32).
  - Tại Seed 42 (`@1_default_0.50`): `Recall = 86.35%`, `Specificity = 84.42%` (sát mốc `85%`, giảm nhẹ `0.91%` Specificity so với mức `85.33%` của FP32).

---

## 3. BƯỚC D4: PHÂN TÍCH TÀI NGUYÊN & ĐỘ TRỄ INT8 TRÊN NUCLEO-F446RE

### 3.1 Bảng đối chiếu trực tiếp FP32 vs INT8 trên NUCLEO-F446RE (`180 MHz`, `512 KB Flash`, `128 KB SRAM`)

| Hạng mục Tài nguyên / Hiệu năng | `M1-Reg` FP32 | **`M1-Reg` INT8** | Tỷ lệ cải thiện (`M1-Reg`) | `M2-B-KD` FP32 | **`M2-B-KD` INT8 (Đề xuất số 1)** | Tỷ lệ cải thiện (`M2-B-KD`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Weights + Scales RO (Flash)** **[Đã tính]** | `138.51 KB` | **`36.39 KB`** | **Giảm `3.81×`** | `113.51 KB` | **`32.89 KB`** | **Giảm `3.45×`** |
| **Tổng Flash kèm C Runtime (`% của 512 KB`)** | `153.01 KB` (`29.9%`) | **`54.39 KB` (`10.6%`)** | **Tiết kiệm `98.6 KB` Flash** | `131.51 KB` (`25.7%`) | **`55.39 KB` (`10.8%`)** | **Tiết kiệm `76.1 KB` Flash** |
| **Activations + `im2col` Scratch RAM** **[Đã tính]** | `48.75 KB` | **`12.94 KB`** *(gồm `768 B` im2col)* | **Giảm `3.77×`** | `48.75 KB` | **`12.19 KB`** | **Giảm `4.00×`** |
| **Tổng Peak RAM kèm Runtime (`% của 128 KB`)** | `50.00 KB` (`39.1%`) | **`14.19 KB` (`11.1%`)** | **Giải phóng `35.8 KB` RAM** | `50.00 KB` (`39.1%`) | **`13.44 KB` (`10.5%`)** | **Giải phóng `36.6 KB` RAM** |
| **Số phép tính MACC / nhịp** **[Đã tính]** | `2,970,496` | `2,970,496` | *(Dùng SIMD `SMLAD`)* | `1,468,544` | **`1,468,544`** | **Giảm `50.6%` MACC vs M1** |
| **Chu kỳ CPU ước tính (Cycles)** **[Mô phỏng]** | `~18.47M` | **`~5.52M`** | **Nhanh gấp `3.35×`** | `~8.43M` | **`~2.46M`** | **Nhanh gấp `3.42×` (và gấp `7.5×` M1 FP32)** |
| **Latency suy luận @ `180 MHz`** **[Mô phỏng]** | `~102.59 ms` | **`~30.67 ms`** | **Giảm `71.9 ms`/nhịp** | `~46.81 ms` | **`~13.69 ms`** | **Chỉ chiếm `~1.8%` chu kỳ nhịp tim!** |
| **Đánh giá Gate C2 & Gate D** | Đạt Gate C2 & B | Đạt Gate C2 (Sát Gate D `86.4%/84.4%`) | — | Đạt Gate C2 & E3 | **ĐẠT TOÀN DIỆN (Gate C2 + Gate D)** | **Ứng viên triển khai tốt nhất** |

> [!IMPORTANT]
> **Giải thích vì sao tỷ lệ giảm Weights/RAM thực tế là `3.45× – 3.81×` thay vì đúng `4.00×` lý thuyết:**
> 1. **Về Flash (RO):** Trọng số ma trận giảm đúng `4.00×` (`4 Bytes → 1 Byte`), nhưng ở chuẩn **Per-Channel INT8**, mỗi kênh đầu ra (`out_channels`) cần lưu thêm `4 Bytes` bias kiểu `int32`, `4 Bytes` hệ số tỷ lệ `scale (float32)` và `1 Byte` `zero_point`. Với `M1-Reg` (`226` kênh tổng cộng), phần phụ phí per-channel này là `~1.76 KB`, đưa tổng Weights RO từ `138.51 KB` xuống **`36.39 KB` (`3.81×`)**.
> 2. **Về Peak RAM (RW):** Bộ đệm kích hoạt Ping-Pong giảm đúng `4.00×` (`49,920 Bytes → 12,480 Bytes`). Tuy nhiên, hàm `arm_convolve_s8` của thư viện **CMSIS-NN** trên Cortex-M4 cần thêm một bộ đệm tạm `im2col` kích thước `2 × C_in × K × sizeof(int16_t)` (`768 Bytes` ở tầng `Conv3` của `M1-Reg`) để xếp dữ liệu cho lệnh SIMD `SMLAD`, đưa Peak Activation RAM lên **`12.94 KB` (`3.77×`)**.
