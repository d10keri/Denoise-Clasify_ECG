# Báo cáo Giai đoạn A: Đánh giá Trọng số Mô hình M2-B-KD & Khả năng Xử lý của ESP32-S3 (N16R8)

**Mô hình đánh giá:** `M2-B-KD` ([export/cnn1d_binary_dsB_kd_fp32.onnx](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/cnn1d_binary_dsB_kd_fp32.onnx))  
**Vi điều khiển mục tiêu:** `ESP32-S3 N16R8` (Dual-core Xtensa LX7 @ 240 MHz, 512 KB Internal SRAM, 16 MB Flash, 8 MB Octal PSRAM)  
**Ghi chú nguồn số liệu:**
- **[Đã tính từ đồ thị ONNX tĩnh]**: Kích thước tensor, số lượng tham số, phân phối trọng số, SQNR lượng tử hóa lũy thừa 2 (`2^n`), MACC từng lớp, độ căn lề kênh PIE SIMD và bộ nhớ Flash/SRAM.
- **[Đã kiểm chứng bằng `esp-ppq 1.3.14`]**: Khả năng đọc và ánh xạ toán tử 1D của đồ thị ONNX.
- **[Ước tính lý thuyết]**: Số chu kỳ CPU và thời gian suy luận (ms) trên Xtensa LX7 @ 240 MHz với hiệu suất giả định 15%–30%.

---

## 1. Bảng Cổng Kiểm duyệt Giai đoạn A (Gate A)

| Tiêu chí Gate A | Yêu cầu | Kết quả Đánh giá | Trạng thái |
| :--- | :--- | :--- | :---: |
| **1. Bộ nhớ Trọng số (Flash)** | `< 16,384 KB` (16 MB Flash) | FP32: `113.51 KB`, INT16: `59.38 KB`, INT8: `32.57 KB` (`< 0.7%` Flash) | **PASS** |
| **2. Bộ nhớ Activation (SRAM nội)** | `< 320 KB` SRAM nội khả dụng (không cần PSRAM) | Đỉnh Ping-Pong + I/O: FP32 `50.77 KB`, INT16 `25.88 KB`, INT8 `13.44 KB` | **PASS** |
| **3. Tương thích Toán tử & PIE SIMD** | Hỗ trợ toàn bộ toán tử trong `esp-ppq` / `ESP-DL` | 6/6 toán tử (`Conv`, `Relu`, `MaxPool`, `AveragePool`, `Flatten`, `Gemm`) nạp thành công vào IR của `esp-ppq 1.3.14`; `94.18%` MACC căn lề 16-lane PIE | **PASS** |
| **4. Độ trễ INT8 ước tính (@ 240 MHz)** | `< 10.0 ms / nhịp` | INT8 PIE (16-lane, hiệu suất 15–30%): **`1.3 – 2.6 ms`** (INT16: `2.6 – 5.2 ms`) | **PASS** |

### Kết luận Gate A: **PASS (ĐẠT TOÀN BỘ — ĐỦ ĐIỀU KIỆN CHUYỂN SANG GIAI ĐOẠN B)**

---

## 2. Phân tích Chi tiết Từng Lớp: Cấu trúc, MACC & Độ Tương thích SIMD PIE 128-bit

ESP32-S3 tích hợp tập lệnh vector **PIE (Processor Instruction Extensions) 128-bit**, xử lý đồng thời **16 phần tử INT8** hoặc **8 phần tử INT16** trong một lệnh SIMD. Các phép nhân tích chập đạt hiệu suất tối đa khi số kênh vào/ra là **bội số của 16** (với INT8) hoặc **8** (với INT16).

| # | Lớp (ONNX Node) | Loại Toán tử | Input Shape | Output Shape | Weight Shape | Tham số | MACC | % MACC | Căn lề PIE (INT8 16-lane) |
| :-: | :--- | :--- | :---: | :---: | :---: | ---: | ---: | ---: | :--- |
| 1 | `/features/features.0/Conv` | StandardConv1d (k=7) | `[1, 1, 260]` | `[1, 32, 260]` | `[32, 1, 7]` | `256` | `58,240` | `3.89%` | in_ch=1 (unaligned input), out_ch=32 (16-aligned); low SIMD utilization on input reduction |
| 2 | `/features/features.1/Relu` | ReLU (in-place fused) | `[1, 32, 260]` | `[1, 32, 260]` | `—` | `0` | `0` | `0.0%` | Non-MAC / activation / pooling layer |
| 3 | `/features/features.2/MaxPool` | MaxPool1d (k=2) | `[1, 32, 260]` | `[1, 32, 130]` | `—` | `0` | `8,320` | `0.56%` | Non-MAC / activation / pooling layer |
| 4 | `/features/features.3/Conv` | DepthwiseConv1d (k=9, g=32) | `[1, 32, 130]` | `[1, 32, 130]` | `[32, 1, 9]` | `320` | `37,440` | `2.5%` | Vectorizes across 32 channels (multiple of 16: True) |
| 5 | `/features/features.4/Relu` | ReLU (in-place fused) | `[1, 32, 130]` | `[1, 32, 130]` | `—` | `0` | `0` | `0.0%` | Non-MAC / activation / pooling layer |
| 6 | `/features/features.5/Conv` | PointwiseConv1d (1x1) | `[1, 32, 130]` | `[1, 64, 130]` | `[64, 32, 1]` | `2,112` | `266,240` | `17.78%` | in_ch=32, out_ch=64 (100% 16-lane INT8 & 8-lane INT16 aligned) |
| 7 | `/features/features.6/Relu` | ReLU (in-place fused) | `[1, 64, 130]` | `[1, 64, 130]` | `—` | `0` | `0` | `0.0%` | Non-MAC / activation / pooling layer |
| 8 | `/features/features.7/MaxPool` | MaxPool1d (k=2) | `[1, 64, 130]` | `[1, 64, 65]` | `—` | `0` | `8,320` | `0.56%` | Non-MAC / activation / pooling layer |
| 9 | `/features/features.8/Conv` | DepthwiseConv1d (k=7, g=64) | `[1, 64, 65]` | `[1, 64, 65]` | `[64, 1, 7]` | `512` | `29,120` | `1.94%` | Vectorizes across 64 channels (multiple of 16: True) |
| 10 | `/features/features.9/Relu` | ReLU (in-place fused) | `[1, 64, 65]` | `[1, 64, 65]` | `—` | `0` | `0` | `0.0%` | Non-MAC / activation / pooling layer |
| 11 | `/features/features.10/Conv` | PointwiseConv1d (1x1) | `[1, 64, 65]` | `[1, 128, 65]` | `[128, 64, 1]` | `8,320` | `532,480` | `35.56%` | in_ch=64, out_ch=128 (100% 16-lane INT8 & 8-lane INT16 aligned) |
| 12 | `/features/features.11/Relu` | ReLU (in-place fused) | `[1, 128, 65]` | `[1, 128, 65]` | `—` | `0` | `0` | `0.0%` | Non-MAC / activation / pooling layer |
| 13 | `/features/features.12/MaxPool` | MaxPool1d (k=2) | `[1, 128, 65]` | `[1, 128, 32]` | `—` | `0` | `8,192` | `0.55%` | Non-MAC / activation / pooling layer |
| 14 | `/features/features.13/Conv` | DepthwiseConv1d (k=5, g=128) | `[1, 128, 32]` | `[1, 128, 32]` | `[128, 1, 5]` | `768` | `20,480` | `1.37%` | Vectorizes across 128 channels (multiple of 16: True) |
| 15 | `/features/features.14/Relu` | ReLU (in-place fused) | `[1, 128, 32]` | `[1, 128, 32]` | `—` | `0` | `0` | `0.0%` | Non-MAC / activation / pooling layer |
| 16 | `/features/features.15/Conv` | PointwiseConv1d (1x1) | `[1, 128, 32]` | `[1, 128, 32]` | `[128, 128, 1]` | `16,512` | `524,288` | `35.01%` | in_ch=128, out_ch=128 (100% 16-lane INT8 & 8-lane INT16 aligned) |
| 17 | `/features/features.16/Relu` | ReLU (in-place fused) | `[1, 128, 32]` | `[1, 128, 32]` | `—` | `0` | `0` | `0.0%` | Non-MAC / activation / pooling layer |
| 18 | `/avg_pool/AveragePool` | AveragePool1d (k=32) | `[1, 128, 32]` | `[1, 128, 1]` | `—` | `0` | `4,096` | `0.27%` | Non-MAC / activation / pooling layer |
| 19 | `/flatten/Flatten` | Flatten (zero-copy view) | `[1, 128, 1]` | `[1, 128]` | `—` | `0` | `0` | `0.0%` | Non-MAC / activation / pooling layer |
| 20 | `/classifier/Gemm` | Linear / Gemm | `[1, 128]` | `[1, 2]` | `[2, 128]` | `258` | `256` | `0.02%` | in_features=128 (16-aligned dot-product), out_features=2 |
| **Tổng** | **20 Nodes (8 Weighted)** | **DS-CNN (`G=1, D=2`)** | `(1,1,260)` | `(1,2)` | — | **`29,058`** | **`1,497,472`** | **`100.0%`** | **`94.18%` MACC căn lề 16-lane** |

### Nhận xét về Phân bố Tính toán trên ESP32-S3:
- **Tổng số phép tính:** `1,468,544` MACC nhân-cộng (Conv + Gemm, khớp `cost_report.json`) + `28,928` phép so sánh/cộng gộp (Pooling) = **`1,497,472`** tổng thao tác.
- **Pointwise Conv1d (`1x1`) chiếm `88.35%` tổng MACC (`1,323,008` MACC):** Toàn bộ 3 lớp Pointwise (`32->64`, `64->128`, `128->128`) đều có số kênh vào và ra là **bội số chẵn của 16** (`32, 64, 128`). Đây là dạng phép nhân ma trận dày đặc lý tưởng nhất cho tập lệnh vector PIE 128-bit của ESP32-S3 (không bị đệm kênh thừa).
- **Depthwise Conv1d chiếm `5.81%` tổng MACC (`87,040` MACC):** Vector hóa dọc theo chiều kênh (`32, 64, 128` kênh — đều chia hết cho 16).
- **Standard Conv1d đầu tiên (`1->32, k=7`) chỉ chiếm `3.89%` tổng MACC (`58,240` MACC):** Dù `in_channels = 1` không tận dụng được chiều sâu kênh đầu vào, tỷ trọng MACC của lớp này rất nhỏ (`3.89%`) nên hầu như không ảnh hưởng tới tổng thời gian suy luận.

---

## 3. Đánh giá Phân phối Trọng số (Weights) & Sai số Lượng tử hóa Lũy thừa 2 (`scale = 2^n`)

Khác với ONNXRuntime (dùng `scale` số thực bất kỳ theo từng kênh), **ESP-DL / ESP-PPQ** quy định bước lượng tử có dạng lũy thừa của 2 (`scale = 2^exponent`) để vi điều khiển dịch bit (`shift`) thay vì nhân số thực. Bảng dưới đây phân tích trực tiếp các tensor trọng số đã gộp BatchNorm trong `cnn1d_binary_dsB_kd_fp32.onnx`:

| Lớp Trọng số | Shape | `[Min, Max]` | `|W|_max` | `|W|_p99` | Tỷ lệ Outlier (`max/p99`) | INT8 Per-Tensor (`exp`, SQNR) | INT8 Per-Channel (SQNR, `exp` range) | INT16 Per-Tensor (`exp`, SQNR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `/features/features.0/Conv` (StandardConv1d (k=7)) | `[32, 1, 7]` | `[-3.579, 3.610]` | `3.6102` | `2.7614` | `1.31x` | `2^-5` (`38.82 dB`) | **`45.51 dB`** (`2^-9..2^-5`) | `2^-13` (**`88.13 dB`**) |
| `/features/features.3/Conv` (DepthwiseConv1d (k=9, g=32)) | `[32, 1, 9]` | `[-2.171, 1.856]` | `2.1712` | `1.7701` | `1.23x` | `2^-5` (`37.57 dB`) | **`44.35 dB`** (`2^-8..2^-5`) | `2^-13` (**`85.72 dB`**) |
| `/features/features.5/Conv` (PointwiseConv1d (1x1)) | `[64, 32, 1]` | `[-1.236, 1.169]` | `1.2359` | `0.9125` | `1.35x` | `2^-6` (`37.89 dB`) | **`43.23 dB`** (`2^-8..2^-6`) | `2^-14` (**`85.99 dB`**) |
| `/features/features.8/Conv` (DepthwiseConv1d (k=7, g=64)) | `[64, 1, 7]` | `[-2.888, 3.213]` | `3.2126` | `2.3878` | `1.35x` | `2^-5` (`38.45 dB`) | **`45.48 dB`** (`2^-8..2^-5`) | `2^-13` (**`86.35 dB`**) |
| `/features/features.10/Conv` (PointwiseConv1d (1x1)) | `[128, 64, 1]` | `[-0.710, 0.588]` | `0.7101` | `0.4180` | `1.7x` | `2^-7` (`37.34 dB`) | **`41.81 dB`** (`2^-9..2^-7`) | `2^-15` (**`85.44 dB`**) |
| `/features/features.13/Conv` (DepthwiseConv1d (k=5, g=128)) | `[128, 1, 5]` | `[-3.187, 2.990]` | `3.1866` | `2.4203` | `1.32x` | `2^-5` (`39.66 dB`) | **`46.07 dB`** (`2^-8..2^-5`) | `2^-13` (**`87.5 dB`**) |
| `/features/features.15/Conv` (PointwiseConv1d (1x1)) | `[128, 128, 1]` | `[-0.507, 0.485]` | `0.5068` | `0.3278` | `1.55x` | `2^-7` (`34.31 dB`) | **`41.07 dB`** (`2^-9..2^-7`) | `2^-15` (**`82.43 dB`**) |
| `/classifier/Gemm` (Linear / Gemm) | `[2, 128]` | `[-0.135, 0.136]` | `0.1361` | `0.1343` | `1.01x` | `2^-9` (`42.7 dB`) | **`42.7 dB`** (`2^-9..2^-9`) | `2^-17` (**`91.08 dB`**) |

### Nhận xét về Trọng số Mô hình khi đưa xuống ESP32-S3:
1. **Không có hiện tượng bùng nổ trọng số (Weight Explosion) sau khi gộp BatchNorm:** Giá trị `|W|_max` của cả 8 lớp đều nằm trong khoảng gọn `0.1361 – 3.6102`. Tỷ lệ `max / p99` chỉ từ `1.01x – 1.70x` (rất ít outlier).
2. **Độ lợi của Per-Channel Quantization:** Lượng tử hóa **Per-Channel INT8** theo lũy thừa 2 nâng SQNR từ `34.31 – 39.66 dB` (Per-Tensor) lên **`41.07 – 46.07 dB`** (cao hơn ~5–7 dB trên toàn bộ các lớp Conv).
3. **Lớp đầu (`features.0/Conv`) và lớp cuối (`classifier/Gemm`):** Chỉ có `256` và `258` tham số (gồm bias). Nếu áp dụng cấu hình **Mixed Precision (C3: đặt lớp đầu và lớp cuối ở INT16)**, SQNR trọng số của hai lớp này đạt tới **`88.13 dB` và `91.08 dB`** mà dung lượng Flash chỉ tăng thêm chưa tới `1 KB`.

---

## 4. Phân tích Độ Nhạy Biên độ Sóng ECG ở Đầu vào (`calib_train.npy`)

Dữ liệu nhịp ECG đầu vào `(1, 1, 260)` đã chuẩn hóa z-score theo từng nhịp (`mean = 0, std = 1`), có dải giá trị thực tế từ **`-4.44358` đến `+7.08285`**:
- Phân vị biên độ tuyệt đối `|x|`: Trung vị `p50 = 0.3819`, `p90 = 1.3929`, `p99 = 3.943`, `p99.9 = 5.9523`, `max = 7.0828`.
- **Đánh đổi cốt lõi khi lượng tử hóa đầu vào INT8 theo lũy thừa 2 (`scale = 2^exponent`):**

| Exponent đầu vào (`INT8`) | Bước lượng tử (`scale = 2^exp`) | Ngưỡng bão hòa (`±127 * scale`) | Số mức rời rạc cho sóng P/T nhỏ (`0.30`) | % Mẫu bị cắt ngọn (Clipped Samples) | % Nhịp có đỉnh R bị chạm ngưỡng | SQNR Tín hiệu Đầu vào |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`2^-3`** | `0.12500` | `±15.875` | **`2 mức`** | `0.0%` | `0.0%` | `28.85 dB` |
| **`2^-4`** | `0.06250` | `±7.9375` | **`5 mức`** | `0.0%` | `0.0%` | `34.89 dB` |
| **`2^-5`** | `0.03125` | `±3.9688` | **`10 mức`** | `0.9795%` | `52.17%` | `19.06 dB` |
| **`2^-6`** | `0.01562` | `±1.9844` | **`19 mức`** | `6.5545%` | `99.67%` | `9.0 dB` |
| **`INT16` (`2^-12`)** | `0.000244` | `±8.0` | **`1229 mức`** | `0.0%` | `0.0%` | **`83.04 dB`** |

- **Ý nghĩa kỹ thuật:**
  - Với `exponent = -4` (`scale = 0.0625`), không có nhịp nào bị cắt đỉnh QRS (`0%` clipped), nhưng sóng P/T biên độ nhỏ (`~0.30`) chỉ được biểu diễn bằng **5 mức nguyên** (`0, 1, 2, 3, 4, 5`).
  - Với `exponent = -5` (`scale = 0.03125`), độ phân giải sóng P/T tăng gấp đôi (**10 mức**), nhưng **`51.67%` số nhịp bị cắt đỉnh R** ở ngưỡng `±3.97` (tuy nhiên chỉ chiếm `1.04%` tổng số mẫu toàn chuỗi do đỉnh R rất hẹp).
  - Đây là lý do ở **Giai đoạn B & C**, chúng ta cần so sánh thực nghiệm giữa `C0/C1` (calib tự động), `C2` (`exp = -4` vs `-5`), `C3` (**Mixed Precision: lớp đầu INT16** giúp giữ nguyên **1,229 mức** cho sóng P/T và `0%` cắt đỉnh!), và `B` (INT16 toàn mạng).

---

## 5. Ngân sách Bộ nhớ & Ước tính Thời gian Suy luận trên ESP32-S3 (N16R8)

| Hạng mục | Cấu hình A: `FP32` (C thuần) | Cấu hình B: `INT16` (ESP-DL PIE) | Cấu hình C: `INT8` (ESP-DL PIE) | Giới hạn Phần cứng ESP32-S3 N16R8 |
| :--- | :---: | :---: | :---: | :---: |
| **Trọng số trong Flash** (`[Đã tính từ ONNX]`) | `113.51 KB` | `59.38 KB` | **`32.57 KB`** | `16,384 KB` (16 MB Flash) |
| **Đỉnh Activation Ping-Pong** (`[Đã tính từ ONNX]`) | `48.75 KB` | `24.38 KB` | **`12.19 KB`** | `512 KB` SRAM nội (`~320 KB` trống) |
| **Tổng Arena + Buffer I/O** (`[Đã tính từ ONNX]`) | `50.77 KB` | `25.88 KB` | **`13.44 KB`** | Không cần dùng tới 8 MB PSRAM |
| **Tập lệnh Thực thi trên Chip** | FPU Vô hướng (`1 MAC/lệnh`) | PIE SIMD 128-bit (`8 lanes INT16`) | **PIE SIMD 128-bit (`16 lanes INT8`)** | Dual-core Xtensa LX7 @ 240 MHz |
| **Chu kỳ CPU ước tính** (`[Ước tính lý thuyết]`) | `2,994,944 – 5,989,888` | `623,946 – 1,247,893` | **`311,973 – 623,946`** | 240,000 chu kỳ / ms |
| **Độ trễ suy luận / nhịp** (`[Ước tính lý thuyết]`) | **`12.48 – 24.96 ms`** | **`2.6 – 5.2 ms`** | **`1.3 – 2.6 ms`** | Mục tiêu Gate A: `< 10 ms` (**PASS**) |

---

## 6. Kiểm chứng Tương thích Toán tử với `esp-ppq 1.3.14` (`[Đã kiểm chứng]`)

- **Trạng thái nạp đồ thị 1D trực tiếp vào `esp-ppq`:** `True`
- **Các toán tử `esp-ppq` nhận diện thành công từ file ONNX:** `['AveragePool', 'Conv', 'Flatten', 'Gemm', 'MaxPool', 'Relu']`
- **Các nền tảng đích (`TargetPlatform`) hỗ trợ cho ESP32-S3:** `['ESPDL_INT8', 'ESPDL_INT16', 'ESPDL_S3_INT8', 'ESPDL_S3_INT16', 'ESPDL_H_PRE_INT16', 'ESPDL_S3_H_PRE_INT16', 'ESPDL_C_INT8', 'ESPDL_C_INT16', 'ESPDL_C_H_PRE_INT16', 'ESPDL_W8A16', 'ESPDL_S3_W8A16', 'ESPDL_C_W8A16']`
- **Ghi chú:** esp-ppq 1.3.14 parses all 6 ONNX 1D operators (Conv, Relu, MaxPool, AveragePool, Flatten, Gemm) directly into its internal IR graph without error.
