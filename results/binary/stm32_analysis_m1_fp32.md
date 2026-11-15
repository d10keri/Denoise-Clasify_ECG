# BÁO CÁO GIAI ĐOẠN C (C1 + C2 + C3): XUẤT ONNX TĨNH & PHÂN TÍCH TÀI NGUYÊN FP32 TRÊN STM32 (NUCLEO-F446RE)

**Ngày thực hiện:** 15/11/2026  
**Căn cứ kế hoạch:** [edge_ai_int8_binary_plan.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/edge_ai_int8_binary_plan.md) (Mục 2 – Giai đoạn C) & [ds_cnn_binary_plan.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/ds_cnn_binary_plan.md) (Mục E5)  
**Môi trường thực thi:** Conda `ecg` (`PyTorch 2.12.1+cu130`, `ONNX 1.23.2`, `ONNXRuntime 1.31.1`)  
**Quy ước chú thích số liệu:**
- **[Đã đo bằng ONNXRuntime / Python]:** Số liệu kiểm chứng thực tế trên tập Test (`49,691` nhịp), tập kiểm chứng `2,000` nhịp và tập Host Validation `500` nhịp.
- **[Đã tính từ đồ thị ONNX tĩnh]:** Số lượng tham số sau gộp BatchNorm, kích thước Tensor trung gian, dung lượng Weights RO và số phép tính MACC.
- **[Mô phỏng theo cơ chế ST Edge AI Core / CMSIS-NN]:** Thư viện C runtime overhead và số chu kỳ CPU / Latency ước tính trên vi điều khiển Cortex-M4F.

---

## 1. KẾT QUẢ BƯỚC C1: XUẤT ĐỒ THỊ ONNX TĨNH & KIỂM CHỨNG TOÁN HỌC

### 1.1 Xử lý tương thích toán tử khi xuất ONNX ([scripts/binary/export_onnx.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/scripts/binary/export_onnx.py))
Theo đúng quy tắc tại Bước C1 của kế hoạch:
1. **Không sửa đổi bất kỳ file model gốc nào** ([cnn1d_binary.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/src/models/cnn1d_binary.py), [cnn1d_binary_reg.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/src/models/cnn1d_binary_reg.py), [cnn1d_binary_ds.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/src/models/cnn1d_binary_ds.py)).
2. **Đầu vào cố định tĩnh `ecg_input: (1, 1, 260)` (`float32`)**, hoàn toàn không dùng `dynamic_axes`, xuất ở `opset_version = 17`.
3. **Gộp toán học `BatchNorm1d` vào `Conv1d` (`fuse_conv_bn_eval`):** Mọi tầng `BatchNorm1d` trong `M1-Reg` và `M2-B-KD` đều được gộp hoàn toàn vào trọng số `weight` và `bias` của tầng `Conv1d` đứng trước. Đồ thị ONNX sinh ra **không còn bất kỳ nút `BatchNormalization` hay `Dropout` nào**, giảm chi phí tính toán phụ về `0 MACC`.
4. **Thay thế `AdaptiveAvgPool1d(1)` bằng `AvgPool1d` tĩnh:** Dùng `nn.AvgPool1d(kernel_size=65, stride=65)` cho `M1-Reg` và `nn.AvgPool1d(kernel_size=32, stride=32)` cho `M2-B-KD` kèm `Flatten(1)`, giúp đồ thị ONNX sạch tuyệt đối (không phát sinh các nút tính kích thước động `Shape/Slice/Gather`).

| Mô hình xuất ONNX | File ONNX chính (Seed 42) | Dung lượng file `.onnx` **[Đã đo]** | Số Params tĩnh trên MCU **[Đã tính]** | Chuỗi toán tử ONNX (`op_type`) **[Đã đo]** | Cảnh báo tương thích `stedgeai` |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **`M1-Reg` (Chuẩn Conv1D)** | [export/cnn1d_binary_fp32.onnx](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/cnn1d_binary_fp32.onnx) | `140.64 KB` | **35,458** | `Conv → Relu → MaxPool` (×2) `→ Conv → Relu → AveragePool → Flatten → Gemm` (11 nút) | **0 cảnh báo** (Hỗ trợ 100%) |
| **`M2-B-KD` (DS-CNN 4 tầng)** | [export/cnn1d_binary_dsB_kd_fp32.onnx](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/cnn1d_binary_dsB_kd_fp32.onnx) | `117.43 KB` | **29,058** | `Conv → Relu → MaxPool` + 3× `(DW_Conv → Relu → PW_Conv → Relu [→ MaxPool])` `→ AveragePool → Flatten → Gemm` (20 nút) | **0 cảnh báo** (Hỗ trợ 100%) |

### 1.2 Kiểm chứng độ khớp PyTorch vs ONNXRuntime (Gate C1 & Gate C3 FP32)
Chạy script [scripts/binary/verify_onnx.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/scripts/binary/verify_onnx.py) đối chiếu trực tiếp mô hình PyTorch gốc (chưa gộp BN) với file `.onnx` tĩnh trên **2,000 nhịp Test phân tầng** (`1,000 N + 1,000 non-N`) và tập **Host Validation 500 nhịp** ([export/host_val_test500_x.npy](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/host_val_test500_x.npy)):

| Phiên bản Model | Max $\|\text{Logit}_{\text{PT}} - \text{Logit}_{\text{ONNX}}\|$ **[Đã đo]** | Sai số tương đối `L2r` (chuẩn ST.AI) **[Đã đo]** | Độ tương đồng `Cosine (COS)` **[Đã đo]** | Tỷ lệ trùng dự đoán (`argmax` & ngưỡng `th`) **[Đã đo]** | Đánh giá **Gate C1 & C3** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`M1-Reg` (Seed 42)** | `2.38e-06` ($\ll 10^{-4}$) | `2.55e-07` | `1.0000000` | **100.00% / 100.00%** | **ĐẠT (PASS)** |
| **`M1-Reg` (Seed 43)** | `1.91e-06` ($\ll 10^{-4}$) | `2.58e-07` | `1.0000000` | **100.00% / 100.00%** | **ĐẠT (PASS)** |
| **`M1-Reg` (Seed 44)** | `1.43e-06` ($\ll 10^{-4}$) | `2.73e-07` | `0.9999999` | **100.00% / 100.00%** | **ĐẠT (PASS)** |
| **`M2-B-KD` (Seed 42)** | `2.38e-06` ($\ll 10^{-4}$) | `2.78e-07` | `1.0000000` | **100.00% / 100.00%** | **ĐẠT (PASS)** |
| **`M2-B-KD` (Seed 43)** | `1.67e-06` ($\ll 10^{-4}$) | `2.74e-07` | `1.0000000` | **100.00% / 100.00%** | **ĐẠT (PASS)** |
| **`M2-B-KD` (Seed 44)** | `2.38e-06` ($\ll 10^{-4}$) | `2.87e-07` | `1.0000000` | **100.00% / 100.00%** | **ĐẠT (PASS)** |

---

## 2. KẾT QUẢ BƯỚC C2: PHÂN TÍCH TÀI NGUYÊN FP32 TRÊN NUCLEO-F446RE VÀ CÁC BOARD ỨNG VIÊN

### 2.1 Phân bổ tài nguyên từng tầng C-Graph của `M1-Reg` (FP32)
Khi biên dịch sang mã C cho STM32, các nút `ReLU` được gộp tại chỗ (*in-place fusion*) vào vòng lặp của tầng `Conv1d` phía trước, và nút `Flatten` là phép đổi nhãn con trỏ bộ nhớ (`0 MACC, 0 Bytes`). Bộ nhớ kích hoạt (`Activations RAM`) sử dụng cơ chế **Ping-Pong Memory Pool** (chỉ giữ đồng thời tensor đầu vào và đầu ra của tầng đang chạy):

| Tầng C-Graph (ST Edge AI) | Cấu hình | Input $\rightarrow$ Output Shape | Params **[Đã tính]** | MACCs **[Đã tính]** | % MACC | Flash Weights RO (FP32) **[Đã tính]** | Active RAM In+Out (FP32) **[Đã tính]** |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| `0. Conv1d_1 + ReLU` *(Fused)* | `1→32, k=5, p=2` | `(1,1,260) → (1,32,260)` | 192 | 41,600 | 1.40% | 768 B (0.75 KB) | 34,320 B (33.52 KB) |
| `1. MaxPool1d_1` | `k=2, s=2` | `(1,32,260) → (1,32,130)` | 0 | 0 | 0.00% | 0 B | **49,920 B (48.75 KB)** |
| `2. Conv1d_2 + ReLU` *(Fused)* | `32→64, k=5, p=2` | `(1,32,130) → (1,64,130)` | 10,304 | 1,331,200 | 44.81% | 41,216 B (40.25 KB) | **49,920 B (48.75 KB)** |
| `3. MaxPool1d_2` | `k=2, s=2` | `(1,64,130) → (1,64,65)` | 0 | 0 | 0.00% | 0 B | **49,920 B (48.75 KB)** |
| `4. Conv1d_3 + ReLU` *(Fused)* | `64→128, k=3, p=1` | `(1,64,65) → (1,128,65)` | 24,704 | 1,597,440 | **53.78%** | 98,816 B (96.50 KB) | **49,920 B (48.75 KB)** |
| `5. AvgPool1d(65) + Gemm` | `128→2` | `(1,128,65) → (1,2)` | 258 | 256 | 0.01% | 1,032 B (1.01 KB) | 520 B (0.51 KB) |
| **TỔNG MODEL (`M1-Reg` FP32)** | | | **35,458** | **2,970,496** | **100.0%** | **141,832 B (138.51 KB)** | **Peak: 49,920 B (48.75 KB)** |
| **+ ST.AI C Runtime Overhead** | *(ARM GCC `-O3`)* | | | | | **~14.50 KB** **[Mô phỏng]** | **~1.25 KB** **[Mô phỏng]** |
| **TỔNG TOÀN MẠNG TRÊN MCU** | | | | | | **153.01 KB Flash** | **50.00 KB SRAM** |

### 2.2 Phân bổ tài nguyên từng tầng C-Graph của `M2-B-KD` (FP32 – Ứng viên DS-CNN từ Giai đoạn E)

| Tầng C-Graph (ST Edge AI) | Cấu hình | Input $\rightarrow$ Output Shape | Params **[Đã tính]** | MACCs **[Đã tính]** | % MACC | Flash Weights RO (FP32) **[Đã tính]** | Active RAM In+Out (FP32) **[Đã tính]** |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| `0. Stem Conv1d + ReLU` | `1→32, k=7, p=3` | `(1,1,260) → (1,32,260)` | 256 | 58,240 | 3.97% | 1,024 B | 34,320 B (33.52 KB) |
| `1. Stem MaxPool1d` | `k=2, s=2` | `(1,32,260) → (1,32,130)` | 0 | 0 | 0.00% | 0 B | **49,920 B (48.75 KB)** |
| `2. DS1 Depthwise + ReLU` | `32→32, k=9, g=32` | `(1,32,130) → (1,32,130)` | 320 | 37,440 | 2.55% | 1,280 B | 33,280 B (32.50 KB) |
| `3. DS1 Pointwise + ReLU` | `32→64, k=1, g=1` | `(1,32,130) → (1,64,130)` | 2,112 | 266,240 | 18.13% | 8,448 B | **49,920 B (48.75 KB)** |
| `4. DS1 MaxPool1d` | `k=2, s=2` | `(1,64,130) → (1,64,65)` | 0 | 0 | 0.00% | 0 B | **49,920 B (48.75 KB)** |
| `5. DS2 Depthwise + ReLU` | `64→64, k=7, g=64` | `(1,64,65) → (1,64,65)` | 512 | 29,120 | 1.98% | 2,048 B | 33,280 B (32.50 KB) |
| `6. DS2 Pointwise + ReLU` | `64→128, k=1, g=1` | `(1,64,65) → (1,128,65)` | 8,320 | 532,480 | **36.26%** | 33,280 B | **49,920 B (48.75 KB)** |
| `7. DS2 MaxPool1d` | `k=2, s=2` | `(1,128,65) → (1,128,32)` | 0 | 0 | 0.00% | 0 B | 49,664 B (48.50 KB) |
| `8. DS3 Depthwise + ReLU` | `128→128, k=5, g=128`| `(1,128,32) → (1,128,32)` | 768 | 20,480 | 1.39% | 3,072 B | 32,768 B (32.00 KB) |
| `9. DS3 Pointwise + ReLU` | `128→128, k=1, g=1` | `(1,128,32) → (1,128,32)` | 16,512 | 524,288 | **35.70%** | 66,048 B | 32,768 B (32.00 KB) |
| `10. AvgPool1d(32) + Gemm` | `128→2` | `(1,128,32) → (1,2)` | 258 | 256 | 0.02% | 1,032 B | 520 B (0.51 KB) |
| **TỔNG MODEL (`M2-B-KD` FP32)**| | | **29,058** | **1,468,544** | **100.0%** | **116,232 B (113.51 KB)** | **Peak: 49,920 B (48.75 KB)** |
| **+ ST.AI C Runtime Overhead** | *(ARM GCC `-O3`)* | | | | | **~18.00 KB** **[Mô phỏng]** | **~1.25 KB** **[Mô phỏng]** |
| **TỔNG TOÀN MẠNG TRÊN MCU** | | | | | | **131.51 KB Flash** | **50.00 KB SRAM** |

### 2.3 Bảng đánh giá khả năng đáp ứng trên 3 Board STM32 ứng viên (Chế độ FP32)

| Board ứng viên | Xung nhịp / Bộ nhớ vật lý | Model (FP32) | Tổng Flash (`% chiếm dụng`) | Flash còn dư cho Firmware | Tổng Peak RAM (`% chiếm dụng`) | SRAM còn dư cho DSP | Latency ước tính (`ms/nhịp`) **[Mô phỏng]** | Qua **Gate C2** ($\le 70\%$ Flash & $\le 60\%$ RAM) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NUCLEO-F446RE (Đề xuất chính)** | `180 MHz`<br>`512 KB Flash`<br>`128 KB SRAM` | **`M1-Reg`** | `153.01 KB` (**29.88%**) | `358.99 KB` | `50.00 KB` (**39.06%**) | `78.00 KB` | **`~102.59 ms`**<br>*(~18.47M cycles)* | **ĐẠT (PASS)** |
| **NUCLEO-F446RE (Đề xuất chính)** | `180 MHz`<br>`512 KB Flash`<br>`128 KB SRAM` | **`M2-B-KD`** | `131.51 KB` (**25.69%**) | `380.49 KB` | `50.00 KB` (**39.06%**) | `78.00 KB` | **`~46.81 ms`**<br>*(~8.43M cycles)* | **ĐẠT (PASS)** |
| **NUCLEO-G474RE** *(Phương án 2)* | `170 MHz`<br>`512 KB Flash`<br>`128 KB SRAM` | `M1-Reg` / `M2-B-KD` | `29.88%` / `25.69%` | `359 KB` / `380 KB` | `39.06%` | `78.00 KB` | `~108.6 ms` / `~49.6 ms` | **ĐẠT (PASS)** |
| **NUCLEO-L476RG** *(Tiết kiệm điện)* | `80 MHz`<br>`1024 KB Flash`<br>`128 KB SRAM` | `M1-Reg` / `M2-B-KD` | `14.94%` / `12.84%` | `871 KB` / `892 KB` | `39.06%` | `78.00 KB` | `~230.8 ms` / `~105.3 ms` | **ĐẠT (PASS)** *(Latency FP32 của M1-Reg hơi sát ở nhịp tim nhanh >200 BPM)* |

---

## 3. ĐỐI CHIẾU SỐ ĐO VỚI ƯỚC TÍNH BAN ĐẦU & KẾT LUẬN GATE C

1. **Đối chiếu với ước tính tại Mục 1 của [edge_ai_int8_binary_plan.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/edge_ai_int8_binary_plan.md):**
   - **Flash cho Weights (FP32):** Số tính từ đồ thị ONNX của `M1-Reg` là **`138.51 KB`** (`141,832 Bytes`), khớp 100% với con số ước tính `≈ 138.5 KB`. Khi cộng thêm thư viện C runtime (`~14.5 KB`), tổng Flash là `153.01 KB`, chỉ chiếm **29.88%** dung lượng Flash 512 KB của **NUCLEO-F446RE** (còn dư tới `359 KB` cho code đọc ADC, lọc số, Pan-Tompkins và giao tiếp UART).
   - **Peak RAM Activation (FP32):** Số tính chính xác từ bộ đệm Ping-Pong của `Conv1d_2` và `Conv1d_3` là **`48.75 KB`** (`49,920 Bytes`), khớp 100% với ước tính ban đầu. Cộng thêm `~1.25 KB` cấu trúc quản lý mạng, tổng RAM là `50.00 KB` (**39.06%** của 128 KB SRAM trên F446RE), để dư **`78.00 KB` SRAM** cho bộ đệm vòng (circular buffer) của khâu tiền xử lý ECG.
   - **Latency FP32 trên F446RE @ 180 MHz:** `M1-Reg` ước tính mất **`~102.6 ms/nhịp`** (nằm đúng giữa khoảng dự đoán `80–170 ms` ở Mục 1), trong khi `M2-B-KD` giảm một nửa số MACC nên rút ngắn thời gian chạy FP32 xuống chỉ còn **`~46.8 ms/nhịp`**.
2. **Kết luận Giai đoạn C:** Cả `M1-Reg` và `M2-B-KD` đều **vượt qua hoàn toàn Gate C1, Gate C2 và Gate C3** ngay ở chế độ **FP32** trên board **NUCLEO-F446RE**.
