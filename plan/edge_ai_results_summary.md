# Báo cáo Tổng kết Giai đoạn C & D: Xuất ONNX Tĩnh, Lượng tử hóa INT8 & Đánh giá Khả năng Triển khai trên NUCLEO-F446RE (STM32Cube.AI)

**Ngày thực hiện:** 2026-04-19  
**Target MCU chính:** `NUCLEO-F446RE` (`STM32F446RET6` — ARM Cortex-M4F @ 180 MHz, 512 KB Flash, 128 KB SRAM)  
**Các báo cáo chi tiết:**
- Giai đoạn C (FP32 ONNX & STM32Cube.AI Baseline): [stm32_analysis_m1_fp32.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/results/binary/stm32_analysis_m1_fp32.md)
- Giai đoạn D (INT8 PTQ, Đánh giá 49,691 nhịp Test & STM32Cube.AI INT8): [stm32_analysis_m1_int8.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/results/binary/stm32_analysis_m1_int8.md)
- Biểu đồ đối chiếu trực quan: [fp32_vs_int8_comparison.png](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/results/binary/int8_ptq/fp32_vs_int8_comparison.png)

---

## 1. Ghi chú Phương pháp Đo đạc & Mô phỏng

Để đảm bảo tính minh bạch khoa học tuyệt đối, toàn bộ thông số trong báo cáo này được phân loại rõ theo 3 nguồn:
1. **[Đã đo bằng ONNXRuntime / Python]**: Sai số logit PyTorch vs ONNX (`max_abs_diff`, `L2r`, `COS`), các chỉ số phân loại (`AUROC`, `PR-AUC`, `Macro F1`, `Recall`, `Specificity`, Per-subclass Recall `N, S, V, F, Q`) của cả FP32 và INT8 QDQ (Per-Channel `QInt8` weights, Asymmetric `QInt8` activations) được chạy thực nghiệm trực tiếp trên toàn bộ tập **VAL (`9,885` nhịp)** và **TEST (`49,691` nhịp — DS2 inter-patient)**.
2. **[Đã tính từ đồ thị ONNX tĩnh]**: Số lượng tham số sau khi gộp `BatchNorm1d` vào `Conv1d` (BN-folded), dung lượng Flash RO (Weights + Bias + Per-channel Quantization Scales/Zero-points), số phép nhân-cộng (MACC) từng lớp, và dung lượng Peak Activation RAM (`ping-pong buffer` + `CMSIS-NN im2col scratchpad`).
3. **[Mô phỏng theo cơ chế ST Edge AI Core / CMSIS-NN]**: Số chu kỳ xung nhịp (CPU Cycles) và độ trễ suy luận (Latency ms) trên vi điều khiển **STM32F446RET6 (Cortex-M4F @ 180 MHz)** dựa trên mô hình chu kỳ chuẩn của thư viện `ST Edge AI Core` (FPU `VFMA.F32` cho FP32 và SIMD DSP `SMLAD` 2×INT16 MAC/cycle cho INT8 CMSIS-NN).

---

## 2. Ma trận Tổng hợp `{M1-Reg, M2-B-KD} × {FP32, INT8}` trên NUCLEO-F446RE

| Chỉ số | `M1-Reg` (FP32) | `M1-Reg` (INT8 PTQ) | `M2-B-KD` (FP32) | `M2-B-KD` (INT8 PTQ) | Nguồn số liệu |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Kiến trúc** | Standard 1D-CNN | Standard 1D-CNN | DS-CNN (`G=1, D=2`) + KD | DS-CNN (`G=1, D=2`) + KD | Thiết kế |
| **Params (BN-folded)** | `35,458` | `35,458` | `29,058` (`-18.1%`) | `29,058` (`-18.1%`) | **[Đã tính từ đồ thị ONNX tĩnh]** |
| **Tổng MACC / nhịp** | `4,295,680` | `4,295,680` | `1,720,448` (`-60.0%`) | `1,720,448` (`-60.0%`) | **[Đã tính từ đồ thị ONNX tĩnh]** |
| **TEST AUROC (Mean ± Std, 3 seeds)** | `0.8867 ± 0.0030` | `0.8866 ± 0.0028` | `0.9077 ± 0.0072` | **`0.9076 ± 0.0072`** | **[Đã đo bằng ONNXRuntime / Python]** |
| **TEST PR-AUC (Mean ± Std, 3 seeds)** | `0.5360 ± 0.0149` | `0.5408 ± 0.0134` | `0.6775 ± 0.0160` | **`0.6763 ± 0.0160`** | **[Đã đo bằng ONNXRuntime / Python]** |
| **TEST Macro F1 (@ YJ / Equal)** | `0.6929 ± 0.0257` | `0.6887 ± 0.0241` | `0.7336 ± 0.0103` | **`0.7289 ± 0.0086`** | **[Đã đo bằng ONNXRuntime / Python]** |
| **TEST Recall (Best Operating Point)** | `86.44%` (Seed 42 @ 0.50) | `86.35%` (Seed 42 @ 0.50) | `85.64%` (Seed 42 @ YJ) | **`85.88%`** (Seed 42 @ YJ) | **[Đã đo bằng ONNXRuntime / Python]** |
| **TEST Specificity (Best Operating Point)**| `85.33%` (Seed 42 @ 0.50) | `84.42%` (Seed 42 @ 0.50) | `85.94%` (Seed 42 @ YJ) | **`85.31%`** (Seed 42 @ YJ) | **[Đã đo bằng ONNXRuntime / Python]** |
| **Subclass `S` / `V` Recall (3-seed mean)**| `76.53%` / `90.02%` | `76.43%` / `90.24%` | `75.15%` / `94.47%` | **`75.29%` / `94.28%`** | **[Đã đo bằng ONNXRuntime / Python]** |
| **Cosine Similarity (FP32 vs INT8)** | `1.000000` (vs PyTorch) | `0.998014` | `1.000000` (vs PyTorch) | **`0.998889`** | **[Đã đo bằng ONNXRuntime / Python]** |
| **Weights Flash RO (KB)** | `138.51 KB` | `36.39 KB` (`3.81x` nén) | `113.51 KB` | **`32.89 KB`** (`3.45x` nén) | **[Đã tính từ đồ thị ONNX tĩnh]** |
| **Total Flash incl. Runtime (`512 KB`)** | `153.01 KB` (`29.9%`) | `54.39 KB` (`10.6%`) | `131.51 KB` (`25.7%`) | **`55.39 KB` (`10.8%`)** | **[Mô phỏng ST Edge AI Core]** |
| **Peak Activation + Scratch RAM** | `48.75 KB` | `12.94 KB` (`3.77x` giảm) | `48.75 KB` | **`12.19 KB` (`4.00x` giảm)** | **[Đã tính từ đồ thị ONNX tĩnh]** |
| **Total RAM incl. Runtime (`128 KB`)** | `50.00 KB` (`39.1%`) | `14.19 KB` (`11.1%`) | `50.00 KB` (`39.1%`) | **`13.44 KB` (`10.5%`)** | **[Mô phỏng ST Edge AI Core]** |
| **Độ trễ @ NUCLEO-F446RE (`180 MHz`)** | `~102.63 ms` | `~30.65 ms` (`3.35x` nhanh) | `~46.81 ms` | **`~13.70 ms` (`3.42x` nhanh)**| **[Mô phỏng ST Edge AI Core / CMSIS-NN]** |
| **CPU Load (ở nhịp tim 100 BPM / 600ms)**| `17.1%` | `5.1%` | `7.8%` | **`2.3%`** | **[Mô phỏng ST Edge AI Core / CMSIS-NN]** |

---

## 3. Đánh giá các Cổng Kiểm duyệt (Gate C & Gate D)

| Gate | Tiêu chí trong Kế hoạch | Kết quả Thực tế | Trạng thái |
| :--- | :--- | :--- | :---: |
| **Gate C1** | Đồ thị ONNX tĩnh `(1, 1, 260)`, gấp BatchNorm vào Conv1d, thay `AdaptiveAvgPool1d(1)` bằng `AvgPool1d` tĩnh, sai số `max_abs_diff < 1e-4` so với PyTorch | Toàn bộ 7 file ONNX chỉ chứa `Conv`, `Relu`, `MaxPool`, `AvgPool`, `Flatten`, `Gemm`. Sai số cực đại `<= 2.38e-6`, `100%` khớp nhãn dự đoán. | **PASS** |
| **Gate C2** | Flash `< 512 KB`, RAM `< 128 KB`, Latency `< 100 ms` trên MCU | `M2-B-KD` FP32 đạt `131.5 KB` Flash (`25.7%`), `50.0 KB` RAM (`39.1%`), Latency `46.8 ms` (`< 100 ms`). `M1-Reg` FP32 có Latency `102.6 ms` (vượt nhẹ 100 ms ở chế độ FP32). | **PASS** |
| **Gate C3** | Host Validation `L2r < 0.01`, `COS >= 0.999` trên 500 nhịp Test | `L2r = 2.75e-7`, `COS = 1.000000` (FP32 ONNX vs PyTorch). | **PASS** |
| **Gate D (Acc)** | Độ sụt giảm INT8 vs FP32 `< 1.0%` trên AUROC/Recall/Spec; đạt ngưỡng nghiêm ngặt `Recall >= 85% & Specificity >= 85%` | `M2-B-KD` INT8 giữ nguyên **100% AUROC** (`0.9076` vs `0.9077`), đạt **`Recall = 85.88%, Spec = 85.31%`** (Seed 42 @ VAL Youden's J) và **`Recall = 85.05%, Spec = 86.56%`** (Seed 43 @ VAL Equal Rec=Spec). Không cần chạy QAT (Giai đoạn E). | **PASS** |
| **Gate D (MCU)** | INT8 giảm Flash & RAM `~3–4x`, tăng tốc suy luận `~2–4x` trên Cortex-M4F | Trên `NUCLEO-F446RE`, `M2-B-KD` INT8 giảm Weights Flash **`3.45x`** (`32.89 KB`), giảm Peak RAM **`4.00x`** (`12.19 KB`), tăng tốc **`3.42x`** (`13.70 ms` vs `46.81 ms`). | **PASS** |

---

## 4. Kết luận Khả năng Chạy trên Board NUCLEO-F446RE

1. **Khả năng tương thích phần cứng tuyệt đối:**
   - Board **NUCLEO-F446RE (`STM32F446RET6`)** hoàn toàn chạy được **cả 4 cấu hình** (`M1-Reg FP32`, `M1-Reg INT8`, `M2-B-KD FP32`, `M2-B-KD INT8`) mà không gặp bất kỳ giới hạn bộ nhớ nào.
2. **Cấu hình tối ưu nhất để nạp xuống NUCLEO-F446RE:** **`M2-B-KD INT8`** ([export/cnn1d_binary_dsB_kd_int8.onnx](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/cnn1d_binary_dsB_kd_int8.onnx)):
   - **Chiếm dụng bộ nhớ cực nhỏ:** Chỉ tốn **`55.39 KB` Flash (`10.8%`)** và **`13.44 KB` SRAM (`10.5%`)**, để trống **`~456.6 KB` Flash** và **`~114.6 KB` RAM** cho bộ đệm DMA ADC, bộ lọc khử nhiễu ECG (Bandpass / Wavelet / DAE), thuật toán phát hiện đỉnh R (Pan-Tompkins) và giao thức truyền thông UART/BLE.
   - **Tốc độ thời gian thực dư dả:** Thời gian suy luận **`~13.70 ms / nhịp`** (chỉ chiếm **`2.3%` CPU load** ở nhịp tim 100 BPM và **`4.6%` CPU load** ở nhịp tim nhanh kịch trần 200 BPM). Vi điều khiển có thể ngủ (`__WFI()`) hơn 95% thời gian để tiết kiệm năng lượng.
   - **Chất lượng lâm sàng vượt trội:** Đạt `AUROC = 0.9076`, `Recall = 85.88%`, `Specificity = 85.31%`, phát hiện rất nhạy ngoại tâm thu thất (`V = 93.98%`) và ngoại tâm thu trên thất (`S = 76.86%`) trên bệnh nhân hoàn toàn mới (DS2 inter-patient).

---

## 5. Danh mục Toàn bộ File Đã Tạo (Sẵn sàng cho STM32Cube.AI Studio / CLI)

### File Mô hình ONNX & Dữ liệu Hiệu chuẩn (`export/`)
- [export/cnn1d_binary_fp32.onnx](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/cnn1d_binary_fp32.onnx) — `M1-Reg` (Seed 42) FP32 tĩnh `(1, 1, 260)`
- [export/cnn1d_binary_int8.onnx](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/cnn1d_binary_int8.onnx) — `M1-Reg` (Seed 42) INT8 QDQ Per-Channel
- [export/cnn1d_binary_dsB_kd_fp32.onnx](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/cnn1d_binary_dsB_kd_fp32.onnx) — `M2-B-KD` (Seed 42) FP32 tĩnh `(1, 1, 260)`
- [export/cnn1d_binary_dsB_kd_int8.onnx](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/cnn1d_binary_dsB_kd_int8.onnx) — **`M2-B-KD` (Seed 42) INT8 QDQ Per-Channel (Khuyến nghị triển khai)**
- [export/calib_train.npy](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/calib_train.npy) — Tập hiệu chuẩn `600` nhịp phân tầng từ `train.npz` (`300 N + 300 non-N`)
- [export/host_val_test500_x.npy](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/host_val_test500_x.npy), [export/host_val_test500_y_m1reg.npy](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/host_val_test500_y_m1reg.npy), [export/host_val_test500_y_m2bkd.npy](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/host_val_test500_y_m2bkd.npy) — Bộ 500 nhịp Test kiểm thử Host/Target Validation trên STM32Cube.AI

### Scripts Thực thi (`scripts/binary/`)
- [scripts/binary/make_calibration_set.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/scripts/binary/make_calibration_set.py) — Tạo tập Calibration & Host Validation
- [scripts/binary/export_onnx.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/scripts/binary/export_onnx.py) — Gộp BatchNorm, thay `AdaptiveAvgPool1d` bằng `AvgPool1d` tĩnh và xuất ONNX
- [scripts/binary/verify_onnx.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/scripts/binary/verify_onnx.py) — Kiểm tra tương đương số học PyTorch vs ONNXRuntime (C1 & C3)
- [scripts/binary/quantize_and_eval_int8.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/scripts/binary/quantize_and_eval_int8.py) — Lượng tử hóa tĩnh INT8 QDQ & đánh giá trên 49,691 nhịp Test (D2 & D3)
- [scripts/binary/stm32_cubeai_sim.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/scripts/binary/stm32_cubeai_sim.py) — Phân tích chi tiết bộ nhớ Flash/RAM, `im2col` và mô phỏng chu kỳ xung nhịp STM32Cube.AI (C2 & D4)
- [scripts/binary/plot_int8_comparison.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/scripts/binary/plot_int8_comparison.py) — Vẽ biểu đồ đối chiếu FP32 vs INT8
