# Plan hoàn chỉnh: đưa M2-B-KD lên ESP32-S3 (N16R8) — đánh giá khả năng → FP32/INT16/INT8 → QEMU

Thay thế [esp32s3_qemu_plan.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/esp32s3_qemu_plan.md) và [esp32s3_qemu_plan_v2.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/esp32s3_qemu_plan_v2.md).

**Mục tiêu:**
1. Đánh giá trọng số và tài nguyên của model so với năng lực ESP32-S3 (giai đoạn A).
2. So sánh FP32 / INT16 / INT8 về độ chính xác, có kiểm soát độ nhạy biên độ của sóng ECG.
3. Chạy được trên QEMU ESP32-S3. Đo thời gian thật khi có board.

**Đã có:** ESP-IDF 5.5.2, `qemu-xtensa 9.2.2`, `esp-ppq 1.3.14` (Conda `ecg`), `export/cnn1d_binary_dsB_kd_fp32.onnx`, `export/calib_train.npy`, `export/host_val_test500_x.npy`.
**Thư mục mới:** `esp32s3/` (không sửa file cũ).
**Terminal:** **T-PY** = Conda `ecg` (`C:\Users\vietc\.conda\envs\ecg\python.exe`). **T-IDF** = Command Prompt trong VS Code (đã nạp ESP-IDF; kiểm tra `idf.py --version` ra `v5.5.x`).

> [!NOTE]
> API ESP-PPQ/ESP-DL viết theo v3.x. Cần đối chiếu README của `espressif/esp-dl` trong `managed_components/` sau khi tải. Các số có nhãn `[ước tính]` là phép tính lý thuyết, chưa đo thật.

## Tổng quan các giai đoạn

| GĐ | Nội dung | Môi trường | Thời gian | Cổng |
|---|---|---|---|---|
| A | Đánh giá trọng số và tài nguyên so với ESP32-S3 | T-PY | ~30 phút | Gate A |
| B | Lượng tử hóa (INT16, INT8 và biến thể biên độ) | T-PY | ~40 phút | — |
| C | Đánh giá độ chính xác trên host (VAL/TEST) và độ nhạy biên độ | T-PY | ~30 phút | Gate H |
| D | Mã C FP32 (baseline chip) + test vectors | T-PY | ~40 phút | — |
| E | Firmware ESP-IDF 3 chế độ | T-IDF | ~50 phút | — |
| F | Chạy QEMU | T-IDF | ~15 phút | Gate Q |
| G | Báo cáo và chuẩn bị đo board | T-PY | ~15 phút | — |

Thứ tự ưu tiên khi thiếu thời gian: A → B (chỉ INT16, INT8 mặc định) → C → E → F; phần còn lại là mở rộng.

---

## Giai đoạn A. Đánh giá trọng số model so với ESP32-S3

**Để làm gì:** xác nhận model có vừa và chạy được trên chip trước khi tốn công làm firmware; biết trước chỗ nào chậm hoặc tốn bộ nhớ.

### A1. Năng lực ESP32-S3 N16R8 (đối chiếu tài liệu Espressif)
| Tài nguyên | Giá trị | Ghi chú |
|---|---|---|
| CPU | Xtensa LX7 ×2, 240 MHz | Chạy model trên 1 nhân là đủ |
| SRAM nội | 512 KB | Còn lại cho ứng dụng sau khi trừ IDF, khoảng 300 KB trở lên (kiểm chứng bằng `idf.py size`) |
| Flash | 16 MB | Weights nằm đây, đọc qua cache (XIP) |
| PSRAM | 8 MB (octal) | Chậm hơn SRAM, chưa cần cho model này |
| SIMD | PIE 128-bit | INT8: 16 lane; INT16: 8 lane |
| FPU | Đơn (FP32) | Không có SIMD cho FP32 |

### A2. Số liệu model (đã tính từ ONNX FP32 tĩnh)

| Lớp | Loại | Trọng số (shape) | Số tham số | Kênh có bội 16? |
|---|---|---|---|---|
| 1 | Conv 7 (1→32) | 32×1×7 | 224 | Vào = 1: SIMD khó tận dụng |
| 2 | Depthwise 9 (32) | 32×1×9 | 288 | 32 ✓ |
| 3 | Pointwise 32→64 | 64×32×1 | 2,048 | ✓ |
| 4 | Depthwise 7 (64) | 64×1×7 | 448 | ✓ |
| 5 | Pointwise 64→128 | 128×64×1 | 8,192 | ✓ |
| 6 | Depthwise 5 (128) | 128×1×5 | 640 | ✓ |
| 7 | Pointwise 128→128 | 128×128×1 | 16,384 | ✓ |
| 8 | Gemm 128→2 | 2×128 | 256 | Ra = 2: nhỏ, không đáng kể |
| — | Bias | — | 578 | — |
| **Tổng** | | | **29,058** | |

Nhận xét: lớp pointwise chiếm khoảng 90% MACC (tính từ shape) và có kênh là bội của 16, nên khả năng cao tận dụng tốt PIE. Lớp depthwise và lớp đầu (1 kênh vào) là chỗ SIMD kém hiệu quả hơn, nhưng đóng góp MACC nhỏ.

### A3. Dung lượng trọng số
| Định dạng | Kích thước ước tính | Ghi chú |
|---|---|---|
| FP32 | 116,232 B (≈113.5 KB) | 29,058 × 4 B |
| INT16 | ≈ 57–60 KB | trọng số 2 B + bias rộng hơn |
| INT8 | ≈ 30–33 KB | trọng số 1 B + bias 32-bit (≈2.3 KB) |
Cả ba đều nhỏ hơn rất nhiều so với 16 MB flash (≤ 0.7%).

### A4. Bộ nhớ activation (đỉnh theo cặp tensor vào/ra, tính từ shape)
| Điểm đỉnh | Số phần tử | INT8 | INT16 | FP32 |
|---|---|---|---|---|
| Vào → conv1 (260 + 8,320) | 8,580 | ≈ 8.4 KB | ≈ 16.8 KB | ≈ 33.5 KB |
| Pointwise 1 (4,160 + 8,320) | 12,480 | **≈ 12.2 KB** | ≈ 24.4 KB | ≈ 48.8 KB |
Con số 12.2 KB (INT8) và 48.8 KB (FP32) khớp báo cáo STM32 cũ. Tất cả nhỏ hơn rất nhiều so với 512 KB SRAM nội nên không cần PSRAM.

### A5. Chi phí tính toán và ước tính thời gian
- Tổng MACC: **1.72 M** (từ báo cáo trước).
- Ước tính thô `[ước tính]`, công thức: `cycles = MACC / (MAC_mỗi_chu_kỳ × hiệu_suất)`, `thời gian = cycles / 240 MHz`.

| Cấu hình | MAC/chu kỳ lý thuyết | Hiệu suất giả định | Thời gian ước tính |
|---|---|---|---|
| FP32 (C thuần, FPU) | ≈ 0.25–0.5 | — | ≈ 14–29 ms |
| INT16 (PIE) | 8 | 15–30% | ≈ 3.0–6.0 ms |
| INT8 (PIE) | 16 | 15–30% | ≈ 1.5–3.0 ms |
Đây là khoảng ước tính để lập kỳ vọng; con số thật chỉ có khi đo trên board (giai đoạn G). Tham chiếu STM32F446 INT8 ≈ 13.7 ms (khác chip, chỉ để hình dung tỷ lệ).

### A6. Kiểm tra tương thích toán tử
Liệt kê toán tử của ONNX: `Conv` (cả depthwise), `Relu`, `MaxPool`, `AveragePool`, `Flatten`, `Gemm`. Đối chiếu danh sách toán tử ESP-DL/ESP-PPQ hỗ trợ. Đánh dấu rủi ro: `Conv1d`/`AveragePool` 1D (có thể cần bản 2D tương đương, xem B).

### Việc làm ở A
Tạo `scripts/esp32s3/assess_fit.py` (file mới), đọc ONNX và tạo `esp32s3/results/fit_assessment.json` + `fit_assessment.md` gồm các bảng A2–A6 (tính từ ONNX, không gõ tay) và tham số hiệu suất có thể chỉnh.

**Gate A:** đạt khi (1) weights + activation của cả 3 định dạng vừa SRAM/flash, (2) không có toán tử nào chắc chắn không hỗ trợ (hoặc đã có phương án 2D), (3) thời gian ước tính INT8 < 10 ms. Nếu không đạt, dừng và xem lại kiến trúc trước khi làm tiếp.

---

## Giai đoạn B. Lượng tử hóa bằng ESP-PPQ (T-PY)

**Vì sao cần:** MCU chạy INT8/INT16 theo quy ước riêng của ESP-DL (scale = 2^n, đối xứng). Phải dùng `esp-ppq` lượng tử từ **ONNX FP32 + calib**, không dùng file INT8 QDQ của onnxruntime.
**ESP-DL chỉ chạy INT8/INT16, không chạy FP32** nên baseline FP32 trên chip dùng mã C thuần (giai đoạn D).

### Vấn đề biên độ (đã đo trên `calib_train.npy`)
Dữ liệu z-score từng nhịp (mean 0, std 1), giá trị từ −4.44 đến +7.08.

| Thống kê | Giá trị |
|---|---|
| \|x\| trung vị / p99 / p99.9 / p99.99 / tối đa | 0.38 / 3.94 / 5.95 / 6.76 / 7.08 |
| absmax mỗi nhịp: p1 / trung vị / p99 | 2.27 / 4.01 / 6.86 |

- Phủ tới 7.08 cần bước lượng tử `2^-4 = 0.0625` (INT8: 127×0.0625 = 7.94). Sóng P/T khoảng 0.2–0.4 chỉ còn ~3–6 mức.
- Bước mịn hơn (`2^-5 = 0.031`) buộc phải cắt ở ±3.97, mất đỉnh QRS.
- INT16 (`2^-12`) gần như xóa vấn đề này.

### Các cấu hình tạo ra
| Mã | Cấu hình | Mục đích |
|---|---|---|
| B | INT16 toàn mạng | Chính xác cao, an toàn biên độ |
| C0 | INT8 mặc định | Mốc INT8 |
| C1 | INT8, calib thêm 200 nhịp V/S/F có đỉnh cao (lấy từ **train.npz**, không dùng VAL/TEST) | Tránh scale quá nhỏ |
| C2a / C2b | INT8, exponent đầu vào −4 (không cắt) / −5 (cắt ±3.97) | Đánh đổi mịn sóng nhỏ ↔ mất đỉnh |
| C3 | Mixed: lớp đầu và lớp cuối INT16, còn lại INT8 | Giữ biên độ ở hai đầu, vẫn nhanh |

### Việc làm
`esp32s3/quant/quantize_all.py`: với mỗi cấu hình gọi `espdl_quantize_onnx(...)`:
```python
espdl_quantize_onnx(
    onnx_import_file="export/cnn1d_binary_dsB_kd_fp32.onnx",
    espdl_export_file="esp32s3/quant/<ten>.espdl",
    calib_dataloader=dl, calib_steps=8, input_shape=[1, 1, 260],
    target="esp32s3", num_of_bits=8,   # hoặc 16
    collate_fn=lambda b: b[0].to("cpu"), device="cpu",
    error_report=True, export_test_values=True, verbose=1)
```
- Mixed (C3): dùng `dispatching_override` chỉ định lớp 16-bit (lấy tên lớp từ `onnx.load(...).graph.node`).
- Ép exponent (C2): theo quantize config input của esp-ppq; nếu khó, bỏ C2 vì C1 và C3 thường đủ.
- **Dự phòng Conv1d:** nếu esp-ppq báo không hỗ trợ, tạo ONNX 2D tương đương (input `(1,1,1,260)`, kernel `(1,k)`) bằng `scripts/esp32s3/make_onnx_2d.py`, copy trọng số đã gộp BN.

Đầu ra: `esp32s3/quant/{b_int16,c0,c1,c2a,c2b,c3}.espdl` + `quant_log.txt` (lỗi theo lớp).
**Đạt khi:** B và C0 được sinh ra; các biến thể khác lỗi thì ghi vào báo cáo.

---

## Giai đoạn C. Đánh giá độ chính xác trên host (T-PY)

**Để làm gì:** kết quả độ chính xác chính thức, không phụ thuộc QEMU (QEMU không mô phỏng chu kỳ và có thể không chạy được PIE).

`esp32s3/quant/eval_host.py`:
- Chạy đồ thị lượng tử hóa mô phỏng của ESP-PPQ trên 9,885 VAL và 49,691 TEST với FP32 (mốc), B, C0–C3.
- Ghi: AUROC, PR-AUC, Recall/Spec tại ngưỡng chọn từ VAL (cùng cách chọn như Giai đoạn D trước), recall lớp S/V/F, cosine với logit FP32. Dùng seed 42 để so công bằng với [edge_ai_results_summary.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/edge_ai_results_summary.md).
- **Phép thử độ nhạy biên độ:** nhân input với `{0.8, 0.9, 1.0, 1.1, 1.25}` (giả lập lệch độ lợi ADC), đo lại AUROC của FP32, B và các biến thể INT8.

Đầu ra: `esp32s3/results/host_eval.json`, bảng markdown, biểu đồ.
**Gate H:** có cấu hình INT8 hoặc mixed với `ΔAUROC ≤ 0.005` so với FP32 trên TEST. Nếu không có, chọn INT16 làm phương án chính và ghi nhận INT8 chưa đạt.
Nếu C0 đã đạt Gate H thì bỏ qua C1–C3 ở các bước sau.

---

## Giai đoạn D. Mã C FP32 và test vectors (T-PY)

**D1. Mã C FP32 (baseline chip):** `esp32s3/c_fp32/gen_c.py` đọc ONNX FP32, xuất `model_fp32.h` (trọng số `const float[]` trong flash) và `model_fp32.c` (các hàm `conv1d`, `dwconv1d`, `relu`, `maxpool`, `avgpool`, `gemm`, buffer ping-pong tĩnh). Kiểm tra trên PC bằng gcc (MSYS2 `C:\msys64\ucrt64\bin`): so ONNXRuntime trên 500 nhịp, sai số < 1e-4.
**Để làm gì:** có mốc FP32 chạy được trên chip/QEMU (không dùng PIE nên QEMU chạy ổn), và để đo PIE nhanh hơn bao nhiêu.

**D2. Test vectors:** `esp32s3/quant/make_vectors.py` chọn **50 nhịp** (25 N + 25 non-N, gồm V/S/F) từ `host_val_test500_x.npy`, xuất `test_vectors.h`: input float, logit FP32, logit mô phỏng ESP-PPQ của từng cấu hình INT.

---

## Giai đoạn E. Firmware ESP-IDF 3 chế độ (T-IDF)

```
cd esp32s3\ecg_qemu
idf.py set-target esp32s3
idf.py add-dependency "espressif/esp-dl^3"
```
- Nhúng `.espdl` qua `EMBED_FILES` (`b_int16.espdl` và cấu hình INT8 tốt nhất từ Gate H), thêm `model_fp32.c`.
- `sdkconfig.defaults`: flash 16 MB, stack main 16 KB, **chưa bật PSRAM** (model nhỏ, giảm rủi ro QEMU).
- `main.cpp` chạy lần lượt fp32 / int16 / int8 trên 50 nhịp, in:
  `MODE=<..> agree=<n>/50 max_logit_diff=<x> match_host=<n>/50 avg_dt_us=<x>`
- Đầu vào INT: `x_int = clamp(round(x / 2^exp_in))`, đầu ra `y = y_int * 2^exp_out` (ưu tiên hàm quantize/dequantize của ESP-DL nếu có). `match_host` so logit chip với logit mô phỏng ESP-PPQ để tách lỗi triển khai khỏi lỗi thuật toán lượng tử hóa.

Build: `idf.py build`. Lỗi dự kiến: thiếu header → tìm tên đúng trong `managed_components\espressif__esp-dl\`; tràn stack → tăng stack main.

---

## Giai đoạn F. Chạy QEMU ESP32-S3 (T-IDF)

Cách 1: `idf.py qemu monitor`. Cách 2 (nếu không hỗ trợ target):
```
python -m esptool --chip esp32s3 merge_bin --fill-flash-size 16MB -o build\qemu_flash.bin @build\flasher_args.json
qemu-system-xtensa -nographic -machine esp32s3 -drive file=build\qemu_flash.bin,if=mtd,format=raw
```
Thoát: `Ctrl+A` rồi `X`. Thoát monitor: `Ctrl+]`.

**Gate Q:** mỗi chế độ in đủ dòng kết quả, `match_host ≥ 48/50`, chế độ FP32 có `max_logit_diff < 1e-3`.

> [!WARNING]
> **Rủi ro lớn nhất:** kernel ESP-DL trên S3 dùng lệnh PIE, QEMU có thể báo `Illegal instruction` ở INT8/INT16.
> **Plan B:** FP32 (C thuần) vẫn chạy trên QEMU để kiểm chứng khung chạy; INT8/INT16 kiểm chứng trên board thật. Ghi rõ trong báo cáo, không gán `avg_dt_us` của QEMU cho thời gian thật.

---

## Giai đoạn G. Báo cáo và chuẩn bị đo board

`reports/esp32s3_fp32_int16_int8.md` gồm:
- Bảng đánh giá khả năng (từ giai đoạn A).
- Bảng `{FP32, B, C0..C3}` × `{AUROC, PR-AUC, Recall, Spec, recall S/V/F, độ nhạy biên độ, kích thước model, kích thước firmware, trạng thái QEMU}`.
- Kết luận cấu hình chọn cho board.
- Nhãn nguồn số liệu: **[tính từ ONNX]**, **[ước tính]**, **[host/ESP-PPQ]**, **[QEMU]**, **[board thật — chưa đo]**.

**Khi có board** (cùng firmware): đọc `avg_dt_us` của 3 chế độ, rồi đối chiếu với khoảng ước tính ở A5. Quan sát: số kênh có là bội của 16/8 (căn lề PIE); weights đặt ở flash hay chép sang SRAM; buffer ở SRAM nội hay PSRAM.

---

## Điểm quyết định
1. **Sau Gate A:** không đạt thì dừng, xem lại kiến trúc.
2. **Sau Gate H:** C0 đạt thì bỏ C1–C3; chưa đạt thì thử C1 → C3 (C2 chỉ để hiểu đánh đổi); không INT8 nào đạt thì dùng INT16 cho board.
3. **Sau Gate Q:** nếu PIE crash trên QEMU, chuyển Plan B.
4. **Quyết định cuối về tốc độ:** chỉ sau khi có số đo trên board.

## Ngoài phạm vi (để sau)
Hiểu sâu bố cục bộ nhớ PIE/PSRAM, tối ưu latency, QAT, tích hợp ADC, lọc nhiễu và phát hiện đỉnh R.
