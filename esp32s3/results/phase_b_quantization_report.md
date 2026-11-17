# Báo cáo Giai đoạn B: Lượng tử hóa Mô hình M2-B-KD bằng ESP-PPQ (v1.3.14) cho ESP32-S3

**Mô hình nguồn:** `M2-B-KD` ([export/cnn1d_binary_dsB_kd_fp32.onnx](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/cnn1d_binary_dsB_kd_fp32.onnx) — thông qua bản sao làm việc để bảo toàn tuyệt đối file gốc)  
**Script thực thi:** [esp32s3/quant/quantize_all.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/quant/quantize_all.py)  
**Log sai số từng lớp (ESP-PPQ):** [esp32s3/quant/quant_log.txt](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/quant/quant_log.txt)  
**Chi tiết cấu hình & Exponent từng Tensor:** [esp32s3/quant/quant_summary.json](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/quant/quant_summary.json)

---

## 1. Tổng hợp 7 Cấu hình `.espdl` Đã Sinh Thành công (`[Đã đo bằng ESP-PPQ 1.3.14]`)

Toàn bộ 7 cấu hình đều xuất trực tiếp từ đồ thị 1D (`Conv1d`, `MaxPool1d`, `AveragePool1d`, `Flatten`, `Gemm`) sang định dạng `.espdl` (kèm `.json` và `.info`) mà **không cần chuyển đổi sang 2D**.

| Mã Cấu hình | File `.espdl` | Dung lượng (`.espdl`) | Nền tảng Đích (`TargetPlatform`) | Exponent Đầu vào (`ecg_input`) | Exponent Đầu ra (`logits`) | Tỷ lệ Nhiễu/Tín hiệu tại Lớp cuối (`Gemm`) | Khớp nhãn FP32 (`Host 500`) | Cosine vs FP32 (`Host 500`) | Sai số Logit TB (`Mean Diff`) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`B` (`b_int16`)** | [b_int16.espdl](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/quant/b_int16.espdl) | `72.39 KB` | `ESPDL_S3_INT16` (`w16a16`) | `2^-12` | `2^-12` | **`0.000%`** | **`100.0%` (`500/500`)** | **`0.999996`** | `0.0008` |
| **`C0` (Mặc định)** | [c0.espdl](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/quant/c0.espdl) | `42.09 KB` | `ESPDL_S3_INT8` (`w8a8`, KL 600 nhịp) | `2^-4` | `2^-5` | `7.020%` | `97.0%` (`485/500`) | `0.970298` | `0.2329` |
| **`C1` (Calib đỉnh cao)** | [c1.espdl](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/quant/c1.espdl) | `42.09 KB` | `ESPDL_S3_INT8` (`w8a8`, KL 800 nhịp) | `2^-4` | `2^-5` | **`1.239%`** | **`98.2%` (`491/500`)** | **`0.990797`** | `0.1280` |
| **`C2a` (Ép `exp=-4`)** | [c2a.espdl](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/quant/c2a.espdl) | `42.09 KB` | `ESPDL_S3_INT8` (`w8a8`, không cắt đỉnh) | `2^-4` | `2^-5` | `7.020%` | `97.0%` (`485/500`) | `0.970298` | `0.2329` |
| **`C2b` (Ép `exp=-5`)** | [c2b.espdl](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/quant/c2b.espdl) | `42.09 KB` | `ESPDL_S3_INT8` (`w8a8`, cắt `±3.97`) | `2^-5` | `2^-5` | `32.624%` | `97.4%` (`487/500`) | `0.964747` | `0.2891` |
| **`C3` (Mixed 16+8)** | [c3.espdl](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/quant/c3.espdl) | **`44.12 KB`** | Stem + Gemm: `INT16`, DS blocks: `INT8` | `2^-12` | `2^-12` | **`0.534%`** | **`99.6%` (`498/500`)** | **`0.992078`** | `0.1194` |
| **`C_W8A16` (Mở rộng)**| [c_w8a16.espdl](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/quant/c_w8a16.espdl) | `45.95 KB` | `ESPDL_S3_W8A16` (Weight 8b, Act 16b) | `2^-12` | `2^-12` | **`0.271%`** | **`99.4%` (`497/500`)** | **`0.999574`** | `0.0430` |

---

## 2. Phân tích Cơ chế Sai số theo Lớp (Tại sao `C1` và `C3` Vượt trội `C0`?)

Bằng việc kiểm tra trực tiếp bảng `NOISE:SIGNAL POWER RATIO` trong [quant_log.txt](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/quant/quant_log.txt) và bảng exponent trong [quant_summary.json](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/quant/quant_summary.json), chúng ta xác định được nguyên nhân chính xác gây mất độ chính xác ở `C0` mặc định:

| Lớp Mạng | Noise Ratio `B (INT16)` | Noise Ratio `C0 (INT8 mặc định)` | Noise Ratio `C1 (INT8 + Calib đỉnh cao)` | Noise Ratio `C2b (Ép exp_in = -5)` | Noise Ratio `C3 (Mixed INT16+INT8)` | Noise Ratio `C_W8A16` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `features.0/Conv` (`1->32, k=7`) | `0.000%` | `0.277%` | `0.277%` | `2.115%` | **`0.000%`** (INT16) | `0.105%` |
| `features.3/Conv` (`DW k=9, g=32`) | `0.000%` | `0.495%` | `0.495%` | `2.584%` | **`0.148%`** | `0.105%` |
| `features.5/Conv` (`PW 32->64`) | `0.000%` | `4.506%` | **`1.317%`** | `6.725%` | **`0.453%`** | `0.205%` |
| `features.8/Conv` (`DW k=7, g=64`) | `0.000%` | `5.726%` | **`2.310%`** | `8.043%` | **`0.931%`** | `0.237%` |
| `features.10/Conv` (`PW 64->128`) | `0.000%` | `6.118%` | **`2.479%`** | `7.593%` | **`1.036%`** | `0.301%` |
| `features.13/Conv` (`DW k=5, g=128`)| `0.000%` | `7.151%` | **`3.550%`** | `9.154%` | **`1.618%`** | `0.379%` |
| `features.15/Conv` (`PW 128->128`) | `0.000%` | `5.765%` | **`2.558%`** | `8.676%` | **`1.297%`** | `0.290%` |
| `classifier/Gemm` (`128->2`) | **`0.000%`** | `7.020%` | **`1.239%`** | `32.624%` | **`0.534%`** (INT16) | **`0.271%`** |

### Giải thích Kỹ thuật:
1. **Điểm nghẽn ẩn tại `features.3/Conv -> features.4/Relu` trong `C0`:**
   - Ở `C0` (hiệu chuẩn trên 600 nhịp chuẩn), bộ quan sát KL gán cho đầu ra `features.4/Relu_output_0` hệ số `exponent = -5` (`scale = 0.03125`, cắt ngọn tại `3.97`). Khi gặp các nhịp ngoại tâm thu (`V/S/F`) có biên độ QRS lớn, đặc trưng sau lớp Depthwise đầu tiên vượt quá `3.97` và bị cắt ngọn hàng loạt, làm nhiễu tăng vọt từ `0.495%` ở `features.3` lên `4.506%` ở `features.5` và `7.020%` ở đầu ra `Gemm`.
2. **Hiệu quả của `C1` (Thêm 200 nhịp `V/S/F` đỉnh cao từ `train.npz` vào tập hiệu chuẩn):**
   - Nhờ thấy đủ phân phối đỉnh cao của các nhịp bất thường trong tập train, bộ quan sát KL tự động chỉnh `features.4/Relu_output_0` từ `exponent = -5` lên **`exponent = -4`** (`scale = 0.0625`, ngưỡng `7.94`).
   - Chỉ riêng thay đổi này giúp giảm **5.6 lần** tỷ lệ nhiễu tại đầu ra (`7.020% -> 1.239%`), nâng Cosine similarity từ `0.9703` lên **`0.9908`** và tỷ lệ khớp nhãn từ `97.0%` lên **`98.2%`** mà không tốn thêm bất kỳ byte bộ nhớ hay chu kỳ CPU nào!
3. **Đánh đổi của `C2a` (`exp_in = -4`) vs `C2b` (`exp_in = -5`):**
   - `C0` mặc định vốn đã chọn `exp_in = -4`, nên `C2a` trùng khớp hoàn toàn với `C0`.
   - Khi ép `exp_in = -5` (`C2b`), các nhịp bình thường/biên độ nhỏ có độ phân giải sóng P/T mịn gấp đôi nên tỷ lệ khớp nhãn tăng nhẹ (`97.4%` vs `97.0%`), nhưng các nhịp có đỉnh R cao bị cắt tại `±3.97` ngay từ đầu vào khiến sai số năng lượng tín hiệu (`NOISE:SIGNAL POWER RATIO`) tại lớp `Gemm` tăng lên `32.6%`.
4. **Sức mạnh của `C3` (Mixed Precision INT16 + INT8):**
   - Đặt khối đầu (`features.0/Conv` + `Relu` + `MaxPool`, chỉ chiếm `3.89%` MACC) và lớp cuối (`classifier/Gemm`, chỉ chiếm `0.02%` MACC) ở **INT16 (`exponent = -12`)** với bộ quan sát `minmax`, đồng thời giữ **toàn bộ 3 khối DS-CNN ở giữa (`96.09%` tổng MACC)** ở **INT8 16-lane PIE SIMD**.
   - Kết quả: Loại bỏ hoàn toàn hiện tượng mất sóng P/T nhỏ lẫn cắt đỉnh R ở đầu vào (`0.000%` nhiễu tại `features.0/Conv`), giảm nhiễu toàn mạng xuống chỉ còn **`0.534%`**, đạt tỷ lệ khớp nhãn **`99.6%` (`498/500` nhịp)** và dung lượng file `.espdl` chỉ **`44.12 KB`** (lớn hơn `C1` vỏn vẹn `2.03 KB`).
