# Báo cáo Tiến trình Triển khai Mô hình ECG `M2-B-KD` lên ESP32-S3 (Từ bước xuất `.onnx` đến hết Giai đoạn F0)

**Ngày lập báo cáo:** 2026-11-19  
**Mô hình triển khai:** `M2-B-KD` (`CNN1DBinaryDS`, `G=1, D=2`, `29,058` tham số sau gộp BatchNorm, đầu vào `(1, 1, 260)`, đầu ra `(1, 2)`)  
**Phần cứng mục tiêu:** `ESP32-S3 (N16R8)` — Dual-core Xtensa LX7 @ 240 MHz, PIE SIMD 128-bit, 512 KB Internal SRAM, 16 MB Flash, 8 MB PSRAM  
**Kế hoạch tham chiếu:**
- Kế hoạch gốc: [plan/esp32s3_deploy_plan.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/esp32s3_deploy_plan.md)
- Kế hoạch hiện hành (hiệu chỉnh sau Giai đoạn C): [plan/esp32s3_deploy_plan_v3.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/esp32s3_deploy_plan_v3.md)
- *(Ghi chú: File `plan/esp32s3_qemu_plan_v2.md` không tồn tại trong thư mục `plan/`; báo cáo này đối chiếu theo hai bản kế hoạch hiện có nêu trên.)*

---

## 1. Tổng quan Tiến độ Toàn tuyến

| STT | Tiến trình | Môi trường | Cổng kiểm duyệt | Trạng thái | Tài liệu / Đầu ra chính |
| :-: | :--- | :---: | :---: | :---: | :--- |
| **0** | Xuất ONNX tĩnh & tập Calibration/Validation | Python | Gate C1–D | **HOÀN THÀNH** | [onnx_export_summary.json](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/onnx_export_summary.json), [edge_ai_results_summary.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/edge_ai_results_summary.md) |
| **A** | Đánh giá trọng số mô hình & năng lực ESP32-S3 | Python | **Gate A** | **PASS** | [fit_assessment.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/fit_assessment.md), [fit_assessment.json](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/fit_assessment.json) |
| **B** | Lượng tử hóa 7 cấu hình sang `.espdl` (ESP-PPQ) | Python (`esp-ppq 1.3.14`) | — | **HOÀN THÀNH** | [phase_b_quantization_report.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/phase_b_quantization_report.md), `esp32s3/quant/*.espdl` |
| **C** | Đánh giá trên Host (Seed 42) → **Phát hiện cần Plan v3** | Python | Gate H (cũ) | **ĐÃ ĐỔI SANG v3** | [host_eval.json](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/host_eval.json), [esp32s3_deploy_plan_v3.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/esp32s3_deploy_plan_v3.md) |
| **C2** | Kiểm tra nhất quán, chạy 3 Seeds (`42,43,44`) & Shortlist | Python | **Gate H2** | **PASS** | [host_eval_report.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/host_eval_report.md), [shortlist.json](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/shortlist.json) |
| **F0** | Smoke test sớm ESP-DL & PIE SIMD trên QEMU ESP32-S3 | ESP-IDF v5.5.2 + QEMU 9.2.2 | **Gate Q0** | **PASS (4/4)** | [qemu_smoke.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/qemu_smoke.md), [f0_qemu.log](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/f0_qemu.log) |
| **D** | Sinh mã C thuần FP32 + 50 test vector có chủ đích | Python + GCC | Gate D | **CHƯA LÀM (Kế tiếp)** | `esp32s3/c_fp32/`, `test_vectors.h`, `thresholds.h` |
| **E** | Xây dựng Firmware ESP-IDF đa chế độ (`5` chế độ) | ESP-IDF v5.5.2 | — | **CHƯA LÀM** | `esp32s3/ecg_qemu/` |
| **F** | Chạy QEMU đầy đủ trên 50 nhịp × 5 chế độ | QEMU 9.2.2 | Gate Q | **CHƯA LÀM** | `esp32s3/results/qemu_run.log` |
| **G** | Báo cáo tổng kết & chuẩn bị đo board thật | Python | — | **CHƯA LÀM** | `reports/esp32s3_fp32_int16_int8.md` |

---

## 2. Chi tiết Từng Tiến trình: Ý nghĩa — Vấn đề — Cách khắc phục

### Bước 0: Xuất đồ thị ONNX tĩnh & Chuẩn bị tập Hiệu chuẩn (`export/`)
- **Ý nghĩa:** Chuyển đổi mô hình PyTorch đã huấn luyện (`M2-B-KD`, 3 seeds `42, 43, 44`, kèm `M1-Reg` để đối chiếu) sang định dạng trung gian ONNX opset 17 có kích thước tĩnh `(1, 1, 260) -> (1, 2)`, tương thích tuyệt đối với các trình biên dịch MCU (`STM32Cube.AI`, `ESP-PPQ`). Đồng thời tạo tập hiệu chuẩn phân tầng (`calib_train.npy`: `600` nhịp từ `train.npz`) và bộ kiểm chứng nhanh (`host_val_test500_x.npy`: `500` nhịp từ TEST).
- **Kết quả chính:**
  - `M2-B-KD` (`cnn1d_binary_dsB_kd_seed{42,43,44}_fp32.onnx`): dung lượng `117.43 KB` (`120,247 bytes`), `29,058` tham số BN-folded, chỉ gồm đúng 6 loại toán tử chuẩn (`Conv`, `Relu`, `MaxPool`, `AveragePool`, `Flatten`, `Gemm`).
  - Sai số số học ONNX vs PyTorch `<= 2.38e-6`, `COS = 1.000000`, khớp nhãn `100%`.
- **Vấn đề gặp phải:**
  1. `BatchNorm1d` rời rạc gây tốn bộ nhớ/phép tính và làm hỏng scale khi lượng tử hóa tĩnh trên MCU.
  2. `AdaptiveAvgPool1d(1)` là toán tử kích thước động không được nhiều runtime nhúng hỗ trợ.
- **Cách đã khắc phục:**
  - Gộp toán học (fold) toàn bộ `BatchNorm1d` vào trọng số và bias của `Conv1d` ngay trong [scripts/binary/export_onnx.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/scripts/binary/export_onnx.py).
  - Thay `AdaptiveAvgPool1d(1)` bằng `AveragePool1d(kernel_size=32, stride=32)` tĩnh, cố định hình dạng tensor toàn đồ thị.

---

### Giai đoạn A: Đánh giá Trọng số Mô hình & Khả năng Xử lý của ESP32-S3 (`Gate A — PASS`)
- **Ý nghĩa:** Kiểm tra định lượng xem kiến trúc `M2-B-KD` có "vừa vặn" với tài nguyên Flash, SRAM nội (không cần PSRAM), tập lệnh vector **PIE SIMD 128-bit** (`16 lanes INT8`, `8 lanes INT16`), và đặc tính lượng tử hóa lũy thừa 2 (`scale = 2^exponent`) của ESP32-S3 hay không, trước khi tốn công lượng tử hóa và viết firmware.
- **Kết quả chính** ([fit_assessment.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/fit_assessment.md)):
  - **Tài nguyên:** Tổng `1,468,544` Conv/Gemm MACC (`1,497,472` tổng ops). Arena SRAM đỉnh chỉ `13.44 KB` (INT8) / `25.88 KB` (INT16) / `50.77 KB` (FP32) — nằm gọn trong `~320 KB` SRAM nội khả dụng, không cần PSRAM.
  - **Độ phù hợp PIE SIMD:** `94.18%` tổng MACC (toàn bộ 3 khối Depthwise/Pointwise có kênh `32, 64, 128`) chia hết cho `16`, tận dụng tối đa 16-lane PIE không đệm thừa.
  - **Ước tính tốc độ @ 240 MHz:** INT8 `1.3–2.6 ms/nhịp`, INT16 `2.6–5.2 ms/nhịp`, FP32 `12.5–25.0 ms/nhịp` (đều đạt Gate A `< 10 ms` cho INT8).
  - **Tương thích toán tử:** `esp-ppq 1.3.14` đọc trực tiếp cả 6 toán tử 1D từ ONNX không cần chuyển sang 2D.
- **Vấn đề phát hiện:**
  - **Đánh đổi biên độ sóng ECG ở đầu vào INT8 lũy thừa 2:** Dữ liệu ECG sau z-score nằm trong `[-4.44, +7.08]`. Nếu chọn `exp_in = -4` (`scale = 0.0625`, ngưỡng `±7.94`), không cắt đỉnh R (`0%` clipped) nhưng sóng P/T nhỏ (`~0.30`) chỉ còn **5 mức rời rạc**. Nếu chọn `exp_in = -5` (`scale = 0.03125`, ngưỡng `±3.97`), sóng P/T tăng lên **10 mức** nhưng **`52.17%` số nhịp bị cắt ngọn đỉnh R**. Trong khi đó INT16 (`exp = -12`) giữ được **1,229 mức** cho sóng P/T và `0%` cắt đỉnh (`SQNR = 83.04 dB`).
- **Cách khắc phục (định hướng sang Giai đoạn B):** Thiết kế ma trận so sánh 7 cấu hình lượng tử hóa khác nhau, bao gồm tăng cường tập hiệu chuẩn đỉnh cao (`C1`), ép exponent đầu vào (`C2a/C2b`), **Mixed Precision INT16 đầu/cuối + INT8 ở giữa (`C3`)**, `W8A16` (`C_W8A16`), và `INT16` toàn mạng (`B`).

---

### Giai đoạn B: Lượng tử hóa 7 Cấu hình bằng `ESP-PPQ 1.3.14` sang `.espdl`
- **Ý nghĩa:** Sinh trực tiếp các file nhị phân `.espdl` (đã đóng gói layout trọng số PIE và bias 20/40-bit cho ESP32-S3) cho 7 cấu hình thử nghiệm, đo mức nhiễu lượng tử hóa từng lớp (`NOISE:SIGNAL POWER RATIO`) để hiểu chính xác lớp nào gây sai số.
- **Kết quả chính** ([phase_b_quantization_report.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/phase_b_quantization_report.md)):
  - Xuất thành công 7 file `.espdl` trong `esp32s3/quant/`: `b_int16.espdl` (`72.39 KB`), `c_w8a16.espdl` (`45.95 KB`), `c3.espdl` (`44.12 KB`), và `c0/c1/c2a/c2b.espdl` (`42.09 KB`).
  - **Điểm đột phá tìm ra:** Ở `C0` mặc định, lớp `features.4/Relu_output_0` bị gán `exp = -5` (cắt ngọn tại `3.97`), làm nhiễu cuối mạng (`Gemm`) vọt lên `7.020%`. Khi bổ sung 200 nhịp `V/S/F` có biên độ đỉnh cao từ `train.npz` vào tập hiệu chuẩn (`calib_c1_highpeak.npy`, 800 nhịp), `C1` tự động nâng `features.4/Relu_output_0` lên `exp = -4`, **giảm 5.6 lần nhiễu cuối mạng** (`7.020% -> 1.239%`) mà không tăng bất kỳ byte bộ nhớ hay chu kỳ tính toán nào. `C3` (Mixed INT16 Stem+Gemm) giảm nhiễu tiếp xuống `0.534%` với kích thước chỉ `44.12 KB`.
- **Vấn đề gặp phải:**
  1. `espdl_quantize_onnx` tự chạy `onnxsim.simplify` và **ghi đè trực tiếp file ONNX đầu vào**.
  2. Dung lượng `.espdl` thực tế (`42.09 KB` INT8, `72.39 KB` INT16) lớn hơn ước tính lý thuyết ở Giai đoạn A (`32.57 KB` / `59.38 KB`) do kèm metadata FlatBuffers, test values và bias đóng gói 20/40-bit.
- **Cách đã khắc phục:**
  - Trong [esp32s3/quant/quantize_all.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/quant/quantize_all.py), luôn sao chép `.onnx` sang bản làm việc tạm trước khi gọi `espdl_quantize_onnx`, bảo toàn nguyên vẹn `export/cnn1d_binary*.onnx`.
  - Ghi nhận dung lượng `.espdl` thực tế để tính kích thước phân vùng app ở Giai đoạn E (`CONFIG_PARTITION_TABLE_SINGLE_APP_LARGE=y`).

---

### Giai đoạn C: Đánh giá trên Host (Seed 42) & Lý do Nâng cấp lên Plan v3
- **Ý nghĩa:** Đánh giá đầy đủ cả 7 cấu hình lượng tử hóa trên toàn bộ tập `VAL (9,885 nhịp)` và `TEST (49,691 nhịp — DS2 inter-patient)` cùng phép thử độ nhạy biên độ `α ∈ {0.5, 0.75, 1.0, 1.25, 1.5}`.
- **7 vấn đề nghiêm trọng phát hiện từ kết quả Giai đoạn C:**
  1. **AUROC bị nhiễu không làm Gate được:** AUROC của `C0, C1, C2b, C3` lại *cao hơn* FP32 (`+0.0012` đến `+0.0065`); `C2b` "đạt" AUROC nhưng khớp nhãn chỉ `96.69%`.
  2. **Suy giảm ẩn ở lớp hiếm `F` (`388` nhịp):** `C1` và `C3` giảm recall lớp `F` tới `-3.9%`, `C2b` giảm `-9.3%` — bị che khuất nếu chỉ nhìn Recall/Spec nhị phân tổng.
  3. **Bất ổn ngưỡng ở `C1`:** `C1` đạt `85.01% / 85.88%` ở ngưỡng cố định nhưng **trượt** xuống `86.70% / 83.93%` khi hiệu chỉnh lại ngưỡng trên VAL.
  4. **Chỉ có 1 seed (Seed 42):** Độ chênh giữa các cấu hình nhỏ hơn độ lệch chuẩn liên-seed của chính mô hình FP32 (`±0.0072` AUROC).
  5. **Chưa kiểm chứng nhất quán CPU vs CUDA:** File `.espdl` ở Bước B tạo trên CPU, còn `eval_host.py` chạy lại lượng tử hóa trên CUDA.
  6. **`C2a` dư thừa & giới hạn của phép thử `α`:** `C2a` trùng 100% với `C0`; phép thử `α` thực hiện sau z-score nên chỉ là stress-test, không bao phủ rủi ro tiền xử lý ADC thật.
  7. **Thiếu bước kiểm thử QEMU sớm & test vector 50 nhịp ngẫu nhiên không đủ độ phân biệt** (các cấu hình khớp nhau `98–99.9%`).
- **Cách đã khắc phục:** Viết lại kế hoạch mới [plan/esp32s3_deploy_plan_v3.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/esp32s3_deploy_plan_v3.md) bổ sung **Giai đoạn C2** (nhất quán CPU/CUDA, đánh giá 3 seeds, Gate H2 5 tiêu chí theo PR-AUC + khớp nhãn + recall từng lớp `S/V/F` + cả 2 ngưỡng), chèn **Giai đoạn F0** (fail-fast QEMU sớm trước khi viết mã D/E), và đổi bộ 50 test vector ở Giai đoạn D sang chọn có chủ đích.

---

### Giai đoạn C2 (theo Plan v3): Nhất quán, 3 Seeds (`42, 43, 44`) & Gate H2 (`PASS`)
- **Ý nghĩa:** Khắc phục triệt để các lỗ hổng thống kê của Giai đoạn C để chốt danh sách ứng viên (`shortlist.json`) đáng tin cậy trước khi đưa vào firmware.
- **Kết quả chính** ([host_eval_report.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/host_eval_report.md), [consistency_check.json](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/consistency_check.json)):
  - **C2.1 (Nhất quán):** Khớp **100% exponent (`0` tensor lệch trên cả 6 cấu hình × 56 tensor)** giữa `.espdl` (Giai đoạn B), CPU và CUDA; sai số logit CPU vs CUDA bằng `0.00000000` trên mọi cấu hình INT8.
  - **C2.2 & C2.3 (Gate H2 trên 3 seeds `42, 43, 44` × 49,691 nhịp TEST):**
    - **`B (INT16 w16a16)` (`72.39 KB`):** **PASS tuyệt đối 5/5** (`ΔPR-AUC = +0.0001`, Khớp nhãn `99.99%`, `ΔF = 0.00%`, Worst seed drop `-0.02%`).
    - **`C_W8A16 (W8A16)` (`45.95 KB`):** **PASS tuyệt đối 5/5** (`ΔPR-AUC = +0.0019`, Khớp nhãn `99.47%`, `ΔF = -0.95%` cố định / `-0.43%` hiệu chỉnh, Worst seed drop `-0.39%`).
    - **`C3 (Mixed INT16+INT8)` (`44.12 KB`):** **PASS theo Quy tắc C2.4.4** (`ΔPR-AUC = +0.0061`, Khớp nhãn `99.07%`, Seed 42 đạt `85.80% / 85.26%` cả 2 ngưỡng, `ΔF = -2.15% / -1.46%` nằm gọn trong `±1` độ lệch chuẩn liên-seed của FP32 là `±3.10%`).
    - **`C1 (INT8 Peak-Calib)` (`42.09 KB`):** **FAIL Gate H2** (Khớp nhãn `98.69% < 99%`, Spec hiệu chỉnh Seed 42 `83.93% < 85%`), nhưng **giữ lại làm `SPEED_CANDIDATE`** để đo tốc độ đối chiếu trên board thật.
    - **Loại bỏ:** `C0`, `C2a`, `C2b`.
- **Vấn đề kỹ thuật khi chạy script & cách khắc phục:**
  - Lỗi `KeyError: 'recall'` do hàm đánh giá trả về key `recall_non_N`, `specificity_N`, `subclass_recall` → đã sửa trong [eval_host.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/quant/eval_host.py).
  - Lỗi `AttributeError` khi unpickle `ppq_graph` do `esp-ppq` thay tham chiếu `Variable` bằng string khi pickle → sửa sang dựng trực tiếp đồ thị sống trong RAM bằng `espdl_quantize_onnx(..., skip_export=True)`.

---

### Giai đoạn F0 (theo Plan v3): Smoke Test ESP-DL trên QEMU ESP32-S3 (`Gate Q0 — PASS 4/4`)
- **Ý nghĩa:** Kiểm tra sớm ("fail-fast") xem `qemu-system-xtensa` (`esp_develop_9.2.2_20250817`) có chạy được `espressif/esp-dl v3.3.15` và tập lệnh PIE (TIE728) hay không, nhằm quyết định hướng đi (dùng PIE trực tiếp hay chuyển Plan B `target="c"` / `esp32c3`) trước khi viết firmware Giai đoạn E.
- **Kết quả chính** ([qemu_smoke.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/qemu_smoke.md), [f0_qemu.log](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/f0_qemu.log)):
  - Cả **4/4 mô hình shortlist chuẩn phần cứng (`target="esp32s3"`)** đều vượt qua `model->test()` **khớp bit-exact (`err=0`, `Test Pass!`)** trên QEMU ESP32-S3:
    - `B (INT16)`: `[18712, -24134]` (`exp=-12`), `PASS`
    - `C_W8A16`: `[17386, -21610]` (`exp=-12`), `PASS`
    - `C3 (Mixed)`: `[14065, -18948]` (`exp=-12`), `PASS`
    - `C1 (INT8)`: `[127, -128]` (`exp=-5`), `PASS`
  - **Không cần** chuyển sang `esp32c3` hay duy trì bộ `.espdl` riêng `target="c"`.
- **4 vấn đề kỹ thuật lớn phát hiện trong F0 & cách đã khắc phục:**
  1. **Crash trước `app_main` (`Coprocessor exception` CP0):** Biến toàn cục `AttentionFloatExpLut` trong `esp-dl` dùng lệnh float trong `do_global_ctors()` trước khi FreeRTOS bật `CPENABLE` → **Khắc phục:** Đăng ký `ESP_SYSTEM_INIT_FN(enable_coprocessors_before_ctors, CORE, BIT(0), 100)` đặt `CPENABLE = 0x09` (bật cả FPU CP0 và PIE CP3).
  2. **Crash `IllegalInstruction` tại `0x4037827f` trong `AveragePool` INT8:** Toàn bộ 9 lớp `Conv`/`DWConv`/`PWConv` + 3 `MaxPool` (`>99.7%` MACC) chạy mượt bằng kernel PIE `tie728` thật trên QEMU; duy nhất `dl_tie728_s8_avg_pool2d_hwc_sum` dùng `EE.ST.QACC_*` chưa được QEMU 9.2.2 giả lập → **Khắc phục cho QEMU:** Bật macro `DL_QEMU_AVOID_ST_QACC 1` để riêng `AveragePool` dùng vòng lặp C `avgpool2d_hwc_sum` (bit-identical).
  3. **2 lỗi giả lập QEMU ở thanh ghi 40-bit `ACCX` tại nhánh `n_remainder` của `Gemm` (`out_ch=2 < 8`):** Qua đo inline-asm (`probe_qemu_accx`), phát hiện QEMU 9.2.2 bị lỗi không mở rộng dấu khi `EE.LD.ACCX.IP` nạp bias 40-bit âm (làm `EE.VMULAS` kế tiếp bão hòa `INT40_MAX`), và `EE.SRS.ACCX` với `shift >= 8` làm hỏng bit dấu (`-165326 >> 8` thành `2147483325`) → **Khắc phục cho QEMU:** Bật macro `DL_QEMU_GEMM_SCALAR_FALLBACK 1` trong `dl_module_gemm.hpp` riêng cho `Gemm` có `output_channel < 8` (chỉ chiếm `0.017%` MACC).
  4. **Căn chỉnh 16-byte cho `.espdl` trong flash:** `EMBED_FILES` chỉ căn lề 4-byte khiến `FbsLoader` cảnh báo và lệch địa chỉ tải vector PIE → **Khắc phục:** Cấp phát/sao chép qua buffer căn lề 16-byte (`heap_caps_aligned_alloc(16, ...)`) khi nạp model.
  5. *(Lỗi công cụ phụ)*: `esptool merge_bin` không đọc được `@flasher_args.json` → sửa [run_f0_smoke.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/scripts/esp32s3/run_f0_smoke.py) parse JSON trực tiếp trong Python.

---

## 3. Các Vấn đề Hiện tại (Sau khi xong Giai đoạn F0)

1. **Quản lý bản vá QEMU khi chuyển sang Firmware chính (Giai đoạn E) & Board thật:**
   - Hai macro workaround (`DL_QEMU_AVOID_ST_QACC`, `DL_QEMU_GEMM_SCALAR_FALLBACK`) hiện đang sửa trực tiếp trong `esp32s3/f0_smoke/managed_components/espressif__esp-dl/`. Khi tạo dự án chính `esp32s3/ecg_qemu`, cần tự động hóa việc áp dụng/gỡ bỏ 2 macro này (qua script hoặc cờ CMake/Kconfig) để đảm bảo **bản build cho board ESP32-S3 thật tắt hoàn toàn workaround và chạy 100% PIE phần cứng**.
2. **Chưa có mốc chuẩn FP32 C thuần và bộ 50 test vector có chủ đích:**
   - Giai đoạn F0 mới chạy trên 1 nhịp test vector mặc định bên trong mỗi `.espdl`. Chưa có kiểm chứng chéo trên 50 nhịp khó (nhịp bất đồng nhãn giữa các cấu hình, nhịp lớp `F`, nhịp lớp `V` đỉnh cao) cũng như chưa có mã C FP32 để đối chứng trực tiếp trên cùng firmware.
3. **Chưa có số đo độ trễ thực tế trên silicon ESP32-S3:**
   - Các giá trị `dt_qemu` ở F0 (`7.8 ms` – `25.8 ms`) chỉ là thời gian giả lập QEMU, **không dùng để kết luận tốc độ thật**. Đặc biệt, khoảng cách tốc độ thực giữa `C_W8A16` (activation 16-bit, có thể gần `B_INT16`) so với `C3` (Mixed, 96% MACC là INT8) và `C1` (INT8 thuần) chỉ quyết định được sau khi nạp lên board thật.
4. **Phạm vi tiền xử lý tín hiệu trên chip (ngoài phạm vi Plan v3, nhưng cần lưu ý cho triển khai thực tế):**
   - Toàn bộ đánh giá hiện tại giả định đầu vào đã được cắt cửa sổ 260 mẫu quanh đỉnh R và chuẩn hóa z-score (`mean=0, std=1`). Khi triển khai sản phẩm thực, sai lệch biên độ từ ADC thật sẽ tác động trực tiếp qua khâu z-score trên chip này.

---

## 4. Hướng Xử lý Tiếp theo (Theo Plan v3: `D → E → F → G`)

1. **Giai đoạn D — Sinh mã C FP32 & Bộ 50 Test Vector có chủ đích (~50 phút, Gate D):**
   - **D1:** Viết `esp32s3/c_fp32/gen_c.py` đọc `export/cnn1d_binary_dsB_kd_fp32.onnx`, sinh `model_fp32.h` (`~113 KB` flash) và `model_fp32.c` (dùng 2 buffer ping-pong tĩnh `~66.5 KB` SRAM). Biên dịch trên PC bằng GCC (MSYS2) và kiểm tra trên 500 nhịp `host_val_test500_x.npy` đạt **Gate D1: `max_logit_diff < 1e-4`** so với ONNXRuntime.
   - **D2:** Viết `esp32s3/quant/make_vectors.py` rút đúng **50 nhịp từ TEST** (`20` nhịp bất đồng nhãn giữa ít nhất 1 ứng viên với FP32 + `10` nhịp lớp `F` + `10` nhịp lớp `V` đỉnh cao + `10` nhịp lớp `N`), xuất `test_vectors.h`, `test_vectors_meta.json`, và `thresholds.h` (ngưỡng chênh logit `logit1 - logit0 > ln(th/(1-th))`, không cần softmax trên chip).
2. **Giai đoạn E — Firmware ESP-IDF Đa chế độ `esp32s3/ecg_qemu` (~60 phút):**
   - Dựng dự án ESP-IDF v5.5.2 (`CONFIG_PARTITION_TABLE_SINGLE_APP_LARGE=y`, `CONFIG_COMPILER_OPTIMIZATION_PERF=y`, `CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384`, không bật PSRAM).
   - Nhúng cả 5 chế độ: `fp32` (C thuần) + 4 `.espdl` shortlist (`b_int16`, `c_w8a16`, `c3`, `c1`) kèm `ESP_SYSTEM_INIT_FN` bật `CPENABLE = 0x09` và căn lề 16-byte.
   - Đóng gói gọn 2 macro QEMU workaround để bật khi chạy QEMU và tắt chỉ bằng 1 cờ khi nạp board thật.
   - Kiểm tra `idf.py size` (Flash & SRAM nội).
3. **Giai đoạn F — Chạy QEMU Đầy đủ (`50 nhịp × 5 chế độ`, ~20 phút, Gate Q):**
   - Thu log `esp32s3/results/qemu_run.log`.
   - Kiểm duyệt **Gate Q**: FP32 đạt `max_logit_diff < 1e-3` so với ONNX; cả 4 chế độ INT (`b_int16`, `c_w8a16`, `c3`, `c1`) đạt `match_ppq >= 49/50` và `max_lsb_diff <= 1`.
4. **Giai đoạn G — Báo cáo Tổng kết & Quy tắc Chốt Cấu hình trên Board thật (~25 phút):**
   - Lập báo cáo tổng hợp `reports/esp32s3_fp32_int16_int8.md` gắn nhãn rõ nguồn số liệu (`[tính từ ONNX]`, `[host/ESP-PPQ]`, `[QEMU]`, `[board thật — chưa đo]`).
   - Khi có board ESP32-S3 N16R8 thật: tắt 2 macro QEMU, nạp cùng firmware, đo `avg_dt_us` cho 5 chế độ và áp dụng quy tắc G3 (nếu `C_W8A16`/`B` chậm hơn `C3`/`C1` dưới `20%` → chọn `C_W8A16` hoặc `B`; nếu `C3` nhanh hơn rõ rệt → chọn `C3` do đã đạt Gate H2 theo quy tắc `C2.4.4`).
