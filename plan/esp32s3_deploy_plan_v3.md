# Plan v3: Đưa M2-B-KD lên ESP32-S3 (N16R8) — hiệu chỉnh sau Giai đoạn C

**Trạng thái:** thay thế các bước D–G của [esp32s3_deploy_plan.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/esp32s3_deploy_plan.md). Giai đoạn A và B giữ nguyên và đã hoàn thành. Giai đoạn C được tách thành C (đã chạy, chỉ seed 42) và **C2 (bổ sung, mới)**.
**Mục tiêu cuối:** chọn cấu hình lượng tử hóa cho board thật, kiểm chứng pipeline trên QEMU ESP32-S3, có thước đo độ chính xác đủ tin cậy trước khi đo thời gian trên phần cứng.
**Kết quả đã có:** [phase_b_quantization_report.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/phase_b_quantization_report.md), [host_eval_report.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/host_eval_report.md), [fit_assessment.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/fit_assessment.md), các file `.espdl` trong `esp32s3/quant/`.

---

## 0. Vì sao cần plan mới

Plan cũ viết trước khi có số liệu Giai đoạn C. Kết quả C cho thấy 7 điểm khiến D–G phải đổi.

| # | Phát hiện từ kết quả C (seed 42, 49,691 nhịp TEST) | Hệ quả với plan cũ |
|---|---|---|
| 1 | AUROC của C0, C1, C2b, C3 **cao hơn** FP32 (+0.0012 đến +0.0065). Lượng tử hóa không thể làm mô hình tốt hơn, đây là nhiễu. | Gate H cũ (`\|ΔAUROC\| ≤ 0.005`) không phát hiện suy giảm thật. C2b "đạt" AUROC (+0.0065) nhưng khớp nhãn chỉ 96.69%, cosine 0.957. |
| 2 | Lớp F (388 nhịp TEST) giảm recall: C1 và C3 −3.9 điểm, C2b −9.3, C_W8A16 −1.3, B 0. Tóm tắt Giai đoạn C trước đã bỏ sót điều này. | Cần gate theo từng lớp con, không chỉ Recall/Spec nhị phân. |
| 3 | C1 đạt Recall/Spec ≥ 85% nhưng sát vạch (Recall 85.01%) và **trượt** khi hiệu chỉnh lại ngưỡng trên VAL (86.70% / 83.93%). C3, B, C_W8A16 đạt cả hai cách. | Việc "đạt Gate" của C1 phụ thuộc cách chọn ngưỡng. Cần gate ở cả ngưỡng cố định lẫn ngưỡng hiệu chỉnh. |
| 4 | Chỉ có 1 seed. Chênh AUROC giữa các cấu hình (≤ 0.0065) nhỏ hơn độ lệch chuẩn giữa các seed của M2-B-KD (0.0072 theo Giai đoạn D trước). | Không thể kết luận cấu hình nào tốt hơn nếu chưa chạy seed 43, 44. |
| 5 | `eval_host.py` dựng lại đồ thị lượng tử hóa trên GPU, còn các `.espdl` ở Giai đoạn B được tạo trên CPU. Chưa kiểm tra hai bản có cùng exponent từng lớp. | Kết quả C có thể không đại diện cho file `.espdl` sẽ nạp lên chip. |
| 6 | C2a trùng hoàn toàn C0 vì C0 đã tự chọn exponent đầu vào −4. Phép thử biên độ α áp dụng **sau** z-score, mà firmware thật phải tự z-score từng nhịp. | Bỏ C2a. Phép thử α chỉ là stress-test; rủi ro biên độ thực tế nằm ở khâu tiền xử lý trên chip. |
| 7 | Ứng viên đã đổi: B, C_W8A16, C3, C1 (không còn 1 INT16 + 1 INT8). Cả 4 `.espdl` chỉ 42–72 KB. Bộ 50 nhịp test ngẫu nhiên cũ khó phân biệt các cấu hình khớp nhau 98–99.9%. Rủi ro QEMU không chạy được lệnh PIE vẫn chưa được thử. | Phải đổi cách chọn test vector, thêm bước thử QEMU sớm, mở rộng firmware sang nhiều `.espdl`. |

**Giữ nguyên từ plan cũ:** A (khả năng phần cứng), B (sinh `.espdl`), nguyên tắc "QEMU kiểm chứng đúng/sai, không đo tốc độ", baseline FP32 bằng mã C thuần.

---

## Sơ đồ giai đoạn và cổng

| GĐ | Nội dung | Môi trường | Thời gian | Cổng |
|---|---|---|---|---|
| C2 | Kiểm tra nhất quán, chạy 3 seed, gate mới, chọn ứng viên | T-PY | ~60 phút | Gate H2 |
| F0 | Thử QEMU sớm với ví dụ ESP-DL | T-IDF | ~25 phút | Gate Q0 |
| D | Mã C FP32 + test vector mới | T-PY | ~50 phút | Gate D |
| E | Firmware ESP-IDF nhiều chế độ | T-IDF | ~60 phút | — |
| F | Chạy QEMU đầy đủ | T-IDF | ~20 phút | Gate Q |
| G | Báo cáo + chuẩn bị đo board và tiền xử lý | T-PY | ~25 phút | — |

**T-PY** = Conda `ecg` (`C:\Users\vietc\.conda\envs\ecg\python.exe`, chạy với quyền ngoài sandbox). **T-IDF** = Command Prompt đã nạp ESP-IDF 5.5.2 (`idf.py --version` phải ra `v5.5.x`).
**Thứ tự bắt buộc:** C2 → F0 → D → E → F → G. F0 làm trước D vì kết quả F0 quyết định có đầu tư mã D, E cho INT8/INT16 hay không.

---

## Giai đoạn C2. Kiểm tra nhất quán, nhiều seed, Gate H2

**Mục đích:** có số liệu đáng tin để chọn cấu hình trước khi viết firmware.

### C2.1 Kiểm tra nhất quán exponent giữa `.espdl` và đồ thị đánh giá
1. Tạo `esp32s3/quant/check_consistency.py`.
2. Với mỗi cấu hình trong `{b_int16, c0, c1, c2b, c3, c_w8a16}`: dựng lại đồ thị bằng đúng thông số của `quantize_all.py` (device `cpu`, cùng calib, cùng setting), trích exponent từng tensor bằng `extract_graph_quant_details`.
3. So với `quant_summary.json` (đã sinh ở B). Ghi danh sách tensor lệch.
4. Lặp lại với `device=cuda` (cách `eval_host.py` đang dùng), so với kết quả CPU.
5. Đầu ra: `esp32s3/results/consistency_check.json`.

**Xử lý kết quả:**
- Không lệch: giữ nguyên số liệu C, đánh dấu "đã xác nhận".
- Lệch ở một vài tensor: đổi `eval_host.py` sang `device=cpu`, chạy lại toàn bộ C trên CPU (khoảng 5–8 phút), dùng kết quả mới.

### C2.2 Chạy lại với seed 43 và 44
1. Tham số hóa `eval_host.py`: thêm đối số `--seed {42,43,44}` và đường dẫn ONNX tương ứng `export/cnn1d_binary_dsB_kd_seed{N}_fp32.onnx`. Tên node và tên đầu vào (`ecg_input`) giống seed 42 nên dùng được lại.
2. Với mỗi seed: FP32 → chọn ngưỡng từ VAL của FP32 seed đó → lượng tử hóa lại 6 cấu hình (bỏ C2a) bằng cùng calib (`calib_train.npy`, `calib_c1_highpeak.npy`) → đánh giá trên 9,885 VAL và 49,691 TEST.
3. Không cần xuất `.espdl` cho seed 43, 44 (`skip_export=True`), chỉ cần số liệu đánh giá.
4. Tập calib chung, không đụng VAL/TEST, giữ nguyên quy tắc đã dùng.
5. Đầu ra: `esp32s3/results/host_eval_seed{43,44}.json`, và `host_eval_3seeds.json` tổng hợp (trung bình, độ lệch chuẩn, giá trị tệ nhất).
6. Bổ sung vào mỗi bảng: recall từng lớp S, V, F và khoảng ±1 độ lệch chuẩn giữa 3 seed.

### C2.3 Gate H2 (thay Gate H cũ)
Đánh giá bằng **trung bình 3 seed**, kèm giá trị tệ nhất:

| Tiêu chí | Ngưỡng |
|---|---|
| ΔPR-AUC so với FP32 | ≤ 0.010 (giá trị tuyệt đối) |
| Khớp nhãn với FP32 tại ngưỡng vận hành | ≥ 99% |
| Giảm recall từng lớp S, V, F so với FP32 | ≤ 2 điểm % mỗi lớp |
| Recall ≥ 85% **và** Specificity ≥ 85% | Đạt ở **cả** ngưỡng FP32 cố định **và** ngưỡng hiệu chỉnh VAL |
| Seed tệ nhất | Không trượt Recall/Spec quá 0.5 điểm so với 85% |

AUROC vẫn ghi trong bảng nhưng **không** dùng làm cổng.

**Dự kiến theo seed 42 (chỉ để định hướng, cần C2.2 xác nhận):**

| Cấu hình | ΔPR-AUC | Khớp nhãn | Δ recall F | Recall/Spec cả 2 ngưỡng | Dự kiến |
|---|---|---|---|---|---|
| B (INT16) | +0.0002 | 99.98% | 0 | Đạt | Đạt |
| C_W8A16 | −0.0086 | 99.32% | −1.3 | Đạt | Đạt |
| C3 (Mixed) | −0.0025 | 99.09% | −3.9 | Đạt | Trượt riêng lớp F |
| C1 (INT8 calib đỉnh) | −0.0059 | 98.48% | −3.9 | Trượt ở ngưỡng hiệu chỉnh | Trượt |
| C0 (INT8 mặc định) | −0.0496 | 97.91% | +0.5 | Trượt (Spec 84.64%) | Trượt |
| C2b (exp −5) | −0.0227 | 96.69% | −9.3 | Đạt | Trượt |

### C2.4 Chọn ứng viên cho firmware
1. **Bắt buộc mang theo:** B và C_W8A16 (an toàn về độ chính xác).
2. **Ứng viên nhanh:** C3 và C1. Mang theo cả hai **ngay cả khi trượt** Gate H2, nhưng gắn nhãn "độ chính xác chưa đạt Gate H2", vì chỉ đo được tốc độ thật trên board. Quyết định cuối ở bước G.
3. **Loại:** C0, C2a, C2b khỏi firmware.
4. Nếu C2.2 cho thấy chênh lệch lớp F giữa C3/C1 và FP32 nằm trong ±1 độ lệch chuẩn giữa các seed: bỏ nhãn "chưa đạt", coi như đạt.
5. Cập nhật `esp32s3/results/host_eval_report.md` thành bản 3 seed (ghi rõ các hiệu chỉnh), thêm `shortlist.json` liệt kê cấu hình mang theo.

**Gate H2:** đạt khi có ít nhất 2 cấu hình đạt toàn bộ tiêu chí (kỳ vọng B và C_W8A16). Nếu chưa đạt, dừng lại, xem lại calib hoặc cân nhắc W8A16 mixed.

---

## Giai đoạn F0. Thử QEMU sớm (fail-fast)

**Mục đích:** biết ngay QEMU có chạy được kernel ESP-DL INT8/INT16 của ESP32-S3 không, trước khi đầu tư D, E.

### F0.1 Chuẩn bị
1. T-IDF: kiểm tra `idf.py --version` ra `v5.5.x`.
2. Kiểm tra `qemu-system-xtensa --version` ra `9.2.2`. Nếu không thấy, nạp lại môi trường bằng `idf_cmd_init.bat`.
3. Tạo thư mục `esp32s3/f0_smoke/`.

### F0.2 Dựng dự án thử
1. Lấy ví dụ nhỏ của `espressif/esp-dl` (ví dụ `hello_world`; tên chính xác kiểm tra bằng README của component hoặc `idf.py create-project-from-example "espressif/esp-dl:<tên>"`).
2. `idf.py set-target esp32s3`, không bật PSRAM.
3. `idf.py build`.

### F0.3 Chạy QEMU
- Cách 1: `idf.py qemu monitor`.
- Cách 2 (nếu IDF không hỗ trợ target này):
```
python -m esptool --chip esp32s3 merge_bin --fill-flash-size 16MB -o build\qemu_flash.bin @build\flasher_args.json
qemu-system-xtensa -nographic -machine esp32s3 -drive file=build\qemu_flash.bin,if=mtd,format=raw
```
- Thoát: `Ctrl+A` rồi `X`; thoát monitor: `Ctrl+]`.

### F0.4 Đọc kết quả

| Quan sát | Kết luận | Hành động |
|---|---|---|
| Chạy xong, in kết quả suy luận | QEMU chạy được kernel PIE | Tiếp tục D, E, F như kế hoạch |
| `Illegal instruction` / panic trong kernel conv | QEMU không mô phỏng đủ lệnh PIE | Chuyển sang **Plan B** bên dưới |
| Build lỗi, không liên quan PIE | Lỗi cấu hình | Sửa rồi thử lại, tối đa 2 lần |

### Plan B (nếu QEMU không chạy được PIE)
1. FP32 (mã C thuần, giai đoạn D) vẫn chạy trên QEMU để kiểm chứng khung chạy.
2. Thử sinh `.espdl` bằng `target="c"` của `esp-ppq` (bản đã cài có hỗ trợ nền tảng `ESPDL_C_INT8`, `ESPDL_C_INT16`) rồi chạy trên QEMU ESP32-S3. Cần xác minh ESP-DL có chọn kernel C dựa trên nền tảng của file model hay theo cấu hình build.
3. Nếu vẫn không được, dùng QEMU `esp32c3` (cần cài `qemu-riscv32`, khoảng 200 MB) cho phần kiểm chứng logic; kernel PIE kiểm chứng trên board thật.
4. Ghi rõ vào báo cáo: QEMU chỉ kiểm chứng pipeline, không kiểm chứng kernel PIE.

**Gate Q0:** ghi lại kết quả F0 (chạy được / crash) vào `esp32s3/results/qemu_smoke.md`. Không có tiêu chí "đạt/trượt", chỉ quyết định hướng đi.

---

## Giai đoạn D. Mã C FP32 và bộ test vector

### D1. Sinh mã C FP32
1. Tạo `esp32s3/c_fp32/gen_c.py`.
2. Đọc ONNX `export/cnn1d_binary_dsB_kd_fp32.onnx` (đã gộp BatchNorm). Duyệt node theo thứ tự: `Conv` (thường, depthwise), `Relu`, `MaxPool` (k=2, s=2), `AveragePool` (k=32), `Flatten`, `Gemm`.
3. Xuất `model_fp32.h`: trọng số và bias dạng `const float[]` (flash, khoảng 113 KB).
4. Xuất `model_fp32.c` với các hàm `conv1d`, `dwconv1d`, `relu`, `maxpool`, `avgpool`, `gemm`. Padding theo thuộc tính `pads` của từng node. Hai bộ đệm ping-pong tĩnh, mỗi bộ 32×260 float (≈ 33 KB), tổng ≈ 66.5 KB trong SRAM.
5. Kiểm tra trên PC: biên dịch bằng gcc (MSYS2 `C:\msys64\ucrt64\bin`), chạy trên 500 nhịp `host_val_test500_x.npy`, so với ONNXRuntime.

**Gate D1:** sai số logit tối đa < 1e-4 so với ONNXRuntime trên 500 nhịp. Nếu sai: kiểm tra padding và thứ tự trục, đây là hai lỗi hay gặp nhất.

### D2. Bộ test vector mới (thay bộ 50 nhịp ngẫu nhiên)
**Vì sao đổi:** các cấu hình khớp nhau 98–99.9%, nên 50 nhịp ngẫu nhiên thường chỉ cho 0–1 nhịp khác biệt, không đủ để phát hiện lỗi triển khai.

1. Tạo `esp32s3/quant/make_vectors.py`.
2. Chọn từ TEST (đây là bước kiểm chứng firmware, không dùng để chọn ngưỡng hay đánh giá), tổng **50 nhịp**:
   - 20 nhịp mà ít nhất một cấu hình ứng viên bất đồng nhãn với FP32 (lấy từ logit đã tính ở C2).
   - 10 nhịp lớp F.
   - 10 nhịp lớp V có biên độ đỉnh cao (top phân vị của |x|).
   - 10 nhịp lớp N ngẫu nhiên (seed cố định).
3. Với mỗi nhịp, lưu: dữ liệu float (260 mẫu), nhãn 5 lớp và nhị phân, logit FP32 (ONNX), logit mô phỏng ESP-PPQ cho từng cấu hình ứng viên, và logit nguyên (INT) tương ứng nếu trích được.
4. Xuất `esp32s3/ecg_qemu/main/test_vectors.h`.
5. Lưu thêm `esp32s3/quant/test_vectors_meta.json`: chỉ số nhịp trong TEST, lớp, lý do được chọn.
6. Xuất file ngưỡng `thresholds.h`: ngưỡng chọn từ VAL cho từng cấu hình (từ C2), dạng ngưỡng trên **chênh logit** `logit₁ − logit₀ > ln(th/(1−th))`, không dùng softmax trên chip.

**Gate D:** (1) D1 đạt; (2) `test_vectors.h` biên dịch được; (3) trong 20 nhịp "bất đồng", mỗi nhịp có ít nhất một cấu hình khác FP32 (kiểm tra bằng script).

---

## Giai đoạn E. Firmware ESP-IDF nhiều chế độ

### E1. Dựng dự án
```
cd esp32s3\ecg_qemu
idf.py set-target esp32s3
idf.py add-dependency "espressif/esp-dl^3"
```

### E2. Cấu hình (`sdkconfig.defaults`)
- Flash 16 MB, `CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384`.
- **Chưa bật PSRAM.** Model nhỏ, giảm rủi ro QEMU.
- Tối ưu hóa biên dịch ở chế độ hiệu năng (`CONFIG_COMPILER_OPTIMIZATION_PERF=y`). Mặc định là mức debug sẽ làm mốc thời gian FP32 C kém hơn thực tế khi so với kernel đã tối ưu.
- **Bảng phân vùng:** tổng dung lượng app dự kiến (bốn `.espdl` ≈ 200 KB + trọng số FP32 ≈ 113 KB + test vector ≈ 52 KB + thư viện ESP-DL) có thể vượt phân vùng mặc định 1 MB. Chọn `CONFIG_PARTITION_TABLE_SINGLE_APP_LARGE`, hoặc tạo bảng phân vùng riêng. Kiểm tra bằng `idf.py size` sau khi build.

### E3. Nhúng model
1. Copy các `.espdl` ứng viên (từ `shortlist.json`: B, C_W8A16, C3, C1) vào `main/`.
2. `main/CMakeLists.txt`: `idf_component_register(SRCS "main.cpp" "model_fp32.c" INCLUDE_DIRS "." EMBED_FILES "b_int16.espdl" "c_w8a16.espdl" "c3.espdl" "c1.espdl")`.
3. Thêm `model_fp32.c/.h`, `test_vectors.h`, `thresholds.h` vào `main/`.

### E4. Viết `main.cpp`
API theo ESP-DL v3, cần kiểm chứng với README của component sau khi tải.

Với mỗi chế độ trong `{fp32, b_int16, c_w8a16, c3, c1}` và mỗi nhịp trong 50 nhịp:
1. **Nạp model** (một lần mỗi chế độ): `dl::Model` từ vùng flash nhúng; lấy tensor đầu vào, đầu ra.
2. **Lượng tử hóa đầu vào:** `x_int = clamp(round(x / 2^exp_in))` về dải 8-bit hoặc 16-bit; đọc `exp_in` từ tensor, không gõ cứng. Ưu tiên hàm quantize có sẵn của ESP-DL nếu có.
3. **Chạy** `model->run()`; đo thời gian bằng `esp_timer_get_time()` (số liệu QEMU **không** dùng, ghi rõ).
4. **Giải lượng tử đầu ra:** `y = y_int × 2^exp_out`.
5. **Quyết định:** so `logit₁ − logit₀` với ngưỡng của đúng cấu hình trong `thresholds.h`.
6. **So sánh:**
   - `match_ppq`: logit chip so với logit mô phỏng ESP-PPQ (kỳ vọng bit-exact, sai số tối đa 1 LSB `2^exp_out`).
   - `agree_fp32`: nhãn so với FP32.
7. In một dòng mỗi chế độ:
   `MODE=<..> match_ppq=<n>/50 max_lsb_diff=<x> agree_fp32=<n>/50 avg_dt_us=<x> heap_internal=<x>`
8. Riêng FP32: in `max_logit_diff` so với logit FP32 (ONNX).

### E5. Build
`idf.py build`. Sau khi build, chạy `idf.py size` để kiểm tra mức dùng flash và RAM.

**Lỗi thường gặp:** thiếu header → tìm tên đúng trong `managed_components\espressif__esp-dl\`; tràn stack → tăng stack main; tràn phân vùng app → xem E2.

---

## Giai đoạn F. Chạy QEMU đầy đủ

1. Chạy theo cách đã dùng ở F0 (`idf.py qemu monitor` hoặc `qemu-system-xtensa` thủ công).
2. Thu log ra file: `esp32s3/results/qemu_run.log`.
3. Ghi lại cho từng chế độ: chạy xong / crash (kèm loại lỗi).

**Gate Q:**
- Chế độ FP32: `max_logit_diff < 1e-3` so với ONNX.
- Mỗi chế độ INT chạy được trên QEMU: `match_ppq ≥ 49/50` và `max_lsb_diff ≤ 1`.
- Chế độ INT không chạy được trên QEMU: ghi "chuyển sang kiểm chứng trên board", **không** tính là lỗi.
- Không gán `avg_dt_us` của QEMU cho thời gian thật.

Nếu F0 đã chuyển sang Plan B, thực hiện theo Plan B thay vì tiêu chí INT ở trên.

---

## Giai đoạn G. Báo cáo, chuẩn bị đo board, bước tiếp theo

### G1. Báo cáo
`reports/esp32s3_fp32_int16_int8.md` gồm:
- Bảng Gate H2 (3 seed) và lý do chọn ứng viên.
- Bảng chế độ × `{AUROC, PR-AUC, Recall, Spec, recall S/V/F, khớp nhãn, kích thước .espdl, kích thước firmware, trạng thái QEMU}`.
- Cập nhật kích thước thực tế: `.espdl` INT8 là 42 KB, INT16 là 72 KB (cao hơn ước tính 32/59 KB ở Giai đoạn A, do thêm metadata và 20/40-bit bias).
- Nhãn nguồn số liệu: **[tính từ ONNX]**, **[ước tính]**, **[host/ESP-PPQ]**, **[QEMU]**, **[board thật — chưa đo]**.

### G2. Chuẩn bị đo trên board thật
- Cùng firmware, đọc `avg_dt_us` của 5 chế độ, so với khoảng ước tính ở A5 (INT8 1.3–2.6 ms, INT16 2.6–5.2 ms, FP32 12.5–25 ms).
- Lưu ý riêng cho W8A16: kích hoạt 16-bit nên tốc độ có thể gần INT16 hơn INT8; đây là lý do đo trực tiếp.
- Quan sát: số kênh có là bội của 16/8 (căn lề PIE); weights đặt ở flash hay chép sang SRAM; buffer ở SRAM nội hay PSRAM.

### G3. Quy tắc chọn cấu hình cuối (sau khi có số đo board)
1. Nếu chênh lệch thời gian giữa C_W8A16 (hoặc B) và C3/C1 nhỏ hơn 20%: chọn C_W8A16 hoặc B, ưu tiên độ chính xác.
2. Nếu C3/C1 nhanh hơn rõ rệt: chọn C3 nếu Gate H2 3 seed chấp nhận mức giảm recall F; nếu không, chọn C_W8A16.
3. Cấu hình INT8 thuần C1 chỉ được chọn nếu đạt Gate H2 ở cả ngưỡng cố định và ngưỡng hiệu chỉnh.

### G4. Việc cần làm sau plan này (ngoài phạm vi)
- **Tiền xử lý trên chip:** cắt cửa sổ 260 mẫu quanh đỉnh R, z-score từng nhịp (mean 0, std 1) ngay trên chip. Biên độ ADC thực chỉ ảnh hưởng qua bước này; phép thử α ở C đã mô phỏng sau z-score nên không bao phủ rủi ro này.
- Lọc nhiễu, phát hiện đỉnh R, lấy mẫu ADC.
- Tối ưu độ trễ sâu hơn, QAT nếu cần cải thiện lớp F.

---

## Rủi ro và cách xử lý

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Exponent giữa `.espdl` và đồ thị đánh giá lệch | `consistency_check.json` có tensor lệch | Đánh giá lại trên CPU (C2.1) |
| Lớp F tụt ở C1/C3 khi chạy 3 seed | Δ recall F vượt ±1 độ lệch chuẩn | Giữ W8A16 và B làm cấu hình chính; C3/C1 chỉ là ứng viên tốc độ |
| QEMU không chạy PIE | `Illegal instruction` ở F0 | Plan B (F0) |
| API ESP-DL khác bản v3.x | Lỗi biên dịch | Đối chiếu README và ví dụ trong `managed_components` |
| Firmware tràn phân vùng | Lỗi link hoặc `idf.py size` báo vượt | Dùng `SINGLE_APP_LARGE` hoặc bảng phân vùng riêng (E2) |
| Thời gian FP32 C bị sai lệch vì build debug | Số đo FP32 chậm bất thường | Bật tối ưu hiệu năng (E2) |

## Tổng thời gian
Khoảng **4–4.5 giờ** cho C2, F0, D, E, F, G (chưa tính xử lý lỗi). Nếu cần rút gọn, thứ tự cắt giảm: bỏ C1 khỏi firmware → chỉ chạy B, C_W8A16, C3 và FP32.
