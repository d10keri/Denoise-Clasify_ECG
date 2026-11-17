# Báo cáo Giai đoạn F0 — Smoke Test ESP-DL trên QEMU ESP32-S3 (Gate Q0)

**Ngày thực hiện:** 2026-11-17  
**Mục tiêu:** Kiểm tra khả năng chạy `espressif/esp-dl` (v3.3.15) và tập lệnh PIE (TIE728) của ESP32-S3 trên `qemu-system-xtensa` (`esp_develop_9.2.2_20250817`), phát hiện sớm giới hạn giả lập trước khi xây dựng firmware Giai đoạn E/F.  
**Dự án thử nghiệm:** [esp32s3/f0_smoke/main/main.cpp](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/f0_smoke/main/main.cpp) | **Log chạy:** [esp32s3/results/f0_qemu.log](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/f0_qemu.log) | **Script chạy:** [scripts/esp32s3/run_f0_smoke.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/scripts/esp32s3/run_f0_smoke.py)

---

## 1. Tóm tắt kết quả (Gate Q0)

> **Kết luận hướng đi:** **Hybrid PIE trên QEMU** (`CONFIG_PIE_V1_BOOST = 1` cho toàn bộ 9 lớp `Conv`/`DepthwiseConv`/`PointwiseConv` + 3 lớp `MaxPool`, kết hợp 2 cờ workaround cho QEMU tại `AveragePool` INT8 và `Gemm` `n_remainder`), sử dụng **trực tiếp** 4 file `.espdl` mục tiêu `esp32s3` đã xuất ở Giai đoạn B. **Không cần** chuyển sang `esp32c3` hay xuất lại model `target="c"`. Khi nạp lên **board ESP32-S3 thật**, chỉ cần tắt 2 macro workaround (`DL_QEMU_AVOID_ST_QACC`, `DL_QEMU_GEMM_SCALAR_FALLBACK`) để toàn bộ đồ thị chạy PIE 100%.

| Mô hình shortlist (`target="esp32s3"`) | Kích thước `.espdl` | Kết quả `model->test()` | Logit đầu ra (INT) | Dequant `logits` | `dt_qemu` (us, không dùng làm latency thật) |
|---|---|---|---|---|---|
| `B` (`b_int16.espdl`, W16A16) | 74,128 bytes (72.39 KB) | **PASS** (`err=0`) | `int16=[18712, -24134]` (`exp=-12`) | `[4.56836, -5.89209]` | 25,834 |
| `C_W8A16` (`c_w8a16.espdl`, W8A16) | 47,056 bytes (45.95 KB) | **PASS** (`err=0`) | `int16=[17386, -21610]` (`exp=-12`) | `[4.24463, -5.27588]` | 13,731 |
| `C3` (`c3.espdl`, Mixed INT16+INT8) | 45,184 bytes (44.12 KB) | **PASS** (`err=0`) | `int16=[14065, -18948]` (`exp=-12`) | `[3.43384, -4.62598]` | 13,531 |
| `C1` (`c1.espdl`, INT8 Peak-Calib) | 43,104 bytes (42.09 KB) | **PASS** (`err=0`) | `int8=[127, -128]` (`exp=-5`) | `[3.96875, -4.00000]` | 7,805 |

Cả **4/4 mô hình ứng viên** đều khớp bit-exact với test vector ground-truth bên trong file `.espdl` (`Test Pass!`).

---

## 2. Các vấn đề kỹ thuật phát hiện trên QEMU ESP32-S3 & cách khắc phục

Trong quá trình chạy thực nghiệm trên `ESP-IDF v5.5.2` + `espressif/esp-dl v3.3.15` + `qemu-xtensa esp_develop_9.2.2_20250817`, đã xác định được **4 vấn đề cụ thể**:

### (1) Crash lúc khởi động trước `app_main`: `Coprocessor exception` (FPU CP0)
- **Nguyên nhân:** `esp-dl` định nghĩa biến C++ toàn cục `AttentionFloatExpLut attention_float_exp_lut_data{}` trong `dl/base/dl_base_attention_common.hpp`, có constructor thực thi lệnh dấu phẩy động (`mul.s`, Coprocessor 0) ngay trong `do_global_ctors()`, trước khi FreeRTOS bật `CPENABLE`. Lưu ý `__attribute__((constructor(101)))` trong `main.cpp` không giải quyết được vì thứ tự link của `libespressif__esp-dl.a` đứng trước `libmain.a`.
- **Khắc phục (bắt buộc cho cả QEMU và firmware Giai đoạn E):** Đăng ký hàm khởi tạo hệ thống mức sớm bằng `ESP_SYSTEM_INIT_FN(enable_coprocessors_before_ctors, CORE, BIT(0), 100)` với `wsr %0, cpenable; rsync` (`0x09`: bật cả FPU CP0 và PIE CP3) trong [main.cpp](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/f0_smoke/main/main.cpp#L9-L14).

### (2) `IllegalInstruction` trong `dl_tie728_s8_avg_pool2d_hwc_sum` (`PC = 0x4037827f`)
- **Quan sát:** Toàn bộ **9 lớp tích chập PIE** (`Conv_0`, 3x `DWConv`, 3x `PWConv` và 3x `MaxPool` — chiếm >99.7% tổng MACCs của mô hình) chạy hoàn toàn bình thường bằng tập lệnh `tie728` trên QEMU. Crash `IllegalInstruction` xảy ra **duy nhất** tại lớp `AveragePool` (`dl_tie728_s8_avg_pool2d.S`), khi gọi `EE.VCMP.EQ.S8` / `EE.ST.QACC_L.L.128.IP` để dump QACC ra bộ đệm.
- **Khắc phục cho QEMU:** Bật `DL_QEMU_AVOID_ST_QACC 1` trong [dl_define_private.hpp](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/f0_smoke/managed_components/espressif__esp-dl/dl/dl_define_private.hpp#L15) để riêng đường dẫn INT8 trong [dl_base_avg_pool2d.cpp](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/f0_smoke/managed_components/espressif__esp-dl/dl/base/dl_base_avg_pool2d.cpp#L107) dùng vòng lặp C `avgpool2d_hwc_sum` (cho kết quả bit-identical với kernel PIE của `esp-dl`).

### (3) Hai lỗi giả lập QEMU đối với thanh ghi tích lũy 40-bit `ACCX` ở nhánh `n_remainder` (`Gemm`)
- Lớp `classifier/Gemm` có `output_channel = 2` (`< 8`), đi vào nhánh vô hướng `n_remainder` (sử dụng thanh ghi 40-bit `ACCX`, chỉ chiếm 256 / 1,497,472 ≈ `0.017%` MACC). Kiểm tra trực tiếp bằng inline-assembly (`probe_qemu_accx`) chứng minh QEMU 9.2.2 mắc 2 lỗi với `ACCX` khi giá trị âm:
  1. `EE.LD.ACCX.IP` nạp bias 40-bit âm (vd `-1435`) vào `accx_1 = 0x000000ff` mà không mở rộng dấu sang 64-bit, khiến `EE.VMULAS.*.ACCX` kế tiếp bão hòa về `INT40_MAX` (`accx_1 = 0x7f, accx_0 = 0xffffffff`).
  2. `EE.SRS.ACCX rd, shift, 0` với `shift >= 8` làm hỏng bit dấu (`-165326 >> 8` trả về `2147483325 = 0x7ffffebd`).
- **Khắc phục cho QEMU:** Bật `DL_QEMU_GEMM_SCALAR_FALLBACK 1` trong [dl_define_private.hpp](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/f0_smoke/managed_components/espressif__esp-dl/dl/dl_define_private.hpp#L16) và xử lý fallback trong [dl_module_gemm.hpp](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/f0_smoke/managed_components/espressif__esp-dl/dl/module/include/dl_module_gemm.hpp#L118-L162) khi `output_height == 1 && output_width == 1 && output_channel < 8 && mac_shift >= 0`.
- Lưu ý: Trên **board thật ESP32-S3**, cả `EE.LD.ACCX.IP` lẫn `EE.SRS.ACCX` và `dl_tie728_s8_avg_pool2d_hwc_sum` đều chạy bằng phần cứng PIE thật; không dùng 2 macro này khi build cho silicon thật.

### (4) Căn chỉnh 16-byte cho dữ liệu FlatBuffers nhúng từ flash (`.rodata`)
- `EMBED_FILES` đặt `.espdl` tại địa chỉ chỉ căn lề 4-byte (`0x3c134f5c`), gây cảnh báo `FbsLoader: The address of fbs model in flash is not aligned with 16 bytes.`
- Trong firmware cần căn lề 16-byte cho buffer hoặc file `.espdl` nhúng (`__attribute__((aligned(16)))` / copy qua buffer `heap_caps_aligned_alloc(16, ...)` khi cần chạy zero-copy tensor PIE).

---

## 3. Kiểm chứng Plan B (đối chiếu để chốt kiến trúc)
- Đã kiểm tra `_SELECT_TARGET` trong `cmake/compile_finalize.py` (`"esp32s3": "pie_v1"` vs `"c"`). Khi chuyển `esp32s3 -> c` + `CONFIG_PIE_V1_BOOST = 0`, firmware biên dịch và chạy hoàn tất không crash trên QEMU, nhưng báo sai giá trị với `.espdl` chuẩn `target="esp32s3"` vì layout trọng số khác (`(N/16)WC16` so với layout C), tức nếu đi hướng Plan B phải duy trì 2 bộ `.espdl` riêng biệt.
- Nhờ 2 workaround nhỏ gọn ở mục 2, toàn bộ 4 mô hình `.espdl` chuẩn phần cứng (`target="esp32s3"`) **vừa kiểm chứng trực tiếp được 9 lớp Conv/MaxPool chạy PIE thực trên QEMU, vừa khớp bit-exact 4/4 đầu ra**.

## 4. Quyết định áp dụng cho Giai đoạn D → E → F
1. **Giữ nguyên 4 file `.espdl` đã lượng tử hóa ở Giai đoạn B (`b_int16`, `c_w8a16`, `c3`, `c1`)** cho cả QEMU và board thật.
2. **Tiếp tục Giai đoạn D** (mã C thuần FP32 + bộ 50 test vector có chủ đích từ TEST + ngưỡng `thresholds.h`).
3. **Trong Giai đoạn E (`esp32s3/ecg_qemu`)**:
   - Thêm `ESP_SYSTEM_INIT_FN(..., 100)` bật `CPENABLE = 0x09` trước global constructors.
   - Đảm bảo 16-byte alignment khi nạp từng `.espdl`.
   - Đóng gói script áp dụng 2 macro QEMU workaround cho `managed_components/espressif__esp-dl` (tách bạch rõ bằng cờ biên dịch để dễ dàng tắt khi nạp board thật).
4. **Trong Giai đoạn F**: Giữ nguyên bộ tiêu chí **Gate Q** chuẩn (`match_ppq >= 49/50` và `max_lsb_diff <= 1` cho cả 4 chế độ INT trên QEMU ESP32-S3, và `max_logit_diff < 1e-3` cho FP32).
