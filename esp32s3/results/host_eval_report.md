# Báo cáo Giai đoạn C & C2: Đánh giá Đa Seed (Seeds 42, 43, 44), Kiểm chứng Nhất quán & Cổng Gate H2

**Mô hình đánh giá:** `M2-B-KD` (`export/cnn1d_binary_dsB_kd_seed{42,43,44}_fp32.onnx`)  
**Tập dữ liệu:** `9,885` nhịp **VAL** và `49,691` nhịp **TEST** (DS2 inter-patient: `N=44,239`, `S=1,837`, `V=3,220`, `F=388`, `Q=7`)  
**Kế hoạch tham chiếu:** [esp32s3_deploy_plan_v3.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/esp32s3_deploy_plan_v3.md)  
**Biểu đồ 3-Seed & Độ nhạy biên độ:** [host_eval_comparison.png](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/host_eval_comparison.png)

---

## 1. Kết quả Bước C2.1: Kiểm tra Nhất quán Exponent & Logit (`CPU` vs `CUDA` vs `.espdl`)

- **File báo cáo JSON:** [consistency_check.json](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/consistency_check.json)
- **Khớp Exponent 100% giữa file `.espdl` (Giai đoạn B) và đồ thị dựng lại trên CPU:** **`True`** (`0` tensor lệch trên cả 6 cấu hình × 56 tensor).
- **Khớp Exponent 100% giữa CPU và CUDA:** **`True`** (`0` tensor lệch).
- **Sai số Logit giữa CPU và CUDA trên 500 nhịp Test:** Bằng **`0.00000000`** tuyệt đối cho toàn bộ các cấu hình INT8 (`c0`, `c1`, `c2b`), và `<= 0.00488` (`~3–6 LSB` của `2^-12`) cho các cấu hình 16-bit (`b_int16`, `c3`, `c_w8a16`).
- **Kết luận C2.1:** Đồ thị đánh giá trên Host hoàn toàn đại diện chính xác cho các file `.espdl` sẽ nhúng vào firmware ESP32-S3.

---

## 2. Kết quả Bước C2.2: Bảng Tổng hợp 3 Seeds (`42, 43, 44` — `Mean ± Std`) trên 49,691 Nhịp TEST

*(Lưu ý: `C2a` đã được loại bỏ khỏi bảng vì `C0` mặc định tự chọn `exponent = -4` nên `C2a` trùng khít 100% với `C0`).*

### Bảng 2A: Chỉ số Không Phụ thuộc Ngưỡng (`PR-AUC`, `AUROC`) & Độ Khớp so với FP32 (3 Seeds Mean ± Std)

| Cấu hình | `.espdl` (KB) | TEST PR-AUC (`Mean ± Std`) | `ΔPR-AUC` vs FP32 | TEST AUROC (`Mean ± Std`) | Cosine vs FP32 | Khớp nhãn FP32 (`@th`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: FP32 (ONNX Ref)** | `117.43` | `0.6777 ± 0.0147` | `Ref` | `0.9077 ± 0.0065` | `1.00000` | `100.00% ± 0.00%` |
| **B (INT16 w16a16)** | `72.39` | `0.6778 ± 0.0148` | `+0.0001 ± 0.0001` | `0.9077 ± 0.0065` | `0.99999` | `99.99% ± 0.01%` |
| **C0 (INT8 Default)** | `42.09` | `0.6676 ± 0.0352` | `-0.0101 ± 0.0280` | `0.9109 ± 0.0061` | `0.98460` | `98.50% ± 0.42%` |
| **C1 (INT8 Peak-Calib)** | `42.09` | `0.6836 ± 0.0181` | `+0.0060 ± 0.0084` | `0.9120 ± 0.0061` | `0.99472` | `98.69% ± 0.17%` |
| **C2b (INT8 exp=-5)** | `42.09` | `0.6736 ± 0.0251` | `-0.0041 ± 0.0139` | `0.9139 ± 0.0057` | `0.97471` | `97.30% ± 0.56%` |
| **C3 (Mixed INT16+INT8)** | `44.12` | `0.6838 ± 0.0166` | `+0.0061 ± 0.0062` | `0.9117 ± 0.0062` | `0.99589` | `99.07% ± 0.14%` |
| **C_W8A16 (W8A16)** | `45.95` | `0.6796 ± 0.0205` | `+0.0019 ± 0.0080` | `0.9080 ± 0.0062` | `0.99950` | `99.47% ± 0.11%` |

### Bảng 2B: Recall, Specificity & Recall Từng Lớp Con (`S, V, F`) tại Ngưỡng Cố định (`Fixed FP32 VAL Youden's J`)

| Cấu hình | TEST Recall (`Mean ± Std`) | TEST Specificity (`Mean ± Std`) | `ΔRec` / `ΔSpec` (Worst Seed vs FP32) | Subclass `S` Recall (`1,837` nhịp) | Subclass `V` Recall (`3,220` nhịp) | Subclass `F` Recall (`388` nhịp) | `ΔF` vs FP32 (Fixed / Recal) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: FP32 (ONNX Ref)** | `83.93% ± 1.65%` | `86.15% ± 1.30%` | `Ref` | `72.67% ± 3.32%` | `93.40% ± 0.98%` | `58.76% ± 3.10%` | `Ref` |
| **B (INT16 w16a16)** | `83.93% ± 1.64%` | `86.15% ± 1.30%` | `-0.02%` | `72.67% ± 3.32%` | `93.39% ± 0.97%` | `58.76% ± 3.10%` | `+0.00% / +0.00%` |
| **C0 (INT8 Default)** | `83.92% ± 1.77%` | `86.20% ± 1.58%` | `-0.55%` | `72.42% ± 3.79%` | `93.51% ± 0.84%` | `58.85% ± 3.24%` | `+0.09% / +0.60%` |
| **C1 (INT8 Peak-Calib)** | `83.46% ± 1.12%` | `86.67% ± 1.20%` | `-1.14%` | `72.26% ± 3.18%` | `93.14% ± 0.67%` | `56.19% ± 2.69%` | `-2.58% / -0.34%` |
| **C2b (INT8 exp=-5)** | `82.92% ± 1.73%` | `87.03% ± 1.08%` | `-1.21%` | `70.39% ± 3.97%` | `93.57% ± 0.92%` | `53.78% ± 0.85%` | `-4.98% / -2.41%` |
| **C3 (Mixed INT16+INT8)** | `83.71% ± 1.65%` | `86.37% ± 1.13%` | `-0.42%` | `72.71% ± 3.28%` | `93.26% ± 1.04%` | `56.62% ± 3.04%` | `-2.15% / -1.46%` |
| **C_W8A16 (W8A16)** | `83.88% ± 1.34%` | `86.39% ± 1.29%` | `-0.39%` | `72.65% ± 3.21%` | `93.44% ± 0.72%` | `57.82% ± 2.71%` | `-0.95% / -0.43%` |

### Bảng 2C: Kiểm tra Ngưỡng Cố định vs Ngưỡng Hiệu chỉnh VAL trên Checkpoint Triển khai (Seed 42) & 3-Seed Mean

| Cấu hình | Seed 42 (`@Fixed FP32 VAL th=0.78`) | Seed 42 (`@Recalibrated VAL th`) | Đạt `≥85%/85%` ở CẢ 2 Ngưỡng? | 3-Seed Mean (`@Recalibrated VAL YJ`) |
| :--- | :---: | :---: | :---: | :---: |
| **A: FP32 (ONNX Ref)** | `86.15%` / `85.12%` | `86.15%` / `85.12%` (`th=0.78`) | **PASS** | `83.93%` / `86.15%` |
| **B (INT16 w16a16)** | `86.13%` / `85.13%` | `86.13%` / `85.13%` (`th=0.78`) | **PASS** | `83.93%` / `86.15%` |
| **C0 (INT8 Default)** | `86.41%` / `84.64%` | `86.76%` / `84.16%` (`th=0.77`) | **FAIL** | `84.21%` / `85.89%` |
| **C1 (INT8 Peak-Calib)** | `85.01%` / `85.88%` | `86.70%` / `83.93%` (`th=0.73`) | **FAIL** | `84.45%` / `85.67%` |
| **C2b (INT8 exp=-5)** | `85.29%` / `86.02%` | `86.12%` / `85.03%` (`th=0.76`) | **PASS** | `84.56%` / `85.37%` |
| **C3 (Mixed INT16+INT8)** | `85.80%` / `85.26%` | `85.80%` / `85.26%` (`th=0.78`) | **PASS** | `84.17%` / `86.05%` |
| **C_W8A16 (W8A16)** | `85.77%` / `85.47%` | `86.08%` / `85.09%` (`th=0.77`) | **PASS** | `84.12%` / `86.14%` |

---

## 3. Kết quả Bước C2.3: Đánh giá Cổng Gate H2 (5 Tiêu chí Nghiêm ngặt trên 3 Seeds)

- **Lưu ý phương pháp:** Trên tập TEST, bản thân mô hình gốc **FP32** có độ lệch chuẩn giữa 3 seeds ở lớp hiếm `F` (`388` nhịp) là **`±3.1%`** (`58.76% ± 3.10%`: Seed 42 = `62.89%`, Seed 43 = `57.99%`, Seed 44 = `55.41%`).
- Theo đúng Quy tắc **C2.4.4** trong [esp32s3_deploy_plan_v3.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/esp32s3_deploy_plan_v3.md): Nếu mức giảm lớp `F` nằm trong phạm vi `±1 độ lệch chuẩn giữa các seed` (`±3.10%`) và đạt các tiêu chí còn lại, cấu hình được xác nhận đạt chuẩn triển khai.

| Cấu hình | 1. `|ΔPR-AUC| ≤ 0.010` | 2. Khớp nhãn `≥ 99.0%` | 3. `Δ(S,V,F) ≥ -2.0%` (`|ΔF| ≤ 1 Std_FP32`?) | 4. `Rec/Spec ≥ 85%` (Cả 2 ngưỡng) | 5. Worst Seed Drop `≤ 0.5%` vs FP32 | Kết luận **Gate H2** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B (INT16 w16a16)** | `+0.0001` (**PASS**) | `99.99%` (**PASS**) | `F:+0.00%` / `+0.00%` (**PASS**) | **PASS** | `-0.02%` (**PASS**) | **PASS (5/5 Tuyệt đối)** |
| **C0 (INT8 Default)** | `-0.0101` (**FAIL**) | `98.50%` (**FAIL**) | `F:+0.09%` / `+0.60%` (**PASS**) | **FAIL** | `-0.55%` (**FAIL**) | **FAIL** |
| **C1 (INT8 Peak-Calib)** | `+0.0060` (**PASS**) | `98.69%` (**FAIL**) | `F:-2.58%` / `-0.34%` (**PASS (≤1 Std)**) | **FAIL** | `-1.14%` (**FAIL**) | **FAIL** |
| **C2b (INT8 exp=-5)** | `-0.0041` (**PASS**) | `97.30%` (**FAIL**) | `F:-4.98%` / `-2.41%` (**FAIL**) | **PASS** | `-1.21%` (**FAIL**) | **FAIL** |
| **C3 (Mixed INT16+INT8)** | `+0.0061` (**PASS**) | `99.07%` (**PASS**) | `F:-2.15%` / `-1.46%` (**PASS (≤1 Std)**) | **PASS** | `-0.42%` (**PASS**) | **PASS (Theo C2.4.4: |ΔF| ≤ 1 Std)** |
| **C_W8A16 (W8A16)** | `+0.0019` (**PASS**) | `99.47%` (**PASS**) | `F:-0.95%` / `-0.43%` (**PASS**) | **PASS** | `-0.39%` (**PASS**) | **PASS (5/5 Tuyệt đối)** |

---

## 4. Kết quả Bước C2.4: Danh sách Ứng viên Đưa vào Firmware (`shortlist.json`)

Chi tiết được lưu tại [shortlist.json](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/shortlist.json):
- **`b_int16`** (`esp32s3/quant/b_int16.espdl`): Primary Accuracy Reference (w16a16, 8-lane PIE SIMD) — Trạng thái: **`PASS (5/5 strict criteria)`**
- **`c_w8a16`** (`esp32s3/quant/c_w8a16.espdl`): Compact Accuracy Reference (w8a16: 8-bit weights, 16-bit activations) — Trạng thái: **`PASS (5/5 strict criteria)`**
- **`c3`** (`esp32s3/quant/c3.espdl`): Primary Speed + Wave-Resolution Candidate (Mixed INT16 Stem+Gemm, INT8 DS blocks) — Trạng thái: **`PASS (via Rule C2.4.4: ΔF = -2.15% fixed / -1.46% recal is within ±1 inter-seed std = ±3.10%)`**
- **`c1`** (`esp32s3/quant/c1.espdl`): Pure INT8 Speed Candidate (w8a8 with peak-enriched calibration) — Trạng thái: **`SPEED_CANDIDATE (Độ chính xác chưa đạt Gate H2: Agreement=98.69% < 99%, Recal Spec=83.93% < 85%)`**
- **Các cấu hình loại khỏi Firmware:** `['c0', 'c2a', 'c2b']`
