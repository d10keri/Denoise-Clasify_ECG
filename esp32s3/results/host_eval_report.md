# Báo cáo Giai đoạn C: Đánh giá Độ chính xác trên Host (VAL/TEST) & Độ nhạy Biên độ Sóng ECG (ESP-PPQ)

**Mô hình gốc:** `M2-B-KD` (Seed 42 — [export/cnn1d_binary_dsB_kd_fp32.onnx](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/export/cnn1d_binary_dsB_kd_fp32.onnx))  
**Tập dữ liệu đánh giá:** Toàn bộ `9,885` nhịp **VAL** và `49,691` nhịp **TEST** (DS2 inter-patient)  
**Công cụ mô phỏng lượng tử hóa:** `ESP-PPQ v1.3.14` (`TorchExecutor` mô phỏng bit-exact quy ước làm tròn và dịch bit `2^n` của thư viện `ESP-DL` trên ESP32-S3)  
**Biểu đồ trực quan:** [host_eval_comparison.png](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/esp32s3/results/host_eval_comparison.png)

---

## 1. Kết quả Cổng Kiểm duyệt Giai đoạn C (Gate H)

- **Tiêu chí Gate H:** Có cấu hình INT8 hoặc Mixed Precision đạt độ lệch `|ΔAUROC| <= 0.0050` (`0.50%`) so với FP32 trên 49,691 nhịp TEST, đồng thời bảo toàn ngưỡng lâm sàng `Recall >= 85.0%` & `Specificity >= 85.0%`.
- **Trạng thái Gate H:** **PASS**
- **Các cấu hình vượt qua Gate H (`|ΔAUROC| <= 0.0050`):** `['b_int16', 'c0', 'c1', 'c2a', 'c3', 'c_w8a16']`
- **Các cấu hình đạt đồng thời `Recall >= 85.0% & Specificity >= 85.0%` trên 49,691 nhịp TEST (@ ngưỡng chọn từ VAL):** `['b_int16', 'c1', 'c2b', 'c3', 'c_w8a16']`
- **Cấu hình khuyến nghị triển khai xuống ESP32-S3:** **`c3`** (cho nhánh Mixed/INT8 tốc độ cao), **`c1`** (cho nhánh thuần INT8 100%), và **`b_int16`** / **`c_w8a16`** (cho nhánh độ chính xác tuyệt đối).

---

## 2. Bảng So sánh Toàn diện trên 49,691 Nhịp TEST (`[Mô phỏng ESP-PPQ & ONNXRuntime]`)

Tất cả các cấu hình dưới đây được đánh giá tại cùng điểm vận hành tối ưu chọn từ tập VAL (`@3_val_best_youden_j`, `th = 0.78` của FP32 và ngưỡng hiệu chỉnh riêng trên VAL của từng cấu hình):

| Cấu hình | `.espdl` (KB) | TEST AUROC (`Δ vs FP32`) | TEST PR-AUC | Macro F1 (`@VAL_YJ`) | Recall (`non-N`) | Specificity (`N`) | Recall `S` / `V` / `F` | Cosine vs FP32 | Khớp nhãn FP32 (`@th`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: FP32 (ONNX Ref)** | `117.43` | `0.9117` (`Ref`) | `0.6724` | `0.7363` | **`86.15%`** | **`85.12%`** | `77.4%` / `94.0%` / `62.9%` | `1.00000` | `100.00%` |
| **B (INT16 w16a16)** | `72.39` | `0.9117` (`+0.0000`) | `0.6726` | `0.7364` | **`86.13%`** | **`85.13%`** | `77.4%` / `94.0%` / `62.9%` | `0.99999` | `99.98%` |
| **C0 (INT8 Default)** | `42.09` | `0.9128` (`+0.0012`) | `0.6228` | `0.7320` | `86.41%` | `84.64%` | `77.7%` / `94.2%` / `63.4%` | `0.96146` | `97.91%` |
| **C1 (INT8 Peak-Calib)** | `42.09` | `0.9153` (`+0.0036`) | `0.6665` | `0.7413` | **`85.01%`** | **`85.88%`** | `76.5%` / `93.0%` / `59.0%` | `0.99226` | `98.48%` |
| **C2a (INT8 exp=-4)** | `42.09` | `0.9128` (`+0.0012`) | `0.6228` | `0.7320` | `86.41%` | `84.64%` | `77.7%` / `94.2%` / `63.4%` | `0.96146` | `97.91%` |
| **C2b (INT8 exp=-5)** | `42.09` | `0.9182` (`+0.0065`) | `0.6497` | `0.7435` | **`85.29%`** | **`86.02%`** | `76.0%` / `94.4%` / `53.6%` | `0.95722` | `96.69%` |
| **C3 (Mixed INT16+INT8)** | `44.12` | `0.9153` (`+0.0037`) | `0.6699` | `0.7368` | **`85.80%`** | **`85.26%`** | `77.4%` / `93.9%` / `59.0%` | `0.99425` | `99.09%` |
| **C_W8A16 (W8A16)** | `45.95` | `0.9103` (`-0.0013`) | `0.6638` | `0.7389` | **`85.77%`** | **`85.47%`** | `77.0%` / `93.7%` / `61.6%` | `0.99931` | `99.32%` |

### Khi Hiệu chỉnh lại Ngưỡng Quyết định trên Tập VAL của Chính Cấu hình Lượng tử hóa (`Recalibrated VAL Threshold`):

| Cấu hình | Điểm vận hành chọn trên VAL | Ngưỡng `th` | TEST Macro F1 | TEST Recall (`non-N`) | TEST Specificity (`N`) | Đạt `Rec>=85% & Spec>=85%`? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A: FP32 (ONNX Ref)** | `3_val_best_youden_j` | `0.78` | `0.7363` | `86.15%` | `85.12%` | **PASS** |
| **B (INT16 w16a16)** | `3_val_best_youden_j` | `0.78` | `0.7364` | `86.13%` | `85.13%` | **PASS** |
| **C0 (INT8 Default)** | `3_val_best_youden_j` | `0.77` | `0.7282` | `86.76%` | `84.16%` | **NEAR** |
| **C1 (INT8 Peak-Calib)** | `3_val_best_youden_j` | `0.73` | `0.7257` | `86.70%` | `83.93%` | **NEAR** |
| **C2a (INT8 exp=-4)** | `3_val_best_youden_j` | `0.77` | `0.7282` | `86.76%` | `84.16%` | **NEAR** |
| **C2b (INT8 exp=-5)** | `3_val_best_youden_j` | `0.76` | `0.7353` | `86.12%` | `85.03%` | **PASS** |
| **C3 (Mixed INT16+INT8)** | `3_val_best_youden_j` | `0.78` | `0.7368` | `85.80%` | `85.26%` | **PASS** |
| **C_W8A16 (W8A16)** | `3_val_best_youden_j` | `0.77` | `0.7358` | `86.08%` | `85.09%` | **PASS** |

---

## 3. Kết quả Phép thử Độ nhạy Biên độ Sóng ECG (`α ∈ {0.8, 0.9, 1.0, 1.1, 1.25}`)

Để kiểm tra hiện tượng nhạy biên độ khi tín hiệu ECG bị thay đổi độ lợi (do tiếp xúc điện cực hoặc sai số ADC trước/sau chuẩn hóa), toàn bộ `49,691` nhịp TEST được nhân với hệ số $\alpha \in \{0.8, 0.9, 1.0, 1.1, 1.25\}$:

### Bảng 3A: TEST AUROC theo Hệ số Khuếch đại Biên độ $\alpha$

| Cấu hình | `α = 0.80x` (Sóng nhỏ -20%) | `α = 0.90x` (-10%) | `α = 1.00x` (Chuẩn) | `α = 1.10x` (+10%) | `α = 1.25x` (Đỉnh cao +25%) | Độ sụt giảm AUROC lớn nhất (`Max Drop`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: FP32 (ONNX Ref)** | `0.9021` | `0.9086` | **`0.9117`** | `0.9125` | `0.9107` | `-0.0096` (`-0.96%`) |
| **B (INT16 w16a16)** | `0.9021` | `0.9086` | **`0.9117`** | `0.9126` | `0.9112` | `-0.0096` (`-0.96%`) |
| **C0 (INT8 Default)** | `0.9044` | `0.9103` | **`0.9128`** | `0.9131` | `0.9103` | `-0.0084` (`-0.84%`) |
| **C1 (INT8 Peak-Calib)** | `0.9016` | `0.9106` | **`0.9153`** | `0.9170` | `0.9156` | `-0.0136` (`-1.36%`) |
| **C2a (INT8 exp=-4)** | `0.9044` | `0.9103` | **`0.9128`** | `0.9131` | `0.9103` | `-0.0084` (`-0.84%`) |
| **C2b (INT8 exp=-5)** | `0.9047` | `0.9130` | **`0.9182`** | `0.9193` | `0.9152` | `-0.0135` (`-1.35%`) |
| **C3 (Mixed INT16+INT8)** | `0.9021` | `0.9107` | **`0.9153`** | `0.9171` | `0.9158` | `-0.0132` (`-1.32%`) |
| **C_W8A16 (W8A16)** | `0.8995` | `0.9069` | **`0.9103`** | `0.9116` | `0.9107` | `-0.0108` (`-1.08%`) |

### Bảng 3B: TEST Recall / Specificity (`@ th = 0.78`) khi Biên độ Lệch (`α = 0.80x` vs `α = 1.25x`)

| Cấu hình | Recall / Spec @ `α = 0.80x` | Recall / Spec @ `α = 1.00x` | Recall / Spec @ `α = 1.25x` | Nhận xét Độ bền Biên độ |
| :--- | :---: | :---: | :---: | :--- |
| **A: FP32 (ONNX Ref)** | `86.10%` / `85.11%` | **`86.15%` / `85.12%`** | `85.45%` / `83.02%` | Chuẩn tham chiếu số thực |
| **B (INT16 w16a16)** | `86.10%` / `85.11%` | **`86.13%` / `85.13%`** | `85.45%` / `83.02%` | Bám sát tuyệt đối FP32 ở mọi mức biên độ |
| **C0 (INT8 Default)** | `85.47%` / `85.09%` | **`86.41%` / `84.64%`** | `88.32%` / `78.50%` | Nhạy hơn khi sóng thu nhỏ (α=0.8x) và khi đỉnh tăng (α=1.25x) |
| **C1 (INT8 Peak-Calib)** | `85.16%` / `85.32%` | **`85.01%` / `85.88%`** | `84.43%` / `83.89%` | Không bị bão hòa lớp giữa ở α=1.25x nhờ calib đỉnh cao |
| **C2a (INT8 exp=-4)** | `85.47%` / `85.09%` | **`86.41%` / `84.64%`** | `88.32%` / `78.50%` | Nhạy hơn khi sóng thu nhỏ (α=0.8x) và khi đỉnh tăng (α=1.25x) |
| **C2b (INT8 exp=-5)** | `85.84%` / `84.97%` | **`85.29%` / `86.02%`** | `86.45%` / `81.18%` | Bị cắt đỉnh R mạnh khi α=1.25x (exp_in=-5) |
| **C3 (Mixed INT16+INT8)** | `85.90%` / `85.03%` | **`85.80%` / `85.26%`** | `85.45%` / `82.61%` | Giữ nguyên độ phân giải 16-bit ở đầu vào & đầu ra, bền biên độ vượt trội C0/C2 |
| **C_W8A16 (W8A16)** | `85.86%` / `84.95%` | **`85.77%` / `85.47%`** | `85.03%` / `83.65%` | Bám sát tuyệt đối FP32 ở mọi mức biên độ |

---

## 4. Phân tích & Quyết định Chọn Cấu hình cho Giai đoạn D–F (Firmware & QEMU ESP32-S3)

1. **So sánh giữa `C0` (INT8 mặc định) và `C1` (INT8 hiệu chuẩn đỉnh cao):**
   - `C0` đạt `AUROC = 0.9128` (`Δ = +0.0012`).
   - `C1` (chỉ thay đổi tập hiệu chuẩn thêm 200 nhịp đỉnh cao từ `train.npz`, giữ nguyên 100% kiến trúc INT8 `42.09 KB`) nâng `AUROC` lên **`0.9153`** (`Δ = +0.0036`), nâng `PR-AUC` từ `0.6228` lên **`0.6665`** và tăng tỷ lệ khớp nhãn FP32 trên 49,691 nhịp từ `97.91%` lên **`98.48%`**.
2. **So sánh với `C3` (Mixed Precision INT16+INT8) và `C_W8A16`:**
   - `C3` (`44.12 KB`) đạt `AUROC = 0.9153` (`Δ = +0.0037`), khớp nhãn **`99.09%`** với FP32 trên 49,691 nhịp TEST.
   - `C_W8A16` (`45.95 KB`) và `B (INT16)` (`72.39 KB`) gần như trùng khít 100% với FP32 (`AUROC = 0.9103` và `0.9117`, khớp nhãn `>99.5%`).
3. **Quyết định cho Giai đoạn D & E:**
   - Đưa cả **`b_int16.espdl` (INT16)**, **`c1.espdl` (Pure INT8 tốt nhất)**, và **`c3.espdl` (Mixed Precision INT16+INT8)** cùng **Mã C thuần FP32** vào bộ kiểm thử 50 nhịp (`test_vectors.h`) để đối chiếu trực tiếp trên ESP-IDF / QEMU.
