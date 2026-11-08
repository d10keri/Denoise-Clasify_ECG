# PHÂN TÍCH & KẾ HOẠCH: CHUYỂN M1-REG SANG DEPTHWISE SEPARABLE CNN (M2-DS)

**Ngày lập:** 08/11/2026 | **Trạng thái:** KẾ HOẠCH – chưa viết code, chưa train, không sửa file hiện có
**Model gốc:** [`CNN1DBinaryRegularized`](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/src/models/cnn1d_binary_reg.py) (M1-Reg), checkpoint chính `checkpoints/cnn1d_binary_reg_best_seed42.pt`
**Mục tiêu:** Giảm chi phí tính toán, **ưu tiên giữ hoặc tăng độ chính xác** so với M1-Reg.
**Chú thích số liệu:** **[Đã đo]** từ file kết quả · **[Đã tính]** tính tay từ kiến trúc (sẽ kiểm lại bằng script ở Bước E0) · **[Ước tính]** / **[Nhận định]** chưa kiểm chứng.

---

## 1. ĐIỂM XUẤT PHÁT: M1-REG ĐANG TỐN TÀI NGUYÊN Ở ĐÂU?

| Tầng (sau khi gộp BN) | Params | MACCs | % MACC |
|---|---:|---:|---:|
| Conv1 1→32, k5, L=260 | 192 | 41,600 | 1.4% |
| Conv2 32→64, k5, L=130 | 10,304 | 1,331,200 | 44.8% |
| Conv3 64→128, k3, L=65 | 24,704 | 1,597,440 | 53.8% |
| Linear 128→2 | 258 | 256 | ~0% |
| **Tổng** | **35,458** | **2,970,496** | |

**[Đã tính]** Conv2 và Conv3 chiếm 98.6% MACC, nên đây là hai tầng cần thay bằng DS-Conv. Conv1 chỉ có 1 kênh vào nên DS-Conv không có lợi gì ở tầng này, cần giữ nguyên.

**Chất lượng tham chiếu của M1-Reg trên TEST [Đã đo]:**

| Chỉ số | Mean ± Std (3 seed) |
|---|---|
| AUROC | 0.8867 ± 0.0026 |
| PR-AUC | 0.5360 ± 0.0090 |
| Recall @0.50 | 0.8597 |
| Specificity @0.50 | 0.8347 |
| Recall nhịp S | 76.0% |

> [!NOTE]
> **Quan sát đáng lưu ý [Đã đo]:** M1-Reg có AUROC cao hơn M1 nhưng **PR-AUC thấp hơn** (0.536 so với 0.642). Nghĩa là ở vùng xác suất cao, M1-Reg có nhiều dương tính giả hơn. Vì vậy, khi so sánh với model DS, cần theo dõi cả PR-AUC chứ không chỉ AUROC.

### 1.1 Phát hiện quan trọng: trường tiếp nhận (receptive field) của M1-Reg rất hẹp **[Đã tính]**

| Tầng | Trường tiếp nhận (mẫu) | Bước nhảy |
|---|---:|---:|
| Conv1 k5 | 5 | 1 |
| MaxPool | 6 | 2 |
| Conv2 k5 | 14 | 2 |
| MaxPool | 16 | 4 |
| Conv3 k3 | **24** | 4 |

- 24 mẫu tại 360 Hz chỉ khoảng **67 ms**, ngắn hơn cả một phức bộ QRS (khoảng 80–120 ms).
- Mỗi đặc trưng trước tầng `AdaptiveAvgPool1d` chỉ "nhìn" được một mẩu nhỏ của nhịp tim. Model không thể thấy quan hệ **sóng P – QRS – sóng T** trong cùng một đặc trưng.
- **[Nhận định]** Đây là một lý do khiến nhịp **S** khó nhận diện: nhịp S chủ yếu khác N ở sóng P và thời điểm xuất hiện, chứ không ở hình dạng QRS.

**Hệ quả cho thiết kế DS-CNN:** Trong DS-Conv, phần depthwise rất rẻ. Tăng kernel depthwise từ 5 lên 9 chỉ tốn thêm vài chục nghìn MACC. Đây là cơ hội **mở rộng trường tiếp nhận gần như miễn phí**, tức vừa giảm chi phí vừa có khả năng tăng độ chính xác.

---

## 2. NHỮNG GÌ CẦN THAY ĐỔI (TỪ M1-REG → M2-DS)

### 2.1 Khối DS-Conv chuẩn đề xuất

```
DSConvBlock(C_in → C_out, k_dw):
  Conv1d(C_in, C_in, k_dw, groups=C_in, padding=k_dw//2, bias=False)  # depthwise: lọc theo thời gian, từng kênh
  BatchNorm1d(C_in) → ReLU
  Conv1d(C_in, C_out, 1, bias=False)                                   # pointwise: trộn kênh
  BatchNorm1d(C_out) → ReLU
  [MaxPool1d(2)] → [Dropout1d(p)]
```

| # | Thay đổi | Lý do |
|---|---|---|
| 1 | Conv2, Conv3 → DSConvBlock | Cắt khoảng 70–80% MACC của hai tầng nặng nhất. |
| 2 | **BN + ReLU giữa depthwise và pointwise** (kiểu MobileNet) | Tăng tính phi tuyến, bù phần dung lượng bị mất. BN được gộp vào conv khi xuất model (0 chi phí trên MCU), ReLU không tốn MACC. |
| 3 | **Kernel depthwise lớn hơn (7–9)** | Mở rộng trường tiếp nhận từ 24 lên khoảng 86 mẫu (≈ 240 ms) để thấy được P–QRS–T. Chi phí tăng không đáng kể. |
| 4 | **Thêm 1 khối DS thứ 4 + 1 MaxPool** | Phần tiết kiệm được từ DS cho phép tăng độ sâu, giúp đặc trưng cấp cao hơn và trường tiếp nhận rộng hơn. |
| 5 | `bias=False` cho conv trước BN | Bias bị BN triệt tiêu nên thừa. Cách làm chuẩn, giúp gộp BN gọn hơn. |
| 6 | Giữ BN + Dropout + Augmentation + AdamW + Cosine của M1-Reg | Công thức chống overfitting đã được chứng minh hiệu quả **[Đã đo]**. Giảm nhẹ dropout (0.10 / 0.20) vì model DS nhỏ hơn, dễ bị thiếu khớp (underfit) hơn. |
| 7 | Tăng số epoch 30 → 40–50 | **[Nhận định]** Mạng DS thường hội tụ chậm hơn. Vẫn lưu checkpoint theo Val AUROC nên không lo overfit muộn. |

### 2.2 Ba biến thể đề xuất **[Đã tính – trọng số sau khi gộp BN]**

**M2-A – Thay thế trực tiếp (để đối chứng):** Conv2 và Conv3 thay bằng DS, giữ nguyên kernel 5 và 3, giữ 3 tầng.

**M2-B – Ưu tiên độ chính xác (ĐỀ XUẤT CHÍNH):**

```
Input (1, 260)
→ Conv1d 1→32, k7 + BN + ReLU + MaxPool2          → (32, 130)
→ DS 32→64,  k_dw=9 + MaxPool2 + Dropout1d(0.10)  → (64, 65)
→ DS 64→128, k_dw=7 + MaxPool2 + Dropout1d(0.10)  → (128, 32)
→ DS 128→128, k_dw=5                              → (128, 32)
→ AdaptiveAvgPool1d(1) → Dropout(0.20) → Linear 128→2
```

**M2-C – Ưu tiên chi phí:** cùng cấu trúc với M2-B nhưng độ rộng giảm còn 24 → 48 → 96 → 96.

| Biến thể | Params | MACCs | So với M1-Reg | Trường tiếp nhận | Peak activation RAM FP32 |
|---|---:|---:|---:|---:|---:|
| M1-Reg (gốc) | 35,458 | 2,970,496 | – | 24 mẫu (67 ms) | ≈ 48.75 KB |
| M2-A thay trực tiếp | 11,330 | 873,856 | **−71%** | 24 mẫu | ≈ 48.75 KB |
| **M2-B (đề xuất)** | **29,058** | **1,468,544** | **−51%** | **86 mẫu (239 ms)** | ≈ 48.75 KB |
| M2-C gọn | 16,802 | 853,344 | **−71%** | 86 mẫu | ≈ 36.6 KB |

Chi tiết MACC của M2-B:

| Thành phần | MACCs |
|---|---:|
| Stem | 58,240 |
| DS2 (dw / pw) | 37,440 / 266,240 |
| DS3 (dw / pw) | 29,120 / 532,480 |
| DS4 (dw / pw) | 20,480 / 524,288 |
| Linear | 256 |

**Nhận xét:**
- **M2-C có chi phí ngang M2-A** (≈ 0.85M MACC) nhưng sâu hơn và trường tiếp nhận rộng gấp 3.6 lần. **[Nhận định]** Nếu cần model nhỏ, M2-C nhiều khả năng chính xác hơn M2-A.
- **Peak RAM không giảm khi dùng DS-Conv**, vì kích thước tensor đầu ra giữ nguyên và còn có thêm tensor trung gian sau depthwise. RAM chỉ giảm khi dùng INT8 (≈ ¼) hoặc khi giảm độ rộng kênh (M2-C).
- **Latency không giảm tỷ lệ thuận với MACC.** Kernel depthwise trên CMSIS-NN thường kém hiệu quả hơn conv chuẩn. Phải đo bằng ST Edge AI Core (Giai đoạn C) mới biết, không suy ra từ MACC.

---

## 3. KẾ HOẠCH THỰC HIỆN (chỉ tạo file mới)

### E0 – Script đo chi phí model (làm trước khi train)
- **File:** `scripts/binary/model_cost_report.py`
- **Việc làm:** dùng forward hook để đếm params (sau khi gộp BN), MACC, trường tiếp nhận và kích thước tensor cho M1-Reg, M2-A, M2-B, M2-C.
- **Đầu ra:** `results/binary/m2_ds_fp32/cost_report.json`.
- **Gate E0:** số đếm bằng script khớp bảng 2.2 (khi khớp, đổi nhãn [Đã tính] thành [Đã đo]).

### E1 – Định nghĩa model
- **File:** `src/models/cnn1d_binary_ds.py`
- **Nội dung:**
  - Class `DSConvBlock`.
  - Class `CNN1DBinaryDS(variant="A"|"B"|"C")` đọc cấu hình từ một dict.
  - Khối `__main__` tự kiểm tra: input `(1,1,260)` cho ra output `(1,2)`, và số params khớp E0.

### E2 – Huấn luyện
- **File:** `train_cnn1d_binary_ds.py --variant B`
- **Công thức:** giữ nguyên M1-Reg (augmentation ±3 mẫu, scale [0.88, 1.12], nhiễu 0.015; class weights [0.5546, 5.0786]; AdamW wd = 1e-3; Cosine). Riêng số epoch là 40 và dropout là 0.10 / 0.20.
- **Seed:** 42, 43, 44. Chọn checkpoint theo Val AUROC.
- **Đầu ra:**
  - `checkpoints/cnn1d_binary_ds{B}_best_seed{S}.pt`
  - `results/binary/m2_ds_fp32/B/history_seed{S}.json`
- **Thứ tự chạy:**
  1. **M2-B** trước (biến thể chính).
  2. M2-A (để đo xem các cải tiến thiết kế đóng góp bao nhiêu).
  3. M2-C (chỉ khi M2-B đạt yêu cầu và muốn model nhỏ hơn nữa).

### E3 – Đánh giá & so sánh
- **File:** `evaluate_cnn1d_binary_ds.py --variant B`
- **Giao thức:** dùng lại `select_thresholds_on_val` và `eval_at_threshold` từ `evaluate_cnn1d_binary.py`. Ngưỡng chọn trên VAL, áp lên TEST, báo cáo mean ± std của 3 seed.
- **Đầu ra:**
  - `results/binary/m2_ds_fp32/B/test_metrics.json`
  - `results/binary/m2_ds_fp32/comparison_vs_m1reg.md` (bảng chất lượng cùng params và MACC)

**Gate E3 – cổng chấp nhận model DS (ngưỡng đề xuất, chưa kiểm chứng):**

| Tiêu chí | Ngưỡng |
|---|---|
| Recall và Specificity trên TEST tại ngưỡng chọn từ VAL | ≥ 0.85 ở ít nhất 1 seed; trung bình 3 seed giảm không quá 1–2 điểm so với M1-Reg |
| TEST AUROC (trung bình) | ≥ 0.877 (M1-Reg − 0.01) |
| TEST PR-AUC (trung bình) | ≥ 0.526 |
| Recall nhịp S | giảm không quá 5 điểm so với M1-Reg (≈ 76%) |
| MACC | ≤ 50% của M1-Reg (M2-B đạt 49.4%) |

### E4 – Chưng cất tri thức (Knowledge Distillation), chỉ khi M2-B chưa qua Gate E3
- **File:** `train_cnn1d_binary_ds_kd.py`
- **Teacher:** trung bình xác suất của 3 checkpoint M1-Reg (chạy offline, không ảnh hưởng MCU).
- **Hàm mất mát:** `L = α·CE_weighted(y) + (1−α)·T²·KL(student_T ‖ teacher_T)`, với α = 0.5 và T = 3 (sẽ quét T ∈ {2, 4}).
- **[Nhận định]** Đây là cách bù độ chính xác cho model nhỏ mà **không tốn thêm chi phí trên MCU**.

### E5 – Chuyển sang Giai đoạn C/D của plan chính
- Model DS tốt nhất đi tiếp qua: xuất ONNX → `analyze` trên STM32Cube AI Studio → INT8.
- Bắt buộc so sánh latency và RAM đo được với M1-Reg, vì MACC giảm không có nghĩa là latency giảm tương ứng.

---

## 4. CÁC PHƯƠNG PHÁP BỔ SUNG (xếp theo hiệu quả / chi phí)

| Ưu tiên | Phương pháp | Tác động độ chính xác | Chi phí trên MCU | Ghi chú |
|---|---|---|---|---|
| **1** | **Đặc trưng RR-interval** (RR trước, RR sau, tỷ lệ RR so với trung bình cục bộ) ghép vào trước Linear | **[Nhận định] Cao nhất cho nhịp S.** Nhịp S khác N chủ yếu ở việc đến sớm, điều mà cửa sổ 260 mẫu đã chuẩn hóa không thể hiện được. | Thêm 3 input, khoảng 6 params | Tính được từ `beat_indices` và `record_ids` đã có trong npz, **không cần build lại dataset**. Chỉ dùng trung bình RR *nhân quả* để tránh dùng dữ liệu tương lai. RR sau làm trễ quyết định 1 nhịp. Thay đổi giao diện input (model 2 đầu vào), nên cần **kế hoạch riêng (M3)**. |
| 2 | Knowledge Distillation (E4) | Trung bình – cao | 0 | Chỉ ảnh hưởng quá trình train. |
| 3 | Kernel depthwise lớn + khối thứ 4 (đã có trong M2-B) | Trung bình | Thấp | |
| 4 | Residual connection ở DS4 (128→128, cùng shape) | Thấp – trung bình, giúp hội tụ | Thêm 1 phép `Add`, thêm khoảng 16 KB RAM (FP32) | ST hỗ trợ phép `Add`. Cân nhắc nếu M2-B khó hội tụ. |
| 5 | Squeeze-and-Excitation nhẹ (tỷ lệ giảm 8) | Thấp – trung bình | ≈ +4K params, thêm phép sigmoid và nhân | Để sau, chỉ làm khi còn dư ngân sách tài nguyên. |
| 6 | **ReLU6 / Hardtanh** thay ReLU | Không đổi ở FP32, **giúp INT8 ổn định hơn** | 0 | Giới hạn dải activation, đặc biệt có lợi cho depthwise vốn nhạy với lượng tử hóa. |
| 7 | QAT (Quantization-Aware Training) | Giữ chất lượng khi chuyển INT8 | 0 | Chỉ làm khi PTQ INT8 làm mất hơn 2 điểm (Gate D). DS-Conv dễ cần QAT hơn conv chuẩn. |

**Không khuyến nghị:**
- **Ensemble nhiều model trên MCU:** chi phí nhân lên nhiều lần.
- **Giảm kernel Conv1:** làm hẹp thêm trường tiếp nhận vốn đã nhỏ.
- **Cắt bớt layer:** đi ngược với phát hiện ở mục 1.1.

---

## 5. RỦI RO

| Rủi ro | Biện pháp |
|---|---|
| Model DS thiếu khớp (underfit), chất lượng giảm | Dùng M2-B (rộng và sâu hơn M2-A), giảm dropout, tăng epoch, dùng KD (E4) |
| Depthwise nhạy với INT8 | Per-channel quantization, ReLU6, QAT dự phòng |
| Latency thực tế không giảm như kỳ vọng từ MACC | Đo bằng ST tool. Nếu depthwise chậm, chuyển sang M2-C hoặc giữ M1-Reg cùng INT8 |
| Kernel lớn hơn không giúp ích | M2-A là đối chứng để tách riêng tác động của từng thay đổi |
| Dao động giữa các seed (inter-patient) | Luôn chạy 3 seed, báo cáo mean ± std, chọn ngưỡng chỉ trên VAL |

---

## 6. KẾT LUẬN & ĐỀ XUẤT

- **Phương án chính: M2-B.**
  - MACC giảm khoảng **51%** (2.97M → 1.47M), params giảm 18% **[Đã tính]**.
  - Trường tiếp nhận tăng từ 24 lên 86 mẫu. **[Nhận định]** Đây là yếu tố có khả năng **giữ hoặc cải thiện** độ chính xác, đặc biệt với nhịp S.
- **Nếu cần nhẹ hơn nữa: M2-C** (giảm khoảng 71% MACC). Chỉ chọn sau khi M2-B qua Gate E3.
- **Hướng có tiềm năng tăng độ chính xác lớn nhất: đặc trưng RR-interval.** Hướng này nằm ngoài phạm vi thay DS-Conv, nên lập kế hoạch riêng (M3) sau khi có kết quả M2-B.
- **Việc nhỏ nhất làm tiếp:** Bước **E0 + E1** – viết `model_cost_report.py` và `cnn1d_binary_ds.py`, kiểm chứng params, MACC và trường tiếp nhận của các biến thể (chưa train).
