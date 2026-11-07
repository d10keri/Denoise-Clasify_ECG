# KẾ HOẠCH CHI TIẾT: 1D-CNN ECG 2 LỚP (N vs non-N) → INT8 → STM32

**Phiên bản:** 2.0 (07/11/2026) | **Trạng thái:** KẾ HOẠCH – chưa thực thi, chưa sửa file nào đang có trong project
**Căn cứ:** [edge_ai_initial_assessment.md](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/plan/edge_ai_initial_assessment.md), [cnn1d.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/src/models/cnn1d.py), [train_cnn1d.py](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/train_cnn1d.py), [test_metrics.json](file:///c:/d2l-en/Denoise_Classify%20ECG/Denoise-Clasify_ECG/results/cnn1d/test_metrics.json)
**Chú thích số liệu:** **[Đã đo]** từ file kết quả · **[Đã tính]** tính từ kiến trúc · **[Ước tính]** chưa kiểm chứng · **[Tài liệu ST]** từ tra cứu ST (cần đối chiếu datasheet khi mua board).

---

## 0. CÁC QUYẾT ĐỊNH ĐÃ CHỐT

| Hạng mục | Quyết định |
|---|---|
| Bài toán | **Nhị phân: lớp 0 = N, lớp 1 = non-N (S+V+F+Q)** |
| Hướng tối ưu | (2) Compiler memory optimization → (1) PTQ INT8 → (3) DS-Conv *(chỉ khi cần)* |
| Ngưỡng chất lượng tạm thời *(chưa kiểm chứng, sẽ chỉnh sau khi chạy thật)* | **Recall (non-N) ≥ 0.85; Specificity ≥ 0.85; INT8 so với FP32: giảm ≤ 1–2 điểm %** |
| Công cụ | STM32Cube AI Studio (miễn phí **[Tài liệu ST]**) + ST Edge AI Core; chưa có board → làm trên PC |
| Quy tắc | Chỉ tạo **file mới**; không sửa `cnn1d.py`, `ecg_dataset.py`, dữ liệu `data/processed/*.npz`, checkpoint cũ |
| Môi trường | Conda `ecg` (`C:\Users\vietc\.conda\envs\ecg\python.exe`). Các thư viện thêm (onnx, onnxruntime…) cần **xin phép trước khi cài** |

**Phân bố nhãn sau khi gộp:** Test: N = 44,239, non-N = 5,452 (**11.0%**) **[Đã tính]**; Train ≈ 90/10 **[Ước tính từ class_weights, cần xác nhận ở Bước 1.1]**.
Accuracy không còn đáng tin (đoán toàn N đã ≈ 89%) → dùng **Recall, Specificity, F1(non-N), AUROC, PR-AUC**.

---

## 1. CHỌN STM32 (miễn phí công cụ, hiệu năng vừa phải)

**Điểm mấu chốt:** STM32Cube AI Studio / ST Edge AI Core miễn phí và hỗ trợ hầu hết dòng Cortex-M (M4, M7, M33…) **[Tài liệu ST]**; chip mạnh kèm NPU (STM32N6) không cần cho model này. Model rất nhỏ nên **không cần chip cao cấp**:

| Nhu cầu của model | FP32 (M1) | INT8 (M1) | Ghi chú |
|---|---:|---:|---|
| Flash cho weights | ≈ 138.5 KB | ≈ 34.6 KB | **[Đã tính]** + code runtime/firmware (~vài chục KB, **[Ước tính]**) |
| Peak RAM activation | ≈ 48.75 KB | ≈ 12.2 KB | **[Đã tính từ kích thước tensor]**; compiler ST có thể khác – sẽ đo ở Bước 3 |
| MACCs/nhịp | 2.97M | 2.97M | **[Đã tính]** |

### Ứng viên (thông số theo datasheet từ hiểu biết thông thường – **cần xác nhận lại khi mua**)

| Board | MCU / Core | Xung | Flash / RAM | Nhận xét cho dự án |
|---|---|---|---|---|
| **NUCLEO-F446RE (đề xuất chính)** | F446RE, Cortex-M4F | 180 MHz | 512 KB / 128 KB | Rẻ, phổ biến, tài liệu nhiều. Đủ Flash/RAM cho cả FP32 lẫn INT8. Có FPU + DSP/SIMD. |
| NUCLEO-G474RE (phương án 2) | G474RE, Cortex-M4F | 170 MHz | 512 KB / 128 KB | Tương đương F446; có ADC/bộ tăng tốc toán học (CORDIC, FMAC) hữu ích cho tiền xử lý ECG sau này. |
| NUCLEO-L476RG (tiết kiệm điện) | L476RG, Cortex-M4F | 80 MHz | 1 MB / 128 KB | Đủ bộ nhớ, nhưng xung thấp → latency ≈ gấp ~2 lần F446. Chọn nếu ưu tiên pin. |
| NUCLEO-H743ZI2 (dư sức, đắt hơn) | H743ZI, Cortex-M7 | 400–480 MHz | 2 MB / ~1 MB | Mạnh nhất nhưng không cần cho 3M MACCs; chỉ dùng nếu muốn mở rộng model sau này. |

**Ước tính latency cho M1 (3M MACCs) trên F446RE @180 MHz:** INT8 ≈ 15–50 ms, FP32 ≈ 80–170 ms **[Ước tính thô – giả định ~1–3 chu kỳ/MAC (INT8) và ~5–10 chu kỳ/MAC (FP32); chỉ số thật phải đo]**. Cả hai đều ≪ khoảng cách giữa 2 nhịp (~250–1000 ms), nên **về mặt thời gian chạy CNN, F446RE dư dùng**.

> **Kết luận chọn chip:** Chọn **NUCLEO-F446RE** (hoặc G474RE). Memory không phải rào cản lớn với M1; rào cản thực tế nằm ở **tiền xử lý thời gian thực** (lọc nhân quả + phát hiện R-peak) chứ không phải CNN. **Đây là nhận định kỹ thuật; sẽ được xác nhận hoặc bác bỏ bằng báo cáo `analyze` ở Bước 3 (chọn chip trong công cụ để xem chính thức).** Với số liệu trên, DS-Conv nhiều khả năng **không cần thiết** cho mục tiêu ban đầu – chỉ xét nếu muốn chạy trên chip nhỏ hơn hoặc cần latency thấp hơn.

---

## 2. CÁC BƯỚC THỰC HIỆN CỤ THỂ

Mỗi giai đoạn có **Đầu vào → Việc làm → Đầu ra → Tiêu chí qua cổng (Gate)**. Không qua cổng thì không sang bước sau.

### GIAI ĐOẠN A – Chuẩn bị và đường chuẩn

**A1. Kiểm tra dữ liệu nhị phân (chỉ đọc).**
- File mới: `scripts/binary/check_binary_labels.py`
- Làm: nạp `train/val/test.npz`, ánh xạ `y>0 → 1`, in số lượng N/non-N mỗi split; xác nhận tỷ lệ train (thay số **[Ước tính]**).
- Đầu ra: `results/binary/label_counts.json`.
- **Gate A1:** có số lượng thực của cả 3 split.

**A2. Baseline "miễn phí" từ model 5 lớp.**
- File mới: `scripts/binary/eval_binary_from_5class.py`
- Làm: nạp `checkpoints/cnn1d_best.pt` (chế độ eval), tính `P(non-N) = 1 − softmax(logits)[N]` trên val và test; tính Recall, Specificity, F1, AUROC, PR-AUC; quét ngưỡng trên **val** để chọn ngưỡng, báo cáo test một lần.
- Đầu ra: `results/binary/baseline_from_5class.json`.
- **Gate A2:** có mốc xuất phát. (Chưa cần đạt ngưỡng 0.85.)

### GIAI ĐOẠN B – Model 2 lớp FP32 (M1)

**B1. Định nghĩa model + dataset nhị phân (file mới).**
- `src/models/cnn1d_binary.py`: `CNN1DBinary` = y hệt `CNN1DBaseline` nhưng `Linear(128, 2)` (35,458 params **[Đã tính]**).
- `src/data/binary_dataset.py`: wrapper bọc `ECGClassificationDataset`, trả nhãn `int(y>0)`; không sửa file gốc.
- Kiểm chứng: in số params = 35,458; forward `(1,1,260)` → `(1,2)`.

**B2. Train.**
- File mới: `train_cnn1d_binary.py`, cấu hình rõ ràng (ghi seed, lr, batch…), tham số tương tự `train_cnn1d.py` (Adam, lr 1e-3, batch 64, 30 epoch) để so sánh công bằng.
- Loss: `CrossEntropyLoss` có trọng số lớp tính **từ train** (hai lớp).
- Chọn checkpoint tốt nhất theo **chỉ số trên val** (đề xuất: AUROC hoặc F1(non-N)); lưu `checkpoints/cnn1d_binary_best_seed{S}.pt`.
- Chạy **3 seed** (ví dụ 42, 43, 44) để biết độ dao động (val/test baseline hiện lệch nhiều **[Đã đo]**).
- Đầu ra: `results/binary/m1_fp32/history_seed{S}.json`.

**B3. Chọn ngưỡng và đánh giá.**
- File mới: `evaluate_cnn1d_binary.py`.
- Chọn ngưỡng quyết định trên **val** sao cho Recall(non-N) ≥ 0.85 và Specificity tối đa; áp ngưỡng cố định lên test; báo cáo trung bình ± độ lệch chuẩn của 3 seed.
- Đầu ra: `results/binary/m1_fp32/test_metrics.json`, confusion matrix.
- **Gate B (FP32 đủ tốt): Recall ≥ 0.85 VÀ Specificity ≥ 0.85 trên test (giá trị tạm thời).**
  - Nếu **không đạt**: dừng nén. Xử lý chất lượng gốc trước (xem Mục 4: hướng khi không đạt) – vì nén model yếu không có ý nghĩa.

### GIAI ĐOẠN C – Xuất ONNX và phân tích bằng ST (Hướng 2: memory)

> Cần hỏi bạn xác nhận trước khi cài `onnx`, `onnxruntime` vào Conda `ecg`.

**C1. Xuất ONNX (file mới).**
- `scripts/binary/export_onnx.py`: model ở `eval()`, input tĩnh `(1,1,260)`, **không** dùng dynamic axes; opset ổn định; xuất `export/cnn1d_binary_fp32.onnx`.
- Nếu `AdaptiveAvgPool1d(1)` xuất ra đồ thị lạ (nhiều `Shape/Slice…`): trong file export, dùng bản wrapper thay bằng `AvgPool1d(65)` hoặc `mean(-1)` (số học tương đương vì chiều dài tại đó cố định = 65) – **không sửa model gốc**.
- Kiểm chứng: `scripts/binary/verify_onnx.py` chạy `onnxruntime` trên ≥ 1,000 nhịp test, so logit với PyTorch.
- **Gate C1:** sai khác logit lớn nhất ≤ ~1e-4 và dự đoán trùng 100% *(ngưỡng gợi ý)*.

**C2. Phân tích tài nguyên bằng STM32Cube AI Studio.**
- Làm: nạp `.onnx` → chọn lần lượt NUCLEO-F446RE, G474RE, L476RG → Analyze; ghi **Flash, RAM (Peak), MACC, % MACC theo tầng, tầng nào được gộp (ReLU fuse), có cảnh báo toán tử nào**.
- Có thể kiểm tra thêm `stedgeai supported-ops` để xác nhận toán tử.
- Đầu ra: bảng `results/binary/stm32_analysis_m1_fp32.md` (số **[Đã đo bằng công cụ ST]**, ghi rõ phiên bản AI Studio/Edge AI Core).
- **Gate C2:** Flash và Peak RAM của FP32 nằm trong giới hạn chip ứng viên với dư địa cho firmware (đề xuất: dùng ≤ ~70% Flash và ≤ ~60% RAM; *ngưỡng gợi ý*). So sánh với số **[Ước tính]** ở Mục 1 và cập nhật nếu lệch.

**C3. Host validation FP32.**
- Làm: trong AI Studio chạy Validate on desktop với dữ liệu thật (một tệp `.npy` gồm vài trăm nhịp test, trong đó có đủ cả N và non-N, kèm đầu ra tham chiếu).
- **Gate C3:** sai số so với ONNX nhỏ (công cụ báo cáo L2r/COS *(đề xuất COS ≈ 1)*).

### GIAI ĐOẠN D – Lượng tử hóa INT8 sau huấn luyện (Hướng 1)

**D1. Tạo tập hiệu chuẩn.**
- File mới: `scripts/binary/make_calibration_set.py`
- Lấy từ **train** (không dùng test), ~600 nhịp **phân tầng**: ~300 N, ~300 non-N (có đủ S, V, F theo tỷ lệ thực). Cố định seed; lưu `export/calib_train.npy` (float32, `(600,1,260)`).

**D2. Lượng tử hóa — chọn MỘT đường và giữ nhất quán:**
- *Đường 1 (đề xuất vì cùng công cụ với mã sinh ra):* lượng tử hóa trong STM32Cube AI Studio / ST Edge AI Core, per-channel nếu có; dùng `calib_train.npy`.
- *Đường 2 (dự phòng):* static PTQ bằng `onnxruntime.quantization` (định dạng QDQ, per-channel weights) rồi nạp ONNX đã lượng tử hóa vào ST.
- Ghi lại phương pháp, phiên bản, tham số.

**D3. Đánh giá INT8 vs FP32 trên val/test (Host validation).**
- Cùng dữ liệu, cùng ngưỡng cố định chọn từ FP32 (và thử hiệu chỉnh lại ngưỡng trên val nếu cần).
- Báo cáo: Recall, Specificity, F1, AUROC, tỷ lệ dự đoán trùng FP32, sai khác logit. Xem riêng Recall theo lớp gốc (S, V, F) để thấy lớp nào bị ảnh hưởng.
- **Gate D (INT8 đủ tốt):** Recall ≥ 0.85, Specificity ≥ 0.85, và **mất ≤ 1–2 điểm %** Recall/Specificity so với FP32 (giá trị tạm thời).
- Nếu không đạt, thử lần lượt: tăng/đổi tập hiệu chuẩn → per-channel → giữ tầng cuối Linear ở độ chính xác cao hơn → (dự phòng) QAT (cần bổ sung kế hoạch riêng).

**D4. Phân tích tài nguyên INT8.**
- Chạy lại Analyze trên bản INT8: ghi Flash, Peak RAM, MACC, chu kỳ ước tính (nếu công cụ cung cấp).
- Đầu ra: `results/binary/stm32_analysis_m1_int8.md`; bảng so sánh FP32 vs INT8 (đối chiếu kỳ vọng ≈ 4× giảm).

### GIAI ĐOẠN E – DS-Conv (Hướng 3) — **CHỈ KHI CẦN**

**Điều kiện kích hoạt:** (a) kết quả D4 cho thấy M1 INT8 vẫn không phù hợp chip mục tiêu; hoặc (b) latency đo thật vượt ngân sách; hoặc (c) bạn muốn chạy trên chip nhỏ hơn.
- File mới: `src/models/cnn1d_binary_ds.py` (M2): Conv1 giữ nguyên; Conv2, Conv3 thành depthwise + pointwise 1×1 (11,330 params; 0.87M MACCs **[Đã tính]**).
- Train cùng protocol (3 seed) → **Gate E1:** FP32 chênh ≤ ~2 điểm so với M1.
- Lượng tử hóa lại từ D1–D4. **Chú ý:** depthwise nhạy với INT8; cần per-channel/QAT dự phòng.
- Không suy latency từ MACCs: depthwise thường kém hiệu quả hơn → phải đo.

### GIAI ĐOẠN F – Tổng hợp quyết định

- Bảng {M1, (M2)} × {FP32, INT8}: Recall/Specificity/F1/AUROC (mean ± std), Flash, Peak RAM, MACC, ghi rõ nguồn số liệu.
- File: `plan/edge_ai_results_summary.md` (tạo sau khi có số liệu thật).
- Chọn cấu hình nhỏ nhất vẫn qua Gate B và D.

### GIAI ĐOẠN G – Khi có board (ngoài phạm vi các bước PC)

1. Mua board (đề xuất NUCLEO-F446RE). 2. Tạo project trong AI Studio / CubeMX/CubeIDE, Validate on target để đo **latency thực**. 3. Nạp qua CubeProgrammer/ST-LINK. 4. Thiết kế lại tiền xử lý thời gian thực (xem Mục 3).

---

## 3. RÀO CẢN LỚN NHẤT: TIỀN XỬ LÝ THỜI GIAN THỰC (không phải CNN)

Dữ liệu train được tạo bằng `filtfilt` (lọc 2 chiều toàn bản ghi) và cắt nhịp quanh **nhãn R-peak của bác sĩ** (`segmentation.py`); trên MCU không có cả hai. Trước khi tin kết quả "chạy được trên STM32", cần một bước mô phỏng trên PC (kế hoạch riêng, **ngoài phạm vi tài liệu này**):
- bộ lọc nhân quả (causal IIR biquad / FIR);
- bộ phát hiện R-peak (ví dụ Pan-Tompkins);
- đánh giá lại chất lượng model khi cửa sổ lấy theo R-peak do thuật toán tìm ra (có sai lệch vài mẫu).

Nếu bỏ qua bước này, chất lượng trên board có thể thấp hơn nhiều so với số đo offline.

---

## 4. RỦI RO VÀ HƯỚNG XỬ LÝ

| Rủi ro | Mức | Biện pháp |
|---|---|---|
| FP32 2 lớp không đạt Gate B (baseline 5 lớp yếu: Macro-F1 0.3282 **[Đã đo]**; lớp S, F rất yếu) | **Cao** | Hướng khi không đạt (theo thứ tự rẻ → tốn): chỉnh ngưỡng/trọng số lớp; augmentation (nhiễu nhẹ, dịch pha) ; tăng dung lượng nhẹ; lấy đặc trưng RR-interval (cần thêm đầu vào) ; xét lại mục tiêu ngưỡng. **Không nén khi chưa qua Gate B.** |
| Val/test lệch (inter-patient) → dao động | Cao | 3 seed; báo mean ± std; AUROC/PR-AUC |
| Chọn ngưỡng/checkpoint trên test | Cao | Chỉ dùng val |
| Calibration thiếu non-N | TB | Lấy mẫu phân tầng (D1) |
| INT8 từ hai công cụ cho kết quả khác | TB | Chọn một đường (D2); validate chính artefact đưa xuống MCU |
| Pool/ONNX không sạch | Thấp–TB | Wrapper export (C1) |
| Cùng bệnh nhân 201 (train) & 202 (test) | TB | Ghi chú giới hạn khi báo cáo |
| Ước tính latency/Flash/RAM sai | TB | Thay bằng số từ AI Studio (C2, D4) và đo trên board (G) |

---

## 5. NHẬN XÉT VỀ KHẢ NĂNG THỰC THI, KẾT QUẢ KỲ VỌNG VÀ BƯỚC TIẾP THEO

- **Thực thi:** Giai đoạn A–D làm được hoàn toàn trên PC, không cần board, không tốn phí công cụ **[Tài liệu ST]**. Rủi ro lớn nhất là chất lượng FP32 (Gate B), không phải dung lượng/chip.
- **Kết quả kỳ vọng (nhận định, chưa phải số đo):** M1 INT8 khoảng 35 KB weights và ≈ 12 KB activation, thoải mái trên F446RE. Mất chất lượng do PTQ INT8 thường nhỏ với model nhỏ như thế này, nhưng ECG cần kiểm chứng; đây là điều Gate D sẽ trả lời.
- **Hướng tiếp theo:** (1) Làm A1–A2 (chỉ đọc) để biết điểm xuất phát thật; (2) B để biết FP32 2 lớp có qua Gate B; (3) hỏi xác nhận để cài `onnx`/`onnxruntime` và cài STM32Cube AI Studio rồi làm C; (4) D. Chỉ cân nhắc E nếu A–D chưa đủ. Cân nhắc mua NUCLEO-F446RE sau khi qua Gate C2/D4.

**Nhiệm vụ nhỏ nhất làm tiếp theo:** Bước A1 + A2 (chỉ đọc, không train, không cài thêm thư viện).
