# HƯỚNG DẪN KIẾN TRÚC MÔ HÌNH VÀ THAM SỐ FINE-TUNE ECG DENOISING

> **Dự án:** ECG Denoising & Classification (Level 2 — Phase 2)  
> **Mục tiêu:** Hướng dẫn trực quan, dễ hiểu dành cho người học Deep Learning cơ bản về 3 kiến trúc khử nhiễu ECG: DAE, 1D-CNN và 1D-ResNet, kèm theo sổ tay tham số fine-tune chi tiết.

---

## 1. NGUYÊN LÝ CHUNG CỦA BÀI TOÁN ECG DENOISING CÓ GIÁM SÁT

Trong bài toán Supervised Waveform Denoising (khử nhiễu có giám sát):
- **Đầu vào ($X_{\text{noisy}}$):** Đoạn tín hiệu ECG bị nhiễm nhiễu, kích thước `(Batch, 1, 1024)`.
- **Mô hình ($\mathcal{M}_{\theta}$):** Hàm biến đổi phi tuyến chứa các trọng số cần học $\theta$, dự đoán tín hiệu sạch $\hat{X} = \mathcal{M}_{\theta}(X_{\text{noisy}})$.
- **Tín hiệu tham chiếu ($X_{\text{clean}}$):** Đoạn tín hiệu sạch tương ứng đã được lọc khử trôi dải thông $[0.05, 60\text{ Hz}]$, kích thước `(Batch, 1, 1024)`.
- **Mục tiêu huấn luyện:** Tối thiểu hóa sai số giữa $\hat{X}$ và $X_{\text{clean}}$ thông qua hàm mất mát (Loss Function).

```
Tín hiệu nhiễu: X_noisy (1, 1024) ───► [ MÔ HÌNH HỌC SÂU ] ───► Dự đoán: X_hat (1, 1024)
                                                                        │
                                       Tín hiệu sạch: X_clean (1, 1024) ──► [ LOSS (MSE/L1) ] ──► Cập nhật trọng số
```

---

## 2. NGHIÊN CỨU CHI TIẾT 3 KIẾN TRÚC MÔ HÌNH

### 2.1 Model 1 — Denoising Autoencoder (DAE)

#### A. Nguyên lý hoạt động
Denoising Autoencoder hoạt động dựa trên nguyên lý **thắt nút cổ chai thông tin (Information Bottleneck)**:
1. **Bộ mã hóa (Encoder):** Nén tín hiệu đầu vào 1024 mẫu qua nhiều tầng tích chập và giảm chiều (stride hoặc max pooling), ép tín hiệu vào một không gian biểu diễn cô đọng gọi là **Bottleneck (Latent Space)**.
2. **Nút cổ chai (Bottleneck):** Không gian này có kích thước nhỏ hơn nhiều so với tín hiệu ban đầu. Do nhiễu (BW, MA, EM) thường có tính chất ngẫu nhiên hoặc entropy cao, nút cổ chai ép mạng chỉ giữ lại những quy luật cấu trúc mạnh nhất (chính là hình thái nhịp tim ECG).
3. **Bộ giải mã (Decoder):** Tái tạo lại tín hiệu từ biểu diễn cô đọng trở về kích thước ban đầu 1024 mẫu thông qua các lớp phóng to (Transposed Convolution hoặc Upsampling + Convolution).

```
[X_noisy: 1x1024] 
       │
       ▼ Conv1D (Stride=2)
   [16 x 512]
       │
       ▼ Conv1D (Stride=2)
   [32 x 256]
       │
       ▼ Conv1D (Stride=2)
   [64 x 128]  ◄─── [ BOTTLENECK / LATENT SPACE ] (Nén thông tin, loại bỏ nhiễu)
       │
       ▼ ConvTranspose1D (Stride=2)
   [32 x 256]
       │
       ▼ ConvTranspose1D (Stride=2)
   [16 x 512]
       │
       ▼ ConvTranspose1D (Stride=2) + Conv1D
[X_hat: 1x1024]
```

#### B. Cấu trúc đề xuất ở mức khái niệm (Conceptual Architecture)
- **Đầu vào:** `(Batch, 1, 1024)`
- **Encoder:** 3 hoặc 4 khối `Conv1d -> BatchNorm1d (hoặc LayerNorm) -> LeakyReLU -> MaxPool1d(2)`. Số kênh tăng dần: $1 \to 16 \to 32 \to 64$. Chiều dài giảm dần: $1024 \to 512 \to 256 \to 128$.
- **Bottleneck:** `(Batch, 64, 128)` (Tỉ lệ nén chiều thời gian: $8\times$).
- **Decoder:** 3 hoặc 4 khối `ConvTranspose1d (stride=2) -> BatchNorm1d -> LeakyReLU`. Số kênh giảm dần: $64 \to 32 \to 16 \to 1$. Chiều dài tăng dần: $128 \to 256 \to 512 \to 1024$.
- **Lớp cuối:** `Conv1d(16, 1, kernel_size=3, padding=1)` không dùng activation (hoặc dùng Tanh nếu tín hiệu nằm trong $[-1, 1]$).

#### C. Ưu điểm và hạn chế đối với ECG Denoising
- **Ưu điểm:** Khử nhiễu biên độ lớn rất mạnh. Do bị ép qua nút cổ chai, các thành phần nhiễu tần số cao hoặc phân kỳ mạnh bị gọt bỏ hiệu quả.
- **Hạn chế lớn (Nguy cơ làm mượt tín hiệu — Oversmoothing):** Khi nén chiều thời gian ($1024 \to 128$), các chi tiết biên độ hẹp và dốc đứng như **đỉnh sóng R (R-peak)** hoặc các khía phức bộ QRS rất dễ bị "san phẳng". Hậu quả là sóng R sau khi khử nhiễu bị thấp đi hoặc bè ra, làm mất giá trị chẩn đoán lâm sàng.
- **Độ phức tạp & Edge AI:** Rất gọn nhẹ, số lượng tham số ít ($\approx 50\text{k} - 150\text{k}$ params), dễ nạp vào vi điều khiển MCU/DSP.

---

### 2.2 Model 2 — 1D-CNN Denoiser (Fully Convolutional Network)

#### A. Nguyên lý hoạt động
1D-CNN Denoiser hoạt động như một hệ thống **bộ lọc thích nghi thông minh (Trainable Digital Filters)**:
- Không giảm chiều tín hiệu qua thời gian (giữ nguyên độ dài 1024 bằng `padding='same'`).
- Tín hiệu đi qua một chuỗi các lớp tích chập một chiều (`Conv1d`). Mỗi kernel trượt dọc theo tín hiệu và tính tích chập cục bộ:
  $$y[t] = \sum_{k} w[k] \cdot x[t - k] + b$$
- **Trường tiếp nhận (Receptive Field):** Là khoảng độ dài tín hiệu đầu vào mà một nơ-ron ở tầng sâu có thể "nhìn thấy". 
  - Nếu trường tiếp nhận quá nhỏ ($\approx 15 - 30$ mẫu $\approx 0.05 - 0.08$ giây), mô hình chỉ nhìn thấy một sườn dốc của sóng R, không thể nhận biết được sóng trôi chậm (BW) kéo dài hàng trăm mẫu.
  - Để tăng trường tiếp nhận mà không làm bùng nổ số lượng tham số, kiến trúc hiện đại sử dụng **Dilated Convolution (tích chập giãn nở)** với tỉ lệ giãn $d = 1, 2, 4, 8, 16...$.

```
[X_noisy: 1x1024]
       │
       ▼ Conv1D (k=7, d=1, pad=3)  ──► [32 x 1024]  (Học đặc trưng tần số cao: đỉnh nhọn QRS)
       │
       ▼ Conv1D (k=7, d=2, pad=6)  ──► [64 x 1024]  (Mở rộng trường nhìn)
       │
       ▼ Conv1D (k=7, d=4, pad=12) ──► [64 x 1024]  (Mở rộng trường nhìn)
       │
       ▼ Conv1D (k=7, d=8, pad=24) ──► [32 x 1024]  (Nhìn thấy chu kỳ trôi đường đẳng điện BW)
       │
       ▼ Conv1D (k=3, d=1, pad=1)  ──► [X_hat: 1x1024] (Tái cấu trúc dạng sóng)
```

#### B. Cấu trúc đề xuất ở mức khái niệm
- **Đầu vào:** `(Batch, 1, 1024)`
- **Các khối tích chập:** 5 đến 7 tầng Conv1D liên tiếp:
  - Tầng 1: `Conv1d(1, 32, kernel_size=9, padding=4) -> BatchNorm1d -> LeakyReLU`
  - Tầng 2: `Conv1d(32, 64, kernel_size=7, dilation=2, padding=6) -> BatchNorm1d -> LeakyReLU`
  - Tầng 3: `Conv1d(64, 64, kernel_size=7, dilation=4, padding=12) -> BatchNorm1d -> LeakyReLU`
  - Tầng 4: `Conv1d(64, 32, kernel_size=7, dilation=8, padding=24) -> BatchNorm1d -> LeakyReLU`
  - Tầng 5: `Conv1d(32, 1, kernel_size=5, padding=2)` (Đầu ra 1 kênh)
- **Đầu ra:** `(Batch, 1, 1024)`

#### C. Ưu điểm và hạn chế đối với ECG Denoising
- **Ưu điểm:** Bảo toàn hình thái cục bộ cực kỳ tốt. Do không có bước pooling/downsampling, đỉnh sóng R và các khía sắc nhọn của QRS không bị mất độ phân giải thời gian.
- **Hạn chế:** Nếu chỉ dùng CNN thông thường không có dilation, mô hình sẽ gặp khó khăn lớn khi khử **Baseline Wander (BW)** vì nhiễu BW biến thiên rất chậm (chu kỳ vài giây), đòi hỏi kernel size khổng lồ.
- **Độ phức tạp & Edge AI:** Tính toán song song cực nhanh, bộ nhớ đệm nhỏ, cực kỳ phù hợp triển khai Edge AI/Wearable ECG.

---

### 2.3 Model 3 — 1D-ResNet Denoiser (Deep Residual Network)

#### A. Nguyên lý hoạt động
1D-ResNet giải quyết bài toán suy giảm đạo hàm (vanishing gradient) khi mạng sâu bằng cơ chế **đường tắt (Residual / Skip Connection)**:
- Thay vì bắt các tầng học ánh xạ trực tiếp $H(x)$, khối residual bắt các tầng chỉ cần học phần sai khác:
  $$F(x) = H(x) - x \implies H(x) = F(x) + x$$
- **Tại sao ResNet cực kỳ phù hợp với Denoising?**
  Bản chất tín hiệu nhiễu là: $x_{\text{noisy}} = x_{\text{clean}} + n_{\text{noise}}$.
  Có 2 cách tiếp cận trong ResNet:
  1. **Dự đoán tín hiệu sạch:** Mỗi khối bổ sung dần các chi tiết còn thiếu và tinh chỉnh dạng sóng qua từng cấp độ sâu.
  2. **Dự đoán phần dư (Residual Noise Estimation):** Mạng học cách dự đoán chính xác thành phần nhiễu $\hat{n} = \mathcal{M}(x_{\text{noisy}})$, sau đó lấy $x_{\text{clean}} = x_{\text{noisy}} - \hat{n}$.

```
             x (Đầu vào khối)
             │───────────────┐ (Identity shortcut)
             ▼               │
     [ Conv1D + BN + Act ]   │
             │               │
     [ Conv1D + BN ]         │
             │               │
             ▼               │
         Cộng: F(x) + x ◄────┘
             │
             ▼ Act (LeakyReLU)
```

#### B. Cấu trúc đề xuất ở mức khái niệm
- **Đầu vào:** `(Batch, 1, 1024)`
- **Stem Conv:** `Conv1d(1, 32, kernel_size=7, padding=3) -> BatchNorm1d -> LeakyReLU`
- **Residual Blocks:** Chuỗi 4 đến 6 Residual Blocks (mỗi block gồm 2 lớp Conv1D kèm skip connection):
  - Block 1 & 2: 32 filters, kernel 5, dilation 1 & 2
  - Block 3 & 4: 64 filters, kernel 5, dilation 4 & 8 (nếu có channel change, dùng $1\times 1$ conv ở shortcut)
  - Block 5 & 6: 32 filters, kernel 3, dilation 1
- **Global Shortcut (Tùy chọn):** Nối tắt trực tiếp từ đầu vào $X_{\text{noisy}}$ tới đầu ra để mạng hoạt động theo cơ chế trừ nhiễu.
- **Đầu ra:** `(Batch, 1, 1024)`

#### C. Ưu điểm và hạn chế đối với ECG Denoising
- **Ưu điểm:** Khả năng biểu diễn mạnh mẽ nhất trong 3 mô hình. Vừa học được đặc trưng tần số thấp (BW) ở các tầng sâu, vừa bảo toàn nguyên vẹn độ sắc nhọn của đỉnh R nhờ đường truyền thẳng shortcut.
- **Hạn chế:** Số lượng tham số và chi phí tính toán (FLOPs) cao nhất trong 3 mô hình. Có nguy cơ quá khớp (overfitting) với các dạng sóng của tập Train nếu không kiểm soát regularization tốt.
- **Độ phức tạp & Edge AI:** Nặng hơn 1D-CNN từ $2\times$ đến $4\times$, vẫn chạy tốt trên các thiết bị nhúng có NPU/GPU mini (như Raspberry Pi 4, Jetson Nano), nhưng đòi hỏi tối ưu lượng tử hóa (Quantization INT8) nếu chạy trên vi điều khiển năng lượng siêu thấp (ARM Cortex-M).

---

## 3. BẢNG SO SÁNH TỔNG QUAN 3 MÔ HÌNH

| Tiêu chí | Model 1: Denoising Autoencoder (DAE) | Model 2: 1D-CNN Denoiser | Model 3: 1D-ResNet Denoiser |
|---|---|---|---|
| **Cơ chế cốt lõi** | Nén qua Bottleneck rồi tái tạo | Trích xuất đặc trưng cục bộ bằng Conv đa tầng | Học sai số phần dư qua Skip Connection |
| **Bảo toàn đỉnh R / QRS** | Kém - Trung bình (nguy cơ làm mượt) | Rất tốt (giữ nguyên độ phân giải thời gian) | Xuất sắc (vừa sâu vừa giữ chi tiết) |
| **Xử lý trôi đẳng điện (BW)** | Tốt (nhờ biểu diễn trừu tượng ở bottleneck) | Khá (cần Dilated Conv để đủ trường nhìn) | Rất tốt (kết hợp trường nhìn rộng và độ sâu) |
| **Xử lý nhiễu điện cực (EM)** | Trung bình (dễ nhầm xung EM thành sóng R) | Khá | Tốt (nhận diện được các mẫu phức tạp) |
| **Số lượng tham số** | Nhỏ (~50k - 150k) | Rất nhỏ (~30k - 100k) | Trung bình (~150k - 500k) |
| **Tốc độ suy luận (Latency)** | Rất nhanh | Nhanh nhất | Trung bình - Nhanh |
| **Mức độ phù hợp Edge AI** | Cao | Rất cao | Khá cao (cần tối ưu hóa) |
| **Khuyến nghị thứ tự thử nghiệm** | Thử thứ 3 (đối chứng cơ chế nén) | **Thử đầu tiên (Baseline chuẩn)** | **Thử thứ hai (Mô hình nâng cao)** |

---

## 4. SỔ TAY THAM SỐ FINE-TUNE CHI TIẾT (CHO NGƯỜI MỚI BẮT ĐẦU)

### 4.1 Tham số chung cho cả 3 mô hình

| Tên tham số | Phạm vi gợi ý | Tác động khi TĂNG | Tác động khi GIẢM | Dấu hiệu cần điều chỉnh | Phân loại |
|---|---|---|---|---|---|
| **Learning Rate (LR)** | $10^{-3}$ đến $10^{-4}$ | Học nhanh hơn; nếu quá cao sẽ phân kỳ (Loss thành NaN/nhảy loạn) | Học ổn định; nếu quá thấp sẽ học rất chậm hoặc kẹt ở nghiệm xấu | Loss không giảm hoặc rung lắc dữ dội quanh cực tiểu | Tham số học |
| **Batch Size** | 32 hoặc 64 | Gradient ổn định hơn, tính toán song song tốt hơn; tốn VRAM hơn | Gradient ồn hơn (có tính ngẫu nhiên giúp thoát cực tiểu địa phương) | Tràn bộ nhớ GPU (OOM) $\implies$ Giảm Batch Size | Kỹ thuật / Tài nguyên |
| **Optimizer** | Adam hoặc AdamW | AdamW áp dụng chuẩn hóa weight decay độc lập, giúp tổng quát hóa tốt hơn Adam thường | — | Luôn ưu tiên dùng AdamW với `betas=(0.9, 0.999)` | Tối ưu hóa |
| **Loss Function** | MSE, L1 (MAE), hoặc Smooth L1 | **MSE:** Phạt rất nặng sai số lớn, nhưng có xu hướng làm mượt đỉnh nhọn.<br>**L1:** Giữ biên độ sắc nhọn của đỉnh R tốt hơn nhiều. | — | Nếu thấy sóng R dự đoán bị lùn đi $\implies$ Đổi sang L1 Loss hoặc kết hợp $\mathcal{L} = \text{MSE} + \lambda \text{L1}$ | Mục tiêu tối ưu |
| **Weight Decay** | $10^{-4}$ đến $10^{-5}$ | Ép trọng số nhỏ, chống overfitting mạnh | Cho phép mô hình học tự do hơn; dễ overfitting | Train Loss tiếp tục giảm mạnh nhưng Val Loss tăng dần $\implies$ Tăng Weight Decay | Regularization |
| **LR Scheduler** | ReduceLROnPlateau hoặc Cosine | Giúp mô hình hội tụ sâu hơn vào những epoch cuối | — | Val Loss đi ngang không cải thiện sau 5-7 epoch $\implies$ Giảm LR đi $2\times$ hoặc $5\times$ | Tối ưu hóa |
| **Số Epoch & Early Stopping** | 30 - 50 epochs (Patience = 7 - 10) | Học kỹ hơn; chạy lâu hơn | Dừng sớm tránh lãng phí thời gian khi mô hình đã bão hòa | Val SNR không tăng thêm trong 8 epoch liên tiếp | Kiểm soát huấn luyện |
| **Window Length** | **1024 mẫu** (Cố định) | Bao quát nhiều chu kỳ tim hơn | Nhìn hẹp hơn | Đã được chốt trong thiết kế Dataset Level 2 | **Tham số Dataset** |
| **Normalization** | `record_max_abs` (Cố định) | Bảo toàn đúng tỉ lệ công suất vật lý của tín hiệu và nhiễu | — | Tuyệt đối không đổi sang Z-score từng nhịp | **Tham số Dataset** |

---

### 4.2 Tham số riêng của Denoising Autoencoder (DAE)

| Tham số | Giá trị gợi ý | Ý nghĩa kỹ thuật và Sự đánh đổi (Trade-off) |
|---|---|---|
| **Độ sâu Encoder/Decoder** | 3 đến 4 tầng | Càng nhiều tầng, tính trừu tượng càng cao. Nhưng nếu quá sâu, tín hiệu bị thu nhỏ quá mức qua mỗi tầng sẽ làm mất định vị thời gian của đỉnh R. |
| **Kích thước Bottleneck** | $128$ mẫu (nén $8\times$) | **Sự đánh đổi cốt lõi:** Nén càng mạnh ($16\times, 32\times$) thì lọc nhiễu càng sạch nhưng sóng tim càng bị bẹp rúm/làm mượt. Nén nhẹ ($2\times, 4\times$) thì giữ được sóng R nhưng nhiễu vẫn lọt qua. |
| **Phương pháp Upsampling** | `ConvTranspose1d` vs `Upsample + Conv1d` | `ConvTranspose1d` học trọng số linh hoạt nhưng dễ tạo ra gợn sóng bàn cờ (checkerboard artifact). `nn.Upsample(mode='linear') + Conv1d` cho đường cong mượt và tự nhiên hơn. |
| **Số kênh trung gian** | $16 \to 32 \to 64$ | Tăng số kênh giúp bù đắp lượng thông tin bị mất khi giảm chiều dài thời gian. |

---

### 4.3 Tham số riêng của 1D-CNN Denoiser

| Tham số | Giá trị gợi ý | Ý nghĩa kỹ thuật và Tác động đến trường tiếp nhận (Receptive Field) |
|---|---|---|
| **Số lớp Conv1D** | 5 đến 7 lớp | Mỗi lớp mở rộng trường nhìn thêm một khoảng $(k - 1)$. Đủ 6 lớp với kernel 7 sẽ đạt trường nhìn $\approx 37$ mẫu. |
| **Kernel Size** | 7 hoặc 9 ở tầng đầu, 5 ở tầng cuối | Kernel lớn ở tầng đầu giúp gom các mẫu lân cận để dập nhiễu co cơ tần số cao (MA). Kernel nhỏ ở tầng cuối giúp vẽ lại đường nét tinh vi. |
| **Dilation Rate** | $[1, 2, 4, 8, 16]$ | **Cực kỳ quan trọng:** Giúp trường nhìn tăng theo cấp số nhân ($RF > 500$ mẫu) mà không làm tăng số lượng tham số. Đây là chìa khóa để 1D-CNN khử được nhiễu trôi chậm (BW). |
| **Activation Function** | `LeakyReLU(0.1)` | Tránh hiện tượng "chết nơ-ron" của ReLU thông thường khi tín hiệu ECG có nửa chu kỳ âm. |

---

### 4.4 Tham số riêng của 1D-ResNet Denoiser

| Tham số | Giá trị gợi ý | Ý nghĩa kỹ thuật và Cách lựa chọn |
|---|---|---|
| **Số Residual Blocks** | 4 đến 6 blocks | 4 blocks là đủ cho tín hiệu 1024 mẫu. Tăng lên 8 blocks làm mạng quá nặng và dễ quá khớp trên tập Train gồm 19 bệnh nhân. |
| **Số filters cơ sở** | 32 filters (mở rộng lên 64) | Đủ để học đa dạng các dạng biến thiên hình thái nhịp tim (bình thường, ngoại tâm thu, bloc nhánh). |
| **Cơ chế dự đoán** | Dự đoán sạch vs Dự đoán trừ nhiễu | **Khuyên dùng:** Cho mạng dự đoán trực tiếp tín hiệu sạch trước. Cơ chế trừ nhiễu $\hat{x} = x - \hat{n}$ chỉ nên thử nghiệm nếu nhận thấy mô hình gặp khó khăn trong việc học nền tĩnh. |
| **Normalization trong Block** | `BatchNorm1d` | Giúp ổn định phân phối trung gian giữa các tầng sâu, cho phép đẩy Learning Rate cao hơn mà không phân kỳ. |

---

## 5. THỨ TỰ FINE-TUNE ĐỀ XUẤT CHO NGƯỜI MỚI BẮT ĐẦU

Để tránh việc thử nghiệm mò mẫm vô định hướng, hãy tuân theo quy trình 4 giai đoạn chuẩn mực:

```
[ GIAI ĐOẠN 1: CỐ ĐỊNH NỀN TẢNG ]
  Cố định Window 1024, AdamW, Batch Size 32, Seed 42, Kiến trúc mặc định nhỏ gọn.
                   │
                   ▼
[ GIAI ĐOẠN 2: TINH CHỈNH TỐC ĐỘ HỌC VÀ HÀM MẤT MÁT ]
  Tìm Learning Rate tối ưu (1e-3, 5e-4, 1e-4). Thử nghiệm MSE vs L1 Loss.
                   │
                   ▼
[ GIAI ĐOẠN 3: ĐIỀU CHỈNH TRƯỜNG NHÌN VÀ DUNG LƯỢNG MÔ HÌNH ]
  Tăng Dilation Rate nếu nhiễu BW còn sót. Tinh chỉnh tỉ lệ Bottleneck (DAE) hoặc số Block (ResNet).
                   │
                   ▼
[ GIAI ĐOẠN 4: CHỐNG QUÁ KHỚP DỰA TRÊN DẤU HIỆU THỰC TẾ ]
  Chỉ can thiệp Weight Decay, Dropout nếu khoảng cách giữa Train Loss và Val Loss giãn rộng.
```

1. **Những tham số CỐ ĐỊNH BAN ĐẦU:**
   - Window size: $1024$, Stride: $512$.
   - Normalization: `record_max_abs`.
   - Batch size: $32$.
   - Optimizer: `AdamW`.
   - Random seed: $42$.
2. **Những tham số ƯU TIÊN THỬ TRƯỚC:**
   - **Learning Rate:** Bắt đầu với $10^{-3}$, giảm xuống $5 \times 10^{-4}$ và $10^{-4}$. Quan sát tốc độ giảm của Loss trong 5 epoch đầu.
   - **Hàm mất mát:** So sánh **MSE Loss** với **L1 Loss**. Đo đạc xem L1 có bảo toàn biên độ đỉnh R tốt hơn không.
3. **Những tham số CHỈ ĐIỀU CHỈNH KHI CÓ DẤU HIỆU CỤ THỂ:**
   - *Dấu hiệu 1: Đỉnh R bị tù và lùn $\implies$* Chuyển sang L1 Loss, giảm độ nén bottleneck (ở DAE), hoặc thêm shortcut.
   - *Dấu hiệu 2: Đường đẳng điện vẫn bị võng/uốn lượn sau khử nhiễu $\implies$* Tăng Dilation Rate ở 1D-CNN và 1D-ResNet để mở rộng Receptive Field.
   - *Dấu hiệu 3: Train loss giảm rất thấp nhưng Val loss tăng $\implies$* Tăng `weight_decay` từ $10^{-5}$ lên $10^{-4}$, hoặc thêm `Dropout(0.1)`.
4. **Kiểm soát tính công bằng khi so sánh 3 mô hình:**
   - Khi so sánh DAE, 1D-CNN và 1D-ResNet, **bắt buộc phải giữ nguyên cùng một hàm Loss, cùng Learning Rate scheduler, cùng Batch Size và cùng tập dữ liệu đánh giá**. Chỉ thay đổi phần kiến trúc bên trong.
