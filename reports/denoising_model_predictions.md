# BÁO CÁO DỰ ĐOÁN VÀ SO SÁNH LÝ THUYẾT BA MÔ HÌNH ECG DENOISING

> **Dự án:** ECG Denoising & Classification (Level 2 — Phase 2)  
> **Mục tiêu:** Đưa ra các dự đoán định tính trước thực nghiệm dựa trên đặc trưng dữ liệu đã thẩm định và nguyên lý học sâu.  
> **Nguyên tắc khoa học:** Phân biệt rạch ròi giữa kỳ vọng lý thuyết và kết quả đo đạc thực tế; **tuyệt đối không tự bịa đặt các con số định lượng (như dB hay RMSE) chưa qua đo lường**.

---

## 1. BỐI CẢNH DỮ LIỆU VÀ THỬ THÁCH THỰC NGHIỆM

Khi đánh giá hiệu năng khử nhiễu của các mô hình trên tập dữ liệu Level 2 hiện tại, có ba yếu tố thực tế mang tính quyết định:
1. **Đặc trưng độ dài cửa sổ (1024 mẫu $\approx 2.84$ giây):** Cửa sổ chứa từ 2 đến 4 nhịp tim. Mô hình phải vừa nắm bắt được hình thái siêu ngắn của đỉnh R ($\approx 20 - 40\text{ ms} \approx 7 - 15\text{ mẫu}$), vừa phải theo dõi được độ cong thoải của đường trôi đẳng điện ($\approx 1 - 2\text{ giây} \approx 360 - 720\text{ mẫu}$).
2. **Ba loại nhiễu với cơ chế vật lý hoàn toàn khác nhau:**
   - `bw` (Baseline Wander): Nhiễu tần số rất thấp ($< 1\text{ Hz}$).
   - `ma` (Muscle Artifact): Nhiễu dải rộng tần số cao, có tính ngẫu nhiên.
   - `em` (Electrode Motion): Nhiễu chuyển động điện cực, có xung đột biến biên độ lớn, phổ tần số trùng một phần với phức bộ QRS.
3. **Phân chia độc lập bệnh nhân và khối thời gian nhiễu:** Mô hình không thể học vẹt dạng nhiễu tĩnh mà bắt buộc phải học tính quy luật tổng quát của sóng tim để lọc bỏ nhiễu.

---

## 2. BẢNG DỰ ĐOÁN NĂNG LỰC ĐỊNH TÍNH CHO TỪNG MÔ HÌNH

### 2.1 Model 1 — Denoising Autoencoder (DAE)

| Khía cạnh | Phân tích và Dự đoán lý thuyết |
|---|---|
| **Kỳ vọng về ưu điểm** | - Khả năng loại bỏ nhiễu ngẫu nhiên và nhiễu có biên độ áp đảo cực kỳ mạnh mẽ, đặc biệt ở mức SNR âm ($-6\text{ dB}, 0\text{ dB}$).<br>- Cấu trúc nút cổ chai (bottleneck) ép mạng chỉ lưu giữ các thành phần năng lượng chính, giúp đường đẳng điện sau khử nhiễu rất phẳng và sạch. |
| **Hạn chế có thể xuất hiện** | - **Nguy cơ làm mượt (Oversmoothing) cao nhất:** Do thông tin thời gian bị nén $4\times - 8\times$, biên độ đỉnh R gần như chắc chắn sẽ bị suy giảm (R-peak attenuation).<br>- Dễ làm mất sóng P hoặc sóng T nhỏ nếu nút cổ chai quá hẹp.<br>- Nguy cơ xuất hiện gợn sóng nhân tạo nếu dùng `ConvTranspose1d`. |
| **Điều kiện phát huy tốt** | - Kích thước nút cổ chai được tinh chỉnh vừa vặn (không nén quá mức).<br>- Tín hiệu đầu vào bị nhiễm nhiễu rất nặng khiến việc tái tạo đại thể quan trọng hơn việc bảo toàn từng vi chi tiết. |
| **Khả năng cạnh tranh** | - Có thể đạt SNR cải thiện tốt ở mức SNR âm, nhưng chỉ số biến dạng hình thái (như PRD hay tương quan sóng) dự kiến sẽ kém hơn các mô hình giữ nguyên độ phân giải. |
| **Mức độ tin cậy của dự đoán** | **Rất cao (High Confidence):** Đây là hiện tượng thắt nút cổ chai kinh điển đã được chứng minh trong xử lý ảnh và tín hiệu âm thanh/y sinh. |

---

### 2.2 Model 2 — 1D-CNN Denoiser (Fully Convolutional / Dilated)

| Khía cạnh | Phân tích và Dự đoán lý thuyết |
|---|---|
| **Kỳ vọng về ưu điểm** | - **Bảo toàn hình thái QRS và đỉnh R rất tốt:** Do giữ nguyên độ dài 1024 mẫu qua toàn bộ các tầng (không pooling), độ sắc nhọn của sóng tim không bị mất mát thông tin vị trí.<br>- Triệt tiêu nhiễu cơ (MA) tần số cao rất tự nhiên nhờ cơ chế hoạt động tương đương các bộ lọc thông thấp thích nghi.<br>- Tốc độ huấn luyện và suy luận nhanh nhất, số lượng tham số ít nhất. |
| **Hạn chế có thể xuất hiện** | - Nếu không dùng **Dilated Convolution** (tích chập giãn nở), trường tiếp nhận sẽ quá hẹp, khiến mô hình gần như "bó tay" trước nhiễu trôi đường đẳng điện chậm (`bw`).<br>- Dễ bị nhiễu điện cực (`em`) đột biến lọt qua nếu kernel chỉ nhìn cục bộ. |
| **Điều kiện phát huy tốt** | - Thiết kế mạng có trường tiếp nhận đủ rộng ($RF > 400\text{ mẫu}$) thông qua việc tăng hệ số dilation ($1, 2, 4, 8, 16$).<br>- Phù hợp nhất khi bài toán ưu tiên độ trung thực của dạng sóng nhịp tim cho bước phân loại tiếp theo. |
| **Khả năng cạnh tranh** | - Dự đoán sẽ vượt trội DAE về khả năng bảo toàn đỉnh R và độ chính xác hình thái.<br>- Có thể ngang ngửa hoặc chỉ thua một khoảng nhỏ so với ResNet ở các ca nhiễu phức tạp (`em`), nhưng bù lại nhẹ hơn nhiều. |
| **Mức độ tin cậy của dự đoán** | **Cao (High Confidence):** Dựa trên đặc tính toán học của phép tích chập 1D bảo toàn kích thước. |

---

### 2.3 Model 3 — 1D-ResNet Denoiser (Deep Residual Network)

| Khía cạnh | Phân tích và Dự đoán lý thuyết |
|---|---|
| **Kỳ vọng về ưu điểm** | - **Cân bằng toàn diện nhất:** Vừa có độ sâu mạng lớn để trích xuất ngữ cảnh trừu tượng, vừa có đường truyền tắt (skip connection) truyền thẳng các chi tiết tần số cao (sườn dốc sóng R) về đầu ra.<br>- Khả năng xử lý tốt nhất đối với loại nhiễu khó nhất là `em` (dịch chuyển điện cực) nhờ phân cấp đặc trưng nhiều tầng.<br>- Ít bị hiện tượng biến mất đạo hàm, huấn luyện ổn định dù tăng số tầng. |
| **Hạn chế có thể xuất hiện** | - **Nguy cơ quá khớp (Overfitting):** Số lượng trọng số lớn nhất. Với tập Train gồm 19 bệnh nhân, mô hình có thể ghi nhớ một số đặc trưng hình thái tim cụ thể thay vì học quy luật khử nhiễu thuần túy.<br>- Tốn tài nguyên tính toán và bộ nhớ VRAM hơn $2\times - 3\times$ so với 1D-CNN cơ bản. |
| **Điều kiện phát huy tốt** | - Cần áp dụng điều chuẩn (Regularization) hợp lý: Weight Decay, Dropout nhẹ, và điều tiết số lượng Residual Blocks (khoảng 4 đến 6 blocks, không nên quá sâu).<br>- Huấn luyện đủ số epoch với Learning Rate scheduler phù hợp. |
| **Khả năng cạnh tranh** | - Dự đoán có tiềm năng đạt tổng điểm cao nhất về cả mức độ giảm lỗi tái tạo (MSE/RMSE) lẫn độ sạch của tín hiệu đầu ra trên đa dạng các loại nhiễu. |
| **Mức độ tin cậy của dự đoán** | **Trung bình - Cao (Medium-to-High Confidence):** Về mặt lý thuyết ResNet có dung lượng lớn nhất, nhưng mức độ vượt trội thực tế so với 1D-CNN cần được kiểm chứng bằng thực nghiệm chạy đo lường. |

---

## 3. SO SÁNH KHẢ NĂNG XỬ LÝ THEO TỪNG LOẠI NHIỄU CỤ THỂ

```
                 BW (Trôi chậm <1Hz)      MA (Nhiễu cơ dải rộng)      EM (Xung đột biến điện cực)
               ┌───────────────────────┬───────────────────────────┬────────────────────────────┐
DAE            │ Tốt (Nén khử trôi tốt)│ Khá (Có thể mờ đỉnh nhọn) │ Kém-TB (Dễ nhầm xung EM)   │
1D-CNN         │ Khá (Cần Dilation lớn)│ Rất tốt (Lọc tần số cao)  │ Khá                        │
1D-ResNet      │ Rất tốt               │ Rất tốt                   │ Tốt nhất                   │
               └───────────────────────┴───────────────────────────┴────────────────────────────┘
```

1. **Đối với nhiễu Baseline Wander (`bw`):**
   - *DAE:* Nén tín hiệu vào bottleneck giúp loại bỏ rất tốt xu hướng trôi tần số thấp, vì sóng trôi chậm bị ép thành một vài thông số trung bình.
   - *1D-CNN:* Phụ thuộc hoàn toàn vào Receptive Field. Nếu Receptive Field $< 200$ mẫu, tín hiệu đầu ra vẫn sẽ bị lượn sóng. Nếu có Dilated Conv ($RF > 500$ mẫu), 1D-CNN xử lý tốt tương đương DAE.
   - *1D-ResNet:* Xử lý rất tốt nhờ mạng sâu kết hợp trường nhìn rộng.
2. **Đối với nhiễu Muscle Artifact (`ma`):**
   - *DAE:* Lọc sạch gợn nhiễu, nhưng có nguy cơ gọt bớt các khía tự nhiên của sóng tim.
   - *1D-CNN & 1D-ResNet:* Là thế mạnh tuyệt đối. Các kernel tích chập hoạt động như các bộ lọc FIR tối ưu, gạt bỏ thành phần nhiễu tần số cao mà vẫn giữ đúng biên độ đỉnh R.
3. **Đối với nhiễu Electrode Motion (`em`):**
   - Đây là loại nhiễu khó nhất vì các xung điện cực có thể có biên độ lớn và thời gian kéo dài tương tự một nhịp ngoại tâm thu thất (PVC).
   - *DAE:* Dễ bị "lừa", tái tạo xung nhiễu EM thành một phức bộ QRS giả.
   - *1D-ResNet:* Nhờ khả năng phân tích đa cấp độ sâu, ResNet có cơ hội phân biệt tốt nhất giữa hình thái QRS sinh lý và xung dịch chuyển điện cực giả tạo.

---

## 4. ĐÁNH GIÁ KHẢ NĂNG TRIỂN KHAI EDGE AI / THIẾT BỊ ĐEO

Khi triển khai trên các thiết bị y tế cá nhân hoặc vòng đeo tay theo dõi tim mạch liên tục (Edge AI):
- **1D-CNN:** Là ứng viên **số 1 về tính khả thi Edge AI**. Kiến trúc hoàn toàn là các phép tích chập trực tiếp nối tiếp nhau, dễ dàng chuyển đổi sang định dạng ONNX, TensorRT, TFLite, hoặc lượng tử hóa sang INT8 mà không bị suy giảm độ chính xác.
- **DAE:** Khá tốt, nhưng các lớp Upsampling hoặc ConvTranspose1D đôi khi gặp hạn chế về mức độ hỗ trợ tối ưu hóa trên một số dòng chip DSP/MCU giá rẻ.
- **1D-ResNet:** Phù hợp với các thiết bị biên có năng lực tính toán tầm trung (như Raspberry Pi, vi xử lý y tế chuyên dụng ARM Cortex-A), nhưng sẽ tốn pin hơn do số lượng phép tính (FLOPs) lớn hơn.

---

## 5. KHUYẾN NGHỊ MÔ HÌNH NÊN ƯU TIÊN NGHIÊN CỨU TRƯỚC

Dựa trên phân tích kỹ thuật và trình độ bắt đầu của người học, **lộ trình nghiên cứu và thực nghiệm được khuyến nghị như sau**:

### Ưu tiên số 1: Nghiên cứu và xây dựng 1D-CNN Denoiser trước tiên
- **Lý do kỹ thuật:**
  1. **Tính trực quan và dễ hiểu:** 1D-CNN là cấu trúc thẳng một chiều (feed-forward), không có cơ chế nén-giải nén phức tạp như Autoencoder và không có cấu trúc phân nhánh như ResNet.
  2. **Tốc độ lặp nhanh (Fast Iteration):** Mô hình nhẹ, huấn luyện nhanh, giúp bạn nhanh chóng hoàn thiện pipeline huấn luyện, vòng lặp đánh giá và kiểm tra hàm mất mát.
  3. **Tạo mốc chuẩn (Baseline Benchmark) vững chắc:** Kết quả của 1D-CNN sẽ đóng vai trò là "thước đo chuẩn" để bạn đánh giá xem: liệu khi chuyển sang mô hình phức tạp hơn (như ResNet) thì hiệu năng có thực sự tăng lên tương xứng với chi phí tính toán hay không.

### Ưu tiên số 2: Mở rộng lên 1D-ResNet Denoiser
- **Lý do kỹ thuật:**
  Ngay sau khi đã có baseline 1D-CNN hoạt động trơn tru, bạn chỉ cần thay thế các khối tích chập đơn lẻ bằng các Residual Block. Việc này cho phép bạn kiểm chứng trực tiếp: *"Liệu cơ chế Skip Connection có thực sự giải quyết được bài toán nhiễu EM và khôi phục sóng tim tốt hơn CNN cơ bản hay không?"*

### Ưu tiên số 3: Thử nghiệm Denoising Autoencoder (DAE) để đối chứng
- **Lý do kỹ thuật:**
  DAE là đại diện cho trường phái nén thông tin qua không gian ẩn (latent space). Việc thử nghiệm DAE sẽ giúp bạn quan sát trực quan sự đánh đổi kinh điển giữa **mức độ làm sạch nhiễu** và **nguy cơ làm mượt mất đỉnh R**.

---

> [!NOTE]
> **Tuyên bố về tính chất của báo cáo:**  
> Toàn bộ các so sánh trên đây là nhận định khoa học mang tính định tính và suy luận kiến trúc trước thực nghiệm. Mọi kết luận định lượng chính xác (như mức tăng SNR bao nhiêu dB, sai số RMSE cụ thể trên từng loại nhiễu) sẽ chỉ được khẳng định khi bước huấn luyện và đánh giá thực nghiệm chính thức được thực hiện ở các phase tiếp theo.
