# Phase 2 Implementation Plan: MLP Baseline

## Mục tiêu

Xây dựng baseline MLP đầu tiên để phân loại 5 lớp ECG từ dataset MITDB hiện tại và tạo ra kết quả test có thể báo cáo.

**Ràng buộc cứng:**
- Không sửa bất kỳ file nào từ Phase 1 (dataset, preprocessing, segmentation, splits).
- Không dùng `test.npz` trong bất kỳ bước nào trước T16.
- Tất cả hyperparameters nằm trong config file, không hardcode trong code.

---

## Context Phase 1 (đã hoàn thành)

| Item | Giá trị |
|---|---|
| Dataset splits | train / val / test (inter-patient De Chazal 2004) |
| Input shape | `(260,)` float32 — per-beat z-score normalized |
| Classes | 0=N, 1=S, 2=V, 3=F, 4=Q |
| Train beats | 41,116 |
| Val beats | 9,885 |
| Test beats | 49,691 |
| Class weights (precomputed) | [0.2218, 10.611, 2.865, 20.610, 2055.800] |
| DataLoader ready | `src/data/ecg_dataset.py` → `make_dataloader()` |
| NPZ files | `data/processed/{train,val,test}.npz` |

---

## Cấu trúc thư mục cuối Phase 2

```
Denoise-Clasify_ECG/
├── configs/
│   ├── dataset_config.yaml          # Phase 1 — KHÔNG SỬA
│   └── mlp_config.yaml              # [TẠO MỚI] — T12
│
├── src/
│   ├── data/                        # Phase 1 — KHÔNG SỬA
│   │   ├── ecg_dataset.py
│   │   └── ...
│   └── models/                      # [TẠO MỚI]
│       ├── __init__.py              # [TẠO MỚI] — T10
│       └── mlp.py                   # [TẠO MỚI] — T10
│
├── train.py                         # [SỬA] — T13 (hiện rỗng)
├── evaluate.py                      # [SỬA] — T16 (hiện rỗng)
│
├── checkpoints/
│   └── mlp_baseline/
│       └── best_model.pt            # [TẠO khi train] — T15
│
├── results/
│   └── mlp_baseline/                # [TẠO MỚI] — T17
│       ├── test_metrics.json
│       ├── test_metrics.csv
│       ├── confusion_matrix.png
│       └── training_history.json
│
├── data/
│   ├── processed/                   # Phase 1 — KHÔNG SỬA
│   └── metadata/                    # Phase 1 — KHÔNG SỬA
│
└── plans/
    ├── dataset_plan.md              # Phase 1 — KHÔNG SỬA
    └── phase_2_mlp.md               # file này
```

---

## Task Overview

```
T10 (Model) → T11 (Loss) → T12 (Config) → T13 (Train Loop)
                                               ↓
                                           T14 (Validation)
                                               ↓
                                           T15 (Checkpoint)
                                               ↓
                                           T16 (Test Eval)
                                               ↓
                                           T17 (Metrics & Report)
```

---

## T10 — Model Definition

### Mục tiêu
Tạo MLP baseline đơn giản, không thêm kiến trúc phức tạp.

### Kiến trúc
```
Input(260) → Linear(260→128) → ReLU → Linear(128→64) → ReLU → Linear(64→5)
```

- Input: 260 features (1 beat, flatten)
- Hidden layer 1: 128 neurons, ReLU activation
- Hidden layer 2: 64 neurons, ReLU activation
- Output: 5 logits (không có Softmax — CrossEntropyLoss tự xử lý)
- Không có Dropout, BatchNorm, hay skip connections ở baseline này

### Files tạo mới

| File | Nội dung |
|---|---|
| `src/models/__init__.py` | Export `MLPBaseline` |
| `src/models/mlp.py` | Class `MLPBaseline(nn.Module)` |

### src/models/mlp.py — Skeleton

```python
import torch.nn as nn

class MLPBaseline(nn.Module):
    """
    MLP Baseline: 260 → 128 → 64 → 5
    Input: (batch, 260) float32
    Output: (batch, 5) logits
    """
    def __init__(self, input_size: int, hidden1: int, hidden2: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)
```

### Input / Output

| | Giá trị |
|---|---|
| Input tensor | `(batch_size, 260)` float32 |
| Output tensor | `(batch_size, 5)` float32 logits |

### Dependency
- Không phụ thuộc vào task khác. Có thể làm đầu tiên độc lập.

### Cách test T10

```python
import torch
from src.models.mlp import MLPBaseline

model = MLPBaseline(input_size=260, hidden1=128, hidden2=64, num_classes=5)
x = torch.randn(64, 260)
out = model(x)
assert out.shape == (64, 5), f"Expected (64,5), got {out.shape}"
print("T10 PASS: output shape", out.shape)
```

### Điều kiện PASS T10
- [ ] `out.shape == (64, 5)` với batch_size=64
- [ ] Model chạy được `model.forward()` không lỗi
- [ ] `sum(p.numel() for p in model.parameters())` hợp lý (≈ 42,053 params)

---

## T11 — Loss & Class Weights

### Mục tiêu
Cấu hình `CrossEntropyLoss` với class weights lấy từ **training set** để xử lý class imbalance nặng (Q chỉ có ~20 beats trong train).

### Class Weights — Nguồn và Cách Tính

**Nguồn:** Class weights đã được tính sẵn trong Phase 1 tại `data/metadata/dataset_info.json`:
```json
"class_weights": [0.2218, 10.611, 2.865, 20.610, 2055.800]
```

**Cách tính (sklearn `compute_class_weight='balanced'`):**
```
weight[c] = n_total / (n_classes × n_samples_in_class_c)
```

Ý nghĩa: class hiếm (Q, F) được nhân hệ số lớn hơn, loss contribution của chúng cao hơn → model không thể bỏ qua chúng.

**Cách đưa class weights vào CrossEntropyLoss:**

```python
import torch
import torch.nn as nn
import json

# Load weights từ metadata (chỉ dùng train split để tính)
with open("data/metadata/dataset_info.json") as f:
    info = json.load(f)
class_weights = torch.tensor(info["class_weights"], dtype=torch.float32)

# Đưa vào loss — weights tự động nhân với logits của đúng class
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

**Tại sao không dùng test để tính weight:**
- Test set là unseen data. Dùng bất kỳ thống kê nào từ test set trước khi đánh giá là data leakage.
- Class weights được tính **chỉ** từ `y_train` và đã lưu vào `dataset_info.json` ở Phase 1.

### Files liên quan

| File | Hành động |
|---|---|
| `data/metadata/dataset_info.json` | Đọc (KHÔNG SỬA) |
| `train.py` | Sử dụng class_weights khi khởi tạo loss |

### Input / Output T11

| | Giá trị |
|---|---|
| Input | `dataset_info.json["class_weights"]` |
| Output | `nn.CrossEntropyLoss(weight=tensor([...]))` object |

### Dependency
- T10 phải PASS (cần model để test loss forward)

### Cách test T11

```python
import torch, torch.nn as nn

weights = torch.tensor([0.2218, 10.611, 2.865, 20.610, 2055.800])
criterion = nn.CrossEntropyLoss(weight=weights)

logits = torch.randn(8, 5)
labels = torch.randint(0, 5, (8,))
loss = criterion(logits, labels)

assert loss.item() > 0
assert not torch.isnan(loss)
print(f"T11 PASS: loss = {loss.item():.4f}")
```

### Điều kiện PASS T11
- [ ] Loss > 0, không NaN, không Inf
- [ ] `criterion.weight` phải khớp với 5 class weights từ `dataset_info.json`
- [ ] Weights chỉ lấy từ train metadata, không từ val hoặc test

---

## T12 — Optimizer & Training Config

### Mục tiêu
Tạo file config tập trung cho tất cả hyperparameters Phase 2. **Không hardcode** bất kỳ giá trị nào trong `train.py`.

### File tạo mới: `configs/mlp_config.yaml`

```yaml
# Phase 2 MLP Baseline Configuration
model:
  name: "MLPBaseline"
  input_size: 260          # phải khớp window.size trong dataset_config.yaml
  hidden1: 128
  hidden2: 64
  num_classes: 5

training:
  seed: 42
  epochs: 30
  batch_size: 64
  learning_rate: 1.0e-3    # Adam baseline LR
  optimizer: "adam"
  use_class_weights: true   # load từ dataset_info.json

paths:
  processed_dir: "data/processed"
  metadata_dir: "data/metadata"
  checkpoint_dir: "checkpoints/mlp_baseline"
  results_dir: "results/mlp_baseline"

checkpoint:
  monitor: "val_macro_f1"   # metric để chọn best model
  mode: "max"               # cao hơn = tốt hơn

logging:
  log_every_epoch: true
  save_history: true
```

### Quy tắc load config trong code

```python
import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config("configs/mlp_config.yaml")
lr  = cfg["training"]["learning_rate"]   # 1e-3
```

### Optimizer được khởi tạo như thế nào

```python
import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=cfg["training"]["learning_rate"]    # từ config, không hardcode
)
```

### Input / Output T12

| | Giá trị |
|---|---|
| Input | N/A (tạo mới) |
| Output | `configs/mlp_config.yaml` được load thành `dict` |

### Dependency
- Không phụ thuộc task khác (có thể làm song song T10, T11).

### Cách test T12

```python
import yaml
cfg = yaml.safe_load(open("configs/mlp_config.yaml"))

assert cfg["model"]["input_size"] == 260
assert cfg["model"]["num_classes"] == 5
assert cfg["training"]["seed"] == 42
assert cfg["training"]["epochs"] == 30
assert cfg["training"]["batch_size"] == 64
assert abs(cfg["training"]["learning_rate"] - 1e-3) < 1e-8
print("T12 PASS: config loaded correctly")
```

### Điều kiện PASS T12
- [ ] `mlp_config.yaml` tồn tại và load không lỗi
- [ ] Tất cả fields bắt buộc có mặt: `model`, `training`, `paths`, `checkpoint`, `logging`
- [ ] `input_size=260` khớp Phase 1
- [ ] Không có giá trị nào bị hardcode trong `train.py`

---

## T13 — Training Loop

### Mục tiêu
Implement vòng lặp training chuẩn: load data → forward → loss → backward → optimizer step. Ghi log train loss và train metrics mỗi epoch.

### File sửa: `train.py` (hiện rỗng)

**Cấu trúc `train.py`:**

```python
# train.py
import torch, yaml, json, random
import numpy as np
from pathlib import Path
from src.models.mlp import MLPBaseline
from src.data.ecg_dataset import make_dataloader

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_one_epoch(model, loader, criterion, optimizer, device) -> dict:
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()                 # (1) reset gradients
        logits = model(X_batch)               # (2) forward pass
        loss   = criterion(logits, y_batch)   # (3) compute loss
        loss.backward()                       # (4) backward pass
        optimizer.step()                      # (5) update weights

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y_batch.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(all_labels, all_preds)
    return {"loss": avg_loss, **metrics}

def main():
    cfg = yaml.safe_load(open("configs/mlp_config.yaml"))
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ... khởi tạo model, optimizer, criterion, dataloaders
    # ... vòng lặp epochs: train_one_epoch → validate_one_epoch → checkpoint

if __name__ == "__main__":
    main()
```

### Input / Output T13

| | Giá trị |
|---|---|
| Input | `data/processed/train.npz`, `configs/mlp_config.yaml` |
| Output | Train loss và metrics in ra console mỗi epoch |

### Dependency
- T10 PASS (model)
- T11 PASS (loss)
- T12 PASS (config)

### Cách test T13

```bash
# Chạy 2 epoch để verify không crash
python train.py
# Expected output:
# Epoch 1/30 | Train Loss: X.XXXX | Train Acc: XX.XX% | Train Macro-F1: 0.XXXX
# Epoch 2/30 | Train Loss: X.XXXX | Train Acc: XX.XX% | Train Macro-F1: 0.XXXX
```

- Train loss phải finite (không NaN, không Inf)
- Seed cố định: chạy 2 lần → kết quả epoch 1 giống nhau

### Điều kiện PASS T13
- [ ] `train.py` chạy không lỗi ít nhất 2 epochs
- [ ] In ra train loss và train metrics mỗi epoch
- [ ] Seed cố định: deterministic khi chạy lại
- [ ] Không load bất kỳ data từ `test.npz`

---

## T14 — Validation

### Mục tiêu
Sau mỗi epoch training, chạy validation loop trên `val.npz` để theo dõi tình trạng overfitting và cung cấp signal cho việc chọn checkpoint.

### Quy tắc validation
- `model.eval()` **bắt buộc** trước khi vào validation loop
- `torch.no_grad()` **bắt buộc** để không tính gradients (tiết kiệm memory)
- Không gọi `optimizer.step()` hoặc `loss.backward()` trong validation
- Validation chỉ dùng `val.npz`, không dùng `test.npz`

### Metrics theo dõi

| Metric | Mô tả |
|---|---|
| `val_loss` | CrossEntropyLoss trung bình trên val set |
| `val_accuracy` | % beats phân loại đúng |
| `val_macro_precision` | Macro-averaged Precision (5 classes) |
| `val_macro_recall` | Macro-averaged Recall (5 classes) |
| `val_macro_f1` | **Metric chính** để chọn checkpoint |
| `val_f1_per_class` | F1 riêng của từng class N/S/V/F/Q |

### Skeleton validate_one_epoch

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def validate_one_epoch(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(all_labels, all_preds)
    return {"loss": avg_loss, **metrics}
```

### Input / Output T14

| | Giá trị |
|---|---|
| Input | `data/processed/val.npz`, model weights sau mỗi epoch |
| Output | Dict metrics: `{val_loss, val_accuracy, val_macro_f1, val_f1_per_class}` |

### Dependency
- T13 PASS (training loop phải hoạt động trước)

### Cách test T14

```
# Console output sau mỗi epoch:
Epoch 1/30 | Train Loss: X.XXXX | Val Loss: X.XXXX | Val Macro-F1: 0.XXXX
```

Kiểm tra thêm:
- `model.training == False` trong validation loop (xác nhận `model.eval()` hoạt động)
- Không có gradient được tính (`torch.no_grad()`)

### Điều kiện PASS T14
- [ ] `model.eval()` được gọi trước validation
- [ ] `torch.no_grad()` bao quanh toàn bộ validation loop
- [ ] 5 metrics được tính: Accuracy, Macro P, Macro R, Macro F1, per-class F1
- [ ] Val metrics xuất hiện trong console output mỗi epoch
- [ ] Không gọi `optimizer.step()` hoặc `loss.backward()` trong validation

---

## T15 — Best Checkpoint

### Mục tiêu
Lưu checkpoint tốt nhất dựa trên `val_macro_f1` trong quá trình training. Không bao giờ dùng test set để chọn checkpoint.

### Thông tin lưu vào checkpoint

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_macro_f1": best_val_f1,
    "config": cfg,                  # toàn bộ mlp_config.yaml
}
torch.save(checkpoint, Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pt")
```

| Field | Mục đích |
|---|---|
| `epoch` | Biết model được train đến epoch bao nhiêu |
| `model_state_dict` | Weights để load lại inference |
| `optimizer_state_dict` | Resume training nếu cần |
| `val_macro_f1` | Verify checkpoint đúng khi load |
| `config` | Reproduce lại exact model architecture |

### Logic chọn best checkpoint

```python
best_val_f1 = -1.0

for epoch in range(cfg["training"]["epochs"]):
    train_metrics = train_one_epoch(...)
    val_metrics   = validate_one_epoch(...)

    if val_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = val_metrics["macro_f1"]
        save_checkpoint(model, optimizer, epoch, val_metrics, cfg)
        print(f"  → New best Val Macro-F1: {best_val_f1:.4f} (saved checkpoint)")
```

### Input / Output T15

| | Giá trị |
|---|---|
| Input | Val Macro-F1 mỗi epoch |
| Output | `checkpoints/mlp_baseline/best_model.pt` |

### Dependency
- T14 PASS (phải có val_macro_f1 để so sánh)

### Cách test T15

```python
import torch
ckpt = torch.load("checkpoints/mlp_baseline/best_model.pt", map_location="cpu")

assert "model_state_dict" in ckpt
assert "optimizer_state_dict" in ckpt
assert "epoch" in ckpt
assert "val_macro_f1" in ckpt
assert "config" in ckpt
assert ckpt["config"]["model"]["input_size"] == 260

# Verify model có thể load lại
from src.models.mlp import MLPBaseline
model = MLPBaseline(260, 128, 64, 5)
model.load_state_dict(ckpt["model_state_dict"])
print(f"T15 PASS: best checkpoint epoch={ckpt['epoch']}, val_macro_f1={ckpt['val_macro_f1']:.4f}")
```

### Điều kiện PASS T15
- [ ] File `checkpoints/mlp_baseline/best_model.pt` tồn tại sau training
- [ ] Checkpoint chứa đủ 5 keys: `epoch`, `model_state_dict`, `optimizer_state_dict`, `val_macro_f1`, `config`
- [ ] Model có thể load lại từ checkpoint và forward không lỗi
- [ ] Checkpoint **không** được chọn dựa trên test set

---

## T16 — Final Test / Evaluate

### Mục tiêu
Load best checkpoint và chạy **một lần duy nhất** trên `test.npz`. Đây là lần đầu tiên và duy nhất test set được dùng.

### Quy tắc cứng
- Không training trong bước này
- Không fine-tuning, không hyperparameter tuning dựa trên kết quả test
- Gọi `model.eval()` và `torch.no_grad()` bắt buộc
- Chạy trong `evaluate.py`, không trong `train.py`

### Cấu trúc `evaluate.py`

```python
# evaluate.py
import torch, json, yaml
from pathlib import Path
from src.models.mlp import MLPBaseline
from src.data.ecg_dataset import make_dataloader

def evaluate(config_path: str, checkpoint_path: str):
    cfg  = yaml.safe_load(open(config_path))
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Khởi tạo model từ config (không hardcode architecture)
    model = MLPBaseline(
        input_size  = cfg["model"]["input_size"],
        hidden1     = cfg["model"]["hidden1"],
        hidden2     = cfg["model"]["hidden2"],
        num_classes = cfg["model"]["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()   # BẮT BUỘC

    # Load test set
    test_loader = make_dataloader(
        data_dir   = cfg["paths"]["processed_dir"],
        split      = "test",          # CHỈ test
        batch_size = cfg["training"]["batch_size"],
        shuffle    = False,           # không shuffle khi evaluate
    )

    all_preds, all_labels = [], []
    with torch.no_grad():            # BẮT BUỘC
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.to(device))
            preds  = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_batch.tolist())

    metrics = compute_all_metrics(all_labels, all_preds)
    return metrics

if __name__ == "__main__":
    metrics = evaluate("configs/mlp_config.yaml", "checkpoints/mlp_baseline/best_model.pt")
    print(metrics)
```

### Input / Output T16

| | Giá trị |
|---|---|
| Input | `checkpoints/mlp_baseline/best_model.pt`, `data/processed/test.npz` |
| Output | Dict chứa tất cả test metrics |

### Dependency
- T15 PASS (checkpoint phải tồn tại)

### Cách test T16

```bash
python evaluate.py
# Expected output:
# Test Accuracy:      XX.XX%
# Test Macro-F1:      0.XXXX
# Test Macro-P:       0.XXXX
# Test Macro-R:       0.XXXX
```

### Điều kiện PASS T16
- [ ] `evaluate.py` chạy không lỗi
- [ ] `model.eval()` và `torch.no_grad()` được dùng
- [ ] Chỉ load `test.npz`, không load `train.npz` hay `val.npz` trong evaluate
- [ ] Không có bất kỳ `optimizer.step()` hay `loss.backward()` trong file

---

## T17 — Metrics & Reporting

### Mục tiêu
Tính và lưu toàn bộ kết quả test có thể báo cáo: metrics JSON/CSV, confusion matrix PNG, training history JSON.

### Metrics cần tính

**Tổng thể:**

| Metric | Hàm sklearn |
|---|---|
| Accuracy | `accuracy_score` |
| Macro Precision | `precision_score(average='macro')` |
| Macro Recall | `recall_score(average='macro')` |
| Macro F1 | `f1_score(average='macro')` |

**Per-class (5 classes: N/S/V/F/Q):**

| Metric | Hàm |
|---|---|
| Precision per class | `precision_score(average=None)` |
| Recall per class | `recall_score(average=None)` |
| F1 per class | `f1_score(average=None)` |
| Support per class | `classification_report` |

**Confusion Matrix:**
- Dùng `sklearn.metrics.confusion_matrix`
- Vẽ bằng `matplotlib` + `seaborn.heatmap`
- Normalize theo row (recall per class)

### Files output

| File | Nội dung |
|---|---|
| `results/mlp_baseline/test_metrics.json` | Tất cả metrics dạng JSON |
| `results/mlp_baseline/test_metrics.csv` | Metrics dạng bảng CSV |
| `results/mlp_baseline/confusion_matrix.png` | Heatmap confusion matrix |
| `results/mlp_baseline/training_history.json` | Loss và metrics mỗi epoch |

### Cấu trúc `test_metrics.json`

```json
{
  "model": "MLPBaseline",
  "checkpoint": "checkpoints/mlp_baseline/best_model.pt",
  "best_epoch": 22,
  "test_accuracy": 0.8765,
  "macro_precision": 0.6543,
  "macro_recall": 0.5891,
  "macro_f1": 0.6123,
  "per_class": {
    "N": {"precision": 0.92, "recall": 0.95, "f1": 0.935, "support": 43873},
    "S": {"precision": 0.45, "recall": 0.38, "f1": 0.412, "support": 556},
    "V": {"precision": 0.78, "recall": 0.72, "f1": 0.748, "support": 3788},
    "F": {"precision": 0.30, "recall": 0.25, "f1": 0.272, "support": 388},
    "Q": {"precision": 0.55, "recall": 0.48, "f1": 0.513, "support": 8}
  }
}
```

### Cấu trúc `training_history.json`

```json
{
  "epochs": [1, 2, "...", 30],
  "train_loss": ["..."],
  "val_loss": ["..."],
  "train_accuracy": ["..."],
  "val_accuracy": ["..."],
  "train_macro_f1": ["..."],
  "val_macro_f1": ["..."],
  "best_epoch": 22
}
```

### Input / Output T17

| | Giá trị |
|---|---|
| Input | Test predictions từ T16, training history từ T13/T14 |
| Output | 4 files trong `results/mlp_baseline/` |

### Dependency
- T16 PASS (cần test predictions)
- T13/T14 (cần training history — được thu thập suốt quá trình train)

### Cách test T17

```python
import json
from pathlib import Path

results_dir = Path("results/mlp_baseline")
assert (results_dir / "test_metrics.json").exists()
assert (results_dir / "test_metrics.csv").exists()
assert (results_dir / "confusion_matrix.png").exists()
assert (results_dir / "training_history.json").exists()

with open(results_dir / "test_metrics.json") as f:
    m = json.load(f)
assert "macro_f1" in m
assert "per_class" in m
assert set(m["per_class"].keys()) == {"N", "S", "V", "F", "Q"}
print("T17 PASS: all result files exist and have correct structure")
```

### Điều kiện PASS T17
- [ ] 4 files output tồn tại trong `results/mlp_baseline/`
- [ ] `test_metrics.json` có đủ keys: `test_accuracy`, `macro_f1`, `per_class` với 5 classes
- [ ] `confusion_matrix.png` có thể mở được (valid image)
- [ ] `training_history.json` có `epochs`, `train_loss`, `val_loss`, `val_macro_f1`, `best_epoch`

---

## Dependency Graph đầy đủ

```
T10 (Model) ─────────────────────────┐
T11 (Loss) ──────────────────────────┤
T12 (Config) ────────────────────────┴──→ T13 (Train Loop) ──→ T14 (Validation) ──→ T15 (Checkpoint) ──→ T16 (Test Eval) ──→ T17 (Metrics)
```

---

## Tổng kết files Phase 2

### Files tạo mới

| File | Task |
|---|---|
| `src/models/__init__.py` | T10 |
| `src/models/mlp.py` | T10 |
| `configs/mlp_config.yaml` | T12 |
| `checkpoints/mlp_baseline/best_model.pt` | T15 (generated khi train) |
| `results/mlp_baseline/test_metrics.json` | T17 |
| `results/mlp_baseline/test_metrics.csv` | T17 |
| `results/mlp_baseline/confusion_matrix.png` | T17 |
| `results/mlp_baseline/training_history.json` | T17 |

### Files sửa

| File | Task | Thay đổi |
|---|---|---|
| `train.py` | T13 | Implement training loop (hiện rỗng) |
| `evaluate.py` | T16 | Implement evaluation (hiện rỗng) |

### Files KHÔNG SỬA (Phase 1)

| File | Lý do |
|---|---|
| `src/data/*.py` | Phase 1 đã PASS, không đụng vào |
| `data/processed/*.npz` | Dataset đã finalized |
| `data/metadata/dataset_info.json` | Source of truth, chỉ đọc |
| `configs/dataset_config.yaml` | Phase 1 config |
| `plans/dataset_plan.md` | Phase 1 documentation |

---

## Điều kiện PASS Phase 2 (tổng thể)

| Task | Điều kiện PASS |
|---|---|
| T10 | `MLPBaseline(260,128,64,5).forward(randn(64,260)).shape == (64,5)` |
| T11 | `CrossEntropyLoss(weight=...)` chạy không NaN, weights đúng 5 values |
| T12 | `mlp_config.yaml` load được, 6 required fields có mặt |
| T13 | `train.py` chạy 2 epochs không crash, loss được log mỗi epoch |
| T14 | Val metrics in ra sau mỗi epoch, `model.eval()` + `no_grad()` confirmed |
| T15 | `best_model.pt` tồn tại, load được, 5 keys đủ |
| T16 | `evaluate.py` chạy trên test.npz, không training |
| T17 | 4 result files tồn tại, JSON structure đúng |

**Phase 2 DONE khi tất cả T10–T17 đều PASS.**
