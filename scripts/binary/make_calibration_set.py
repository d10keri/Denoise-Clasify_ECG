"""
D1 & C3. Tạo tập dữ liệu hiệu chuẩn (Calibration Set) cho lượng tử hóa INT8
và tập dữ liệu kiểm thử trên máy tính (Host Validation Set) cho STM32Cube.AI.

Quy tắc nghiêm ngặt (theo plan/edge_ai_int8_binary_plan.md):
1. Tập Calibration (D1):
   - Lấy HOÀN TOÀN từ tập TRAIN (`data/processed/train.npz`), tuyệt đối không chạm vào TEST.
   - Tổng số: 600 nhịp phân tầng (stratified):
     + 300 nhịp lớp N (lớp 0)
     + 300 nhịp lớp non-N (lớp 1: gồm S, V, F, Q lấy theo tỷ lệ thực trong tập Train;
       lớp Q chỉ có 4 mẫu trong train nên giữ toàn bộ 4 mẫu).
   - Định dạng: `export/calib_train.npy`, kiểu `float32`, kích thước `(600, 1, 260)`.
2. Tập Host Validation (C3):
   - Lấy 500 nhịp phân tầng từ tập TEST (`data/processed/test.npz`):
     + 250 nhịp N + 250 nhịp non-N (đủ S, V, F, Q) để nạp nhanh vào STM32Cube AI Studio GUI
       (Validate on desktop) hoặc chạy script kiểm thử nhanh.
   - Lưu `export/host_val_test500_x.npy` `(500, 1, 260)` và `export/host_val_test500_meta.json`.
"""
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
EXPORT_DIR = PROJECT_ROOT / "export"
SEED = 42
CLASS_NAMES_5 = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}


def build_stratified_subset(
    X: np.ndarray,
    y5: np.ndarray,
    n_normal_target: int,
    n_abnormal_target: int,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    idx_n = np.where(y5 == 0)[0]
    chosen_n = rng.choice(idx_n, size=n_normal_target, replace=False)

    # Phân bổ số lượng cho các lớp con trong non-N (S=1, V=2, F=3, Q=4)
    abn_classes = [1, 2, 3, 4]
    idx_by_c = {c: np.where(y5 == c)[0] for c in abn_classes}
    total_abn = sum(len(idx_by_c[c]) for c in abn_classes)

    # Tính số mẫu mỗi lớp theo tỷ lệ thực, đảm bảo các lớp hiếm (như Q) có mặt đầy đủ
    alloc = {}
    remaining = n_abnormal_target
    # Xử lý lớp siêu hiếm Q trước
    for c in [4]:
        avail = len(idx_by_c[c])
        take = min(avail, max(1, int(round(n_abnormal_target * avail / total_abn))))
        alloc[c] = take
        remaining -= take

    # Phân bổ phần còn lại cho S, V, F theo tỷ lệ trong tập
    svf_total = sum(len(idx_by_c[c]) for c in [1, 2, 3])
    raw_counts = {
        c: int(np.floor(remaining * len(idx_by_c[c]) / svf_total)) for c in [1, 2, 3]
    }
    diff = remaining - sum(raw_counts.values())
    # Cộng phần dư vào lớp lớn nhất (V, sau đó S)
    for c in [2, 1, 3]:
        if diff <= 0:
            break
        raw_counts[c] += 1
        diff -= 1

    for c in [1, 2, 3]:
        alloc[c] = min(raw_counts[c], len(idx_by_c[c]))

    chosen_abn_list = []
    for c in abn_classes:
        chosen_c = rng.choice(idx_by_c[c], size=alloc[c], replace=False)
        chosen_abn_list.append(chosen_c)

    chosen_abn = np.concatenate(chosen_abn_list, axis=0)
    all_indices = np.concatenate([chosen_n, chosen_abn], axis=0)
    rng.shuffle(all_indices)

    X_sub = X[all_indices].astype(np.float32)
    if X_sub.ndim == 2:
        X_sub = np.expand_dims(X_sub, axis=1)  # (N, 1, 260)
    y5_sub = y5[all_indices].astype(np.int64)
    y2_sub = (y5_sub > 0).astype(np.int64)

    breakdown = {CLASS_NAMES_5[c]: int(np.sum(y5_sub == c)) for c in range(5)}
    return X_sub, y5_sub, y2_sub, all_indices, breakdown


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Tạo Calibration Set (600 nhịp từ TRAIN)
    train_npz = np.load(PROCESSED_DIR / "train.npz", allow_pickle=False)
    X_train = train_npz["X"]
    y5_train = train_npz["y"].astype(np.int64)

    X_calib, y5_calib, y2_calib, idx_calib, calib_breakdown = build_stratified_subset(
        X_train, y5_train, n_normal_target=300, n_abnormal_target=300, seed=SEED
    )

    calib_path = EXPORT_DIR / "calib_train.npy"
    np.save(calib_path, X_calib)
    np.savez_compressed(
        EXPORT_DIR / "calib_train_labeled.npz",
        X=X_calib,
        y5=y5_calib,
        y2=y2_calib,
        train_indices=idx_calib,
    )

    # 2. Tạo Host Validation Subset (500 nhịp từ TEST cho STM32Cube.AI Desktop Validation)
    test_npz = np.load(PROCESSED_DIR / "test.npz", allow_pickle=False)
    X_test = test_npz["X"]
    y5_test = test_npz["y"].astype(np.int64)

    X_hval, y5_hval, y2_hval, idx_hval, hval_breakdown = build_stratified_subset(
        X_test, y5_test, n_normal_target=250, n_abnormal_target=250, seed=SEED
    )

    hval_x_path = EXPORT_DIR / "host_val_test500_x.npy"
    np.save(hval_x_path, X_hval)
    np.savez_compressed(
        EXPORT_DIR / "host_val_test500_labeled.npz",
        X=X_hval,
        y5=y5_hval,
        y2=y2_hval,
        test_indices=idx_hval,
    )

    meta = {
        "seed": SEED,
        "calibration_set": {
            "source": "data/processed/train.npz",
            "file_npy": str(calib_path.relative_to(PROJECT_ROOT)),
            "shape": list(X_calib.shape),
            "dtype": str(X_calib.dtype),
            "n_normal": int(np.sum(y2_calib == 0)),
            "n_abnormal": int(np.sum(y2_calib == 1)),
            "class_5_breakdown": calib_breakdown,
            "min_val": round(float(X_calib.min()), 6),
            "max_val": round(float(X_calib.max()), 6),
            "mean_val": round(float(X_calib.mean()), 6),
            "std_val": round(float(X_calib.std()), 6),
        },
        "host_validation_subset": {
            "source": "data/processed/test.npz",
            "file_npy": str(hval_x_path.relative_to(PROJECT_ROOT)),
            "shape": list(X_hval.shape),
            "dtype": str(X_hval.dtype),
            "n_normal": int(np.sum(y2_hval == 0)),
            "n_abnormal": int(np.sum(y2_hval == 1)),
            "class_5_breakdown": hval_breakdown,
        },
    }

    meta_path = EXPORT_DIR / "calibration_and_host_val_info.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("=== HOÀN TẤT BƯỚC D1 & C3: TẠO TẬP CALIBRATION VÀ HOST VALIDATION ===")
    print(f"1. Calibration Set: {calib_path}")
    print(f"   - Shape: {X_calib.shape}, Dtype: {X_calib.dtype}")
    print(f"   - Phân bố: N={np.sum(y2_calib==0)}, non-N={np.sum(y2_calib==1)} -> {calib_breakdown}")
    print(f"2. Host Validation Set: {hval_x_path}")
    print(f"   - Shape: {X_hval.shape}, Dtype: {X_hval.dtype}")
    print(f"   - Phân bố: N={np.sum(y2_hval==0)}, non-N={np.sum(y2_hval==1)} -> {hval_breakdown}")


if __name__ == "__main__":
    main()
