"""
A1. Kiểm tra dữ liệu nhị phân (N vs non-N) trên 3 tập train / val / test.
Chỉ đọc dữ liệu từ data/processed/*.npz, không sửa đổi bất kỳ dữ liệu gốc nào.
Xuất kết quả ra: results/binary/label_counts.json
"""
import json
from pathlib import Path
import numpy as np

PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = Path("results/binary/label_counts.json")
CLASS_NAMES_5 = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
CLASS_NAMES_2 = {0: "N", 1: "non-N (S+V+F+Q)"}


def analyze_split(split: str) -> dict:
    npz_path = PROCESSED_DIR / f"{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    y5 = data["y"].astype(np.int64)
    y2 = (y5 > 0).astype(np.int64)
    total = int(len(y5))

    counts_5 = {
        CLASS_NAMES_5[c]: int(np.sum(y5 == c)) for c in range(5)
    }
    n_normal = int(np.sum(y2 == 0))
    n_abnormal = int(np.sum(y2 == 1))

    result = {
        "split": split,
        "total_beats": total,
        "binary_counts": {
            "N (0)": n_normal,
            "non-N (1)": n_abnormal,
        },
        "binary_ratios": {
            "N_pct": round(100.0 * n_normal / total, 4),
            "non_N_pct": round(100.0 * n_abnormal / total, 4),
        },
        "original_5class_counts": counts_5,
        "non_N_breakdown_pct_of_non_N": {
            CLASS_NAMES_5[c]: round(100.0 * counts_5[CLASS_NAMES_5[c]] / max(n_abnormal, 1), 4)
            for c in range(1, 5)
        },
    }
    return result


def main():
    splits_summary = {}
    for split in ["train", "val", "test"]:
        splits_summary[split] = analyze_split(split)

    # Tính balanced class weights cho bài toán 2 lớp từ tập train: w_c = Total / (2 * N_c)
    train_total = splits_summary["train"]["total_beats"]
    train_n = splits_summary["train"]["binary_counts"]["N (0)"]
    train_abn = splits_summary["train"]["binary_counts"]["non-N (1)"]
    w_n = train_total / (2.0 * train_n)
    w_abn = train_total / (2.0 * train_abn)
    pos_weight_bce = train_n / max(train_abn, 1)

    output = {
        "task": "Binary Classification Label Audit (0 = N, 1 = non-N [S, V, F, Q])",
        "splits": splits_summary,
        "train_binary_class_weights_balanced": {
            "weight_N_0": round(w_n, 6),
            "weight_non_N_1": round(w_abn, 6),
            "raw_list": [round(w_n, 6), round(w_abn, 6)],
            "pos_weight_ratio_N_over_non_N": round(pos_weight_bce, 6),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=== KẾT QUẢ KIỂM TRA NHÃN NHỊ PHÂN (A1) ===")
    for split, info in splits_summary.items():
        n0 = info["binary_counts"]["N (0)"]
        n1 = info["binary_counts"]["non-N (1)"]
        p0 = info["binary_ratios"]["N_pct"]
        p1 = info["binary_ratios"]["non_N_pct"]
        c5 = info["original_5class_counts"]
        print(
            f"[{split.upper():5s}] Total: {info['total_beats']:6,d} | "
            f"N(0): {n0:6,d} ({p0:5.2f}%) | non-N(1): {n1:5,d} ({p1:5.2f}%) "
            f"--> [S:{c5['S']}, V:{c5['V']}, F:{c5['F']}, Q:{c5['Q']}]"
        )
    print(
        f"\nBalanced Class Weights (Train 2-class): "
        f"w_N = {w_n:.4f}, w_non_N = {w_abn:.4f} (Tỷ lệ N/non-N = {pos_weight_bce:.2f}x)"
    )
    print(f"Đã lưu kết quả tại: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
