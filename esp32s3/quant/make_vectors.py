"""
Phase D2: Generate targeted 50-beat firmware verification vectors (test_vectors.h),
logit-diff decision thresholds (thresholds.h), and metadata (test_vectors_meta.json),
and verify Gate D (compilation with GCC/G++ + disagreement beat check).
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_NPZ = PROJECT_ROOT / "data" / "processed" / "test.npz"
LOGITS_NPZ = PROJECT_ROOT / "esp32s3" / "quant" / "test49691_seed42_logits.npz"
HOST_EVAL_JSON = PROJECT_ROOT / "esp32s3" / "results" / "host_eval.json"
QUANT_DIR = PROJECT_ROOT / "esp32s3" / "quant"
MAIN_DIR = PROJECT_ROOT / "esp32s3" / "ecg_qemu" / "main"
RESULTS_DIR = PROJECT_ROOT / "esp32s3" / "results"
GCC_PATH = Path(r"C:\msys64\ucrt64\bin\gcc.exe")
GPP_PATH = Path(r"C:\msys64\ucrt64\bin\g++.exe")

SHORTLIST = ["b_int16", "c_w8a16", "c3", "c1"]
ALL_MODES = ["fp32"] + SHORTLIST
CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def prob_to_logit_diff(th: float) -> float:
    return math.log(th / (1.0 - th))


def fmt_f32(v: float, prec: int = 9) -> str:
    s = f"{float(v):.{prec}g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s + "f"


def get_output_quant_params(cfg: str):
    """Read output tensor ('logits') scale, exponent, and bit_width from PPQ json."""
    jpath = QUANT_DIR / f"{cfg}.json"
    data = json.loads(jpath.read_text(encoding="utf-8"))
    gemm_cfg = data["configs"]["/classifier/Gemm"]["logits"]
    dom = str(gemm_cfg["dominator"])
    scale = float(data["values"][dom]["scale"])
    exp = int(round(math.log2(scale)))
    assert abs(scale - (2.0 ** exp)) < 1e-12, f"{cfg}: non-pow2 scale {scale}"
    bit_width = int(gemm_cfg["bit_width"])
    qmin = int(gemm_cfg["quant_min"])
    qmax = int(gemm_cfg["quant_max"])
    return {
        "scale": scale,
        "exp_out": exp,
        "bit_width": bit_width,
        "quant_min": qmin,
        "quant_max": qmax,
    }


def select_50_beats(X: np.ndarray, y5: np.ndarray, logits_map: dict, th_logit_diff: dict, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(y5)

    # Binary predictions using each config's VAL Youden's J logit-diff threshold
    preds = {}
    for m in ALL_MODES:
        lg = logits_map[m]
        preds[m] = (lg[:, 1] - lg[:, 0]) > th_logit_diff[m]

    pred_fp32 = preds["fp32"]
    disagree_per_cfg = {cfg: np.where(preds[cfg] != pred_fp32)[0] for cfg in SHORTLIST}
    any_disagree = np.zeros(n, dtype=bool)
    for cfg in SHORTLIST:
        any_disagree |= (preds[cfg] != pred_fp32)

    selected = []
    selected_set = set()
    categories = {}

    def add_indices(idxs, cat_name, limit):
        added = 0
        for idx in idxs:
            idx = int(idx)
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
                categories[idx] = cat_name
                added += 1
                if added >= limit:
                    break
        return added

    # 1. 20 disagreement beats: prioritize diverse coverage across b_int16, c_w8a16, c3, c1
    quotas = [("b_int16", 4), ("c_w8a16", 5), ("c3", 5), ("c1", 6)]
    for cfg, q in quotas:
        pool = np.array([i for i in disagree_per_cfg[cfg] if i not in selected_set], dtype=np.int64)
        if len(pool) > 0:
            chosen = rng.choice(pool, size=min(q, len(pool)), replace=False)
            add_indices(chosen, "disagreement", q)

    # Fill up to 20 from any remaining disagreement beats
    if len(selected) < 20:
        rem_pool = np.array([i for i in np.where(any_disagree)[0] if i not in selected_set], dtype=np.int64)
        need = 20 - len(selected)
        chosen = rng.choice(rem_pool, size=need, replace=False)
        add_indices(chosen, "disagreement", need)

    assert len(selected) == 20, f"Expected 20 disagreement beats, got {len(selected)}"

    # 2. 10 class F beats (y5 == 3)
    f_pool = np.array([i for i in np.where(y5 == 3)[0] if i not in selected_set], dtype=np.int64)
    chosen_f = rng.choice(f_pool, size=10, replace=False)
    add_indices(chosen_f, "class_F", 10)

    # 3. 10 class V high-peak beats (y5 == 2, top 5% of max(|x|))
    peaks = np.max(np.abs(X), axis=1)
    v_all = np.where(y5 == 2)[0]
    p95_v = float(np.percentile(peaks[v_all], 95))
    v_high_pool = np.array(
        [i for i in v_all if peaks[i] >= p95_v and i not in selected_set], dtype=np.int64
    )
    chosen_v = rng.choice(v_high_pool, size=10, replace=False)
    # Sort high-peak V beats by peak descending for readability
    chosen_v = chosen_v[np.argsort(-peaks[chosen_v])]
    add_indices(chosen_v, "class_V_highpeak", 10)

    # 4. 10 random class N beats (y5 == 0)
    n_pool = np.array([i for i in np.where(y5 == 0)[0] if i not in selected_set], dtype=np.int64)
    chosen_n = rng.choice(n_pool, size=10, replace=False)
    add_indices(chosen_n, "class_N_random", 10)

    assert len(selected) == 50 and len(selected_set) == 50
    return selected, categories, preds, peaks, p95_v


def write_thresholds_h(out_path: Path, th_prob: dict, th_logit_diff: dict):
    content = f"""/* Auto-generated by esp32s3/quant/make_vectors.py */
/* Decision thresholds from Phase C2 (Seed 42 VAL Youden's J). */
/* Decision rule on MCU (no softmax needed): (logit[1] - logit[0]) > TH_LOGIT_DIFF_<MODE> */

#ifndef THRESHOLDS_H
#define THRESHOLDS_H

typedef enum {{
    ECG_MODE_FP32    = 0,
    ECG_MODE_B_INT16 = 1,
    ECG_MODE_C_W8A16 = 2,
    ECG_MODE_C3      = 3,
    ECG_MODE_C1      = 4,
    ECG_MODE_COUNT   = 5
}} ecg_mode_t;

/* Probability thresholds selected on VAL (Youden's J, Seed 42) */
#define TH_PROB_FP32     ({th_prob['fp32']:.6f}f)
#define TH_PROB_B_INT16  ({th_prob['b_int16']:.6f}f)
#define TH_PROB_C_W8A16  ({th_prob['c_w8a16']:.6f}f)
#define TH_PROB_C3       ({th_prob['c3']:.6f}f)
#define TH_PROB_C1       ({th_prob['c1']:.6f}f)

/* Equivalent logit-difference thresholds: ln(th / (1 - th)) */
#define TH_LOGIT_DIFF_FP32     ({th_logit_diff['fp32']:.9f}f)
#define TH_LOGIT_DIFF_B_INT16  ({th_logit_diff['b_int16']:.9f}f)
#define TH_LOGIT_DIFF_C_W8A16  ({th_logit_diff['c_w8a16']:.9f}f)
#define TH_LOGIT_DIFF_C3       ({th_logit_diff['c3']:.9f}f)
#define TH_LOGIT_DIFF_C1       ({th_logit_diff['c1']:.9f}f)

/* Fixed FP32 VAL threshold for cross-config fixed-threshold comparison */
#define TH_LOGIT_DIFF_FIXED_FP32 TH_LOGIT_DIFF_FP32

static const float ECG_TH_LOGIT_DIFF[ECG_MODE_COUNT] = {{
    TH_LOGIT_DIFF_FP32,
    TH_LOGIT_DIFF_B_INT16,
    TH_LOGIT_DIFF_C_W8A16,
    TH_LOGIT_DIFF_C3,
    TH_LOGIT_DIFF_C1
}};

static const char *const ECG_MODE_NAMES[ECG_MODE_COUNT] = {{
    "fp32",
    "b_int16",
    "c_w8a16",
    "c3",
    "c1"
}};

static inline int ecg_decide_binary(float logit0, float logit1, float th_logit_diff) {{
    return ((logit1 - logit0) > th_logit_diff) ? 1 : 0;
}}

#endif /* THRESHOLDS_H */
"""
    out_path.write_text(content, encoding="utf-8")


def write_test_vectors_h(
    out_path: Path,
    selected: list,
    X: np.ndarray,
    y5: np.ndarray,
    logits_map: dict,
    int_logits_map: dict,
    qparams: dict,
):
    n = len(selected)
    lines = [
        "/* Auto-generated by esp32s3/quant/make_vectors.py */",
        "/* 50 targeted TEST beats: 20 disagreement + 10 class F + 10 high-peak V + 10 random N */",
        "",
        "#ifndef TEST_VECTORS_H",
        "#define TEST_VECTORS_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define NUM_TEST_VECTORS {n}",
        "#define TEST_VECTOR_LEN  260",
        "",
        f"#define EXP_OUT_B_INT16 ({qparams['b_int16']['exp_out']})",
        f"#define EXP_OUT_C_W8A16 ({qparams['c_w8a16']['exp_out']})",
        f"#define EXP_OUT_C3      ({qparams['c3']['exp_out']})",
        f"#define EXP_OUT_C1      ({qparams['c1']['exp_out']})",
        "",
    ]

    # Test indices
    lines.append(f"static const int32_t TEST_INDICES[NUM_TEST_VECTORS] = {{")
    for i in range(0, n, 10):
        lines.append("    " + ", ".join(str(idx) for idx in selected[i : i + 10]) + ",")
    lines.append("};\n")

    # 5-class and binary labels
    y5_sel = [int(y5[idx]) for idx in selected]
    y2_sel = [1 if c != 0 else 0 for c in y5_sel]
    lines.append("static const uint8_t TEST_Y5[NUM_TEST_VECTORS] = {")
    for i in range(0, n, 10):
        lines.append("    " + ", ".join(str(v) for v in y5_sel[i : i + 10]) + ",")
    lines.append("};\n")

    lines.append("static const uint8_t TEST_Y2[NUM_TEST_VECTORS] = {")
    for i in range(0, n, 10):
        lines.append("    " + ", ".join(str(v) for v in y2_sel[i : i + 10]) + ",")
    lines.append("};\n")

    # Waveforms (50 x 260 floats = 52 KB)
    lines.append("static const float TEST_X[NUM_TEST_VECTORS][TEST_VECTOR_LEN] = {")
    for bi, idx in enumerate(selected):
        row = X[idx]
        lines.append(f"    /* Beat {bi} (TEST idx={idx}, y5={CLASS_NAMES[y5_sel[bi]]}) */ {{")
        for j in range(0, 260, 10):
            chunk = row[j : j + 10]
            lines.append("        " + ", ".join(fmt_f32(v, 8) for v in chunk) + ",")
        lines.append("    },")
    lines.append("};\n")

    # Float reference logits for FP32 and each shortlisted config
    def append_float_logits(c_name: str, key: str):
        arr = logits_map[key][selected]
        lines.append(f"static const float {c_name}[NUM_TEST_VECTORS][2] = {{")
        for bi in range(n):
            lines.append(f"    {{{fmt_f32(arr[bi, 0], 9)}, {fmt_f32(arr[bi, 1], 9)}}},")
        lines.append("};\n")

    append_float_logits("REF_LOGITS_FP32", "fp32")
    append_float_logits("REF_LOGITS_B_INT16", "b_int16")
    append_float_logits("REF_LOGITS_C_W8A16", "c_w8a16")
    append_float_logits("REF_LOGITS_C3", "c3")
    append_float_logits("REF_LOGITS_C1", "c1")

    # Exact integer logits for each shortlisted config
    def append_int_logits(c_type: str, c_name: str, key: str):
        arr = int_logits_map[key][selected]
        lines.append(f"static const {c_type} {c_name}[NUM_TEST_VECTORS][2] = {{")
        for bi in range(n):
            lines.append(f"    {{{int(arr[bi, 0])}, {int(arr[bi, 1])}}},")
        lines.append("};\n")

    append_int_logits("int16_t", "REF_INT_LOGITS_B_INT16", "b_int16")
    append_int_logits("int16_t", "REF_INT_LOGITS_C_W8A16", "c_w8a16")
    append_int_logits("int16_t", "REF_INT_LOGITS_C3", "c3")
    append_int_logits("int8_t", "REF_INT_LOGITS_C1", "c1")

    lines.append("#endif /* TEST_VECTORS_H */")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_headers_compile(main_dir: Path, gcc_path: Path, gpp_path: Path):
    test_src = QUANT_DIR / "_tmp_verify_headers.c"
    test_exe = QUANT_DIR / "_tmp_verify_headers.exe"
    test_src.write_text(
        """#include <stdio.h>
#include "thresholds.h"
#include "test_vectors.h"

int main(void) {
    int disagree_cnt = 0;
    for (int i = 0; i < 20; ++i) {
        int p_fp32 = ecg_decide_binary(REF_LOGITS_FP32[i][0], REF_LOGITS_FP32[i][1], TH_LOGIT_DIFF_FP32);
        int p_b    = ecg_decide_binary(REF_LOGITS_B_INT16[i][0], REF_LOGITS_B_INT16[i][1], TH_LOGIT_DIFF_B_INT16);
        int p_w8   = ecg_decide_binary(REF_LOGITS_C_W8A16[i][0], REF_LOGITS_C_W8A16[i][1], TH_LOGIT_DIFF_C_W8A16);
        int p_c3   = ecg_decide_binary(REF_LOGITS_C3[i][0], REF_LOGITS_C3[i][1], TH_LOGIT_DIFF_C3);
        int p_c1   = ecg_decide_binary(REF_LOGITS_C1[i][0], REF_LOGITS_C1[i][1], TH_LOGIT_DIFF_C1);
        if (p_b != p_fp32 || p_w8 != p_fp32 || p_c3 != p_fp32 || p_c1 != p_fp32) {
            disagree_cnt++;
        }
    }
    printf("COMPILE_CHECK_OK vectors=%d disagree_20=%d/20\\n", NUM_TEST_VECTORS, disagree_cnt);
    return (disagree_cnt == 20) ? 0 : 1;
}
""",
        encoding="utf-8",
    )

    try:
        # 1. C11 compile check
        r_c = subprocess.run(
            [str(gcc_path), "-O2", "-Wall", "-Wextra", "-std=c11", f"-I{main_dir}", str(test_src), "-o", str(test_exe)],
            capture_output=True,
            text=True,
        )
        if r_c.returncode != 0:
            raise RuntimeError(f"GCC C11 check failed:\n{r_c.stderr}")

        r_run = subprocess.run([str(test_exe)], capture_output=True, text=True)
        if r_run.returncode != 0:
            raise RuntimeError(f"Header verification binary failed:\n{r_run.stdout}\n{r_run.stderr}")

        # 2. C++17 compile check (since main.cpp in Phase E is C++)
        r_cpp = subprocess.run(
            [str(gpp_path), "-O2", "-Wall", "-Wextra", "-std=c++17", "-x", "c++", f"-I{main_dir}", str(test_src), "-o", str(test_exe)],
            capture_output=True,
            text=True,
        )
        if r_cpp.returncode != 0:
            raise RuntimeError(f"G++ C++17 check failed:\n{r_cpp.stderr}")

        return {
            "c11_compile_ok": True,
            "c11_warnings": r_c.stderr.strip(),
            "cpp17_compile_ok": True,
            "cpp17_warnings": r_cpp.stderr.strip(),
            "runtime_output": r_run.stdout.strip(),
        }
    finally:
        test_src.unlink(missing_ok=True)
        test_exe.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Phase D2: Create targeted 50-beat test vectors & thresholds.h")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    MAIN_DIR.mkdir(parents=True, exist_ok=True)

    # Load TEST data
    d_test = np.load(str(TEST_NPZ))
    X = d_test["X"].astype(np.float32)
    if X.ndim == 3:
        X = X[:, 0, :]
    y5 = d_test["y"].astype(np.int64)
    assert X.shape == (49691, 260), f"Unexpected TEST X shape: {X.shape}"

    # Load C2 Seed 42 logits
    d_lg = np.load(str(LOGITS_NPZ))
    logits_map = {m: d_lg[m].astype(np.float32) for m in ALL_MODES}

    # Load Seed 42 VAL Youden's J thresholds from host_eval.json
    he = json.loads(HOST_EVAL_JSON.read_text(encoding="utf-8"))["models"]
    th_prob = {m: float(he[m]["recalibrated_val_thresholds"]["3_val_best_youden_j"]) for m in ALL_MODES}
    th_logit_diff = {m: prob_to_logit_diff(th_prob[m]) for m in ALL_MODES}

    # Extract output quantization parameters and verify exact integer logits
    qparams = {}
    int_logits_map = {}
    for cfg in SHORTLIST:
        qp = get_output_quant_params(cfg)
        qparams[cfg] = qp
        scaled = logits_map[cfg].astype(np.float64) / qp["scale"]
        rounded = np.round(scaled)
        max_frac_err = float(np.max(np.abs(scaled - rounded)))
        assert max_frac_err < 1e-4, f"{cfg}: non-integer scaled logits (max err={max_frac_err})"
        int_arr = np.clip(rounded, qp["quant_min"], qp["quant_max"]).astype(np.int32)
        int_logits_map[cfg] = int_arr
        qp["max_int_reconstruction_err"] = max_frac_err

    # Select 50 targeted beats
    selected, categories, preds, peaks, p95_v = select_50_beats(X, y5, logits_map, th_logit_diff, seed=args.seed)

    # Verify Gate D criterion (3): every beat in first 20 has >= 1 config disagreeing with FP32
    disagree_details = []
    per_cfg_disagree_in_20 = {cfg: 0 for cfg in SHORTLIST}
    for i, idx in enumerate(selected[:20]):
        dis = [cfg for cfg in SHORTLIST if preds[cfg][idx] != preds["fp32"][idx]]
        assert len(dis) >= 1, f"Beat {i} (TEST idx={idx}) has no disagreeing config!"
        for c in dis:
            per_cfg_disagree_in_20[c] += 1
        disagree_details.append(dis)

    # Export headers
    th_h_path = MAIN_DIR / "thresholds.h"
    tv_h_path = MAIN_DIR / "test_vectors.h"
    write_thresholds_h(th_h_path, th_prob, th_logit_diff)
    write_test_vectors_h(tv_h_path, selected, X, y5, logits_map, int_logits_map, qparams)

    # Export metadata JSON
    meta_records = []
    for bi, idx in enumerate(selected):
        dis_cfgs = [cfg for cfg in SHORTLIST if preds[cfg][idx] != preds["fp32"][idx]]
        meta_records.append(
            {
                "vector_id": bi,
                "test_index": int(idx),
                "y5": int(y5[idx]),
                "y5_name": CLASS_NAMES[int(y5[idx])],
                "y2": int(y5[idx] != 0),
                "selection_category": categories[idx],
                "peak_abs_amplitude": round(float(peaks[idx]), 6),
                "fp32_pred": int(preds["fp32"][idx]),
                "disagreeing_configs": dis_cfgs,
                "fp32_logits": [round(float(v), 6) for v in logits_map["fp32"][idx]],
                "ppq_int_logits": {
                    cfg: [int(int_logits_map[cfg][idx, 0]), int(int_logits_map[cfg][idx, 1])]
                    for cfg in SHORTLIST
                },
            }
        )

    meta_obj = {
        "num_vectors": len(selected),
        "seed": args.seed,
        "v_highpeak_p95_threshold": round(p95_v, 6),
        "category_counts": {
            "disagreement": sum(1 for c in categories.values() if c == "disagreement"),
            "class_F": sum(1 for c in categories.values() if c == "class_F"),
            "class_V_highpeak": sum(1 for c in categories.values() if c == "class_V_highpeak"),
            "class_N_random": sum(1 for c in categories.values() if c == "class_N_random"),
        },
        "disagreement_20_coverage_by_config": per_cfg_disagree_in_20,
        "thresholds_val_youden_j": {
            m: {"prob": th_prob[m], "logit_diff": round(th_logit_diff[m], 9)} for m in ALL_MODES
        },
        "output_quant_params": qparams,
        "vectors": meta_records,
    }
    meta_path = QUANT_DIR / "test_vectors_meta.json"
    meta_path.write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")

    # Compile & runtime verify headers with GCC and G++
    compile_res = verify_headers_compile(MAIN_DIR, GCC_PATH, GPP_PATH)

    # Check Gate D1 result if present
    d1_path = RESULTS_DIR / "gate_d1_fp32_c.json"
    d1_pass = False
    d1_max_diff = None
    if d1_path.exists():
        d1_data = json.loads(d1_path.read_text(encoding="utf-8"))
        d1_pass = bool(d1_data.get("pc_verification", {}).get("gate_d1_pass", False))
        d1_max_diff = d1_data.get("pc_verification", {}).get("max_logit_diff")

    gate_d = {
        "gate_d1_fp32_c_pass": d1_pass,
        "gate_d1_max_logit_diff": d1_max_diff,
        "gate_d2_headers_compile": compile_res,
        "gate_d2_disagreement_20_all_valid": True,
        "disagreement_20_coverage_by_config": per_cfg_disagree_in_20,
        "test_vectors_h_size_kb": round(tv_h_path.stat().st_size / 1024.0, 2),
        "thresholds_h_size_kb": round(th_h_path.stat().st_size / 1024.0, 2),
        "gate_d_overall_pass": d1_pass and compile_res["c11_compile_ok"] and compile_res["cpp17_compile_ok"],
    }
    gate_d_path = RESULTS_DIR / "gate_d_summary.json"
    gate_d_path.write_text(json.dumps(gate_d, indent=2), encoding="utf-8")

    print("Gate D Summary:")
    print(json.dumps(gate_d, indent=2))
    if not gate_d["gate_d_overall_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
