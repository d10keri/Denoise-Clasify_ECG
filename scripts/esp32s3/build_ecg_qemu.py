"""
Phase E: Configure, patch toggleable QEMU guards, build esp32s3/ecg_qemu,
and collect idf.py size metrics.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJ_DIR = PROJECT_ROOT / "esp32s3" / "ecg_qemu"
RESULTS_DIR = PROJECT_ROOT / "esp32s3" / "results"

ESPRESSIF_ROOT = Path(r"C:\Users\vietc\ESP_SDK\Espressif")
IDF_PATH = ESPRESSIF_ROOT / "frameworks" / "esp-idf-v5.5.2"
IDF_PYTHON = ESPRESSIF_ROOT / "python_env" / "idf5.5_py3.11_env" / "Scripts" / "python.exe"
IDF_PY = IDF_PATH / "tools" / "idf.py"
IDF_TOOLS_PY = IDF_PATH / "tools" / "idf_tools.py"


def get_idf_env() -> dict:
    env = os.environ.copy()
    env["IDF_PATH"] = str(IDF_PATH)
    env["IDF_TOOLS_PATH"] = str(ESPRESSIF_ROOT)
    env["IDF_PYTHON_ENV_PATH"] = str(IDF_PYTHON.parents[1])

    res = subprocess.run(
        [str(IDF_PYTHON), str(IDF_TOOLS_PY), "export", "--format", "key-value"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    for line in res.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            if k == "PATH":
                v = v.replace("%PATH%", env.get("PATH", "")).replace("$PATH", env.get("PATH", ""))
            env[k] = v
    return env


def run_idf(env: dict, args: list[str], label: str) -> tuple[int, str]:
    cmd = [str(IDF_PYTHON), str(IDF_PY), "-C", str(PROJ_DIR)] + args
    print(f"[Phase E] {label}: {' '.join(args)}")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = (res.stdout or "") + "\n" + (res.stderr or "")
    return res.returncode, out


def apply_qemu_guard_patches(esp_dl_dir: Path):
    """Idempotently add #if-guarded QEMU 9.2.2 workarounds (controlled by CMake ECG_QEMU_WORKAROUNDS)."""
    # 1. dl_base_avg_pool2d.cpp
    avg_file = esp_dl_dir / "dl" / "base" / "dl_base_avg_pool2d.cpp"
    avg_txt = avg_file.read_text(encoding="utf-8")
    old_avg = "#elif CONFIG_PIE_V1_BOOST\n"
    new_avg = "#elif CONFIG_PIE_V1_BOOST && !(defined(DL_QEMU_AVOID_ST_QACC) && DL_QEMU_AVOID_ST_QACC)\n"
    if "DL_QEMU_AVOID_ST_QACC" not in avg_txt:
        assert old_avg in avg_txt, f"Could not find target line in {avg_file}"
        avg_file.write_text(avg_txt.replace(old_avg, new_avg, 1), encoding="utf-8")
        print(f"[Phase E] Patched {avg_file.name} with DL_QEMU_AVOID_ST_QACC guard")
    else:
        print(f"[Phase E] {avg_file.name} already guarded")

    # 2. dl_module_gemm.hpp
    gemm_file = esp_dl_dir / "dl" / "module" / "include" / "dl_module_gemm.hpp"
    gemm_txt = gemm_file.read_text(encoding="utf-8")
    old_gemm = '        if (task_size == 1) { // single task\n            forward_args((void *)&m_args.get_args(0));'
    new_gemm = """        if (task_size == 1) { // single task
#if defined(DL_QEMU_GEMM_SCALAR_FALLBACK) && DL_QEMU_GEMM_SCALAR_FALLBACK
            const auto &a0 = m_args.get_args(0);
            if (a0.output_height == 1 && a0.output_width == 1 && a0.output_channel < 8 && a0.mac_shift >= 0) {
                const int64_t *b = (const int64_t *)a0.bias_element;
                const int shift = a0.mac_shift;
                const int64_t round_add = (shift > 0) ? (1LL << (shift - 1)) : 0;
                for (int n = 0; n < a0.output_channel; n++) {
                    int64_t acc = b ? b[n] : 0;
                    if (quant_type == QUANT_TYPE_SYMM_8BIT) {
                        const int8_t *x = (const int8_t *)a0.input_element;
                        const int8_t *w = (const int8_t *)a0.filter_element + n * a0.input_channel;
                        for (int c = 0; c < a0.input_channel; c++) {
                            acc += (int64_t)x[c] * (int64_t)w[c];
                        }
                    } else if (quant_type == QUANT_TYPE_SYMM_W8A16) {
                        const int16_t *x = (const int16_t *)a0.input_element;
                        const int8_t *w = (const int8_t *)a0.filter_element + n * a0.input_channel;
                        for (int c = 0; c < a0.input_channel; c++) {
                            acc += (int64_t)x[c] * (int64_t)w[c];
                        }
                    } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
                        const int16_t *x = (const int16_t *)a0.input_element;
                        const int16_t *w = (const int16_t *)a0.filter_element + n * a0.input_channel;
                        for (int c = 0; c < a0.input_channel; c++) {
                            acc += (int64_t)x[c] * (int64_t)w[c];
                        }
                    }
                    int64_t v = (shift > 0) ? ((acc + round_add) >> shift) : acc;
                    if (this->activation == ReLU && v < 0) v = 0;
                    if (quant_type == QUANT_TYPE_SYMM_8BIT) {
                        if (v > 127) v = 127;
                        if (v < -128) v = -128;
                        ((int8_t *)a0.output_element)[n] = (int8_t)v;
                    } else {
                        if (v > 32767) v = 32767;
                        if (v < -32768) v = -32768;
                        ((int16_t *)a0.output_element)[n] = (int16_t)v;
                    }
                }
            } else {
                forward_args((void *)&m_args.get_args(0));
            }
#else
            forward_args((void *)&m_args.get_args(0));
#endif"""
    if "DL_QEMU_GEMM_SCALAR_FALLBACK" not in gemm_txt:
        assert old_gemm in gemm_txt, f"Could not find target block in {gemm_file}"
        gemm_file.write_text(gemm_txt.replace(old_gemm, new_gemm, 1), encoding="utf-8")
        print(f"[Phase E] Patched {gemm_file.name} with DL_QEMU_GEMM_SCALAR_FALLBACK guard")
    else:
        print(f"[Phase E] {gemm_file.name} already guarded")


def main():
    parser = argparse.ArgumentParser(description="Build esp32s3/ecg_qemu firmware and measure size")
    parser.add_argument("--real-board", action="store_true", help="Disable QEMU workarounds for real ESP32-S3 silicon")
    args = parser.parse_args()

    env = get_idf_env()
    esp_dl_dir = PROJ_DIR / "managed_components" / "espressif__esp-dl"

    # Step 1: set-target (downloads esp-dl if not present)
    if not (PROJ_DIR / "sdkconfig").exists() or not esp_dl_dir.exists():
        rc, out = run_idf(env, ["set-target", "esp32s3"], "set-target esp32s3")
        if rc != 0:
            print(out[-3000:])
            sys.exit(rc)

    # Step 2: apply toggleable QEMU guards in esp-dl
    apply_qemu_guard_patches(esp_dl_dir)

    # Step 3: build
    qemu_flag = "OFF" if args.real_board else "ON"
    rc, out = run_idf(env, [f"-DECG_QEMU_WORKAROUNDS={qemu_flag}", "build"], f"build (ECG_QEMU_WORKAROUNDS={qemu_flag})")
    if rc != 0:
        print(out[-4000:])
        sys.exit(rc)
    print("\n".join(out.strip().splitlines()[-15:]))

    # Step 4: idf.py size
    rc_sz, out_sz = run_idf(env, ["size"], "idf.py size")
    if rc_sz != 0:
        print(out_sz[-2000:])
        sys.exit(rc_sz)

    rc_sc, out_sc = run_idf(env, ["size-components"], "idf.py size-components")
    size_report = f"=== ECG_QEMU_WORKAROUNDS={qemu_flag} ===\n\n--- idf.py size ---\n{out_sz}\n--- idf.py size-components ---\n{out_sc}\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    size_path = RESULTS_DIR / "ecg_qemu_size.txt"
    size_path.write_text(size_report, encoding="utf-8")
    print("\n--- idf.py size ---")
    print(out_sz)
    print(f"Saved full size report to {size_path}")


if __name__ == "__main__":
    main()
