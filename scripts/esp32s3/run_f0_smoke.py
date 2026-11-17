"""
Phase F0: Build & Run ESP-DL Smoke Test on QEMU ESP32-S3 (qemu-xtensa 9.2.2)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
F0_DIR = PROJECT_ROOT / "esp32s3" / "f0_smoke"
RESULTS_DIR = PROJECT_ROOT / "esp32s3" / "results"

ESPRESSIF_ROOT = Path(r"C:\Users\vietc\ESP_SDK\Espressif")
IDF_PATH = ESPRESSIF_ROOT / "frameworks" / "esp-idf-v5.5.2"
IDF_PYTHON = ESPRESSIF_ROOT / "python_env" / "idf5.5_py3.11_env" / "Scripts" / "python.exe"
IDF_PY = IDF_PATH / "tools" / "idf.py"
IDF_TOOLS_PY = IDF_PATH / "tools" / "idf_tools.py"
QEMU_EXE = (
    ESPRESSIF_ROOT
    / "tools"
    / "qemu-xtensa"
    / "esp_develop_9.2.2_20250817"
    / "qemu"
    / "bin"
    / "qemu-system-xtensa.exe"
)


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


def run_build(env: dict) -> tuple[int, str]:
    print("[F0] Running idf.py build in esp32s3/f0_smoke ...")
    res = subprocess.run(
        [str(IDF_PYTHON), str(IDF_PY), "-C", str(F0_DIR), "build"],
        capture_output=True,
        text=True,
        env=env,
    )
    out = (res.stdout or "") + "\n" + (res.stderr or "")
    return res.returncode, out


def merge_qemu_flash(env: dict) -> tuple[int, str, Path]:
    build_dir = F0_DIR / "build"
    flash_bin = build_dir / "qemu_flash.bin"
    flasher_json = json.loads((build_dir / "flasher_args.json").read_text(encoding="utf-8"))

    cmd = [
        str(IDF_PYTHON),
        "-m",
        "esptool",
        "--chip",
        "esp32s3",
        "merge_bin",
        "--fill-flash-size",
        "4MB",
        "-o",
        str(flash_bin),
    ]
    cmd.extend(flasher_json.get("write_flash_args", []))
    for offset, rel_file in flasher_json.get("flash_files", {}).items():
        cmd.extend([offset, rel_file])

    print("[F0] Merging flash image for QEMU:", " ".join(cmd))
    res = subprocess.run(
        cmd,
        cwd=str(build_dir),
        capture_output=True,
        text=True,
        env=env,
    )
    out = (res.stdout or "") + "\n" + (res.stderr or "")
    return res.returncode, out, flash_bin


def run_qemu_with_timeout(flash_bin: Path, timeout_sec: float = 10.0) -> str:
    print(f"[F0] Launching QEMU esp32s3 ({QEMU_EXE.name}) with {timeout_sec}s timeout ...")
    cmd = [
        str(QEMU_EXE),
        "-nographic",
        "-machine",
        "esp32s3",
        "-drive",
        f"file={flash_bin},if=mtd,format=raw",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    try:
        out, _ = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = get_idf_env()

    app_bin = F0_DIR / "build" / "f0_smoke.bin"
    if not app_bin.exists() or "--rebuild" in sys.argv:
        rc_build, build_log = run_build(env)
        (RESULTS_DIR / "f0_build.log").write_text(build_log, encoding="utf-8")
        if rc_build != 0:
            print(f"[ERROR] Build failed with code {rc_build}. Last 40 lines of build log:")
            print("\n".join(build_log.splitlines()[-40:]))
            sys.exit(rc_build)
    else:
        print(f"[F0] Using already built binary: {app_bin}")

    rc_merge, merge_log, flash_bin = merge_qemu_flash(env)
    if rc_merge != 0:
        print(f"[ERROR] Merge flash failed with code {rc_merge}:\n{merge_log}")
        sys.exit(rc_merge)
    print("[OK] Merged flash image:", flash_bin)

    qemu_log = run_qemu_with_timeout(flash_bin, timeout_sec=10.0)
    (RESULTS_DIR / "f0_qemu.log").write_text(qemu_log, encoding="utf-8")
    print("\n=== QEMU OUTPUT PREVIEW ===")
    print(qemu_log[-3000:])


if __name__ == "__main__":
    main()
