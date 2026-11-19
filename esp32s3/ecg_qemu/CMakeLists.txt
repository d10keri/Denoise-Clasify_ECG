cmake_minimum_required(VERSION 3.16)

option(ECG_QEMU_WORKAROUNDS "Enable QEMU 9.2.2 PIE workarounds (set OFF for real ESP32-S3 silicon)" ON)
if(ECG_QEMU_WORKAROUNDS)
    add_compile_definitions(DL_QEMU_AVOID_ST_QACC=1 DL_QEMU_GEMM_SCALAR_FALLBACK=1)
else()
    add_compile_definitions(DL_QEMU_AVOID_ST_QACC=0 DL_QEMU_GEMM_SCALAR_FALLBACK=0)
endif()

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(ecg_qemu)
