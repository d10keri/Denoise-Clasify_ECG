#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_private/startup_internal.h"
#include "dl_model_base.hpp"

// Runs in do_system_init_fn() BEFORE do_global_ctors() on ESP32-S3!
ESP_SYSTEM_INIT_FN(enable_coprocessors_before_ctors, CORE, BIT(0), 100)
{
    uint32_t cpen = 0x09; // Bit 0 = FPU (CP0), Bit 3 = PIE (CP3)
    asm volatile("wsr %0, cpenable; rsync" : : "r"(cpen));
    return ESP_OK;
}

extern const uint8_t b_int16_start[] asm("_binary_b_int16_espdl_start");
extern const uint8_t b_int16_end[]   asm("_binary_b_int16_espdl_end");
extern const uint8_t c_w8a16_start[] asm("_binary_c_w8a16_espdl_start");
extern const uint8_t c_w8a16_end[]   asm("_binary_c_w8a16_espdl_end");
extern const uint8_t c3_start[]      asm("_binary_c3_espdl_start");
extern const uint8_t c3_end[]        asm("_binary_c3_espdl_end");
extern const uint8_t c1_start[]      asm("_binary_c1_espdl_start");
extern const uint8_t c1_end[]        asm("_binary_c1_espdl_end");

struct SmokeModelEntry {
    const char *name;
    const uint8_t *start;
    const uint8_t *end;
};

extern "C" void app_main(void)
{
    uint32_t cpen = 0x09;
    asm volatile("wsr %0, cpenable; rsync" : : "r"(cpen));

    printf("\n=== F0 SMOKE TEST: ESP-DL ON QEMU ESP32-S3 (4 SHORTLIST MODELS) ===\n");

    const SmokeModelEntry entries[] = {
        {"B (INT16 w16a16)",     b_int16_start, b_int16_end},
        {"C_W8A16 (W8A16)",      c_w8a16_start, c_w8a16_end},
        {"C3 (Mixed INT16+INT8)",c3_start,      c3_end},
        {"C1 (INT8 Peak-Calib)", c1_start,      c1_end},
    };

    int pass_count = 0;
    for (size_t i = 0; i < sizeof(entries)/sizeof(entries[0]); i++) {
        size_t model_size = entries[i].end - entries[i].start;
        uint8_t *aligned_model = (uint8_t *)heap_caps_aligned_alloc(16, model_size, MALLOC_CAP_8BIT);
        memcpy(aligned_model, entries[i].start, model_size);

        printf("\n------------------------------------------------------------\n");
        printf("[F0] Testing %s (size=%u bytes, aligned=%p)...\n",
               entries[i].name, (unsigned int)model_size, (void *)aligned_model);

        dl::Model *model = new dl::Model((const char *)aligned_model, fbs::MODEL_LOCATION_IN_FLASH_RODATA);
        int64_t t0 = esp_timer_get_time();
        esp_err_t err = model->test();
        int64_t dt = esp_timer_get_time() - t0;

        dl::TensorBase *out = model->get_output();
        if (out && out->data) {
            int exp = (int)out->exponent;
            float scale = (exp >= 0) ? (float)(1 << exp) : (1.0f / (float)(1 << (-exp)));
            if (out->get_dtype() == dl::DATA_TYPE_INT8) {
                int8_t *p = (int8_t *)out->data;
                printf("[F0] %s output int8=[%d, %d] (exp=%d) -> dequant=[%.5f, %.5f]\n",
                       entries[i].name, (int)p[0], (int)p[1], exp, p[0] * scale, p[1] * scale);
            } else if (out->get_dtype() == dl::DATA_TYPE_INT16) {
                int16_t *p = (int16_t *)out->data;
                printf("[F0] %s output int16=[%d, %d] (exp=%d) -> dequant=[%.5f, %.5f]\n",
                       entries[i].name, (int)p[0], (int)p[1], exp, p[0] * scale, p[1] * scale);
            }
        }
        printf("[F0] %s RESULT: %s (err=%d, dt_qemu=%lld us)\n",
               entries[i].name, (err == ESP_OK) ? "PASS" : "FAIL", (int)err, (long long)dt);
        if (err == ESP_OK) {
            pass_count++;
        }
        delete model;
        heap_caps_free(aligned_model);
    }

    printf("\n============================================================\n");
    printf("F0_SMOKE_SUMMARY: %d/4 SHORTLIST MODELS PASSED ON QEMU ESP32-S3\n", pass_count);
    printf("=== F0 SMOKE TEST COMPLETE ===\n");
}
