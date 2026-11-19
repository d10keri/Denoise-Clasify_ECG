#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_private/startup_internal.h"
#include "esp_timer.h"

#include "dl_define.hpp"
#include "dl_model_base.hpp"
#include "dl_tool.hpp"

#include "model_fp32.h"
#include "test_vectors.h"
#include "thresholds.h"

/* Enable FPU (CP0) and PIE (CP3) before global C++ constructors (esp-dl AttentionFloatExpLut) */
ESP_SYSTEM_INIT_FN(enable_coprocessors_before_ctors, CORE, BIT(0), 100)
{
    uint32_t cpen = 0x09;
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

struct QuantModeSpec {
    ecg_mode_t mode;
    const char *name;
    const uint8_t *bin_start;
    const uint8_t *bin_end;
    const float (*ref_logits)[2];
    const int16_t (*ref_int16)[2];
    const int8_t (*ref_int8)[2];
    int expected_exp_out;
};

static void quantize_input_beat(const float *src, dl::TensorBase *in_tensor)
{
    const int exp_in = (int)in_tensor->exponent;
    const float rescale = DL_RESCALE(exp_in);

    if (in_tensor->get_dtype() == dl::DATA_TYPE_INT16) {
        int16_t *dst = (int16_t *)in_tensor->data;
        for (int j = 0; j < TEST_VECTOR_LEN; ++j) {
            int q = dl::tool::round(src[j] * rescale);
            dst[j] = (int16_t)DL_CLIP(q, DL_QUANT16_MIN, DL_QUANT16_MAX);
        }
    } else if (in_tensor->get_dtype() == dl::DATA_TYPE_INT8) {
        int8_t *dst = (int8_t *)in_tensor->data;
        for (int j = 0; j < TEST_VECTOR_LEN; ++j) {
            int q = dl::tool::round(src[j] * rescale);
            dst[j] = (int8_t)DL_CLIP(q, DL_QUANT8_MIN, DL_QUANT8_MAX);
        }
    } else {
        printf("[ERR] Unsupported input dtype=%d\n", (int)in_tensor->get_dtype());
    }
}

static void run_fp32_mode(int fp32_preds[NUM_TEST_VECTORS])
{
    float logits[2] = {0.0f, 0.0f};
    float max_logit_diff = 0.0f;
    int64_t total_us = 0;
    size_t heap_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

    for (int i = 0; i < NUM_TEST_VECTORS; ++i) {
        int64_t t0 = esp_timer_get_time();
        model_fp32_infer(TEST_X[i], logits);
        total_us += (esp_timer_get_time() - t0);

        float d0 = fabsf(logits[0] - REF_LOGITS_FP32[i][0]);
        float d1 = fabsf(logits[1] - REF_LOGITS_FP32[i][1]);
        if (d0 > max_logit_diff) max_logit_diff = d0;
        if (d1 > max_logit_diff) max_logit_diff = d1;

        fp32_preds[i] = ecg_decide_binary(logits[0], logits[1], TH_LOGIT_DIFF_FP32);
    }

    double avg_dt_us = (double)total_us / (double)NUM_TEST_VECTORS;
    printf("MODE=fp32 max_logit_diff=%.7f avg_dt_us=%.1f heap_internal=%u\n",
           max_logit_diff, avg_dt_us, (unsigned int)heap_free);
}

static void run_quant_mode(const QuantModeSpec &spec, const int fp32_preds[NUM_TEST_VECTORS])
{
    const size_t model_size = (size_t)(spec.bin_end - spec.bin_start);
    const size_t heap_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

    uint8_t *aligned_buf = (uint8_t *)heap_caps_aligned_alloc(16, model_size, MALLOC_CAP_8BIT);
    if (!aligned_buf) {
        printf("MODE=%s ERROR=alloc_failed size=%u\n", spec.name, (unsigned int)model_size);
        return;
    }
    memcpy(aligned_buf, spec.bin_start, model_size);

    dl::Model *model = new dl::Model((const char *)aligned_buf, fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    if (!model || !model->is_built()) {
        printf("MODE=%s ERROR=model_build_failed\n", spec.name);
        delete model;
        heap_caps_free(aligned_buf);
        return;
    }

    const size_t heap_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    const size_t ram_used = (heap_before > heap_internal) ? (heap_before - heap_internal) : 0;

    dl::TensorBase *in_t = model->get_input();
    dl::TensorBase *out_t = model->get_output();
    const int exp_in = (int)in_t->exponent;
    const int exp_out = (int)out_t->exponent;
    const float out_scale = DL_SCALE(exp_out);
    const float th_diff = ECG_TH_LOGIT_DIFF[spec.mode];

    int match_ppq = 0;
    int exact_0lsb = 0;
    int max_lsb_diff = 0;
    int agree_fp32 = 0;
    int agree_ppq_label = 0;
    int64_t total_us = 0;

    for (int i = 0; i < NUM_TEST_VECTORS; ++i) {
        quantize_input_beat(TEST_X[i], in_t);

        int64_t t0 = esp_timer_get_time();
        model->run(dl::RUNTIME_MODE_SINGLE_CORE);
        total_us += (esp_timer_get_time() - t0);

        int32_t y_int[2] = {0, 0};
        int32_t ref_int[2] = {0, 0};
        if (out_t->get_dtype() == dl::DATA_TYPE_INT16) {
            const int16_t *p = (const int16_t *)out_t->data;
            y_int[0] = (int32_t)p[0];
            y_int[1] = (int32_t)p[1];
            ref_int[0] = (int32_t)spec.ref_int16[i][0];
            ref_int[1] = (int32_t)spec.ref_int16[i][1];
        } else {
            const int8_t *p = (const int8_t *)out_t->data;
            y_int[0] = (int32_t)p[0];
            y_int[1] = (int32_t)p[1];
            ref_int[0] = (int32_t)spec.ref_int8[i][0];
            ref_int[1] = (int32_t)spec.ref_int8[i][1];
        }

        int d0 = abs((int)(y_int[0] - ref_int[0]));
        int d1 = abs((int)(y_int[1] - ref_int[1]));
        int lsb_diff = (d0 > d1) ? d0 : d1;
        if (lsb_diff > max_lsb_diff) {
            max_lsb_diff = lsb_diff;
        }
        if (lsb_diff <= 1) {
            match_ppq++;
        }
        if (lsb_diff == 0) {
            exact_0lsb++;
        }

        float y_f[2] = {y_int[0] * out_scale, y_int[1] * out_scale};
        int pred = ecg_decide_binary(y_f[0], y_f[1], th_diff);
        if (pred == fp32_preds[i]) {
            agree_fp32++;
        }
        int ref_pred = ecg_decide_binary(spec.ref_logits[i][0], spec.ref_logits[i][1], th_diff);
        if (pred == ref_pred) {
            agree_ppq_label++;
        }
    }

    double avg_dt_us = (double)total_us / (double)NUM_TEST_VECTORS;
    printf("MODE=%s match_ppq=%d/%d max_lsb_diff=%d agree_fp32=%d/%d avg_dt_us=%.1f heap_internal=%u\n",
           spec.name, match_ppq, NUM_TEST_VECTORS, max_lsb_diff,
           agree_fp32, NUM_TEST_VECTORS, avg_dt_us, (unsigned int)heap_internal);
    printf("  [INFO] %s exact_0lsb=%d/%d agree_ppq_label=%d/%d exp_in=%d exp_out=%d(expect=%d) espdl_bytes=%u ram_used=%u\n",
           spec.name, exact_0lsb, NUM_TEST_VECTORS, agree_ppq_label, NUM_TEST_VECTORS,
           exp_in, exp_out, spec.expected_exp_out, (unsigned int)model_size, (unsigned int)ram_used);

    delete model;
    heap_caps_free(aligned_buf);
}

extern "C" void app_main(void)
{
    uint32_t cpen = 0x09;
    asm volatile("wsr %0, cpenable; rsync" : : "r"(cpen));

#ifndef DL_QEMU_AVOID_ST_QACC
#define DL_QEMU_AVOID_ST_QACC 0
#endif
#ifndef DL_QEMU_GEMM_SCALAR_FALLBACK
#define DL_QEMU_GEMM_SCALAR_FALLBACK 0
#endif

    printf("\n====================================================================\n");
    printf("ECG M2-B-KD MULTI-MODE VERIFICATION FIRMWARE (ESP32-S3, %d BEATS)\n", NUM_TEST_VECTORS);
    printf("Build flags: DL_QEMU_AVOID_ST_QACC=%d, DL_QEMU_GEMM_SCALAR_FALLBACK=%d\n",
           DL_QEMU_AVOID_ST_QACC, DL_QEMU_GEMM_SCALAR_FALLBACK);
    printf("NOTE: avg_dt_us is valid ONLY on real silicon; DO NOT use QEMU timing.\n");
    printf("====================================================================\n");

    int fp32_preds[NUM_TEST_VECTORS];
    run_fp32_mode(fp32_preds);

    const QuantModeSpec modes[] = {
        {ECG_MODE_B_INT16, "b_int16", b_int16_start, b_int16_end, REF_LOGITS_B_INT16, REF_INT_LOGITS_B_INT16, nullptr, EXP_OUT_B_INT16},
        {ECG_MODE_C_W8A16, "c_w8a16", c_w8a16_start, c_w8a16_end, REF_LOGITS_C_W8A16, REF_INT_LOGITS_C_W8A16, nullptr, EXP_OUT_C_W8A16},
        {ECG_MODE_C3,      "c3",      c3_start,      c3_end,      REF_LOGITS_C3,      REF_INT_LOGITS_C3,      nullptr, EXP_OUT_C3},
        {ECG_MODE_C1,      "c1",      c1_start,      c1_end,      REF_LOGITS_C1,      nullptr, REF_INT_LOGITS_C1,      EXP_OUT_C1},
    };

    for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); ++m) {
        run_quant_mode(modes[m], fp32_preds);
    }

    printf("====================================================================\n");
    printf("ECG_QEMU_ALL_MODES_DONE\n");
    printf("====================================================================\n");
}
