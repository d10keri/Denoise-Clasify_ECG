#include <stdio.h>
#include <stdlib.h>
#include "model_fp32.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <in.bin> <out.bin> <num_samples>\n", argv[0]);
        return 1;
    }
    const char *in_path = argv[1];
    const char *out_path = argv[2];
    const int n = atoi(argv[3]);

    FILE *fin = fopen(in_path, "rb");
    if (!fin) { perror("fopen in"); return 2; }
    FILE *fout = fopen(out_path, "wb");
    if (!fout) { perror("fopen out"); fclose(fin); return 3; }

    float x[MODEL_FP32_INPUT_LEN];
    float y[MODEL_FP32_OUTPUT_LEN];
    for (int i = 0; i < n; ++i) {
        if (fread(x, sizeof(float), MODEL_FP32_INPUT_LEN, fin) != MODEL_FP32_INPUT_LEN) {
            fprintf(stderr, "Short read at sample %d\n", i);
            fclose(fin);
            fclose(fout);
            return 4;
        }
        model_fp32_infer(x, y);
        fwrite(y, sizeof(float), MODEL_FP32_OUTPUT_LEN, fout);
    }
    fclose(fin);
    fclose(fout);
    return 0;
}
