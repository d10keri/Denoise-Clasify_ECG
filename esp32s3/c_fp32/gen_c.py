"""
Phase D1: Generate pure C FP32 inference code (model_fp32.h and model_fp32.c)
from BN-folded ONNX model export/cnn1d_binary_dsB_kd_fp32.onnx,
then verify bit-level / numerical parity against ONNXRuntime on 500 beats
using MSYS2 GCC.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX = PROJECT_ROOT / "export" / "cnn1d_binary_dsB_kd_fp32.onnx"
DEFAULT_TEST_X = PROJECT_ROOT / "export" / "host_val_test500_x.npy"
OUT_DIR = Path(__file__).resolve().parent
GCC_PATH = Path(r"C:\msys64\ucrt64\bin\gcc.exe")


def sanitize_c_ident(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    if s and s[0].isdigit():
        s = "w_" + s
    return s


def get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.INTS:
                return list(a.ints)
            if a.type == onnx.AttributeProto.INT:
                return int(a.i)
            if a.type == onnx.AttributeProto.FLOAT:
                return float(a.f)
            if a.type == onnx.AttributeProto.FLOATS:
                return list(a.floats)
            if a.type == onnx.AttributeProto.STRING:
                return a.s.decode("utf-8")
    return default


def format_float_array(arr: np.ndarray, c_name: str, per_line: int = 8) -> str:
    flat = np.ascontiguousarray(arr, dtype=np.float32).ravel()
    lines = [f"static const float {c_name}[{flat.size}] = {{"]
    for i in range(0, flat.size, per_line):
        chunk = flat[i : i + per_line]
        vals = ", ".join(f"{float(v):.9g}f" for v in chunk)
        lines.append(f"    {vals},")
    lines.append("};")
    return "\n".join(lines)


def generate_c_code(onnx_path: Path, out_dir: Path):
    model = onnx.load(str(onnx_path))
    graph = model.graph

    inits = {init.name: numpy_helper.to_array(init).astype(np.float32) for init in graph.initializer}

    # Verify input shape (1, 1, 260)
    in_dims = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    assert len(in_dims) == 3 and in_dims[0] == 1, f"Unexpected input shape: {in_dims}"
    c_cur, len_cur = int(in_dims[1]), int(in_dims[2])
    input_len = c_cur * len_cur

    out_dims = [d.dim_value for d in graph.output[0].type.tensor_type.shape.dim]
    assert len(out_dims) == 2 and out_dims[0] == 1, f"Unexpected output shape: {out_dims}"
    output_len = int(out_dims[1])

    # Track ping-pong buffers: None means external `input`, 0 means g_buf_a, 1 means g_buf_b
    cur_buf = None
    max_buf_elems = 0
    total_params = 0
    weight_defs = []
    used_inits = set()
    call_lines = []

    def buf_name(b):
        if b is None:
            return "input"
        return "g_buf_a" if b == 0 else "g_buf_b"

    def next_buf(b):
        return 0 if b != 0 else 1

    for idx, node in enumerate(graph.node):
        op = node.op_type
        if op == "Conv":
            w_name, b_name = node.input[1], node.input[2]
            w = inits[w_name]
            b = inits[b_name]
            w_c = "w_" + sanitize_c_ident(w_name)
            b_c = "w_" + sanitize_c_ident(b_name)
            if w_name not in used_inits:
                weight_defs.append(format_float_array(w, w_c))
                used_inits.add(w_name)
                total_params += w.size
            if b_name not in used_inits:
                weight_defs.append(format_float_array(b, b_c))
                used_inits.add(b_name)
                total_params += b.size

            group = get_attr(node, "group", 1)
            k_shape = get_attr(node, "kernel_shape", [w.shape[2]])
            pads = get_attr(node, "pads", [0, 0])
            strides = get_attr(node, "strides", [1])
            dilations = get_attr(node, "dilations", [1])
            assert dilations == [1], f"Unsupported dilation: {dilations}"

            k = int(k_shape[0])
            pad_l, pad_r = int(pads[0]), int(pads[1])
            stride = int(strides[0])
            c_out = int(w.shape[0])
            len_out = (len_cur + pad_l + pad_r - k) // stride + 1

            dst_buf = next_buf(cur_buf)
            max_buf_elems = max(max_buf_elems, c_out * len_out)

            if group == 1:
                assert w.shape == (c_out, c_cur, k), f"Unexpected Conv weight shape {w.shape}"
                call_lines.append(
                    f"    /* Node {idx}: Conv {node.name} ({c_cur}x{len_cur} -> {c_out}x{len_out}, k={k}, pad=[{pad_l},{pad_r}], s={stride}) */"
                )
                call_lines.append(
                    f"    conv1d({buf_name(cur_buf)}, {buf_name(dst_buf)}, {w_c}, {b_c}, "
                    f"{c_cur}, {len_cur}, {c_out}, {k}, {pad_l}, {pad_r}, {stride});"
                )
            elif group == c_cur and group == c_out:
                assert w.shape == (c_out, 1, k), f"Unexpected DWConv weight shape {w.shape}"
                call_lines.append(
                    f"    /* Node {idx}: DWConv {node.name} ({c_cur}x{len_cur} -> {c_out}x{len_out}, k={k}, pad=[{pad_l},{pad_r}], s={stride}) */"
                )
                call_lines.append(
                    f"    dwconv1d({buf_name(cur_buf)}, {buf_name(dst_buf)}, {w_c}, {b_c}, "
                    f"{c_cur}, {len_cur}, {k}, {pad_l}, {pad_r}, {stride});"
                )
            else:
                raise ValueError(f"Unsupported grouped conv: group={group}, c_in={c_cur}, c_out={c_out}")

            cur_buf = dst_buf
            c_cur, len_cur = c_out, len_out

        elif op == "Relu":
            assert cur_buf is not None
            call_lines.append(f"    /* Node {idx}: Relu {node.name} ({c_cur}x{len_cur}) */")
            call_lines.append(f"    relu({buf_name(cur_buf)}, {c_cur * len_cur});")

        elif op == "MaxPool":
            k_shape = get_attr(node, "kernel_shape")
            pads = get_attr(node, "pads", [0, 0])
            strides = get_attr(node, "strides", [1])
            ceil_mode = get_attr(node, "ceil_mode", 0)
            assert ceil_mode == 0, f"Unsupported ceil_mode={ceil_mode}"
            k = int(k_shape[0])
            pad_l, pad_r = int(pads[0]), int(pads[1])
            stride = int(strides[0])
            len_out = (len_cur + pad_l + pad_r - k) // stride + 1

            dst_buf = next_buf(cur_buf)
            max_buf_elems = max(max_buf_elems, c_cur * len_out)
            call_lines.append(
                f"    /* Node {idx}: MaxPool {node.name} ({c_cur}x{len_cur} -> {c_cur}x{len_out}, k={k}, s={stride}) */"
            )
            call_lines.append(
                f"    maxpool1d({buf_name(cur_buf)}, {buf_name(dst_buf)}, "
                f"{c_cur}, {len_cur}, {k}, {pad_l}, {pad_r}, {stride});"
            )
            cur_buf = dst_buf
            len_cur = len_out

        elif op == "AveragePool":
            k_shape = get_attr(node, "kernel_shape")
            pads = get_attr(node, "pads", [0, 0])
            strides = get_attr(node, "strides", [1])
            assert pads == [0, 0], f"Unsupported AveragePool pads={pads}"
            k = int(k_shape[0])
            stride = int(strides[0])
            len_out = (len_cur - k) // stride + 1

            dst_buf = next_buf(cur_buf)
            max_buf_elems = max(max_buf_elems, c_cur * len_out)
            call_lines.append(
                f"    /* Node {idx}: AveragePool {node.name} ({c_cur}x{len_cur} -> {c_cur}x{len_out}, k={k}, s={stride}) */"
            )
            call_lines.append(
                f"    avgpool1d({buf_name(cur_buf)}, {buf_name(dst_buf)}, "
                f"{c_cur}, {len_cur}, {k}, {stride});"
            )
            cur_buf = dst_buf
            len_cur = len_out

        elif op == "Flatten":
            call_lines.append(f"    /* Node {idx}: Flatten {node.name} ({c_cur * len_cur}) - no-op in C layout */")

        elif op == "Gemm":
            w_name, b_name = node.input[1], node.input[2]
            w = inits[w_name]
            b = inits[b_name]
            trans_b = get_attr(node, "transB", 0)
            alpha = get_attr(node, "alpha", 1.0)
            beta = get_attr(node, "beta", 1.0)
            assert abs(alpha - 1.0) < 1e-6 and abs(beta - 1.0) < 1e-6
            if trans_b == 0:
                w = w.T.copy()
            in_dim = c_cur * len_cur
            out_dim = int(w.shape[0])
            assert w.shape == (out_dim, in_dim), f"Unexpected Gemm weight shape {w.shape}"

            w_c = "w_" + sanitize_c_ident(w_name)
            b_c = "w_" + sanitize_c_ident(b_name)
            if w_name not in used_inits:
                weight_defs.append(format_float_array(w, w_c))
                used_inits.add(w_name)
                total_params += w.size
            if b_name not in used_inits:
                weight_defs.append(format_float_array(b, b_c))
                used_inits.add(b_name)
                total_params += b.size

            call_lines.append(f"    /* Node {idx}: Gemm {node.name} ({in_dim} -> {out_dim}) */")
            call_lines.append(
                f"    gemm({buf_name(cur_buf)}, output, {w_c}, {b_c}, {in_dim}, {out_dim});"
            )
            c_cur, len_cur = out_dim, 1

        else:
            raise ValueError(f"Unsupported ONNX op: {op}")

    assert c_cur == output_len, f"Final output len {c_cur} != {output_len}"

    # Write model_fp32.h
    header_content = f"""/* Auto-generated by esp32s3/c_fp32/gen_c.py from {onnx_path.name} */
/* Total parameters: {total_params} ({total_params * 4 / 1024.0:.2f} KB in flash) */
/* Max ping-pong buffer: {max_buf_elems} floats x 2 = {max_buf_elems * 8} bytes ({max_buf_elems * 8 / 1024.0:.2f} KB SRAM) */

#ifndef MODEL_FP32_H
#define MODEL_FP32_H

#define MODEL_FP32_INPUT_LEN  {input_len}
#define MODEL_FP32_OUTPUT_LEN {output_len}
#define MODEL_FP32_BUF_ELEMS  {max_buf_elems}
#define MODEL_FP32_NUM_PARAMS {total_params}

#ifdef __cplusplus
extern "C" {{
#endif

void model_fp32_infer(const float *input, float *output);

#ifdef __cplusplus
}}
#endif

#ifdef MODEL_FP32_DEFINE_WEIGHTS

{chr(10).join(weight_defs)}

#endif /* MODEL_FP32_DEFINE_WEIGHTS */

#endif /* MODEL_FP32_H */
"""

    # Write model_fp32.c
    c_content = f"""/* Auto-generated by esp32s3/c_fp32/gen_c.py from {onnx_path.name} */

#include <float.h>
#include <stddef.h>

#define MODEL_FP32_DEFINE_WEIGHTS
#include "model_fp32.h"

/* Two static ping-pong activation buffers in internal SRAM */
static float g_buf_a[MODEL_FP32_BUF_ELEMS];
static float g_buf_b[MODEL_FP32_BUF_ELEMS];

static void conv1d(
    const float *in,
    float *out,
    const float *weight,
    const float *bias,
    int c_in,
    int len_in,
    int c_out,
    int k,
    int pad_l,
    int pad_r,
    int stride)
{{
    const int len_out = (len_in + pad_l + pad_r - k) / stride + 1;
    for (int oc = 0; oc < c_out; ++oc) {{
        const float b = bias[oc];
        const float *w_oc = weight + (size_t)oc * (size_t)c_in * (size_t)k;
        float *out_oc = out + (size_t)oc * (size_t)len_out;
        for (int t = 0; t < len_out; ++t) {{
            float acc = b;
            const int in_start = t * stride - pad_l;
            for (int ic = 0; ic < c_in; ++ic) {{
                const float *in_ic = in + (size_t)ic * (size_t)len_in;
                const float *w_ic = w_oc + (size_t)ic * (size_t)k;
                for (int ki = 0; ki < k; ++ki) {{
                    const int idx = in_start + ki;
                    if (idx >= 0 && idx < len_in) {{
                        acc += in_ic[idx] * w_ic[ki];
                    }}
                }}
            }}
            out_oc[t] = acc;
        }}
    }}
}}

static void dwconv1d(
    const float *in,
    float *out,
    const float *weight,
    const float *bias,
    int c,
    int len_in,
    int k,
    int pad_l,
    int pad_r,
    int stride)
{{
    const int len_out = (len_in + pad_l + pad_r - k) / stride + 1;
    for (int ch = 0; ch < c; ++ch) {{
        const float b = bias[ch];
        const float *in_ch = in + (size_t)ch * (size_t)len_in;
        const float *w_ch = weight + (size_t)ch * (size_t)k;
        float *out_ch = out + (size_t)ch * (size_t)len_out;
        for (int t = 0; t < len_out; ++t) {{
            float acc = b;
            const int in_start = t * stride - pad_l;
            for (int ki = 0; ki < k; ++ki) {{
                const int idx = in_start + ki;
                if (idx >= 0 && idx < len_in) {{
                    acc += in_ch[idx] * w_ch[ki];
                }}
            }}
            out_ch[t] = acc;
        }}
    }}
}}

static void relu(float *buf, int n)
{{
    for (int i = 0; i < n; ++i) {{
        if (buf[i] < 0.0f) {{
            buf[i] = 0.0f;
        }}
    }}
}}

static void maxpool1d(
    const float *in,
    float *out,
    int c,
    int len_in,
    int k,
    int pad_l,
    int pad_r,
    int stride)
{{
    const int len_out = (len_in + pad_l + pad_r - k) / stride + 1;
    for (int ch = 0; ch < c; ++ch) {{
        const float *in_ch = in + (size_t)ch * (size_t)len_in;
        float *out_ch = out + (size_t)ch * (size_t)len_out;
        for (int t = 0; t < len_out; ++t) {{
            float m = -FLT_MAX;
            const int in_start = t * stride - pad_l;
            for (int ki = 0; ki < k; ++ki) {{
                const int idx = in_start + ki;
                if (idx >= 0 && idx < len_in) {{
                    const float v = in_ch[idx];
                    if (v > m) {{
                        m = v;
                    }}
                }}
            }}
            out_ch[t] = m;
        }}
    }}
}}

static void avgpool1d(
    const float *in,
    float *out,
    int c,
    int len_in,
    int k,
    int stride)
{{
    const int len_out = (len_in - k) / stride + 1;
    const float inv_k = 1.0f / (float)k;
    for (int ch = 0; ch < c; ++ch) {{
        const float *in_ch = in + (size_t)ch * (size_t)len_in;
        float *out_ch = out + (size_t)ch * (size_t)len_out;
        for (int t = 0; t < len_out; ++t) {{
            float sum = 0.0f;
            const int in_start = t * stride;
            for (int ki = 0; ki < k; ++ki) {{
                sum += in_ch[in_start + ki];
            }}
            out_ch[t] = sum * inv_k;
        }}
    }}
}}

static void gemm(
    const float *in,
    float *out,
    const float *weight,
    const float *bias,
    int in_dim,
    int out_dim)
{{
    for (int oc = 0; oc < out_dim; ++oc) {{
        float acc = bias[oc];
        const float *w_oc = weight + (size_t)oc * (size_t)in_dim;
        for (int ic = 0; ic < in_dim; ++ic) {{
            acc += in[ic] * w_oc[ic];
        }}
        out[oc] = acc;
    }}
}}

void model_fp32_infer(const float *input, float *output)
{{
{chr(10).join(call_lines)}
}}
"""

    h_path = out_dir / "model_fp32.h"
    c_path = out_dir / "model_fp32.c"
    h_path.write_text(header_content, encoding="utf-8")
    c_path.write_text(c_content, encoding="utf-8")

    return {
        "h_path": str(h_path),
        "c_path": str(c_path),
        "total_params": total_params,
        "weights_kb": round(total_params * 4 / 1024.0, 2),
        "ping_pong_elems_each": max_buf_elems,
        "ping_pong_total_bytes": max_buf_elems * 8,
        "ping_pong_total_kb": round(max_buf_elems * 8 / 1024.0, 2),
    }


def verify_on_pc(onnx_path: Path, test_x_path: Path, out_dir: Path, gcc_path: Path):
    X = np.load(str(test_x_path)).astype(np.float32)
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    n = X.shape[0]
    assert X.shape == (n, 1, 260), f"Unexpected test_x shape: {X.shape}"

    # 1. Run ONNXRuntime reference
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    ort_logits = np.empty((n, 2), dtype=np.float32)
    for i in range(n):
        ort_logits[i] = sess.run([out_name], {in_name: X[i : i + 1]})[0][0]

    # 2. Write PC verification harness
    verify_c = out_dir / "verify_pc.c"
    verify_c.write_text(
        """#include <stdio.h>
#include <stdlib.h>
#include "model_fp32.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <in.bin> <out.bin> <num_samples>\\n", argv[0]);
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
            fprintf(stderr, "Short read at sample %d\\n", i);
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
""",
        encoding="utf-8",
    )

    in_bin = out_dir / "tmp_test500_in.bin"
    out_bin = out_dir / "tmp_test500_out.bin"
    exe_path = out_dir / "verify_pc.exe"

    np.ascontiguousarray(X.reshape(n, 260), dtype=np.float32).tofile(str(in_bin))

    # 3. Compile with GCC
    cmd_gcc = [
        str(gcc_path),
        "-O2",
        "-Wall",
        "-Wextra",
        "-std=c11",
        str(out_dir / "model_fp32.c"),
        str(verify_c),
        "-o",
        str(exe_path),
    ]
    res = subprocess.run(cmd_gcc, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"GCC compilation failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

    # 4. Run verify_pc.exe
    res_run = subprocess.run([str(exe_path), str(in_bin), str(out_bin), str(n)], capture_output=True, text=True)
    if res_run.returncode != 0:
        raise RuntimeError(f"verify_pc.exe failed:\n{res_run.stderr}")

    c_logits = np.fromfile(str(out_bin), dtype=np.float32).reshape(n, 2)

    # Clean up temporary binaries
    in_bin.unlink(missing_ok=True)
    out_bin.unlink(missing_ok=True)

    abs_diff = np.abs(c_logits - ort_logits)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    argmax_match = int(np.sum(np.argmax(c_logits, axis=1) == np.argmax(ort_logits, axis=1)))
    gate_d1_pass = max_diff < 1e-4

    return {
        "num_beats": n,
        "gcc_warnings": res.stderr.strip(),
        "max_logit_diff": max_diff,
        "mean_logit_diff": mean_diff,
        "argmax_match": f"{argmax_match}/{n}",
        "gate_d1_threshold": 1e-4,
        "gate_d1_pass": gate_d1_pass,
        "sample0_ort": [float(v) for v in ort_logits[0]],
        "sample0_c": [float(v) for v in c_logits[0]],
    }


def main():
    parser = argparse.ArgumentParser(description="Phase D1: Generate & verify pure C FP32 model")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--test-x", type=Path, default=DEFAULT_TEST_X)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--gcc", type=Path, default=GCC_PATH)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    gen_info = generate_c_code(args.onnx, args.out_dir)
    print("Generated C FP32 model:")
    print(json.dumps(gen_info, indent=2))

    summary = {"generation": gen_info}
    if not args.no_verify:
        ver_info = verify_on_pc(args.onnx, args.test_x, args.out_dir, args.gcc)
        summary["pc_verification"] = ver_info
        print("\nPC Verification (GCC vs ONNXRuntime):")
        print(json.dumps(ver_info, indent=2))
        if not ver_info["gate_d1_pass"]:
            sys.exit(1)

    report_path = PROJECT_ROOT / "esp32s3" / "results" / "gate_d1_fp32_c.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved Gate D1 report to {report_path}")


if __name__ == "__main__":
    main()
