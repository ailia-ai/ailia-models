"""
Export GPT-SoVITS v2-Pro int8 quantized ONNX models.

Tested versions:
    onnxruntime 1.24.4
    onnx 1.20.1

Requirements:
    pip install onnxruntime onnx numpy

Usage:
    python export_int8.py
    python export_int8.py --output_dir /path/to/output

This script quantizes the following GPT-SoVITS v2-Pro models to int8:
  - cnhubert (SSL feature extractor): 360MB -> 119MB
  - t2s_fsdec (T2S first-stage decoder): 293MB -> 78MB
  - t2s_sdec (T2S stage decoder): 293MB -> 78MB

Models not quantized:
  - t2s_encoder.onnx (11MB, mostly Embedding/Conv, only 2 MatMul nodes)
  - vits.onnx (vocoder, 251MB, mostly Conv)
  - sv.onnx (speaker verification, 174MB, all Conv)

The quantization uses onnxruntime's MatMulNBitsQuantizer (8-bit).
For t2s_fsdec and t2s_sdec, Gemm nodes are first converted to MatMul+Add.
"""

import os
import sys
import argparse
import subprocess

import onnx
from onnx import numpy_helper, helper, TensorProto
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from onnxruntime.quantization.quant_utils import QuantFormat


def download_model(model_name, remote_path):
    """Download model file from remote storage if not present."""
    if os.path.exists(model_name):
        print(f"  {model_name} already exists, skipping download.")
        return
    url = remote_path + model_name
    print(f"  Downloading {model_name} ...")
    subprocess.check_call(["wget", "-q", url, "-O", model_name])


def convert_gemm_to_matmul(model):
    """Convert Gemm nodes to MatMul+Add with in-place weight transpose."""
    graph = model.graph
    init_map = {t.name: i for i, t in enumerate(graph.initializer)}
    nodes_to_remove = []
    nodes_to_add = []

    for node in graph.node:
        if node.op_type != "Gemm":
            continue
        transB = 0
        for attr in node.attribute:
            if attr.name == "transB":
                transB = attr.i

        A = node.input[0]
        B = node.input[1]

        if transB and B in init_map:
            idx = init_map[B]
            w = numpy_helper.to_array(graph.initializer[idx])
            w_t = w.T.copy()
            new_name = B + "_transposed"
            graph.initializer.append(numpy_helper.from_array(w_t, name=new_name))
            B = new_name

        matmul_out = f"{node.name}_matmul_out"
        if len(node.input) > 2 and node.input[2]:
            nodes_to_add.append(
                onnx.helper.make_node(
                    "MatMul", inputs=[A, B], outputs=[matmul_out],
                    name=f"{node.name}_matmul",
                )
            )
            nodes_to_add.append(
                onnx.helper.make_node(
                    "Add", inputs=[matmul_out, node.input[2]], outputs=node.output,
                    name=f"{node.name}_add",
                )
            )
        else:
            nodes_to_add.append(
                onnx.helper.make_node(
                    "MatMul", inputs=[A, B], outputs=node.output,
                    name=f"{node.name}_matmul",
                )
            )
        nodes_to_remove.append(node)

    for node in nodes_to_remove:
        graph.node.remove(node)
    for node in nodes_to_add:
        graph.node.append(node)

    print(f"  Converted {len(nodes_to_remove)} Gemm nodes to MatMul+Add")
    return model


def fix_topk(model):
    """Fix TopK K input shape for onnxruntime 1.24+ compatibility."""
    graph = model.graph
    nodes_fixed = 0
    for node in list(graph.node):
        if node.op_type != "TopK":
            continue
        k_input = node.input[1]
        reshape_out = k_input + "_reshaped_1d"
        shape_const_name = k_input + "_reshape_shape"
        graph.initializer.append(
            helper.make_tensor(shape_const_name, TensorProto.INT64, [1], [1])
        )
        idx = list(graph.node).index(node)
        graph.node.insert(
            idx,
            helper.make_node(
                "Reshape",
                inputs=[k_input, shape_const_name],
                outputs=[reshape_out],
                name=node.name + "_reshape_k",
            ),
        )
        node.input[1] = reshape_out
        nodes_fixed += 1
    if nodes_fixed:
        print(f"  Fixed {nodes_fixed} TopK nodes")
    return model


def quantize_model(input_path, output_path, has_gemm=False):
    """Quantize an ONNX model to int8 using MatMulNBitsQuantizer."""
    if has_gemm:
        print("  Loading and converting Gemm to MatMul+Add...")
        model = onnx.load(input_path)
        model = convert_gemm_to_matmul(model)
    else:
        model = input_path

    print("  Quantizing to int8 (block_size=128, symmetric) ...")
    quant = MatMulNBitsQuantizer(
        model=model,
        bits=8,
        block_size=128,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
    )
    quant.process()

    result = quant.model.model
    nbits = sum(1 for n in result.graph.node if n.op_type == "MatMulNBits")
    print(f"  MatMulNBits nodes created: {nbits}")

    result = fix_topk(result)

    print(f"  Saving: {output_path}")
    onnx.save(result, output_path)

    size = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Output size: {size:.0f}MB")


def generate_prototxt(onnx_path):
    """Generate prototxt from ONNX model using onnx2prototxt.py."""
    prototxt_path = onnx_path + ".prototxt"
    script_url = "https://raw.githubusercontent.com/ailia-ai/export-to-onnx/master/onnx2prototxt.py"
    script_path = "onnx2prototxt.py"

    if not os.path.exists(script_path):
        print("  Downloading onnx2prototxt.py ...")
        subprocess.check_call(["wget", "-q", script_url, "-O", script_path])

    print(f"  Generating prototxt for {onnx_path} ...")
    subprocess.check_call([sys.executable, script_path, onnx_path, prototxt_path])
    return prototxt_path


def main():
    parser = argparse.ArgumentParser(
        description="Export GPT-SoVITS v2-Pro int8 quantized models"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="..",
        help="Output directory (default: parent dir)",
    )
    args = parser.parse_args()

    remote_path = "https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro/"
    work_dir = os.path.abspath(args.output_dir)
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    models = [
        ("cnhubert.onnx", "cnhubert_int8.onnx", False),
        ("t2s_fsdec.onnx", "t2s_fsdec_int8.onnx", True),
        ("t2s_sdec.opt.onnx", "t2s_sdec_int8.onnx", True),
    ]

    # Step 1: Download
    print("[1/3] Downloading original models ...")
    for orig, _, _ in models:
        download_model(orig, remote_path)

    # Step 2: Quantize
    print("[2/3] Quantizing models to int8 ...")
    output_files = []
    for orig, out, has_gemm in models:
        print(f"\n--- {orig} ---")
        out_path = os.path.join(work_dir, out)
        quantize_model(orig, out_path, has_gemm=has_gemm)
        output_files.append(out_path)

    # Step 3: Generate prototxt
    print("\n[3/3] Generating prototxt ...")
    for out_path in output_files:
        generate_prototxt(out_path)

    print("\nDone! Generated files:")
    for out_path in output_files:
        print(f"  - {out_path}")
        print(f"  - {out_path}.prototxt")


if __name__ == "__main__":
    main()
