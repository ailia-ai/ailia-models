"""
Export BEVFormer-tiny int4 quantized ONNX model.

Tested versions:
    onnxruntime 1.24.4
    onnx 1.20.1

Requirements:
    pip install onnxruntime onnx numpy

Usage:
    python export_bevformer_int4.py
    python export_bevformer_int4.py --output_dir /path/to/output

This script quantizes the BEVFormer-tiny model to int4 using:
  - MatMulNBits (4-bit) for linear layers (MatMul/Gemm nodes)

Unlike the int8 variant, Conv layers are NOT quantized (no ConvInteger /
QConvLinear). Only MatMulNBits is applied. Calibration-free dynamic
quantization (no calibration dataset required).
"""

import os
import sys
import argparse
import subprocess

import numpy as np
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
    """Convert Gemm nodes to MatMul+Add with in-place weight transpose.

    MatMulNBitsQuantizer does not support Gemm, so we convert Gemm nodes to
    MatMul (with transposed weight initializers) + Add (for bias).
    """
    graph = model.graph

    # Clear external data references so weights are accessible as inline data
    for tensor in graph.initializer:
        while len(tensor.external_data) > 0:
            tensor.external_data.pop()
        tensor.ClearField("data_location")

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
        elif transB:
            transpose_out = f"{node.name}_transB_out"
            nodes_to_add.append(helper.make_node(
                "Transpose", inputs=[B], outputs=[transpose_out],
                name=f"{node.name}_transpose", perm=[1, 0]))
            B = transpose_out

        matmul_out = f"{node.name}_matmul_out"
        if len(node.input) > 2 and node.input[2]:
            nodes_to_add.append(helper.make_node(
                "MatMul", inputs=[A, B], outputs=[matmul_out],
                name=f"{node.name}_matmul"))
            nodes_to_add.append(helper.make_node(
                "Add", inputs=[matmul_out, node.input[2]], outputs=node.output,
                name=f"{node.name}_add"))
        else:
            nodes_to_add.append(helper.make_node(
                "MatMul", inputs=[A, B], outputs=node.output,
                name=f"{node.name}_matmul"))
        nodes_to_remove.append(node)

    for node in nodes_to_remove:
        graph.node.remove(node)
    for node in nodes_to_add:
        graph.node.append(node)

    print(f"  Converted {len(nodes_to_remove)} Gemm nodes to MatMul+Add")
    return model


def remove_unused_initializers(model):
    """Remove initializers not referenced by any node or graph input."""
    graph = model.graph
    used = set()
    for node in graph.node:
        for inp in node.input:
            used.add(inp)
    for inp in graph.input:
        used.add(inp.name)

    to_remove = [t for t in graph.initializer if t.name not in used]
    for t in to_remove:
        graph.initializer.remove(t)
    if to_remove:
        print(f"  Removed {len(to_remove)} unused initializers")
    return model


def generate_prototxt(onnx_path):
    """Generate prototxt from ONNX model using onnx2prototxt.py."""
    prototxt_path = onnx_path + ".prototxt"
    script_url = ("https://raw.githubusercontent.com/ailia-ai/export-to-onnx/"
                  "master/onnx2prototxt.py")
    script_path = "onnx2prototxt.py"

    if not os.path.exists(script_path):
        print("  Downloading onnx2prototxt.py ...")
        subprocess.check_call(["wget", "-q", script_url, "-O", script_path])

    print(f"  Generating prototxt for {onnx_path} ...")
    subprocess.check_call([sys.executable, script_path, onnx_path, prototxt_path])
    return prototxt_path


def main():
    parser = argparse.ArgumentParser(
        description="Export BEVFormer-tiny int4 quantized model")
    parser.add_argument(
        "--output_dir", type=str, default="..",
        help="Output directory for quantized model files (default: parent dir)")
    args = parser.parse_args()

    remote_path = "https://storage.googleapis.com/ailia-models/bevformer/"
    work_dir = os.path.abspath(args.output_dir)
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    # Step 1: Download original model
    print("[1/4] Downloading original model ...")
    orig = "bevformer_tiny.onnx"
    download_model(orig, remote_path)

    # Step 2: Load and convert Gemm to MatMul+Add
    print("[2/4] Converting Gemm to MatMul+Add ...")
    model = onnx.load(orig)
    model = convert_gemm_to_matmul(model)

    # Step 3: Quantize MatMul to MatMulNBits (int4)
    print("[3/4] Quantizing MatMul to MatMulNBits (int4) ...")
    quant = MatMulNBitsQuantizer(
        model=model,
        bits=4,
        block_size=128,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
    )
    quant.process()

    result = quant.model.model

    nbits = sum(1 for n in result.graph.node if n.op_type == "MatMulNBits")
    convint = sum(1 for n in result.graph.node if n.op_type == "ConvInteger")
    conv = sum(1 for n in result.graph.node if n.op_type == "Conv")
    print(f"  MatMulNBits nodes: {nbits}")
    print(f"  ConvInteger nodes: {convint}")
    print(f"  Conv nodes (unquantized): {conv}")

    # Cleanup unused initializers and save
    remove_unused_initializers(result)

    out_path = os.path.join(work_dir, "bevformer_tiny_int4.onnx")
    print(f"  Saving: {out_path}")
    onnx.save(result, out_path)

    orig_size = os.path.getsize(orig) / 1024 / 1024
    out_size = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Original size: {orig_size:.0f}MB")
    print(f"  Quantized size: {out_size:.0f}MB")

    # Step 4: Generate prototxt
    print("[4/4] Generating prototxt ...")
    prototxt_path = generate_prototxt(out_path)

    print(f"\nDone! Generated files:")
    print(f"  - {out_path}")
    print(f"  - {prototxt_path}")


if __name__ == "__main__":
    main()
