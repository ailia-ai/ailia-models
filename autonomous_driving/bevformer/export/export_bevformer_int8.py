"""
Export BEVFormer-tiny int8 quantized ONNX model.

Tested versions:
    onnxruntime 1.24.4
    onnx 1.20.1

Requirements:
    pip install onnxruntime onnx numpy

Usage:
    python export_bevformer_int8.py
    python export_bevformer_int8.py --output_dir /path/to/output

This script quantizes the BEVFormer-tiny model to int8 using two techniques:
  - MatMulNBits (8-bit) for linear layers (MatMul/Gemm nodes)
  - QConvLinear for Conv layers: DynamicQuantizeLinear -> ConvInteger
        -> DequantizeLinear(int32) -> Add(bias)

QConvLinear replaces each Conv node with:
    1. DynamicQuantizeLinear: float -> uint8 + x_scale + x_zp (1 node)
    2. ConvInteger: uint8 input x int8 weight -> int32 (1 node)
    3. Mul: x_scale * w_scale -> deq_scale (1 node)
    4. DequantizeLinear: int32 -> float via deq_scale (1 node)
    5. Add: float + bias (1 node)

ConvInteger outputs int32 (no clipping), DequantizeLinear fuses Cast+Mul.
Calibration-free dynamic quantization (no calibration dataset required).
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


def _extract_conv_attrs(node):
    """Extract Conv node attributes as kwargs dict for helper.make_node."""
    attrs = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.INTS:
            attrs[attr.name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.INT:
            attrs[attr.name] = attr.i
        elif attr.type == onnx.AttributeProto.STRING:
            attrs[attr.name] = attr.s
    return attrs


def convert_conv_to_qconvlinear(model):
    """Convert Conv nodes to QConvLinear using DynamicQuantizeLinear + ConvInteger.

    QConvLinear replaces each Conv with:
        DynamicQuantizeLinear(input) -> x_uint8, x_scale, x_zp   (1 node)
        ConvInteger(x_uint8, w_int8, x_zp, w_zp) -> int32        (1 node)
        Mul(x_scale, w_scale) -> deq_scale                       (1 node)
        DequantizeLinear(int32, deq_scale, 0) -> float            (1 node)
        Add(float, bias_4d) -> output                             (1 node)

    ConvInteger outputs int32 (no clipping, full precision).
    DequantizeLinear on int32 fuses Cast + scale multiplication.
    """
    graph = model.graph
    init_map = {t.name: t for t in graph.initializer}

    # Shared constants
    zp_int8 = "_qcl_zp8"
    zp_int32 = "_qcl_zp32"

    graph.initializer.extend([
        numpy_helper.from_array(np.int8(0), name=zp_int8),
        numpy_helper.from_array(np.int32(0), name=zp_int32),
    ])

    nodes_to_remove = []
    nodes_to_add = []
    count = 0

    for node in list(graph.node):
        if node.op_type != "Conv":
            continue

        w_name = node.input[1]
        if w_name not in init_map:
            continue

        w_data = numpy_helper.to_array(init_map[w_name])
        w_abs_max = float(np.abs(w_data).max())
        if w_abs_max < 1e-10:
            continue

        w_scale_val = w_abs_max / 127.0
        w_int8 = np.clip(
            np.round(w_data / w_scale_val), -127, 127
        ).astype(np.int8)

        p = f"_qcl{count}"
        count += 1

        # Static initializers: int8 weight and w_scale
        qw = f"{p}_w"
        ws = f"{p}_ws"
        graph.initializer.append(numpy_helper.from_array(w_int8, name=qw))
        graph.initializer.append(numpy_helper.from_array(
            np.float32(w_scale_val), name=ws))

        # Pre-reshape bias to (1,C,1,1)
        has_bias = len(node.input) > 2 and node.input[2]
        bias_4d = None
        if has_bias and node.input[2] in init_map:
            b_data = numpy_helper.to_array(init_map[node.input[2]])
            bias_4d = f"{p}_b4d"
            graph.initializer.append(numpy_helper.from_array(
                b_data.reshape(1, -1, 1, 1).astype(np.float32), name=bias_4d))

        inp = node.input[0]
        conv_attrs = _extract_conv_attrs(node)

        # DynamicQuantizeLinear: float -> uint8, x_scale, x_zp
        nodes_to_add.append(helper.make_node(
            "DynamicQuantizeLinear", [inp],
            [f"{p}_xu8", f"{p}_xs", f"{p}_xzp"]))

        # ConvInteger: uint8 x int8 -> int32
        nodes_to_add.append(helper.make_node(
            "ConvInteger",
            [f"{p}_xu8", qw, f"{p}_xzp", zp_int8],
            [f"{p}_ci"], **conv_attrs))

        # deq_scale = x_scale * w_scale
        nodes_to_add.append(helper.make_node(
            "Mul", [f"{p}_xs", ws], [f"{p}_ds"]))

        # DequantizeLinear: int32 * deq_scale -> float
        if bias_4d:
            nodes_to_add.append(helper.make_node(
                "DequantizeLinear",
                [f"{p}_ci", f"{p}_ds", zp_int32], [f"{p}_dq"]))
            nodes_to_add.append(helper.make_node(
                "Add", [f"{p}_dq", bias_4d], node.output))
        else:
            nodes_to_add.append(helper.make_node(
                "DequantizeLinear",
                [f"{p}_ci", f"{p}_ds", zp_int32], node.output))

        nodes_to_remove.append(node)

    for n in nodes_to_remove:
        graph.node.remove(n)
    graph.node.extend(nodes_to_add)

    print(f"  Converted {len(nodes_to_remove)} Conv to QConvLinear (int8)")
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
        description="Export BEVFormer-tiny int8 quantized model")
    parser.add_argument(
        "--output_dir", type=str, default="..",
        help="Output directory for quantized model files (default: parent dir)")
    args = parser.parse_args()

    remote_path = "https://storage.googleapis.com/ailia-models/bevformer/"
    work_dir = os.path.abspath(args.output_dir)
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    # Step 1: Download original model
    print("[1/5] Downloading original model ...")
    orig = "bevformer_tiny.onnx"
    download_model(orig, remote_path)

    # Step 2: Load and convert Gemm to MatMul+Add
    print("[2/5] Converting Gemm to MatMul+Add ...")
    model = onnx.load(orig)
    model = convert_gemm_to_matmul(model)

    # Step 3: Convert Conv to QConvLinear (int8)
    print("[3/5] Converting Conv to QConvLinear (int8) ...")
    model = convert_conv_to_qconvlinear(model)

    # Step 4: Quantize MatMul to MatMulNBits (int8)
    print("[4/5] Quantizing MatMul to MatMulNBits (int8) ...")
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
    convint = sum(1 for n in result.graph.node if n.op_type == "ConvInteger")
    dynq = sum(1 for n in result.graph.node if n.op_type == "DynamicQuantizeLinear")
    print(f"  MatMulNBits nodes: {nbits}")
    print(f"  ConvInteger nodes: {convint}")
    print(f"  DynamicQuantizeLinear nodes: {dynq}")

    # Cleanup unused initializers and save
    remove_unused_initializers(result)

    out_path = os.path.join(work_dir, "bevformer_tiny_int8.onnx")
    print(f"  Saving: {out_path}")
    onnx.save(result, out_path)

    orig_size = os.path.getsize(orig) / 1024 / 1024
    out_size = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Original size: {orig_size:.0f}MB")
    print(f"  Quantized size: {out_size:.0f}MB")

    # Step 5: Generate prototxt
    print("[5/5] Generating prototxt ...")
    prototxt_path = generate_prototxt(out_path)

    print(f"\nDone! Generated files:")
    print(f"  - {out_path}")
    print(f"  - {prototxt_path}")


if __name__ == "__main__":
    main()
