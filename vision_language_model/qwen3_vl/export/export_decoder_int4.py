"""
Export Qwen3-VL-4B language model (decoder) int4 quantized ONNX model.

Tested versions:
    onnxruntime 1.24.4
    onnx 1.20.1

Requirements:
    pip install onnxruntime onnx numpy

Usage:
    python export_decoder_int4.py
    python export_decoder_int4.py --output_dir /path/to/output

This script quantizes the Qwen3-VL-4B language model to int4 (4-bit
weight-only quantization) using onnxruntime's MatMulNBitsQuantizer.
Only the decoder is quantized; the vision encoder and embed_tokens stay fp32.
The quantized weights still exceed the 2GB protobuf limit, so the model is
saved with external data (*_weights.pb).
"""

import argparse
import os
import subprocess
import sys
import urllib.request

import numpy as np
import onnx
from onnx import numpy_helper

try:
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
except ImportError:  # onnxruntime < 1.22
    from onnxruntime.quantization.matmul_4bits_quantizer import (
        MatMul4BitsQuantizer as MatMulNBitsQuantizer,
    )
from onnxruntime.quantization.quant_utils import QuantFormat

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/qwen3_vl/"
ORIGINAL_MODEL = "qwen3_vl_4b_instruct_language_model.onnx"
ORIGINAL_PB = "qwen3_vl_4b_instruct_language_model_weights.pb"
QUANTIZED_MODEL = "qwen3_vl_4b_instruct_language_model_int4.onnx"
QUANTIZED_PB = "qwen3_vl_4b_instruct_language_model_int4_weights.pb"


def download_model(model_name, remote_path):
    """Download model file from remote storage if not present."""
    if os.path.exists(model_name):
        print(f"  {model_name} already exists, skipping download.")
        return
    url = remote_path + model_name
    print(f"  Downloading {model_name} ...")
    urllib.request.urlretrieve(url, model_name)


def generate_prototxt(onnx_path):
    """Generate prototxt from ONNX model using onnx2prototxt.py."""
    prototxt_path = onnx_path + ".prototxt"
    script_url = "https://raw.githubusercontent.com/ailia-ai/export-to-onnx/master/onnx2prototxt.py"
    script_path = "onnx2prototxt.py"

    if not os.path.exists(script_path):
        print("  Downloading onnx2prototxt.py ...")
        urllib.request.urlretrieve(script_url, script_path)

    print(f"  Generating prototxt for {onnx_path} ...")
    subprocess.check_call([sys.executable, script_path, onnx_path])
    return prototxt_path


def fold_transposed_weights(model):
    """Fold weight-initializer -> Transpose -> MatMul patterns.

    torch.onnx.export emits nn.Linear as Transpose(weight) followed by
    MatMul, so the MatMul input is not a constant and MatMulNBitsQuantizer
    skips it ("MatMul doesn't have const weight"). Transpose the weight data
    offline and feed it to MatMul directly so it can be quantized.
    """
    graph = model.graph
    initializers = {t.name: t for t in graph.initializer}

    consumers = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    folded = 0
    for node in list(graph.node):
        if node.op_type != "Transpose" or node.input[0] not in initializers:
            continue
        tensor = initializers[node.input[0]]
        if len(tensor.dims) != 2:
            continue
        perm = next((list(a.ints) for a in node.attribute if a.name == "perm"), None)
        if perm is not None and perm != [1, 0]:
            continue
        used_by = consumers.get(node.output[0], [])
        if not used_by or any(
            n.op_type != "MatMul" or n.input[1] != node.output[0] for n in used_by
        ):
            continue

        weight = numpy_helper.to_array(tensor)
        transposed = numpy_helper.from_array(
            np.ascontiguousarray(weight.T), name=tensor.name + "_transposed"
        )
        graph.initializer.append(transposed)
        for matmul in used_by:
            matmul.input[1] = transposed.name
        graph.node.remove(node)
        folded += 1

    # drop original initializers that are no longer referenced
    used = {name for node in graph.node for name in node.input}
    for tensor in list(graph.initializer):
        if tensor.name not in used:
            graph.initializer.remove(tensor)

    print(f"  Folded {folded} transposed weights into MatMul")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen3-VL-4B decoder int4 quantized model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="..",
        help="Output directory for quantized model files (default: parent dir)",
    )
    args = parser.parse_args()

    work_dir = os.path.abspath(args.output_dir)
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    # Step 1: Download original fp32 ONNX model
    print("[1/3] Downloading original fp32 model ...")
    download_model(ORIGINAL_MODEL, REMOTE_PATH)
    download_model(ORIGINAL_PB, REMOTE_PATH)

    # Step 2: Quantize language model to int4
    print("[2/3] Quantizing language model to int4 ...")
    print("  Loading model (with external data) ...")
    model = onnx.load(ORIGINAL_MODEL)
    model = fold_transposed_weights(model)

    print("  Quantizing to int4 (block_size=128, symmetric) ...")
    quant = MatMulNBitsQuantizer(
        model=model,
        block_size=128,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
    )
    quant.process()

    print(f"  Saving quantized model: {QUANTIZED_MODEL}")
    if os.path.exists(QUANTIZED_PB):
        os.remove(QUANTIZED_PB)
    onnx.save_model(
        quant.model.model,
        QUANTIZED_MODEL,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=QUANTIZED_PB,
        size_threshold=1024,
    )

    # Step 3: Generate prototxt
    print("[3/3] Generating prototxt ...")
    prototxt_path = generate_prototxt(QUANTIZED_MODEL)

    print("\nDone! Generated files:")
    print(f"  - {QUANTIZED_MODEL}")
    print(f"  - {QUANTIZED_PB}")
    print(f"  - {prototxt_path}")


if __name__ == "__main__":
    main()
