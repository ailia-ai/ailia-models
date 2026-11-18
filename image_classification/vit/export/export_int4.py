"""Export ViT-B/16 as an int4 MatMulNBits ONNX model.

Tested versions:
    onnxruntime 1.23.2
    onnx 1.23.2

Requirements:
    pip install onnxruntime onnx numpy

Usage:
    cd image_classification/vit/export
    python3 export_int4.py
"""

import argparse
import collections
import logging
import os
import subprocess
import sys
import urllib.request

import onnx

try:
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
except ImportError:  # onnxruntime < 1.22
    from onnxruntime.quantization.matmul_4bits_quantizer import (
        MatMul4BitsQuantizer as MatMulNBitsQuantizer,
    )
from onnxruntime.quantization.quant_utils import QuantFormat


REMOTE_PATH = "https://storage.googleapis.com/ailia-models/vit/"
ORIGINAL_MODEL = "ViT-B_16-224.onnx"
QUANTIZED_MODEL = "ViT-B_16-224-int4.onnx"
EXPECTED_QUANTIZED_NODES = 72
EXPECTED_DYNAMIC_MATMUL_NODES = 24


def download_model(model_name):
    """Download a model from ailia-models storage when it is not present."""
    if os.path.exists(model_name):
        print(f"  {model_name} already exists, skipping download.")
        return
    url = REMOTE_PATH + model_name
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, model_name)


def generate_prototxt(onnx_path):
    """Generate the ailia prototxt corresponding to an ONNX model."""
    script_name = "onnx2prototxt.py"
    if not os.path.exists(script_name):
        script_url = (
            "https://raw.githubusercontent.com/ailia-ai/"
            "export-to-onnx/master/onnx2prototxt.py"
        )
        print(f"  Downloading {script_url} ...")
        urllib.request.urlretrieve(script_url, script_name)

    print(f"  Generating prototxt for {onnx_path} ...")
    subprocess.check_call([sys.executable, script_name, onnx_path])
    return onnx_path + ".prototxt"


def validate_result(model):
    """Check graph structure and Neutron-compatible MatMulNBits attributes."""
    counts = collections.Counter(node.op_type for node in model.graph.node)
    if counts["MatMulNBits"] != EXPECTED_QUANTIZED_NODES:
        raise RuntimeError(
            f"expected {EXPECTED_QUANTIZED_NODES} MatMulNBits nodes, "
            f"got {counts['MatMulNBits']}"
        )
    if counts["MatMul"] != EXPECTED_DYNAMIC_MATMUL_NODES:
        raise RuntimeError(
            f"expected {EXPECTED_DYNAMIC_MATMUL_NODES} dynamic MatMul nodes, "
            f"got {counts['MatMul']}"
        )

    for node in model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        attrs = {
            attr.name: onnx.helper.get_attribute_value(attr)
            for attr in node.attribute
        }
        if attrs["bits"] != 4 or attrs["K"] % 16 or attrs["N"] % 128:
            raise RuntimeError(
                f"{node.name} is not compatible with the Neutron EP: {attrs}"
            )
        if len(node.input) > 3:
            raise RuntimeError(f"{node.name} has an unsupported zero-point input")

    onnx.checker.check_model(model)
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Export ViT-B/16 int4 MatMulNBits ONNX model"
    )
    parser.add_argument(
        "--output_dir",
        default="..",
        help="output directory containing the fp32 and int4 models",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    print("[1/3] Preparing fp32 model ...")
    download_model(ORIGINAL_MODEL)

    print("[2/3] Quantizing constant-weight MatMul nodes to int4 ...")
    model = onnx.load(ORIGINAL_MODEL)
    logging.getLogger(
        "onnxruntime.quantization.matmul_nbits_quantizer"
    ).setLevel(logging.WARNING)
    quantizer = MatMulNBitsQuantizer(
        model=model,
        block_size=128,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
    )
    quantizer.process()
    result = quantizer.model.model
    counts = validate_result(result)

    print(f"  Saving {QUANTIZED_MODEL} ...")
    onnx.save(result, QUANTIZED_MODEL)
    print(
        f"  MatMulNBits={counts['MatMulNBits']}, "
        f"dynamic MatMul={counts['MatMul']}, Gemm={counts['Gemm']}"
    )

    print("[3/3] Generating ailia prototxt ...")
    prototxt_path = generate_prototxt(QUANTIZED_MODEL)

    print("\nDone! Generated files:")
    print(f"  - {os.path.join(output_dir, QUANTIZED_MODEL)}")
    print(f"  - {os.path.join(output_dir, prototxt_path)}")


if __name__ == "__main__":
    main()
