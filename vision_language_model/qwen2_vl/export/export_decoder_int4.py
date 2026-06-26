"""
Export Qwen2-VL-2B LLM decoder int4 quantized ONNX model.

Tested versions:
    onnxruntime 1.24.4
    onnx 1.20.1

Requirements:
    pip install onnxruntime onnx numpy

Usage:
    python export_decoder_int4.py
    python export_decoder_int4.py --output_dir /path/to/output

This script quantizes the Qwen2-VL-2B LLM decoder to int4 (4-bit weight-only
quantization) using onnxruntime's MatMulNBitsQuantizer.
"""

import os
import sys
import argparse
import subprocess

import onnx
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
        description="Export Qwen2-VL-2B decoder int4 quantized model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="..",
        help="Output directory for quantized model files (default: parent dir)",
    )
    args = parser.parse_args()

    remote_path = "https://storage.googleapis.com/ailia-models/qwen2_vl/"
    work_dir = os.path.abspath(args.output_dir)
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    # Step 1: Download original fp32 ONNX model
    print("[1/3] Downloading original fp32 model ...")
    original_model = "Qwen2-VL-2B.onnx"
    original_pb = "Qwen2-VL-2B_weights.pb"
    download_model(original_model, remote_path)
    download_model(original_pb, remote_path)

    # Step 2: Quantize LLM decoder to int4
    print("[2/3] Quantizing LLM decoder to int4 ...")
    quantized_model = os.path.join(work_dir, "Qwen2-VL-2B_int4.onnx")
    print("  Quantizing to int4 (block_size=128, symmetric) ...")
    quant = MatMulNBitsQuantizer(
        model=original_model,
        bits=4,
        block_size=128,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
    )
    quant.process()

    print(f"  Saving quantized model: {quantized_model}")
    onnx.save(quant.model.model, quantized_model)

    # Step 3: Generate prototxt
    print("[3/3] Generating prototxt ...")
    prototxt_path = generate_prototxt(quantized_model)

    print("\nDone! Generated files:")
    print(f"  - {quantized_model}")
    print(f"  - {prototxt_path}")


if __name__ == "__main__":
    main()
