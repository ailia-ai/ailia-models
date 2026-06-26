"""
Export SenseVoice int4/int8 quantized ONNX models.

Tested versions:
    onnxruntime 1.24.4
    onnx 1.20.1

Requirements:
    pip install onnxruntime onnx numpy

Usage:
    python export_quantized.py
    python export_quantized.py --bits 4
    python export_quantized.py --bits 8

Size reduction:
  sensevoice_small: 894MB -> 123MB (int4) / 234MB (int8)
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
        description="Export SenseVoice quantized models"
    )
    parser.add_argument(
        "--bits", type=int, default=4, choices=[4, 8],
        help="Quantization bits (4 or 8, default: 4)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="..",
        help="Output directory (default: parent dir)",
    )
    args = parser.parse_args()

    remote_path = "https://storage.googleapis.com/ailia-models/sensevoice/"
    work_dir = os.path.abspath(args.output_dir)
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    suffix = f"int{args.bits}"
    orig = "sensevoice_small.onnx"
    out = f"sensevoice_small_{suffix}.onnx"

    # Step 1: Download
    print("[1/3] Downloading original model ...")
    download_model(orig, remote_path)

    # Step 2: Quantize
    print(f"[2/3] Quantizing to {suffix} ...")
    out_path = os.path.join(work_dir, out)
    quant = MatMulNBitsQuantizer(
        model=orig, bits=args.bits, block_size=128, is_symmetric=True,
        accuracy_level=4, quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
    )
    quant.process()

    result = quant.model.model
    nbits = sum(1 for n in result.graph.node if n.op_type == "MatMulNBits")
    print(f"  MatMulNBits nodes created: {nbits}")
    print(f"  Saving: {out_path}")
    onnx.save(result, out_path)

    size = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Output size: {size:.0f}MB")

    # Step 3: Generate prototxt
    print("[3/3] Generating prototxt ...")
    prototxt_path = generate_prototxt(out_path)

    print(f"\nDone! Generated files:")
    print(f"  - {out_path}")
    print(f"  - {prototxt_path}")


if __name__ == "__main__":
    main()
