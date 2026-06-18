"""
Export Whisper V3 Turbo int4/int8 quantized ONNX models.

Tested versions:
    onnxruntime 1.24.4
    onnx 1.20.1

Requirements:
    pip install onnxruntime onnx numpy

Usage:
    python export_turbo_quantized.py
    python export_turbo_quantized.py --bits 4
    python export_turbo_quantized.py --bits 8

This script quantizes the Whisper V3 Turbo encoder and decoder to int4 or int8
using onnxruntime's MatMulNBitsQuantizer.

Size reduction:
  encoder: 2.4GB (onnx+pb) -> 349MB (int4) / 649MB (int8)
  decoder: 910MB -> 343MB (int4) / 425MB (int8)
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


def quantize_model(input_path, output_path, bits):
    """Quantize an ONNX model using MatMulNBitsQuantizer."""
    print(f"  Quantizing to int{bits} (block_size=128, symmetric) ...")
    quant = MatMulNBitsQuantizer(
        model=input_path,
        bits=bits,
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
        description="Export Whisper V3 Turbo quantized models"
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

    remote_path = "https://storage.googleapis.com/ailia-models/whisper/"
    work_dir = os.path.abspath(args.output_dir)
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    suffix = f"int{args.bits}"
    models = [
        ("encoder_turbo.opt.onnx", f"encoder_turbo_{suffix}.onnx",
         "encoder_turbo_weights.opt.pb"),
        ("decoder_turbo_fix_kv_cache.opt.onnx",
         f"decoder_turbo_fix_kv_cache_{suffix}.onnx", None),
    ]

    # Step 1: Download
    print("[1/3] Downloading original models ...")
    for orig, _, pb in models:
        download_model(orig, remote_path)
        if pb:
            download_model(pb, remote_path)

    # Step 2: Quantize
    print(f"[2/3] Quantizing models to {suffix} ...")
    output_files = []
    for orig, out, _ in models:
        print(f"\n--- {orig} ---")
        out_path = os.path.join(work_dir, out)
        quantize_model(orig, out_path, args.bits)
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
