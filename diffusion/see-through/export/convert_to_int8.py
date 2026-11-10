"""
Quantize the See-through fp16 ONNX models (UNets / text encoders) to int8 with MatMulNBits.

Tested versions:
    onnx : 1.22.0
    onnxruntime : 1.23.2

Usage:
    python convert_to_fp16.py --model all   # the int8 models are made from the fp16 ones
    python convert_to_int8.py --model all
    python convert_to_int8.py --model layerdiff_unet

This applies weight-only 8-bit quantization (as in sdxl/export): MatMul nodes
with a constant 2D weight are replaced by MatMulNBits (8-bit, block_size=128,
symmetric). Everything else, Conv included, stays as in the fp16 model, so no
calibration dataset is needed.

The source is the fp16 model rather than the fp32 one: the fp32 layerdiff UNet
has 16GB of weights, too much to quantize in memory, and the fp16 model already
has its weight Transposes folded, so the MatMuls read their weight directly
(MatMulNBitsQuantizer skips a weight behind a Transpose).

The VAEs are not quantized: they are dominated by Conv, and see_through.py uses
the fp16 VAEs with --precision int8.
"""

import argparse
import os
import subprocess
import sys
import urllib.request

import onnx
from onnx import TensorProto
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)
from onnxruntime.quantization.quant_utils import QuantFormat

MODELS = [
    "layerdiff_text_encoder",
    "layerdiff_text_encoder_2",
    "layerdiff_unet",
    "marigold_text_encoder",
    "marigold_unet",
]

# A protobuf message cannot exceed 2GB, so a larger model keeps its weights in
# a separate *_int8_weights.pb.
INLINE_LIMIT = 1900 * 1024 * 1024


def clear_external_refs(model):
    """Drop external data references so the in-memory raw_data is used."""
    for tensor in model.graph.initializer:
        del tensor.external_data[:]
        tensor.data_location = TensorProto.DEFAULT


def remove_unused_initializers(model):
    """Remove initializers not referenced by any node or graph output."""
    graph = model.graph
    used = {name for node in graph.node for name in node.input}
    used |= {o.name for o in graph.output}
    removed = 0
    # deleted by index, as remove() compares the messages, weights included
    for i in reversed(range(len(graph.initializer))):
        if graph.initializer[i].name not in used:
            del graph.initializer[i]
            removed += 1
    if removed:
        print(f"  Removed {removed} unused initializers")


def generate_prototxt(onnx_path):
    """Generate prototxt from ONNX model using onnx2prototxt.py."""
    script_url = (
        "https://raw.githubusercontent.com/ailia-ai/export-to-onnx/"
        "master/onnx2prototxt.py"
    )
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "onnx2prototxt.py"
    )
    if not os.path.exists(script_path):
        print("  Downloading onnx2prototxt.py ...")
        urllib.request.urlretrieve(script_url, script_path)

    print(f"  Generating prototxt for {onnx_path} ...")
    subprocess.check_call([sys.executable, script_path, onnx_path])


def convert(stem, model_dir):
    src = os.path.join(model_dir, stem + "_fp16.onnx")
    dst = os.path.join(model_dir, stem + "_int8.onnx")
    if not os.path.exists(src):
        print(f"  {src} not found, skipping. (run convert_to_fp16.py first)")
        return

    print(f"  Loading {src} ...")
    model = onnx.load(src)  # external data (_fp16_weights.pb) is loaded too
    clear_external_refs(model)

    print("  Quantizing MatMul to MatMulNBits (int8) ...")
    config = dict(
        block_size=128,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
    )
    quant = MatMulNBitsQuantizer(
        model=model,
        algo_config=DefaultWeightOnlyQuantConfig(bits=8, **config),
        **config,
    )
    quant.process()
    result = quant.model.model
    del model, quant

    nbits = sum(1 for n in result.graph.node if n.op_type == "MatMulNBits")
    matmul = sum(1 for n in result.graph.node if n.op_type == "MatMul")
    print(f"  MatMulNBits nodes: {nbits}")
    print(f"  MatMul nodes left (activation x activation): {matmul}")

    remove_unused_initializers(result)

    print(f"  Saving {dst} ...")
    dst_pb = os.path.join(model_dir, stem + "_int8_weights.pb")
    for path in (dst, dst_pb):
        if os.path.exists(path):
            os.remove(path)  # external data is appended to, so remove it first
    weight_bytes = sum(len(t.raw_data) for t in result.graph.initializer)
    if weight_bytes > INLINE_LIMIT:
        onnx.save(
            result,
            dst,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(dst_pb),
            size_threshold=1024,
        )
    else:
        onnx.save(result, dst)
    del result

    generate_prototxt(dst)

    def total_size(onnx_path, pb_path):
        size = os.path.getsize(onnx_path)
        if os.path.exists(pb_path):
            size += os.path.getsize(pb_path)
        return size

    src_size = total_size(src, os.path.join(model_dir, stem + "_fp16_weights.pb"))
    dst_size = total_size(dst, dst_pb)
    print(f"  {stem}: {src_size / 1024**2:.0f}MB (fp16) -> {dst_size / 1024**2:.0f}MB")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize See-through ONNX models to int8 (MatMulNBits)"
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=MODELS + ["all"],
        help="model to convert",
    )
    parser.add_argument(
        "--model_dir",
        default="..",
        help="directory containing the fp16 ONNX models",
    )
    args = parser.parse_args()

    targets = MODELS if args.model == "all" else [args.model]
    for name in targets:
        print(f"[{name}]")
        convert(name, args.model_dir)


if __name__ == "__main__":
    main()
