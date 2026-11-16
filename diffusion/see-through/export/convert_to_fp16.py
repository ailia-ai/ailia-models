"""
Convert the See-through fp32 ONNX models to fp16.

Tested versions:
    onnx : 1.22.0
    onnxruntime : 1.23.2

Usage:
    python convert_to_fp16.py --model all
    python convert_to_fp16.py --model layerdiff_unet

The fp32 models (*.onnx and *_weights.pb) are expected in the parent directory
(downloaded automatically by see_through.py on the first run). Each model is
written next to it as *_fp16.onnx, with a prototxt.

The converted models keep fp32 inputs/outputs (keep_io_types=True), so
see_through.py feeds the same fp32 arrays to both the fp32 and the fp16 models.
A model whose fp32 weights are in a separate *_weights.pb keeps them in
*_fp16_weights.pb, so the file layout stays the same.

The converter is onnxruntime's rather than onnxconverter-common's, which no
longer imports with onnx 1.22 (onnx.mapping was removed).
"""

import argparse
import os
import subprocess
import sys
import urllib.request

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnxruntime.transformers.float16 import (
    DEFAULT_OP_BLOCK_LIST,
    convert_float_to_float16,
)

MODELS = [
    "layerdiff_text_encoder",
    "layerdiff_text_encoder_2",
    "layerdiff_vae_encoder",
    "layerdiff_unet",
    "layerdiff_vae_decoder",
    "marigold_text_encoder",
    "marigold_vae_encoder",
    "marigold_unet",
    "marigold_vae_decoder",
]

# DEFAULT_OP_BLOCK_LIST of onnxruntime + Resize and Pow (as in sdxl/export).
# ailia expects the scales of Resize in fp32, and the converter would make them
# fp16 as Resize is not in the default list.
OP_BLOCK_LIST = DEFAULT_OP_BLOCK_LIST + ["Resize", "Pow"]


def consumers_of(graph):
    consumers = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    return consumers


def load_weights_as_fp16(model, model_dir):
    """Read the external weights, halving those the converter would halve anyway.

    The UNet has 16GB of fp32 weights. Loading all of them and then converting
    would need the fp32 and the fp16 copy in memory at once, so each tensor is
    read from the weight file on its own and halved right away. A weight that
    feeds an op of OP_BLOCK_LIST stays fp32, as the converter would leave it.
    """
    consumers = consumers_of(model.graph)
    halved = 0
    for tensor in model.graph.initializer:
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        info = {e.key: e.value for e in tensor.external_data}
        path = os.path.join(model_dir, info["location"])
        offset = int(info.get("offset", 0))
        length = int(info["length"])
        with open(path, "rb") as f:
            f.seek(offset)
            raw = f.read(length)
        del tensor.external_data[:]
        tensor.data_location = TensorProto.DEFAULT
        blocked = any(
            c.op_type in OP_BLOCK_LIST for c in consumers.get(tensor.name, [])
        )
        if tensor.data_type == TensorProto.FLOAT and not blocked:
            raw = np.frombuffer(raw, dtype=np.float32)
            raw = np.clip(raw, -65504, 65504).astype(np.float16).tobytes()
            tensor.data_type = TensorProto.FLOAT16
            halved += 1
        tensor.raw_data = raw
    print(f"  Halved {halved} external weights while loading")


def fold_weight_transposes(model):
    """Store the weights that go through a Transpose already transposed.

    The exporter wrote most MatMul weights of the UNets as
    initializer -> Transpose -> MatMul (6.9GB of the 7.6GB of the layerdiff
    UNet). The Transposes are applied here and removed from the graph, so the
    runtime does not have to transpose the weights itself.
    """
    graph = model.graph
    init = {t.name: t for t in graph.initializer}
    consumers = consumers_of(graph)
    output_names = {o.name for o in graph.output}

    folded = []
    for node in graph.node:
        if node.op_type != "Transpose" or node.input[0] not in init:
            continue
        if node.output[0] in output_names:
            continue
        src = init[node.input[0]]
        arr = numpy_helper.to_array(src)
        perm = next((list(a.ints) for a in node.attribute if a.name == "perm"), None)
        arr = np.ascontiguousarray(arr.transpose(perm))
        # the new initializer takes the name of the Transpose output, so the
        # consumers need no rewiring
        graph.initializer.append(numpy_helper.from_array(arr, node.output[0]))
        folded.append(node)

    # deleted by index, as remove() compares the messages, weights included
    folded_outputs = {node.output[0] for node in folded}
    for i in reversed(range(len(graph.node))):
        node = graph.node[i]
        if node.op_type == "Transpose" and node.output[0] in folded_outputs:
            del graph.node[i]
    # drop the original initializers no node reads any more
    used = {name for node in graph.node for name in node.input}
    used |= output_names
    for i in reversed(range(len(graph.initializer))):
        if graph.initializer[i].name not in used:
            del graph.initializer[i]
    # value_info of a Transpose output now names an initializer
    for i in reversed(range(len(graph.value_info))):
        if graph.value_info[i].name in folded_outputs:
            del graph.value_info[i]
    print(f"  Folded {len(folded)} weight Transposes")


def retarget_cast_nodes(model):
    """Rewrite the model's own Cast(to=float32) nodes to Cast(to=float16).

    The converter leaves the `to` attribute of the Casts already in the graph
    alone, so a Cast(to=float32) in the attention stays fp32 and the fp16 node
    after it gets mismatched inputs. The Casts the converter inserted itself, to
    a graph output or into an op of OP_BLOCK_LIST, have to stay fp32.
    """
    graph = model.graph
    output_names = {o.name for o in graph.output}
    consumers = consumers_of(graph)

    changed = 0
    for node in graph.node:
        if node.op_type != "Cast":
            continue
        to_attr = next(a for a in node.attribute if a.name == "to")
        if to_attr.i != TensorProto.FLOAT:
            continue
        out = node.output[0]
        if out in output_names:
            continue
        if any(c.op_type in OP_BLOCK_LIST for c in consumers.get(out, [])):
            continue
        to_attr.i = TensorProto.FLOAT16
        changed += 1
    print(f"  Retargeted {changed} Cast nodes to fp16")


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
    src = os.path.join(model_dir, stem + ".onnx")
    dst = os.path.join(model_dir, stem + "_fp16.onnx")
    if not os.path.exists(src):
        print(f"  {src} not found, skipping. (run see_through.py once to download)")
        return

    print(f"  Loading {src} ...")
    # shape inference runs before the weights are loaded, as a model over 2GB
    # cannot be serialized for it (the converter needs the types of the values
    # to insert the Casts around the blocked ops)
    model = onnx.load(src, load_external_data=False)
    external = any(
        t.data_location == TensorProto.EXTERNAL for t in model.graph.initializer
    )
    model = onnx.shape_inference.infer_shapes(model)
    load_weights_as_fp16(model, os.path.abspath(model_dir))
    fold_weight_transposes(model)

    print("  Converting to fp16 ...")
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,  # value_info is already filled in above
        op_block_list=OP_BLOCK_LIST,
    )
    del model
    retarget_cast_nodes(model_fp16)

    print(f"  Saving {dst} ...")
    dst_pb = os.path.join(model_dir, stem + "_fp16_weights.pb")
    for path in (dst, dst_pb):
        if os.path.exists(path):
            os.remove(path)  # external data is appended to, so remove it first
    if external:
        onnx.save(
            model_fp16,
            dst,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(dst_pb),
            size_threshold=1024,
        )
    else:
        onnx.save(model_fp16, dst)
    del model_fp16

    generate_prototxt(dst)

    def total_size(onnx_path, pb_path):
        size = os.path.getsize(onnx_path)
        if os.path.exists(pb_path):
            size += os.path.getsize(pb_path)
        return size

    src_size = total_size(src, os.path.join(model_dir, stem + "_weights.pb"))
    dst_size = total_size(dst, dst_pb)
    print(f"  {stem}: {src_size / 1024**2:.0f}MB -> {dst_size / 1024**2:.0f}MB")


def main():
    parser = argparse.ArgumentParser(description="Convert See-through ONNX models to fp16")
    parser.add_argument(
        "--model",
        default="all",
        choices=MODELS + ["all"],
        help="model to convert",
    )
    parser.add_argument(
        "--model_dir",
        default="..",
        help="directory containing the fp32 ONNX models",
    )
    args = parser.parse_args()

    targets = MODELS if args.model == "all" else [args.model]
    for name in targets:
        print(f"[{name}]")
        convert(name, args.model_dir)


if __name__ == "__main__":
    main()
