"""
Convert SDXL fp32 ONNX models (UNet / text encoders) to int8 with MatMulNBits.

Tested versions:
    onnxruntime 1.24.4
    onnx 1.20.1

Requirements:
    pip install onnxruntime onnx numpy

Usage:
    python convert_to_int8.py --model all
    python convert_to_int8.py --model unet

This applies weight-only dynamic 8-bit quantization (same approach as
bevformer): MatMul nodes with a constant weight are replaced by MatMulNBits
(8-bit, block_size=128, symmetric). Gemm nodes are first converted to
MatMul+Add because MatMulNBitsQuantizer does not support Gemm. Activations
stay fp32, so no calibration dataset is needed.

Conv layers are NOT quantized: unlike bevformer-tiny, the SDXL UNet is
dominated by the transformer MatMuls (1002 MatMul vs 51 Conv), so
quantizing Conv adds little size reduction for extra quality risk.

The VAE is intentionally left fp32.
"""

import argparse
import os
import subprocess
import sys
import urllib.request

import onnx
from onnx import helper, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)
from onnxruntime.quantization.quant_utils import QuantFormat

MODELS = {
    "unet": "sdxl_unet",
    "clip_l": "sdxl_text_encoder_clip_l",
    "open_clip_bigg": "sdxl_text_encoder_open_clip_bigg",
    "refiner_unet": "sdxl_refiner_unet",
}


def clear_external_refs(model):
    """Drop external data references so the in-memory raw_data is used."""
    for tensor in model.graph.initializer:
        while len(tensor.external_data) > 0:
            tensor.external_data.pop()
        tensor.ClearField("data_location")


def convert_gemm_to_matmul(model):
    """Convert Gemm nodes to MatMul+Add with in-place weight transpose.

    MatMulNBitsQuantizer does not support Gemm, so we convert Gemm nodes to
    MatMul (with transposed weight initializers) + Add (for bias).
    """
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


def fold_transposed_weights(model):
    """Fold initializer -> Transpose -> MatMul patterns into the weight.

    generative-models 由来の UNet は Linear の重みを
    initializer -> Transpose -> MatMul の形で持っており、このままだと
    MatMulNBitsQuantizer が全て skip してしまう。Transpose を重み側へ
    畳み込み、MatMul の B 入力を 2D initializer に直結させる。
    """
    graph = model.graph
    init_map = {t.name: i for i, t in enumerate(graph.initializer)}
    producers = {o: n for n in graph.node for o in n.output}

    folded = 0
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        producer = producers.get(node.input[1])
        if producer is None or producer.op_type != "Transpose":
            continue
        if not producer.input or producer.input[0] not in init_map:
            continue
        perm = [1, 0]
        for attr in producer.attribute:
            if attr.name == "perm":
                perm = list(attr.ints)
        weight = graph.initializer[init_map[producer.input[0]]]
        if len(weight.dims) != 2 or perm != [1, 0]:
            continue

        new_name = producer.input[0] + "_transposed"
        if new_name not in init_map:
            arr = numpy_helper.to_array(weight).T.copy()
            graph.initializer.append(numpy_helper.from_array(arr, name=new_name))
            init_map[new_name] = len(graph.initializer) - 1
        node.input[1] = new_name
        folded += 1

    # 参照されなくなった Transpose ノードを削除する
    output_names = {o.name for o in graph.output}
    used = set()
    for node in graph.node:
        used.update(node.input)
    dead = [
        n for n in graph.node
        if n.op_type == "Transpose"
        and not any(o in used or o in output_names for o in n.output)
    ]
    for n in dead:
        graph.node.remove(n)

    print(f"  Folded {folded} Transpose weights into MatMul "
          f"(removed {len(dead)} Transpose nodes)")
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
    script_url = (
        "https://raw.githubusercontent.com/ailia-ai/export-to-onnx/"
        "master/onnx2prototxt.py"
    )
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "onnx2prototxt.py")
    if not os.path.exists(script_path):
        print("  Downloading onnx2prototxt.py ...")
        urllib.request.urlretrieve(script_url, script_path)

    print(f"  Generating prototxt for {onnx_path} ...")
    subprocess.check_call([sys.executable, script_path, onnx_path])


def convert(stem, model_dir):
    src = os.path.join(model_dir, stem + ".onnx")
    dst = os.path.join(model_dir, stem + "_int8.onnx")
    if not os.path.exists(src):
        print(f"  {src} not found, skipping. (run sdxl.py once to download)")
        return

    print(f"  Loading {src} ...")
    model = onnx.load(src)  # external data (weights.pb) is loaded too
    clear_external_refs(model)

    model = convert_gemm_to_matmul(model)
    model = fold_transposed_weights(model)
    model = remove_unused_initializers(model)

    print("  Quantizing MatMul to MatMulNBits (int8) ...")
    quant = MatMulNBitsQuantizer(
        model=model,
        block_size=128,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
        algo_config=DefaultWeightOnlyQuantConfig(
            bits=8,
            block_size=128,
            is_symmetric=True,
            accuracy_level=4,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=("MatMul",),
        ),
    )
    quant.process()
    result = quant.model.model

    nbits = sum(1 for n in result.graph.node if n.op_type == "MatMulNBits")
    matmul = sum(1 for n in result.graph.node if n.op_type == "MatMul")
    print(f"  MatMulNBits nodes: {nbits}")
    print(f"  MatMul nodes (activation x activation): {matmul}")

    remove_unused_initializers(result)

    print(f"  Saving {dst} ...")
    dst_pb = os.path.join(model_dir, stem + "_int8_weights.pb")
    for path in (dst, dst_pb):
        if os.path.exists(path):
            os.remove(path)  # 外部データは追記書き込みされるため先に消す
    onnx.save(
        result,
        dst,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=stem + "_int8_weights.pb",
        size_threshold=1024,
    )
    del model, quant, result

    generate_prototxt(dst)

    src_size = os.path.getsize(src.replace(".onnx", "_weights.pb"))
    dst_size = os.path.getsize(dst.replace(".onnx", "_weights.pb"))
    print(f"  {stem}: {src_size / 1024**2:.0f}MB -> {dst_size / 1024**2:.0f}MB")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize SDXL ONNX models to int8 (MatMulNBits)"
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=list(MODELS.keys()) + ["all"],
        help="model to convert",
    )
    parser.add_argument(
        "--model_dir",
        default="..",
        help="directory containing the fp32 ONNX models",
    )
    args = parser.parse_args()

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]
    for name in targets:
        print(f"[{name}]")
        convert(MODELS[name], args.model_dir)


if __name__ == "__main__":
    main()
