import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.external_data_helper import load_external_data_for_tensor, uses_external_data

"""
Split the combined SigLIP2 ONNX model (input_ids + pixel_values -> logits_per_image)
into a text encoder and an image encoder.

- encode_image: pixel_values -> image_embeds (L2 normalized)
- encode_text:  input_ids -> text_embeds (L2 normalized), logit_scale (exp applied), logit_bias

logits_per_image can be reconstructed as:
    image_embeds @ text_embeds.T * logit_scale + logit_bias
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model_type",
    default="base-patch16-224",
    choices=("base-patch16-224", "large-patch16-256", "giant-patch16-256"),
    help="model type",
)
parser.add_argument(
    "--input_dir",
    default="..",
    help="directory containing the combined onnx model",
)
parser.add_argument(
    "--output_dir",
    default="..",
    help="directory to save the separated onnx models",
)
args = parser.parse_args()

DIC_STEM = {
    "base-patch16-224": "siglip2-base-patch16-224",
    "large-patch16-256": "siglip2-large-patch16-256",
    "giant-patch16-256": "siglip2-giant-opt-patch16-256",
}


def reaches_input(graph, producer, tensor_name, input_name):
    """Check if tensor_name depends on the graph input input_name."""
    stack = [tensor_name]
    visited = set()
    while stack:
        name = stack.pop()
        if name in visited:
            continue
        visited.add(name)
        if name == input_name:
            return True
        node = producer.get(name)
        if node is not None:
            stack.extend(node.input)
    return False


def find_split_tensors(graph):
    """
    Walk backward from logits_per_image to locate the normalized embedding
    tensors and the logit_scale/logit_bias parameters.

    Expected tail of the graph:
        MatMul(text_embeds, Cast(Transpose(image_embeds)))
        -> Mul(., Exp(logit_scale)) -> Add(., logit_bias)
        -> Transpose -> logits_per_image
    """
    producer = {o: n for n in graph.node for o in n.output}
    initializers = {i.name: i for i in graph.initializer}

    node = producer["logits_per_image"]
    if node.op_type == "Transpose":
        node = producer[node.input[0]]

    assert node.op_type == "Add", f"unexpected op: {node.op_type}"
    add_inputs = list(node.input)
    mul_out = next(x for x in add_inputs if x in producer and producer[x].op_type == "Mul")
    logit_bias_name = next(x for x in add_inputs if x != mul_out)

    node = producer[mul_out]  # Mul
    matmul_out = next(
        x for x in node.input if x in producer and producer[x].op_type == "MatMul"
    )
    exp_out = next(x for x in node.input if x != matmul_out)
    logit_scale_name = producer[exp_out].input[0]  # Exp input

    matmul = producer[matmul_out]

    embeds = []
    for x in matmul.input:
        name = x
        # skip Cast/Transpose inserted before MatMul
        while name in producer and producer[name].op_type in ("Cast", "Transpose"):
            name = producer[name].input[0]
        embeds.append(name)

    text_embeds = next(
        x for x in embeds if reaches_input(graph, producer, x, "input_ids")
    )
    image_embeds = next(
        x for x in embeds if reaches_input(graph, producer, x, "pixel_values")
    )

    return image_embeds, text_embeds, logit_scale_name, logit_bias_name, initializers


def get_scalar(graph, initializers, base_dir, name):
    """Get a scalar parameter value from an initializer or a Constant node."""
    if name in initializers:
        tensor = initializers[name]
        if uses_external_data(tensor):
            load_external_data_for_tensor(tensor, base_dir)
        return onnx.numpy_helper.to_array(tensor).astype(np.float32).reshape(-1)
    for node in graph.node:
        if name in node.output and node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    return (
                        onnx.numpy_helper.to_array(attr.t)
                        .astype(np.float32)
                        .reshape(-1)
                    )
    raise ValueError(f"scalar parameter not found: {name}")


def extract_subgraph(model, base_dir, input_names, outputs, extra_initializers=()):
    """
    Extract the subgraph required to compute the given output tensors.

    outputs: list of (source tensor name, renamed output name)
    extra_initializers: list of (numpy array, output name) appended as
        Identity outputs (used for logit_scale / logit_bias)
    """
    graph = model.graph
    producer = {o: n for n in graph.node for o in n.output}
    initializers = {i.name: i for i in graph.initializer}
    graph_inputs = {i.name: i for i in graph.input}

    needed_nodes = set()
    needed_inits = []
    visited = set()
    stack = [name for name, _ in outputs]
    while stack:
        name = stack.pop()
        if name in visited or name in graph_inputs:
            continue
        visited.add(name)
        if name in initializers:
            needed_inits.append(initializers[name])
            continue
        node = producer.get(name)
        if node is None:
            continue
        if id(node) not in needed_nodes:
            needed_nodes.add(id(node))
            stack.extend(node.input)

    nodes = [n for n in graph.node if id(n) in needed_nodes]

    # load external data for required initializers only
    for tensor in needed_inits:
        if uses_external_data(tensor):
            load_external_data_for_tensor(tensor, base_dir)

    # rename outputs via Identity
    output_vi = []
    for tensor, new_name in outputs:
        nodes.append(helper.make_node("Identity", [tensor], [new_name]))
        output_vi.append(helper.make_tensor_value_info(new_name, TensorProto.FLOAT, None))

    # embed scalar parameters as outputs
    for array, new_name in extra_initializers:
        init_name = new_name + "_value"
        needed_inits.append(onnx.numpy_helper.from_array(array, init_name))
        nodes.append(helper.make_node("Identity", [init_name], [new_name]))
        output_vi.append(helper.make_tensor_value_info(new_name, TensorProto.FLOAT, None))

    new_graph = helper.make_graph(
        nodes,
        graph.name,
        [graph_inputs[name] for name in input_names],
        output_vi,
        initializer=needed_inits,
    )
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version

    return new_model


def save_model(model, output_path, use_external_data):
    if use_external_data:
        location = os.path.splitext(os.path.basename(output_path))[0] + "_weights.pb"
        pb_path = os.path.join(os.path.dirname(output_path), location)
        if os.path.exists(pb_path):
            os.remove(pb_path)
        onnx.save_model(
            model,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=location,
            size_threshold=1024,
        )
        print(f"saved: {output_path} (+ {location})")
    else:
        onnx.save_model(model, output_path)
        print(f"saved: {output_path}")


def main():
    stem = DIC_STEM[args.model_type]
    input_path = os.path.join(args.input_dir, stem + ".onnx")

    print(f"loading: {input_path}")
    model = onnx.load(input_path, load_external_data=False)
    base_dir = os.path.dirname(os.path.abspath(input_path))
    graph = model.graph

    (
        image_embeds,
        text_embeds,
        logit_scale_name,
        logit_bias_name,
        initializers,
    ) = find_split_tensors(graph)
    print(f"image_embeds: {image_embeds}")
    print(f"text_embeds: {text_embeds}")

    logit_scale = np.exp(get_scalar(graph, initializers, base_dir, logit_scale_name))
    logit_bias = get_scalar(graph, initializers, base_dir, logit_bias_name)
    print(f"logit_scale (exp applied): {logit_scale}")
    print(f"logit_bias: {logit_bias}")

    # follow the combined model: large/giant use external weights (.pb)
    use_external_data = any(uses_external_data(i) for i in graph.initializer)

    print("extracting image encoder...")
    image_model = extract_subgraph(
        model,
        base_dir,
        ["pixel_values"],
        [(image_embeds, "image_embeds")],
    )
    save_model(
        image_model,
        os.path.join(args.output_dir, stem + "-encode_image.onnx"),
        use_external_data,
    )

    # reload to drop external data loaded in memory for the image encoder
    model = onnx.load(input_path, load_external_data=False)

    print("extracting text encoder...")
    text_model = extract_subgraph(
        model,
        base_dir,
        ["input_ids"],
        [(text_embeds, "text_embeds")],
        extra_initializers=[(logit_scale, "logit_scale"), (logit_bias, "logit_bias")],
    )
    save_model(
        text_model,
        os.path.join(args.output_dir, stem + "-encode_text.onnx"),
        use_external_data,
    )

    print("done.")


if __name__ == "__main__":
    main()
