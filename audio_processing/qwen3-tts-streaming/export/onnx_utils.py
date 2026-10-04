"""How the exported models are written out, shared by the export and the fp16 pass.

Kept separate from export_onnx.py so that convert_to_fp16.py does not need torch or
qwen-tts installed to lay a converted model out the same way.
"""

import os
import subprocess
import sys
import urllib.request

import onnx

# A protobuf message cannot exceed 2GB, so a model whose weights come to more
# than this keeps them in a separate .onnx.data file and everything else stores
# them inline. The margin is for the graph itself.
INLINE_LIMIT = 1900 * 1024 * 1024

ONNX2PROTOTXT_URL = (
    "https://raw.githubusercontent.com/ailia-ai/export-to-onnx/master/onnx2prototxt.py"
)


def graph_tensors(graph):
    """Every tensor in a graph, including the ones held by node attributes.

    Weights folded into a Constant node live in an attribute rather than in
    graph.initializer, and the exporter writes those to their own external file
    too, so they have to be walked as well to find all of them.
    """
    for tensor in graph.initializer:
        yield tensor
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            for tensor in attribute.tensors:
                yield tensor
            if attribute.HasField("g"):
                yield from graph_tensors(attribute.g)
            for subgraph in attribute.graphs:
                yield from graph_tensors(subgraph)


def external_data_threshold(model):
    """Pick a size_threshold that leaves at least one initializer inline.

    ailia reads every weight as zero when all of a model's initializers live in
    the external data file, so the smallest one is always kept in the ONNX
    itself. Small tensors are kept inline anyway: onnx shape inference cannot
    read external data, and ops such as Slice need their operand values to infer
    shapes. onnx compares sys.getsizeof(raw_data) (the payload plus the bytes
    object overhead) against the threshold, so the same measure is used here.
    """
    initializers = model.graph.initializer
    sizes = [sys.getsizeof(t.raw_data) for t in initializers if t.HasField("raw_data")]
    if not sizes or len(sizes) < len(initializers):
        # an initializer without raw_data is never externalized, so one already
        # stays inline
        return 1024
    return max(1024, min(sizes) + 1)


def save_model(model, path):
    """Write a model out, inline if its weights fit and in one data file if not.

    Both torch exporters can leave weights in files of their own -- the dynamo one
    always does -- and one file per tensor is unwieldy to upload. A model that fits
    inline gets its weights back in the ONNX, so it needs no sidecar at all; the
    rest are rewritten into a single data file next to the model.
    """
    location = os.path.basename(path) + ".data"
    directory = os.path.dirname(path)
    weight_bytes = sum(
        len(tensor.raw_data) for tensor in graph_tensors(model.graph)
        if tensor.HasField("raw_data")
    )
    inline = weight_bytes <= INLINE_LIMIT
    print(f"  storing {weight_bytes / 1e9:.2f}GB of weights "
          + ("in the ONNX itself" if inline else f"in {location}") + " ...")

    # onnx appends to an existing data file, so any older one has to go first
    if os.path.exists(os.path.join(directory, location)):
        os.remove(os.path.join(directory, location))

    if inline:
        onnx.save_model(model, path)
        return
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=location,
        size_threshold=external_data_threshold(model),
        convert_attribute=False,
    )
    # onnx writes the data file with the process umask, make it world readable
    # like the model itself so it can be uploaded as is
    os.chmod(os.path.join(directory, location), 0o644)


def consolidate_external_data(path):
    """Rewrite a just exported model into the layout save_model() describes."""
    directory = os.path.dirname(path)
    model = onnx.load(path, load_external_data=False)
    externals = {
        entry.value
        for tensor in graph_tensors(model.graph)
        if tensor.data_location == onnx.TensorProto.EXTERNAL
        for entry in tensor.external_data
        if entry.key == "location"
    }

    model = onnx.load(path)
    weight_bytes = sum(
        len(tensor.raw_data) for tensor in graph_tensors(model.graph)
        if tensor.HasField("raw_data")
    )
    if weight_bytes <= INLINE_LIMIT and not externals:
        return

    # the weights are in memory now, so the files they came from can go
    for name in externals:
        os.remove(os.path.join(directory, name))
    save_model(model, path)


def generate_prototxt(onnx_path):
    """Generate the ailia prototxt from an ONNX model using onnx2prototxt.py."""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx2prototxt.py")
    if not os.path.exists(script_path):
        print("  downloading onnx2prototxt.py ...")
        urllib.request.urlretrieve(ONNX2PROTOTXT_URL, script_path)
    print(f"  generating {os.path.basename(onnx_path)}.prototxt ...")
    subprocess.check_call([sys.executable, script_path, onnx_path])
