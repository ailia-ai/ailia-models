"""Export Qwen3-VL embed_tokens as a standalone ONNX model.

The language model ONNX takes inputs_embeds instead of input_ids so that
image embeddings from the vision encoder can be inserted at the image token
positions outside the graph. Qwen3-VL uses tie_word_embeddings=false, so
embed_tokens.weight cannot be recovered from lm_head either. This script
exports the token embedding lookup as a minimal ONNX graph (a single Gather):

    input_ids [batch, seq] (int64) -> inputs_embeds [batch, seq, hidden] (fp32)

The weight is taken from the original Hugging Face checkpoint by default.
If --npy is given (or the pre-saved .npy from the previous export exists
next to the sample), it is used instead to avoid downloading the checkpoint.

The 8b weight (151936 x 4096 fp32 = 2.5GB) exceeds the 2GB protobuf limit,
so it is stored as external data (*_weights.pb). The 4b weight (1.6GB) is
kept inside a single .onnx file.

Usage:
    python export_embed_tokens.py --model_type 4b
    python export_embed_tokens.py --model_type 8b
"""

import argparse
import json
import math
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MODELS = {
    "4b": {
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
        "hidden_size": 2560,
        "output": "qwen3_vl_4b_instruct_embed_tokens.onnx",
        "external_data": False,  # 151936 x 2560 fp32 = 1.6GB < 2GB
    },
    "8b": {
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
        "hidden_size": 4096,
        "output": "qwen3_vl_8b_instruct_embed_tokens.onnx",
        "external_data": True,  # 151936 x 4096 fp32 = 2.5GB > 2GB
    },
}

VOCAB_SIZE = 151936
OPSET_VERSION = 17


def load_weight_from_npy(npy_path):
    print(f"Loading embed_tokens.weight from {npy_path}")
    weight = np.load(npy_path)
    return np.ascontiguousarray(weight.astype(np.float32, copy=False))


def load_weight_from_hf(repo_id):
    """Download only the safetensors shard containing embed_tokens.weight."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    print(f"Loading embed_tokens.weight from {repo_id}")
    index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    key = next(k for k in weight_map if k.endswith("embed_tokens.weight"))
    shard_path = hf_hub_download(repo_id, weight_map[key])
    with safe_open(shard_path, framework="pt") as f:
        tensor = f.get_tensor(key)  # bf16
    return tensor.to(torch.float32).numpy()


# Keep every tensor below 2GB: protobuf cannot hold larger tensors in memory,
# and ailia cannot read a single external-data tensor over 2GB either.
MAX_TENSOR_BYTES = 2**31 - 2**20


def build_model(weight, external_location=None):
    hidden_size = weight.shape[1]

    if external_location is None:
        initializers = [numpy_helper.from_array(weight, name="embed_tokens.weight")]
        nodes = [
            helper.make_node(
                "Gather",
                inputs=["embed_tokens.weight", "input_ids"],
                outputs=["inputs_embeds"],
                axis=0,
                name="/embed_tokens/Gather",
            )
        ]
    else:
        # The 8b weight (2.5GB) exceeds the per-tensor limit, so split it
        # along the hidden axis into <2GB chunks stored as external data,
        # gather each chunk and concatenate the (small) gathered outputs.
        bytes_per_col = weight.shape[0] * weight.itemsize
        cols_per_chunk = MAX_TENSOR_BYTES // bytes_per_col
        num_chunks = math.ceil(hidden_size / cols_per_chunk)

        initializers = []
        nodes = []
        gathered = []
        offset = 0
        with open(external_location, "wb") as f:
            for i in range(num_chunks):
                chunk = np.ascontiguousarray(
                    weight[:, i * cols_per_chunk : (i + 1) * cols_per_chunk]
                )
                raw = chunk.tobytes()
                f.write(raw)

                tensor = TensorProto()
                tensor.name = f"embed_tokens.weight.{i}"
                tensor.data_type = TensorProto.FLOAT
                tensor.dims.extend(chunk.shape)
                tensor.data_location = TensorProto.EXTERNAL
                for key, value in [
                    ("location", os.path.basename(external_location)),
                    ("offset", str(offset)),
                    ("length", str(len(raw))),
                ]:
                    entry = tensor.external_data.add()
                    entry.key = key
                    entry.value = value
                initializers.append(tensor)
                offset += len(raw)

                nodes.append(
                    helper.make_node(
                        "Gather",
                        inputs=[tensor.name, "input_ids"],
                        outputs=[f"inputs_embeds.{i}"],
                        axis=0,
                        name=f"/embed_tokens/Gather_{i}",
                    )
                )
                gathered.append(f"inputs_embeds.{i}")

        nodes.append(
            helper.make_node(
                "Concat",
                inputs=gathered,
                outputs=["inputs_embeds"],
                axis=2,
                name="/embed_tokens/Concat",
            )
        )

    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT64, ["batch", "seq"]
    )
    inputs_embeds = helper.make_tensor_value_info(
        "inputs_embeds", TensorProto.FLOAT, ["batch", "seq", hidden_size]
    )

    graph = helper.make_graph(
        nodes,
        "qwen3_vl_embed_tokens",
        [input_ids],
        [inputs_embeds],
        initializer=initializers,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET_VERSION)]
    )


def verify(onnx_path, weight):
    import onnxruntime

    sess = onnxruntime.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    input_ids = np.array([[0, 1, 12345, VOCAB_SIZE - 1]], dtype=np.int64)
    out = sess.run(["inputs_embeds"], {"input_ids": input_ids})[0]
    expected = weight[input_ids[0]][None, :, :]
    assert out.shape == (1, 4, weight.shape[1]), out.shape
    assert np.array_equal(out, expected)
    print(f"Verified: {onnx_path} matches numpy lookup")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_type",
        default="4b",
        choices=list(MODELS.keys()),
        help="Qwen3-VL model type: 4b or 8b (default: 4b)",
    )
    parser.add_argument(
        "--npy",
        default=None,
        help="path to a pre-saved embed_tokens .npy "
        "(default: use ../<output>.npy if it exists, otherwise download from HF)",
    )
    args = parser.parse_args()

    config = MODELS[args.model_type]

    npy_path = args.npy
    if npy_path is None:
        default_npy = os.path.join(
            os.path.dirname(__file__), "..", config["output"].replace(".onnx", ".npy")
        )
        if os.path.exists(default_npy):
            npy_path = default_npy

    if npy_path is not None:
        weight = load_weight_from_npy(npy_path)
    else:
        weight = load_weight_from_hf(config["repo_id"])

    assert weight.shape == (VOCAB_SIZE, config["hidden_size"]), weight.shape

    output_path = config["output"]
    external_location = (
        output_path.replace(".onnx", "_weights.pb")
        if config["external_data"]
        else None
    )
    model = build_model(weight, external_location)
    onnx.save_model(model, output_path)
    print(f"Saved: {output_path}")

    if not config["external_data"]:
        # onnx.checker cannot handle external data files over 2GB; the
        # onnxruntime run in verify() covers validation for the 8b model.
        onnx.checker.check_model(output_path)
    verify(output_path, weight)


if __name__ == "__main__":
    main()
