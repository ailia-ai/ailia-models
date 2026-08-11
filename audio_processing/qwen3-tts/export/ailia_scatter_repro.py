"""Minimal reproduction: ailia writes the wrong slots of a fixed length KV cache.

A static shape decode loop keeps its KV cache in a buffer of a fixed length and
writes the step's keys and values at cache_position instead of appending them.
index_copy says that in one ScatterElements node, and ailia 1.6.1 gets it right
when a call writes one position and wrong when it writes more than one, which the
prompt of a decode loop does.

Two ONNX models of about 2KB, both taking (past, new, cache_position) and
returning past with new written at cache_position:

    scatter.onnx   past.index_copy(2, cache_position, new), one ScatterElements
    onehot.onnx    the same thing as a matmul: a [seq, cache_len] one hot of the
                   positions spreads new over the buffer and clears the slots it
                   overwrites

The expected output is exact -- it is a copy, not arithmetic -- so both should
come back bit for bit. onnxruntime does on both. ailia 1.6.1:

    scatter.onnx   seq=1 0.0e+00   seq=2 4.2e+00   seq=4 4.8e+00   seq=8 5.0e+00
    onehot.onnx    seq=1 0.0e+00   seq=2 0.0e+00   seq=4 0.0e+00   seq=8 0.0e+00

Nothing else about the two models differs, and neither has a KV cache, a rotary
embedding or an attention of its own: one ScatterElements on a
[1, 8, cache_len, 128] buffer is the whole graph.

In qwen3_tts_talker_<p>_static.onnx this came out as the prompt, which writes its
whole length in one call, leaving the cache wrong from the first call on: the
logits ailia returned were off by 47.9 against onnxruntime, which is enough to
change every sampled token. This is why export_onnx.py writes the cache with
cache_write() rather than index_copy.

Usage:
    python3 ailia_scatter_repro.py                 # build both models and compare
    python3 ailia_scatter_repro.py --onnx_dir DIR  # where to write them
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import onnxruntime
import torch
from torch import nn

import ailia

CACHE_LEN = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
SEQ_LENGTHS = [1, 2, 4, 8]
START = 5

MODELS = ["scatter", "onehot"]


class Scatter(nn.Module):
    """One ScatterElements, which is what index_copy exports to."""

    def forward(self, past, new, cache_position):
        return past.index_copy(2, cache_position, new)


class OneHot(nn.Module):
    """The same write as a matmul against a one hot of the positions."""

    def forward(self, past, new, cache_position):
        positions = torch.arange(past.shape[2])
        onehot = (positions[None, :] == cache_position[:, None]).to(past.dtype)
        spread = torch.einsum("bhsd,sm->bhmd", new, onehot)
        keep = (1.0 - onehot.sum(0))[None, None, :, None]
        return past * keep + spread


def build(onnx_dir):
    """Export both models and their prototxt."""
    past = torch.zeros(1, NUM_KV_HEADS, CACHE_LEN, HEAD_DIM)
    new = torch.zeros(1, NUM_KV_HEADS, 1, HEAD_DIM)
    cache_position = torch.tensor([START])

    for name, module in [("scatter", Scatter()), ("onehot", OneHot())]:
        path = os.path.join(onnx_dir, name + ".onnx")
        print(f"exporting {name}.onnx ...")
        torch.onnx.export(
            module,
            (past, new, cache_position),
            path,
            input_names=["past", "new", "cache_position"],
            output_names=["out"],
            dynamic_axes={"new": {2: "seq_len"}, "cache_position": {0: "seq_len"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx2prototxt.py")
        subprocess.check_call([sys.executable, script, path])


def compare(onnx_dir):
    """Run both models on both runtimes and print the error against the expected copy."""
    rng = np.random.default_rng(0)
    disagrees = []
    for name in MODELS:
        path = os.path.join(onnx_dir, name + ".onnx")
        session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        net = ailia.Net(path + ".prototxt", path, env_id=0)
        print(f"\n{name}.onnx")
        for seq_len in SEQ_LENGTHS:
            past = rng.standard_normal(
                (1, NUM_KV_HEADS, CACHE_LEN, HEAD_DIM)).astype(np.float32)
            new = rng.standard_normal(
                (1, NUM_KV_HEADS, seq_len, HEAD_DIM)).astype(np.float32)
            cache_position = np.arange(START, START + seq_len, dtype=np.int64)

            expected = past.copy()
            expected[:, :, START:START + seq_len, :] = new

            inputs = [past, new, cache_position]
            names = [i.name for i in session.get_inputs()]
            onnx_out = session.run(None, dict(zip(names, inputs)))[0]
            for index, value in enumerate(inputs):
                net.set_input_blob_shape(value.shape, index)
            ailia_out = net.run(inputs)[0]

            onnx_error = np.abs(onnx_out - expected).max()
            ailia_error = np.abs(ailia_out - expected).max()
            print(f"  seq={seq_len}  onnxruntime {onnx_error:.1e}   ailia {ailia_error:.1e}")
            if ailia_error > onnx_error and name not in disagrees:
                disagrees.append(name)
    print()
    if disagrees:
        print("ailia disagrees with onnxruntime on: " + ", ".join(disagrees))
    else:
        print("ailia agrees with onnxruntime on both models")
    return disagrees


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx_dir", default=".", help="where to write the models")
    args = parser.parse_args()

    os.makedirs(args.onnx_dir, exist_ok=True)
    build(args.onnx_dir)
    compare(args.onnx_dir)


if __name__ == "__main__":
    main()
