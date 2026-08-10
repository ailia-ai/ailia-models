"""Minimal reproduction: ailia returns stale gather results in a decode loop.

Two ONNX models under 100KB each, identical except for where the input embedding
comes from:

    gather.onnx   the embedding is gathered from a constant table inside the
                  graph, by a row index that comes in as an input
    input.onnx    the same embedding arrives as a graph input, gathered on the
                  Python side

Everything else matches: one attention layer with a rotary embedding, a KV cache
passed in and out, and a 4D additive mask. Both are driven the way a decode loop
drives a model, with a seq=2 prefill and then seq=1 steps, a growing cache and a
different table row every step.

Both models produce the same values, so the two runs should agree to rounding.
ailia 1.6.1 does on input.onnx and does not on gather.onnx, where the first two
calls match and every call after that is off by six orders of magnitude more:

    gather.onnx   call 0 0.0e+00  call 1 5.6e-09  call 2 1.3e-02  call 3 1.7e-02 ...
    input.onnx    call 0 0.0e+00  call 1 5.6e-09  call 2 1.0e-08  call 3 4.7e-09 ...

Both halves are needed to see it: with the rotary embedding dropped
(--stage attn) gather.onnx is correct too, and so is a graph that gathers without
a KV cache. In the real model, where the weights are trained rather than random,
the same shape of error comes out as O(1) on logits of O(10), which is enough to
change every sampled token.

This is why qwen3_tts_talker_*.onnx and qwen3_tts_code_predictor_*.onnx take
their codec embeddings as an input instead of holding the tables, and why
qwen3_tts_codec_embedding_*.onnx is a separate model: a graph that only gathers
is correct, however many times it is called.

Usage:
    python3 ailia_gather_repro.py
    python3 ailia_gather_repro.py --stage attn     # the variant that stays correct
"""

import argparse
import os
import subprocess
import sys
import urllib.request

import numpy as np
import onnxruntime
import torch
from torch import nn

import ailia

ROWS, HIDDEN, HEADS, HEAD_DIM = 64, 64, 8, 8
STEPS = 8
STAGES = ["attn", "rope"]
ONNX2PROTOTXT_URL = (
    "https://raw.githubusercontent.com/ailia-ai/export-to-onnx/master/onnx2prototxt.py"
)


def make_table():
    generator = torch.Generator().manual_seed(1234)
    return torch.randn(ROWS, HIDDEN, generator=generator) * 0.05


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Layer(nn.Module):
    def __init__(self, stage, gather):
        super().__init__()
        self.stage = stage
        self.gather = gather
        self.register_buffer("table", make_table(), persistent=False)
        self.register_buffer(
            "inv_freq",
            1.0 / (1000000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM)),
            persistent=False,
        )
        self.q_proj = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False)

    def forward(self, rows, embeds, attention_mask, position_ids, past_key, past_value):
        hidden_states = self.table[rows[0]][None] if self.gather else embeds

        shape = (1, -1, HEADS, HEAD_DIM)
        query = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)

        if self.stage == "rope":
            inv_freq = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
            freqs = (inv_freq @ position_ids[:, None, :].float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)
            query = query * cos + rotate_half(query) * sin
            key = key * cos + rotate_half(key) * sin

        key = torch.cat([past_key, key], dim=2)
        value = torch.cat([past_value, value], dim=2)
        scores = query @ key.transpose(-1, -2) / (HEAD_DIM ** 0.5) + attention_mask
        attention = (scores.softmax(dim=-1) @ value).transpose(1, 2)
        out = self.o_proj(attention.reshape(1, -1, HEADS * HEAD_DIM))
        return out[:, -1:, :], key, value


def causal_mask(seq_len, past_len=0):
    mask = np.full((seq_len, seq_len + past_len), -np.inf, dtype=np.float32)
    for i in range(seq_len):
        mask[i, : past_len + i + 1] = 0.0
    return mask[None, None]


def build(stage, gather, out_dir):
    path = os.path.join(out_dir, f"{'gather' if gather else 'input'}.onnx")
    torch.manual_seed(0)
    model = Layer(stage, gather).eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (torch.zeros(1, 2, dtype=torch.int64), torch.randn(1, 2, HIDDEN),
             torch.from_numpy(causal_mask(2)), torch.arange(2)[None],
             torch.zeros(1, HEADS, 0, HEAD_DIM), torch.zeros(1, HEADS, 0, HEAD_DIM)),
            path,
            input_names=["rows", "embeds", "attention_mask", "position_ids",
                         "past_key", "past_value"],
            output_names=["out", "present_key", "present_value"],
            dynamic_axes={
                "rows": {1: "seq"}, "embeds": {1: "seq"},
                "attention_mask": {2: "seq", 3: "total_len"},
                "position_ids": {1: "seq"},
                "past_key": {2: "past_len"}, "past_value": {2: "past_len"},
                "present_key": {2: "total_len"}, "present_value": {2: "total_len"},
            },
            opset_version=17,
            do_constant_folding=True,
            # the same exporter the real models use; the dynamo one builds a
            # different graph and does not show the problem
            dynamo=False,
        )

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx2prototxt.py")
    if not os.path.exists(script):
        urllib.request.urlretrieve(ONNX2PROTOTXT_URL, script)
    subprocess.check_call([sys.executable, script, path], stdout=subprocess.DEVNULL)
    return path


def calls(gather, table):
    """A seq=2 prefill and then seq=1 steps, a different row every step."""
    indices = [(i * 13 + 5) % ROWS for i in range(STEPS + 1)]
    for step in range(STEPS):
        if step == 0:
            rows = np.array([indices[:2]], np.int64)
            position_ids = np.array([[0, 1]], np.int64)
            mask = causal_mask(2)
        else:
            rows = np.array([[indices[step + 1]]], np.int64)
            position_ids = np.array([[step + 1]], np.int64)
            mask = causal_mask(1, step + 1)
        embeds = (np.zeros((1, rows.shape[1], HIDDEN), np.float32) if gather
                  else table[rows[0]][None].astype(np.float32))
        yield {"rows": rows, "embeds": embeds, "attention_mask": mask,
               "position_ids": position_ids}


def sweep_onnxruntime(path, gather, table):
    session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    cache = [np.zeros((1, HEADS, 0, HEAD_DIM), np.float32) for _ in range(2)]
    outputs = []
    for named in calls(gather, table):
        feed = dict(zip(names, [named[n] for n in names if n in named] + cache))
        out = session.run(None, feed)
        outputs.append(out[0].copy())
        cache = [np.ascontiguousarray(out[1]), np.ascontiguousarray(out[2])]
    return outputs


def sweep_ailia(path, gather, table, env_id):
    net = ailia.Net(stream=path + ".prototxt", weight=path, env_id=env_id)
    names = [i.name for i in onnxruntime.InferenceSession(
        path, providers=["CPUExecutionProvider"]).get_inputs()]
    cache = [np.zeros((1, HEADS, 0, HEAD_DIM), np.float32) for _ in range(2)]
    outputs = []
    for named in calls(gather, table):
        values = [named[n] for n in names if n in named] + cache
        for index, value in enumerate(values):
            net.set_input_blob_shape(value.shape, index)
        out = [np.array(x, copy=True) for x in net.run(values)]
        outputs.append(out[0])
        cache = [out[1], out[2]]
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", default="rope", choices=STAGES,
                        help="rope reproduces the bug, attn does not")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("-e", "--env_id", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    table = make_table().numpy()
    print(f"ailia {ailia.get_version()}  env_id {args.env_id}  stage {args.stage}")
    for gather in (True, False):
        path = build(args.stage, gather, args.output_dir)
        reference = sweep_onnxruntime(path, gather, table)
        actual = sweep_ailia(path, gather, table, args.env_id)
        diffs = [float(np.abs(a - o).max()) for a, o in zip(actual, reference)]
        first_bad = next((i for i, d in enumerate(diffs) if d > 1e-3), None)
        print(f"  {os.path.basename(path):<12} {os.path.getsize(path) / 1e3:6.1f} KB  "
              f"first wrong call {first_bad}")
        print("     " + "  ".join(f"call {i} {d:.1e}" for i, d in enumerate(diffs)))


if __name__ == "__main__":
    main()
