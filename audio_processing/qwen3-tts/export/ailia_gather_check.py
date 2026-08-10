"""Run the gather bug sample on onnxruntime and ailia and compare.

Takes the models as they are, so it needs only numpy, onnxruntime and ailia; use
ailia_gather_repro.py, which needs torch, to build them from scratch instead.
--download fetches them:

    python3 ailia_gather_check.py --download
    python3 ailia_gather_check.py --input_dir /path/to/models -e 1

Four models, in two pairs that differ only in where the input embedding comes
from. The _gather half reads it from a table inside the graph, indexed by an
input; the _input half takes the embedding as an input instead. Each model is
driven the way a decode loop drives one -- a seq=2 prefill, then seq=1 steps with
a growing KV cache -- and is compared against itself on the two runtimes, so the
two halves need not agree with each other.

    gather_rope    one attention layer with rotary embeddings   FAILS on ailia
    input_rope     the same, embedding passed in                ok
    gather_attn    the rope dropped                             ok
    gather_attn    the rope dropped, embedding passed in        ok

Expected output on ailia 1.6.1: every model agrees with onnxruntime to ~1e-08
except gather_rope, which agrees for two calls and is then off by ~1e-02. Both
halves of that model are needed for it: the gather has to feed the graph, and the
rotary embeddings have to be there.
"""

import argparse
import os
import urllib.request

import numpy as np
import onnxruntime

import ailia

MODELS = ["gather_rope", "input_rope", "gather_attn", "input_attn"]
HIDDEN, HEADS, HEAD_DIM, ROWS = 64, 8, 8, 64
STEPS = 8
TOLERANCE = 1e-3
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/qwen3-tts/gather_bug/"


def download(input_dir):
    os.makedirs(input_dir, exist_ok=True)
    for name in MODELS:
        for suffix in (".onnx", ".onnx.prototxt"):
            path = os.path.join(input_dir, name + suffix)
            if os.path.exists(path):
                continue
            print(f"downloading {name + suffix} ...")
            urllib.request.urlretrieve(REMOTE_PATH + name + suffix, path)


def causal_mask(seq_len, past_len=0):
    mask = np.full((seq_len, seq_len + past_len), -np.inf, dtype=np.float32)
    for i in range(seq_len):
        mask[i, : past_len + i + 1] = 0.0
    return mask[None, None]


def calls():
    """A seq=2 prefill and then seq=1 steps, different values every step.

    rows and embeds are both provided; a model only has one of them as an input
    and the other is ignored.
    """
    rng = np.random.default_rng(0)
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
        embeds = (rng.standard_normal((1, rows.shape[1], HIDDEN)) * 0.05).astype(np.float32)
        yield {"rows": rows, "embeds": embeds, "attention_mask": mask,
               "position_ids": position_ids}


def sweep_onnxruntime(path, names):
    session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    cache = [np.zeros((1, HEADS, 0, HEAD_DIM), np.float32) for _ in range(2)]
    outputs = []
    for named in calls():
        values = [named[name] for name in names if name in named] + cache
        out = session.run(None, dict(zip(names, values)))
        outputs.append(np.asarray(out[0]).copy())
        cache = [np.ascontiguousarray(out[1]), np.ascontiguousarray(out[2])]
    return outputs


def sweep_ailia(path, names, env_id):
    net = ailia.Net(stream=path + ".prototxt", weight=path, env_id=env_id)
    cache = [np.zeros((1, HEADS, 0, HEAD_DIM), np.float32) for _ in range(2)]
    outputs = []
    for named in calls():
        values = [named[name] for name in names if name in named] + cache
        for index, value in enumerate(values):
            net.set_input_blob_shape(value.shape, index)
        out = [np.array(x, copy=True) for x in net.run(values)]
        outputs.append(out[0])
        cache = [out[1], out[2]]
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input_dir", default=".", help="where the models are")
    parser.add_argument("--download", action="store_true",
                        help="fetch any model that is not in input_dir yet")
    parser.add_argument("-e", "--env_id", type=int, default=1)
    args = parser.parse_args()

    if args.download:
        download(args.input_dir)

    print(f"ailia {ailia.get_version()}  onnxruntime {onnxruntime.__version__}  "
          f"env_id {args.env_id}")
    missing, failed = [], []
    for name in MODELS:
        path = os.path.join(args.input_dir, name + ".onnx")
        if not os.path.exists(path):
            missing.append(name)
            continue
        # the exporter drops whichever of rows / embeds the model does not use, so
        # the inputs to feed come from the model itself
        names = [i.name for i in onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"]).get_inputs()]
        reference = sweep_onnxruntime(path, names)
        actual = sweep_ailia(path, names, args.env_id)
        diffs = [float(np.abs(a - o).max()) for a, o in zip(actual, reference)]
        first_bad = next((i for i, d in enumerate(diffs) if d > TOLERANCE), None)
        if first_bad is not None:
            failed.append(name)
        print(f"  {name:<12} {'MISMATCH from call ' + str(first_bad) if first_bad is not None else 'match':<22}"
              f" worst {max(diffs):.1e}")
        print("     " + "  ".join(f"call {i} {d:.1e}" for i, d in enumerate(diffs)))

    if missing:
        print("not found (pass --download): " + ", ".join(missing))
    print("\nailia disagrees with onnxruntime on: "
          + (", ".join(failed) if failed else "nothing"))
    if failed == ["gather_rope"]:
        print("this is the expected result on ailia 1.6.1")


if __name__ == "__main__":
    main()
