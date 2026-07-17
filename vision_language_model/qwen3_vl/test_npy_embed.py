"""Standalone verification of the qwen3_vl embed_tokens backends.

Compares the three embedding lookup paths used by qwen3_vl.py against each
other on the same input_ids:

    npy   : numpy fancy indexing on qwen3_vl_*_embed_tokens.npy (reference)
    ort   : qwen3_vl_*_embed_tokens.onnx via onnxruntime
    ailia : qwen3_vl_*_embed_tokens.onnx via ailia

Missing model files are downloaded automatically.

Usage:
    python test_npy_embed.py --model_type 4b
    python test_npy_embed.py --model_type 8b
"""

import argparse
import sys

import numpy as np

sys.path.append("../../util")
from model_utils import check_and_download_file, check_and_download_models

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/qwen3_vl/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_type",
        default="4b",
        choices=["4b", "8b"],
        help="Qwen3-VL model type: 4b or 8b (default: 4b)",
    )
    args = parser.parse_args()

    npy_path = f"qwen3_vl_{args.model_type}_instruct_embed_tokens.npy"
    onnx_path = f"qwen3_vl_{args.model_type}_instruct_embed_tokens.onnx"

    check_and_download_file(npy_path, REMOTE_PATH)
    check_and_download_models(onnx_path, onnx_path + ".prototxt", REMOTE_PATH)
    if args.model_type == "8b":
        check_and_download_file(
            f"qwen3_vl_{args.model_type}_instruct_embed_tokens_weights.pb", REMOTE_PATH
        )

    table = np.load(npy_path, mmap_mode="r")
    vocab_size, hidden_size = table.shape
    print(f"embed_tokens: vocab={vocab_size} hidden={hidden_size}")

    # edge cases (first/last token) + random ids, prefill-like and decode-like
    rng = np.random.default_rng(0)
    test_inputs = [
        np.array([[0, 1, 12345, vocab_size - 1]], dtype=np.int64),
        rng.integers(0, vocab_size, size=(1, 97), dtype=np.int64),
        np.array([[151645]], dtype=np.int64),  # single token (decode step)
    ]

    def reference(input_ids):
        return np.asarray(table[input_ids])

    import onnxruntime

    sess = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def run_ort(input_ids):
        return sess.run(["inputs_embeds"], {"input_ids": input_ids})[0]

    import ailia

    net = ailia.Net(onnx_path + ".prototxt", onnx_path)

    def run_ailia(input_ids):
        return net.predict([input_ids])[0]

    ok = True
    for input_ids in test_inputs:
        expected = reference(input_ids)
        for name, runner in [("ort", run_ort), ("ailia", run_ailia)]:
            out = runner(input_ids)
            match = np.array_equal(out, expected)
            max_diff = float(np.abs(out - expected).max())
            print(
                f"input {str(input_ids.shape):10s} {name:6s} "
                f"{'OK' if match else 'NG'}  max_diff={max_diff:g}"
            )
            ok &= match

    print("PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
