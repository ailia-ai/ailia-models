"""
Benchmark Qwen2-VL decoder for FP32 vs INT4 comparison.
Usage:
    python3 benchmark_decoder.py --model_type int4                       # M=1 decode (GEMV)
    python3 benchmark_decoder.py --model_type int4 --prefill             # prefill (GEMM)
    python3 benchmark_decoder.py --model_type int4 --prefill --tokens 64 # prefill with 64 tokens
    python3 benchmark_decoder.py --model_type int4 --onnxruntime
    python3 benchmark_decoder.py --model_type int4 --onnxruntime --prefill
"""
import sys
import time
import argparse

import numpy as np

sys.path.append("../../util")
from model_utils import check_and_download_models  # noqa


REMOTE_PATH = "https://storage.googleapis.com/ailia-models/qwen2_vl/"
NUM_KV = 28
HIDDEN_SIZE = 1536
NUM_HEADS = 2
HEAD_DIM = 128
WARMUP = 3
ITERATIONS = 20


def print_results(label, times, seq_len, tokens=1):
    avg = sum(times) / len(times)
    median = sorted(times)[len(times) // 2]
    min_t = min(times)
    max_t = max(times)

    mode = "prefill (GEMM)" if tokens > 1 else "decode (GEMV)"
    print(f"\n=== Qwen2-VL Decoder Benchmark ({label}) ===")
    print(f"Mode:    {mode}, M={tokens}")
    print(f"Seq length: {seq_len}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Average: {avg:.2f} ms")
    print(f"Median:  {median:.2f} ms")
    print(f"Min:     {min_t:.2f} ms")
    print(f"Max:     {max_t:.2f} ms")
    print(f"All times: {[f'{t:.2f}' for t in times]}")


def benchmark_ailia(args, weight_path, model_path, input_ids, inputs_embeds,
                    position_ids, attention_mask, past_key_values, seq_len):
    import ailia

    memory_mode = ailia.get_memory_mode(
        reduce_constant=True,
        ignore_input_with_initializer=True,
        reduce_interstage=False,
        reuse_interstage=True,
    )
    net = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=memory_mode)

    # First run to initialize
    all_inputs = [input_ids, inputs_embeds, position_ids, attention_mask, *past_key_values]
    output = net.predict(all_inputs)

    is_prefill = input_ids.shape[1] > 1

    if is_prefill:
        # Prefill mode: use predict() each time (no KV cache reuse)
        for i in range(WARMUP):
            net.predict(all_inputs)

        times = []
        for i in range(ITERATIONS):
            start = time.perf_counter()
            net.predict(all_inputs)
            end = time.perf_counter()
            times.append((end - start) * 1000)
    else:
        # Decode mode: use copy_blob_data for KV cache
        def run_decode_step():
            key_shapes = []
            value_shapes = []
            for j in range(NUM_KV):
                key_shapes.append(net.get_blob_shape(net.find_blob_index_by_name(f"key_cache_out{j}")))
                value_shapes.append(net.get_blob_shape(net.find_blob_index_by_name(f"value_cache_out{j}")))
            net.set_input_blob_data(input_ids, net.find_blob_index_by_name("input_ids"))
            net.set_input_blob_data(inputs_embeds, net.find_blob_index_by_name("inputs_embeds"))
            net.set_input_blob_data(position_ids, net.find_blob_index_by_name("position_ids"))
            cur_kv_len = key_shapes[0][2]
            cur_mask = np.ones((1, cur_kv_len + 1), dtype=np.int64)
            net.set_input_blob_data(cur_mask, net.find_blob_index_by_name("attention_mask"))
            for j in range(NUM_KV):
                net.set_input_blob_shape(key_shapes[j], net.find_blob_index_by_name(f"key_cache{j}"))
                net.set_input_blob_shape(value_shapes[j], net.find_blob_index_by_name(f"value_cache{j}"))
                net.copy_blob_data(f"key_cache{j}", f"key_cache_out{j}")
                net.copy_blob_data(f"value_cache{j}", f"value_cache_out{j}")
            net.update()

        for i in range(WARMUP):
            run_decode_step()

        times = []
        for i in range(ITERATIONS):
            key_shapes = []
            value_shapes = []
            for j in range(NUM_KV):
                key_shapes.append(net.get_blob_shape(net.find_blob_index_by_name(f"key_cache_out{j}")))
                value_shapes.append(net.get_blob_shape(net.find_blob_index_by_name(f"value_cache_out{j}")))
            net.set_input_blob_data(input_ids, net.find_blob_index_by_name("input_ids"))
            net.set_input_blob_data(inputs_embeds, net.find_blob_index_by_name("inputs_embeds"))
            net.set_input_blob_data(position_ids, net.find_blob_index_by_name("position_ids"))
            cur_kv_len = key_shapes[0][2]
            cur_mask = np.ones((1, cur_kv_len + 1), dtype=np.int64)
            net.set_input_blob_data(cur_mask, net.find_blob_index_by_name("attention_mask"))
            for j in range(NUM_KV):
                net.set_input_blob_shape(key_shapes[j], net.find_blob_index_by_name(f"key_cache{j}"))
                net.set_input_blob_shape(value_shapes[j], net.find_blob_index_by_name(f"value_cache{j}"))
                net.copy_blob_data(f"key_cache{j}", f"key_cache_out{j}")
                net.copy_blob_data(f"value_cache{j}", f"value_cache_out{j}")

            start = time.perf_counter()
            net.update()
            end = time.perf_counter()
            times.append((end - start) * 1000)

    print_results(f"ailia {args.model_type.upper()}", times, seq_len, tokens=input_ids.shape[1])


def benchmark_onnxruntime(args, weight_path, input_ids, inputs_embeds,
                          position_ids, attention_mask, past_key_values, seq_len):
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(weight_path, sess_options)

    # Build feed dict
    def build_feed(kv_cache):
        feed = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        for j in range(NUM_KV):
            feed[f"key_cache{j}"] = kv_cache[j * 2]
            feed[f"value_cache{j}"] = kv_cache[j * 2 + 1]
        return feed

    output_names = [o.name for o in session.get_outputs()]

    # First run to initialize
    feed = build_feed(past_key_values)
    outputs = session.run(output_names, feed)

    def extract_kv_from_outputs(outputs):
        """Extract KV cache from outputs for next iteration."""
        kv = []
        for j in range(NUM_KV):
            kv.append(outputs[1 + j * 2])      # key_cache_out{j}
            kv.append(outputs[1 + j * 2 + 1])  # value_cache_out{j}
        return kv

    is_prefill = input_ids.shape[1] > 1

    if is_prefill:
        # Prefill mode: use same inputs each time (no KV cache reuse)
        feed = build_feed(past_key_values)
        for i in range(WARMUP):
            session.run(output_names, feed)

        times = []
        for i in range(ITERATIONS):
            feed = build_feed(past_key_values)
            start = time.perf_counter()
            session.run(output_names, feed)
            end = time.perf_counter()
            times.append((end - start) * 1000)
    else:
        # Decode mode: use output KV cache for subsequent runs
        current_kv = extract_kv_from_outputs(outputs)

        for i in range(WARMUP):
            cur_kv_len = current_kv[0].shape[2]
            cur_mask = np.ones((1, cur_kv_len + 1), dtype=np.int64)
            feed = {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
                "attention_mask": cur_mask,
            }
            for j in range(NUM_KV):
                feed[f"key_cache{j}"] = current_kv[j * 2]
                feed[f"value_cache{j}"] = current_kv[j * 2 + 1]
            outputs = session.run(output_names, feed)
            current_kv = extract_kv_from_outputs(outputs)

        times = []
        for i in range(ITERATIONS):
            cur_kv_len = current_kv[0].shape[2]
            cur_mask = np.ones((1, cur_kv_len + 1), dtype=np.int64)
            feed = {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
                "attention_mask": cur_mask,
            }
            for j in range(NUM_KV):
                feed[f"key_cache{j}"] = current_kv[j * 2]
                feed[f"value_cache{j}"] = current_kv[j * 2 + 1]

            start = time.perf_counter()
            outputs = session.run(output_names, feed)
            end = time.perf_counter()
            times.append((end - start) * 1000)

            current_kv = extract_kv_from_outputs(outputs)

    print_results(f"ONNX Runtime {args.model_type.upper()}", times, seq_len, tokens=input_ids.shape[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="fp32", choices=["fp32", "int4"])
    parser.add_argument("--env_id", type=int, default=-1)
    parser.add_argument("--seq_len", type=int, default=100,
                        help="Simulated past sequence length for KV cache")
    parser.add_argument("--onnxruntime", action="store_true",
                        help="Use ONNX Runtime instead of ailia")
    parser.add_argument("--prefill", action="store_true",
                        help="Benchmark prefill (M>1, GEMM) instead of decode (M=1, GEMV)")
    parser.add_argument("--tokens", type=int, default=908,
                        help="Number of input tokens for prefill mode (default: 908, "
                             "based on demo.jpeg 683x1024 -> grid 48x74 -> 888 image + ~20 text tokens)")
    args = parser.parse_args()

    if args.model_type == "int4":
        weight_path = "Qwen2-VL-2B_int4.onnx"
        model_path = "Qwen2-VL-2B_int4.onnx.prototxt"
    else:
        weight_path = "Qwen2-VL-2B.onnx"
        model_path = "Qwen2-VL-2B.onnx.prototxt"

    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    seq_len = args.seq_len

    if args.prefill:
        # Prefill mode: M tokens as input, no KV cache (frame 0)
        num_tokens = args.tokens
        input_ids = np.ones((1, num_tokens), dtype=np.int64)
        inputs_embeds = np.zeros((0, num_tokens, HIDDEN_SIZE), dtype=np.float32)
        # position_ids: (3, 1, num_tokens)
        pos = np.arange(num_tokens, dtype=np.int64).reshape(1, 1, num_tokens)
        position_ids = np.concatenate([pos, pos, pos], axis=0)  # (3, 1, num_tokens)
        attention_mask = np.ones((1, num_tokens), dtype=np.int64)
        # Empty KV cache
        past_key_values = [
            np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32)
            for _ in range(NUM_KV * 2)
        ]
    else:
        # Decode mode: single token, existing KV cache (M=1, GEMV)
        input_ids = np.array([[1]], dtype=np.int64)
        inputs_embeds = np.zeros((0, 1, HIDDEN_SIZE), dtype=np.float32)
        position_ids = np.array([[[seq_len]], [[seq_len]], [[seq_len]]], dtype=np.int64)
        attention_mask = np.ones((1, seq_len + 1), dtype=np.int64)
        past_key_values = [
            np.random.randn(1, NUM_HEADS, seq_len, HEAD_DIM).astype(np.float32)
            for _ in range(NUM_KV * 2)
        ]

    if args.onnxruntime:
        benchmark_onnxruntime(args, weight_path, input_ids, inputs_embeds,
                              position_ids, attention_mask, past_key_values, seq_len)
    else:
        benchmark_ailia(args, weight_path, model_path, input_ids, inputs_embeds,
                        position_ids, attention_mask, past_key_values, seq_len)


if __name__ == "__main__":
    main()
