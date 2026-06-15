"""
Benchmark dynamic shape overhead: ailia vs ONNX Runtime.

Simulates autoregressive decoding where the KV cache grows by 1 token
each step (dynamic shape). Measures per-step latency breakdown:
  - shape setup / data copy overhead
  - actual compute (update/run)

Usage:
    python3 benchmark_dynamic.py                           # ailia fp32, CPU BLAS
    python3 benchmark_dynamic.py --model_type int4         # ailia int4
    python3 benchmark_dynamic.py --onnxruntime             # ONNX Runtime fp32
    python3 benchmark_dynamic.py --model_type int4 --onnxruntime
    python3 benchmark_dynamic.py --profile                 # ailia layer profile
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


def print_table(label, records, steps):
    """Print per-step timing table and summary."""
    print(f"\n{'='*70}")
    print(f" {label}")
    print(f"{'='*70}")

    has_breakdown = "setup_ms" in records[0]

    if has_breakdown:
        print(f"{'Step':>5} {'KV_len':>7} {'Setup_ms':>10} {'Compute_ms':>11} {'Total_ms':>10}")
        print(f"{'-'*5} {'-'*7} {'-'*10} {'-'*11} {'-'*10}")
        for r in records:
            print(f"{r['step']:5d} {r['kv_len']:7d} {r['setup_ms']:10.2f} "
                  f"{r['compute_ms']:11.2f} {r['total_ms']:10.2f}")
    else:
        print(f"{'Step':>5} {'KV_len':>7} {'Total_ms':>10}")
        print(f"{'-'*5} {'-'*7} {'-'*10}")
        for r in records:
            print(f"{r['step']:5d} {r['kv_len']:7d} {r['total_ms']:10.2f}")

    totals = [r["total_ms"] for r in records]
    avg = sum(totals) / len(totals)
    med = sorted(totals)[len(totals) // 2]
    print(f"\nSteps: {steps},  Avg: {avg:.2f} ms,  Median: {med:.2f} ms,  "
          f"Min: {min(totals):.2f} ms,  Max: {max(totals):.2f} ms")

    if has_breakdown:
        setups = [r["setup_ms"] for r in records]
        computes = [r["compute_ms"] for r in records]
        print(f"  Setup   avg: {sum(setups)/len(setups):.2f} ms")
        print(f"  Compute avg: {sum(computes)/len(computes):.2f} ms")


def benchmark_ailia_dynamic(args):
    import ailia

    if args.model_type == "int4":
        weight_path = "Qwen2-VL-2B_int4.onnx"
        model_path = "Qwen2-VL-2B_int4.onnx.prototxt"
    else:
        weight_path = "Qwen2-VL-2B.onnx"
        model_path = "Qwen2-VL-2B.onnx.prototxt"
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    memory_mode = ailia.get_memory_mode(
        reduce_constant=True,
        ignore_input_with_initializer=True,
        reduce_interstage=False,
        reuse_interstage=True,
    )
    net = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=memory_mode)

    if args.profile:
        net.set_profile_mode(ailia.PROFILE_AVERAGE)

    # --- Initial prefill ---
    init_tokens = args.init_tokens
    input_ids = np.ones((1, init_tokens), dtype=np.int64)
    inputs_embeds = np.zeros((0, init_tokens, HIDDEN_SIZE), dtype=np.float32)
    pos = np.arange(init_tokens, dtype=np.int64).reshape(1, 1, init_tokens)
    position_ids = np.concatenate([pos, pos, pos], axis=0)
    attention_mask = np.ones((1, init_tokens), dtype=np.int64)
    past_key_values = [
        np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32)
        for _ in range(NUM_KV * 2)
    ]

    print("Running initial prefill...")
    output = net.predict(
        [input_ids, inputs_embeds, position_ids, attention_mask, *past_key_values]
    )
    print(f"Prefill done. Initial KV cache length = {init_tokens}")

    # --- Warmup decode steps ---
    decode_input_ids = np.array([[1]], dtype=np.int64)
    decode_inputs_embeds = np.zeros((0, 1, HIDDEN_SIZE), dtype=np.float32)

    for w in range(args.warmup):
        key_shapes = []
        value_shapes = []
        for j in range(NUM_KV):
            key_shapes.append(net.get_blob_shape(net.find_blob_index_by_name(f"key_cache_out{j}")))
            value_shapes.append(net.get_blob_shape(net.find_blob_index_by_name(f"value_cache_out{j}")))

        cur_kv_len = key_shapes[0][2]
        cur_mask = np.ones((1, cur_kv_len + 1), dtype=np.int64)
        decode_pos = np.array([[[cur_kv_len]], [[cur_kv_len]], [[cur_kv_len]]], dtype=np.int64)

        net.set_input_blob_data(decode_input_ids, net.find_blob_index_by_name("input_ids"))
        net.set_input_blob_data(decode_inputs_embeds, net.find_blob_index_by_name("inputs_embeds"))
        net.set_input_blob_data(decode_pos, net.find_blob_index_by_name("position_ids"))
        net.set_input_blob_data(cur_mask, net.find_blob_index_by_name("attention_mask"))
        for j in range(NUM_KV):
            net.set_input_blob_shape(key_shapes[j], net.find_blob_index_by_name(f"key_cache{j}"))
            net.set_input_blob_shape(value_shapes[j], net.find_blob_index_by_name(f"value_cache{j}"))
            net.copy_blob_data(f"key_cache{j}", f"key_cache_out{j}")
            net.copy_blob_data(f"value_cache{j}", f"value_cache_out{j}")
        net.update()

    # --- Measured decode steps (KV cache grows each step = dynamic shape) ---
    records = []
    for step in range(args.steps):
        # ---- Phase 1: shape setup + data copy ----
        t0 = time.perf_counter()

        key_shapes = []
        value_shapes = []
        for j in range(NUM_KV):
            key_shapes.append(net.get_blob_shape(net.find_blob_index_by_name(f"key_cache_out{j}")))
            value_shapes.append(net.get_blob_shape(net.find_blob_index_by_name(f"value_cache_out{j}")))

        cur_kv_len = key_shapes[0][2]
        cur_mask = np.ones((1, cur_kv_len + 1), dtype=np.int64)
        decode_pos = np.array([[[cur_kv_len]], [[cur_kv_len]], [[cur_kv_len]]], dtype=np.int64)

        net.set_input_blob_data(decode_input_ids, net.find_blob_index_by_name("input_ids"))
        net.set_input_blob_data(decode_inputs_embeds, net.find_blob_index_by_name("inputs_embeds"))
        net.set_input_blob_data(decode_pos, net.find_blob_index_by_name("position_ids"))
        net.set_input_blob_data(cur_mask, net.find_blob_index_by_name("attention_mask"))
        for j in range(NUM_KV):
            net.set_input_blob_shape(key_shapes[j], net.find_blob_index_by_name(f"key_cache{j}"))
            net.set_input_blob_shape(value_shapes[j], net.find_blob_index_by_name(f"value_cache{j}"))
            net.copy_blob_data(f"key_cache{j}", f"key_cache_out{j}")
            net.copy_blob_data(f"value_cache{j}", f"value_cache_out{j}")

        t1 = time.perf_counter()

        # ---- Phase 2: compute ----
        net.update()

        t2 = time.perf_counter()

        setup_ms = (t1 - t0) * 1000
        compute_ms = (t2 - t1) * 1000
        total_ms = (t2 - t0) * 1000
        records.append({
            "step": step,
            "kv_len": cur_kv_len,
            "setup_ms": setup_ms,
            "compute_ms": compute_ms,
            "total_ms": total_ms,
        })

    print_table(f"ailia {args.model_type.upper()} Dynamic Decode (env_id={args.env_id})",
                records, args.steps)

    if args.profile:
        print(f"\n{'='*70}")
        print(" ailia Layer Profile (averaged over all steps)")
        print(f"{'='*70}")
        print(net.get_summary())


def benchmark_ort_dynamic(args):
    import onnxruntime as ort

    if args.model_type == "int4":
        weight_path = "Qwen2-VL-2B_int4.onnx"
        model_path = "Qwen2-VL-2B_int4.onnx.prototxt"
    else:
        weight_path = "Qwen2-VL-2B.onnx"
        model_path = "Qwen2-VL-2B.onnx.prototxt"
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(weight_path, sess_options,
                                   providers=["CPUExecutionProvider"])
    output_names = [o.name for o in session.get_outputs()]

    # --- Initial prefill ---
    init_tokens = args.init_tokens
    input_ids = np.ones((1, init_tokens), dtype=np.int64)
    inputs_embeds = np.zeros((0, init_tokens, HIDDEN_SIZE), dtype=np.float32)
    pos = np.arange(init_tokens, dtype=np.int64).reshape(1, 1, init_tokens)
    position_ids = np.concatenate([pos, pos, pos], axis=0)
    attention_mask = np.ones((1, init_tokens), dtype=np.int64)
    past_key_values = [
        np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32)
        for _ in range(NUM_KV * 2)
    ]

    def build_feed(ids, embeds, pos_ids, mask, kv):
        feed = {
            "input_ids": ids,
            "inputs_embeds": embeds,
            "position_ids": pos_ids,
            "attention_mask": mask,
        }
        for j in range(NUM_KV):
            feed[f"key_cache{j}"] = kv[j * 2]
            feed[f"value_cache{j}"] = kv[j * 2 + 1]
        return feed

    def extract_kv(outputs):
        kv = []
        for j in range(NUM_KV):
            kv.append(outputs[1 + j * 2])
            kv.append(outputs[1 + j * 2 + 1])
        return kv

    print("Running initial prefill...")
    feed = build_feed(input_ids, inputs_embeds, position_ids, attention_mask, past_key_values)
    outputs = session.run(output_names, feed)
    current_kv = extract_kv(outputs)
    print(f"Prefill done. Initial KV cache length = {init_tokens}")

    # --- Warmup decode steps ---
    decode_input_ids = np.array([[1]], dtype=np.int64)
    decode_inputs_embeds = np.zeros((0, 1, HIDDEN_SIZE), dtype=np.float32)

    for w in range(args.warmup):
        cur_kv_len = current_kv[0].shape[2]
        cur_mask = np.ones((1, cur_kv_len + 1), dtype=np.int64)
        decode_pos = np.array([[[cur_kv_len]], [[cur_kv_len]], [[cur_kv_len]]], dtype=np.int64)
        feed = build_feed(decode_input_ids, decode_inputs_embeds, decode_pos, cur_mask, current_kv)
        outputs = session.run(output_names, feed)
        current_kv = extract_kv(outputs)

    # --- Measured decode steps ---
    records = []
    for step in range(args.steps):
        cur_kv_len = current_kv[0].shape[2]
        cur_mask = np.ones((1, cur_kv_len + 1), dtype=np.int64)
        decode_pos = np.array([[[cur_kv_len]], [[cur_kv_len]], [[cur_kv_len]]], dtype=np.int64)

        t0 = time.perf_counter()

        feed = build_feed(decode_input_ids, decode_inputs_embeds, decode_pos, cur_mask, current_kv)

        t1 = time.perf_counter()

        outputs = session.run(output_names, feed)

        t2 = time.perf_counter()

        current_kv = extract_kv(outputs)

        setup_ms = (t1 - t0) * 1000
        compute_ms = (t2 - t1) * 1000
        total_ms = (t2 - t0) * 1000
        records.append({
            "step": step,
            "kv_len": cur_kv_len,
            "setup_ms": setup_ms,
            "compute_ms": compute_ms,
            "total_ms": total_ms,
        })

    print_table(f"ONNX Runtime {args.model_type.upper()} Dynamic Decode (CPU)",
                records, args.steps)


def benchmark_ailia_static(args):
    """Benchmark with fixed KV cache size (no shape change between steps)."""
    import ailia

    if args.model_type == "int4":
        weight_path = "Qwen2-VL-2B_int4.onnx"
        model_path = "Qwen2-VL-2B_int4.onnx.prototxt"
    else:
        weight_path = "Qwen2-VL-2B.onnx"
        model_path = "Qwen2-VL-2B.onnx.prototxt"
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    memory_mode = ailia.get_memory_mode(
        reduce_constant=True,
        ignore_input_with_initializer=True,
        reduce_interstage=False,
        reuse_interstage=True,
    )
    net = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=memory_mode)

    # Use a fixed seq_len for all steps (no dynamic shape change)
    seq_len = args.init_tokens + args.warmup + args.steps
    input_ids = np.array([[1]], dtype=np.int64)
    inputs_embeds = np.zeros((0, 1, HIDDEN_SIZE), dtype=np.float32)
    position_ids = np.array([[[seq_len]], [[seq_len]], [[seq_len]]], dtype=np.int64)
    attention_mask = np.ones((1, seq_len + 1), dtype=np.int64)
    past_key_values = [
        np.random.randn(1, NUM_HEADS, seq_len, HEAD_DIM).astype(np.float32)
        for _ in range(NUM_KV * 2)
    ]

    # First run
    all_inputs = [input_ids, inputs_embeds, position_ids, attention_mask, *past_key_values]
    net.predict(all_inputs)

    # Warmup with copy_blob_data (same shape every time)
    for w in range(args.warmup):
        key_shapes = []
        value_shapes = []
        for j in range(NUM_KV):
            key_shapes.append(net.get_blob_shape(net.find_blob_index_by_name(f"key_cache_out{j}")))
            value_shapes.append(net.get_blob_shape(net.find_blob_index_by_name(f"value_cache_out{j}")))
        net.set_input_blob_data(input_ids, net.find_blob_index_by_name("input_ids"))
        net.set_input_blob_data(inputs_embeds, net.find_blob_index_by_name("inputs_embeds"))
        net.set_input_blob_data(position_ids, net.find_blob_index_by_name("position_ids"))
        net.set_input_blob_data(attention_mask, net.find_blob_index_by_name("attention_mask"))
        for j in range(NUM_KV):
            net.set_input_blob_shape(key_shapes[j], net.find_blob_index_by_name(f"key_cache{j}"))
            net.set_input_blob_shape(value_shapes[j], net.find_blob_index_by_name(f"value_cache{j}"))
            net.copy_blob_data(f"key_cache{j}", f"key_cache_out{j}")
            net.copy_blob_data(f"value_cache{j}", f"value_cache_out{j}")
        net.update()

    # Measured steps (same shape = static)
    records = []
    for step in range(args.steps):
        t0 = time.perf_counter()

        key_shapes = []
        value_shapes = []
        for j in range(NUM_KV):
            key_shapes.append(net.get_blob_shape(net.find_blob_index_by_name(f"key_cache_out{j}")))
            value_shapes.append(net.get_blob_shape(net.find_blob_index_by_name(f"value_cache_out{j}")))
        net.set_input_blob_data(input_ids, net.find_blob_index_by_name("input_ids"))
        net.set_input_blob_data(inputs_embeds, net.find_blob_index_by_name("inputs_embeds"))
        net.set_input_blob_data(position_ids, net.find_blob_index_by_name("position_ids"))
        net.set_input_blob_data(attention_mask, net.find_blob_index_by_name("attention_mask"))
        for j in range(NUM_KV):
            net.set_input_blob_shape(key_shapes[j], net.find_blob_index_by_name(f"key_cache{j}"))
            net.set_input_blob_shape(value_shapes[j], net.find_blob_index_by_name(f"value_cache{j}"))
            net.copy_blob_data(f"key_cache{j}", f"key_cache_out{j}")
            net.copy_blob_data(f"value_cache{j}", f"value_cache_out{j}")

        t1 = time.perf_counter()
        net.update()
        t2 = time.perf_counter()

        setup_ms = (t1 - t0) * 1000
        compute_ms = (t2 - t1) * 1000
        total_ms = (t2 - t0) * 1000
        records.append({
            "step": step,
            "kv_len": seq_len,
            "setup_ms": setup_ms,
            "compute_ms": compute_ms,
            "total_ms": total_ms,
        })

    print_table(f"ailia {args.model_type.upper()} STATIC Decode (env_id={args.env_id}, fixed kv_len={seq_len})",
                records, args.steps)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark dynamic shape overhead: ailia vs ONNX Runtime")
    parser.add_argument("--model_type", type=str, default="fp32",
                        choices=["fp32", "int4"])
    parser.add_argument("--env_id", type=int, default=1,
                        help="ailia environment id (default: 1 = CPU-BLAS)")
    parser.add_argument("--onnxruntime", action="store_true",
                        help="Use ONNX Runtime instead of ailia")
    parser.add_argument("--static", action="store_true",
                        help="Also run static-shape baseline for ailia (no shape change)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable ailia layer-level profiling")
    parser.add_argument("--init_tokens", type=int, default=100,
                        help="Number of initial prefill tokens (sets initial KV cache size)")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of decode steps to measure")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup decode steps")
    args = parser.parse_args()

    if args.onnxruntime:
        benchmark_ort_dynamic(args)
    else:
        benchmark_ailia_dynamic(args)
        if args.static:
            print("\n")
            benchmark_ailia_static(args)


if __name__ == "__main__":
    main()
