"""Fold the fused code predictor's shape arithmetic so the CUDA EP owns every node.

    python simplify_frame.py --onnx_dir ../models

CUDA graph capture needs the whole graph on the CUDA EP, and onnxruntime keeps
shape computations (Shape/ConstantOfShape/Expand/Equal/Where chains that the
tracer emits for expand() and full_like()) on the CPU. Every shape in this graph is
static, so constant folding removes them. Tries onnxsim first and falls back to
onnxruntime's offline optimizer; then lists what is still placed on the CPU and
times the result with and without a CUDA graph.
"""
import argparse
import collections
import os
import re
import subprocess
import sys
import time

import numpy as np
import onnx
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onnxruntime as ort


def cpu_nodes(path):
    """Nodes the CUDA EP leaves on the CPU, from onnxruntime's verbose session log."""
    code = (
        "import onnxruntime as ort; ort.set_default_logger_severity(0); ort.set_default_logger_verbosity(1); "
        f"ort.InferenceSession(r'{path}', providers=['CUDAExecutionProvider','CPUExecutionProvider'])"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = r.stderr + r.stdout
    m = re.search(r"placed on \[CPUExecutionProvider\][^\n]*\n((?:.*\n){0,400})", log)
    if "All nodes placed on [CUDAExecutionProvider]" in log or "All nodes have been placed on [CUDAExecutionProvider]" in log:
        return []
    if not m:
        return ["(could not parse placement log; " + str(len(log)) + " chars)"]
    ops = re.findall(r"\(([A-Za-z]+)\)", m.group(1)) or re.findall(r"\b([A-Z][A-Za-z]+)\b", m.group(1))
    return ops


def time_fused(path, cuda_graph):
    po = {"cudnn_conv_algo_search": "HEURISTIC"}
    if cuda_graph:
        po["enable_cuda_graph"] = "1"
    s = ort.InferenceSession(path, providers=[("CUDAExecutionProvider", po), "CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    hidden = s.get_inputs()[0].shape[-1]                 # 1024 (0.6B) or 2048 (1.7B)
    group_vocab = s.get_inputs()[2].shape[-1]
    feed = [rng.standard_normal((1, 1, hidden)).astype(np.float32), np.array([123], np.int64),
            np.zeros((15, group_vocab), np.float32), np.array([1.0], np.float32)]
    b = s.io_binding()
    vals = []
    for i, arr in zip(s.get_inputs(), feed):
        v = ort.OrtValue.ortvalue_from_numpy(arr, "cuda", 0); vals.append(v); b.bind_ortvalue_input(i.name, v)
    for o in s.get_outputs():
        b.bind_output(o.name, "cuda", 0)
    for _ in range(3):
        s.run_with_iobinding(b)
    t = time.perf_counter()
    for _ in range(20):
        s.run_with_iobinding(b)
    return (time.perf_counter() - t) * 50, b.get_outputs()[0].numpy().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_dir", default="../models")
    ap.add_argument("-p", "--parameter_num", default="0.6B")
    ap.add_argument("-m", "--model", default="custom_voice", choices=["base", "custom_voice"])
    a = ap.parse_args()
    suffix = {"base": "", "custom_voice": "_custom_voice"}[a.model]
    src = os.path.join(a.onnx_dir, f"qwen3_tts_code_predictor_frame_{a.parameter_num}{suffix}.onnx")
    out = src.replace(".onnx", "_sim.onnx")

    print("before: CPU nodes:", collections.Counter(cpu_nodes(src)).most_common(12), flush=True)
    ok = False
    try:
        import onnxsim
        t = time.perf_counter()
        model, ok = onnxsim.simplify(onnx.load(src))
        print(f"onnxsim {time.perf_counter() - t:.0f}s ok={ok}", flush=True)
        if ok:
            onnx.save(model, out)
            # ailia reads the graph from a prototxt next to the weights
            import onnx2prototxt
            onnx2prototxt.onnx2prototxt(out)
    except Exception as e:  # noqa: BLE001
        print("onnxsim failed:", repr(e)[:200], flush=True)
    if not ok:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        so.optimized_model_filepath = out
        ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
        print("saved ORT offline-optimized model", flush=True)
    m = onnx.load(out, load_external_data=False)
    print("after ops:", collections.Counter(n.op_type for n in m.graph.node).most_common(14))
    print("after: CPU nodes:", collections.Counter(cpu_nodes(out)).most_common(12), flush=True)

    ms, toks = time_fused(out, False)
    print(f"[fused sim] io-bound {ms:.1f} ms/frame tokens={toks[:5]}...")
    try:
        ms, toks2 = time_fused(out, True)
        print(f"[fused sim] CUDA graph {ms:.1f} ms/frame tokens equal: {toks == toks2}")
    except Exception as e:  # noqa: BLE001
        print("CUDA graph failed:", str(e)[:300])


if __name__ == "__main__":
    main()
