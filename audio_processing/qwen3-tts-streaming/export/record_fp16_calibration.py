"""Record real inputs of every model of the streaming sample for the fp16 conversion.

    python record_fp16_calibration.py -p 0.6B -m custom_voice --onnx_dir ../models

Runs ../qwen3-tts-streaming.py on onnxruntime (CUDA if available, the growing
cache talker so that the cache is visible) for a few sentences with the fp32
models (the sample itself loads the *_fp16 files; their names are rewritten here), catching the inputs of each model call, and writes
<onnx_dir>/calibration_<p>[_custom_voice].pkl: {model name: [[inputs], ...]}.
convert_to_fp16.py --calibration uses them to find the MatMul / Conv nodes
whose fp32 activations exceed the fp16 range (the Qwen3 residual stream has
channels around 5e4 and an MLP product around 1.6e5 in the code predictor),
which then stay fp32.
"""
import argparse
import importlib.util
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

TEXTS = [
    "ご注文の冷蔵庫の配送日は、明日午前10時頃になります。お手数をおかけしますが、よろしくお願いいたします。",
    "RTX 3080 の在庫はございますが、価格につきましてはお問い合わせください。",
]
MAX_CALLS = {"prompt": 20, "talker": 3, "code_predictor": 40, "decoder": 6, "encoder": 1}   # a talker call is 117 MB of cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--parameter_num", default="0.6B")
    ap.add_argument("-m", "--model", default="custom_voice", choices=["base", "custom_voice"])
    ap.add_argument("--onnx_dir", default="../models")
    a = ap.parse_args()

    onnx_dir = os.path.abspath(a.onnx_dir)
    os.chdir(os.path.join(HERE, ".."))
    sys.argv = ["qwen3-tts-streaming.py", "--onnx", "--quiet", "-p", a.parameter_num, "-m", a.model,
                "--cuda_graph", "none"]              # plain sessions: the talker's buffers can be read back
    spec = importlib.util.spec_from_file_location("sample", "qwen3-tts-streaming.py")
    sample = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sample)
    # the sample loads the fp16 models; the calibration needs the fp32 ones
    for key, value in list(vars(sample).items()):
        if key.startswith(("WEIGHT_PATH_", "MODEL_PATH_")) and isinstance(value, str):
            setattr(sample, key, value.replace("_fp16", ""))
    tts = sample.Qwen3TTSStreaming(None, sample.args.env_id)

    rec = {}

    def hook_run(net, name):
        orig = net.run

        def run(*inputs):
            if len(rec.setdefault(name, [])) < MAX_CALLS[name]:
                rec[name].append([i.copy() for i in inputs])
            return orig(*inputs)
        net.run = run

    def hook_step(net, name):
        """The fixed buffer talker: the graph inputs of a step are x, the mask,
        position_ids, cache_position and the 56 cache buffers (device resident,
        read back here), the same layout the fp32 calibration probe runs."""
        import numpy as np
        orig = net.step

        def step(x, position):
            if len(rec.setdefault(name, [])) < MAX_CALLS[name]:
                mask = np.zeros((1, 1, 1, net.buf), np.float32)
                mask[..., position + 1:] = -np.inf
                rec[name].append([np.ascontiguousarray(x, dtype=np.float32), mask,
                                  np.array([[position]], np.int64), np.array([position], np.int64)]
                                 + [c.numpy().copy() for c in net.cache])
            return orig(x, position)
        net.step = step

    hook_run(tts.prompt, "prompt")
    hook_run(tts.code_predictor, "code_predictor")
    hook_run(tts.decoder, "decoder")
    if tts.decoder_ramp is not None:
        hook_run(tts.decoder_ramp, "decoder")
    hook_step(tts.talker, "talker")
    for text in TEXTS:
        src = sample.new_source(tts)
        src.push(text)
        src.close()
        for _ in tts.stream(src, sample.args.speaker, sample.args.language):
            pass
    suffix = {"base": "", "custom_voice": "_custom_voice"}[a.model]
    out = os.path.join(onnx_dir, f"calibration_{a.parameter_num}{suffix}.pkl")
    with open(out, "wb") as f:
        pickle.dump(rec, f)
    for name, calls in rec.items():
        print(f"{name}: {len(calls)} calls, {sum(x.nbytes for c in calls for x in c) / 1e6:.0f} MB")
    print("wrote", out)


if __name__ == "__main__":
    main()
