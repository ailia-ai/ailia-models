"""Convert the exported Qwen3-TTS ONNX to fp16.

    onnxruntime 1.28.0
    onnx 1.22.0

Usage:
    python3 convert_to_fp16.py --parameter_num 0.6B --onnx_dir .
    python3 convert_to_fp16.py --parameter_num 0.6B --only talker

Writes qwen3_tts_<name>_<p>_fp16.onnx next to each fp32 model, with a prototxt, and
needs neither torch nor qwen-tts.

The fused code predictor (qwen3_tts_code_predictor_<p>.opt.onnx ->
qwen3_tts_code_predictor_<p>_fp16.opt.onnx) is the exception to the whole-graph
conversion below: its graph ends in 15 sampling blocks (lm_head, top-k, Gumbel
noise), and a full fp16 version of it overflows. It is converted the way the
streaming sample converts every model -- weights and MatMul / Gather in fp16 with
a Cast on the way in and out, everything else fp32 -- by importing
../../qwen3-tts-streaming/export/convert_to_fp16.py, so both samples ship the
same file for it.

The encoder is left in fp32 for a reason given at MODELS below. Two things stay in
fp32 in every model that is converted:

  the graph inputs and outputs
      keep_io_types, so ../qwen3-tts.py feeds and reads the same arrays whether it
      loaded the fp32 or the fp16 models, and nothing in it changes but the paths.

  the rotary embedding
      The angle is inv_freq * position_id, and the talker's positions run past
      2000, where fp16 has a spacing of 2.0. Rounding an angle in radians that
      coarsely would leave cos and sin unrelated to the position, so every node
      from the position ids down to the Cos and Sin is kept in fp32 and only what
      they multiply is halved. Only the two models with a position_ids input have
      such nodes; the decoder's Sin belongs to its snake activations, whose
      argument is an activation rather than a position, and is converted.
"""

import argparse
import collections
import os

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

from onnx_utils import generate_prototxt, save_model

# The converter is onnxruntime's rather than the onnxconverter-common one the other
# samples in this repository use: onnxconverter-common 1.16.0 crashes on these
# graphs as soon as a Cast feeds more than one node ("'list' object has no
# attribute 'input'" in remove_unnecessary_cast_node), and skipping that cleanup
# leaves a graph both runtimes reject for mixing fp16 and fp32 on one Add.
# onnxruntime ships a maintained fork of the same function with the same options.

# The encoder is not converted. Its audio_codes are codebook indices, and in fp16
# 29 of the 3232 the reference audio produces come out different, in codebooks 3
# and 5..15 where the residual being quantised is small enough for fp16 to flip a
# near tie. Those 16 codebooks are the voice prompt, so the reference audio would
# no longer be encoded the same way -- for 114MB of the 4.3GB set.
MODELS = ["decoder", "prompt", "codec_embedding", "talker", "code_predictor", "code_predictor_opt"]

STREAMING_EXPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "qwen3-tts-streaming", "export")


def convert_fused(path, out_path, parameter_num):
    """The fused code predictor, with the streaming sample's converter (see the module doc)."""
    import importlib.util
    import sys
    script = os.path.join(STREAMING_EXPORT, "convert_to_fp16.py")
    if not os.path.exists(script):
        raise FileNotFoundError(f"{script} is needed to convert the fused code predictor")
    sys.path.insert(0, STREAMING_EXPORT)
    spec = importlib.util.spec_from_file_location("streaming_convert_to_fp16", script)
    streaming = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(streaming)
    # no recorded calls here: the converter falls back to the activation ranges
    # it recorded on the shipped models (KNOWN_BLOCKS), which keep one MatMul of
    # the 0.6B predictor in fp32 compute
    streaming.convert(path, out_path, name="code_predictor_frame", parameter_num=parameter_num)


def rotary_nodes(graph):
    """The nodes that turn position_ids into cos and sin.

    Walking back from every Cos and Sin would also pick up the speech tokenizer
    decoder's snake activations, which are most of that graph and have nothing to
    do with positions, so a Cos or Sin only counts when its own ancestry reaches
    the position_ids input.
    """
    if not any(value.name == "position_ids" for value in graph.input):
        return []
    producer = {output: node for node in graph.node for output in node.output}
    blocked = set()
    for seed in graph.node:
        if seed.op_type not in ("Cos", "Sin"):
            continue
        seen, queue, from_positions = set(), collections.deque([seed]), False
        while queue:
            node = queue.popleft()
            if node.name in seen:
                continue
            seen.add(node.name)
            for name in node.input:
                if name == "position_ids":
                    from_positions = True
                elif name in producer:
                    queue.append(producer[name])
        if from_positions:
            blocked |= seen
    return sorted(blocked)


def convert(path, out_path):
    model = onnx.load(path)
    blocked = rotary_nodes(model.graph)
    print(f"  keeping {len(blocked)} rotary embedding nodes in fp32")
    converted = convert_float_to_float16(
        model,
        keep_io_types=True,
        node_block_list=blocked,
        disable_shape_infer=True,
    )
    save_model(converted, out_path)
    generate_prototxt(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--parameter_num", default="0.6B", choices=["0.6B", "1.7B"])
    parser.add_argument("--onnx_dir", default=".", help="where the fp32 models are")
    parser.add_argument("--output_dir", default=None, help="defaults to onnx_dir")
    parser.add_argument("--only", default=None, choices=MODELS)
    args = parser.parse_args()

    output_dir = args.output_dir or args.onnx_dir
    os.makedirs(output_dir, exist_ok=True)
    for name in [args.only] if args.only else MODELS:
        if name == "code_predictor_opt":
            stem = f"qwen3_tts_code_predictor_{args.parameter_num}"
            path = os.path.join(args.onnx_dir, stem + ".opt.onnx")
            out_path = os.path.join(output_dir, stem + "_fp16.opt.onnx")
            print(f"converting {stem}.opt.onnx ...")
            convert_fused(path, out_path, args.parameter_num)
        else:
            stem = f"qwen3_tts_{name}_{args.parameter_num}"
            path = os.path.join(args.onnx_dir, stem + ".onnx")
            out_path = os.path.join(output_dir, stem + "_fp16.onnx")
            print(f"converting {stem}.onnx ...")
            convert(path, out_path)
        print(f"  {os.path.getsize(path) / 1e6:8.1f} MB -> "
              f"{os.path.getsize(out_path) / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
