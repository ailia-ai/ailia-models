"""Convert the exported Qwen3-TTS ONNX to fp16: weights and MatMul/Conv in fp16,
every other tensor fp32.

Usage:
    python record_fp16_calibration.py -p 0.6B -m custom_voice --onnx_dir ../models/qwen3-tts-streaming
    python convert_to_fp16.py -p 0.6B -m custom_voice --onnx_dir ../models/qwen3-tts-streaming
    python convert_to_fp16.py -p 0.6B -m custom_voice --onnx_dir ../models/qwen3-tts-streaming --only talker_static

Writes qwen3_tts_<name>_<p>[_custom_voice][_static]_fp16.onnx next to each fp32
model (the fused predictor's qwen3_tts_code_predictor_frame_<p>..._sim.onnx becomes
qwen3_tts_code_predictor_<p>..._fp16.onnx: frame / _sim are build steps, not variants), with a prototxt, and needs neither torch nor qwen-tts (the calibration
recording runs the sample on onnxruntime).

Only the weight-carrying ops (MatMul, Conv, ConvTranspose and the embedding
lookups) are computed in fp16, with a Cast on the way in and out; residual
adds, norms, activations, masks and the sampling stay fp32. A full fp16
conversion is not possible for these graphs: the Qwen3 residual stream has
channels around 5e4 and the code predictor's MLP product reaches 1.7e5, past
fp16's 65504, so the norm after them becomes NaN. Even a MatMul whose own
input or output exceeds the range (found by running the recorded fp32 calls,
e.g. the code predictor's layer 2 down_proj) stays fp32. Halves the file size;
on onnxruntime CUDA the talker and the code predictor run at the fp32 speed,
the decoder's fp16 convolutions are slower (56 -> 89 ms per 80-frame window).

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
# name -> file stem suffix after the parameter / family part
MODELS = {
    "decoder": "",
    "prompt": "",
    "codec_embedding": "",
    "talker_static": "_static",            # qwen3_tts_talker_<p>[_custom_voice]_static.onnx (the only talker used)
    "code_predictor_frame": "_sim",        # the onnxsim output of the fused predictor
}
FAMILY_SUFFIX = {"base": "", "custom_voice": "_custom_voice"}


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


# (op_type, input index) pairs that carry shapes or indices, not data
INDEX_INPUTS = {
    ("Slice", 1), ("Slice", 2), ("Slice", 3), ("Slice", 4),
    ("Pad", 1), ("Reshape", 1), ("Expand", 1), ("ConstantOfShape", 0), ("Tile", 1),
    ("Range", 0), ("Range", 1), ("Range", 2), ("Gather", 1), ("GatherElements", 1),
    ("ScatterElements", 1), ("ScatterND", 1), ("Unsqueeze", 1), ("Squeeze", 1),
    ("Resize", 2), ("Resize", 3), ("TopK", 1), ("Split", 1), ("OneHot", 0), ("OneHot", 1),
}


def index_nodes(graph):
    """The nodes whose values end up as a shape, a slice bound or an index.

    Such values are sometimes computed through float ops (a Cast to float, a
    Div, a Floor, a Cast back): in fp16 an index above 2048 loses its low bits
    (the decoder's 14805 became 14800), so the whole chain stays fp32.
    """
    producer = {output: node for node in graph.node for output in node.output}
    blocked, queue = set(), collections.deque()
    for node in graph.node:
        for i, name in enumerate(node.input):
            if (node.op_type, i) in INDEX_INPUTS and name in producer:
                queue.append(producer[name])
    while queue:
        node = queue.popleft()
        if node.name in blocked:
            continue
        blocked.add(node.name)
        if node.op_type in ("Shape", "Size", "TopK", "ArgMax", "ArgMin", "NonZero"):
            continue        # an integer from a shape or a comparison: the data upstream is unaffected
        for name in node.input:
            if name in producer:
                queue.append(producer[name])
    return sorted(blocked)


def convert_prompt(path, out_path):
    """The prompt and the codec embedding run on the CPU, where an fp16 MatMul is
    slow (130 ms a call), so only the embedding table is halved: the table
    initializer becomes fp16 and a Cast back to fp32 follows the gather; the
    projection / sum stays fp32."""
    import numpy as np
    from onnx import numpy_helper
    model = onnx.load(path)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    # the table is an initializer, or (codec embedding) the tensor of a Constant node
    for node in graph.node:
        if node.op_type == "Constant" and node.attribute[0].name == "value":
            inits[node.output[0]] = node.attribute[0].t
    gather = max((n for n in graph.node if n.op_type in ("Gather", "GatherND") and n.input[0] in inits),
                 key=lambda n: np.prod(inits[n.input[0]].dims))
    table = inits[gather.input[0]]
    half = numpy_helper.from_array(numpy_helper.to_array(table).astype(np.float16), table.name)
    table.CopyFrom(half)
    out = gather.output[0]
    gather.output[0] = out + "_fp16"
    cast = onnx.helper.make_node("Cast", [out + "_fp16"], [out], name=gather.name + "_cast", to=onnx.TensorProto.FLOAT)
    idx = list(graph.node).index(gather)
    graph.node.insert(idx + 1, cast)
    for vi in graph.value_info:
        if vi.name == out:
            graph.value_info.remove(vi)
            break
    print(f"  embedding table {list(table.dims)} -> fp16, projection kept fp32")
    save_model(model, out_path)
    generate_prototxt(out_path)


def sampling_nodes(graph):
    """In the fused code predictor, the lm_head projections and everything after
    them: the logits (values around +-20 over a 2048 vocabulary) and the
    top-k / Gumbel-max selection stay fp32, where a near tie is decided the same
    way as in the reference; only the transformer body is halved."""
    consumers = collections.defaultdict(list)
    for node in graph.node:
        for name in node.input:
            consumers[name].append(node)
    queue = collections.deque(n for n in graph.node if "lm_head" in n.name)
    blocked = set()
    while queue:
        node = queue.popleft()
        if node.name in blocked:
            continue
        blocked.add(node.name)
        if node.op_type in ("TopK", "ArgMax", "Gather", "GatherElements"):
            # the chosen token index: what follows is the next step's embedding
            # lookup and transformer layers, which must not be blocked (in the
            # unrolled graph every later step is downstream of this step's logits;
            # blocking them kept 14 of the 15 steps in fp32 with a Cast on every
            # weight, 4 GB of fp32 weight copies on ailia)
            continue
        for output in node.output:
            queue.extend(consumers[output])
    return sorted(blocked)


def rmsnorm_nodes(graph):
    """The statistics half of every RMSNorm: Pow(x, 2) -> ReduceMean -> Add eps ->
    Sqrt -> Reciprocal/Div. The talker's hidden states reach +-90, so the sum of
    1024 squares passes fp16's 65504 and the norm output collapses to zero
    (onnxruntime only fuses the pattern into its fp32-accumulating kernel when
    it recognises it, which it does not in the fused code predictor). These few
    elementwise nodes stay fp32; x * rstd and the weight stay fp16.
    """
    from onnx import numpy_helper
    consumers = collections.defaultdict(list)
    for node in graph.node:
        for name in node.input:
            consumers[name].append(node)
    consts = {i.name: numpy_helper.to_array(i) for i in graph.initializer if i.dims in ([], [1])}
    for node in graph.node:
        if node.op_type == "Constant":
            consts[node.output[0]] = onnx.helper.get_attribute_value(node.attribute[0])
            if not hasattr(consts[node.output[0]], "size"):
                consts[node.output[0]] = numpy_helper.to_array(consts[node.output[0]])
    blocked = set()
    for node in graph.node:
        if node.op_type != "Pow" or len(node.input) < 2:
            continue
        exp = consts.get(node.input[1])
        if exp is None or exp.size != 1 or float(exp.reshape(-1)[0]) != 2.0:
            continue
        chain, cur = [node], node
        for _ in range(5):
            nxt = [c for c in consumers[cur.output[0]] if c.op_type in ("ReduceMean", "Add", "Sqrt", "Reciprocal", "Div", "Cast")]
            if not nxt:
                break
            cur = nxt[0]
            chain.append(cur)
            if cur.op_type in ("Reciprocal", "Div"):
                break
        if any(n.op_type == "ReduceMean" for n in chain):
            blocked |= {n.name for n in chain}
    return sorted(blocked)


# The ops that carry the weights. Only these are computed in fp16; every other
# node (residual adds, norms, activations, masks, sampling) stays fp32, because
# the Qwen3 residual stream has channels around 5e4 and the code predictor's
# MLP product reaches 1.6e5: fp16 (max 65504) turns them into inf.
WEIGHT_OPS = ("MatMul", "Gemm", "Conv", "ConvTranspose", "Gather", "GatherND")
FP16_LIMIT = 2 ** 15     # a weight op whose fp32 input or output passes this stays fp32 too
CALIBRATED = {}          # model name -> blocked weight ops, shared with the static talker

# What the calibration found on the shipped models, so the same files come out of
# a conversion without recorded calls (record_fp16_calibration.py needs the fp32
# models and a run of the sample). Of all the weight ops in the five models only
# one sees activations beyond fp16: the 0.6B code predictor's down projection of
# step 2 (the MLP product reaches 1.6e5), for the Base and the CustomVoice
# checkpoint alike. The 1.7B models and everything else stay within range.
KNOWN_BLOCKS = {
    ("code_predictor_frame", "0.6B"): ["/mlp/down_proj_2/MatMul"],
    ("code_predictor_frame", "1.7B"): [],
    ("talker_static", "0.6B"): [], ("talker_static", "1.7B"): [],
    ("decoder", "0.6B"): [], ("decoder", "1.7B"): [],
}


def calibrated_blocks(model, calls, tmp_dir):
    """Names of the weight ops whose activations do not fit fp16, measured on
    recorded fp32 calls (record_fp16_calibration.py)."""
    import numpy as np
    import onnxruntime as ort
    graph = model.graph
    inits = {i.name for i in graph.initializer}
    watch = {}
    for node in graph.node:
        if node.op_type in WEIGHT_OPS and node.op_type not in ("Gather", "GatherND"):   # tables hold small values
            for t in (node.input[0], node.output[0]):
                if t and t not in inits:
                    watch.setdefault(t, []).append(node.name)
    dbg = onnx.ModelProto()
    dbg.CopyFrom(model)
    outputs = {o.name for o in dbg.graph.output}
    inputs = {i.name for i in dbg.graph.input}
    for t in watch:
        if t not in outputs and t not in inputs:
            dbg.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    dbg_path = os.path.join(tmp_dir, "calibration_probe.onnx")
    onnx.save(dbg, dbg_path, save_as_external_data=True, all_tensors_to_one_file=True,
              location="calibration_probe.onnx.data")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    session = ort.InferenceSession(dbg_path, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    in_names = [i.name for i in session.get_inputs()]
    out_names = [o.name for o in session.get_outputs()]
    absmax = {}
    for call in calls:
        feed = dict(zip(in_names, call))
        for t, v in feed.items():
            if t in watch and v.dtype.kind == "f":
                absmax[t] = max(absmax.get(t, 0.0), float(np.abs(v).max()))
        for t, v in zip(out_names, session.run(None, feed)):
            if t in watch and v.dtype.kind == "f":
                finite = v[np.isfinite(v)]
                absmax[t] = max(absmax.get(t, 0.0), float(np.abs(finite).max()) if finite.size else 0.0)
    for f in (dbg_path, dbg_path + ".data"):
        if os.path.exists(f):
            os.remove(f)
    blocked = {}
    for t, nodes in watch.items():
        if absmax.get(t, 0.0) > FP16_LIMIT:
            for n in nodes:
                blocked[n] = max(blocked.get(n, 0.0), absmax[t])
    for n, v in sorted(blocked.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {n}: activation {v:.0f}")
    return set(blocked)


def convert_weights_only(path, out_path):
    """Store the weights in fp16, compute in fp32: every fp32 initializer feeding a
    MatMul / Conv / ConvTranspose / Gather becomes fp16 with a Cast back to fp32 in
    front of the op. onnxruntime folds Cast(initializer) at session creation,
    so at run time the graph is the fp32 one (same kernels, same speed, fp32
    memory); only the file is halved. Used for the decoder, whose fp16
    convolutions are slower than fp32 ones on cuDNN."""
    import numpy as np
    from onnx import numpy_helper
    model = onnx.load(path)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    casted, new_nodes, n_bytes = {}, [], 0
    for node in graph.node:
        if node.op_type not in WEIGHT_OPS:
            new_nodes.append(node)
            continue
        for k, name in enumerate(node.input):
            t = inits.get(name)
            if t is None or t.data_type != onnx.TensorProto.FLOAT or np.prod(t.dims) < 4096:
                continue
            if name not in casted:
                arr = numpy_helper.to_array(t)
                t.CopyFrom(numpy_helper.from_array(arr.astype(np.float16), name))
                n_bytes += arr.nbytes // 2
                casted[name] = name + "_fp32"
                new_nodes.append(onnx.helper.make_node("Cast", [name], [casted[name]], name=name + "_cast",
                                                       to=onnx.TensorProto.FLOAT))
            node.input[k] = casted[name]
        new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)
    print(f"  {len(casted)} weight tensors stored as fp16 ({n_bytes / 1e6:.0f} MB saved), computed in fp32")
    save_model(model, out_path)
    generate_prototxt(out_path)


def convert(path, out_path, name=None, calibration=None, weights_only=(), parameter_num=None):
    if name in ("prompt", "codec_embedding"):
        return convert_prompt(path, out_path)      # a lookup table plus a little fp32 math
    if name in weights_only:
        return convert_weights_only(path, out_path)
    # the converter only inserts a Cast between an fp32 node and an fp16 node when
    # it knows the tensor's type: the exporter's graphs carry no value_info, so
    # shapes are inferred first (through a file when the model exceeds 2GB)
    if os.path.getsize(path) + sum(os.path.getsize(f) for f in [path + ".data"] if os.path.exists(f)) < 2 ** 31 - 2 ** 20:
        model = onnx.shape_inference.infer_shapes(onnx.load(path), strict_mode=False)
    else:
        # over 2GB the proto cannot be serialized in memory: infer through a file
        # in the same directory, so that the external data is still found
        inferred = os.path.join(os.path.dirname(path), os.path.basename(out_path) + ".shapes.tmp")
        onnx.shape_inference.infer_shapes_path(path, inferred, strict_mode=False)
        model = onnx.load(inferred)
        os.remove(inferred)
    rotary = rotary_nodes(model.graph)
    index = index_nodes(model.graph)
    sampling = sampling_nodes(model.graph) if name == "code_predictor_frame" else []
    norm = rmsnorm_nodes(model.graph)
    others = {n.name for n in model.graph.node if n.op_type not in WEIGHT_OPS}
    print(f"  {len(model.graph.node) - len(others)} weight ops go to fp16, {len(others)} other nodes stay fp32")
    calibrated = set()
    calls = (calibration or {}).get({"code_predictor_frame": "code_predictor", "talker_static": "talker"}.get(name, name))
    if calls:
        calibrated = calibrated_blocks(model, calls, os.path.dirname(out_path))
        CALIBRATED[name] = calibrated
        print(f"  {len(calibrated)} weight ops stay fp32 for their activation range ({len(calls)} recorded calls)")
    elif (name, parameter_num) in KNOWN_BLOCKS:
        calibrated = set(KNOWN_BLOCKS[(name, parameter_num)])
        print(f"  no calibration data: {len(calibrated)} weight ops stay fp32 from the recorded list (KNOWN_BLOCKS)")
    else:
        print("  WARNING: no calibration data for this model, weight ops with large activations may overflow")
    blocked = sorted(set(rotary) | set(index) | set(sampling) | set(norm) | others | calibrated)
    print(f"  keeping {len(rotary)} rotary embedding nodes and {len(index)} shape/index nodes in fp32")
    converted = convert_float_to_float16(
        model,
        keep_io_types=True,
        node_block_list=blocked,
        disable_shape_infer=True,
    )
    removed = remove_fp16_round_trips(converted.graph)
    print(f"  removed {removed} fp32 -> fp16 -> fp32 cast round trips")
    save_model(converted, out_path)
    generate_prototxt(out_path)


def remove_fp16_round_trips(graph):
    """Drop Cast(to fp16) -> Cast(to fp32) pairs.

    Between two nodes that both stay fp32 the converter still routes the edge
    through fp16, which only rounds the value: on the decoder's index chain a
    Ceil result of 14805 came back as 14800 and a Conv no longer matched its
    residual. Removing the pair never changes a result that was correct.
    """
    FP32, FP16 = onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16
    consumers = collections.defaultdict(list)
    for node in graph.node:
        for name in node.input:
            consumers[name].append(node)
    cast_to = lambda node: next((a.i for a in node.attribute if a.name == "to"), None)
    removed, drop = 0, set()
    for node in graph.node:
        if node.op_type != "Cast" or cast_to(node) != FP16:
            continue
        users = consumers[node.output[0]]
        if not users or any(u.op_type != "Cast" or cast_to(u) != FP32 for u in users):
            continue
        if any(node.output[0] == o.name for o in graph.output):
            continue
        for back in users:
            for user in consumers[back.output[0]]:
                user.input[:] = [node.input[0] if n == back.output[0] else n for n in user.input]
            for o in graph.output:
                if o.name == back.output[0]:
                    o.name = node.input[0]
            drop.add(back.name)
        drop.add(node.name)
        removed += 1
    kept = [n for n in graph.node if n.name not in drop]
    del graph.node[:]
    graph.node.extend(kept)
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--parameter_num", default="0.6B", choices=["0.6B", "1.7B"])
    parser.add_argument("--onnx_dir", default=".", help="where the fp32 models are")
    parser.add_argument("--output_dir", default=None, help="defaults to onnx_dir")
    parser.add_argument("--only", default=None, choices=list(MODELS))
    parser.add_argument("-m", "--model", default="base", choices=list(FAMILY_SUFFIX),
                        help="checkpoint family the files were exported from")
    parser.add_argument("--weights_only", default="",
                        help="comma list of models whose weights are stored fp16 but computed fp32 "
                             "(Cast folded by onnxruntime at load); default: none, every model casts its activations")
    parser.add_argument("--calibration", default=None,
                        help="recorded inputs from record_fp16_calibration.py (default: "
                             "<onnx_dir>/calibration_<p>[_custom_voice].pkl when it exists)")
    args = parser.parse_args()

    import pickle
    calib_path = args.calibration or os.path.join(
        args.onnx_dir, f"calibration_{args.parameter_num}{FAMILY_SUFFIX[args.model]}.pkl")
    calibration = pickle.load(open(calib_path, "rb")) if os.path.exists(calib_path) else None
    print("calibration:", calib_path if calibration else "none")

    output_dir = args.output_dir or args.onnx_dir
    os.makedirs(output_dir, exist_ok=True)
    for name in [args.only] if args.only else MODELS:
        base_name = "talker" if name == "talker_static" else name
        stem = f"qwen3_tts_{base_name}_{args.parameter_num}{FAMILY_SUFFIX[args.model]}{MODELS[name]}"
        path = os.path.join(args.onnx_dir, stem + ".onnx")
        if not os.path.exists(path):
            print(f"skipping {stem}.onnx (not found)")
            continue
        out_stem = f"qwen3_tts_code_predictor_{args.parameter_num}{FAMILY_SUFFIX[args.model]}"             if name == "code_predictor_frame" else stem
        out_path = os.path.join(output_dir, out_stem + "_fp16.onnx")
        print(f"converting {stem}.onnx ...")
        convert(path, out_path, name, calibration, tuple(x for x in args.weights_only.split(",") if x),
                parameter_num=args.parameter_num)
        print(f"  {os.path.getsize(path) / 1e6:8.1f} MB -> "
              f"{os.path.getsize(out_path) / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
