"""
Convert embedder.onnx to embedder_dynamic.onnx which accepts an arbitrary
length reference audio.

The original embedder.onnx was exported by tracing SpeechEmbedder
(https://github.com/maum-ai/voicefilter) with a 301 frame (3 sec) mel
spectrogram, so `mel.unfold(1, window=80, stride=40)` was frozen into
6 fixed Slice + Concat nodes and the LSTM initial states were baked in
with batch=6. Inputs shorter than 280 frames fail with a Concat shape
mismatch in ailia.

This script rewrites the graph so that the sliding windows are computed
at runtime (Shape -> Range -> Gather), which is equivalent to unfold:

    T = Shape(dvec_mel)[1]
    N = (T - window) / stride + 1
    starts = Range(0, N, 1) * stride                  # (N,)
    idx = starts[:, None] + Range(0, window, 1)[None] # (N, window)
    out = Gather(dvec_mel, idx, axis=1)               # (40, N, window)

and shrinks the all-zero LSTM h0/c0 initializers from (1, 6, 768) to
(1, 1, 768) so the existing Expand nodes broadcast them to any N.

Usage:
    python3 export_embedder_dynamic.py [--input ../embedder.onnx] [--output ../embedder_dynamic.onnx]
"""
import argparse

import numpy as np
import onnx
from onnx import helper, TensorProto

WINDOW = 80
STRIDE = 40

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='../embedder.onnx')
parser.add_argument('--output', default='../embedder_dynamic.onnx')
args = parser.parse_args()

m = onnx.load(args.input)
g = m.graph

# Nodes 0..30 are the unrolled unfold:
# 18x Constant, 6x Slice, 6x Unsqueeze, 1x Concat -> output '45'
assert g.node[30].op_type == 'Concat' and g.node[30].output[0] == '45'
del g.node[:31]


def const(name, value):
    arr = np.array(value, dtype=np.int64)
    return helper.make_node(
        'Constant', [], [name],
        value=helper.make_tensor(name + '_t', TensorProto.INT64, arr.shape, arr.flatten()))


new_nodes = [
    # T = Shape(dvec_mel)[1]  (scalar)
    helper.make_node('Shape', ['dvec_mel'], ['uf_shape']),
    const('uf_one_s', 1),
    helper.make_node('Gather', ['uf_shape', 'uf_one_s'], ['uf_T'], axis=0),
    # N = (T - WINDOW) / STRIDE + 1  (scalar)
    const('uf_window_s', WINDOW),
    const('uf_stride_s', STRIDE),
    helper.make_node('Sub', ['uf_T', 'uf_window_s'], ['uf_Tm']),
    helper.make_node('Div', ['uf_Tm', 'uf_stride_s'], ['uf_Nd']),
    helper.make_node('Add', ['uf_Nd', 'uf_one_s'], ['uf_N']),
    # starts = Range(0, N, 1) * STRIDE  -> (N,)
    const('uf_zero_s', 0),
    helper.make_node('Range', ['uf_zero_s', 'uf_N', 'uf_one_s'], ['uf_r']),
    helper.make_node('Mul', ['uf_r', 'uf_stride_s'], ['uf_starts']),
    # win = Range(0, WINDOW, 1) -> (WINDOW,)
    helper.make_node('Range', ['uf_zero_s', 'uf_window_s', 'uf_one_s'], ['uf_win']),
    # idx = starts[:, None] + win[None, :] -> (N, WINDOW)
    helper.make_node('Unsqueeze', ['uf_starts'], ['uf_starts2'], axes=[1]),
    helper.make_node('Unsqueeze', ['uf_win'], ['uf_win2'], axes=[0]),
    helper.make_node('Add', ['uf_starts2', 'uf_win2'], ['uf_idx']),
    # gather along time axis: (40, T) -> (40, N, WINDOW)  == old '45'
    helper.make_node('Gather', ['dvec_mel', 'uf_idx'], ['45'], axis=1),
]
for i, n in enumerate(new_nodes):
    g.node.insert(i, n)

# LSTM h0/c0 were traced as zeros with batch=6 baked in; shrink to batch=1
# so the Expand nodes can broadcast to any window count N
for t in g.initializer:
    a = onnx.numpy_helper.to_array(t)
    if a.ndim == 3 and a.shape[1] == 6:
        assert not a.any(), t.name
        t.CopyFrom(onnx.numpy_helper.from_array(
            np.zeros((a.shape[0], 1, a.shape[2]), dtype=a.dtype), t.name))

# make the time axis dynamic in the declared input
g.input[0].type.tensor_type.shape.dim[1].dim_param = 'T'

onnx.checker.check_model(m)
onnx.save(m, args.output)
print('saved', args.output)
