"""
CLIP inference — builtin backend.
Uses ViT-L14-encode_image.onnx and ViT-L14-encode_text.onnx (existing ailia CLIP models).
"""

import os
import sys

import numpy as np

# ======================
# Model file constants
# ======================

VIS_WEIGHT = "ViT-L14-encode_image.onnx"
VIS_MODEL = "ViT-L14-encode_image.onnx.prototxt"
TXT_WEIGHT = "ViT-L14-encode_text.onnx"
TXT_MODEL = "ViT-L14-encode_text.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/clip/"

# ======================
# SimpleTokenizer
# ======================

_clip_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "image_classification",
    "clip",
)
sys.path.insert(0, _clip_dir)
from simple_tokenizer import SimpleTokenizer  # type: ignore

sys.path.pop(0)

_simple_tokenizer = SimpleTokenizer()


def _simple_tokenize(texts, context_length=77):
    sot = _simple_tokenizer.encoder["<|startoftext|>"]
    eot = _simple_tokenizer.encoder["<|endoftext|>"]
    result = np.zeros((len(texts), context_length), dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = [sot] + _simple_tokenizer.encode(text) + [eot]
        result[i, : min(len(tokens), context_length)] = tokens[:context_length]
    return result


# ======================
# Inference
# ======================


def run_vision(net, img_tensor, onnx):
    """Run CLIP vision encoder.

    img_tensor : (1, 3, 224, 224) float32 — caller handles preprocessing.
    Returns    : (D,) L2-normalised feature vector.
    """
    if not onnx:
        output = net.predict([img_tensor])
    else:
        output = net.run(None, {"image": img_tensor})
    feats = output[0]  # (1, D) — not L2-normalised in the ONNX graph
    feats = feats / np.linalg.norm(feats, ord=2, axis=-1, keepdims=True)
    return feats[0]  # (D,)


def run_text(net, texts, onnx, clip_tokenizer=None):
    """Run CLIP text encoder.

    texts          : list[str]
    clip_tokenizer : unused (kept for API compatibility with custom backend).
    Returns        : (N, D) L2-normalised feature matrix.
    """
    token_ids = _simple_tokenize(texts)  # (N, 77)
    if not onnx:
        output = net.predict([token_ids])
    else:
        output = net.run(None, {"text": token_ids})
    feats = output[0]  # (N, D) — not L2-normalised in the ONNX graph
    return feats / np.linalg.norm(feats, ord=2, axis=-1, keepdims=True)
