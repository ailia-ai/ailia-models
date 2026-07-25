"""
CLIP inference — local backend.
Uses clip_vision_encoder.onnx and clip_text_encoder.onnx (independently exported models).
L2-normalisation is baked into the ONNX graphs.
"""

import numpy as np

# ======================
# Model file constants
# ======================

VIS_WEIGHT = "clip_vision_encoder.onnx"
VIS_MODEL = "clip_vision_encoder.onnx.prototxt"
TXT_WEIGHT = "clip_text_encoder.onnx"
TXT_MODEL = "clip_text_encoder.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/semantic-segment-anything/"

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
        output = net.run(None, {"pixel_values": img_tensor})
    return output[0][0]  # (D,) — already L2-normalised inside the ONNX graph


def run_text(net, texts, onnx, clip_tokenizer=None):
    """Run CLIP text encoder.

    texts          : list[str]
    clip_tokenizer : CLIPTokenizerFast instance — required.
    Returns        : (N, D) L2-normalised feature matrix.
    """
    enc = clip_tokenizer(
        texts, return_tensors="np", padding=True, truncation=True, max_length=77
    )
    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)
    if not onnx:
        output = net.predict([input_ids, attention_mask])
    else:
        output = net.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
    return output[0]  # (N, D) — already L2-normalised inside the ONNX graph
