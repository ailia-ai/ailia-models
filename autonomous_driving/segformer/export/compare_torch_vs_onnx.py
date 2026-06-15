"""Compare PyTorch (HuggingFace) vs ONNX outputs end-to-end on input.jpg.

Generates two side-by-side overlays plus a difference map and prints
numerical agreement statistics.
"""

import argparse
import os
import sys

import numpy as np
import cv2
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))


# Same map as in segformer.py / segformer_onnx_export.py
MODEL_VARIANTS = {
    'cityscapes-1024-1024': (1024, 1024, 19, 'cityscapes'),
    'cityscapes-768-768':   (768,  768,  19, 'cityscapes'),
    'cityscapes-640-1280':  (640,  1280, 19, 'cityscapes'),
    'cityscapes-512-1024':  (512,  1024, 19, 'cityscapes'),
    'ade-512-512':          (512,  512,  150, 'ade'),
}

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)

CITYSCAPES_LABELS = [
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle',
]


def preprocess(img_bgr):
    img_resized = cv2.resize(
        img_bgr, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = (img_rgb - IMG_MEAN) / IMG_STD
    img_chw = img_rgb.transpose(2, 0, 1)
    return np.expand_dims(img_chw, axis=0).astype(np.float32)


def upsample_argmax(logits, orig_h, orig_w):
    logits = logits[0]  # (C, h, w)
    C, h, w = logits.shape
    upsampled = np.empty((C, orig_h, orig_w), dtype=np.float32)
    for c in range(C):
        upsampled[c] = cv2.resize(
            logits[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return np.argmax(upsampled, axis=0).astype(np.int32), upsampled


def colorize(pred, palette):
    rgb = palette[np.clip(pred, 0, palette.shape[0] - 1)]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def overlay(img_bgr, color_bgr, alpha=ALPHA):
    blended = (img_bgr.astype(np.float32) * (1 - alpha)
               + color_bgr.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def class_summary(pred, prefix=''):
    unique, counts = np.unique(pred, return_counts=True)
    total = pred.size
    rows = sorted(
        [(100.0 * c / total, CITYSCAPES_LABELS[int(u)] if u < len(CITYSCAPES_LABELS) else f'class_{u}')
         for u, c in zip(unique, counts)],
        reverse=True)
    print(f'{prefix}top classes:')
    for ratio, name in rows[:8]:
        print(f'  {prefix}  {name}: {ratio:.2f}%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='cityscapes-1024-1024',
                    choices=list(MODEL_VARIANTS.keys()))
    ap.add_argument('--input', default=os.path.join(HERE, '..', 'input.jpg'))
    ap.add_argument('--output', default=None,
                    help='Output PNG path (default: ../compare_<variant>.png)')
    cfg = ap.parse_args()

    global IMAGE_HEIGHT, IMAGE_WIDTH
    IMAGE_HEIGHT, IMAGE_WIDTH, _, _ = MODEL_VARIANTS[cfg.variant]
    hf_id = f'nvidia/segformer-b0-finetuned-{cfg.variant}'
    onnx_path = os.path.join(HERE, '..', f'segformer_b0_{cfg.variant}.onnx')
    out_path = cfg.output or os.path.join(
        HERE, '..', f'compare_{cfg.variant}.png')

    print(f'Loading image: {cfg.input}')
    img_bgr = cv2.imread(cfg.input)
    h, w = img_bgr.shape[:2]
    print(f'  image: {w}x{h}')

    blob = preprocess(img_bgr)
    print(f'  preprocessed blob: {blob.shape} dtype={blob.dtype}')

    # ---- PyTorch reference ----
    print(f'\nLoading PyTorch model from {hf_id} ...')
    model = SegformerForSemanticSegmentation.from_pretrained(hf_id)
    model.eval()
    with torch.no_grad():
        torch_logits = model(pixel_values=torch.from_numpy(blob)).logits.numpy()
    print(f'  torch logits: {torch_logits.shape}')

    # ---- ONNX Runtime ----
    print(f'\nLoading ONNX model from {onnx_path} ...')
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    onnx_logits = session.run(None, {input_name: blob})[0]
    print(f'  onnx  logits: {onnx_logits.shape}')

    # Numerical comparison on raw logits
    diff = np.abs(torch_logits - onnx_logits)
    print('\n=== Logit comparison ===')
    print(f'  max abs diff:  {diff.max():.6e}')
    print(f'  mean abs diff: {diff.mean():.6e}')
    print(f'  PT logit range: [{torch_logits.min():.3f}, {torch_logits.max():.3f}]')

    # Argmax & per-pixel agreement on the upsampled output
    torch_pred, _ = upsample_argmax(torch_logits, h, w)
    onnx_pred, _ = upsample_argmax(onnx_logits, h, w)
    agree = (torch_pred == onnx_pred).mean() * 100.0
    print(f'  per-pixel argmax agreement: {agree:.4f}%')

    class_summary(torch_pred, prefix='[torch] ')
    class_summary(onnx_pred, prefix='[onnx]  ')

    # Render overlays
    palette = CITYSCAPES_PALETTE
    torch_overlay = overlay(img_bgr, colorize(torch_pred, palette))
    onnx_overlay = overlay(img_bgr, colorize(onnx_pred, palette))

    # Difference map (yellow where the two predictions disagree)
    diff_mask = (torch_pred != onnx_pred).astype(np.uint8) * 255
    diff_color = np.zeros_like(img_bgr)
    diff_color[..., 1] = diff_mask  # G
    diff_color[..., 2] = diff_mask  # R  -> yellow
    diff_overlay = overlay(img_bgr, diff_color, alpha=0.6)

    # Compose 3-panel comparison
    h_pad = 30
    label_h = 30
    panels = []
    for title, vis in [('torch', torch_overlay),
                       ('onnx', onnx_overlay),
                       (f'diff ({100 - agree:.2f}% disagreement)', diff_overlay)]:
        canvas = np.zeros((vis.shape[0] + label_h, vis.shape[1], 3), dtype=np.uint8)
        canvas[label_h:] = vis
        cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(canvas)
    sep = np.full((panels[0].shape[0], 4, 3), 255, dtype=np.uint8)
    composite = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)

    cv2.imwrite(out_path, composite)
    print(f'\nSaved comparison image: {out_path}')


if __name__ == '__main__':
    main()
