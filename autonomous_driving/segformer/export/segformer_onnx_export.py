"""
Export SegFormer-B0 (Cityscapes / ADE20K) to ONNX.

This script downloads pretrained weights from the HuggingFace Model Hub
(nvidia/segformer-b0-finetuned-*) and exports the model as a single ONNX
file. The exported model takes a normalized RGB image and returns the
per-class logits at 1/4 of the input resolution.

Usage:
    # Default: Cityscapes 1024x1024
    python3 segformer_onnx_export.py

    # Specify variant by name (one of the keys in MODEL_VARIANTS below)
    python3 segformer_onnx_export.py --variant cityscapes-512-1024

    # Verify against PyTorch with ONNX Runtime
    python3 segformer_onnx_export.py --verify
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import onnx

from transformers import SegformerForSemanticSegmentation


# All publicly available SegFormer-B0 variants on the HuggingFace Hub.
# key: short name used by --variant
# value: (HF model id, default input height, default input width)
MODEL_VARIANTS = {
    'cityscapes-1024-1024': (
        'nvidia/segformer-b0-finetuned-cityscapes-1024-1024', 1024, 1024),
    'cityscapes-768-768': (
        'nvidia/segformer-b0-finetuned-cityscapes-768-768', 768, 768),
    'cityscapes-640-1280': (
        'nvidia/segformer-b0-finetuned-cityscapes-640-1280', 640, 1280),
    'cityscapes-512-1024': (
        'nvidia/segformer-b0-finetuned-cityscapes-512-1024', 512, 1024),
    'ade-512-512': (
        'nvidia/segformer-b0-finetuned-ade-512-512', 512, 512),
}


class SegformerWrapper(nn.Module):
    """Wrap HuggingFace SegformerForSemanticSegmentation to output logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def export_model(args):
    if args.variant not in MODEL_VARIANTS:
        raise ValueError(
            f'Unknown variant: {args.variant}. '
            f'Choose from: {list(MODEL_VARIANTS.keys())}')

    hf_id, default_h, default_w = MODEL_VARIANTS[args.variant]
    img_h = args.img_h if args.img_h is not None else default_h
    img_w = args.img_w if args.img_w is not None else default_w

    print(f'Loading {hf_id} (input: {img_h}x{img_w})...')
    model = SegformerForSemanticSegmentation.from_pretrained(hf_id)
    model.eval()
    wrapper = SegformerWrapper(model)

    num_params = sum(p.numel() for p in model.parameters())
    num_labels = model.config.num_labels
    print(f'Parameters: {num_params:,}')
    print(f'Num labels: {num_labels}')

    dummy = torch.randn(1, 3, img_h, img_w)
    print('Testing forward pass...')
    with torch.no_grad():
        logits = wrapper(dummy)
    print(f'  logits: {tuple(logits.shape)}')

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__) or '.', '..',
            f'segformer_b0_{args.variant}.onnx')

    print(f'Exporting to {output_path} (opset={args.opset})...')
    torch.onnx.export(
        wrapper,
        (dummy,),
        output_path,
        input_names=['pixel_values'],
        output_names=['logits'],
        opset_version=args.opset,
        do_constant_folding=True,
    )

    # Merge external data into a single .onnx file (for small B0 it
    # usually fits in a single file, but keep the same logic as bevformer).
    print('Merging weights into single ONNX file...')
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx.save(onnx_model, output_path, save_as_external_data=False)
    data_path = output_path + '.data'
    if os.path.exists(data_path):
        os.remove(data_path)
        print(f'Removed {data_path}')

    print('Verifying ONNX model...')
    onnx_model = onnx.load(output_path)
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print(f'  Warning: ONNX checker: {e}')

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f'ONNX model saved: {output_path} ({file_size:.1f} MB)')

    print('\nModel inputs:')
    for inp in onnx_model.graph.input:
        name = inp.name
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f'  {name}: {shape}')
    print('Model outputs:')
    for out in onnx_model.graph.output:
        name = out.name
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f'  {name}: {shape}')

    if args.verify:
        verify_onnx_runtime(output_path, dummy, logits)

    return output_path


def verify_onnx_runtime(onnx_path, dummy, pt_logits):
    import onnxruntime as ort

    print('\n--- ONNX Runtime Verification ---')
    session = ort.InferenceSession(
        onnx_path, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    inputs = {input_name: dummy.numpy()}
    ort_logits = session.run(None, inputs)[0]

    pt_np = pt_logits.detach().numpy()
    diff = float(np.abs(ort_logits - pt_np).max())
    print(f'  logits max diff: {diff:.6f}')
    tol = 1e-3
    if diff < tol:
        print(f'  PASSED (tolerance={tol})')
    else:
        print(f'  WARNING: diff exceeds tolerance={tol} '
              '(may be acceptable for float32 precision)')


def main():
    parser = argparse.ArgumentParser(
        description='Export SegFormer-B0 to ONNX')
    parser.add_argument(
        '--variant', type=str, default='cityscapes-1024-1024',
        choices=list(MODEL_VARIANTS.keys()),
        help='Model variant to export.')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output ONNX file path.')
    parser.add_argument(
        '--opset', type=int, default=18,
        help='ONNX opset version.')
    parser.add_argument(
        '--img_h', type=int, default=None,
        help='Override input image height.')
    parser.add_argument(
        '--img_w', type=int, default=None,
        help='Override input image width.')
    parser.add_argument(
        '--verify', action='store_true',
        help='Verify with ONNX Runtime after export.')
    args = parser.parse_args()

    export_model(args)


if __name__ == '__main__':
    main()
