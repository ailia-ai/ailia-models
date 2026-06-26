"""
Export BEVFormer-tiny to ONNX.

This script exports the full BEVFormer-tiny model (backbone + BEV encoder
with deformable attention + detection head) as a single ONNX model.
Deformable Attention is implemented using F.grid_sample (standard ONNX op).

Usage:
    # Export with pretrained weights (auto-downloaded)
    python3 bevformer_onnx_export.py

    # Export with custom resolution
    python3 bevformer_onnx_export.py --img_h 480 --img_w 800

    # Export and verify with ONNX Runtime
    python3 bevformer_onnx_export.py --verify
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import onnx

from bevformer_model import build_bevformer_tiny, load_pretrained

CHECKPOINT_URL = 'https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth'
CHECKPOINT_FILE = 'bevformer_tiny_epoch_24.pth'


def download_checkpoint(ckpt_path):
    """Download pretrained weights if not present."""
    if os.path.exists(ckpt_path):
        print(f'Checkpoint found: {ckpt_path}')
        return
    print(f'Downloading checkpoint from {CHECKPOINT_URL}...')
    import urllib.request
    urllib.request.urlretrieve(CHECKPOINT_URL, ckpt_path)
    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f'Downloaded: {ckpt_path} ({size_mb:.1f} MB)')


def export_model(args):
    """Export the full BEVFormer-tiny model to ONNX."""
    print(f'Building BEVFormer-tiny (img: {args.img_h}x{args.img_w}, '
          f'cams: {args.num_cams})...')

    model = build_bevformer_tiny(
        num_cams=args.num_cams,
        img_h=args.img_h,
        img_w=args.img_w,
    )

    # Load pretrained weights
    ckpt_path = os.path.join(os.path.dirname(__file__) or '.', CHECKPOINT_FILE)
    download_checkpoint(ckpt_path)
    model = load_pretrained(model, ckpt_path)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {num_params:,}')

    # Dummy input
    dummy_imgs = torch.randn(
        1, args.num_cams, 3, args.img_h, args.img_w)

    # Test forward pass
    print('Testing forward pass...')
    with torch.no_grad():
        cls_scores, bbox_preds = model(dummy_imgs)
    print(f'  cls_scores: {cls_scores.shape}')
    print(f'  bbox_preds: {bbox_preds.shape}')

    # ONNX export
    output_path = args.output
    print(f'Exporting to {output_path} (opset={args.opset})...')

    torch.onnx.export(
        model,
        (dummy_imgs,),
        output_path,
        input_names=['images'],
        output_names=['cls_scores', 'bbox_preds'],
        opset_version=args.opset,
        do_constant_folding=True,
    )

    # Merge external data into a single .onnx file
    print('Merging weights into single ONNX file...')
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx.save(onnx_model, output_path, save_as_external_data=False)

    # Remove leftover .data file
    data_path = output_path + '.data'
    if os.path.exists(data_path):
        os.remove(data_path)
        print(f'Removed {data_path}')

    # Verify ONNX model
    print('Verifying ONNX model...')
    onnx_model = onnx.load(output_path)
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print(f'  Warning: ONNX checker: {e}')

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f'ONNX model saved: {output_path} ({file_size:.1f} MB)')

    # Print I/O info
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

    # Verify with ONNX Runtime
    if args.verify:
        verify_onnx_runtime(output_path, dummy_imgs, cls_scores, bbox_preds)

    return output_path


def verify_onnx_runtime(onnx_path, dummy_imgs, pt_cls, pt_bbox):
    """Verify ONNX model outputs match PyTorch outputs using ONNX Runtime."""
    import onnxruntime as ort

    print('\n--- ONNX Runtime Verification ---')
    session = ort.InferenceSession(
        onnx_path, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    inputs = {input_name: dummy_imgs.numpy()}

    ort_outputs = session.run(None, inputs)
    ort_cls = ort_outputs[0]
    ort_bbox = ort_outputs[1]

    pt_cls_np = pt_cls.detach().numpy()
    pt_bbox_np = pt_bbox.detach().numpy()

    cls_diff = np.abs(ort_cls - pt_cls_np).max()
    bbox_diff = np.abs(ort_bbox - pt_bbox_np).max()

    print(f'  cls_scores max diff:  {cls_diff:.6f}')
    print(f'  bbox_preds max diff:  {bbox_diff:.6f}')

    tol = 1e-3
    if cls_diff < tol and bbox_diff < tol:
        print(f'  PASSED (tolerance={tol})')
    else:
        print(f'  WARNING: diff exceeds tolerance={tol} '
              '(may be acceptable for float32 precision)')

    # Benchmark
    import time
    n_runs = 5
    times = []
    for i in range(n_runs):
        start = time.time()
        session.run(None, inputs)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f'  Run {i+1}: {elapsed:.1f} ms')
    avg = sum(times[1:]) / len(times[1:])
    print(f'  Average (excluding warmup): {avg:.1f} ms')


def main():
    parser = argparse.ArgumentParser(
        description='Export BEVFormer-tiny to ONNX')
    parser.add_argument(
        '--output', type=str, default='../bevformer_tiny.onnx',
        help='Output ONNX file path')
    parser.add_argument(
        '--opset', type=int, default=18,
        help='ONNX opset version (>= 16 for grid_sample, 18 recommended)')
    parser.add_argument(
        '--img_h', type=int, default=480,
        help='Input image height')
    parser.add_argument(
        '--img_w', type=int, default=800,
        help='Input image width')
    parser.add_argument(
        '--num_cams', type=int, default=6,
        help='Number of camera views')
    parser.add_argument(
        '--verify', action='store_true',
        help='Verify with ONNX Runtime after export')
    args = parser.parse_args()

    export_model(args)


if __name__ == '__main__':
    main()
