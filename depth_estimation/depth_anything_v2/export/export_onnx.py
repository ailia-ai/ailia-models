import sys
import os
import argparse

import torch
import torch.nn as nn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--encoder', '-ec', type=str, default='vits',
        help='model type: vits, vitb, vitl'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='output onnx file path'
    )
    parser.add_argument(
        '--opset', type=int, default=17,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--input_size', type=int, default=518,
        help='input image size (default: 518)'
    )
    return parser.parse_args()


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


def main():
    args = get_args()

    assert args.encoder in model_configs, \
        f'encoder should be one of {list(model_configs.keys())}'

    # Clone the Depth-Anything-V2 repo and install dependencies before running:
    #   git clone https://github.com/DepthAnything/Depth-Anything-V2.git
    #   pip install -r Depth-Anything-V2/requirements.txt
    # Then place this script in the Depth-Anything-V2 directory or add it to sys.path.
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        print("Error: Cannot import depth_anything_v2.")
        print("Please clone the Depth-Anything-V2 repository:")
        print("  git clone https://github.com/DepthAnything/Depth-Anything-V2.git")
        print("  cd Depth-Anything-V2")
        print("  pip install -r requirements.txt")
        print("Then run this script from within the Depth-Anything-V2 directory.")
        sys.exit(1)

    config = model_configs[args.encoder]
    model = DepthAnythingV2(**config)

    # Load pretrained weights
    # Download from: https://github.com/DepthAnything/Depth-Anything-V2#pre-trained-models
    weight_path = f'checkpoints/depth_anything_v2_{args.encoder}.pth'
    print(f'Loading weights from {weight_path}...')
    try:
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: Weight file not found at {weight_path}")
        print("Please download the pretrained weights from:")
        print("  https://github.com/DepthAnything/Depth-Anything-V2#pre-trained-models")
        print(f"And place them at {weight_path}")
        sys.exit(1)

    model.eval()

    if args.output is None:
        output_path = f'depth_anything_v2_depth_anything_v2_{args.encoder}.onnx'
    else:
        output_path = args.output

    # Create dummy input
    input_size = args.input_size
    dummy_input = torch.randn(1, 3, input_size, input_size)

    print(f'Exporting {args.encoder} model to {output_path}...')
    print(f'Input size: {input_size}x{input_size}')

    # Use legacy exporter to produce a single self-contained ONNX file
    # Note: dynamic_axes not used because DINOv2 position embeddings
    # are baked in during tracing and do not support non-square inputs.
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=args.opset,
        input_names=['image'],
        output_names=['depth'],
        dynamo=False,
    )

    # If external data file was created, merge into single file
    ext_data_path = output_path + '.data'
    if os.path.exists(ext_data_path):
        import onnx
        print('Merging external data into single ONNX file...')
        onnx_model = onnx.load(output_path, load_external_data=True)
        onnx.save(onnx_model, output_path)
        if os.path.exists(ext_data_path):
            os.remove(ext_data_path)

    print(f'Successfully exported to {output_path}')


if __name__ == '__main__':
    main()
