import sys
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
        '--input_size', type=int, default=504,
        help='input image size (default: 504)'
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

    # Clone the Depth-Anything-3 repo and install dependencies before running:
    #   git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
    #   cd Depth-Anything-3
    #   pip install -e .
    # Then run this script from within the Depth-Anything-3 directory or add it to sys.path.
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError:
        print("Error: Cannot import depth_anything_3.")
        print("Please clone the Depth-Anything-3 repository:")
        print("  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git")
        print("  cd Depth-Anything-3")
        print("  pip install -e .")
        print("Then run this script from within the Depth-Anything-3 directory.")
        sys.exit(1)

    # Load pretrained model
    model_name_map = {
        'vits': 'depth-anything/DA3-Small',
        'vitb': 'depth-anything/DA3-Base',
        'vitl': 'depth-anything/DA3-Large',
    }
    model_name = model_name_map[args.encoder]
    print(f'Loading model {model_name}...')
    da3 = DepthAnything3.from_pretrained(model_name)
    model = da3.model
    model.eval()

    if args.output is None:
        output_path = f'da3_{args.encoder}.onnx'
    else:
        output_path = args.output

    # Create dummy input
    input_size = args.input_size
    dummy_input = torch.randn(1, 3, input_size, input_size)

    print(f'Exporting {args.encoder} model to {output_path}...')
    print(f'Input size: {input_size}x{input_size}')

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=args.opset,
        input_names=['image'],
        output_names=['depth'],
        dynamic_axes={
            'image': {0: 'batch_size', 2: 'height', 3: 'width'},
            'depth': {0: 'batch_size', 1: 'height', 2: 'width'},
        },
    )

    print(f'Successfully exported to {output_path}')


if __name__ == '__main__':
    main()
