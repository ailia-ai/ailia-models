import sys
import argparse

import torch
import torch.nn as nn


def get_args():
    parser = argparse.ArgumentParser(
        description='Export Depth Anything V3 model to ONNX format'
    )
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
        '--input_height', type=int, default=336,
        help='input image height (default: 336, must be multiple of 14)'
    )
    parser.add_argument(
        '--input_width', type=int, default=504,
        help='input image width (default: 504, must be multiple of 14)'
    )
    return parser.parse_args()


model_name_map = {
    'vits': 'depth-anything/DA3-Small',
    'vitb': 'depth-anything/DA3-Base',
    'vitl': 'depth-anything/DA3-Large',
}

config_name_map = {
    'vits': 'depth_anything_3.configs.da3-small',
    'vitb': 'depth_anything_3.configs.da3-base',
    'vitl': 'depth_anything_3.configs.da3-large',
}


class DepthAnything3Wrapper(nn.Module):
    """Wrapper for ONNX export that takes (B, 3, H, W) input
    and returns (B, H, W) depth output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Add view dimension: (B, 3, H, W) -> (B, 1, 3, H, W)
        x = x.unsqueeze(1)
        output = self.model(x)
        return output.depth


def main():
    args = get_args()

    assert args.encoder in model_name_map, \
        f'encoder should be one of {list(model_name_map.keys())}'
    assert args.input_height % 14 == 0, \
        f'input_height must be a multiple of 14, got {args.input_height}'
    assert args.input_width % 14 == 0, \
        f'input_width must be a multiple of 14, got {args.input_width}'

    # Install depth_anything_3 before running:
    #   pip install depth-anything-3
    # Or clone and install from source:
    #   git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
    #   cd Depth-Anything-3
    #   pip install -e .
    try:
        from depth_anything_3.cfg import load_config, create_object
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install required packages:")
        print("  pip install depth-anything-3 safetensors huggingface_hub")
        sys.exit(1)

    # Create model from config
    config_name = config_name_map[args.encoder]
    print(f'Loading config {config_name}...')
    cfg = load_config(config_name)
    model = create_object(cfg)

    # Download and load pretrained weights
    model_name = model_name_map[args.encoder]
    print(f'Downloading weights from {model_name}...')
    weight_path = hf_hub_download(
        repo_id=model_name,
        filename='model.safetensors',
    )
    state_dict = load_file(weight_path)
    # Strip 'model.' prefix from HuggingFace checkpoint keys
    state_dict = {
        k[len('model.'):] if k.startswith('model.') else k: v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Wrap model for ONNX export
    wrapper = DepthAnything3Wrapper(model)
    wrapper.eval()

    if args.output is None:
        output_path = f'da3_{args.encoder}.onnx'
    else:
        output_path = args.output

    # Create dummy input
    input_h = args.input_height
    input_w = args.input_width
    dummy_input = torch.randn(1, 3, input_h, input_w)

    print(f'Exporting {args.encoder} model to {output_path}...')
    print(f'Input size: {input_h}x{input_w}')

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        opset_version=args.opset,
        input_names=['image'],
        output_names=['depth'],
        dynamic_axes={
            'image': {0: 'batch', 2: 'height', 3: 'width'},
            'depth': {0: 'batch', 2: 'height', 3: 'width'},
        },
        dynamo=False,
    )

    print(f'Successfully exported to {output_path}')
    print()
    print('To upload to GCS:')
    print(f'  gsutil cp {output_path} gs://ailia-models/depth_anything_v3/')


if __name__ == '__main__':
    main()
