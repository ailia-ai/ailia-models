#!/bin/bash
# Export vits.onnx from pretrained s2Gv2Pro.pth weights
#
# Prerequisites:
#   git clone -b 20250606v2pro https://github.com/RVC-Boss/GPT-SoVITS.git
#   cd GPT-SoVITS
#   pip install -r requirements.txt
#
# Download pretrained model from HuggingFace:
#   mkdir -p GPT_SoVITS/pretrained_models/v2Pro
#   wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/v2Pro/s2Gv2Pro.pth \
#        -O GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth
#
# Usage:
#   bash export_vits_from_pretrained.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd GPT-SoVITS/GPT_SoVITS || { echo "Please run from the GPT-SoVITS root directory"; exit 1; }

python3 "${SCRIPT_DIR}/export_vits.py" \
    --sovits_path pretrained_models/v2Pro/s2Gv2Pro.pth \
    --output vits.onnx

echo ""
echo "Done! Copy vits.onnx to the gpt-sovits-v2-pro ONNX model directory."
