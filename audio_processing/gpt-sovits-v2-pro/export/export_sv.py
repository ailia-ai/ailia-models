"""
Export sv.onnx (Speaker Verification model) for GPT-SoVITS v2pro.
The SV model (ERes2NetV2) extracts speaker embeddings from fbank features.

Input: fbank features (B, T, 80) - Kaldi-style 80-dim fbank at 16kHz
Output: sv_emb (B, 20480) - speaker verification embedding

Prerequisites:
  git clone -b 20250606v2pro https://github.com/RVC-Boss/GPT-SoVITS.git
  cd GPT-SoVITS && pip install -r requirements.txt

Usage:
  python export_sv.py --sv_path pretrained_eres2netv2w24s4ep4.ckpt --output sv.onnx
"""
import sys
import os
import torch
import torch.nn as nn

# Add GPT-SoVITS eres2net to path
if os.path.exists("/tmp/GPT-SoVITS/GPT_SoVITS/eres2net"):
    sys.path.insert(0, "/tmp/GPT-SoVITS/GPT_SoVITS/eres2net")

from ERes2NetV2 import ERes2NetV2


class SVModel(nn.Module):
    """Wrapper for ERes2NetV2 that uses forward3 (returns flattened embedding)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, fbank_feat):
        # fbank_feat: (B, T, 80)
        x = fbank_feat.permute(0, 2, 1)  # (B, 80, T) => (B, F, T)
        x = x.unsqueeze(1)  # (B, 1, F, T)
        out = torch.relu(self.model.bn1(self.model.conv1(x)))
        out1 = self.model.layer1(out)
        out2 = self.model.layer2(out1)
        out3 = self.model.layer3(out2)
        out4 = self.model.layer4(out3)
        out3_ds = self.model.layer3_ds(out3)
        fuse_out34 = self.model.fuse34(out4, out3_ds)
        return fuse_out34.flatten(start_dim=1, end_dim=2).mean(-1)


def export_sv(sv_path=None, output_path="/tmp/onnx_output/sv.onnx"):
    embedding_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)

    if sv_path and os.path.exists(sv_path):
        print(f"Loading SV weights from {sv_path}")
        pretrained_state = torch.load(sv_path, map_location="cpu", weights_only=False)
        embedding_model.load_state_dict(pretrained_state)
    else:
        print("Using random initialization (for architecture validation)")

    embedding_model.eval()

    sv_model = SVModel(embedding_model)
    sv_model.eval()

    # Dummy input: 5 seconds of fbank at 16kHz (10ms frame shift = 500 frames)
    fbank_feat = torch.randn(1, 500, 80)

    print(f"fbank_feat shape: {fbank_feat.shape}")

    with torch.no_grad():
        sv_emb = sv_model(fbank_feat)
    print(f"sv_emb shape: {sv_emb.shape}")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        sv_model,
        (fbank_feat,),
        output_path,
        input_names=["fbank_feat"],
        output_names=["sv_emb"],
        dynamic_axes={
            "fbank_feat": {1: "time_length"},
        },
        opset_version=17,
        verbose=False,
        dynamo=False,
    )
    print(f"Exported {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sv_path", type=str, default=None)
    parser.add_argument("--output", type=str, default="/tmp/onnx_output/sv.onnx")
    args = parser.parse_args()
    export_sv(args.sv_path, args.output)
