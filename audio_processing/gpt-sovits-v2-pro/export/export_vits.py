"""
Export vits.onnx for GPT-SoVITS v2pro.
This script creates a SynthesizerTrn model with v2Pro architecture and exports it to ONNX.
v2Pro requires an additional sv_emb (speaker verification embedding) input.

Prerequisites:
  git clone -b 20250606v2pro https://github.com/RVC-Boss/GPT-SoVITS.git
  cd GPT-SoVITS && pip install -r requirements.txt

Usage:
  python export_vits.py --sovits_path pretrained_models/v2Pro/s2Gv2Pro.pth --output vits.onnx
"""
import sys
import os
import torch
import torch.nn as nn

# Add GPT-SoVITS to path (adjust as needed)
if os.path.exists("/tmp/GPT-SoVITS/GPT_SoVITS"):
    sys.path.insert(0, "/tmp/GPT-SoVITS/GPT_SoVITS")
    sys.path.insert(0, "/tmp/GPT-SoVITS")

from module.models_onnx import SynthesizerTrn
from text import cleaned_text_to_sequence


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    hann_window = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


class VitsModel(nn.Module):
    """Wrapper for SynthesizerTrn that handles spectrogram computation and sv_emb for v2Pro."""

    def __init__(self, vq_model, hps):
        super().__init__()
        self.vq_model = vq_model
        self.hps = hps

    def forward(self, text_seq, pred_semantic, ref_audio, sv_emb):
        refer = spectrogram_torch(
            ref_audio,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        )
        return self.vq_model(pred_semantic, text_seq, refer, sv_emb=sv_emb)[0, 0]


def export_vits(sovits_path=None, output_path="/tmp/onnx_output/vits.onnx"):
    if sovits_path and os.path.exists(sovits_path):
        print(f"Loading SoVITS weights from {sovits_path}")
        dict_s2 = torch.load(sovits_path, map_location="cpu")
        hps = dict_s2["config"]
        hps["model"]["semantic_frame_rate"] = "25hz"
        if dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            hps["model"]["version"] = "v1"
        elif "sv_emb.weight" in dict_s2["weight"]:
            hps["model"]["version"] = "v2Pro"
        else:
            hps["model"]["version"] = "v2"
        hps = DictToAttrRecursive(hps)
        print(f"  version: {hps.model.version}")
        print(f"  gin_channels: {hps.model.gin_channels}")
        print(f"  n_speakers: {hps.data.n_speakers}")
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
        vq_model.eval()
        print(f"  load_state_dict: {vq_model.load_state_dict(dict_s2['weight'], strict=False)}")
    else:
        print("Using random initialization (for architecture validation)")
        hps_dict = {
            "data": {
                "filter_length": 2048,
                "sampling_rate": 32000,
                "hop_length": 640,
                "win_length": 2048,
                "n_speakers": 300,
            },
            "train": {
                "segment_size": 20480,
            },
            "model": {
                "inter_channels": 192,
                "hidden_channels": 192,
                "filter_channels": 768,
                "n_heads": 2,
                "n_layers": 6,
                "kernel_size": 3,
                "p_dropout": 0,
                "resblock": "1",
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [10, 8, 2, 2, 2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [16, 16, 8, 2, 2],
                "gin_channels": 1024,
                "semantic_frame_rate": "25hz",
                "use_sdp": True,
                "freeze_quantizer": None,
                "version": "v2Pro",
            },
        }
        hps = DictToAttrRecursive(hps_dict)
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
        vq_model.eval()

    vits = VitsModel(vq_model, hps)
    vits.eval()

    # Create dummy inputs
    text_seq = torch.LongTensor(
        [
            cleaned_text_to_sequence(
                [
                    "w", "o3", "sh", "i4", "b", "ai2", "y", "e4",
                    "w", "o3", "sh", "i4", "b", "ai2", "y", "e4",
                    "w", "o3", "sh", "i4", "b", "ai2", "y", "e4",
                ],
                version="v2",
            )
        ]
    )
    pred_semantic = torch.randint(0, 1024, (1, 1, 50)).long()
    ref_audio = torch.randn((1, 32000 * 5)).float()
    sv_emb = torch.randn((1, 20480)).float()

    print(f"text_seq shape: {text_seq.shape}")
    print(f"pred_semantic shape: {pred_semantic.shape}")
    print(f"ref_audio shape: {ref_audio.shape}")
    print(f"sv_emb shape: {sv_emb.shape}")

    # Test forward
    with torch.no_grad():
        audio = vits(text_seq, pred_semantic, ref_audio, sv_emb)
    print(f"Output audio shape: {audio.shape}")

    # Export to ONNX
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        vits,
        (text_seq, pred_semantic, ref_audio, sv_emb),
        output_path,
        input_names=["text_seq", "pred_semantic", "ref_audio", "sv_emb"],
        output_names=["audio"],
        dynamic_axes={
            "text_seq": {1: "text_length"},
            "pred_semantic": {2: "pred_length"},
            "ref_audio": {1: "audio_length"},
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
    parser.add_argument("--sovits_path", type=str, default=None)
    parser.add_argument("--output", type=str, default="/tmp/onnx_output/vits.onnx")
    args = parser.parse_args()
    export_vits(args.sovits_path, args.output)
