"""
GPT-SoVITS v2pro ONNX Export Script

This script exports the GPT-SoVITS v2pro PyTorch models to ONNX format.
It should be run from within the GPT-SoVITS repository (20250606v2pro branch).

Usage:
    git clone -b 20250606v2pro https://github.com/RVC-Boss/GPT-SoVITS.git
    cd GPT-SoVITS/GPT_SoVITS
    python /path/to/export_onnx.py \
        --sovits_path SoVITS_weights_v2pro/your_model.pth \
        --gpt_path GPT_weights_v2pro/your_model.ckpt \
        --output_dir onnx_output

Exports the following ONNX models:
    - cnhubert.onnx       : SSL feature extractor (Chinese HuBERT)
    - t2s_encoder.onnx    : Text-to-Semantic encoder
    - t2s_fsdec.onnx      : Text-to-Semantic first stage decoder
    - t2s_sdec.onnx       : Text-to-Semantic stage decoder
    - vits.onnx           : VITS vocoder

Reference:
    https://github.com/RVC-Boss/GPT-SoVITS/blob/20250606v2pro/GPT_SoVITS/onnx_export.py
"""

import argparse
import os
import sys

import torch
import torchaudio
from torch import nn

# Add the GPT_SoVITS directory to path
now_dir = os.getcwd()
sys.path.append(now_dir)

from AR.models.t2s_lightning_module_onnx import Text2SemanticLightningModule
from feature_extractor import cnhubert
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


class T2SEncoder(nn.Module):
    def __init__(self, t2s, vits):
        super().__init__()
        self.encoder = t2s.onnx_encoder
        self.vits = vits

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        codes = self.vits.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        bert = torch.cat([ref_bert.transpose(0, 1), text_bert.transpose(0, 1)], 1)
        all_phoneme_ids = torch.cat([ref_seq, text_seq], 1)
        bert = bert.unsqueeze(0)
        prompt = prompt_semantic.unsqueeze(0)
        return self.encoder(all_phoneme_ids, bert), prompt


class T2SModel(nn.Module):
    def __init__(self, t2s_path, vits_model):
        super().__init__()
        dict_s1 = torch.load(t2s_path, map_location="cpu")
        self.config = dict_s1["config"]
        self.t2s_model = Text2SemanticLightningModule(
            self.config, "ojbk", is_train=False
        )
        self.t2s_model.load_state_dict(dict_s1["weight"])
        self.t2s_model.eval()
        self.vits_model = vits_model.vq_model
        self.hz = 50
        self.max_sec = self.config["data"]["max_sec"]
        self.t2s_model.model.top_k = torch.LongTensor(
            [self.config["inference"]["top_k"]]
        )
        self.t2s_model.model.early_stop_num = torch.LongTensor(
            [self.hz * self.max_sec]
        )
        self.t2s_model = self.t2s_model.model
        self.t2s_model.init_onnx()
        self.onnx_encoder = T2SEncoder(self.t2s_model, self.vits_model)
        self.first_stage_decoder = self.t2s_model.first_stage_decoder
        self.stage_decoder = self.t2s_model.stage_decoder

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        early_stop_num = self.t2s_model.early_stop_num

        x, prompts = self.onnx_encoder(
            ref_seq, text_seq, ref_bert, text_bert, ssl_content
        )

        prefix_len = prompts.shape[1]

        y, k, v, y_emb, x_example = self.first_stage_decoder(x, prompts)

        stop = False
        for idx in range(1, 1500):
            enco = self.stage_decoder(y, k, v, y_emb, x_example)
            y, k, v, y_emb, logits, samples = enco
            if early_stop_num != -1 and (
                y.shape[1] - prefix_len
            ) > early_stop_num:
                stop = True
            if (
                torch.argmax(logits, dim=-1)[0] == self.t2s_model.EOS
                or samples[0, 0] == self.t2s_model.EOS
            ):
                stop = True
            if stop:
                break
        y[0, -1] = 0

        return y[:, -idx:].unsqueeze(0)

    def export(
        self,
        ref_seq,
        text_seq,
        ref_bert,
        text_bert,
        ssl_content,
        output_dir,
    ):
        torch.onnx.export(
            self.onnx_encoder,
            (ref_seq, text_seq, ref_bert, text_bert, ssl_content),
            f"{output_dir}/t2s_encoder.onnx",
            input_names=[
                "ref_seq",
                "text_seq",
                "ref_bert",
                "text_bert",
                "ssl_content",
            ],
            output_names=["x", "prompts"],
            dynamic_axes={
                "ref_seq": {1: "ref_length"},
                "text_seq": {1: "text_length"},
                "ref_bert": {0: "ref_length"},
                "text_bert": {0: "text_length"},
                "ssl_content": {2: "ssl_length"},
            },
            opset_version=16,
        )
        print(f"Exported {output_dir}/t2s_encoder.onnx")

        x, prompts = self.onnx_encoder(
            ref_seq, text_seq, ref_bert, text_bert, ssl_content
        )

        torch.onnx.export(
            self.first_stage_decoder,
            (x, prompts),
            f"{output_dir}/t2s_fsdec.onnx",
            input_names=["x", "prompts"],
            output_names=["y", "k", "v", "y_emb", "x_example"],
            dynamic_axes={
                "x": {1: "x_length"},
                "prompts": {1: "prompts_length"},
            },
            verbose=False,
            opset_version=16,
        )
        print(f"Exported {output_dir}/t2s_fsdec.onnx")

        y, k, v, y_emb, x_example = self.first_stage_decoder(x, prompts)

        torch.onnx.export(
            self.stage_decoder,
            (y, k, v, y_emb, x_example),
            f"{output_dir}/t2s_sdec.onnx",
            input_names=["iy", "ik", "iv", "iy_emb", "ix_example"],
            output_names=["y", "k", "v", "y_emb", "logits", "samples"],
            dynamic_axes={
                "iy": {1: "iy_length"},
                "ik": {1: "ik_length"},
                "iv": {1: "iv_length"},
                "iy_emb": {1: "iy_emb_length"},
                "ix_example": {1: "ix_example_length"},
            },
            verbose=False,
            opset_version=16,
        )
        print(f"Exported {output_dir}/t2s_sdec.onnx")


class VitsModel(nn.Module):
    def __init__(self, vits_path):
        super().__init__()
        dict_s2 = torch.load(vits_path, map_location="cpu")
        self.hps = dict_s2["config"]
        if dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            self.hps["model"]["version"] = "v1"
        else:
            self.hps["model"]["version"] = "v2"

        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        )
        self.vq_model.eval()
        self.vq_model.load_state_dict(dict_s2["weight"], strict=False)

    def forward(self, text_seq, pred_semantic, ref_audio):
        refer = spectrogram_torch(
            ref_audio,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        )
        return self.vq_model(pred_semantic, text_seq, refer)[0, 0]


class GptSoVits(nn.Module):
    def __init__(self, vits, t2s):
        super().__init__()
        self.vits = vits
        self.t2s = t2s

    def forward(
        self,
        ref_seq,
        text_seq,
        ref_bert,
        text_bert,
        ref_audio,
        ssl_content,
    ):
        pred_semantic = self.t2s(
            ref_seq, text_seq, ref_bert, text_bert, ssl_content
        )
        audio = self.vits(text_seq, pred_semantic, ref_audio)
        return audio

    def export(
        self,
        ref_seq,
        text_seq,
        ref_bert,
        text_bert,
        ref_audio,
        ssl_content,
        output_dir,
    ):
        self.t2s.export(
            ref_seq, text_seq, ref_bert, text_bert, ssl_content, output_dir
        )
        pred_semantic = self.t2s(
            ref_seq, text_seq, ref_bert, text_bert, ssl_content
        )
        torch.onnx.export(
            self.vits,
            (text_seq, pred_semantic, ref_audio),
            f"{output_dir}/vits.onnx",
            input_names=["text_seq", "pred_semantic", "ref_audio"],
            output_names=["audio"],
            dynamic_axes={
                "text_seq": {1: "text_length"},
                "pred_semantic": {2: "pred_length"},
                "ref_audio": {1: "audio_length"},
            },
            opset_version=17,
            verbose=False,
        )
        print(f"Exported {output_dir}/vits.onnx")


class SSLModel(nn.Module):
    def __init__(self, ssl):
        super().__init__()
        self.ssl = ssl

    def forward(self, ref_audio_16k):
        return (
            self.ssl.model(ref_audio_16k)["last_hidden_state"]
            .transpose(1, 2)
        )


def export(sovits_path, gpt_path, output_dir, hubert_path):
    cnhubert.cnhubert_base_path = hubert_path
    ssl_model = cnhubert.get_model()

    vits = VitsModel(sovits_path)
    gpt = T2SModel(gpt_path, vits)
    gpt_sovits = GptSoVits(vits, gpt)
    ssl = SSLModel(ssl_model)

    vits_model = "v2"

    ref_seq = torch.LongTensor(
        [
            cleaned_text_to_sequence(
                [
                    "n",
                    "i2",
                    "h",
                    "ao3",
                    ",",
                    "w",
                    "o3",
                    "sh",
                    "i4",
                    "b",
                    "ai2",
                    "y",
                    "e4",
                ],
                version=vits_model,
            )
        ]
    )
    text_seq = torch.LongTensor(
        [
            cleaned_text_to_sequence(
                [
                    "w",
                    "o3",
                    "sh",
                    "i4",
                    "b",
                    "ai2",
                    "y",
                    "e4",
                    "w",
                    "o3",
                    "sh",
                    "i4",
                    "b",
                    "ai2",
                    "y",
                    "e4",
                    "w",
                    "o3",
                    "sh",
                    "i4",
                    "b",
                    "ai2",
                    "y",
                    "e4",
                ],
                version=vits_model,
            )
        ]
    )
    ref_bert = torch.randn((ref_seq.shape[1], 1024)).float()
    text_bert = torch.randn((text_seq.shape[1], 1024)).float()
    ref_audio = torch.randn((1, 48000 * 5)).float()
    ref_audio_16k = torchaudio.functional.resample(
        ref_audio, 48000, 16000
    ).float()
    ref_audio_sr = torchaudio.functional.resample(
        ref_audio, 48000, vits.hps.data.sampling_rate
    ).float()

    os.makedirs(output_dir, exist_ok=True)

    ssl_content = ssl(ref_audio_16k).float()

    # Export SSL model (cnhubert)
    torch.onnx.export(
        ssl,
        (ref_audio_16k,),
        f"{output_dir}/cnhubert.onnx",
        input_names=["ref_audio_16k"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "ref_audio_16k": {1: "audio_length"},
        },
        opset_version=17,
        verbose=False,
    )
    print(f"Exported {output_dir}/cnhubert.onnx")

    # Export GPT-SoVITS models (t2s_encoder, t2s_fsdec, t2s_sdec, vits)
    gpt_sovits.export(
        ref_seq,
        text_seq,
        ref_bert,
        text_bert,
        ref_audio_sr,
        ssl_content,
        output_dir,
    )

    print(f"\nAll models exported to {output_dir}/")
    print("Exported models:")
    print(f"  - {output_dir}/cnhubert.onnx")
    print(f"  - {output_dir}/t2s_encoder.onnx")
    print(f"  - {output_dir}/t2s_fsdec.onnx")
    print(f"  - {output_dir}/t2s_sdec.onnx")
    print(f"  - {output_dir}/vits.onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export GPT-SoVITS v2pro models to ONNX"
    )
    parser.add_argument(
        "--sovits_path",
        type=str,
        required=True,
        help="Path to SoVITS model weights (.pth)",
    )
    parser.add_argument(
        "--gpt_path",
        type=str,
        required=True,
        help="Path to GPT model weights (.ckpt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="onnx_output",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--hubert_path",
        type=str,
        default="GPT_SoVITS/pretrained_models/chinese-hubert-base",
        help="Path to Chinese HuBERT model",
    )

    args = parser.parse_args()

    export(args.sovits_path, args.gpt_path, args.output_dir, args.hubert_path)
