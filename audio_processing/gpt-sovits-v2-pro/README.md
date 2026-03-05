# GPT-SoVITS V2 Pro

### Input
- A synthesis text and reference audio and reference text for voice cloning

### Output
The Voice file is output as .wav which path is defined as `SAVE_WAV_PATH` in `gpt-sovits-v2-pro.py `.

### Requirements
This model requires pyopenjtalk for g2p.

```
pip3 install -r requirements.txt
```

### Usage
Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample sentence and sample audio,
```
python3 gpt-sovits-v2-pro.py
```

Run with audio prompt.

```
python3 gpt-sovits-v2-pro.py -i "ax株式会社ではAIの実用化のための技術を開発しています。" --ref_audio reference_audio_captured_by_ax.wav --ref_text "水をマレーシアから買わなくてはならない。"
```

Run for english.

```
python3 gpt-sovits-v2-pro.py -i "Hello world. We are testing speech synthesis." --text_language en --ref_audio reference_audio_captured_by_ax.wav --ref_text "水をマレーシアから買わなくてはならない。" --ref_language ja
```

### Architecture

GPT-SoVITS V2 Pro uses the following ONNX models:

| Model | Description | Input | Output |
|-------|-------------|-------|--------|
| cnhubert.onnx | Chinese HuBERT for SSL features | ref_audio_16k (1, N) | ssl_content (1, T, 768) |
| t2s_encoder.onnx | T2S encoder | ref_seq, text_seq, ref_bert, text_bert, ssl_content | x, prompts |
| t2s_fsdec.onnx | T2S first-stage decoder | x, prompts, top_k, top_p, temperature, repetition_penalty | y, k, v, y_emb, x_example |
| t2s_sdec.onnx | T2S stage decoder | iy, ik, iv, iy_emb, ix_example, top_k, top_p, temperature, repetition_penalty | y, k, v, y_emb, logits, samples |
| sv.onnx | Speaker Verification (ERes2NetV2) | fbank_feat (1, T, 80) | sv_emb (1, 20480) |
| vits.onnx | VITS synthesizer with v2Pro weights | text_seq, pred_semantic, ref_audio, sv_emb | audio |

The v2Pro architecture differs from v2 by:
- Speaker verification embedding (sv_emb) from ERes2NetV2 model
- gin_channels=1024 (vs 512 in v2), with ge_to512 projection for MRTE
- PReLU activation on combined reference + speaker embeddings

### ONNX Export

To export the VITS model:

```
git clone -b 20250606v2pro https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -r requirements.txt

# Export VITS
python3 /path/to/export/export_vits.py \
    --sovits_path GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth \
    --output vits.onnx

# Export SV model
python3 /path/to/export/export_sv.py \
    --sv_path GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt \
    --output sv.onnx
```

### Reference
[GPT-SoVITS (20250606v2pro)](https://github.com/RVC-Boss/GPT-SoVITS/tree/20250606v2pro)

### Framework
PyTorch 2.10.0

### Model Format
ONNX opset = 17

### Netron

#### Normal model

- [cnhubert.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro/cnhubert.onnx.prototxt)
- [t2s_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro/t2s_encoder.onnx.prototxt)
- [t2s_fsdec.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro/t2s_fsdec.onnx.prototxt)
- [t2s_sdec.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro/t2s_sdec.opt.onnx.prototxt)
- [sv.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro/sv.onnx.prototxt)
- [vits.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro/vits.onnx.prototxt)
