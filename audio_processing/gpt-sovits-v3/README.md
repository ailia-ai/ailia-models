# GPT-SoVITS V3

### Input
- A synthesis text and reference audio and reference text for voice cloning

### Output
The Voice file is output as .wav which path is defined as `SAVE_WAV_PATH` in `gpt-sovits-v3.py `.

### Requirements
This model requires pyopenjtalk for g2p.

```
pip3 install -r requirements.txt
```

### Usage
Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample sentence and sample audio,
```
python3 gpt-sovits-v3.py 
```

Run with audio prompt.

```
python3 gpt-sovits-v3.py -i "ax株式会社ではAIの実用化のための技術を開発しています。" --ref_audio reference_audio_captured_by_ax.wav --ref_text "水をマレーシアから買わなくてはならない。"
```

Run for english.

```
python3 gpt-sovits-v3.py -i "Hello world. We are testing speech synthesis." --text_language en --ref_audio reference_audio_captured_by_ax.wav --ref_text "水をマレーシアから買わなくてはならない。" --ref_language ja
```

Run for chinese.

```
python3 gpt-sovits-v3.py -i "你好世界。我们正在测试语音合成。" --text_language zh --ref_audio reference_audio_captured_by_ax.wav --ref_text "水をマレーシアから買わなくてはならない。" --ref_language ja
```

### Reference
[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

### Framework
PyTorch 2.5.0

### Model Format
ONNX opset = 17

### Netron

#### Normal model

- [cnhubert.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v3/cnhubert.onnx.prototxt)
- [t2s_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v3/t2s_encoder.onnx.prototxt)
- [t2s_fsdec.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v3/t2s_fsdec.onnx.prototxt)
- [t2s_sdec.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v3/t2s_sdec.opt.onnx.prototxt)
- [vq_model.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v3/vq_model.onnx.prototxt)
- [vq_cfm.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v3/vq_cfm.onnx.prototxt)
- [bigvgan_model.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v3/bigvgan_model.onnx.prototxt)
- [chinese-roberta.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits-v3/chinese-roberta.onnx.prototxt)

