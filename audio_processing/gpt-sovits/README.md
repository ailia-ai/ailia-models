# GPT-SoVITS

### Input
- A synthesis text and reference audio and reference text for voice cloning

### Output
The Voice file is output as .wav which path is defined as `SAVE_WAV_PATH` in `gpt-sovits.py `.

### Requirements
This model requires pyopenjtalk for g2p.

```
pip3 install -r requirements.txt
```

### Usage
Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample sentence and sample audio,
```
python3 gpt-sovits.py 
```

Run with audio prompt.

```
python3 gpt-sovits.py -i "音声合成のテストを行なっています。" --ref_audio reference_audio_captured_by_ax.wav --ref_text "水をマレーシアから買わなくてはならない。"
```

Run for english.

```
python3 gpt-sovits.py -i "Hello world. We are testing speech synthesis." --text_language en --ref_audio reference_audio_captured_by_ax.wav --ref_text "水をマレーシアから買わなくてはならない。" --ref_language ja
```

Run for chinese.

```
python3 gpt-sovits.py -i "你好世界。我们正在测试语音合成。" --text_language zh --ref_audio reference_audio_captured_by_ax.wav --ref_text "水をマレーシアから買わなくてはならない。" --ref_language ja
```

### Reference
[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

### Framework
PyTorch 2.1.2

### Model Format
ONNX opset = 17

### Netron

#### Normal model

- [cnhubert.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits/cnhubert.onnx.prototxt)
- [t2s_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits/t2s_encoder.onnx.prototxt)
- [t2s_fsdec.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits/t2s_fsdec.onnx.prototxt)
- [t2s_sdec.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits/t2s_sdec.onnx.prototxt)
- [vits.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits/vits.onnx.prototxt)

#### Chinese BERT model

- [chinese-roberta.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits/chinese-roberta.onnx.prototxt)

#### Optimized model

- [t2s_sdec.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits/t2s_sdec.opt.onnx.prototxt)
- [t2s_sdec.opt2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gpt-sovits/t2s_sdec.opt2.onnx.prototxt)
