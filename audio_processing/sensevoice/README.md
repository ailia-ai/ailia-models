# SenseVoice

## Input

Audio file

https://huggingface.co/FunAudioLLM/SenseVoiceSmall/tree/main/example

## Output

Recognized speech text
```
うちの中学は弁当制で持っていきない場合は50円の学校販売のパンを買う
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 sensevoice.py
```

If you want to specify the audio, put the file path after the `--input` option.
```bash
$ python3 sensevoice.py --input AUDIO_FILE
```

If you specify the `-V` option, it will be in input mode from the microphone.

```bash
$ python3 sensevoice.py -V
```

By adding the disable_ailia_audio option, you can use the same kaldi as the reference implementation. In that case, installing kaldi is required.
```bash
$ python3 sensevoice.py --disable_ailia_audio
```

```
pip3 install kaldi_native_fbank
```

By adding the `--quantize` option, you can use int4 or int8 quantized models.
```bash
$ python3 sensevoice.py --quantize int4
$ python3 sensevoice.py --quantize int8
```

The quantized models are exported using `export/export_quantized.py`.

## Reference

- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)

## Framework

Pytorch

## Model Format

ONNX opset=14

## Netron

[speech_fsmn_vad_zh-cn-16k-common.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sensevoice/speech_fsmn_vad_zh-cn-16k-common.onnx.prototxt)  
[sensevoice_small.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sensevoice/sensevoice_small.onnx.prototxt)  
