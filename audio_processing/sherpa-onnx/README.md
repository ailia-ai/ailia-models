# sherpa-onnx : Streaming Speech Recognition using Zipformer Transducer

## Input

Audio file (WAV, mono, 16kHz, 16-bit PCM)

## Output

Recognized speech text

## Requirements
This model requires additional module.
```
pip3 install librosa
pip3 install pyaudio  # for microphone input mode
pip3 install onnxruntime  # for --onnx option
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 sherpa-onnx.py
```

If inference fails because the model contains operators that are not supported by the ailia SDK, you can perform inference with ONNX Runtime by
```bash
python3 sherpa-onnx.py --onnx
```
.

If you want to specify the audio, put the file path after the `--input` option.
```bash
python3 sherpa-onnx.py --input AUDIO_FILE
```

If you specify the `-V` option, it will be in input mode from the microphone.

```bash
python3 sherpa-onnx.py -V
```

1. speak into the microphone
2. the recognized text is printed in real time as you speak
3. type `Ctrl+c` to exit


## Reference

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [icefall](https://github.com/k2-fsa/icefall)
- [PengChengStarling](https://github.com/PCL-Voice/PengChengStarling)
- [Hugging Face - sherpa-onnx-streaming-zipformer (ONNX model, token file, sample audio file)](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10/tree/main)

## Framework
PyTorch