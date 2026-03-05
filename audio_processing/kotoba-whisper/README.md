# Kotoba-Whisper

## Input

Audio file

https://github.com/ailia-ai/ailia-models/assets/29946532/b0b83413-cb15-4414-afab-46c4bf71340d

## Output

Recognized speech text
```
ちょっと見ていきましょう。
```

## Requirements

This model requires additional module.
```
pip3 install librosa
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 kotoba-whisper.py
```

If you want to specify the audio, put the file path after the `--input` option.
```bash
$ python3 kotoba-whisper.py --input AUDIO_FILE
```

This `--chunk_length` option should be used when a single large audio file is being transcribed.
```bash
$ python3 kotoba-whisper.py --chunk_length CHUNK_LENGTH
```


## Reference

- [Hugging Face - Kotoba-Whisper](https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[kotoba-whisper-v1.0_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/kotoba-whisper/kotoba-whisper-v1.0_encoder.onnx.prototxt)  
[kotoba-whisper-v1.0_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/kotoba-whisper/kotoba-whisper-v1.0_decoder.onnx.prototxt)  
