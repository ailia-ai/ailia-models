# Parakeet TDT

## Input

Audio file

```
Example
input: 2086-149220-0033.wav
```

## Output

Recognized speech text
```
Transcription: Well, I don't wish to see it any more, observed Phebe, turning away her eyes. It is certainly very like the old portrait.
Score: 4257.02490234375
```

If you specify the `--timestamp` option, it outputs segment timestamps.
```
0.4s - 4.64s : Well, I don't wish to see it any more, observed Phebe, turning away her eyes.
4.96s - 7.04s : It is certainly very like the old portrait.
```

## Requirements

This model requires additional module.
```
pip3 install librosa
pip3 install soundfile
pip3 install nemo_toolkit
pip3 install sentencepiece
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 parakeet-tdt.py
```

If you want to specify the audio, put the file path after the `--input` option.
```bash
$ python3 parakeet-tdt.py --input AUDIO_FILE
```

If you want to print segment timestamps, add the `--timestamp` option.
```bash
$ python3 parakeet-tdt.py --input AUDIO_FILE --timestamp
```

If you want to run on onnxruntime, add the `--onnx` option.
```bash
$ python3 parakeet-tdt.py --onnx
```

## Reference

- [Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [NeMo](https://github.com/NVIDIA/NeMo)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[parakeet-tdt-0.6b-v2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/parakeet-tdt/parakeet-tdt-0.6b-v2.onnx.prototxt)  
[parakeet-tdt-0.6b-v2_encoder_projection.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/parakeet-tdt/parakeet-tdt-0.6b-v2_encoder_projection.onnx.prototxt)  
[parakeet-tdt-0.6b-v2_predictor.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/parakeet-tdt/parakeet-tdt-0.6b-v2_predictor.onnx.prototxt)  
[parakeet-tdt-0.6b-v2_joint.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/parakeet-tdt/parakeet-tdt-0.6b-v2_joint.onnx.prototxt)  
