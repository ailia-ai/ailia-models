# Demucs Music Source Separation

## Input

Audio file

https://github.com/facebookresearch/demucs/blob/main/test.mp3

## Output

Audio file with separated sound sources

## Requirements

This model requires additional module.
```
pip3 install soundfile
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 demucs.py
```

If you want to specify the audio, put the file path after the `--input` option.
```bash
$ python3 demucs.py --input AUDIO_FILE
```

## Reference

- [Demucs](https://github.com/facebookresearch/demucs)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

- [htdemucs_ft_bass.onnx](https://netron.app/?url=https://storage.googleapis.com/ailia-models/demucs/htdemucs_ft_bass.onnx.prototxt)  
- [htdemucs_ft_drums.onnx](https://netron.app/?url=https://storage.googleapis.com/ailia-models/demucs/htdemucs_ft_drums.onnx.prototxt)  
- [htdemucs_ft_other.onnx](https://netron.app/?url=https://storage.googleapis.com/ailia-models/demucs/htdemucs_ft_other.onnx.prototxt)  
- [htdemucs_ft_vocals.onnx](https://netron.app/?url=https://storage.googleapis.com/ailia-models/demucs/htdemucs_ft_vocals.onnx.prototxt)  
