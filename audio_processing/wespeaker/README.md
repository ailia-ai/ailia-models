# WeSpeaker

## Input

Two audio files.
```
Example
input1: examples/00001_spk1.wav
input2: examples/00024_spk1.wav
```

(Wav file from https://huggingface.co/spaces/wenet/wespeaker_demo)

## Output

Degree of similarity.
```
The speakers are 84.8% similar
Welcome, human!
```

## Requirements
This model recommends additional module.

```
pip3 install torch torchaudio
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample audio,
```bash
$ python3 wespeaker.py --input1 examples/00001_spk1.wav --input2 examples/00024_spk1.wav
```

Specify two audio files with the `--input1` and `--input2` options.

By specifying the `--english` option, it uses the English model
and the `--chinese` option for the Chinese.
By specifying the `--multilingual` option, it uses the multilingual model
(VoxBlink2 SimAMResNet34), which is recommended for Japanese and other languages.

```bash
$ python3 wespeaker.py --input1 AUDIO_FILE1 --input2 AUDIO_FILE2 --english
$ python3 wespeaker.py --input1 AUDIO_FILE1 --input2 AUDIO_FILE2 --multilingual
```

## Reference

- [WeSpeaker](https://github.com/wenet-e2e/wespeaker)
- [WeSpeaker Pretrained Models](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
- [Hugging Face - Speaker Verification in WeSpeaker](https://huggingface.co/spaces/wenet/wespeaker_demor)
- [KaldiFeat](https://github.com/yuyq96/kaldifeat)

## Framework

Onnxruntime

## Model Format

ONNX opset=14 (voxceleb_resnet34, cnceleb_resnet34), opset=17 (voxblink2_samresnet34)

## Netron

[voxceleb_resnet34.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/wespeaker/voxceleb_resnet34.onnx.prototxt)  
[cnceleb_resnet34.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/wespeaker/cnceleb_resnet34.onnx.prototxt)  
[voxblink2_samresnet34.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/wespeaker/voxblink2_samresnet34.onnx.prototxt)  
