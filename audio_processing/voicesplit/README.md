# VoiceSplit

## Input

Audio file

- Mixed audio (`mixed.wav`)
- Reference audio of the speaker to extract (`ref-voice.wav`)

Input an audio file that is spoken by multiple people and an audio file that contains the voice of the person you want to extract.
The voice of one person is extracted and output.

## Output

Audio file (`output.wav`)

- Estimated audio of the target speaker

(Audio from the LibriSpeech demo set of [VoiceSplit](https://github.com/Edresson/VoiceSplit))

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 voicesplit.py --input mixed.wav --reference_file ref-voice.wav
```

If you want to specify the mixed audio, put the file path after the `--input` option, and to specify the reference audio, put the file path after the `--reference_file` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 voicesplit.py --input MIXED_WAV --reference_file REFERENCE_WAV --savepath SAVE_PATH
```

The five experiments compared in the VoiceSplit report can be selected with the `--model_type` option.
`exp5` is used by default because it scored the best SI-SNRi.

| Model type | Loss | Activation | Speaker encoder | SI-SNRi |
|:---|:---|:---|:---|---:|
| exp1 | MSE | ReLU | GE2E2k | 6.023 |
| exp2 | Power-Law | ReLU | GE2E2k | 5.699 |
| exp3 | SI-SNR + uPIT | ReLU | GE2E2k | 5.661 |
| exp4 | SI-SNR + uPIT | Mish | GE2E2k | 6.491 |
| exp5 | SI-SNR + uPIT | ReLU | GE2E3k | 6.552 |

SI-SNRi is the value reported in Table 2 of the VoiceSplit report (LibriSpeech test set).

```bash
$ python3 voicesplit.py --input MIXED_WAV --reference_file REFERENCE_WAV --model_type exp4
```

The speaker encoder differs between the experiments and is downloaded automatically.
`exp1` to `exp4` use the GE2E2k encoder, which requires a reference audio of 0.8 sec or longer.
`exp5` uses the GE2E3k encoder, which has no such restriction.

The input audio is resampled to 16 kHz.

## Reference

- [VoiceSplit](https://github.com/Edresson/VoiceSplit)
- [Real-Time Voice Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[voicesplit_exp1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/voicesplit/voicesplit_exp1.onnx.prototxt)  
[voicesplit_exp2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/voicesplit/voicesplit_exp2.onnx.prototxt)  
[voicesplit_exp3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/voicesplit/voicesplit_exp3.onnx.prototxt)  
[voicesplit_exp4.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/voicesplit/voicesplit_exp4.onnx.prototxt)  
[voicesplit_exp5.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/voicesplit/voicesplit_exp5.onnx.prototxt)  
[ge2e2k_embedder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/voicesplit/ge2e2k_embedder.onnx.prototxt)  
[ge2e3k_embedder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/voicesplit/ge2e3k_embedder.onnx.prototxt)
