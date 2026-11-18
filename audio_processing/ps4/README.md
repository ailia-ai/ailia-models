# PS4

## Input

Audio file

- Mixed audio (`mixed.wav`)
- Enrollment audio of the speaker to extract (`ref-voice.wav`)

Input an audio file that is spoken by multiple people and an audio file that contains the voice of the person you want to extract.
The voice of one person is extracted and output.

(Audio from the LibriSpeech demo set of [VoiceSplit](https://github.com/Edresson/VoiceSplit))

## Output

Audio file (`output.wav`)

- Estimated audio of the target speaker

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 ps4.py --input mixed.wav --reference_file ref-voice.wav
```

If you want to specify the mixed audio, put the file path after the `--input` option, and to specify the enrollment audio, put the file path after the `--reference_file` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 ps4.py --input MIXED_WAV --reference_file REFERENCE_WAV --savepath SAVE_PATH
```

The input audio is resampled to 16 kHz.

## Reference

- [PS4](https://github.com/TaurenMountain/PS4)
- [TaurenMountain/PS4](https://huggingface.co/TaurenMountain/PS4)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[ps4_spk_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/ps4/ps4_spk_encoder.onnx.prototxt)  
[ps4_bsrnn.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/ps4/ps4_bsrnn.onnx.prototxt)
