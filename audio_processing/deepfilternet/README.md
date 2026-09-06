# DeepFilterNet

## Input

Audio file (`noisy_snr0.wav`)

- Noisy speech, 48 kHz

## Output

Audio file (`output.wav`)

- Speech with the background noise suppressed

(Audio from the assets of [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet))

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 deepfilternet.py --input noisy_snr0.wav
```

If you want to specify the audio, put the file path after the `--input` option.
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 deepfilternet.py --input NOISY_WAV --savepath SAVE_PATH
```

The model can be selected with the `--model_type` option. `DeepFilterNet3` gives the best
quality of the three and is used by default.

```bash
$ python3 deepfilternet.py --model_type DeepFilterNet2
```

The `--pf` option enables the post-filter, which slightly over-attenuates very noisy sections.
```bash
$ python3 deepfilternet.py --pf
```

For `DeepFilterNet` and `DeepFilterNet2` the post-filter is applied to the ERB mask inside the
model, so a separate `_pf` onnx is downloaded. For `DeepFilterNet3` it is applied to the output
of the model and has no weights, so it is calculated in this script instead.

The `--atten_lim` option limits the noise attenuation in dB by mixing the enhanced signal back
with the noisy signal. For example `12` only suppresses 12 dB and keeps the remaining noise.
```bash
$ python3 deepfilternet.py --atten_lim 12
```

The model works on 48 kHz audio. Other sampling rates are resampled to 48 kHz before the
inference and back to the original rate before saving. Multi channel audio is processed
channel by channel.

## Reference

- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[deepfilternet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/deepfilternet/deepfilternet.onnx.prototxt)  
[deepfilternet2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/deepfilternet/deepfilternet2.onnx.prototxt)  
[deepfilternet3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/deepfilternet/deepfilternet3.onnx.prototxt)  
[deepfilternet_pf.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/deepfilternet/deepfilternet_pf.onnx.prototxt)  
[deepfilternet2_pf.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/deepfilternet/deepfilternet2_pf.onnx.prototxt)
