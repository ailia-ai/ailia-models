# emotion2vec

## Input

Audio file (`test.wav`)

- Speech, 16 kHz mono

(Audio from the `example` directory of [emotion2vec_plus_large](https://huggingface.co/emotion2vec/emotion2vec_plus_large))

## Output

Emotion label with the scores of the 9 classes

```
Emotion: angry
Confidence: 1.0
	angry      1.0000
	surprised  0.0000
	happy      0.0000
	sad        0.0000
	neutral    0.0000
	fearful    0.0000
	disgusted  0.0000
	other      0.0000
	unknown    0.0000
```

## Labels

```
0: angry
1: disgusted
2: fearful
3: happy
4: neutral
5: other
6: sad
7: surprised
8: unknown
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 emotion2vec.py --input test.wav
```

If you want to specify the audio, put the file path after the `--input` option.
```bash
$ python3 emotion2vec.py --input WAV_PATH
```

The model can be selected with the `--model_type` option. `large` is used by default.

| model_type | model | description |
| --- | --- | --- |
| `large` | emotion2vec_plus_large | emotion2vec+ large (~300M) |
| `base` | emotion2vec_plus_base | emotion2vec+ base (~90M) |
| `seed` | emotion2vec_plus_seed | emotion2vec+ seed (~90M) |

```bash
$ python3 emotion2vec.py --model_type base
```

The model works on 16 kHz audio. Other sampling rates are resampled to 16 kHz and multi channel audio is mixed down to mono before the inference.

## Reference

- [emotion2vec](https://github.com/ddlBoJack/emotion2vec)
- [emotion2vec (Hugging Face)](https://huggingface.co/emotion2vec)
- [FunASR](https://github.com/modelscope/FunASR)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[emotion2vec_plus_large.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/emotion2vec/emotion2vec_plus_large.onnx.prototxt)  
[emotion2vec_plus_base.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/emotion2vec/emotion2vec_plus_base.onnx.prototxt)  
[emotion2vec_plus_seed.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/emotion2vec/emotion2vec_plus_seed.onnx.prototxt)
