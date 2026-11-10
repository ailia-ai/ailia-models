# Open MusicLM

Text-to-music generation. An open implementation of MusicLM.

## Input

Text prompt describing the music, e.g.

```
song with synths and flute
```

## Output

Generated audio (`output.wav`, 24 kHz mono)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the default prompt,
```bash
$ python3 open-musiclm.py
```

If you want to specify the prompt, put the text after the `--input` option.
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 open-musiclm.py --input "piano sonata waltz, glittery" --savepath output.wav
```

The length of the generated audio is specified with `--duration` (seconds, default 4, minimum 4).
Longer audio is generated with sliding windows (10 s semantic / 4 s coarse / 2 s fine).
```bash
$ python3 open-musiclm.py --duration 8
```

Generation is stochastic; use `--seed` to change the result.
`--return_coarse_wave` skips the fine stage and decodes the audio from the coarse tokens only (faster, lower quality).

## Pipeline

1. `open_musiclm_clap_text` : text → CLAP text embedding → 12 tokens (residual VQ)
2. `open_musiclm_semantic` : CLAP tokens → semantic tokens (50 Hz), autoregressive
3. `open_musiclm_coarse` : CLAP + semantic tokens → coarse acoustic tokens (75 Hz × 3 quantizers), autoregressive
4. `open_musiclm_fine` : CLAP + coarse tokens → fine acoustic tokens (75 Hz × 5 quantizers), autoregressive
5. `open_musiclm_encodec_decoder` : 8 acoustic tokens per frame → waveform (EnCodec 24 kHz, 6 kbps)

The transformers do not use a kv cache; every generated token requires a full forward pass over the whole sequence
(about 1,850 transformer passes for 4 seconds of audio).

## Reference

- [Open MusicLM](https://github.com/zhvng/open-musiclm)
- [CLAP](https://github.com/LAION-AI/CLAP)
- [MERT](https://huggingface.co/m-a-p/MERT-v0)
- [EnCodec](https://github.com/facebookresearch/encodec)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[open_musiclm_clap_text.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/open-musiclm/open_musiclm_clap_text.onnx.prototxt)  
[open_musiclm_semantic.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/open-musiclm/open_musiclm_semantic.onnx.prototxt)  
[open_musiclm_coarse.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/open-musiclm/open_musiclm_coarse.onnx.prototxt)  
[open_musiclm_fine.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/open-musiclm/open_musiclm_fine.onnx.prototxt)  
[open_musiclm_encodec_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/open-musiclm/open_musiclm_encodec_decoder.onnx.prototxt)
