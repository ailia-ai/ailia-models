# MioTTS

## Input

Text string (Japanese or English) for text-to-speech synthesis.

```
Hello, World.
```

## Output

WAV audio file synthesized at 44.1 kHz (default: `output.wav`).

## Requirements

This model requires additional modules.

```
pip3 install -r requirements.txt
```

The `requirements.txt` installs: `transformers`, `soundfile`, `jinja2`.

## Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the default English input,

```bash
$ python3 mio-tts.py
```

If you want to specify the input text, use the `--input` option.

```bash
$ python3 mio-tts.py --input "Hello, World."
```

For Japanese input,

```bash
$ python3 mio-tts.py --input "日本語の音声合成をテストしています。"
```

You can use `--savepath` to change the output file path.

```bash
$ python3 mio-tts.py --input "Hello, World." --savepath result.wav
```

To select a voice preset explicitly,

```bash
$ python3 mio-tts.py --input "Hello, World." --preset_id en_female
```

Available preset IDs: `jp_female`, `jp_male`, `en_female`, `en_male`.
When `--preset_id` is not specified, the preset is selected automatically based on the detected language (`jp_female` for Japanese, `en_male` for English).

To control generation parameters,

```bash
$ python3 mio-tts.py --input "Hello, World." --temperature 0.9 --top_p 0.95 --seed 42
```

To use greedy decoding (deterministic),

```bash
$ python3 mio-tts.py --input "Hello, World." --greedy
```

## Reference

- [MioTTS-0.1B](https://huggingface.co/Aratako/MioTTS-0.1B)
- [MioCodec-25Hz-44.1kHz-v2](https://huggingface.co/Aratako/MioCodec-25Hz-44.1kHz-v2)

## Framework

Pytorch

## Model Format

ONNX opset=17

<!-- ## Netron

- [miotts_llm_prefill.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mio-tts/miotts_llm_prefill.onnx.prototxt)
- [miocodec_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mio-tts/miocodec_decoder.onnx.prototxt)
- [miocodec_global_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mio-tts/miocodec_global_encoder.onnx.prototxt) -->
