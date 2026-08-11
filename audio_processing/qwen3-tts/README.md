# Qwen3-TTS

## Input

- Input text (String)
- Reference audio (WAV file) - For Voice Clone
- Reference text (String) - Transcript of the reference audio

## Output

- Synthesized audio (WAV file)

## Requirements
- Python 3.12 or higher
- [ailia SDK](https://ailia.jp/sdk/) (Version 1.6.1 or higher recommended)

Install the required Python libraries:
```bash
pip install -r requirements.txt
```

If you use `--disable_ailia_tokenizer` option, this model requires additional module.
```bash
pip3 install transformers
```

If you use `--onnx` option, this model requires additional module.
```bash
pip3 install onnxruntime
```

## Usage

This model supports **Voice Clone (Zero-shot Voice Conversion)** by default. You need to provide a text to synthesize, a reference audio file of the target speaker, and its corresponding text transcript.

The bundled sample reference audio (`clone_2.wav`) and transcript are based on the official Qwen3-TTS Voice Clone example. The original reference audio is published as `clone.wav` at:
`https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav`

For the sample wav,
```bash
$ python3 qwen3-tts.py
```

You can directly pass the text you want to synthesize using the `--input` argument.

```bash
python3 qwen3-tts.py --input "Hello, this is a test of voice cloning." --ref_audio clone_2.wav --ref_text "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you." --savepath output.wav
```

### Model size

Both released Base model sizes are available and selected with
`-p`/`--parameter_num`. The default is `0.6B`.

```bash
python3 qwen3-tts.py -p 1.7B
```

The model files add up to about 4.3GB for `0.6B` and about 8.4GB for `1.7B`.
Every model is a single ONNX except the 1.7B talker, whose weights do not fit
in the 2GB protobuf limit and live in `qwen3_tts_talker_1.7B.onnx.data`.

### Language

By default the language is auto-detected (`--language Auto`). If the synthesized speech is pronounced in the wrong language (for example, Japanese text being read with a Chinese accent), specify the target language explicitly with the `--language` option.

```bash
python qwen3-tts.py --input "こんにちは、今日はいい天気ですね。" --language japanese
```

Supported languages: `Auto` (default), `chinese`, `english`, `japanese`, `korean`, `german`, `french`, `russian`, `portuguese`, `spanish`, `italian`.

### Options

- `-i`, `--input` Direct input text string to synthesize. (e.g. `Hello, this is a test of voice cloning.`)
- `-p`, `--parameter_num` Model size. `0.6B` or `1.7B`. (default: `0.6B`)
- `--ref_audio` Reference audio file path for Voice Clone mode. (e.g. `clone_2.wav`)
- `--ref_text` Reference text file path containing the transcript of the reference audio. (e.g. `clone_2.txt`)
- `--language` Target language for synthesis. `Auto` for auto detection, or one of `chinese`, `english`, `japanese`, `korean`, `german`, `french`, `russian`, `portuguese`, `spanish`, `italian`. (default: `Auto`)
- `--temperature` Sampling temperature for the talker. `0` for greedy decoding. (default: `0.9`, matches the official implementation)
- `--top_k` Top-k sampling for the talker. (default: `50`, matches the official implementation)
- `--repetition_penalty` Repetition penalty for the talker. (default: `1.05`)
- `--subtalker_temperature` Sampling temperature for the subtalker (code predictor). `0` for greedy decoding. (default: `0.9`, matches the official implementation)
- `--subtalker_top_k` Top-k sampling for the subtalker (code predictor). (default: `50`, matches the official implementation)
- `--seed` Random seed for reproducible sampling. (default: `None`)
- `--onnx` Run the models with onnxruntime instead of the ailia SDK.
- `--disable_ailia_tokenizer` Tokenize the text with the transformers tokenizer instead of the ailia tokenizer. The ailia tokenizer reads `tokenizer/vocab.json` and `tokenizer/merges.txt`, which are downloaded with the models; transformers reads the bundled `tokenizer/tokenizer.json`. Both produce the same token ids.
- `--fp16` Use the fp16 models, which halve the download (about 2.3GB for `0.6B` and about 4.3GB for `1.7B`). On a CPU this is slower rather than faster, on the ailia SDK and on onnxruntime alike, so the size is what it buys there; the speed benefit is on a GPU that computes in fp16. The reference audio encoder stays fp32 either way, because its output is codebook indices. See [export/README.md](./export/README.md) for the measurements.
- `-b`, `--benchmark` Report the inference time per model. The talker and the code predictor run once per token, so their line shows the number of calls and the average.
- `--profile` Print the ailia SDK layer profile for every model.
- `-s`, `--savepath` Save path for the output synthesized audio. (default: `output.wav`)

> **Note:** Sampling (`temperature=0.9`, `top_k=50`, `subtalker_temperature=0.9`, `subtalker_top_k=50`) is used by default to match the official implementation. Set both `--temperature 0` and `--subtalker_temperature 0` for deterministic greedy decoding. Use `--seed` to make sampled output reproducible.

## Model Format

ONNX opset = 17 (the decoder uses opset 18)

The ONNX files are generated by [export/export_onnx.py](./export/export_onnx.py);
see [export/README.md](./export/README.md) for the model layout.

## Reference

- [Qwen3-TTS Official Repository](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
- [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)

## Framework

PyTorch
