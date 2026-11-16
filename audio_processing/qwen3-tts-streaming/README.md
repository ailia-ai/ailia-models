# Qwen3-TTS streaming

Text in, audio out, both incremental: the text is fed in as it is produced (for
example by an LLM) and speech comes out while the text is still arriving.

## Input

- Text (a text file or a string, fed in pieces at `--cps` characters per second)
- For the voice clone (`-m base`): a reference audio (WAV) and, for the
  in-context clone, its transcript

## Output

- Synthesized audio, played on the default audio device with `--play` and
  saved to `--savepath`

## Requirements

- Python 3.10 or higher
- [ailia SDK](https://ailia.jp/sdk/) **1.7.0 or higher** (the fused code
  predictor and the CUDA graph segments it relies on), or onnxruntime with
  `--onnx`
- With `--coreml`: macOS 15 or later and `coremltools` (Core ML on the Apple
  Neural Engine)

```bash
pip install -r requirements.txt
```

`sounddevice` is only needed for `--play`. For `--onnx`:

```bash
pip3 install onnxruntime-gpu   # or onnxruntime
```

For `--coreml`:

```bash
pip3 install coremltools
```

## Usage

The sample follows the ailia-models convention for the environment: `-e`
selects the ailia environment, and with `--onnx` the onnxruntime provider
follows it (CUDA for a GPU environment, otherwise CPU). On a machine with a
GPU, pass the cuDNN environment id, e.g. `-e 2` (the ids are listed with
`python3 -c "import ailia; print(ailia.get_environment_list())"`). The CPU is
far from real time (RTF about 5).

Speak a text file as it is fed in (the default is `input.txt`, 293 Japanese
characters at 30 characters per second), echoing the text to the console as it
goes in:

```bash
python3 qwen3-tts-streaming.py -e 2 --play
```

A string works too:

```bash
python3 qwen3-tts-streaming.py -i "ご注文を承りました。少々お待ちください。" -s output.wav
```

Voice clone with the Base model: the reference audio and its transcript go
into the prompt (in-context clone). The first `reference frames + 1 -
transcript tokens` tokens of each utterance have to be known before speech
can start (23 tokens for the bundled `reference.wav`), so the first audio
comes later than with a predefined speaker.

```bash
python3 qwen3-tts-streaming.py -m base --ref_audio reference.wav --ref_text "かしこまりました。" --play
```

With an empty `--ref_text` only the speaker embedding of the reference is
used (Qwen3-TTS's `x_vector_only_mode`): no transcript, the latency of a
predefined speaker, lower similarity.

```bash
python3 qwen3-tts-streaming.py -m base --ref_audio reference.wav --ref_text "" --play
```

On a Mac, `--coreml` runs the same models converted to Core ML: the talker,
the code predictor and the decoder on the Neural Engine in fp16, the prompt and
the codec tables on the CPU. The reference encoder of `-m base` stays ONNX
(ailia, or onnxruntime with `--onnx`); it runs once per voice.

```bash
python3 qwen3-tts-streaming.py --coreml --play
python3 qwen3-tts-streaming.py --coreml -m base --ref_audio reference.wav --ref_text "かしこまりました。" --play
```

Replay the token stream of an LLM (Qwen2.5 token deltas at `--rate` tokens per
second; needs `transformers` and the Qwen2.5-1.5B-Instruct tokenizer):

```bash
python3 qwen3-tts-streaming.py --replay texts.json --rate 30 --results results.jsonl -b
```

### Options

- `-i`, `--input` text file or string (default: `input.txt`)
- `-m`, `--model` `custom_voice` (predefined speakers, default) or `base` (voice clone)
- `-p`, `--parameter_num` `0.6B` (default) or `1.7B`
- `--speaker` CustomVoice speaker: `Ono_Anna` (default, Japanese), `Vivian`,
  `Serena`, `Ryan`, `Aiden`, `Sohee`, `Eric`, `Dylan`, `Uncle_Fu`
- `--language` `Japanese` (default), `Auto`, `chinese`, `english`, `korean`,
  `german`, `french`, `russian`, `portuguese`, `spanish`, `italian`
- `--ref_audio`, `--ref_text` reference for `-m base` (see above)
- `--commit` when text becomes available to the talker: `token` (default:
  per talker token, 3 held back) or `punct` (at spaces and 、。！？ etc.)
- `--cps`, `--piece` feed rate of the input text (default 30 characters per
  second, 2 characters per push)
- `--max_chars` the text is spoken as utterances of whole sentences up to
  this length (default 150; the talker's fixed KV buffer holds about 40 s)
- `--play` play on the default audio device; `--prebuffer` seconds of audio
  to queue before playback starts (default 0.5)
- `--emit_every` frames per audio chunk (default 8 = 0.64 s); `--decode_window`
  frames the decoder sees per chunk (default 80); `--decode_select` frames it
  renders per chunk (default 16, a shape of the decoder model)
- `--temperature`, `--top_k`, `--subtalker_temperature`, `--repetition_penalty`
  sampling (defaults 0.9 / 50 / 0.9 / 1.05, as in the reference implementation;
  the code predictor's top-k of 50 is a constant of its graph, and the penalty
  applies to the talker's generated group-0 tokens)
- `--results` append one JSON line of timings per utterance
- `--coreml` run on Core ML (the Apple Neural Engine) instead of ailia
- `--onnx` run on onnxruntime instead of ailia; `--cuda_graph` which
  fixed-shape graphs onnxruntime replays as CUDA graphs (default
  `talker,code_predictor`)
- `-b` report the time spent in each model, `--debug` per-utterance timings
  and a NaN check on every model output

## How it works

Qwen3-TTS generates 12.5 codec frames per second; the talker consumes one
text token per frame and is simply not stepped while it has caught up with
the text, so speech follows the text as it arrives. The talker's KV cache is a
fixed buffer of 512 positions updated in place; the text of each utterance is
fed one position at a time, every call has the same shapes, and ailia (1.7:
`AILIA_ENABLE_DNN_SEGMENT`, set by the sample) and onnxruntime (CUDA graph)
replay the whole network as one graph. The 15 code groups of a frame come
from one fused graph that also samples them. The decoder always sees an
80-frame window (right padded while the utterance is short): the whole window
goes through its transformer, but only the last 16 frames (the 8 to emit and 8
before them) through the 480x upsampling, which gives the same samples for a
fifth of the work.

An utterance ends when the talker draws EOS, or when EOS is its most likely
token: an fp16 talker leaves EOS slightly less probable, and an utterance that
misses it would run on in silence. As in the reference implementation, only
`codec_eos` ends an utterance (the other special codec tokens, `think_eos`
included, are suppressed) and EOS is not allowed in the first two frames.

The prompt prefix of a voice (role, codec tags, speaker, and for the
in-context clone the reference text on its codec frames) is prefilled once;
the KV buffer keeps it across utterances of the same voice and language, so
each utterance only prefills its own text.

On an RTX 3080 (ailia SDK 1.7.0, cuDNN) the 0.6B CustomVoice model speaks the
bundled `input.txt` with the first audio 1.1 s after the first character and
an RTF of about 0.5, the 1.7B model at an RTF of about 0.55 in 6 GB of GPU
memory; onnxruntime is within a few percent of that. On an Apple M2 with
`--coreml` the 0.6B Base model speaks it at an RTF of about 0.9.

## Models

The models are the ONNX split of [qwen3-tts](../qwen3-tts/) exported from the
CustomVoice (`*_custom_voice`) and Base checkpoints. The talker and the code
predictor are pure fp16 graphs (fp32 inputs and outputs); the prompt, the codec
tables and the decoder keep their weights and MatMul / Conv in fp16 and the
rest in fp32; the encoder stays fp32 (its outputs are codebook indices, which
fp16 flips). `<p>` is `0.6B` or `1.7B`; the Base files have no
`_custom_voice`.

| File | Role |
|---|---|
| `qwen3_tts_prompt_<p>_custom_voice_fp16.onnx` | text tokens -> talker hidden |
| `qwen3_tts_codec_embedding_<p>_custom_voice_fp16.onnx` | the 16 codec tables (read out once) |
| `qwen3_tts_talker_<p>_custom_voice_static_fp16.onnx` | talker with a fixed 512-position KV buffer (1.7B: weights in `.onnx.data`) |
| `qwen3_tts_code_predictor_<p>_custom_voice_fp16.onnx` | groups 1..15 of a frame in one call, sampling inside |
| `qwen3_tts_decoder_window_<p>_fp16.onnx` | codec frames -> waveform (one file for both models: the speech tokenizer is the same) |
| `qwen3_tts_encoder_<p>.onnx` | Base only: reference audio -> codec frames + speaker embedding (fp32) |

With `--coreml` the same split is read as Core ML (`.mlmodelc`, downloaded
zipped from `qwen3-tts-streaming/coreml/`): `qwen3_tts_<name>_<p>[_custom_voice].mlmodelc`
for `prompt`, `codec_embedding`, `code_predictor`, `decoder_window`, and the
talker in two files `talker_<p>[_custom_voice]_0` / `_1` (one file does not
load on the Neural Engine). The talker keeps its KV cache in a Core ML state
(1024 positions). Only 0.6B is converted so far.

The exporters (the ONNX split, the fused predictor, the fixed buffer talker with
its shape arithmetic folded, the fp16 conversion, and the Core ML conversion)
are maintained outside this repository.

## Reference

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS-streaming-input](https://github.com/keless/Qwen3-TTS-streaming-input) (the live text streaming scheme)

## Framework

Pytorch

## Model Format

ONNX opset=23 (talker, code predictor), 17 (the others); Core ML (mlprogram, iOS 18 / macOS 15)

## Netron

- [qwen3_tts_talker_0.6B_custom_voice_static_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/qwen3-tts-streaming/qwen3_tts_talker_0.6B_custom_voice_static_fp16.onnx.prototxt)
- [qwen3_tts_code_predictor_0.6B_custom_voice_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/qwen3-tts-streaming/qwen3_tts_code_predictor_0.6B_custom_voice_fp16.onnx.prototxt)
- [qwen3_tts_decoder_window_0.6B_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/qwen3-tts-streaming/qwen3_tts_decoder_window_0.6B_fp16.onnx.prototxt)
