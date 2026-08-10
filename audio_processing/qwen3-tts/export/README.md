# Qwen3-TTS ONNX export

Scripts that produce the ONNX / prototxt / npy files used by `../qwen3-tts.py`
from the official Qwen3-TTS 12Hz Base checkpoints.

| parameter_num | Hugging Face model |
|---|---|
| `0.6B` | [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) |
| `1.7B` | [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) |

## Setup

```bash
# CPU wheels are enough, the export does not need a GPU
pip install torch==2.10.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install --no-deps qwen-tts==0.1.1
```

`qwen-tts` is the official inference package and provides the reference
implementation the export is built on. It is installed with `--no-deps` because
two of its dependencies are only needed by parts that are not exported here:
`gradio` for its demo, and `pysox` (which does not build everywhere) for its 25Hz
tokenizer. The export scripts stub `pysox` out. Its remaining dependencies are in
`requirements.txt`; note that it pins `transformers` to 4.57.3, which is older
than the version `../requirements.txt` uses at runtime.

## Export

```bash
python3 export_onnx.py --parameter_num 0.6B
python3 export_onnx.py --parameter_num 1.7B
```

The checkpoint is downloaded from the Hub unless `--model_dir` points at a local
snapshot. `--output_dir` selects where the files are written (default: the
current directory). Each module is exported in a separate process, since the
1.7B talker needs about twice its weight size in RAM while ONNX serializes it.

Everything is exported as float32, which comes to about 4GB of output for 0.6B
and about 8GB for 1.7B.

`onnx2prototxt.py` is downloaded from
[ailia-ai/export-to-onnx](https://github.com/ailia-ai/export-to-onnx) on first
use and generates the `.prototxt` next to every `.onnx`.

## Verify

`verify_onnx.py` runs every exported graph through onnxruntime and compares it
with the PyTorch reference module. The talker and the code predictor are checked
for both prefill and one decode step with a KV cache, and the speech tokenizer is
checked at two different lengths:

```bash
python3 verify_onnx.py --parameter_num 1.7B --onnx_dir .
```

## Files

`<p>` is the parameter_num, `H` the talker hidden size (1024 for 0.6B, 2048 for
1.7B) and `Hs` the code predictor hidden size (1024 for both).

| file | input | output |
|---|---|---|
| `qwen3_tts_speaker_encoder_<p>.onnx` | mel `[B, frames, 128]` | speaker embedding `[B, H]` |
| `qwen3_tts_tokenizer_encoder_<p>.onnx` | waveform `[1, 1, L]` | audio codes `[1, 32, T]` |
| `qwen3_tts_tokenizer_decoder_<p>.onnx` | audio codes `[B, 16, T]` | waveform `[1, 1, L]` |
| `qwen3_tts_talker_io_units_<p>.onnx` | text features `[B, seq, 2048]`, talker hidden `[B, 1, H]` | projected text `[B, seq, H]`, codec logits `[B, 1, 3072]` |
| `qwen3_tts_talker_decoder_<p>.onnx` | talker hidden states + 4D mask + position ids + KV cache | hidden states + KV cache |
| `qwen3_tts_subtalker_decoder_<p>.onnx` | talker hidden states + 4D mask + position ids + KV cache | hidden states + KV cache |
| `qwen3_tts_text_embedding_<p>.npy` | | `[text_vocab_size, 2048]` |
| `qwen3_tts_codec_embeddings_<p>.npy` | | `[1, 3072, H]` |
| `qwen3_tts_subtalker_lm_heads_<p>.npy` | | `[15, 2048, Hs]` |
| `qwen3_tts_subtalker_codec_emb_<p>.npy` | | `[15, 2048, H]` |

Notes:

- The talker and the code predictor take their KV cache as plain inputs and
  return it as plain outputs (`past_pkv_*` / `present_pkv_*`, two per layer), so
  `../qwen3-tts.py` can drive the auto regressive loop from Python. The number of
  cached layers is read back from the ONNX input count at runtime instead of from
  the config, because the published `qwen3_tts_talker_decoder_0.6B.onnx` was
  exported with a cache for the first 24 of its 28 layers, while these scripts
  cache all 28.
- Weights only have to be stored in a separate `.onnx.data` file when the model
  does not fit in the 2GB protobuf limit, which is the case for the 1.7B talker
  only. The published 0.6B files additionally use external data for the speaker
  encoder and the talker IO units, and `../qwen3-tts.py` downloads exactly that
  set of files, so `PUBLISHED_EXTERNAL_DATA` in `export_onnx.py` keeps a
  re-exported 0.6B on the same layout.
- The talker uses an mRoPE with three sections, but Qwen3-TTS Base gives all
  three the same position id, which makes it equivalent to the plain rotary
  embedding. The exported graph therefore takes 2D `[B, seq]` position ids.
- The embedding tables are exported as npy instead of ONNX because they are only
  used as lookup tables on the Python side. `codec_embeddings` keeps a leading
  axis of 1 so that the runtime can index the group 0 table as
  `codec_embeddings[0]`. The published `qwen3_tts_codec_embeddings_0.6B.npy` has
  16 entries on that axis, of which only the first is the talker table and the
  rest are unused; both shapes work with `../qwen3-tts.py`.
- The speech tokenizer weights are identical for both sizes, so the two
  `qwen3_tts_tokenizer_{encoder,decoder}_*.onnx` pairs have the same content.
  They are exported per size to keep the runtime file names uniform.

## Upload

The generated files go to the `qwen3-tts/` folder of the
[ailia-models bucket](https://console.cloud.google.com/storage/browser/ailia-models).
