# Qwen3-TTS ONNX export

Scripts that produce the ONNX and prototxt files used by `../qwen3-tts.py` from
the official Qwen3-TTS 12Hz Base checkpoints.

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

Everything is exported as float32, which comes to about 4.5GB of output for
0.6B and about 8.7GB for 1.7B.

`onnx2prototxt.py` is downloaded from
[ailia-ai/export-to-onnx](https://github.com/ailia-ai/export-to-onnx) on first
use and generates the `.prototxt` next to every `.onnx`.

## Verify

`verify_onnx.py` runs every exported graph through onnxruntime and compares it
with the PyTorch reference module. The talker is checked for a prefill and a
decode step with a KV cache, the code predictor for all 15 code groups of a
frame, the codec tables row block by row block, and the speech tokenizer at two
different lengths:

```bash
python3 verify_onnx.py --parameter_num 1.7B --onnx_dir .
```

## Files

`<p>` is the parameter_num, `H` the talker hidden size (1024 for 0.6B, 2048 for
1.7B) and `Hs` the code predictor hidden size (1024 for both).

| file | input | output |
|---|---|---|
| `qwen3_tts_encoder_<p>.onnx` | waveform `[1, 1, L]`, mel `[1, frames, 128]` | audio codes `[1, 32, T]`, speaker embedding `[1, H]` |
| `qwen3_tts_prompt_<p>.onnx` | text token ids `[1, n]`, codec tag ids `[1, m]`, reference codes `[1, 16, T]` | projected text `[1, n, H]`, codec embeddings `[1, m, H]`, summed reference frames `[1, T, H]` |
| `qwen3_tts_talker_<p>.onnx` | hidden states `[1, seq, H]`, 4D mask, position ids, KV cache | codec logits `[1, 1, 3072]`, hidden state `[1, 1, H]`, KV cache |
| `qwen3_tts_code_predictor_<p>.onnx` | hidden states `[1, seq, H]`, head rows `[2048]`, 4D mask, position ids, KV cache | code group logits `[1, 1, 2048]`, KV cache |
| `qwen3_tts_tokenizer_decoder_<p>.onnx` | audio codes `[B, 16, T]` | waveform `[1, 1, L]` |
| `qwen3_tts_codec_tables_<p>.npy` | | the 16 codec embedding tables, `[3072 + 15 * 2048, H]` |

Every matmul is part of a graph, including the text projection and all 16 output
heads, so `../qwen3-tts.py` only gathers rows of `codec_tables`, adds them up and
samples tokens. Keeping the matmuls out of the runtime is not only tidier: numpy's
BLAS keeps its threads spinning after a matmul and starves ailia on the next call,
which cost about 6x on a code predictor step when the output heads were still on
the Python side.

Notes:

- The talker and the code predictor take their KV cache as plain inputs and
  return it as plain outputs (`past_pkv_*` / `present_pkv_*`, two per layer), so
  `../qwen3-tts.py` can drive the auto regressive loop from Python. The number of
  cached layers is read back from the ONNX input count at runtime rather than from
  the config.
- **The codec embedding tables cannot live in the two decode loop graphs.** They
  did, which removed the npy entirely, but on ailia 1.6.1 a table lookup inside
  those graphs returns rows from an earlier call once the loop is a few steps in:
  the code predictor's first two calls match onnxruntime to 1e-4 and every call
  after that is wrong by ~1e+1, and the talker behaves the same way. The lookup is
  correct in isolation, correct as a graph output, and correct on the first calls,
  so the graphs keep every matmul and hand the lookup back to the runtime. With
  the embedding passed in through `inputs_embeds` instead, all 15 code predictor
  calls match onnxruntime to 2e-3, including the 15 different `head_rows`, which
  is why the output heads could stay.
- The code predictor picks its output head from `head_rows`, the 2048 rows of the
  combined head matrix that step needs, rather than deriving them from
  `position_ids`. Indices that come straight from an input are what the other
  table lookups here use as well.
- Weights only have to be stored in a separate `.onnx.data` file when the model
  does not fit in the 2GB protobuf limit. That is the case for `prompt` (the text
  embedding table alone is over 1GB), `tokenizer_decoder` (exported with
  `dynamo=True`, which always externalizes) and the 1.7B `talker`.
- **A model using external data has to keep at least one initializer inline.**
  ailia reads every weight as zero when all of them live in the data file, which
  silently produces zero outputs (and NaN once a graph is run twice under
  CPU-IntelMKL). `external_data_threshold()` derives the `size_threshold` from
  the model so the smallest initializer always stays in the ONNX itself. Note
  that onnx compares `sys.getsizeof(raw_data)`, the payload plus the bytes object
  overhead, against that threshold.
- The exporter also writes weights folded into a Constant node to their own
  external file. Those are attribute tensors rather than initializers, so
  `consolidate_external_data()` walks node attributes as well when it collects the
  files to merge and delete; missing them leaves a few hundred MB of duplicates
  next to the model.
- The talker uses an mRoPE with three sections, but Qwen3-TTS Base gives all
  three the same position id, which makes it equivalent to the plain rotary
  embedding. The exported graph therefore takes 2D `[B, seq]` position ids.
- The speech tokenizer weights are identical for both sizes, so the two
  `qwen3_tts_tokenizer_decoder_*.onnx` have the same content. They are exported
  per size to keep the runtime file names uniform.
- The file names differ from the first published set
  (`speaker_encoder`, `tokenizer_encoder`, `talker_io_units`, `talker_decoder`,
  `subtalker_decoder` and its npy tables), whose split does not match these
  graphs. `tokenizer_decoder` is unchanged and keeps its name.

## Upload

The generated files go to the `qwen3-tts/` folder of the
[ailia-models bucket](https://console.cloud.google.com/storage/browser/ailia-models).
