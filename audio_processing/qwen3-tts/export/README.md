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

Everything is exported as float32, which comes to about 4.3GB of output for
0.6B and about 8.4GB for 1.7B.

`onnx2prototxt.py` is downloaded from
[ailia-ai/export-to-onnx](https://github.com/ailia-ai/export-to-onnx) on first
use and generates the `.prototxt` next to every `.onnx`.

## Verify

`verify_onnx.py` runs every exported graph through onnxruntime and compares it
with the PyTorch reference module. The talker is checked for a prefill and a
decode step with a KV cache, the code predictor for all 15 code groups of a
frame, the codec embedding for every group and a whole frame, and the speech
tokenizer at two different lengths:

```bash
python3 verify_onnx.py --parameter_num 1.7B --onnx_dir .
```

## The ailia gather bug

The split above is shaped by an ailia bug, and two scripts pin it down. Both
compare a model against itself on onnxruntime and on ailia, driving it the way a
decode loop does: a seq=2 prefill, then seq=1 steps with a growing KV cache.

`ailia_gather_check.py` runs the sample models that are published next to the
Qwen3-TTS ONNX. It needs only numpy, onnxruntime and ailia, so it is the one to
hand to someone looking at the SDK:

```bash
python3 ailia_gather_check.py --download
```

```
ailia 1.6.1.45  onnxruntime 1.28.0  env_id 1
  gather_rope  MISMATCH from call 2   worst 1.7e-02
     call 0 0.0e+00  call 1 5.6e-09  call 2 1.3e-02  call 3 1.7e-02  ...
  input_rope   match                  worst 4.7e-09
  gather_attn  match                  worst 7.5e-09
  input_attn   match                  worst 6.5e-09
```

Four models under 100KB each, in two pairs. The `gather_` half of a pair reads its
input embedding from a table inside the graph, indexed by an input; the `input_`
half takes the embedding as an input instead. `_rope` has one attention layer with
rotary embeddings and `_attn` is the same with the rope dropped. Only
`gather_rope` disagrees, so the bug needs both halves: a gather feeding the graph,
and the rotary embeddings. It also needs the KV cache -- a graph that only gathers
is correct however many times it is called, which is what
`qwen3_tts_codec_embedding_<p>.onnx` relies on.

`ailia_gather_repro.py` builds those four models from scratch with torch, and runs
the same comparison:

```bash
python3 ailia_gather_repro.py --stage both
```

## Files

`<p>` is the parameter_num, `H` the talker hidden size (1024 for 0.6B, 2048 for
1.7B) and `Hs` the code predictor hidden size (1024 for both).

| file | input | output |
|---|---|---|
| `qwen3_tts_encoder_<p>.onnx` | waveform `[1, 1, L]`, mel `[1, frames, 128]` | audio codes `[1, 32, T]`, speaker embedding `[1, H]` |
| `qwen3_tts_prompt_<p>.onnx` | text token ids `[1, n]` | projected text `[1, n, H]` |
| `qwen3_tts_codec_embedding_<p>.onnx` | codec table rows `[n, 16]` | their sums `[1, n, H]` |
| `qwen3_tts_talker_<p>.onnx` | hidden states `[1, seq, H]`, 4D mask, position ids, KV cache | codec logits `[1, 1, 3072]`, hidden state `[1, 1, H]`, KV cache |
| `qwen3_tts_code_predictor_<p>.onnx` | hidden states `[1, seq, H]`, head rows `[2048]`, 4D mask, position ids, KV cache | code group logits `[1, 1, 2048]`, KV cache |
| `qwen3_tts_decoder_<p>.onnx` | audio codes `[B, 16, T]` | waveform `[1, 1, L]` |

Every weight is in a graph, including the text projection, the 16 codec embedding
tables and all 16 output heads, so `../qwen3-tts.py` only reshapes arrays and
samples tokens.

`qwen3_tts_codec_embedding_<p>.onnx` holds the 16 tables as one, the talker's 3072
rows first, then the code predictor's 15 tables of 2048, then one all zero row. A
call takes 16 row indices per position and returns their sum, which covers both
callers: a talker step is a whole frame's 16 groups summed, and a code predictor
step is one group with the other 15 pointing at the zero row. It is called ~16
times per audio frame and costs 0.2 ms a call.

Notes:

- The talker and the code predictor take their KV cache as plain inputs and
  return it as plain outputs (`past_pkv_*` / `present_pkv_*`, two per layer), so
  `../qwen3-tts.py` can drive the auto regressive loop from Python. The number of
  cached layers is read back from the ONNX input count at runtime rather than from
  the config.
- **The codec embedding tables cannot live in the two decode loop graphs**, which
  is why they get a model of their own. With them inside, ailia stops following the
  gather's index a few calls into a decode loop: the code predictor's first two
  calls match onnxruntime to 1e-4 and every call after that is wrong by ~1e+1, and
  the talker behaves the same way. See The ailia gather bug above for the sample
  models that isolate it.
- The code predictor picks its output head from `head_rows`, the 2048 rows of the
  combined head matrix that step needs, rather than deriving them from
  `position_ids`. Indices that come straight from an input are what the other
  table lookups here use as well.
- A model whose weights fit in the 2GB protobuf limit is stored as a single ONNX,
  with no `.onnx.data` beside it, which for these models means everything except
  the 1.7B `talker`. Both exporters can leave weights in files of their own -- the
  dynamo one always does -- so `consolidate_external_data()` puts them back in the
  ONNX when they fit and merges them into one data file when they do not. Reading
  the weights from the ONNX itself is also what makes the `prompt` call take 3 ms
  rather than the seconds it took with a 1.27GB sidecar.
- **A model using external data has to keep at least one initializer inline.**
  ailia reads every weight as zero when all of them live in the data file, which
  silently produces zero outputs (and NaN once a graph is run twice under
  CPU-IntelMKL). `external_data_threshold()` derives the `size_threshold` from
  the model so the smallest initializer always stays in the ONNX itself. Note
  that onnx compares `sys.getsizeof(raw_data)`, the payload plus the bytes object
  overhead, against that threshold. Only the 1.7B talker is affected now that
  everything else is a single file.
- The exporter also writes weights folded into a Constant node to their own
  external file. Those are attribute tensors rather than initializers, so
  `consolidate_external_data()` walks node attributes as well when it collects the
  files to merge and delete; missing them leaves a few hundred MB of duplicates
  next to the model.
- The talker uses an mRoPE with three sections, but Qwen3-TTS Base gives all
  three the same position id, which makes it equivalent to the plain rotary
  embedding. The exported graph therefore takes 2D `[B, seq]` position ids.
- The speech tokenizer weights are identical for both sizes, so the two
  `qwen3_tts_decoder_*.onnx` have the same content. They are exported per size to
  keep the runtime file names uniform.
- **Not one file name is shared with the first published set**, so the two
  generations can be told apart by name alone and the older one deleted without
  looking inside anything. That set was `speaker_encoder`, `tokenizer_encoder`,
  `tokenizer_decoder`, `talker_io_units`, `talker_decoder`, `subtalker_decoder` and
  four npy tables, and its split does not match these graphs.
  `qwen3_tts_decoder_<p>.onnx` is the one model whose content did not change:
  it has the same input and output names, dtypes, shapes and opset as the published
  `tokenizer_decoder`, and produces the same waveform to 1.6e-05 on samples in
  [-1, 1], which is fp32 rounding from a slightly different graph decomposition. It
  is renamed anyway so that no name spans both generations.

## Upload

The generated files go to the `qwen3-tts/` folder of the
[ailia-models bucket](https://console.cloud.google.com/storage/browser/ailia-models),
and the four gather bug models to `qwen3-tts/gather_bug/` in the same folder,
which is where `ailia_gather_check.py --download` looks for them.
