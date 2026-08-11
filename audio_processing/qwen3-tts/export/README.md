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

## Static shape

`--static` rebuilds the two modules that carry a KV cache, as
`qwen3_tts_talker_<p>_static.onnx` and
`qwen3_tts_code_predictor_<p>_static.onnx`:

```bash
python3 export_onnx.py --parameter_num 0.6B --static --max_seq_len 512
```

The cache becomes a fixed length buffer that a step writes at `cache_position`
(one extra input, `[seq_len]` int64) instead of one that grows by a step per
call, so no shape in the graph depends on how many steps have run.
`attention_mask` covers the whole buffer and masks the slots not written yet.

This is about the runtime, not the graph. ailia re-infers the shape of the whole
network on every `set_input_blob_shape`, and a growing cache needs
`2 * num_layers` of them per step -- 56 for the 0.6B talker. With a fixed buffer
`../qwen3-tts.py` sets the shapes once and never again.

`--max_seq_len` (default 512) is the talker's buffer length and covers the prompt
as well as the generated tokens, so it caps how much audio one call can produce
(512 frames is about 42 s at 12 Hz, of which the prompt takes the reference
audio's length plus 8). It is a trade: attention reads the whole buffer whatever
the current length is, so a buffer much longer than the sequences actually
generated costs time. The code predictor has no such option -- a frame is always
`num_code_groups` positions, so its buffer is exactly 16 long.

That trade is what the two modules show on a CPU, 0.6B fp32, same machine, same
seed, both ending at step 89 (`ailia 1.6.1.45`, buffer 512, prompt 110):

| | growing cache | fixed buffer |
|---|---|---|
| talker | 129.0 ms/call | 170.2 ms/call |
| code predictor | 17.2 ms/call | 12.8 ms/call |
| total | 45014 ms | 42702 ms |

The code predictor gains because its buffer is 16 long, so reading all of it costs
nothing and dropping the per step shape inference is all that is left. The talker
loses on a CPU because 512 slots is 2.6x the 200 it actually fills. A GPU is the
other way round -- the per layer overhead the fixed buffer removes is larger there
and its bandwidth makes the extra slots cheaper -- and `--max_seq_len` is how to
trade the two.

The output is unchanged: greedy decoding through the whole pipeline gives the same
320 samples (20 frames of 16 code groups) with either pair of models, and the
verify below checks both against the same reference.

## Verify

`verify_onnx.py` runs every exported graph through onnxruntime and compares it
with the PyTorch reference module. The talker is checked for a prefill and a
decode step with a KV cache, the code predictor for all 15 code groups of a
frame, the codec embedding for every group and a whole frame, and the speech
tokenizer at two different lengths:

```bash
python3 verify_onnx.py --parameter_num 1.7B --onnx_dir .
```

`--static` checks the fixed cache variants against the same reference, so a
static model has to reproduce what the growing cache produces:

```bash
python3 verify_onnx.py --parameter_num 0.6B --static --onnx_dir .
```

## fp16

`convert_to_fp16.py` halves the exported models. It works on the ONNX rather than
re-exporting, so it needs neither torch nor qwen-tts:

```bash
python3 convert_to_fp16.py --parameter_num 0.6B --onnx_dir .
```

| the set ../qwen3-tts.py downloads | fp32 | fp16 |
|---|---|---|
| 0.6B | 4.31 GB | 2.28 GB |
| 1.7B | 8.37 GB | 4.31 GB |

The fp16 column includes the encoder, which stays fp32.

**On a CPU fp16 is slower on both runtimes**, so the download size is the whole of
what it buys there. One decode call of the 1.7B models, same inputs, same machine:

| | ailia fp32 | ailia fp16 | onnxruntime fp32 | onnxruntime fp16 |
|---|---|---|---|---|
| `code_predictor` | 17.5 ms | 18.5 ms | 6.7 ms | 10.0 ms |
| `talker` | 222.4 ms | 236.6 ms | 141.1 ms | 144.1 ms |

**The two runtimes do not agree on what fp16 costs in accuracy, because ailia does
not compute in it.** It returns fp32 values from an fp16 gather to 3e-08 and its
fp16 talker logits sit 1.2e-03 from its fp32 ones, while onnxruntime, which does
compute in fp16, puts the same logits 1.4e-02 apart (2.2e-03 for the code
predictor). So on ailia today fp16 is close to fp32 quality at half the download,
and on a runtime that computes in fp16 the talker's logits move by about 1.4e-02
relative, which is enough to change a sampled token.

End to end on ailia with the same seed, the 0.6B run follows the identical token
sequence: EOS at step 89 as in fp32, waveform 1.4e-03 peak apart, 57.8 dB SNR. The
1.7B run diverges, reaching EOS at 63 rather than 57 -- still speech, but a
different sample from the same distribution. Judge fp16 by listening, not by
comparing waveforms.

What the conversion does and does not touch:

- The graph inputs and outputs stay fp32 (`keep_io_types`), so `../qwen3-tts.py`
  feeds and reads the same arrays either way and `--fp16` only changes paths.
- **The rotary embedding stays in fp32.** Its angle is `inv_freq * position_id` and
  the talker's positions pass 2000, where fp16 spacing is 2.0; rounding an angle in
  radians that coarsely would leave cos and sin unrelated to the position. The 26
  nodes from `position_ids` down to Cos and Sin are kept in fp32 in the talker and
  the code predictor. A Cos or Sin only counts when its ancestry reaches
  `position_ids`, because the decoder's snake activations use Sin on an activation
  and blocking those would leave the whole model in fp32.
- **The encoder is not converted at all.** Its `audio_codes` are codebook indices,
  and in fp16 29 of the 3232 the sample's reference audio produces come out
  different, in codebooks 3 and 5..15 where the residual is small enough for fp16 to
  flip a near tie. Those are the voice prompt, and the saving would be 114MB of a
  4.3GB set.
- The converter is onnxruntime's rather than the onnxconverter-common one the other
  samples here use. onnxconverter-common 1.16.0 raises `'list' object has no
  attribute 'input'` on these graphs as soon as a Cast feeds more than one node, and
  skipping that cleanup leaves a model both runtimes reject for binding one Add to
  both fp16 and fp32.

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

## The ailia ScatterElements bug

`--static` ran into a second one. A fixed length cache is written at
`cache_position`, which `index_copy` says in one `ScatterElements` node, and ailia
1.6.1 writes the right slot when a call writes one position and the wrong ones when
it writes more than one. The prompt writes its whole length in one call, so
`qwen3_tts_talker_0.6B_static.onnx` built that way came back from ailia with logits
47.9 off against onnxruntime on the very first call, while the two runtimes agreed
bit for bit on the growing cache model given the same inputs.

`ailia_scatter_repro.py` is the whole thing in two models of about 2KB, one
`ScatterElements` node against the same write done as a matmul:

```bash
python3 ailia_scatter_repro.py
```

```
scatter.onnx
  seq=1  onnxruntime 0.0e+00   ailia 0.0e+00
  seq=2  onnxruntime 0.0e+00   ailia 4.2e+00
  seq=4  onnxruntime 0.0e+00   ailia 4.8e+00
  seq=8  onnxruntime 0.0e+00   ailia 5.0e+00

onehot.onnx
  seq=1  onnxruntime 0.0e+00   ailia 0.0e+00
  seq=2  onnxruntime 0.0e+00   ailia 0.0e+00
  seq=4  onnxruntime 0.0e+00   ailia 0.0e+00
  seq=8  onnxruntime 0.0e+00   ailia 0.0e+00
```

The expected output is a copy rather than arithmetic, so the error is exact: those
are wrong values, not rounding. `cache_write()` in `export_onnx.py` therefore
writes the cache with the one hot matmul, which costs a few more passes over the
buffer and which both runtimes agree on at every length.

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
| `qwen3_tts_talker_<p>_static.onnx` | as above plus cache position `[seq]`, KV cache `[1, kv, max_seq_len, dim]` | as above, KV cache the same length |
| `qwen3_tts_code_predictor_<p>_static.onnx` | as above plus cache position `[seq]`, KV cache `[1, kv, 16, dim]` | as above, KV cache the same length |

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
