# SigLIP2 ONNX Separation

Split the combined SigLIP2 ONNX model (`input_ids` + `pixel_values` -> `logits_per_image`)
into an image encoder and a text encoder.

## Usage

Place the combined onnx model (and its `_weights.pb` for large/giant) in the model
directory (`..`), then run:

```bash
pip install onnx numpy
python separate_onnx.py -m base-patch16-224   # or large-patch16-256, giant-patch16-256
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--model_type` | `base-patch16-224` | Model type: `base-patch16-224`, `large-patch16-256`, `giant-patch16-256` |
| `--input_dir` | `..` | Directory containing the combined onnx model |
| `--output_dir` | `..` | Directory to save the separated onnx models |

## Outputs

- `siglip2-<model>-encode_image.onnx`
  - Input: `pixel_values` `[n, 3, H, W]`
  - Output: `image_embeds` (L2 normalized)
- `siglip2-<model>-encode_text.onnx`
  - Input: `input_ids` `[b, l]`
  - Outputs: `text_embeds` (L2 normalized), `logit_scale` (exp applied), `logit_bias`

If the source model uses external weights (`_weights.pb`), the separated models are
saved with external weights as well (`-encode_image_weights.pb` / `-encode_text_weights.pb`).

`logits_per_image` can be reconstructed as:

```python
logits_per_image = image_embeds @ text_embeds.T * logit_scale + logit_bias
```

## Prototxt

Generate prototxt files with
[onnx2prototxt.py](https://github.com/ailia-ai/export-to-onnx/blob/master/onnx2prototxt.py):

```bash
python onnx2prototxt.py siglip2-<model>-encode_image.onnx
python onnx2prototxt.py siglip2-<model>-encode_text.onnx
```

## Note

`siglip2.py --separate` currently supports base-patch16-224 only.

## Verification

The separated models were verified against the combined model with onnxruntime
(max abs diff of `logits_per_image`: ~4e-6 on base-patch16-224).

```bash
python siglip2.py --separate --onnx
python siglip2.py --separate
```
