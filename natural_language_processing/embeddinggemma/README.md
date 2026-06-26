# EmbeddingGemma

## Input

- Text sequences for embedding. The CLI uses a single query plus one or more documents via `--documents` or falls back to the built-in examples.
- Token shape: (batch, sequence_length)

## Output

- `sentence_embedding` shape: (batch, 768)
- Console shows cosine-similarity ranking between the query and provided documents.

## Requirements

This model requires additional module.

```
pip3 install transformers
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

- Run with the built-in demo (default query and planet documents):
```bash
$ python3 embeddinggemma.py
```

- Similarity search with custom query and documents:
```bash
$ python3 embeddinggemma.py \
	--query "What is the Red Planet?" \
	--documents "Mercury is closest to the Sun" "Mars is called the Red Planet" "Saturn has rings"
```

- Document input: pass one or more strings after `--documents`. The option can be repeated to group documents, e.g.:
```bash
$ python3 embeddinggemma.py --documents "Doc A" "Doc B" --documents "Doc C"
```

## Reference

- [EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma?hl=ja)
- [Hugging Face - EmbeddingGemma](https://huggingface.co/collections/google/embeddinggemma)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[embeddinggemma-300m.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/embeddinggemma/embeddinggemma-300m.onnx.prototxt)
