# Qwen3-Embedding

## Input

Query
```
What is the capital of China?
```

Documents (`documents.txt`)
```
The capital of China is Beijing.
Gravity is a force that attracts two bodies towards each other. ...
```

## Output

The documents in order of similarity
```
[1] (0.765)
- The capital of China is Beijing.

[2] (0.141)
- Gravity is a force that attracts two bodies towards each other. ...
```

For multi-query mode, results are shown as `[query_index-document_index]`.
```
[1-1] (0.765)
- What is the capital of China?
- The capital of China is Beijing.

[2-2] (0.600)
- Explain gravity
- Gravity is a force that attracts two bodies towards each other. ...

[1-2] (0.141)
- What is the capital of China?
- Gravity is a force that attracts two bodies towards each other. ...

[2-1] (0.135)
- Explain gravity
- The capital of China is Beijing.
```

## Requirements

This model requires additional modules.

```
pip3 install "transformers>=4.51.0"
```

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample documents,
```bash
$ python3 qwen3-embedding.py
```

If you want to specify a query, put the query text after the `-i` option.
```bash
$ python3 qwen3-embedding.py -i "QUERY"
```

If you want to specify documents, use the `--document` option (space-separated or repeatable).
```bash
$ python3 qwen3-embedding.py -i "QUERY" --document "DOC1" "DOC2"
$ python3 qwen3-embedding.py -i "QUERY" --document "DOC1" --document "DOC2"
```

If you want to specify a document file, put the file path after `--document`.
```bash
$ python3 qwen3-embedding.py -i "QUERY" --document FILE_PATH
```

For multi-query mode, specify multiple queries space-separated after a single `-i`.
```bash
$ python3 qwen3-embedding.py -i "QUERY1" "QUERY2" --document "DOC1" "DOC2"
```

You can also specify a custom task description for the query instruction.
```bash
$ python3 qwen3-embedding.py -i "QUERY" -t "Given a question, retrieve relevant passages that answer the question"
```

## Reference

- [Hugging Face - Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[Qwen3-Embedding-0.6B.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/qwen3-embedding/Qwen3-Embedding-0.6B.onnx.prototxt)
