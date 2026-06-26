# Ruri-Reranker: Japanese General Reranker

## Input

Query
```
瑠璃色はどんな色？
```

Documents (`documents.txt`)
```
ワシ、タカ、ハゲワシ、ハヤブサ、コンドル、フクロウが代表的である。...
瑠璃、または琉璃（るり）は、仏教の七宝の一つ。...
瑠璃色（るりいろ）は、紫みを帯びた濃い青。...
```

## Output

The scores in order of higher
```
#3 (1.000)
- 瑠璃色（るりいろ）は、紫みを帯びた濃い青。名は、半貴石の瑠璃（ラピスラズリ、英: lapis lazuli）による。JIS慣用色名では「こい紫みの青」（略号 dp-pB）と定義している[1][2]。

#2 (0.081)
- 瑠璃、または琉璃（るり）は、仏教の七宝の一つ。サンスクリットの vaiḍūrya またはそのプラークリット形の音訳である。金緑石のこととも、ラピスラズリであるともいう[1]。

#1 (0.000)
- ワシ、タカ、ハゲワシ、ハヤブサ、コンドル、フクロウが代表的である。これらの猛禽類はリンネ前後の時代(17~18世紀)には鷲類・鷹類・隼類及び梟類に分類された。...
```

## Requirements

This model requires additional module.

```
pip3 install transformers
```

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample documents,
```bash
$ python3 ruri-v3-reranker.py
```

If you want to specify a query, put the query text after the `-i` option.
```bash
$ python3 ruri-v3-reranker.py -i "QUERY"
```

If you want to specify documents, use the `--document` option (space-separated or repeatable).
```bash
$ python3 ruri-v3-reranker.py -i "QUERY" --document "DOC1" "DOC2" "DOC3"
$ python3 ruri-v3-reranker.py -i "QUERY" --document "DOC1" --document "DOC2"
```

If you want to specify a document file, put the file path after `--document`.
```bash
$ python3 ruri-v3-reranker.py -i "QUERY" --document FILE_PATH
```

For multi-query mode (query and document 1-to-1 correspondence),
```bash
$ python3 ruri-v3-reranker.py -i "QUERY1" "QUERY2" --document "DOC1" "DOC2"
```

## Reference

- [Hugging Face - ruri-v3-reranker-310m](https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[ruri-v3-reranker-310m.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/ruri-v3-reranker/ruri-v3-reranker-310m.onnx.prototxt)
