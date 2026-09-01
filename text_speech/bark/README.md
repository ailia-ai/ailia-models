# Bark

## Input

Text

- Example
```
Hello, my name is Suno. And, uh ― and I like pizza. [laughs]
But I also have other interests such as playing tic tac toe.
```

## Output

https://github.com/axinc-ai/ailia-models/assets/29946532/f7f13d36-1e2b-4821-be4f-0927d250a893

## Requirements

This model requires additional module.
```
pip3 install pytorch
pip3 install transformers
pip3 install encodec
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 bark.py
```

If you want to specify the text, put the text after the `--input` option.
```bash
$ python3 bark.py --input TEXT
```

## Reference

- [Bark](https://github.com/suno-ai/bark)

## Framework

Pytorch

## Model Format

ONNX opset=14

## Netron

[text.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bark/text.onnx.prototxt)  
[coarse.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bark/coarse.onnx.prototxt)  
[fine.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bark/fine.onnx.prototxt)  
