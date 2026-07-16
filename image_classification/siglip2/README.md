# SigLIP 2

## Input

![Input](demo.jpg)

(Image from http://images.cocodataset.org/val2017/000000039769.jpg)

## Output

- Zero-Shot Prediction
```bash
1: 2 cats - 65.41%
2: 3 dogs - 32.62%
3: a remote - 1.05%
4: a plane - 0.92%
```

## Requirements
This model requires additional module.

```
pip3 install transformers
pip3 install sentencepiece
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 siglip2.py
```

If you want to specify the input image, put the image path after the `--input` option.
```bash
$ python3 siglip2.py --input IMAGE_PATH
```

You can use `--text` option  if you want to specify a subset of the texture labels to input into the model.  
Default labels is "2 cats", "a plane", "a remote" and "3 dogs".
```bash
$ python3 siglip2.py --text "2 cats" --text "a plane" --text "a remote" --text "3 dogs"
```

By adding the `--model_type` option, you can specify model type which is selected from "base-patch16-224", "base-patch16-224", "giant-patch16-256". (default is base-patch16-224)
```bash
$ python3 siglip2.py --model_type base-patch16-224
```

By adding the `--separate` option, you can use models separated into an image encoder and a text encoder. (base-patch16-224 only)
The text features are computed only once, so this is efficient when processing multiple images with the same labels.
```bash
$ python3 siglip2.py --separate
```

## Reference

- [Hugging Face - SigLIP 2 Base](https://huggingface.co/google/siglip2-base-patch16-224)
- [Zero-shot Image Classification with SigLIP2](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/siglip-zero-shot-image-classification/siglip-zero-shot-image-classification.ipynb)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[siglip2-base-patch16-224.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/siglip2/siglip2-base-patch16-224.onnx.prototxt)  
[siglip2-base-patch16-224-encode_image.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/siglip2/siglip2-base-patch16-224-encode_image.onnx.prototxt)  
[siglip2-base-patch16-224-encode_text.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/siglip2/siglip2-base-patch16-224-encode_text.onnx.prototxt)  
