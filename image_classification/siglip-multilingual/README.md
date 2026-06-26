# SigLIP (base-sized model, multilingual)

## Input

![Input](demo.jpg)

## Output

- Zero-Shot Prediction
```bash
1: 2 cats - 21.57%
2: a remote - 0.01%
3: a plane - 0.00%
```

## Requirements

If you use `--disable_ailia_tokenizer` option, this model requires additional module.
```
pip3 install transformers
pip3 install sentencepiece
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 siglip-multilingual.py
```

If you want to specify the input image, put the image path after the `--input` option.
```bash
$ python3 siglip-multilingual.py --input IMAGE_PATH
```

You can use `--text` option if you want to specify a subset of the candidate labels to input into the model.
Default labels are "2 cats", "a plane" and "a remote". Labels in other languages can also be specified,
since this model is multilingual.
```bash
$ python3 siglip-multilingual.py --text "2 cats" --text "a plane" --text "a remote" --text "3 dogs"
```

Labels passed here are candidate labels, not full sentences: each one is formatted through
`"This is a photo of {}."` before being tokenized, matching what
`transformers.pipeline(task="zero-shot-image-classification", ...)` does internally. Passing an already-complete
sentence (e.g. "a photo of 2 cats") would also get wrapped in that template, producing a sentence inside a
sentence, so stick to bare labels here.

## Reference

- [Hugging Face - SigLIP base-patch16-256-multilingual](https://huggingface.co/google/siglip-base-patch16-256-multilingual)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[siglip-base-patch16-256-multilingual.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/siglip-base-patch16-256-multilingual/siglip-base-patch16-256-multilingual.onnx.prototxt)
