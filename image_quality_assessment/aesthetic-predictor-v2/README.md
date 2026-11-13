# CLIP+MLP Aesthetic Score Predictor

Visualize an aesthetic score predictor (how much people like on average an image) based on a simple neural net that takes CLIP embeddings as inputs.

## Input

![Input](demo.jpg)

Image from https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/F5WsLD0XoAAviiV%20(1).jpeg
Refer to https://github.com/christophschuhmann/improved-aesthetic-predictor/

## Output

- Aesthetic score prediction
```bash
Aesthetic score: 5.2305717
```

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 aesthetic-predictor.py
```

If you want to specify the input image, put the image path after the `--input` option.
```bash
$ python3 aesthetic-predictor.py --input IMAGE_PATH
```

## Reference

- [CLIP+MLP Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)

## Framework

Pytorch

## Model Format

ONNX opset=18

## Netron

[sac+logos+ava1-l14-linearMSE.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/aesthetic-predictor-v2/sac+logos+ava1-l14-linearMSE.onnx.prototxt)
[ViT-L14-encode_image.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/clip/ViT-L14-encode_image.onnx.prototxt)
