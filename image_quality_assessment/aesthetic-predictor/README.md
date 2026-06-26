# LAION-Aesthetics_Predictor V1

## Input

![Input](demo.jpg)

Image from https://thumbs.dreamstime.com/b/lovely-cat-as-domestic-animal-view-pictures-182393057.jpg  
Refer from https://github.com/LAION-AI/aesthetic-predictor/blob/main/asthetics_predictor.ipynb

## Output

- Aesthetic score prediction
```bash
Aesthetic score: 5.0491
```

## Requirements

This model requires an additional module only when using the `--use_open_clip` option.

```
pip3 install open_clip_torch
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

You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 aesthetic-predictor.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--model_type` option, you can specify the model type which is selected from "vit_l_14", "vit_b_32". (default is vit_l_14)
```bash
$ python3 aesthetic-predictor.py --model_type vit_l_14
```

By adding the `--use_open_clip` option, you can use open_clip instead of ailia for CLIP feature extraction.
```bash
$ python3 aesthetic-predictor.py --use_open_clip
```

## Reference

- [LAION-Aesthetics_Predictor V1](https://github.com/LAION-AI/aesthetic-predictor)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[aesthetic_predictor_vit_l_14.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/aesthetic-predictor/aesthetic_predictor_vit_l_14.onnx.prototxt)
[aesthetic_predictor_vit_b_32.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/aesthetic-predictor/aesthetic_predictor_vit_b_32.onnx.prototxt)
