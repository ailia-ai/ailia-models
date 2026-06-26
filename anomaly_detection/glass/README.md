# GLASS : A Unified Anomaly Synthesis Strategy with Gradient Ascent for Industrial Anomaly Detection and Localization

This model needs to be trained with a good quality dataset in Pytorch.

## Input

Normal images

<img src="bottle_000.png" width="256" height="256">

(Image from MVTec AD datasets https://www.mvtec.com/company/research/datasets/mvtec-ad/)

## Output

Anomaly maps

Left to right: input, ground truth, predicted head map

![Output](output.png)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 glass.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 glass.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

You can specify the dataset category using the `--dataset_name` option.  
Available choices are:
- `mvtec_carpet`
- `mvtec_grid`
- `mvtec_leather`
- `mvtec_tile`
- `mvtec_wood`
- `mvtec_bottle`
- `mvtec_cable`
- `mvtec_capsule`
- `mvtec_hazelnut`
- `mvtec_metal_nut`
- `mvtec_pill`
- `mvtec_screw`
- `mvtec_toothbrush`
- `mvtec_transistor`
- `mvtec_zipper`

(default is `mvtec_bottle`)

```bash
$ python3 glass.py --dataset_name mvtec_bottle
```

You can also provide a directory containing ground truth mask files using the `--gt_dir` option.
The ground truth mask filename should correspond to the input image filename by appending _mask before the file extension.
(For example, if the input image is `bottle_000.png`, the corresponding ground truth mask file should be `bottle_000_mask.png`)
```bash
$ python3 glass.py --input IMAGE_PATH --gt_dir ./gt_masks
```

## Reference

[GLASS](https://github.com/cqylunlun/GLASS)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

- [mvtec_carpet](https://netron.app/?url=https://storage.googleapis.com/ailia-models/glass/mvtec_carpet.onnx.prototxt)
- [mvtec_grid](https://netron.app/?url=https://storage.googleapis.com/ailia-models/glass/mvtec_grid.onnx.prototxt)
- [mvtec_leather](https://netron.app/?url=https://storage.googleapis.com/ailia-models/glass/mvtec_leather.onnx.prototxt)
- [mvtec_tile](https://netron.app/?url=https://storage.googleapis.com/ailia-models/glass/mvtec_tile.onnx.prototxt)
- [mvtec_wood](https://netron.app/?url=https://storage.googleapis.com/ailia-models/glass/mvtec_wood.onnx.prototxt)
- [mvtec_bottle](https://netron.app/?url=https://storage.googleapis.com/ailia-models/glass/mvtec_bottle.onnx.prototxt)
- [mvtec_cable](https://netron.app/?url=https://storage.googleapis.com/ailia-models/glass/mvtec_cable.onnx.prototxt)
- [mvtec_capsule](https://netron.app/?url=https://storage.googleapis.com/ailia-models/glass/mvtec_capsule.onnx.prototxt)
- [mvtec_hazelnut](https://netron.app/?url=https://storage.googleapis.com/ailia-models/glass/mvtec_hazelnut.onnx.prototxt)
- [mvtec_metal_nut](https://netron.app/?url=https://storage.googleapis.com/ailia-models/glass/mvtec_metal_nut.onnx.prototxt)
- [mvtec_pill](https://netron.app/?url=https://storage.googleapis.com/ailia-models/glass/mvtec_pill.onnx.prototxt)
- [mvtec_screw](https://netron.app/?url=https://storage.googleapis.com/ailia-models/glass/mvtec_screw.onnx.prototxt)
- [mvtec_toothbrush](https://netron.app/?url=https://storage.googleapis.com/ailia-models/glass/mvtec_toothbrush.onnx.prototxt)
- [mvtec_transistor](https://netron.app/?url=https://storage.googleapis.com/ailia-models/glass/mvtec_transistor.onnx.prototxt)
- [mvtec_zipper](https://netron.app/?url=https://storage.googleapis.com/ailia-models/glass/mvtec_zipper.onnx.prototxt)
