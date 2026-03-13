# MIMO-UNet

## Input

![Input](demo.png)

(Image from https://seungjunnah.github.io/Datasets/gopro.html)

## Output

![Output](output.png)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 mimo-unet.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 mimo-unet.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 mimo-unet.py --video VIDEO_PATH
```

By adding the `--model_type` option, you can specify model type which is selected from "MIMO-UNetPlus", "MIMO-UNet". (default is MIMO-UNetPlus)
```bash
$ python3 mimo-unet.py --model_type MIMO-UNetPlus
```

## Reference

- [MIMO-UNet - Official Pytorch Implementation](https://github.com/chosj95/MIMO-UNet)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[MIMO-UNetPlus.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mimo-unet/MIMO-UNetPlus.onnx.prototxt)  
[MIMO-UNet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mimo-unet/MIMO-UNet.onnx.prototxt)
