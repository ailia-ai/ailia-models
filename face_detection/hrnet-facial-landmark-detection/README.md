# HRNet Facial Landmark Detection

## Input

<img src="input.jpg" width="60%">

- Landmark model input shape: (n, 3, 256, 256)
- Face detector input shape: (1, 3, 384, 672)

## Output

<img src="output.png" width="60%">

- WFLW model: heatmap shape (n, 98, 64, 64) → 98 landmark coordinates
- AFLW model: heatmap shape (n, 19, 64, 64) → 19 landmark coordinates
- COFW model: heatmap shape (n, 29, 64, 64) → 29 landmark coordinates
- 300W model: heatmap shape (n, 68, 64, 64) → 68 landmark coordinates

## Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 hrnet-facial-landmark-detection.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 hrnet-facial-landmark-detection.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.  
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 hrnet-facial-landmark-detection.py --video VIDEO_PATH
```

By default the WFLW model (98 landmarks) is used. You can switch models with the `-m` option.
```bash
$ python3 hrnet-facial-landmark-detection.py -m wflw   # 98 landmarks (default)
$ python3 hrnet-facial-landmark-detection.py -m aflw   # 19 landmarks
$ python3 hrnet-facial-landmark-detection.py -m cofw   # 29 landmarks
$ python3 hrnet-facial-landmark-detection.py -m 300w   # 68 landmarks
```

## Reference

- [HRNet-Facial-Landmark-Detection](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

- [hrnet_w18_wflw_256x256.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hrnet-facial-landmark-detection/hrnet_w18_wflw_256x256.onnx.prototxt)
- [hrnet_w18_aflw_256x256.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hrnet-facial-landmark-detection/hrnet_w18_aflw_256x256.onnx.prototxt)
- [hrnet_w18_cofw_256x256.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hrnet-facial-landmark-detection/hrnet_w18_cofw_256x256.onnx.prototxt)
- [hrnet_w18_300w_256x256.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hrnet-facial-landmark-detection/hrnet_w18_300w_256x256.onnx.prototxt)
