# Depth Anything V2

## Input

* **Image or Video**

![demo image](demo1.png)

The script will perform a monocular depth estimation on the input media.

## Output

* **Depth image**

![result](output.png)

Estimated relative depth with inferno colormap(without option ```-g```),
or single channel grey scale image(with option ```-g```).

Saves to ```./output.png``` by default but it can be specified with the ```-s``` option

## Usage
Internet connection is required when running the script for the first time,
as it will download the necessary model files.

Running this script will estimate the relative depth of the input image/video.
The results will be shown in a separate window(when inferencing on image and video),
or saved as an image(when inferencing on image).

#### Example 1: Inference on prepared demo image.
```bash
$ python3 depth_anything_v2.py
```
The result will be saved to ```output.png``` by default.

#### Example 2: Specify input path, save path, and encoder type.
```bash
$ python3 depth_anything_v2.py -i input.png -s output.png -ec vitl
```
```-i```, ```-s```, ```-ec``` options can be used to specify the
input path, save path, and encoder type separately.

Available encoder types: ```vits```, ```vitb```, ```vitl```

#### Example 3: Inference on Video.
```bash
$ python3 depth_anything_v2.py -v 0
```
argument after the ```-v``` option can be the device id of the webcam,
or the path to the input video.

## Requirements

This model requires ailia SDK 1.2.16 and later.

## Reference

* [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
* [Depth Anything ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[depth_anything_v2_depth_anything_v2_vits.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/depth_anything_v2/depth_anything_v2_depth_anything_v2_vits.onnx.prototxt)
[depth_anything_v2_depth_anything_v2_vitb.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/depth_anything_v2/depth_anything_v2_depth_anything_v2_vitb.onnx.prototxt)
[depth_anything_v2_depth_anything_v2_vitl.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/depth_anything_v2/depth_anything_v2_depth_anything_v2_vitl.onnx.prototxt)
