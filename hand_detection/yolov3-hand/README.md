# yolov3-hand

## Input

![Input](couple.jpg)

- Shape : (1, 3, 448, 448)  
- Range : [0.0, 1.0]

## Output

![Output](output.png)

- category : [0,1]
- probablity : [0.0,1.0]
- position : x, y, w, h [0,1]

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 yolov3-hand.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 yolov3-hand.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 yolov3-hand.py --video VIDEO_PATH
```

## Reference

- [Hand detection branch of Face detection using keras-yolov3](https://github.com/ailia-ai/yolov3-face/tree/hand_detection)

## Framework

Keras 2.2.4

## Model Format

ONNX opset=10

## Netron

[yolov3-hand.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov3-hand/yolov3-hand.opt.onnx.prototxt)

