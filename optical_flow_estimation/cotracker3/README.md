# CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos

## Input

![Input](input.gif)

(Image from https://github.com/facebookresearch/co-tracker/blob/main/gradio_demo/videos/bear.mp4)

Shape : (1, 3, 854, 480)  

## Output

![Output](output.gif)


### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample video,
``` bash
$ python3 cotracker3.py
```

If you want to specify the input video, put the video path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.

```bash
$ python3 cotracker3.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By default, the ailia SDK is used. If you want to use ONNX Runtime, use the --onnx option.
```bash
$ python3 cotracker3.py --onnx
```

## Reference

- [CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos](https://github.com/facebookresearch/co-tracker)

## Framework

Pytorch 2.4

## Model Format

ONNX opset=20

## Netron

[cotracker3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/cotracker3/cotracker3.onnx.prototxt)
