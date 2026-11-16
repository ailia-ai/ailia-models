# Partial Convolution

### input
![input_image](test_5735.JPEG)

Ailia input shape: (1, 3, 224, 224)  


### output
```
class_count=5
+ idx=0
  category=731[plunger, plumber's helper ]
  prob=12.340980529785156
+ idx=1
  category=543[dumbbell ]
  prob=11.191944122314453
+ idx=2
  category=680[nipple ]
  prob=10.75782299041748
+ idx=3
  category=422[barbell ]
  prob=10.286931991577148
+ idx=4
  category=844[switch, electric switch, electrical switch ]
  prob=9.976827621459961
```

### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 partialconv.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 partialconv.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 partialconv.py --video VIDEO_PATH
```

You can select a model from `resnet50 | vgg16_bn | pdresnet50 | pdresnet101 | pdresnet152` by adding `--arch`.
Please note that `pdresnet152` does not currently executable on Mac OS.

### Reference
[Partial Convolution Layer for Padding and Image Inpainting](https://github.com/NVIDIA/partialconv)

### Framework

PyTorch 1.2.0

### Model Format
ONNX opset = 10 

### Netron

[resnet50.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/partialconv/resnet50.onnx.prototxt)

[vgg16_bn.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/partialconv/vgg16_bn.onnx.prototxt)

[pdresnet50.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/partialconv/pdresnet50.onnx.prototxt)
