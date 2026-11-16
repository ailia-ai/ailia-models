# Illumination Correction


### input
<img src='input.png' width='240px'>

- The output image of [ailia-models/dewarpnet](https://github.com/ailia-ai/ailia-models/tree/master/image_manipulation/dewarpnet) is used as the sample input image.
(original image from https://github.com/cvlab-stonybrook/DewarpNet/tree/master/eval/inp)

Ailia input shape: (1, 3, 128, 128)  
Range: [-1, 1]

### output
<img src='output.png' width='240px'>

### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 illnet.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 illnet.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video. (test implementation)
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 illnet.py --video VIDEO_PATH
```

### Reference
[Document Rectification and Illumination Correction using a Patch-based CNN](https://github.com/xiaoyu258/DocProj)


### Framework
PyTorch 1.3


### Model Format
ONNX opset = 10


### Netron
[illnet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/illnet/illnet.onnx.prototxt)
