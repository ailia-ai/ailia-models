# Age Transformation Using a Style-Based Regression Model

## Input

[<img src="img/input.jpg" width=256px>](img/input.jpg)

(Image from https://github.com/yuval-alaluf/SAM/blob/master/notebooks/images/866.jpg)

Shape : (1, 3, 1024, 1024)

Face alignment and reshaped to : (1, 3, 256, 256)  

## Output

![Output](img/output.png)

\* Note: From left to right: original (face aligned) image, 10-, 30-, 50-, 70-, and 90-year-old-individual images.

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 sam.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 sam.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH 
```

By specifying the `-age` option, you can choose the individual's age(s) (between 0 and 100) to be generated (default '10,30,50,70,90').
```bash
$ python3 sam.py -age '0,25,50,75,100' 
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 sam.py --video VIDEO_PATH 
```

By adding the `--use_dlib` option, you can use original version of face alignment.

## Reference

- [SAM](https://github.com/yuval-alaluf/SAM)

- [PSGAN](https://github.com/ailia-ai/ailia-models/tree/master/style_transfer/psgan) (face alignment without dlib)

## Framework

Pytorch 1.10.0

Python 3.6.7+

## Model Format

ONNX opset=11

## Netron

[encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sam/encoder.onnx.prototxt)

[pretrained-encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sam/pretrained-encoder.onnx.prototxt)

[decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sam/decoder.onnx.prototxt)