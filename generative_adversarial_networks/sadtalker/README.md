# SadTalker

## Input

[<img src="input.png" width=256px>](input.png)

(Image from https://github.com/OpenTalker/SadTalker/blob/main/examples/source_image/art_1.png)

[input.wav](input.wav)

(Audio from https://github.com/OpenTalker/SadTalker/blob/main/examples/driven_audio/bus_chinese.wav)

## Output

mp4

## Requirements

This model requires `ffmpeg` and additional module.

```bash
pip3 install -r requirements.txt
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 sadtalker.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
If you want to specify the input audio, put the audio path after the `--audio` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 sadtalker.py --input IMAGE_PATH --audio AUDIO_PATH --savepath SAVE_VIDEO_PATH 
```

By adding the `--enhancer` option, you can enhance the generated face via gfpgan.
```bash
$ python3 sadtalker.py --enhancer
```

To run the full image animation, set the `--preprocess` option to `full`. For better results, also use `--still`.
```bash
$ python3 sadtalker.py -i input_full_body.png --enhancer --preprocess full --still
```

## Reference

- [SadTalker](https://github.com/OpenTalker/SadTalker)
- [retinaface](https://github.com/biubug6/Pytorch_Retinaface)
- [GFPGAN](https://github.com/TencentARC/GFPGAN)

## Framework

Pytorch

## Model Format

ONNX opset=20

## Netron

[animation_generator.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sadtalker/animation_generator.onnx.prototxt)  
[audio2exp.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sadtalker/audio2exp.onnx.prototxt)  
[audio2pose.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sadtalker/audio2pose.onnx.prototxt)  
[face_align.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sadtalker/face_align.onnx.prototxt)  
[face3d_recon.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sadtalker/face3d_recon.onnx.prototxt)  
[kp_detector.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sadtalker/kp_detector.onnx.prototxt)  
[mappingnet_full.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sadtalker/mappingnet_full.onnx.prototxt)  
[mappingnet_not_full.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sadtalker/mappingnet_not_full.onnx.prototxt)  
