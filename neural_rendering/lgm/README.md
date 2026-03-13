# LGM

## Image-to-3D
### Input
![Input](catstatue_rgba.png)

(Image from https://github.com/3DTopia/LGM/blob/main/data_test/catstatue_rgba.png)

### Output
![Output](output.gif)

## Text-to-3D
### Input
Prompt:
```
a hamburger
```

### Output
![Output](hamburger.gif)

## Install
This model requires C++ Compiler.
```
pip3 install -r requirements.txt

git clone https://github.com/MrSecant/diff-gaussian-rasterization.git c-diff-gaussian-rasterization
pip install -e ./c-diff-gaussian-rasterization
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 lgm.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 lgm.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

For Text-to-3D, specify the input prompt after the `--prompt` option.
``` bash
$ python3 lgm.py --prompt "a hamburger"
```

## Reference

- [LGM](https://github.com/3DTopia/LGM)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[unet_image.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/lgm/unet_image.onnx.prototxt)  
[unet_text.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/lgm/unet_text.onnx.prototxt)  
[vae_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/lgm/vae_encoder.onnx.prototxt)  
[vae_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/lgm/vae_decoder.onnx.prototxt)  
[text_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/lgm/text_encoder.onnx.prototxt)  
[image_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/lgm/image_encoder.onnx.prototxt)  
[lgm.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/lgm/lgm.onnx.prototxt)  
