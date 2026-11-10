# Vision Transformer

## input
![input image](input.jpg)

(from https://pixabay.com/photos/labrador-retriever-dog-pet-labrador-6244939/)

<br/>

## output
![output_image](output.png)

<br/>

## usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python vit.py
(ex on CPU)  $ python vit.py -e 0
(ex on BLAS) $ python vit.py -e 1
(ex on GPU)  $ python vit.py -e 2
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 vit.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
$ python3 vit.py -i IMAGE_PATH -s SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.
```bash
$ python3 vit.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
$ python3 vit.py -v VIDEO_PATH -s SAVE_VIDEO_PATH
(ex) $ python3 vit.py --video input.mp4 --savepath output.mp4
```

By adding the `--quantize int4` option, the weight-only int4 model is used.
The transformer's constant-weight linear layers are converted to
`MatMulNBits`; dynamic attention matrix multiplications and the classifier
head remain in fp32.

```bash
$ python3 vit.py --quantize int4
$ python3 vit.py --quantize int4 -b -bc 20
```

The int4 model can be generated from the fp32 model with ONNX Runtime's
`MatMulNBitsQuantizer`:

```bash
$ pip install onnxruntime onnx numpy
$ cd export
$ python3 export_int4.py
$ cd ..
$ python3 vit.py --quantize int4
```

<br/>

## Reference

[Pytorch reimplementation of the Vision Transformer (An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale)](https://github.com/jeonsworld/ViT-pytorch)

<br/>

## Framework
Pytorch

<br/>

## Model Format
ONNX opset = 10

<br/>

## Netron

[ViT-B_16-224.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/vit/ViT-B_16-224.onnx.prototxt)
