# Segment Anything 3

## ONNX models

Download from [Hugging Face](https://huggingface.co/wkentaro/sam3-onnx-models):

```
hf download --local-dir models wkentaro/sam3-onnx-models

models
├── sam3_decoder.onnx
├── sam3_decoder.onnx.data
├── sam3_image_encoder.onnx
├── sam3_image_encoder.onnx.data
├── sam3_language_encoder.onnx
└── sam3_language_encoder.onnx.data
```

## Image mode

### Input

![Input](demo/truck.jpg)

(Image from https://github.com/facebookresearch/sam3/blob/main/assets/images/truck.jpg)

### Output

![Output](demo/image.png)

## Video mode

### Input

![Input](demo/bedroom.png)

(Image from https://github.com/facebookresearch/sam3/blob/main/assets/videos/bedroom.mp4)

### Output

![Output](demo/video.png)
