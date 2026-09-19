# Segment Anything 3

## ONNX models

Download from [Hugging Face](https://huggingface.co/wkentaro/sam3-onnx-models):

```
hf download --local-dir . --include "*.onnx*" wkentaro/sam3-onnx-models
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
