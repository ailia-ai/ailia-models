# Segment Anything 2

## Image mode

### Input

![Input](truck.jpg)

(Image from https://github.com/facebookresearch/segment-anything/blob/main/notebooks/images/truck.jpg)

### Output

![Output](output.png)

## Video mode

### Input

![Input](demo/00000.jpg)

(Image from https://github.com/facebookresearch/segment-anything-2/tree/main/notebooks/videos/bedroom)

### Output

![Output](video_4.png)

## Automatic mask generation mode

### Output

![Output](output_auto.png)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 segment-anything-2.py
```

For the sample video,
```bash
$ python3 segment-anything-2.py -v demo
```

For the webcam,
```bash
$ python3 segment-anything-2.py -v 0 --pos 960 540
```

For automatic mask generation (segment all objects),
```bash
$ python3 segment-anything-2.py --auto
```

You can adjust automatic mask generation parameters.
```bash
$ python3 segment-anything-2.py --auto --points_per_side 32 --pred_iou_thresh 0.8 --stability_score_thresh 0.95
```

By default, the ailia SDK is used. If you want to use ONNX Runtime, use the --onnx option.

```bash
$ python3 segment-anything-2.py --onnx
```

If you want to specify the input image, put the image path after the `--input` option.
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 segment-anything-2.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

If you want to specify the positive point, put the coordinates(x,y) after the `--pos` option.
```bash
$ python3 segment-anything-2.py --pos 500 375
```

And if you want to specify the negative point, put the coordinates after the `--neg` option.
```bash
$ python3 segment-anything-2.py --pos 500 375 --neg 360 405
```

If you want to specify the box, put the coordinates(x1,y1,x2,y2) after the `--box` option.
```bash
$ python3 segment-anything-2.py --box 425 600 700 875
```

These options can be combined.
```bash
$ python3 segment-anything-2.py --pos 500 375 --pos 1125 625
```

```bash
$ python3 segment-anything-2.py --box 425 600 700 875 --neg 575 750
```

By adding the `--model_type` option, you can specify model type which is selected from "hiera_l",  "hiera_b+",  "hiera_s", and "hiera_t". (default is hiera_l)
```bash
$ python3 segment-anything-2.py --model_type hiera_l
```

By adding the `--version` option, you can specify model type which is selected from "2" and "2.1". (default is 2)
```bash
$ python3 segment-anything-2.py --version "2.1"
```

To use legacy ONNX models (4D matmul with batch=1, mask prompt not supported), use the --legacy option.
```bash
$ python3 segment-anything-2.py --legacy
```

To improve the performance of MemoryAttention, you can also reduce the number of reference images in past frames, which is num_mask_mem.
```bash
$ python3 segment-anything-2.py -v 0 --num_mask_mem 2 --max_obj_ptrs_in_encoder 2
```

## Reference

- [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

### SAM2

#### hiera_l

- [image_encoder_hiera_l.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/image_encoder_hiera_l.onnx.prototxt)
- [mask_decoder_hiera_l.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/mask_decoder_hiera_l.onnx.prototxt)
- [prompt_encoder_with_mask_hiera_l.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/prompt_encoder_with_mask_hiera_l.onnx.prototxt)
- [memory_attention_6d_hiera_l.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/memory_attention_6d_hiera_l.onnx.prototxt)
- [memory_encoder_hiera_l.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/memory_encoder_hiera_l.onnx.prototxt)
- [mlp_hiera_l.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/mlp_hiera_l.onnx.prototxt)

#### hiera_b+

- [image_encoder_hiera_b+.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/image_encoder_hiera_b+.onnx.prototxt)
- [mask_decoder_hiera_b+.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/mask_decoder_hiera_b+.onnx.prototxt)
- [prompt_encoder_with_mask_hiera_b+.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/prompt_encoder_with_mask_hiera_b+.onnx.prototxt)
- [memory_attention_6d_hiera_b+.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/memory_attention_6d_hiera_b+.onnx.prototxt)
- [memory_encoder_hiera_b+.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/memory_encoder_hiera_b+.onnx.prototxt)
- [mlp_hiera_b+.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/mlp_hiera_b+.onnx.prototxt)

#### hiera_s

- [image_encoder_hiera_s.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/image_encoder_hiera_s.onnx.prototxt)
- [mask_decoder_hiera_s.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/mask_decoder_hiera_s.onnx.prototxt)
- [prompt_encoder_with_mask_hiera_s.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/prompt_encoder_with_mask_hiera_s.onnx.prototxt)
- [memory_attention_6d_hiera_s.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/memory_attention_6d_hiera_s.onnx.prototxt)
- [memory_encoder_hiera_s.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/memory_encoder_hiera_s.onnx.prototxt)
- [mlp_hiera_s.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/mlp_hiera_s.onnx.prototxt)

#### hiera_t

- [image_encoder_hiera_t.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/image_encoder_hiera_t.onnx.prototxt)
- [mask_decoder_hiera_t.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/mask_decoder_hiera_t.onnx.prototxt)
- [prompt_encoder_with_mask_hiera_t.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/prompt_encoder_with_mask_hiera_t.onnx.prototxt)
- [memory_attention_6d_hiera_t.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/memory_attention_6d_hiera_t.onnx.prototxt)
- [memory_encoder_hiera_t.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/memory_encoder_hiera_t.onnx.prototxt)
- [mlp_hiera_t.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/mlp_hiera_t.onnx.prototxt)

#### legacy

- [prompt_encoder_hiera_l.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/prompt_encoder_hiera_l.onnx.prototxt)
- [memory_attention_hiera_l.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/memory_attention_hiera_l.opt.onnx.prototxt)
- [prompt_encoder_hiera_b+.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/prompt_encoder_hiera_b+.onnx.prototxt)
- [memory_attention_hiera_b+.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/memory_attention_hiera_b+.opt.onnx.prototxt)
- [prompt_encoder_hiera_s.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/prompt_encoder_hiera_s.onnx.prototxt)
- [memory_attention_hiera_s.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/memory_attention_hiera_s.opt.onnx.prototxt)
- [prompt_encoder_hiera_t.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/prompt_encoder_hiera_t.onnx.prototxt)
- [memory_attention_hiera_t.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2/memory_attention_hiera_t.opt.onnx.prototxt)

### SAM2.1

#### hiera_l

- [image_encoder_hiera_l_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/image_encoder_hiera_l_2.1.onnx.prototxt)
- [mask_decoder_hiera_l_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/mask_decoder_hiera_l_2.1.onnx.prototxt)
- [prompt_encoder_with_mask_hiera_l_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/prompt_encoder_with_mask_hiera_l_2.1.onnx.prototxt)
- [memory_attention_6d_hiera_l_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/memory_attention_6d_hiera_l_2.1.onnx.prototxt)
- [memory_encoder_hiera_l_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/memory_encoder_hiera_l_2.1.onnx.prototxt)
- [mlp_hiera_l_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/mlp_hiera_l_2.1.onnx.prototxt)
- [obj_ptr_tpos_proj_hiera_l_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/obj_ptr_tpos_proj_hiera_l_2.1.onnx.prototxt)

#### hiera_b+

- [image_encoder_hiera_b+_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/image_encoder_hiera_b+_2.1.onnx.prototxt)
- [mask_decoder_hiera_b+_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/mask_decoder_hiera_b+_2.1.onnx.prototxt)
- [prompt_encoder_with_mask_hiera_b+_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/prompt_encoder_with_mask_hiera_b+_2.1.onnx.prototxt)
- [memory_attention_6d_hiera_b+_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/memory_attention_6d_hiera_b+_2.1.onnx.prototxt)
- [memory_encoder_hiera_b+_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/memory_encoder_hiera_b+_2.1.onnx.prototxt)
- [mlp_hiera_b+_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/mlp_hiera_b+_2.1.onnx.prototxt)
- [obj_ptr_tpos_proj_hiera_b+_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/obj_ptr_tpos_proj_hiera_b+_2.1.onnx.prototxt)

#### hiera_s

- [image_encoder_hiera_s_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/image_encoder_hiera_s_2.1.onnx.prototxt)
- [mask_decoder_hiera_s_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/mask_decoder_hiera_s_2.1.onnx.prototxt)
- [prompt_encoder_with_mask_hiera_s_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/prompt_encoder_with_mask_hiera_s_2.1.onnx.prototxt)
- [memory_attention_6d_hiera_s_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/memory_attention_6d_hiera_s_2.1.onnx.prototxt)
- [memory_encoder_hiera_s_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/memory_encoder_hiera_s_2.1.onnx.prototxt)
- [mlp_hiera_s_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/mlp_hiera_s_2.1.onnx.prototxt)
- [obj_ptr_tpos_proj_hiera_s_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/obj_ptr_tpos_proj_hiera_s_2.1.onnx.prototxt)

#### hiera_t

- [image_encoder_hiera_t_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/image_encoder_hiera_t_2.1.onnx.prototxt)
- [mask_decoder_hiera_t_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/mask_decoder_hiera_t_2.1.onnx.prototxt)
- [prompt_encoder_with_mask_hiera_t_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/prompt_encoder_with_mask_hiera_t_2.1.onnx.prototxt)
- [memory_attention_6d_hiera_t_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/memory_attention_6d_hiera_t_2.1.onnx.prototxt)
- [memory_encoder_hiera_t_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/memory_encoder_hiera_t_2.1.onnx.prototxt)
- [mlp_hiera_t_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/mlp_hiera_t_2.1.onnx.prototxt)
- [obj_ptr_tpos_proj_hiera_t_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/obj_ptr_tpos_proj_hiera_t_2.1.onnx.prototxt)

#### legacy

- [prompt_encoder_hiera_l_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/prompt_encoder_hiera_l_2.1.onnx.prototxt)
- [memory_attention_hiera_l_2.1.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/memory_attention_hiera_l_2.1.opt.onnx.prototxt)
- [prompt_encoder_hiera_b+_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/prompt_encoder_hiera_b+_2.1.onnx.prototxt)
- [memory_attention_hiera_b+_2.1.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/memory_attention_hiera_b+_2.1.opt.onnx.prototxt)
- [prompt_encoder_hiera_s_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/prompt_encoder_hiera_s_2.1.onnx.prototxt)
- [memory_attention_hiera_s_2.1.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/memory_attention_hiera_s_2.1.opt.onnx.prototxt)
- [prompt_encoder_hiera_t_2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/prompt_encoder_hiera_t_2.1.onnx.prototxt)
- [memory_attention_hiera_t_2.1.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/segment-anything-2.1/memory_attention_hiera_t_2.1.opt.onnx.prototxt)  

## Model information

By default, the new re-exported models are used:
- `prompt_encoder_with_mask` : Supports 4D mask input (B,C,H,W) for Conv2d compatibility.
- `memory_attention_6d` : Uses 6D matmul with dynamic batch support.

With `--legacy`, the old models are used:
- `prompt_encoder` : 3D mask input (mask prompt not supported).
- `memory_attention.opt` : 4D matmul with batch size fixed to 1.
