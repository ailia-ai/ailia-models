# DINOv3

## Input

![Input](coco_000000039769.jpg)

## Output

Patch-level cosine similarity map between a reference patch (red cross) and
every other patch (default reference point: image center).

![Output](output.png)

```bash
last_hidden_state: shape=(1, 785, 768)
pooler_output: shape=(1, 768), norm=48.3022
```

To generate multiple panels at once, pass `--point` multiple times:

```bash
$ python3 dinov3.py --input fruit_market.jpg --model_type vith16plus --resolution 1792 \
    --point 274 39 --point 394 191 --point 352 167 --point 401 346 \
    --point 224 228 --point 560 273 --point 136 337 --point 609 570 \
    -s images/panel.png
```

| | | |
|:---:|:---:|:---:|
| ![panel_0](images/panel_0.png) | ![panel_1](images/panel_1.png) | ![panel_2](images/panel_2.png) |
| ![panel_3](images/panel_3.png) | ![panel_4](images/panel_4.png) | ![panel_5](images/panel_5.png) |
| ![panel_6](images/panel_6.png) | ![panel_7](images/panel_7.png) | |

## PCA Mode

Patch features of a foreground object are projected onto 3 principal components (PCA), then colored with a sigmoid to produce a rainbow visualization.

Input image and foreground mask:

| Input | Mask |
|:---:|:---:|
| ![Input](image_pca.jpg) | ![Mask](image_pca_fg.png) |

Output PCA visualization:

![PCA output](images/pca_output.png)

```bash
$ python3 dinov3.py -m vitl16 --mode pca \
    --mask image_pca_fg.png \
    -i image_pca.jpg \
    -s images/pca_output.png
```

The foreground mask (grayscale, pixel > 127 = foreground) is used to fit PCA only on the object of interest.
Background patches are set to black.
The mask can be generated with the `fg_classifier.pkl` trained in the `foreground_segmentation` notebook.

## Matching Mode

Establishes dense and sparse correspondences between two images using patch-level cosine similarity.

Input images and foreground masks:

| Left Image | Left Mask | Right Image | Right Mask |
|:---:|:---:|:---:|:---:|
| ![Left](image_left.jpg) | ![Left mask](image_left_fg.png) | ![Right](image_right.jpg) | ![Right mask](image_right_fg.png) |

Dense correspondences (PCA color map — same color = same semantic region):

![Dense matching](images/matching.png)

Sparse correspondences (lines connect matched patches, colored by PCA color):

![Sparse matching](images/matching_sparse.png)

```bash
$ python3 dinov3.py -m vitl16 --mode matching \
    --input image_left.jpg \
    --image2 image_right.jpg \
    --mask image_left_fg.png \
    --mask2 image_right_fg.png \
    -s images/matching.png
# -> images/matching.png (dense), images/matching_sparse.png (sparse)
```

## Tracking Mode

Propagates segmentation masks from the first frame to subsequent frames using patch-level feature similarity.

| Frame 0 | Frame 57 | Frame 115 | Frame 173 |
|:---:|:---:|:---:|:---:|
| ![Frame 0](images/tracking_frame0.jpg) | ![Frame 57](images/tracking_frame57.jpg) | ![Frame 115](images/tracking_frame115.jpg) | ![Frame 173](images/tracking_frame173.jpg) |
| ![Mask 0](images/tracking_mask_001.png) | ![Mask 57](images/tracking_mask_002.png) | ![Mask 115](images/tracking_mask_003.png) | ![Mask 173](images/tracking_mask_004.png) |

```bash
$ python3 dinov3.py -m vitl16 --mode tracking \
    --video VIDEO_DIR_OR_FILE \
    --mask first_frame_mask.png \
    --resolution 960 \
    -s tracking.mp4
```

## Requirements

This model requires additional module.

```
pip3 install transformers
```

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 dinov3.py
```

If you want to specify the input image, put the image path after the `--input` option.
```bash
$ python3 dinov3.py --input IMAGE_PATH
```

This model has 5 variants. You can specify the model type with the `--model_type` option.
Default is `convnext_tiny`.
```bash
$ python3 dinov3.py --model_type vits16
$ python3 dinov3.py --model_type vitb16
$ python3 dinov3.py --model_type vitl16
$ python3 dinov3.py --model_type vith16plus
$ python3 dinov3.py --model_type convnext_tiny
```

Neither architecture has a fixed input size (ViT's RoPE position embedding
and ConvNeXt's pure-conv design both generalize to any resolution; the ONNX
models were exported with dynamic height/width axes), so the inference
resolution can be changed with `--resolution`. Default is 896.
```bash
$ python3 dinov3.py --resolution 1344
```

You can change the reference patch used to build the similarity map with the
`--point` option (`x y`, in the original image's pixel coordinates). Default
is the image center.
```bash
$ python3 dinov3.py --point 100 150
```

`--point` can be specified multiple times. One panel is saved per point, with
a `_0`, `_1`, ... suffix added to `--savepath`.
```bash
$ python3 dinov3.py --point 100 150 --point 300 200 -s images/panel.png
# -> images/panel_0.png, images/panel_1.png
```

The output mode can be changed with `--mode`. Default is `similarity`.
```bash
$ python3 dinov3.py --mode pca
$ python3 dinov3.py --mode similarity
$ python3 dinov3.py --mode tracking --video VIDEO_PATH
$ python3 dinov3.py --mode matching --image2 IMAGE2_PATH
```

A foreground mask can be specified with `--mask` (grayscale PNG, pixel > 127 = foreground).
In `pca` mode, PCA is fit only on foreground patches. In `tracking` mode, the mask is used as the initial segmentation for frame 0.
```bash
$ python3 dinov3.py --mode pca --mask MASK_PATH
$ python3 dinov3.py --mode tracking --video VIDEO_DIR_OR_FILE --mask FIRST_FRAME_MASK_PATH
```

## Reference

- [DINOv3](https://github.com/facebookresearch/dinov3)
- [Hugging Face - DINOv3](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[dinov3_vits16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/dinov3/dinov3_vits16.onnx.prototxt)

[dinov3_vitb16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/dinov3/dinov3_vitb16.onnx.prototxt)

[dinov3_vitl16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/dinov3/dinov3_vitl16.onnx.prototxt)

[dinov3_vith16plus.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/dinov3/dinov3_vith16plus.onnx.prototxt)

[dinov3_convnext_tiny.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/dinov3/dinov3_convnext_tiny.onnx.prototxt)
