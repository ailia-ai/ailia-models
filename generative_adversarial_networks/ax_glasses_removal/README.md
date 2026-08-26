# AX Glasses Removal

## Input

<img src="sample.jpg" width="320px">

(Image from the [Council-GAN sample](../council-GAN/README.md), originally from the CelebA dataset.)

## Output

<img src="output.png" width="320px">

## Pipeline

The sample follows the current JavaScript VTO implementation:

1. FaceMesh-v2 detects one face and estimates 478 landmarks.
2. The eye landmarks define a 2:1 removal region. Areas outside the image are
   padded with `BORDER_REFLECT_101`, as in training.
3. The RGB region is resized to 256 x 128, normalized to `[-1, 1]`, and passed
   to the glasses-removal model.
4. The generated clean image is blended with the input using `frame_mask`, the
   binary mask that the model used for inpainting.

The model also returns the segmenter's raw `probability`. It is available as a
debug overlay with `--show-mask`, but is not used for the final composite.

In video mode, FaceMesh tracks from the preceding landmarks instead of running
the face detector on every frame. Landmark and removal-region motion are
smoothed with the same elapsed-time-based filters as the JavaScript demo.

## Usage

The model files are downloaded automatically on the first run.

```bash
python3 ax_glasses_removal.py
```

Specify an input and output path with `--input` and `--savepath`.

```bash
python3 ax_glasses_removal.py \
  --input IMAGE_PATH \
  --savepath SAVE_IMAGE_PATH
```

Use a video file or webcam (`0`) with `--video`.

```bash
python3 ax_glasses_removal.py \
  --video VIDEO_PATH \
  --savepath SAVE_VIDEO_PATH
```

Overlay the raw glasses probability for debugging.

```bash
python3 ax_glasses_removal.py --show-mask
```

## Model specification

### AX Glasses Removal Mobile

- Model: [ax_glasses_removal_mobile.onnx](https://storage.googleapis.com/ailia-models/ax_glasses_removal/ax_glasses_removal_mobile.onnx)
- Input: `image`, shape `(1, 3, 128, 256)`, RGB, range `[-1, 1]`
- Outputs:
  - `clean`, shape `(1, 3, 128, 256)`, RGB, range `[-1, 1]`
  - `frame_mask`, shape `(1, 1, 128, 256)`
  - `probability`, shape `(1, 1, 128, 256)`

### FaceMesh-v2

- Face detector input: `(1, 128, 128, 3)`, RGB, range `[-1, 1]`
- Landmark detector input: `(1, 256, 256, 3)`, RGB, range `[0, 1]`
- Landmark output: 478 normalized `(x, y, z)` points

## Model format

ailia encrypted ONNX (loaded directly without a prototxt)

## Framework

PyTorch

## Reference

- [Face landmark detection guide](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- [Council-GAN sample image](../council-GAN/README.md)
