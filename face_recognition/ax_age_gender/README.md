# ax Age Gender

## Input

![Input](demo.jpg)

(Image
from https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/age-gender-recognition-retail-0013/assets/age-gender-recognition-retail-0001.jpg)

### Face Detector: BlazeFace

- ailia input shape: (1, 3, 256, 256) RGB channel order
- Pixel value range: [-1, 1]

### Age Gender Estimator: ax Age Gender

- ailia input shape: (batch_size, 3, 128, 128) RGB channel order
- Pixel value range: [0, 1] before normalization
- Preprocessing: normalization using ImageNet statistics
- The face is aligned from the BlazeFace eye keypoints into a canonical
  128x128 crop (inter-ocular distance 0.30 of the crop, eye line at 0.40)
  before inference.

## Output

![Output](output.png)

- Estimating gender, age and mask wearing
```bash
### Estimating gender and age ###
gender is: Female (95.61)
age is: 28.1
mask is: nomask (99.95)
```

The person segmentation used by the model is outlined in yellow. The detected
face-mask region is outlined and filled in magenta. Pass `--no-segmentation`
to hide these overlays.

Faces whose head angle falls outside the range the model is accurate on are
detected, boxed in gray and left unscored. Pass `--no-pose-gate` to score
them anyway.

## Usage

Automatically downloads the onnx files on the first run. It is necessary to
be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 ax_age_gender.py
```

If you want to specify the input image, put the image path after the `--input` option.
```bash
$ python3 ax_age_gender.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
You can use --savepath option to specify the output file to save.
```bash
$ python3 ax_age_gender.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```

In video mode, predictions are smoothed for each tracked face by default. The
full age distribution, gender probabilities and mask probabilities are
averaged with an exponential smoothing factor of 0.1. Use `--smooth-alpha 0`
to disable smoothing or specify another value between 0 and 1.

BlazeFace is applied to the full frame, which is intended for portraits and
webcam images where faces are already large enough for the 256x256 detector.

### Additional options

- `--no-pose-gate`: Estimate faces outside the recommended head-pose range.
- `--smooth-alpha`: Set the video smoothing factor; 0 disables smoothing.
- `--no-segmentation`: Hide the person and face-mask segmentation overlays.

## Reference

This model was developed by ailia Inc.
The age is predicted as a distribution over [0, 100] and reduced by
expectation. The model also classifies whether the face is wearing a mask,
and stays accurate on masked faces thanks to the eyes-only alignment.

## Framework

Pytorch

## Model Format

ONNX opset = 11 (FP16)

## Netron

- [ax_age_gender_fp16.onnx](https://storage.googleapis.com/ailia-models/ax_age_gender/ax_age_gender_fp16.onnx)
