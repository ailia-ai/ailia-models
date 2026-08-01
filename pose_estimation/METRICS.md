[Model list](./README.md)

# ailia MODELS : PoseEstimation

## Models for pose estimation

Estimate human pose keypoints from single image.

### Coco

|Name|AP|Method|Publish Date|
|-----|-----|-----|-----|
|[pose_resnet (256x192_pose_resnet_50)](./pose_resnet/)|0.704 (coco2017)|TopDown|2019|
|[openpose](./openpose/)|0.618 (coco2016)|BottomUp|2017|
|[lightweight-human-pose-estimation (mobilenetv1)](./lightweight-human-pose-estimation/)|0.428 (coco2016)|BottomUp|2018|

## Metrics

### AP

- PoseResnet https://github.com/microsoft/human-pose-estimation.pytorch
- LightWeightHumanPose https://arxiv.org/pdf/1811.12004.pdf
- OpenPose https://arxiv.org/pdf/1611.08050.pdf

## Leader board

Pose Estimation
https://paperswithcode.com/task/pose-estimation
