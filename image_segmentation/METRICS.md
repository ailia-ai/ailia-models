[Model list](./README.md)

# ailia MODELS : Image segmentation

## Models for image segmentation

Generate segmentation the input image by category.

### Cityscapes

|Name|mIoU|Publish Date|
|-----|-----|-----|
|[paddleseg (MscaleOCRNet HRNet_w48)](./paddleseg/)|86.89|2021|
|[hrnet (HRNetV2-W48)](./hrnet_segmentation/)|81.1|2019|
|[hrnet (HRNetV2-W18-Small-v2)](./hrnet_segmentation/)|76.2|2019|
|[hrnet (HRNetV2-W18-Small-v1)](./hrnet_segmentation/)|70.3|2019|

### Ade20k

|Name|mIoU|Publish Date|
|-----|-----|-----|
|[dpt-hybrid](./dense_prediction_transformers/)|49.02|2021|

## Metrics

### mIoU

- PaddleSeg https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.4/contrib/CityscapesSOTA
- HRNet https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1
- DPT https://paperswithcode.com/sota/semantic-segmentation-on-ade20k

## Leader board

Semantic Segmentation
https://paperswithcode.com/task/semantic-segmentation
