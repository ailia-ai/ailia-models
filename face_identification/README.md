# ailia MODELS : FaceIdentification

## Models for face identification

Identify person from face.

### Cityscapes

|Name|Accuracy (calculated)|Accuracy (paper)|Training Dataset|Publish Date|
|-----|-----|-----|-----|-----|
|[arcface (resnet18)](./arcface/)|99.93|99.83 (resnet100) (LFW)|LFW|2018|
|[vggface2 (resnet50)](./vggface2/)|99.91|0.891 (TAR@FAR=0.001), 0.947 (TAR@FAR=0.01)|VGGFace2|2018|

## Metrics

### Accuracy (calculated)

We calculated accuracy with LFW-dataset (only 100 person) using this script.

https://github.com/ailia-ai/ailia-models-measurement/tree/main/face_identification

### Accuracy (paper)

We referred accuracy from this paper.

- ArcFace https://arxiv.org/pdf/1801.07698.pdf
- VGGFace2 https://github.com/ox-vgg/vgg_face2 https://github.com/ox-vgg/vgg_face2

## Leader board

Face Identification
https://paperswithcode.com/task/face-identification
