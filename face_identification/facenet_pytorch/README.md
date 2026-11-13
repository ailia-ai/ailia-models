# Face Recognition Using Pytorch

## Input
3 face images.
<figure>
<img src="data/angelina_jolie.jpg" width="150">
<img src="data/bradley_cooper.jpg" width="150">
<img src="data/kate_siegel.jpg" width="150">
<legend>angelina_jolie.jpg 　　　bradley_cooper.jpg　　　kate_siegel.jpg</legend>
<img src="data/paul_rudd.jpg" width="150">
<img src="data/shea_whigham.jpg" width="150">
<legend>paul_rudd.jpg 　　　 　　shea_whigham.jpg</legend>
</figure>

(Image from https://github.com/timesler/facenet-pytorch/tree/master/data/test_images)

## Output

A similarity of each pair of images.
```
[['' 'angelina_jolie' 'bradley_cooper' 'kate_siegel' 'paul_rudd' 'shea_whigham']
 ['angelina_jolie' '0.0' '1.4458625' '0.89089304' '1.445407' '1.3876879']
 ['bradley_cooper' '1.4458625' '0.0' '1.3021401' '1.0183644' '1.0345106']
 ['kate_siegel' '0.89089304' '1.3021401' '0.0' '1.4002758' '1.3784742']
 ['paul_rudd' '1.445407' '1.0183644' '1.4002758' '0.0' '1.0893339']
 ['shea_whigham' '1.3876879' '1.0345106' '1.3784742' '1.0893339' '0.0']]
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample images,
```bash
$ python3 facenet_pytorch.py
```

If you want to specify the image folder, put the file path after the `-d (--dir)` option.  
```bash
$ python3 facenet_pytorch.py -d IMG_DIR_PATH
```
`-w (--weight)` option select trained weights with the following datasets.

- vggface2
- casia-webface
```bash
$ python3 facenet_pytorch.py -w casia-webface
```

## Reference

- [facenet_pytorch](https://github.com/timesler/facenet-pytorch)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron
 - [facenet](https://netron.app/?url=https://storage.googleapis.com/ailia-models/facenet-pytorch/vggface2.onnx.prototxt)
