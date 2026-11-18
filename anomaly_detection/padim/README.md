# PaDiM-Anomaly-Detection-Localization-master

## Input

Normal images

<img src="bottle_000.png" width="256" height="256">

(Image from MVTec AD datasets https://www.mvtec.com/company/research/datasets/mvtec-ad/)

- Original image shape : (n, 3, 900, 900)
- Input shape : (n, 3, 224, 224)

## Output

Anomaly maps

Left to right: input, ground truth, predicted head map, predicted mask, segmentation result

![Output](output.png)

## Usage with MVTec AD datasets

In order to get the feature vector of the normal product, it is necessary to prepare the file of the normal product.  

By default, normal files are got from the `train` directory.  
For the sample image, first download [MVTec AD datasets](https://www.mvtec.com/company/research/datasets/mvtec-ad/) and place `bottle/train/good/*.png` files to the `train` directory.

![Train](padim_train.png)

For the sample image, train with `train` directory and test with `bottle_000.png`,
```bash
$ python3 padim.py
```

For train and test directly from a particular category in MVTec AD datasets.

```bash
$ python3 padim.py -i bottle/test/broken_large --train_dir bottle/train/good --gt_dir bottle/test/ground_truth/broken_large
```

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

## Usage with your datasets

By default, mask image required to calculate the optimal threshold. By specifying the `threshold` option, it is not necessary to prepare the mask image.

With the following command, learn using train folder and verify with test folder.

```bash
$ python3 padim.py --train_dir train --input test --threshold 0.5
```

Use the following command to perform only the test.

```bash
$ python3 padim.py --feat train.pkl --input test --threshold 0.5
```

Now you can give videos to `train_dir` and `video` option. If a video is given, the first 200 frames of the video will be used for training.

```bash
$ python3 padim.py --train_dir train.mp4 --video test.mp4 --threshold 0.5
```

## Options

You can specify the directory of normal product files with the `--train_dir` option.
```bash
$ python3 padim.py --train_dir train
```

The feature vectors created from files in the train directory can be saved to the pickle, numpy or pytorch file.  For that you have to specify desired format as `npy`, `pt` or `pkl`. Depending on --enable_optimization flag, files saved might be originally pytorch tensors (True) or numpy arrays (False). 
```bash
$ python3 padim.py --save_format pt
```

From the second time, by specifying the saved file by `--feat` option,
it can omit the calculation of the feature vector of the normal product.  
The name of the pickle file created is the same as the name of a normal product file directory.
```bash
$ python3 padim.py --feat train.pkl
```

The ground truth files are got from the `gt_masks` directory by default.  
The name of the ground truth file corresponds to the file with `__mask` after the name of the input file.  
You can specify the directory of ground truth files with the `--gt_dir` option.
```bash
$ python3 padim.py --gt_dir gt_masks
```

If you want to specify the input test image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 padim.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--arch` option, you can specify model type which is selected from "resnet18", "wide_resnet50_2".  
(default is resnet18)
```bash
$ python3 padim.py --arch wide_resnet50_2
```

By adding the `--aug` option, you can process with augmentation.  
(default is processing without augmentation)
```bash
$ python3 padim.py --aug
```

By adding the `--enable_optimization` option, you can use optimized code, which significantly speeds up distance matrix calculation.  You have to have cuda compatible GPU and pytorch installed
(default is processing without optimization)
```bash
$ python3 padim.py --enable_optimization True
```

By adding the `--compare_optimization` option, you can compare output of the optimized code with the output of the original code.  You have to have cuda compatible GPU and pytorch installed
(default is processing without comparison)
```bash
$ python3 padim.py --compare_optimization True
```



## PaDiM GUI

You can also use the GUI to train and test.

<img src="padim_gui.png" width="656" height="370">

Start the GUI with the following command.

```bash
$ python3 padim_gui.py
```
Specify the inference mode from the `Set optimization` button and choose the `Trained file format`.

Specify the folder from the `Select train folder` button and press the `Train button`.

Specify the folder from the `Select test folder` button and press the `Test button`.
Inference results are listed in Result images.

Change the Threshold and press the `Test button` again

If you want to benchmark the inference, set  `Benchmark mode`

## Reference

[PaDiM-Anomaly-Detection-Localization-master](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[resnet18.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/padim/resnet18.onnx.prototxt)

[wide_resnet50_2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/padim/wide_resnet50_2.onnx.prototxt)
