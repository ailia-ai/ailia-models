# ailia MODELS tutorial

In this tutorial we will explain how to use ailia from python language.
If you want to use ailia from other languages(C++/C#(Unity)/JNI/Kotlin/Rust/Flutter) see the link at the bottom of this tutorial.

## Requirements

- Python 3.6 and later

## Install ailia SDK

```
pip3 install ailia
```

The ailia SDK is a commercial library. Under certain conditions, it can be used free of charge; however, it is principally paid software. For details, please refer to https://ailia.ai/license/en/ .

## Install required libraries for Python

### For Windows, Mac and Linux

```
pip install -r requirements.txt
```

### For Jetson

```
sudo apt install python3-pip
sudo apt install python3-matplotlib
sudo apt install python3-scipy
pip3 install cython
pip3 install numpy
pip3 install pillow
```

[OpenCV for python3 is pre-installed on Jetson.](https://forums.developer.nvidia.com/t/install-opencv-for-python3-in-jetson-nano/74042/3) You only need to run this command if you get a cv2 import error.

```
sudo apt install nvidia-jetpack
```

* Note that Jetson Orin require ailia 1.2.13 or above. Please contact us if you would like to use an early build of ailia 1.2.13.

### For Raspberry Pi

```
pip3 install numpy
pip3 install opencv-python
pip3 install matplotlib
pip3 install scikit-image
sudo apt-get install libatlas-base-dev
```

## Options

The following options can be specified for each model.

```
optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE/VIDEO, --input IMAGE/VIDEO
                        The default (model-dependent) input data (image /
                        video) path. If a directory name is specified, the
                        model will be run for the files inside. File type is
                        specified by --ftype argument (default: lenna.png)
  -v VIDEO, --video VIDEO
                        Run the inference against live camera image.
                        If an integer value is given, corresponding
                        webcam input will be used. (default: None)
  -s SAVE_PATH, --savepath SAVE_PATH
                        Save path for the output (image / video / text).
                        (default: output.png)
  -b, --benchmark       Running the inference on the same input 5 times to
                        measure execution performance. (Cannot be used in
                        video mode) (default: False)
  -e ENV_ID, --env_id ENV_ID
                        A specific environment id can be specified. By
                        default, the return value of
                        ailia.get_gpu_environment_id will be used (default: 2)
  --env_list            display environment list (default: False)
  --ftype FILE_TYPE     file type list: image | video | audio (default: image)
  --debug               set default logger level to DEBUG (enable to show
                        DEBUG logs) (default: False)
  --profile             set profile mode (enable to show PROFILE logs)
                        (default: False)
  -bc BENCHMARK_COUNT, --benchmark_count BENCHMARK_COUNT
                        set iteration count of benchmark (default: 5)
```                        

Input an image file, perform AI processing, and save the output to a file.

```
python3 yolov3-tiny.py -i input.png -s output.png
```

Input an video file, perform AI processing, and save the output to a video.

```
python3 yolov3-tiny.py -i input.mp4 -s output.mp4
```

Measure the execution time of the AI model.

```
python3 yolov3-tiny.py -b
```

Run AI model on CPU instead of GPU.

```
python3 yolov3-tiny.py -e 0
```

Get a list of executable environments.

```
python3 yolov3-tiny.py --env_list
```

Run the inference against live video stream.
(Press 'Q' to quit)

```
python3 yolov3-tiny.py -v 0
```

## Launcher

You can use a GUI and select the model from the list using the command below. (Press 'Q' to quit each AI model app)

```
python3 launcher.py
```

<img src="launcher.png">



## Demo application for iOS/Android
- [ailia AI showcase for iOS](https://apps.apple.com/jp/app/ailia-ai-showcase/id1522828798)
- [ailia AI showcase for Android](https://play.google.com/store/apps/details?id=jp.axinc.ailia_ai_showcase)
- Contact [us](<mailto:contact@axinc.jp>) for other platforms (Windows/macOS/Linux)

## API Documentations

- [ailia Documentation](https://docs.ailia.ai/en/)
