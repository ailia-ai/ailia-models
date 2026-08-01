# ailia MODELS tutorial

This tutorial explains how to run the models in this repository from Python.

To try it without installing anything, open [hello_ailia.ipynb](hello_ailia.ipynb) in [Google Colaboratory](https://colab.research.google.com/github/ailia-ai/ailia-models/blob/master/hello_ailia.ipynb). It installs the SDK and runs object detection in a few cells.

To use ailia from another language (C++ / C# (Unity) / JNI / Kotlin / Rust / Flutter), see [Other platforms](README.md#other-platforms).

## Requirements

- Python 3.9 to 3.12 (`numpy<2.0` in requirements.txt has no wheel for 3.13 and later)
- git

If Python, pip or git are not set up yet, follow the guide for your OS:
[Python environment setup](https://docs.ailia.ai/en/setup/python/) (Windows / Mac / Linux)

## 1. Install ailia SDK

```
pip3 install ailia
```

The ailia SDK is a commercial library, but it can be used free of charge under certain conditions, including personal non-commercial use and commercial use where the total economic benefit over 12 months is below 100,000 USD. Crediting the ailia SDK is required for the free tiers.

You do not need to set up a license to get started: when you install with pip, an evaluation license file is downloaded automatically and renewed every 30 days.

For the exact terms, please refer to https://ailia.ai/license/en/ .

## 2. Get ailia MODELS

```
git clone https://github.com/ailia-ai/ailia-models
cd ailia-models
pip3 install -r requirements.txt
```

## 3. Run your first model

Each model lives in its own folder and comes with a sample input, so it runs with no arguments. The ONNX file is downloaded automatically on the first run.

```
cd object_detection/yolox
python3 yolox.py
```

This detects objects in `input.jpg` and writes the result to `output.jpg`.

Note that `output.jpg` is a file tracked by git, so `git status` will report it as modified after the run. Every model works this way. Pass `-s` to write the result somewhere else if you want to keep your checkout clean.

To try another model, pick one from the [category list](README.md#models) and run the script of the same name inside its folder.

## Command line options

The options below are shared by every model. Some models add their own; run the script with `-h` to see the full list.

```
python3 yolox.py -h
```

| Option | Description |
|:---|:---|
| `-i`, `--input` | Input file. If a directory is given, every file inside is processed. Multiple paths can be listed. |
| `-s`, `--savepath` | Save path for the output (image / video / text). |
| `-v`, `--video` | Run against a video stream. An integer selects the corresponding webcam. |
| `-b`, `--benchmark` | Run the inference on the same input several times to measure execution performance. Cannot be used in video mode. |
| `-bc`, `--benchmark_count` | Iteration count of the benchmark (default 5). |
| `-e`, `--env_id` | Run on a specific environment. `0` is always CPU. Defaults to the return value of `ailia.get_gpu_environment_id`. |
| `--env_list` | Display the list of available environments. |
| `--ftype` | File type of the input: `image` \| `video` \| `audio`. |
| `--debug` | Show DEBUG logs. |
| `--profile` | Show PROFILE logs. |

Input an image file, perform AI processing, and save the output to a file.

```
python3 yolox.py -i input.jpg -s output.jpg
```

Input a video file, perform AI processing, and save the output to a video.

```
python3 yolox.py -i input.mp4 -s output.mp4
```

Measure the execution time of the AI model.

```
python3 yolox.py -b
```

Run the AI model on CPU instead of GPU.

```
python3 yolox.py -e 0
```

Get a list of executable environments.

```
python3 yolox.py --env_list
```

Run the inference against a live video stream. (Press 'Q' to quit)

```
python3 yolox.py -v 0
```

## GPU acceleration

ailia runs on the GPU through Vulkan or Metal by default. Metal is available on macOS without any setup. For the other backends, see:

- [CUDA Toolkit / cuDNN setup](https://docs.ailia.ai/en/setup/cuda/)
- [Vulkan setup](https://docs.ailia.ai/en/setup/vulkan/)

Use `--env_list` to check which environments were detected, and `-e` to pick one.

## Launcher

You can use a GUI and select the model from the list using the command below. (Press 'Q' to quit each AI model app)

```
python3 launcher.py
```

The launcher uses tkinter, which is not bundled with Python on some Linux distributions. If you get a tkinter import error, install it with your package manager.

```
sudo apt install python3-tk
```

<img src="launcher.png">

## Platform notes

### Jetson

OpenCV for python3 is [pre-installed on Jetson](https://forums.developer.nvidia.com/t/install-opencv-for-python3-in-jetson-nano/74042/3). Run this command only if you get a cv2 import error.

```
sudo apt install nvidia-jetpack
```

Some packages in `requirements.txt` have no wheel for aarch64. If pip fails to build them, install them with apt instead.

```
sudo apt install python3-matplotlib python3-scipy
```

### Raspberry Pi

numpy needs BLAS.

```
sudo apt-get install libatlas-base-dev
```

Vulkan is slow on the Raspberry Pi, so the samples fall back to the CPU by default. Pass `-e` if you want to force a specific environment.

## Demo application for iOS/Android

- [ailia AI showcase for iOS](https://apps.apple.com/jp/app/ailia-ai-showcase/id1522828798)
- [ailia AI showcase for Android](https://play.google.com/store/apps/details?id=jp.axinc.ailia_ai_showcase)
- Contact [us](<mailto:contact@axinc.jp>) for other platforms (Windows/macOS/Linux)

## Documentation

- [ailia SDK documentation](https://docs.ailia.ai/en/sdk/)
- [ailia SDK Python API reference](https://docs.ailia.ai/sdk/python/en/)
