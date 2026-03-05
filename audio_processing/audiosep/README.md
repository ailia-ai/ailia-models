# AudioSep: Separate Anything You Describe

## Input

* **Mixed audio file**

Audio file in wav format with mixed sources. [input.wav](./input.wav)

https://github.com/ailia-ai/ailia-models/assets/53651931/4b761212-a1c7-46dc-b598-a08e4c5ab7ff

This audio file was adapted from the [official audiosep implementation](https://github.com/Audio-AGI/AudioSep)

https://audio-agi.github.io/Separate-Anything-You-Describe/demos/exp31_water/drops_mixture.wav

* **Text condition**

Text description of the sound source you want to separate.

## Output

* **Audio file**

Separated audio source according to the text query.

Saves to ```./output.wav``` by default but it can be specified with the ```--path``` option 

## Usage
Internet connection is required when running the script for the first time, as the model files will be automatically downloaded.

Running this script will separate sound sources from the original input audio file, according to the language query.

#### Example1: Extract sound of thunder
```bash
$ python3 audiosep.py -p "thunder" -i input.wav -s output_thunder.wav
```
https://github.com/ailia-ai/ailia-models/assets/53651931/d0d016dd-a808-4eb6-a4b5-9791f8f1bd2f

#### Example2: Extract sound of waterdrops
```bash
$ python3 audiosep.py -p "water drops" -i input.wav -s output_waterdrops.wav
```
https://github.com/ailia-ai/ailia-models/assets/53651931/7710b6c9-49dc-4d2a-8489-ccbf7fb45591

```.wav``` file containing the sound source separated from the original mixture will be created in both cases.

## Reference

* [AudioSep](https://github.com/Audio-AGI/AudioSep)
* [Separate Anything You Describe](https://audio-agi.github.io/Separate-Anything-You-Describe/)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

* [audiosep_text.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/audiosep/audiosep_text.onnx.prototxt)
* [audiosep_resunet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/audiosep/audiosep_resunet.onnx.prototxt)



