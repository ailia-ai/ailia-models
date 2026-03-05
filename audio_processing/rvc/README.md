# Retrieval-based-Voice-Conversion

## Input

Audio file

https://github.com/ailia-ai/ailia-models/assets/29946532/689bba85-b894-4645-bd2a-8abf928733db

(Audio from https://github.com/ohashi3399/RVC-demo)

## Output

Audio file

https://github.com/ailia-ai/ailia-models/assets/29946532/5c036243-a93b-4627-acf0-90bdb911daee

## Requirements

This model requires additional module.
```
pip3 install librosa
pip3 install soundfile
pip3 install faiss-cpu==1.7.3
pip3 install pyworld==0.3.2
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 rvc.py
```

If you want to specify the audio, put the file path after the `--input` option.
```bash
$ python3 rvc.py --input AUDIO_FILE
```

By adding the `--model_file` option, you can specify vc model file.
```bash
$ python3 rvc.py --model_file AISO-HOWATTO.onnx
```

Specify the f0 option to infer a model that uses f0. You can choice `crepe` or `crepe_tiny` for f0_method.

```bash $ 
python3 rvc.py -i booth.wav -m Rinne.onnx --f0_method crepe_tiny --f0 1 --f0_up_key 11 --tgt_sr 48000
```

By adding the `--file_index` option, you can specify faiss feature file.

```bash $ 
python3 rvc.py -i booth.wav -m Rinne.onnx --f0_method crepe --f0 1 --f0_up_key 11 --tgt_sr 48000 --file_index Rinne.index --index_rate 0.75
```

By adding the `--version` option, you can specify rvc model file version.
```bash
$ python3 rvc.py --model_file rvc_v2.onnx --version 2
```

## Reference

- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [RVC向け学習済みボイスモデルデータ](https://chihaya369.booth.pm/items/4701666)

## Framework

Pytorch

## Model Format

ONNX opset=14

## Netron

- [hubert_base.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/rvc/hubert_base.onnx.prototxt)  
- [AISO-HOWATTO.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/rvc/AISO-HOWATTO.onnx.prototxt)
- [crepe.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/rvc/crepe.onnx.prototxt)
- [crepe_tiny.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/rvc/crepe_tiny.onnx.prototxt)
