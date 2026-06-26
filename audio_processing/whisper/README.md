# Whisper : Robust Speech Recognition via Large-Scale Weak Supervision

## Input

Audio file

https://user-images.githubusercontent.com/29946532/197575850-d4b76831-7a14-41f0-9253-8bcf477b3f7e.mov

## Output

Recognized speech text
```
He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce.
```

## Requirements

This model requires additional module.
```
pip3 install librosa
pip3 install pyaudio  # for microphone input mode
```

If you use `--disable_ailia_tokenizer` option, this model requires additional module.
```
pip3 install transformers
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 whisper.py
```

If you want to specify the audio, put the file path after the `--input` option.
```bash
$ python3 whisper.py --input AUDIO_FILE
```

By adding the `--model_type` option, you can specify model type which is selected from "tiny", "base", "small", "medium". (default is base)
```bash
$ python3 whisper.py --model_type small
```

By giving the `--task translate` option, you can translate it into English.

```bash
$ python3 whisper.py --task translate
```

If you specify the `-V` option, it will be in input mode from the microphone.

```bash
$ python3 whisper.py -V
```

1. speak into the microphone when "Please speak something."
2. end the recording after about 0.5 second of silence and do voice recognition
3. return to 1 again after displaying the forecast results
4. type ``Ctrl+c`` if you want to exit

By adding the `--quantize` option with the turbo model, you can use int4 or int8 quantized models.
```bash
$ python3 whisper.py --model_type turbo --quantize int4
$ python3 whisper.py --model_type turbo --quantize int8
```

The quantized models are exported using `export/export_turbo_quantized.py`.

## Reference

- [Whisper](https://github.com/openai/whisper)
- [Whisper OpenVINO](https://github.com/zhuzilin/whisper-openvino)

## Framework

Pytorch

## Model Format

ONNX opset=11 - 17

## Netron

### Normal models (opset 11)

#### Static Shape (Recommended)

[encoder_tiny.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/encoder_tiny.onnx.prototxt)  
[encoder_base.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/encoder_base.onnx.prototxt)  
[encoder_small.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/encoder_small.onnx.prototxt)  
[encoder_medium.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/encoder_medium.onnx.prototxt)  
[encoder_large.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/encoder_large.onnx.prototxt)  
[encoder_large_v3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/encoder_large_v3.onnx.prototxt)  

[decoder_tiny_fix_kv_cache.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_tiny_fix_kv_cache.onnx.prototxt)  
[decoder_base_fix_kv_cache.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_base_fix_kv_cache.onnx.prototxt)  
[decoder_small_fix_kv_cache.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_small_fix_kv_cache.onnx.prototxt)  
[decoder_medium_fix_kv_cache.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_medium_fix_kv_cache.onnx.prototxt)  
[decoder_large_fix_kv_cache.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_large_fix_kv_cache.onnx.prototxt)  
[decoder_large_v3_fix_kv_cache.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_large_v3_fix_kv_cache.onnx.prototxt)  

#### Dynamic Shape

[decoder_tiny.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_tiny.onnx.prototxt)  
[decoder_base.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_base.onnx.prototxt)  
[decoder_small.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_small.onnx.prototxt)  
[decoder_medium.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_medium.onnx.prototxt)  
[decoder_large.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_large.onnx.prototxt)

### Mean Variance Normalization models (opset 11)

[encoder_tiny.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/encoder_tiny.opt.onnx.prototxt)  
[encoder_base.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/encoder_base.opt.onnx.prototxt)  
[encoder_small.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/encoder_small.opt.onnx.prototxt)  
[encoder_medium.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/encoder_medium.opt.onnx.prototxt)  

[decoder_tiny_fix_kv_cache.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_tiny_fix_kv_cache.opt.onnx.prototxt)  
[decoder_base_fix_kv_cache.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_base_fix_kv_cache.opt.onnx.prototxt)  
[decoder_small_fix_kv_cache.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_small_fix_kv_cache.opt.onnx.prototxt)  
[decoder_medium_fix_kv_cache.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_medium_fix_kv_cache.opt.onnx.prototxt)  

### Fuse ScatterND models (opset 11)

[decoder_tiny_fix_kv_cache.opt2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_tiny_fix_kv_cache.opt2.onnx.prototxt)  
[decoder_base_fix_kv_cache.opt2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_base_fix_kv_cache.opt2.onnx.prototxt)  
[decoder_small_fix_kv_cache.opt2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_small_fix_kv_cache.opt2.onnx.prototxt)  
[decoder_medium_fix_kv_cache.opt2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_medium_fix_kv_cache.opt2.onnx.prototxt)  

### Layer Normalization models (opset 17)

[encoder_tiny.opt3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/encoder_tiny.opt3.onnx.prototxt)  
[encoder_base.opt3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/encoder_base.opt3.onnx.prototxt)  
[encoder_small.opt3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/encoder_small.opt3.onnx.prototxt)  
[encoder_medium.opt3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/encoder_medium.opt3.onnx.prototxt)  
[encoder_turbo.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/encoder_turbo.opt.onnx.prototxt)  

[decoder_tiny_fix_kv_cache.opt3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_tiny_fix_kv_cache.opt3.onnx.prototxt)  
[decoder_base_fix_kv_cache.opt3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_base_fix_kv_cache.opt3.onnx.prototxt)  
[decoder_small_fix_kv_cache.opt3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_small_fix_kv_cache.opt3.onnx.prototxt)  
[decoder_medium_fix_kv_cache.opt3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_medium_fix_kv_cache.opt3.onnx.prototxt)  
[decoder_turbo_fix_kv_cache.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/whisper/decoder_turbo_fix_kv_cache.opt.onnx.prototxt)  
