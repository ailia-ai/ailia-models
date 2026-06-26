# lite-whisper-large-v3-turbo : Low-Rank Compressed Whisper Encoder

## Input

Audio file

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
$ python3 lite-whisper.py
```

If you want to specify the audio, put the file path after the `--input` option.
```bash
$ python3 lite-whisper.py --input AUDIO_FILE
```

By giving the `--task translate` option, you can translate it into English.
```bash
$ python3 lite-whisper.py --task translate
```

If you specify the `-V` option, it will be in input mode from the microphone.
```bash
$ python3 lite-whisper.py -V
```

1. speak into the microphone when "Please speak something."
2. end the recording after about 0.5 second of silence and do voice recognition
3. return to 1 again after displaying the forecast results
4. type ``Ctrl+c`` if you want to exit

## Reference

- [LiteASR](https://github.com/efeslab/LiteASR)
- [lite-whisper-large-v3-turbo](https://huggingface.co/efficient-speech/lite-whisper-large-v3-turbo)
- [Whisper](https://github.com/openai/whisper)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[lite-whisper-large-v3-turbo_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/lite-whisper/lite-whisper-large-v3-turbo_encoder.onnx.prototxt)  
[lite-whisper-large-v3-turbo_encoder.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/lite-whisper/lite-whisper-large-v3-turbo_encoder.opt.onnx.prototxt)  
