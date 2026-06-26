# CosyVoice2

## Input

Audio file

- Audio file

  https://github.com/FunAudioLLM/CosyVoice/blob/main/asset/zero_shot_prompt.wav

- TTS Text

  收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。

- Prompt Text

  希望你以后能够做的比我还好呦。


## Output

- Audio file

  output.wav

## Requirements

This model requires additional module.
```
pip3 install librosa
pip3 install soundfile
pip3 install inflect
pip3 install WeTextProcessing==1.0.3
```

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 cosyvoice2.py
```

If you want to specify audio and prompt_text, put the file path after the `--input` option and the prompt_text after the `--prompt_text` option.
```bash
$ python3 cosyvoice2.py --input zero_shot_prompt.wav --prompt_text "希望你以后能够做的比我还好呦。"
```

You can specify tts_text by adding the `--tts_text` option.
```bash
$ python3 cosyvoice2.py --tts_text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
```

## Reference

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice/tree/main)
- [CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models](https://funaudiollm.github.io/cosyvoice2/)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

- [speech_tokenizer_v2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/cosyvoice/speech_tokenizer_v2.onnx.prototxt)  
- [campplus.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/cosyvoice/campplus.onnx.prototxt)
- [CosyVoice2-0.5B_embed_tokens.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/cosyvoice/CosyVoice2-0.5B_embed_tokens.onnx.prototxt)
- [CosyVoice2-0.5B_flow_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/cosyvoice/CosyVoice2-0.5B_flow_encoder.onnx.prototxt)
- [CosyVoice2-0.5B_flow.decoder.estimator.fp32.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/cosyvoice/CosyVoice2-0.5B_flow.decoder.estimator.fp32.onnx.prototxt)
- [CosyVoice2-0.5B_hift.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/cosyvoice/CosyVoice2-0.5B_hift.onnx.prototxt)

