# Bert-VITS2 JP

## Input

**text (```--text```)**

text that will be converted to speech. (Default: 吾輩は猫である)

**emo text (```--emo```)**

text that represents the emotion when being converted to speech. (Default: 私はいまとてもうれしいです)

**emo audio path(optional) (```--emo-audio```)**

path to audio file that represents the emotion when being converted to speech.
If both `--emo` and `--emo-audio` is present, `--emo-audio` will be used as the reference of emotion.

**speaker id (```--sid```)**

specifies the type of voice that will be used. JP characters' id is in the 196 to 427 range. (Default: 340)

**style text(optional)  (```--style-text```)**

the BERT features of this text will be mixed with the BERT features of the


original input, forcibly stylizing the output speech.

## Output

**speech**

Speech converted from text input. Output path can be specified using the argument ```--savepath```

## Usage

before running the sample script, install the required packages
```bash
cd audio_processing/bert-vits2
pip install -r requirements.txt
```

An Internet connection is required when running the script for the first time, as the model files will be downloaded automatically.

Running the script will convert the input text to speech while also considering the meaning of it using the BERT feature extractor.
The emotion the output speech will have is specified by the emo_text (Although this seems to have minimal effect on the output speech).

Running this script in FP16 environments will result in an error due to the range of the floating point expression. Switch to using CPU if necessary. (This is done by setting the argument ```-e``` to 0 in the example below)
```bash
python3 bert-vits2.py --text 吾輩は猫である --emo 私は今とても嬉しいです -e 0
```
The output of this script will be like this.

https://github.com/ailia-ai/ailia-models/assets/53651931/57d9eb49-78eb-4eea-9cde-eb6fbf09ed96


For more information about the arguments, try running ```python3 bert-vits2-jp.py --help```

## Reference

* [Bert-VITS2 JP-Extra](https://github.com/fishaudio/Bert-VITS2/releases/tag/JP-Exta)

## Framework

Pytorch

## Model Format

ONNX opset=12

## Netron


[enc](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert-vits2/BertVits2.2PT_enc_p.onnx.prototxt')

[emb_g](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert-vits2/BertVits2.2PT_emb.onnx.prototxt')

[dp](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert-vits2/BertVits2.2PT_dp.onnx.prototxt')

[sdp](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert-vits2/BertVits2.2PT_sdp.onnx.prototxt')

[flow](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert-vits2/BertVits2.2PT_flow.onnx.prototxt')

[dec](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert-vits2/BertVits2.2PT_dec.onnx.prototxt')

[clap](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert-vits2/emo_clap.onnx.prototxt')

[bert](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert-vits2/debertav2lc.onnx.prototxt')

[clap_audio](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert-vits2/emo_clap_audio.onnx.prototxt')