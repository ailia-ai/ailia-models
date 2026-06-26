# LatentSync: Audio Conditioned Latent Diffusion Models for Lip Sync

## Input

Video file:

    https://github.com/bytedance/LatentSync/blob/main/assets/demo1_video.mp4

Audio file:

    https://github.com/bytedance/LatentSync/blob/main/assets/demo1_audio.wav

## Output

Video file

## Requirements
This model requires additional module.

```
pip3 install face_alignment
pip3 install librosa
```

Install ffmepg package.

```
apt-get install ffmpeg
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample video, audio,
```bash
$ python3 latentsync.py
```

If you want to specify the video and audio files, put the file paths after the `--video` and `--input` options.  
You can use the `--savepath` option to change the name of the output file to save.
```bash
$ python3 latentsync.py --input AUDIO_FILE --video VIDEO_FILE --savepath OUTPUT_FILE
```

## Reference

- [LatentSync](https://github.com/bytedance/LatentSync/tree/main)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[unet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/latentsync/unet.onnx.prototxt)  
[vae_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/latentsync/vae_encoder.onnx.prototxt)  
[vae_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/latentsync/vae_decoder.onnx.prototxt)  
[whisper_tiny.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/latentsync/whisper_tiny.onnx.prototxt)  
