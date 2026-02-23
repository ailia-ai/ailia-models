# FastSpeech2 : Fast and High-Quality End-to-End Text to Speech

## Input

Text (Supports English and Chinese)

## Output

Audio file (`onnx/result/<dataset>/<text>.wav`).  
Mel spectrogram plot (`onnx/result/<dataset>/<text>.png`).

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

### Single-sentence Synthesis

For the sample text (default):

```bash
$ python3 fastspeech2.py \
  --onnx_fs2 onnx/fastspeech2/ljspeech.onnx \
  --onnx_hifi onnx/hifigan/hifigan_ljspeech.onnx
```

Specify your own text:

```bash
$ python3 fastspeech2.py \
  --onnx_fs2 onnx/fastspeech2/ljspeech.onnx \
  --onnx_hifi onnx/hifigan/hifigan_ljspeech.onnx \
  --text "Hello, this is a test."
```

Specify speaker ID for multi-speaker models:

```bash
$ python3 fastspeech2.py \
  --onnx_fs2 onnx/fastspeech2/ljspeech.onnx \
  --onnx_hifi onnx/hifigan/hifigan_ljspeech.onnx \
  --text "Hello world" \
  --speaker_id 0
```

Control pitch, energy, and speaking rate:

```bash
$ python3 fastspeech2.py \
  --onnx_fs2 onnx/fastspeech2/ljspeech.onnx \
  --onnx_hifi onnx/hifigan/hifigan_ljspeech.onnx \
  --text "Hello world" \
  --pitch_control 1.2 \
  --duration_control 0.8
```

### Multi-Speaker Models

For LibriTTS (English, Multi-Speaker)

```bash
$ python3 fastspeech2.py \
  --text "Hello, I am speaking from a multi-speaker model." \
  --preprocess_config config/LibriTTS/preprocess.yaml \
  --onnx_fs2 onnx/fastspeech2/libritts.onnx \
  --onnx_hifi onnx/hifigan/hifigan.onnx \
  --speaker_id 0
```

For AISHELL-3 (Mandarin, Multi-Speaker):

```bash
$ python3 fastspeech2.py \
  --text "你好" \
  --preprocess_config config/AISHELL3/preprocess.yaml \
  --onnx_fs2 onnx/fastspeech2/aishell3.onnx \
  --onnx_hifi onnx/hifigan/hifigan.onnx \
  --speaker_id 16
```

## Options

### Core Arguments

- `-t`, `--text`: Raw text to synthesize (for single-sentence mode only, default: "Ailia SDK makes it easy to deploy deep learning models.")
- `--speaker_id`: Speaker ID for multi-speaker synthesis (for single-sentence mode only, default: 0)
- `--pitch_control`: Control the pitch of the whole utterance, larger value for higher pitch (default: 1.0)
- `--duration_control`: Control the speed of the whole utterance, larger value for slower speaking rate (default: 1.0)

### Additional Arguments (ailia-specific)

- `--preprocess_config`: Path to preprocess.yaml (default: config/LJSpeech/preprocess.yaml)
- `--onnx_fs2`: Path to FastSpeech2 ONNX file (default: ljspeech.onnx)
- `--onnx_hifi`: Path to HiFi-GAN ONNX file (default: hifigan.onnx)
- `--output_dir`: Output directory for generated audio files (default: onnx/result/ailia)
- `-b`, `--benchmark`: Running the inference on the same input 5 times to measure execution performance
- `--env_id`: The backend environment id

## Model

- [FastSpeech2](https://github.com/ming024/FastSpeech2)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)

## Requirements

- ailia SDK
- g2p_en
- pypinyin
