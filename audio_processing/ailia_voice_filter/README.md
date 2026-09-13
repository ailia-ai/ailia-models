# ailia Voice Filter

Real-time noise reduction for calls, at 48 kHz.

Two graphs ship here. `voicefilter` is conditioned on a d-vector: it keeps the
enrolled speaker and removes everything else, including other voices.
`deepfilternet` is DeepFilterNet3, which denoises but cannot be told a speaker,
so background speech survives it.

Both are streaming exports. Every recurrent tensor is an explicit input and
output, one hop of 480 samples goes in and one comes out, and nothing in the
sample looks ahead beyond what the graph itself does.

## Input

Audio file

- Mixed audio (`mixed.wav`, two speakers)
- Reference audio for the d-vector (`reference.wav`, the speaker to keep)

(Audio from http://swpark.me/voicefilter/)

A second input, `babble_15dB.wav`, is 48 kHz speech in babble noise, used for
the `deepfilternet` example below.

(Audio from https://jmvalin.ca/demo/rnnoise/)

## Output

Audio file

- `output.wav` — the enrolled speaker, extracted from the mixture
- `output_deepfilternet.wav` — `babble_15dB.wav` denoised

## Usage
Automatically downloads the onnx files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample wav,
```bash
$ python3 ailia_voice_filter.py
```

The mixed audio goes after `--input` and the reference audio after
`--reference_file`. `--savepath` changes where the result is written.
```bash
$ python3 ailia_voice_filter.py --input MIXED_WAV --reference_file REFERENCE_WAV --savepath SAVE_PATH
```

`-m deepfilternet` selects DeepFilterNet3. It denoises only, so it takes no
reference audio and refuses one.
```bash
$ python3 ailia_voice_filter.py -m deepfilternet --input babble_15dB.wav --savepath output_deepfilternet.wav
```

Any sample rate is accepted; the audio is resampled to 48 kHz on the way in and
the result is written at 48 kHz.

## Model specification

### ailia Voice Filter (`voicefilter_stream.onnx`)

- 48 kHz, `n_fft` 960, hop 480, Vorbis window at half overlap
- Input: one frame of complex spectrum (`real`, `imaginary`, 481 bins each),
  a 256-d `dvector`, and nine state tensors
- Output: the enhanced spectrum and the nine states
- Latency: 960 samples (20 ms), plus the 480 the analysis framing costs

The graph returns a spectrum, not a mask — the deep filter is complex and spans
5 frames. The caller does the framing and the overlap-add, and multiplies the
spectrum by `1/960` going in and divides by it coming out.

### DeepFilterNet3 (`deepfilternet_stream.onnx`)

- 48 kHz, one hop of 480 samples in and out — the transform is inside the graph
- Output: the denoised hop and eleven states
- Latency: 1440 samples (30 ms)

### Speaker encoder (`speaker_encoder.onnx`)

- 16 kHz, 25600 samples (1.6 s) in, a 256-d GE2E embedding out
- The STFT and the mel filterbank are inside the graph

Enrollment slices the reference audio into 1.6 s partials at 50% overlap,
averages the embeddings and L2 normalises, after normalising the volume to
-30 dBFS. It is not part of the real-time loop: run it once per speaker.

For the first two frames after a reset, six of the states (`erb_cache`,
`df_cache`, `convp_cache`, `enc_h`, `erb_h`, `df_h`) must be discarded and the
previous values sent again. The offline model never sees those frames.

## Measured

`deepfilternet` on `babble_15dB.wav`, framed at 960 samples, comparing the
quietest fifth of frames against the loudest fifth:

| | noise floor | speech |
|:---|---:|---:|
| input | -37.3 dB | -15.7 dB |
| ailia Voice Filter (`-m deepfilternet`) | -56.4 dB | -16.1 dB |
| [rnnoise](../rnnoise/) | -50.3 dB | -16.0 dB |

19 dB of noise removed with the speech level left alone.

`voicefilter` on `mixed.wav` correlates 0.91 with [voicefilter](../voicefilter/)'s
extraction of the same target speaker.

One hop is 10 ms of audio. On an M2 CPU a hop costs 2.0 ms through
`voicefilter` and 2.5 ms through `deepfilternet`, median over a whole file.

## Model format

ailia encrypted ONNX (loaded directly without a prototxt)

## Framework

PyTorch

## Reference

- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- [VoiceSplit](https://github.com/Edresson/VoiceSplit)
- [Real-Time Voice Cloning (GE2E speaker encoder)](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
