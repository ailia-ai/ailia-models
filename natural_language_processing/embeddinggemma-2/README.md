# EmbeddingGemma 2

EmbeddingGemma 2 is a multimodal embedding model by Google DeepMind built on Gemma 4.
It maps text, images, video and audio into a single 768-dimensional vector space,
so that any input can be compared with any other by cosine similarity.

## Input

Queries are texts. Documents (the targets to search) can be texts, images, videos and audio files.

- Documents

  | cats.jpg | jfk.wav | sea-turtle.mp4 |
  |:---:|:---:|:---:|
  | <img src="cats.jpg" width=200px> | speech (11 s) | video (8 s) |

  (Files from https://huggingface.co/datasets/Xenova/transformers.js-docs)

- Queries

  ```
  - cats sleeping on a couch
  - a president's speech about serving your country
  - a turtle swimming in the ocean
  ```

## Output

Similarity (rows: queries, columns: documents) and the documents in order of relevance for each query.

```
[[0.7456 0.4989 0.5014]
 [0.4612 0.7719 0.4902]
 [0.501  0.4831 0.7279]]
Query: cats sleeping on a couch
  [1] (0.7456) image: cats.jpg
  [2] (0.5014) video: sea-turtle.mp4
  [3] (0.4989) audio: jfk.wav
Query: a president's speech about serving your country
  [1] (0.7719) audio: jfk.wav
  [2] (0.4902) video: sea-turtle.mp4
  [3] (0.4612) image: cats.jpg
Query: a turtle swimming in the ocean
  [1] (0.7279) video: sea-turtle.mp4
  [2] (0.5010) image: cats.jpg
  [3] (0.4831) audio: jfk.wav
```

## Requirements

This model requires additional module.

```
pip3 install librosa
```

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample documents and queries,
```bash
$ python3 embeddinggemma-2.py
```

If you want to specify the queries, put the query texts after the `--query` option.
```bash
$ python3 embeddinggemma-2.py --query "QUERY1" "QUERY2"
```

The documents (the targets to search) can be texts, images, videos and audio files, and are given with two options.

- `--input`: documents given as image, video and audio files.
  The type of each file is determined by the extension: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` are videos,
  `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac` are audio files and the others are images.
- `--document`: documents given as texts.

When neither option is given, the sample image, audio and video are used.
```bash
$ python3 embeddinggemma-2.py --input IMAGE_PATH VIDEO_PATH AUDIO_PATH --document "TEXT1" "TEXT2"
```

The `--input`, `--query` and `--document` options can also be repeated.
```bash
$ python3 embeddinggemma-2.py -i IMAGE_PATH -i VIDEO_PATH -q "QUERY1" -q "QUERY2"
```

Text search only,
```bash
$ python3 embeddinggemma-2.py \
    --query "What causes the northern lights?" \
    --document "The northern lights are caused by charged particles from the sun."
```

By adding the `--ffmpeg` option, the audio files are decoded and resampled to 16 kHz mono with the `ffmpeg` command instead of librosa.
The scores of stereo or non-16 kHz audio change slightly (e.g. 0.7719 -> 0.7702 for jfk.wav).
```bash
$ python3 embeddinggemma-2.py --input AUDIO_PATH --ffmpeg
```

By specifying the `--truncate_dim` option, you can shorten the embeddings to 128, 256 or 512 dimensions (Matryoshka Representation Learning).
```bash
$ python3 embeddinggemma-2.py --truncate_dim 256
```

## Reference

- [Hugging Face - google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)
- [EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[embeddinggemma-2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/embeddinggemma-2/embeddinggemma-2.onnx.prototxt)

[embeddinggemma-2_vision.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/embeddinggemma-2/embeddinggemma-2_vision.onnx.prototxt)

[embeddinggemma-2_audio.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/embeddinggemma-2/embeddinggemma-2_audio.onnx.prototxt)
