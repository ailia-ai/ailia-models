# G2PW: A Neural Grapheme-to-Phoneme Converter for Mandarin Chinese

G2PW is a neural network model that converts Chinese (Mandarin) graphemes (characters) into phonemes.

## Input

```
你好世界
```

## Output

**pinyin style**
```
[ni3, hao3, shi4, jie4]
```

**bopomofo style**
```
[ㄋㄧ3, ㄏㄠ3, ㄕ4, ㄐㄧㄝ4]
```

## Usage

Please ensure that the ONNX file is placed in the `G2PWModel/` directory.

To run with the default text:

```bash
$ python3 g2pw.py
```

To specify custom text:

```bash
$ python3 g2pw.py --input '你好世界'
```

To specify the outout style:\
[pinyin]
```bash
$ python3 g2pw.py --style pinyin
```
[Bopomofo]
```bash
$ python3 g2pw.py --style bopomofo
```
## Reference

- [G2PW]  
 https://github.com/GitYCC/g2pW

## Model Format
ONNX opset=12

## Netron

- [g2pW.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/g2pw/1.1/g2pW.onnx.prototxt)

## Python version
Python 3.6.13 