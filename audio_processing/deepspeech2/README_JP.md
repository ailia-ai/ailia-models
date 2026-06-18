# DeepSpeech_ailia
[DeepSpeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)の学習済みモデルを、[ailia SDK](https://ailia.jp)を利用して実行を行います。現在、英語のみが認識対象になります。

# セットアップ

## requirements
requirements.txt参照

## ailia SDKのインストール
[公式のチュートリアル](https://tech.ailia.ai/ailia-sdk-チュートリアル-python-28379dbc9649)を参考にインストールします。

お使いの環境がMac OSの場合、
```
import ailia
```
とモジュールをインポートすると、セキュリティに引っかかり怒られる場合があります。その場合、システム環境設定→セキュリティとプライバシー→一般の「ダウンロードしたアプリケーションを許可」の欄に「"~~"は開発元を確認できないため、開けませんでした。」と表示されるので、「このまま開く」をクリックすると実行が可能になります。三回ほど聞かれるので、繰り返していけば実行出来ます。

## pyaudioのインストール
pyaudioをインストールするために、mac, Linuxの場合はportaudioのインストールが必要です。

Mac OS:
```
brew install portaudio
pip install pyaudio
```

Linux:
```
sudo apt-get install portaudio19-dev
pip install pyaudio
```

Windowsの場合、pythonのバージョンが3.7以上の場合、コンパイルエラーが発生するため、Windowsはpythonのバージョンを3.6以下にして、インストールして下さい。

## ctcdecodeのインストール
言語モデルを利用したBeamDecodeを行うために必要なモジュールです。[リポジトリ](https://github.com/parlance/ctcdecode)からcloneしてインストールを行います。ただし、Windows環境ではコンパイルができないケースがあります。

```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
```
# 使い方
`deepspeech_dynamic.py`が実行ファイルになります。オプションで、音声ファイルを入力とする場合と、PCのマイクから入力するモードを切り替えることができます。
## 音声ファイル入力
サンプリング周波数16kHzの音声ファイルを入力して、予測を行います。
```
python deepspeech_dynamic.py -i <音声ファイルパス> -s <テキストファイル名>
```

## マイク入力
```-V```オプションを指定することで、マイクから入力した音声の予測を行います。
```
python deepspeech_dynamic.py -V
```
実行後、以下のような動作をします。
1. "Please speak something."と表示されたら、マイクに向かって話す
2. 約1秒無音が続くと録音を終了し、音声認識を行う
3. 予測結果を表示後、再び1に戻る
4. 終了したい場合、```Ctrl + c```を入力する

話している最中に録音が終了する、または録音が終わらない場合は、`deepspeech_dynamic.py`の46行目の`THRESHOLD`の値を、お使いのデバイスに合わせて変更します。話してる最中に終了する場合は値を小さく、録音が終わらない場合は値を大きくすることで動作が改善します。

## 共通オプション

### BeamDecoderを利用
`-d`オプションで、言語モデルを利用したBeamDecoderで、認識結果のデコードを行います。

### 学習済みモデルを変更する
`-a`オプションで、他の学習済みモデルを利用することが出来ます。
```
python deepspeech_dynamic.py -a <.onnxファイルのパス>
```