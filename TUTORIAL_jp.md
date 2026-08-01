# ailia MODELS チュートリアル

このチュートリアルでは、このリポジトリのモデルをpythonから実行する方法について解説します。

他の言語（C++/C#(Unity)/JNI/Kotlin/Rust/Flutter）からailiaを使用したい場合は、[Other platforms](README.md#other-platforms) を参照してください。

## 動作条件

- Python 3.7 以降
- git

python、pip、gitの準備ができていない場合は、OSごとの手順を参照してください。
[Python環境のセットアップ](https://docs.ailia.ai/setup/python/)（Windows / Mac / Linux）

## 1. ailia SDKのインストール

以下のコマンドを実行します。（Windowsの場合はコマンドプロンプトやWindows PowerShell、macやLinuxではターミナルで実行）

```
pip3 install ailia
```

ailia SDKは商用ライブラリです。特定の条件下では、無償使用いただけますが、原則として有償ソフトウェアです。詳細は https://ailia.ai/license/ を参照してください。

## 2. ailia MODELSの取得

```
git clone https://github.com/ailia-ai/ailia-models
cd ailia-models
pip3 install -r requirements.txt
```

## 3. 最初のモデルを実行する

各モデルはフォルダごとに分かれており、サンプル入力が同梱されているため、引数なしで実行できます。ONNXファイルは初回実行時に自動的にダウンロードされます。

```
cd object_detection/yolox
python3 yolox.py
```

`input.jpg` から物体を検出し、結果を `output.jpg` に保存します。

他のモデルを試す場合は、[カテゴリ一覧](README.md#models) からモデルを選び、そのフォルダの中にある同名のスクリプトを実行してください。

## コマンドラインオプション

以下のオプションはすべてのモデルで共通です。モデル固有のオプションが追加されている場合があるので、全一覧は `-h` で確認してください。

```
python3 yolox.py -h
```

| オプション | 説明 |
|:---|:---|
| `-i`, `--input` | 入力ファイル。ディレクトリを指定した場合は中にあるファイルすべてに対して推論を実行します。複数指定も可能です。 |
| `-s`, `--savepath` | 出力ファイルの保存先パス。（画像・動画・テキスト） |
| `-v`, `--video` | 映像に対して推論を実行します。int型の引数を指定した場合は、その番号に対応したwebカメラの入力が使われます。 |
| `-b`, `--benchmark` | パフォーマンスを計測するために、同じ入力に対して複数回推論を実行します。ビデオモードでは使用できません。 |
| `-bc`, `--benchmark_count` | ベンチマークモードの実行回数。（デフォルトは5回） |
| `-e`, `--env_id` | 実行環境をenvironment idで指定します。`0` は常にCPUです。デフォルトは `ailia.get_gpu_environment_id` の返り値です。 |
| `--env_list` | 選択できる実行環境の一覧を表示します。 |
| `--ftype` | 入力ファイルの種類: `image` \| `video` \| `audio` |
| `--debug` | デバッグログを出力します。 |
| `--profile` | パフォーマンスのプロファイルログを出力します。 |

画像ファイルを入力し、AIで推論、結果を画像ファイルに保存

```
python3 yolox.py -i input.jpg -s output.jpg
```

動画ファイルを入力し、AIで推論、結果を動画ファイルに保存

```
python3 yolox.py -i input.mp4 -s output.mp4
```

AIの実行時間を計測

```
python3 yolox.py -b
```

AIモデルをGPUではなくCPUで実行

```
python3 yolox.py -e 0
```

選択できる実行環境の一覧を表示

```
python3 yolox.py --env_list
```

カメラからの入力に対してAI推論を実行（終了する際はキーボードの「Q」キーを押す）

```
python3 yolox.py -v 0
```

## GPUアクセラレーション

ailiaはデフォルトでVulkanまたはMetal経由でGPUを使用します。macOSのMetalは設定不要で利用できます。その他のバックエンドについては以下を参照してください。

- [CUDA Toolkit / cuDNN のセットアップ](https://docs.ailia.ai/setup/cuda/)
- [Vulkan のセットアップ](https://docs.ailia.ai/setup/vulkan/)

検出された実行環境は `--env_list` で確認でき、`-e` で選択できます。

## GUI ラウンチャー

以下のコマンドでGUIラウンチャーを表示してマウスで実行することも可能です。（一部のモデルは未対応）

```
python3 launcher.py
```

<img src="launcher.png">

## プラットフォームごとの注意点

### Jetson

[JetsonではOpenCV for python3がプリインストールされています](https://forums.developer.nvidia.com/t/install-opencv-for-python3-in-jetson-nano/74042/3)。cv2 import errorが発生した場合のみ、以下のコマンドを実行してください。

```
sudo apt install nvidia-jetpack
```

`requirements.txt` の一部のパッケージにはaarch64向けのwheelがありません。pipでのビルドに失敗する場合は、aptでインストールしてください。

```
sudo apt install python3-matplotlib python3-scipy
```

### Raspberry Pi

numpyの実行にBLASが必要です。

```
sudo apt-get install libatlas-base-dev
```

Raspberry PiではVulkanが低速なため、サンプルはデフォルトでCPUを使用します。実行環境を明示したい場合は `-e` を指定してください。

## iOS/Android 向けデモアプリ（ストアからダウンロード）

- [ailia AI showcase for iOS](https://apps.apple.com/jp/app/ailia-ai-showcase/id1522828798)
- [ailia AI showcase for Android](https://play.google.com/store/apps/details?id=jp.axinc.ailia_ai_showcase)
- Windows/macOS/Linuxなどのプラットフォーム用は[こちら](<mailto:contact@axinc.jp>)に問い合わせてください

## マニュアル

- [ailia SDK ドキュメント](https://docs.ailia.ai/sdk/)
- [ailia SDK Python API リファレンス](https://docs.ailia.ai/sdk/python/en/)（英語）
