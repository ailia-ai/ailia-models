# ONNXエクスポートの進め方

## 概要

URLで指定されたモデルをONNXに変換して、ailia MODELSに追加する。
エクスポートスクリプトは各モデルの/export/フォルダに格納する。
既存のモデルの構成を踏襲する。
必ずブランチで作業する。

## 必要事項

公開されている全てのモデルサイズを引数で選択できるようにする。
実際にエクスポートしてONNXを生成する。
ONNXと合わせてprototxtが必要。prototxtは下記を使用して生成する。
https://github.com/ailia-ai/export-to-onnx/blob/master/onnx2prototxt.py
元のリポジトリのライセンスファイルもコミットする。

## 生成したモデルのアップロード

生成したONNXとprototxtは下記にアップロードする。
https://console.cloud.google.com/storage/browser/ailia-models

## サンプルの作成

実際に推論を行ってテストする。

## リストへの追加

モデルリストはカテゴリごとに分割されている。
モデルが属するカテゴリのフォルダのREADME.md（例：物体検出ならobject_detection/README.md）と、
scripts/download_all_models.shに追加する。
カテゴリのREADME.md内のパスは、そのフォルダからの相対パス（例：./yolov7/）で記述する。

トップページのREADME.mdにはモデル個別の記載はしない。
カテゴリ一覧の表にあるモデル数と、その上のモデル総数を更新する。
新しいカテゴリを作成した場合のみ、トップページのREADME.mdに行を追加する。

一部のカテゴリのフォルダには、精度比較用のMETRICS.mdがある。
精度を計測した場合はそちらに追加する。
