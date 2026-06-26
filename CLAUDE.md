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

トップページのREADME.mdと、scripts/download_all_models.shに追加する。
