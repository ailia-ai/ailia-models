"""
Convert SDXL fp32 ONNX models (UNet / text encoders) to fp16.

Tested versions:
    onnx : 1.13.1
    onnxconverter-common : 1.15.0

Usage:
    python convert_to_fp16.py --model all
    python convert_to_fp16.py --model unet

The fp32 models (sdxl_*.onnx + sdxl_*_weights.pb) are expected in the
parent directory (downloaded automatically by sdxl.py on the first run).

The converted models keep fp32 inputs/outputs (keep_io_types=True), so
sdxl.py can feed the same fp32 tensors to both fp32 and fp16 models.
Weights are always saved as external data (sdxl_*_fp16_weights.pb) to keep
the same file layout as the fp32 models; the UNet exceeds the 2GB protobuf
limit anyway.

The VAE decoder is intentionally not converted: the SDXL fp32 VAE overflows
in fp16 (NaN in the attention block), which is why stabilityai published the
separate sdxx-vae-fp16-fix weights.
"""

import argparse
import os
import subprocess
import sys
import urllib.request

import onnx
from onnx import TensorProto
from onnxconverter_common.float16 import convert_float_to_float16

MODELS = {
    "unet": "sdxl_unet",
    "clip_l": "sdxl_text_encoder_clip_l",
    "open_clip_bigg": "sdxl_text_encoder_open_clip_bigg",
    "refiner_unet": "sdxl_refiner_unet",
}

# DEFAULT_OP_BLOCK_LIST of onnxconverter-common + Pow.
# Pow を fp32 に残さないと、fp16 変換後に onnx 側で fp32 に戻されて
# initializer (fp16) と不一致になるエラーが出る (whisper と同じ)。
OP_BLOCK_LIST = [
    "ArrayFeatureExtractor",
    "Binarizer",
    "CastMap",
    "CategoryMapper",
    "DictVectorizer",
    "FeatureVectorizer",
    "Imputer",
    "LabelEncoder",
    "LinearClassifier",
    "LinearRegressor",
    "Normalizer",
    "OneHotEncoder",
    "RandomUniformLike",
    "SVMClassifier",
    "SVMRegressor",
    "Scaler",
    "TreeEnsembleClassifier",
    "TreeEnsembleRegressor",
    "ZipMap",
    "NonMaxSuppression",
    "TopK",
    "RoiAlign",
    "Resize",
    "Range",
    "CumSum",
    "Min",
    "Max",
    "Upsample",
    "Pow",
]


def clear_external_refs(model):
    """Drop external data references so the in-memory raw_data is used."""
    for tensor in model.graph.initializer:
        while len(tensor.external_data) > 0:
            tensor.external_data.pop()
        tensor.ClearField("data_location")


def retarget_cast_nodes(model):
    """Rewrite pre-existing Cast(to=float32) nodes to Cast(to=float16).

    convert_float_to_float16 は元モデルに含まれる Cast ノードの to 属性を
    書き換えないため、CLIP/UNet の attention 内にある Cast(to=float32) が
    fp32 のまま残り、fp16 化された後続ノードと型不整合になる。
    keep_io_types / op_block_list のために converter 自身が挿入した
    境界 Cast (グラフ出力へ出すものと、ブロック対象 op へ入れるもの) は
    fp32 のまま残す必要があるので除外する。
    """
    graph = model.graph
    output_names = {o.name for o in graph.output}
    consumers = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    changed = 0
    for node in graph.node:
        if node.op_type != "Cast":
            continue
        to_attr = next(a for a in node.attribute if a.name == "to")
        if to_attr.i != TensorProto.FLOAT:
            continue
        out = node.output[0]
        if out in output_names:
            continue  # keep_io_types の出力境界 Cast
        if any(c.op_type in OP_BLOCK_LIST for c in consumers.get(out, [])):
            continue  # ブロック対象 op (fp32 実行) への入力 Cast
        to_attr.i = TensorProto.FLOAT16
        changed += 1
    print(f"  Retargeted {changed} Cast nodes to fp16")


def generate_prototxt(onnx_path):
    """Generate prototxt from ONNX model using onnx2prototxt.py."""
    script_url = (
        "https://raw.githubusercontent.com/ailia-ai/export-to-onnx/"
        "master/onnx2prototxt.py"
    )
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "onnx2prototxt.py")
    if not os.path.exists(script_path):
        print("  Downloading onnx2prototxt.py ...")
        urllib.request.urlretrieve(script_url, script_path)

    print(f"  Generating prototxt for {onnx_path} ...")
    subprocess.check_call([sys.executable, script_path, onnx_path])


def convert(stem, model_dir):
    src = os.path.join(model_dir, stem + ".onnx")
    dst = os.path.join(model_dir, stem + "_fp16.onnx")
    if not os.path.exists(src):
        print(f"  {src} not found, skipping. (run sdxl.py once to download)")
        return

    print(f"  Loading {src} ...")
    # 2GB 超のモデルは infer_shapes がシリアライズできないため、重みなしで
    # ロードして shape 推論し (op_block_list の境界 Cast 挿入に型情報が必要)、
    # その後で外部データを読み込む。
    model = onnx.load(src, load_external_data=False)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.external_data_helper.load_external_data_for_model(
        model, os.path.abspath(model_dir)
    )
    clear_external_refs(model)

    print("  Converting to fp16 ...")
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,  # value_info は上で付与済み
        op_block_list=OP_BLOCK_LIST,
    )
    del model
    retarget_cast_nodes(model_fp16)

    print(f"  Saving {dst} ...")
    dst_pb = os.path.join(model_dir, stem + "_fp16_weights.pb")
    for path in (dst, dst_pb):
        if os.path.exists(path):
            os.remove(path)  # 外部データは追記書き込みされるため先に消す
    onnx.save(
        model_fp16,
        dst,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=stem + "_fp16_weights.pb",
        size_threshold=1024,
    )
    del model_fp16

    generate_prototxt(dst)

    src_size = os.path.getsize(src.replace(".onnx", "_weights.pb"))
    dst_size = os.path.getsize(dst.replace(".onnx", "_weights.pb"))
    print(f"  {stem}: {src_size / 1024**2:.0f}MB -> {dst_size / 1024**2:.0f}MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SDXL ONNX models to fp16"
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=list(MODELS.keys()) + ["all"],
        help="model to convert",
    )
    parser.add_argument(
        "--model_dir",
        default="..",
        help="directory containing the fp32 ONNX models",
    )
    args = parser.parse_args()

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]
    for name in targets:
        print(f"[{name}]")
        convert(MODELS[name], args.model_dir)


if __name__ == "__main__":
    main()
