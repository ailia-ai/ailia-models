import sys
import os
import importlib.util

_whisper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "whisper")

# whisper/ の内部モジュール (decode_utils, languages, audio_utils 等) を解決するため
sys.path.insert(0, _whisper_dir)

# lite-whisper は turbo のデコーダ/dims を使用するため -m turbo を強制する
if "--model_type" not in sys.argv and "-m" not in sys.argv:
    sys.argv.extend(["-m", "turbo"])

# 音声ファイルが指定されていない場合は ../whisper/demo.wav をデフォルトにする
if "-i" not in sys.argv and "--input" not in sys.argv:
    sys.argv.extend(["--input", os.path.join(_whisper_dir, "demo.wav")])

# ../whisper/whisper.py をモジュールとしてロード
_whisper_py = os.path.join(_whisper_dir, "whisper.py")
_spec = importlib.util.spec_from_file_location("whisper_main", _whisper_py)
_whisper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_whisper)

# lite-whisper 専用のモデル定数
WEIGHT_ENC_LITE_WHISPER_PATH = "lite-whisper-large-v3-turbo_encoder.opt.onnx"
MODEL_ENC_LITE_WHISPER_PATH = "lite-whisper-large-v3-turbo_encoder.opt.onnx.prototxt"
REMOTE_PATH_LITE_WHISPER = "https://storage.googleapis.com/ailia-models/lite-whisper/"

# whisper モジュールの turbo エンコーダパスを lite-whisper のものに差し替える
# （main() 内の model_dic はこの変数を参照して構築される）
_whisper.WEIGHT_ENC_TURBO_PATH = WEIGHT_ENC_LITE_WHISPER_PATH
_whisper.MODEL_ENC_TURBO_PATH = MODEL_ENC_LITE_WHISPER_PATH
_whisper.WEIGHT_ENC_TURBO_PB_PATH = None  # lite-whisper には pb ファイルが不要

if __name__ == "__main__":
    # lite-whisper エンコーダを専用 remote からダウンロードしてから main() を呼ぶ
    # main() 内でも check_and_download_models が呼ばれるが、
    # ファイルが既に存在するためスキップされる
    _whisper.check_and_download_models(
        WEIGHT_ENC_LITE_WHISPER_PATH, MODEL_ENC_LITE_WHISPER_PATH, REMOTE_PATH_LITE_WHISPER
    )
    _whisper.main()
