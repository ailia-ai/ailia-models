import sys
import os
import time
import platform
import numpy as np
import queue

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa
from microphone_utils import start_microphone_input  # noqa

# logger
from logging import getLogger  # noqa: E402
logger = getLogger(__name__)



# ======================
# Parameters
# ======================
WAV_PATH = "demo.wav"
SAVE_TEXT_PATH = "output.txt"

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser("Sherpa-onnx", WAV_PATH, SAVE_TEXT_PATH, input_ftype="audio")
# ... (引数定義は後で追加)

# ======================
# Models
# ======================
from collections import namedtuple

# モデルの次元情報など、共通の構造が必要な場合はnamedtupleを定義
SherpaOnnxModelDims = namedtuple(
    "SherpaOnnxModelDims",
    [
        "sample_rate",
        "feature_dim",
    ],
)

# sherpa-onnxモデルの識別名定数
MODEL_ZIPFORMER_JA_REAZONSPEECH = "zipformer-ja-reazonspeech"
MODEL_ZIPFORMER_MULTI_LANG_STREAMING = "zipformer-multi-lang-streaming"
MODEL_PARAFORMER_BILINGUAL_STREAMING = "paraformer-bilingual-streaming"
# 他のモデルタイプをここに追加

# モデルの詳細辞書
MODEL_CONFIGS = {
    MODEL_ZIPFORMER_JA_REAZONSPEECH: {
        "model_type": "transducer",
        "remote_base_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/",
        "model_dir_name": "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01",
        "files": {
            "tokens": "tokens.txt",
            "encoder": "encoder-epoch-99-avg-1.onnx",
            "decoder": "decoder-epoch-99-avg-1.onnx",
            "joiner": "joiner-epoch-99-avg-1.onnx",
        },
        "dims": SherpaOnnxModelDims(sample_rate=16000, feature_dim=80),
        "languages": ["ja"],
        "description": "Japanese offline Zipformer-Transducer model trained on ReazonSpeech.",
        "is_streaming": False,
    },
    MODEL_ZIPFORMER_MULTI_LANG_STREAMING: {
        "model_type": "transducer",
        "remote_base_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/",
        "model_dir_name": "sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10",
        "files": {
            "tokens": "tokens.txt",
            "encoder": "encoder-epoch-75-avg-11-chunk-16-left-128.int8.onnx",
            "decoder": "decoder-epoch-75-avg-11-chunk-16-left-128.onnx",
            "joiner": "joiner-epoch-75-avg-11-chunk-16-left-128.int8.onnx",
        },
        "dims": SherpaOnnxModelDims(sample_rate=16000, feature_dim=80),
        "languages": ["ar", "en", "id", "ja", "ru", "th", "vi", "zh"],
        "description": "Multi-lingual streaming Zipformer-Transducer model.",
        "is_streaming": True,
        "chunk_size": 16,
        "num_left_chunks": 128,
    },
    MODEL_PARAFORMER_BILINGUAL_STREAMING: {
        "model_type": "paraformer",
        "remote_base_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/",
        "model_dir_name": "sherpa-onnx-streaming-paraformer-bilingual-zh-en",
        "files": {
            "tokens": "tokens.txt",
            "paraformer_encoder": "encoder.int8.onnx",
            "paraformer_decoder": "decoder.int8.onnx",
        },
        "dims": SherpaOnnxModelDims(sample_rate=16000, feature_dim=80),
        "languages": ["zh", "en"],
        "description": "Bilingual Chinese+English streaming Paraformer model.",
        "is_streaming": True,
    },
}

parser.add_argument(
    "-m",
    "--model_type",
    default=MODEL_ZIPFORMER_JA_REAZONSPEECH,
    choices=MODEL_CONFIGS.keys(),
    help="sherpa-onnx model type",
)
parser.add_argument(
    "-V",
    action="store_true",
    help="use microphone input",
)
parser.add_argument(
    "--num-threads",
    type=int,
    default=1,
    help="Number of threads for neural network computation",
)
parser.add_argument(
    "--decoding-method",
    type=str,
    default="greedy_search",
    choices=("greedy_search", "modified_beam_search"),
    help="Valid values are greedy_search and modified_beam_search",
)
parser.add_argument(
    "--provider",
    type=str,
    default="cpu",
    choices=("cpu", "cuda", "coreml"),
    help="Valid values: cpu, cuda, coreml",
)
parser.add_argument(
    "--max-active-paths",
    type=int,
    default=4,
    help="Used only when --decoding-method is modified_beam_search. It specifies number of active paths to keep during decoding.",
)
parser.add_argument(
    "--hotwords-file",
    type=str,
    default="",
    help="The file containing hotwords, one words/phrases per line, like HELLO WORLD 你好世界",
)
parser.add_argument(
    "--hotwords-score",
    type=float,
    default=1.5,
    help="The hotword score of each token for biasing word/phrase. Used only if --hotwords-file is given.",
)
parser.add_argument(
    "--modeling-unit",
    type=str,
    default="",
    help="The modeling unit of the model, valid values are cjkchar, bpe, cjkchar+bpe. Used only when hotwords-file is given.",
)
parser.add_argument(
    "--bpe-vocab",
    type=str,
    default="",
    help="The path to the bpe vocabulary. Used only when hotwords-file is given and modeling-unit is bpe or cjkchar+bpe.",
)
parser.add_argument(
    "--blank-penalty",
    type=float,
    default=0.0,
    help="The penalty applied on blank symbol during decoding.",
)

args = update_parser(parser)

from ailia_audio_utils import load_audio

def recognize_from_audio(recognizer):
    model_config = MODEL_CONFIGS[args.model_type]
    is_streaming = model_config["is_streaming"]

    for audio_path in args.input:
        logger.info(audio_path)

        wav, sample_rate = load_audio(audio_path, for_speech_recognition=True)

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            start_time = time.time()

        if is_streaming:
            stream = recognizer.create_stream()
            chunk_size = int(0.1 * sample_rate) # 100ms
            for i in range(0, len(wav), chunk_size):
                chunk = wav[i : i + chunk_size]
                stream.accept_waveform(sample_rate, chunk)
                while recognizer.is_ready(stream):
                    recognizer.decode_streams([stream])
                result = recognizer.get_result(stream)
                if result:
                    logger.info(f"Partial result: {result}")
            
            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_streams([stream])
            result = recognizer.get_result(stream)
            logger.info(f"Final result: {result}")
        else:
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, wav)
            recognizer.decode_streams([stream])
            result = stream.result.text
            logger.info(f"Final result: {result}")

        if args.benchmark:
            end_time = time.time()
            estimation_time = (end_time - start_time) * 1000
            logger.info(f"\ttotal processing time {estimation_time:.3f} ms")

    logger.info("Script finished successfully.")

def recognize_from_microphone(recognizer, mic_info):
    p = mic_info["p"]
    que = mic_info["que"]
    pause = mic_info["pause"]
    fin = mic_info["fin"]

    model_config = MODEL_CONFIGS[args.model_type]
    sample_rate = model_config["dims"].sample_rate

    stream = recognizer.create_stream()

    try:
        logger.info("Please speak something")
        while p.is_alive():
            try:
                wav = que.get(timeout=0.1)
                stream.accept_waveform(sample_rate, wav)
                while recognizer.is_ready(stream):
                    recognizer.decode_streams([stream])
                result = recognizer.get_result(stream)
                if result:
                    logger.info(f"Partial result: {result}")
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        pass
    finally:
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_streams([stream])
        result = recognizer.get_result(stream)
        logger.info(f"Final result: {result}")
        fin.set()

    logger.info("script finished successfully.")

def main():
    model_config = MODEL_CONFIGS[args.model_type]
    model_dir_name = model_config["model_dir_name"]
    remote_path = model_config["remote_base_url"]

    # モデルのダウンロード
    for file_name in model_config["files"].values():
        check_and_download_file(os.path.join(model_dir_name, file_name), remote_path)

    mic_info = None
    if args.V:
        # in microphone input mode, start thread before load the model.
        mic_info = start_microphone_input(model_config["dims"].sample_rate, sc=False, speaker=False)

    pf = platform.system()
    if pf == "Darwin":
        logger.info(
            "This model not optimized for macOS GPU currently."
            " So we will use BLAS (env_id = 1)."
        )
        args.env_id = 1
    else:
        logger.info(
            "This model uses a lot of memory."
            " If an error occurs during execution, specify -e 0 and execute on the CPU."
        )

    # initialize
    import ailia

    model_config = MODEL_CONFIGS[args.model_type]
    model_dir_name = model_config["model_dir_name"]
    model_type = model_config["model_type"]
    files = model_config["files"]

    # ailia.Netの初期化
    enc_net = ailia.Net(
        os.path.join(model_dir_name, files["encoder"] + ".prototxt"), # .prototxtが必要
        os.path.join(model_dir_name, files["encoder"]),
        env_id=args.env_id,
    )
    dec_net = ailia.Net(
        os.path.join(model_dir_name, files["decoder"] + ".prototxt"),
        os.path.join(model_dir_name, files["decoder"]),
        env_id=args.env_id,
    )
    joi_net = ailia.Net(
        os.path.join(model_dir_name, files["joiner"] + ".prototxt"),
        os.path.join(model_dir_name, files["joiner"]),
        env_id=args.env_id,
    )

    if args.V:
        # microphone input mode
        recognize_from_microphone(enc_net, dec_net, joi_net, mic_info)
    else:
        recognize_from_audio(enc_net, dec_net, joi_net)

    if args.profile:
        if args.onnx:
            prof_file = dec_net.end_profiling()
            print(prof_file)
        else:
            print(dec_net.get_summary())


if __name__ == "__main__":
    main()

