import sys
import time
from logging import getLogger

import ailia
import librosa
import numpy as np
import soundfile as sf

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_LARGE_PATH = "emotion2vec_plus_large.onnx"
MODEL_LARGE_PATH = "emotion2vec_plus_large.onnx.prototxt"
WEIGHT_BASE_PATH = "emotion2vec_plus_base.onnx"
MODEL_BASE_PATH = "emotion2vec_plus_base.onnx.prototxt"
WEIGHT_SEED_PATH = "emotion2vec_plus_seed.onnx"
MODEL_SEED_PATH = "emotion2vec_plus_seed.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/emotion2vec/"

# tokens.txt of the emotion2vec+ models (English part)
LABELS = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
)

WAVE_PATH = "test.wav"
SAMPLE_RATE = 16000

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("emotion2vec", WAVE_PATH, None, input_ftype="audio")
parser.add_argument(
    "-m",
    "--model_type",
    metavar="MODEL_TYPE",
    default="large",
    choices=("large", "base", "seed"),
    help="model type: large | base | seed (emotion2vec+ large / base / seed)",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


def read_wave(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # multi channel -> mono
    if sr != SAMPLE_RATE:
        logger.info(f"resampling {sr}Hz -> {SAMPLE_RATE}Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    return np.ascontiguousarray(audio)


# ======================
# Main functions
# ======================


def predict(net, audio):
    """audio: [T] -> scores [9]

    The waveform normalization, the feature extraction and the classification head
    are all inside the onnx. The first output (frame level embedding) is not used.
    """
    wav = audio[None]  # [1, T]
    if not args.onnx:
        output = net.predict([wav])
    else:
        output = net.run(None, {"wav": wav})

    scores = output[1][0]

    return scores


def recognize_from_audio(net):
    for soundf_path in args.input:
        logger.info(soundf_path)

        audio = read_wave(soundf_path)

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                scores = predict(net, audio)
                end = int(round(time.time() * 1000))
                estimation_time = end - start

                # Loggin
                logger.info(f"\tailia processing estimation time {estimation_time} ms")
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(
                f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
            )
        else:
            scores = predict(net, audio)

        order = np.argsort(scores)[::-1]
        logger.info("Emotion: %s" % LABELS[order[0]])
        logger.info("Confidence: %s" % scores[order[0]])
        for i in order:
            logger.info("\t%-10s %.4f" % (LABELS[i], scores[i]))

    logger.info("Script finished successfully.")


def main():
    dic_model = {
        "large": (WEIGHT_LARGE_PATH, MODEL_LARGE_PATH),
        "base": (WEIGHT_BASE_PATH, MODEL_BASE_PATH),
        "seed": (WEIGHT_SEED_PATH, MODEL_SEED_PATH),
    }
    weight_path, model_path = dic_model[args.model_type]

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    if not args.onnx:
        net = ailia.Net(model_path, weight_path, env_id=args.env_id)
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(weight_path)

    recognize_from_audio(net)


if __name__ == "__main__":
    main()
