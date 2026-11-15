import os
import sys
import time
from logging import getLogger  # noqa: E402

import ailia
import librosa
import numpy as np
import soundfile as sf

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_SPK_ENCODER_PATH = "ps4_spk_encoder.onnx"
MODEL_SPK_ENCODER_PATH = "ps4_spk_encoder.onnx.prototxt"
WEIGHT_BSRNN_PATH = "ps4_bsrnn.onnx"
MODEL_BSRNN_PATH = "ps4_bsrnn.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/ps4/"

WAVE_PATH = "mixed.wav"
SAVE_PATH = "output.wav"

# Audio
SAMPLING_RATE = 16000
WIN = 512
STRIDE = 128

# Speaker encoder features
N_MELS = 80
F_MIN = 20
PRE_EMPHASIS_COEF = 0.97

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("PS4", WAVE_PATH, SAVE_PATH, input_ftype="audio")
parser.add_argument(
    "-r",
    "--reference_file",
    default="ref-voice.wav",
    type=str,
    help="path of the enrollment wav file of the target speaker",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


def load_audio(path):
    wav, sr = librosa.load(path, sr=None, mono=True)
    if sr != SAMPLING_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLING_RATE)

    # peak normalization
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak

    return wav.astype(np.float32)


def stft(wav):
    spec = librosa.stft(
        wav,
        n_fft=WIN,
        hop_length=STRIDE,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    spec_RI = np.stack([spec.real, spec.imag], axis=0)

    return spec_RI.astype(np.float32)


def istft(spec_RI, length):
    spec = spec_RI[0] + 1j * spec_RI[1]
    wav = librosa.istft(
        spec,
        n_fft=WIN,
        hop_length=STRIDE,
        window="hann",
        center=True,
        length=length,
    )

    return wav.astype(np.float32)


def pre_emphasis(wav):
    # the first sample is filtered against the reflected one
    prev = np.concatenate([wav[1:2], wav[:-1]])

    return wav - PRE_EMPHASIS_COEF * prev


def speaker_features(wav):
    wav = pre_emphasis(wav)
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=SAMPLING_RATE,
        n_fft=WIN,
        hop_length=STRIDE,
        win_length=WIN,
        window="hamming",
        center=True,
        pad_mode="reflect",
        power=2.0,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=SAMPLING_RATE / 2,
        htk=True,
        norm=None,
    )
    feats = np.log(mel + 1e-8)
    feats = feats - np.mean(feats, axis=-1, keepdims=True)

    return feats.T.astype(np.float32)


def predict_spk_emb(spk_encoder, wav):
    feats = speaker_features(wav)
    feats = np.expand_dims(feats, axis=0)

    if not args.onnx:
        output = spk_encoder.predict([feats])
    else:
        output = spk_encoder.run(None, {"feats": feats})
    spk_emb = output[0]

    return spk_emb


def predict(bsrnn, wav, spk_emb):
    spec = stft(wav)
    spec = np.expand_dims(spec, axis=0)

    if not args.onnx:
        output = bsrnn.predict([spec, spk_emb])
    else:
        output = bsrnn.run(None, {"spec": spec, "spk_emb": spk_emb})
    est_spec = output[0]

    est_wav = istft(est_spec[0], len(wav))

    return est_wav


# ======================
# Main functions
# ======================


def audio_recognition(models):
    reference_file = args.reference_file
    if not reference_file or not os.path.exists(reference_file):
        logger.error("reference_file:%s is NG." % reference_file)
        sys.exit(-1)

    spk_encoder = models["spk_encoder"]
    bsrnn = models["bsrnn"]

    # prepare enrollment wav
    enroll_wav = load_audio(reference_file)
    spk_emb = predict_spk_emb(spk_encoder, enroll_wav)

    for soundf_path in args.input:
        logger.info(soundf_path)

        # prepare mix wav
        mixed_wav = load_audio(soundf_path)

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                est_wav = predict(bsrnn, mixed_wav, spk_emb)
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
            est_wav = predict(bsrnn, mixed_wav, spk_emb)

        savepath = get_savepath(args.savepath, soundf_path, ext=".wav")
        logger.info(f"saved at : {savepath}")
        sf.write(savepath, est_wav, SAMPLING_RATE, subtype="FLOAT")

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    logger.info("Checking speaker encoder model...")
    check_and_download_models(
        WEIGHT_SPK_ENCODER_PATH, MODEL_SPK_ENCODER_PATH, REMOTE_PATH
    )
    logger.info("Checking separation model...")
    check_and_download_models(WEIGHT_BSRNN_PATH, MODEL_BSRNN_PATH, REMOTE_PATH)

    if not args.onnx:
        env_id = args.env_id

        spk_encoder = ailia.Net(
            MODEL_SPK_ENCODER_PATH, WEIGHT_SPK_ENCODER_PATH, env_id=env_id
        )
        bsrnn = ailia.Net(MODEL_BSRNN_PATH, WEIGHT_BSRNN_PATH, env_id=env_id)
    else:
        import onnxruntime

        cuda = 0 < ailia.get_gpu_environment_id()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )
        spk_encoder = onnxruntime.InferenceSession(
            WEIGHT_SPK_ENCODER_PATH, providers=providers
        )
        bsrnn = onnxruntime.InferenceSession(WEIGHT_BSRNN_PATH, providers=providers)

    models = {"spk_encoder": spk_encoder, "bsrnn": bsrnn}

    audio_recognition(models)


if __name__ == "__main__":
    main()
