import sys
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

WEIGHT_PATH = "nemotronlabs_re_use.onnx"
MODEL_PATH = "nemotronlabs_re_use.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/nemotronlabs-re-use/"

WAVE_PATH = "mic_test2.wav"
SAVE_PATH = "output.wav"

# STFT settings at the base sampling rate
SAMPLING_RATE = 8000
N_FFT = 320
HOP_SIZE = 40
WIN_SIZE = 320

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    "NemotronLabs RE-USE", WAVE_PATH, SAVE_PATH, input_ftype="audio"
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


def make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.T  # (channels, samples)

    return wav, sr


def mag_phase_stft(wav, n_fft, hop_size, win_size):
    spec = librosa.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    mag = np.log1p(np.abs(spec))
    pha = np.angle(spec)

    return mag.astype(np.float32), pha.astype(np.float32)


# ======================
# Main functions
# ======================


def predict(net, wav, sr):
    n_fft = make_even(N_FFT * sr // SAMPLING_RATE)
    hop_size = make_even(HOP_SIZE * sr // SAMPLING_RATE)
    win_size = make_even(WIN_SIZE * sr // SAMPLING_RATE)

    noisy_mag, noisy_pha = mag_phase_stft(wav, n_fft, hop_size, win_size)

    # feedforward
    if not args.onnx:
        output = net.predict([noisy_mag, noisy_pha])
    else:
        output = net.run(None, {"noisy_mag": noisy_mag, "noisy_pha": noisy_pha})
    amp_g, pha_g = output


def audio_recognition(net):
    for soundf_path in args.input:
        logger.info(soundf_path)

        # prepare input data
        noisy_wav, sr = load_audio(soundf_path)

        # inference
        logger.info("Start inference...")
        predict(net, noisy_wav, sr)

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=memory_mode)
    else:
        import onnxruntime

        cuda = 0 < ailia.get_gpu_environment_id()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    audio_recognition(net)


if __name__ == "__main__":
    main()
