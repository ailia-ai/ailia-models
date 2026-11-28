import math
import sys
import time
from logging import getLogger

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
parser.add_argument(
    "--bwe",
    type=int,
    default=None,
    help="resample the input to this sampling rate before enhancement (bandwidth extension)",
)
parser.add_argument(
    "--chunk",
    type=float,
    default=None,
    help="chunk length in seconds. the input is enhanced chunk by chunk with overlap",
)
parser.add_argument(
    "--hop_portion",
    type=float,
    default=0.5,
    help="hop of the chunks as a ratio of the chunk length",
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

    if args.bwe is not None:
        wav = librosa.resample(
            wav, orig_sr=sr, target_sr=args.bwe, res_type="kaiser_best"
        )
        sr = args.bwe

    return wav.astype(np.float32), sr


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


def mag_phase_istft(mag, pha, n_fft, hop_size, win_size):
    mag = np.expm1(np.maximum(mag, 0))
    spec = mag * np.cos(pha) + 1j * mag * np.sin(pha)
    wav = librosa.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window="hann",
        center=True,
    )

    return wav.astype(np.float32)


def pad_or_trim_to_match(reference, target, pad_value=1e-8):
    ref_len = reference.shape[1]
    tgt_len = target.shape[1]

    if tgt_len >= ref_len:
        return target[:, :ref_len]

    padded = np.full((target.shape[0], ref_len), pad_value, dtype=target.dtype)
    padded[:, :tgt_len] = target

    return padded


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

    # silence the frames where more than half of the frequency bins are zero
    mag = np.expm1(np.maximum(amp_g, 0))
    zero_portion = np.sum(mag == 0, axis=1) / mag.shape[1]
    amp_g[:, :, (zero_portion > 0.5)[0]] = 0

    audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop_size, win_size)
    audio_g = pad_or_trim_to_match(wav, audio_g)

    return audio_g


def enhance(net, noisy_wav, sr):
    if args.chunk is None:
        return predict(net, noisy_wav, sr)

    chunk_size = int(args.chunk * sr)
    hop_length = int(args.hop_portion * chunk_size)
    window = librosa.filters.get_window("hann", chunk_size, fftbins=True)
    window = np.maximum(window.astype(np.float32), 1e-7)

    audio_enhanced = np.zeros_like(noisy_wav)
    window_sum = np.zeros_like(noisy_wav)
    num_chunks = max(1, math.ceil((noisy_wav.shape[1] - chunk_size) / hop_length) + 1)
    for c in range(noisy_wav.shape[0]):
        wav = noisy_wav[c : c + 1, :]
        for i in range(num_chunks):
            if i < num_chunks - 1:
                start = i * hop_length
                wav_chunk = wav[:, start : start + chunk_size]
            else:
                wav_chunk = wav[:, -chunk_size:]
            audio_g = predict(net, wav_chunk, sr)
            w = window[0 : audio_g.shape[1]]
            if i < num_chunks - 1:
                audio_enhanced[c : c + 1, start : start + chunk_size] += audio_g * w
                window_sum[c : c + 1, start : start + chunk_size] += w
            else:
                audio_enhanced[c : c + 1, -chunk_size:] += audio_g * w
                window_sum[c : c + 1, -chunk_size:] += w
    audio_enhanced = audio_enhanced / window_sum

    return audio_enhanced


def audio_recognition(net):
    for soundf_path in args.input:
        logger.info(soundf_path)

        # prepare input data
        noisy_wav, sr = load_audio(soundf_path)

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                audio_g = enhance(net, noisy_wav, sr)
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
            audio_g = enhance(net, noisy_wav, sr)

        savepath = get_savepath(args.savepath, soundf_path, ext=".wav")
        logger.info(f"saved at : {savepath}")
        sf.write(savepath, audio_g.T, sr, subtype="FLOAT")

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
