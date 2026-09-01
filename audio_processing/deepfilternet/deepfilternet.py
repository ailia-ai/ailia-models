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

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/deepfilternet/"

WAVE_PATH = "noisy_snr0.wav"
SAVE_PATH = "output.wav"

MODEL_TYPES = ("DeepFilterNet", "DeepFilterNet2", "DeepFilterNet3")

# [df] section of the config.ini that ships with every pretrained model.
# The three models share the same values.
SAMPLE_RATE = 48000
FFT_SIZE = 960
HOP_SIZE = 480
NB_ERB = 32
NB_DF = 96
MIN_NB_ERB_FREQS = 2
NORM_TAU = 1.0

# initial state of the exponential moving average used by the two normalizations
MEAN_NORM_INIT = (-60.0, -90.0)
UNIT_NORM_INIT = (0.001, 0.0001)

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("DeepFilterNet", WAVE_PATH, SAVE_PATH, input_ftype="audio")
parser.add_argument(
    "-m",
    "--model_type",
    metavar="MODEL_TYPE",
    default="DeepFilterNet3",
    choices=MODEL_TYPES,
    help="model type: " + " | ".join(MODEL_TYPES),
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


def read_wave(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(audio.T)  # [T, C] -> [C, T]
    if sr != SAMPLE_RATE:
        logger.info(f"resampling {sr}Hz -> {SAMPLE_RATE}Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    return audio, sr


def freq2erb(freq_hz):
    return 9.265 * np.log1p(freq_hz / (24.7 * 9.265))


def erb2freq(n_erb):
    return 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1.0)


def erb_fb():
    """Number of frequency bins per ERB band. The bands get wider with frequency."""
    freq_width = SAMPLE_RATE / FFT_SIZE
    erb_low = freq2erb(0.0)
    erb_high = freq2erb(SAMPLE_RATE // 2)
    step = (erb_high - erb_low) / NB_ERB

    widths = [0] * NB_ERB
    prev_freq = 0  # last frequency band of the previous erb band
    freq_over = 0  # number of frequency bands already stored in previous erb bands
    for i in range(1, NB_ERB + 1):
        f = erb2freq(erb_low + i * step)
        fb = int(np.round(f / freq_width))
        nb_freqs = fb - prev_freq - freq_over
        if nb_freqs < MIN_NB_ERB_FREQS:
            freq_over = MIN_NB_ERB_FREQS - nb_freqs
            nb_freqs = MIN_NB_ERB_FREQS
        else:
            freq_over = 0
        widths[i - 1] = nb_freqs
        prev_freq = fb

    widths[-1] += 1  # since we have FFT_SIZE/2+1 frequency bins
    too_large = sum(widths) - (FFT_SIZE // 2 + 1)
    if too_large > 0:
        widths[-1] -= too_large

    return np.array(widths, dtype=np.int64)


def fft_window():
    """Vorbis window: sin(pi/2 * sin^2(pi*n/N))"""
    n = np.arange(FFT_SIZE, dtype=np.float64)
    sin = np.sin(0.5 * np.pi * (n + 0.5) / (FFT_SIZE // 2))
    return np.sin(0.5 * np.pi * sin * sin).astype(np.float32)


ERB_WIDTHS = erb_fb()
ERB_EDGES = np.concatenate(([0], np.cumsum(ERB_WIDTHS))).astype(np.int64)
WINDOW = fft_window()
# the normalization of the overlap add is applied in the analysis only
WNORM = np.float32(1.0 / (FFT_SIZE**2 / (2 * HOP_SIZE)))


def get_norm_alpha():
    """Exponential decay factor alpha for a given tau (decay window size [s])."""
    a = math.exp(-(HOP_SIZE / SAMPLE_RATE) / NORM_TAU)
    precision = 3
    alpha = 1.0
    while alpha >= 1.0:
        alpha = round(a, precision)
        precision += 1

    return alpha


ALPHA = get_norm_alpha()


def as_real(x):
    return np.stack((x.real, x.imag), axis=-1).astype(np.float32)


def as_complex(x):
    return (x[..., 0] + 1j * x[..., 1]).astype(np.complex64)


# ======================
# Signal processing
# ======================


def analysis(audio):
    """Real-time STFT. Frame i is windowed over the frames i-1 and i.

    audio: [C, T] -> spec: [C, Tf, F] complex, Tf = T // HOP_SIZE
    """
    channels, samples = audio.shape
    frames = samples // HOP_SIZE

    # the first frame sees an all zero analysis memory
    padded = np.concatenate(
        (
            np.zeros((channels, FFT_SIZE - HOP_SIZE), dtype=np.float32),
            audio[:, : frames * HOP_SIZE],
        ),
        axis=1,
    )
    index = np.arange(frames)[:, None] * HOP_SIZE + np.arange(FFT_SIZE)[None, :]
    buf = padded[:, index] * WINDOW

    return np.fft.rfft(buf, axis=-1).astype(np.complex64) * WNORM


def synthesis(spec):
    """Real-time ISTFT with overlap add.

    spec: [C, Tf, F] complex -> audio: [C, Tf * HOP_SIZE]
    """
    channels, frames = spec.shape[:2]

    # the inverse fft of the rust implementation is not normalized
    x = np.fft.irfft(spec, n=FFT_SIZE, axis=-1) * FFT_SIZE
    x = (x * WINDOW).astype(np.float32)

    # the synthesis memory holds the second half of the previous frame
    audio = x[:, :, :HOP_SIZE].copy()
    audio[:, 1:] += x[:, :-1, HOP_SIZE:]

    return audio.reshape(channels, frames * HOP_SIZE)


def erb(spec, db=True):
    """Mean power per ERB band. spec: [C, Tf, F] -> [C, Tf, NB_ERB]"""
    power = (spec.real**2 + spec.imag**2).astype(np.float32)
    band = np.stack(
        [
            power[..., ERB_EDGES[i] : ERB_EDGES[i + 1]].mean(axis=-1)
            for i in range(NB_ERB)
        ],
        axis=-1,
    )
    if db:
        band = np.log10(band + 1e-10) * 10.0

    return band.astype(np.float32)


def erb_norm(feat, alpha):
    """Exponential mean normalization of the ERB band energies. feat: [C, Tf, NB_ERB]"""
    channels, frames = feat.shape[:2]
    state = np.linspace(*MEAN_NORM_INIT, NB_ERB, dtype=np.float32)
    state = np.repeat(state[None], channels, axis=0)

    out = np.empty_like(feat)
    for i in range(frames):
        state = feat[:, i] * (1 - alpha) + state * alpha
        out[:, i] = (feat[:, i] - state) / 40.0

    return out


def unit_norm(spec, alpha):
    """Exponential unit normalization of the complex spectrum. spec: [C, Tf, NB_DF]"""
    channels, frames = spec.shape[:2]
    state = np.linspace(*UNIT_NORM_INIT, spec.shape[-1], dtype=np.float32)
    state = np.repeat(state[None], channels, axis=0)

    out = np.empty_like(spec)
    for i in range(frames):
        state = np.abs(spec[:, i]) * (1 - alpha) + state * alpha
        out[:, i] = spec[:, i] / np.sqrt(state)

    return out


def df_features(audio):
    """audio: [C, T] -> spec [C, 1, Tf, F, 2], feat_erb [C, 1, Tf, NB_ERB],
    feat_spec [C, 1, Tf, NB_DF, 2]"""
    spec = analysis(audio)
    feat_erb = erb_norm(erb(spec), ALPHA)[:, None]
    feat_spec = as_real(unit_norm(spec[..., :NB_DF], ALPHA))[:, None]

    return as_real(spec)[:, None], feat_erb, feat_spec


# ======================
# Main functions
# ======================


def predict(net, spec, feat_erb, feat_spec):
    if not args.onnx:
        output = net.predict([spec, feat_erb, feat_spec])
    else:
        output = net.run(
            None, {"spec": spec, "feat_erb": feat_erb, "feat_spec": feat_spec}
        )

    enh, m, lsnr = output

    return enh, m, lsnr


def enhance(net, audio):
    orig_len = audio.shape[-1]
    # pad audio to compensate for the delay due to the real-time STFT implementation
    audio = np.pad(audio, ((0, 0), (0, FFT_SIZE)))

    spec, feat_erb, feat_spec = df_features(audio)
    enh, m, lsnr = predict(net, spec, feat_erb, feat_spec)

    enhanced = as_complex(enh.squeeze(1))
    audio = synthesis(enhanced)

    # the STFT/ISTFT loop introduces an algorithmic delay of FFT_SIZE - HOP_SIZE
    d = FFT_SIZE - HOP_SIZE

    return audio[:, d : orig_len + d]


def audio_recognition(net):
    for soundf_path in args.input:
        logger.info(soundf_path)

        audio, sr = read_wave(soundf_path)

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                enhanced = enhance(net, audio)
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
            start = int(round(time.time() * 1000))
            enhanced = enhance(net, audio)
            estimation_time = int(round(time.time() * 1000)) - start
            rtf = estimation_time / (enhanced.shape[-1] / SAMPLE_RATE * 1000)
            logger.info(f"\tprocessing time {estimation_time} ms (RT factor {rtf:.3f})")

        if sr != SAMPLE_RATE:
            enhanced = librosa.resample(enhanced, orig_sr=SAMPLE_RATE, target_sr=sr)

        savepath = get_savepath(args.savepath, soundf_path, ext=".wav")
        logger.info(f"saved at : {savepath}")
        sf.write(savepath, enhanced.T, sr, "PCM_16")

    logger.info("Script finished successfully.")


def main():
    weight_path = "%s.onnx" % args.model_type.lower()
    model_path = weight_path + ".prototxt"

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    if not args.onnx:
        net = ailia.Net(model_path, weight_path, env_id=args.env_id)
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(weight_path)

    audio_recognition(net)


if __name__ == "__main__":
    main()
