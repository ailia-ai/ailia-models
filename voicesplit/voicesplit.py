import os
import sys
import time

import ailia
import librosa
import numpy as np
import soundfile as sf
from audio_utils import Audio
from scipy.ndimage import binary_dilation

# import original modules
sys.path.append("../../util")
# logger
from logging import getLogger  # noqa: E402

from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "voicesplit_exp5.onnx"
MODEL_PATH = "voicesplit_exp5.onnx.prototxt"
WEIGHT_EMB_PATH = "ge2e3k_embedder.onnx"
MODEL_EMB_PATH = "ge2e3k_embedder.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/voicesplit/"

WAVE_PATH = "mixed.wav"
SAVE_PATH = "output.wav"

# Audio
SAMPLING_RATE = 16000

# speaker encoder
MEL_WINDOW_LENGTH = 25  # in milliseconds
MEL_WINDOW_STEP = 10  # in milliseconds
MEL_N_CHANNELS = 40
PARTIALS_N_FRAMES = 160
VAD_WINDOW_LENGTH = 30  # in milliseconds
VAD_MOVING_AVERAGE_WIDTH = 8
VAD_MAX_SILENCE_LENGTH = 6
VAD_RELATIVE_DB = 35.0
AUDIO_NORM_TARGET_DBFS = -30

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("VoiceSplit", WAVE_PATH, SAVE_PATH, input_ftype="audio")
parser.add_argument(
    "-r",
    "--reference_file",
    default="ref-voice.wav",
    type=str,
    help="path of reference wav file",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


def read_wave(path):
    # prepare input data
    wav, source_sr = librosa.load(path, sr=None)
    # Resample the wav if needed
    if source_sr is not None and source_sr != SAMPLING_RATE:
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=SAMPLING_RATE)

    return wav


def normalize_volume(wav, target_dBFS, increase_only=False):
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav**2))
    if dBFS_change < 0 and increase_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters.
    """
    samples_per_window = (VAD_WINDOW_LENGTH * SAMPLING_RATE) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[: len(wav) - (len(wav) % samples_per_window)]

    # Flag the windows that are loud enough relative to the loudest one.
    # webrtcvad is used upstream, but an energy threshold gives the same
    # embedding (cosine similarity over 0.998) without the extra dependency.
    frames = wav.reshape(-1, samples_per_window)
    db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-10)
    voice_flags = (db > db.max() - VAD_RELATIVE_DB).astype(float)

    # Smooth the voice detection with a moving average
    width = VAD_MOVING_AVERAGE_WIDTH
    padded = np.concatenate(
        (np.zeros((width - 1) // 2), voice_flags, np.zeros(width // 2))
    )
    ret = np.cumsum(padded, dtype=float)
    ret[width:] = ret[width:] - ret[:-width]
    audio_mask = np.round(ret[width - 1 :] / width).astype(bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(VAD_MAX_SILENCE_LENGTH + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask]


def wav_to_mel_spectrogram(wav):
    """
    Note: this is not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        y=wav,
        sr=SAMPLING_RATE,
        n_fft=int(SAMPLING_RATE * MEL_WINDOW_LENGTH / 1000),
        hop_length=int(SAMPLING_RATE * MEL_WINDOW_STEP / 1000),
        n_mels=MEL_N_CHANNELS,
    )

    return frames.astype(np.float32).T


def compute_partial_slices(
    n_samples,
    partial_utterance_n_frames=PARTIALS_N_FRAMES,
    min_pad_coverage=0.75,
    overlap=0.5,
):
    """
    Computes where to split an utterance waveform and its corresponding mel
    spectrogram to obtain partial utterances of <partial_utterance_n_frames>
    each. The returned ranges may index further than the length of the
    waveform, so the waveform has to be padded with zeros up to
    wav_slices[-1].stop.
    """
    samples_per_frame = int(SAMPLING_RATE * MEL_WINDOW_STEP / 1000)
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    # Evaluate whether extra padding is warranted or not
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (
        last_wav_range.stop - last_wav_range.start
    )
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]

    return wav_slices, mel_slices


def embed_speaker(embedder, wav):
    wav = normalize_volume(wav, AUDIO_NORM_TARGET_DBFS, increase_only=True)
    wav = trim_long_silences(wav)

    # Compute where to split the utterance into partials and pad if necessary
    wav_slices, mel_slices = compute_partial_slices(len(wav))
    max_wave_length = wav_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

    frames = wav_to_mel_spectrogram(wav)
    frames_batch = np.array([frames[s] for s in mel_slices], dtype=np.float32)

    if not args.onnx:
        output = embedder.predict([frames_batch])
    else:
        output = embedder.run(None, {"mel": frames_batch})

    partial_embeds = output[0]

    # Compute the utterance embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)

    return raw_embed / np.linalg.norm(raw_embed, 2)


def predict(net, mag, dvec):
    if not args.onnx:
        output = net.predict([mag, dvec])
    else:
        output = net.run(None, {"mag": mag, "dvec": dvec})

    mask = output[0]

    return mask


# ======================
# Main functions
# ======================


def audio_recognition(models):
    reference_file = args.reference_file
    if not reference_file or not os.path.exists(reference_file):
        logger.error("reference_file:%s is NG." % reference_file)
        sys.exit(-1)

    net = models["net"]
    embedder = models["embedder"]

    audio = Audio()

    # prepare reference wav
    dvec_wav = read_wave(reference_file)
    dvec = embed_speaker(embedder, dvec_wav)
    dvec = np.expand_dims(dvec, axis=0)

    for soundf_path in args.input:
        logger.info(soundf_path)

        # prepare mix wav
        mixed_wav = read_wave(soundf_path)
        mag, phase = audio.wav2spec(mixed_wav)
        mag = np.expand_dims(mag, axis=0)

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                mask = predict(net, mag, dvec)
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
            mask = predict(net, mag, dvec)

        est_mag = mag * mask
        est_wav = audio.spec2wav(est_mag[0], phase)

        savepath = get_savepath(args.savepath, soundf_path, ext=".wav")
        logger.info(f"saved at : {savepath}")
        sf.write(savepath, est_wav, SAMPLING_RATE, "PCM_24")

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    logger.info("Checking VoiceSplit model...")
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    logger.info("Checking embedder model...")
    check_and_download_models(WEIGHT_EMB_PATH, MODEL_EMB_PATH, REMOTE_PATH)

    if not args.onnx:
        env_id = args.env_id

        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
        embedder = ailia.Net(MODEL_EMB_PATH, WEIGHT_EMB_PATH, env_id=env_id)
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(WEIGHT_PATH)
        embedder = onnxruntime.InferenceSession(WEIGHT_EMB_PATH)

    models = {"net": net, "embedder": embedder}

    audio_recognition(models)


if __name__ == "__main__":
    main()
