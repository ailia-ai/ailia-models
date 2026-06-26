from collections import namedtuple
from fractions import Fraction
import sys
import time
from logging import getLogger
from typing import Optional

import numpy as np
import librosa
import tqdm

import ailia

# import original modules
sys.path.append("../../util")
from model_utils import check_and_download_models  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa

flg_ffmpeg = False

if flg_ffmpeg:
    import ffmpeg

flg_llame = True # save to mp3

try:
    import lameenc
except:
    flg_llame = False

if not flg_llame:
    import soundfile as sf

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_HTDEMUCS_FT_DRUMS_PATH = "htdemucs_ft_drums.onnx"
WEIGHT_HTDEMUCS_FT_BASS_PATH = "htdemucs_ft_bass.onnx"
WEIGHT_HTDEMUCS_FT_OTHER_PATH = "htdemucs_ft_other.onnx"
WEIGHT_HTDEMUCS_FT_VACALS_PATH = "htdemucs_ft_vocals.onnx"
MODEL_HTDEMUCS_FT_DRUMS_PATH = "htdemucs_ft_drums.onnx.prototxt"
MODEL_HTDEMUCS_FT_BASS_PATH = "htdemucs_ft_bass.onnx.prototxt"
MODEL_HTDEMUCS_FT_OTHER_PATH = "htdemucs_ft_other.onnx.prototxt"
MODEL_HTDEMUCS_FT_VACALS_PATH = "htdemucs_ft_vocals.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/demucs/"

SAMPLE_RATE = 44100

WAV_PATH = "test.mp3"
SAVE_WAV_PATH = "output.wav"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    "Demucs Music Source Separation", WAV_PATH, SAVE_WAV_PATH, input_ftype="audio", fp16_support=False
)
parser.add_argument(
    "-m",
    "--model_type",
    default="htdemucs_ft",
    choices=("htdemucs_ft",),
    help="model type",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


TensorChunk = namedtuple("TensorChunk", ["tensor", "offset", "length"])


def load_audio(file: str, sr: int = SAMPLE_RATE):
    if flg_ffmpeg:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=2, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )

        audio = np.copy(np.frombuffer(out, np.float32)).reshape(-1, 2).T
    else:
        # prepare input data
        audio, source_sr = librosa.load(file, sr=None, mono=False)
        # Resample the wav if needed
        if source_sr is not None and source_sr != sr:
            audio = librosa.resample(audio, orig_sr=source_sr, target_sr=sr)

        if audio.ndim == 1:
            audio = audio[None, :]
            audio = np.broadcast_to(audio, (2, audio.shape[-1]))

    return audio


def encode_mp3(wav, path, samplerate=44100, bitrate=320, quality=2, verbose=False):
    """Save given audio as mp3. This should work on all OSes."""
    C, T = wav.shape
    wav = (np.clip(wav, -1, 1) * (2**15 - 1)).astype(np.int16)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(samplerate)
    encoder.set_channels(C)
    encoder.set_quality(quality)  # 2-highest, 7-fastest
    if not verbose:
        encoder.silence()
    wav = wav.transpose(1, 0)
    mp3_data = encoder.encode(wav.tobytes())
    mp3_data += encoder.flush()
    with open(path, "wb") as f:
        f.write(mp3_data)


def tensor_chunk(tensor, offset=0, length=None):
    total_length = (
        tensor.shape[-1] if isinstance(tensor, np.ndarray) else chunk_shape(tensor)[-1]
    )

    if length is None:
        length = total_length - offset
    else:
        length = min(total_length - offset, length)

    if isinstance(tensor, TensorChunk):
        offset = offset + tensor.offset
        tensor = tensor.tensor
    else:
        tensor = tensor
        offset = offset

    return TensorChunk(tensor, offset, length)


def chunk_shape(chunk: TensorChunk):
    shape = list(chunk.tensor.shape)
    shape[-1] = chunk.length
    return shape


def chunk_padded(chunk: TensorChunk, target_length):
    length = chunk_shape(chunk)[-1]
    delta = target_length - length
    total_length = chunk.tensor.shape[-1]

    start = chunk.offset - delta // 2
    end = start + target_length

    correct_start = max(0, start)
    correct_end = min(total_length, end)

    pad_left = correct_start - start
    pad_right = end - correct_end

    out = np.pad(
        chunk.tensor[..., correct_start:correct_end],
        ((0, 0), (0, 0), (pad_left, pad_right)),
    )

    return out


# ======================
# Main functions
# ======================


def apply_model(
    models,
    mix,
    shifts: int = 1,
    split: bool = True,
    overlap: float = 0.25,
    transition_power: float = 1.0,
    segment: Optional[float] = None,
):
    samplerate = SAMPLE_RATE

    batch, channels, length = (
        mix.shape if isinstance(mix, np.ndarray) else chunk_shape(mix)
    )
    if shifts:
        max_shift = int(0.5 * samplerate)
        mix = tensor_chunk(mix)
        padded_mix = chunk_padded(mix, length + 2 * max_shift)
        out = 0.0
        for shift_idx in range(shifts):
            offset = np.random.randint(0, max_shift)
            shifted = tensor_chunk(padded_mix, offset, length + max_shift - offset)
            res = apply_model(
                models,
                shifted,
                shifts=0,
                split=split,
                overlap=overlap,
                transition_power=transition_power,
                segment=segment,
            )
            shifted_out = res
            out += shifted_out[..., max_shift - offset :]

        out /= shifts
        return out
    elif split:
        out = np.zeros((batch, len(models["sources"]), channels, length))
        sum_weight = np.zeros(length)
        segment_length = int(samplerate * (segment if segment else models["segment"]))
        stride = int((1 - overlap) * segment_length)
        offsets = range(0, length, stride)
        # We start from a triangle shaped weight, with maximal weight in the middle
        # of the segment. Then we normalize and take to the power `transition_power`.
        # Large values of transition power will lead to sharper transitions.
        weight = np.concatenate(
            [
                np.arange(1, segment_length // 2 + 1),
                np.arange(segment_length - segment_length // 2, 0, -1),
            ]
        )
        # If the overlap < 50%, this will translate to linear transition when
        # transition_power is 1.
        weight = (weight / weight.max()) ** transition_power
        for offset in tqdm.tqdm(offsets):
            chunk = tensor_chunk(mix, offset, segment_length)
            chunk_out = apply_model(
                models,
                chunk,
                shifts=0,
                split=False,
                overlap=overlap,
                transition_power=transition_power,
                segment=segment,
            )
            chunk_length = chunk_out.shape[-1]
            out[..., offset : offset + segment_length] += (
                weight[:chunk_length] * chunk_out
            )
            sum_weight[offset : offset + segment_length] += weight[:chunk_length]
            offset += segment_length

        out /= sum_weight
        return out
    else:
        valid_length = length
        if "valid_length" in models:
            valid_length = models["valid_length"](length)

        mix = tensor_chunk(mix)
        padded_mix = chunk_padded(mix, valid_length)

        # apply the model
        net = models["net"]
        if not args.onnx:
            output = net.predict([padded_mix])
        else:
            output = net.run(None, {"mix": padded_mix})
        output = output[0]

        delta = output.shape[-1] - length
        if delta:
            output = output[..., delta // 2 : -(delta - delta // 2)]

    return output


def predict(models, audio):
    wav = audio

    ref = wav.mean(axis=0)
    wav -= ref.mean()
    wav /= ref.std() + 1e-8
    mix = wav[None, :]

    shifts = 1
    split = True
    overlap = 0.25
    transition_power = 1.0

    estimates = 0.0
    for k, name in enumerate(models["sources"]):
        models["net"] = models["nets"][name]
        out = apply_model(models, mix, shifts, split, overlap, transition_power)
        for _k in range(len(models["sources"])):
            if _k != k:
                out[:, _k, :, :] *= 0
        estimates += out

    estimates *= ref.std() + 1e-8
    estimates += ref.mean()
    estimates = dict(zip(models["sources"], estimates[0]))

    return estimates


def recognize_from_audio(models):
    # input audio loop
    for audio_path in args.input:
        logger.info(audio_path)

        # prepare input data
        audio = load_audio(audio_path, SAMPLE_RATE)

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            start = int(round(time.time() * 1000))
            output = predict(models, audio)
            end = int(round(time.time() * 1000))
            estimation_time = end - start
            logger.info(f"\ttotal processing time {estimation_time} ms")
        else:
            output = predict(models, audio)

        # save result
        if flg_llame:
            _savepath = get_savepath(args.savepath, audio_path, ext=".mp3")
            for name, wav in output.items():
                wav = wav / max(1.01 * np.max(np.abs(wav)), 1)
                savepath = _savepath.replace(".mp3", f"_{name}.mp3")
                logger.info(f"saved at : {savepath}")
                encode_mp3(wav, savepath, samplerate=SAMPLE_RATE, bitrate=320, quality=2)
        else:
            _savepath = get_savepath(args.savepath, audio_path, ext=".wav")
            for name, wav in output.items():
                wav = wav / max(1.01 * np.max(np.abs(wav)), 1)
                wav = wav.T.astype(np.float32)
                savepath = _savepath.replace(".wav", f"_{name}.wav")
                logger.info(f"saved at : {savepath}")
                sf.write(savepath, wav, SAMPLE_RATE)

    logger.info("Script finished successfully.")


def main():
    segment = Fraction(39, 5)
    models_info = dict(
        htdemucs_ft=dict(
            sources=["drums", "bass", "other", "vocals"],
            weight_files=[
                WEIGHT_HTDEMUCS_FT_DRUMS_PATH,
                WEIGHT_HTDEMUCS_FT_BASS_PATH,
                WEIGHT_HTDEMUCS_FT_OTHER_PATH,
                WEIGHT_HTDEMUCS_FT_VACALS_PATH,
            ],
            model_files=[
                MODEL_HTDEMUCS_FT_DRUMS_PATH,
                MODEL_HTDEMUCS_FT_BASS_PATH,
                MODEL_HTDEMUCS_FT_OTHER_PATH,
                MODEL_HTDEMUCS_FT_VACALS_PATH,
            ],
            segment=segment,
            valid_length=lambda length: int(segment * SAMPLE_RATE),
        )
    )
    model_info = models_info[args.model_type]
    for weight_file, model_file in zip(
        model_info["weight_files"], model_info["model_files"]
    ):
        check_and_download_models(weight_file, model_file, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        nets = {
            name: ailia.Net(model_files, weight_files, env_id=env_id)
            for name, weight_files, model_files in zip(
                model_info["sources"],
                model_info["weight_files"],
                model_info["model_files"],
            )
        }
    else:
        import onnxruntime

        providers = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        nets = {
            name: onnxruntime.InferenceSession(weight_files, providers=providers)
            for name, weight_files in zip(
                model_info["sources"],
                model_info["weight_files"],
            )
        }

    models = dict(
        sources=model_info["sources"],
        nets=nets,
        segment=model_info["segment"],
    )
    if "valid_length" in model_info:
        models["valid_length"] = model_info["valid_length"]

    recognize_from_audio(models)


if __name__ == "__main__":
    main()
