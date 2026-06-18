import sys
import time
from collections import namedtuple
import platform
import queue
import zlib
from logging import getLogger

import numpy as np

# import original modules
sys.path.append("../../util")
from decode_utils import (
    ApplyTimestampRules,
    BeamSearchDecoder,
    GreedyDecoder,
    MaximumLikelihoodRanker,
    SuppressBlank,
    SuppressTokens,
)
from math_utils import softmax
from microphone_utils import start_microphone_input  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa
from languages import LANGUAGES, TO_LANGUAGE_CODE
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WAV_PATH = "demo.wav"
SAVE_TEXT_PATH = "output.txt"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Whisper", WAV_PATH, SAVE_TEXT_PATH, input_ftype="audio")
parser.add_argument(
    "-V",
    action="store_true",
    help="use microphone input",
)
parser.add_argument(
    "-m",
    "--model_type",
    default="small",
    choices=(
        "tiny",
        "base",
        "small",
        "medium",
        "large",
        "large-v3",
        "turbo",
    ),
    help="model type",
)
parser.add_argument(
    "--language",
    type=str,
    default=None,
    choices=sorted(LANGUAGES.keys())
    + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
    help="language spoken in the audio, specify None to perform language detection",
)
parser.add_argument(
    "--temperature", type=float, default=0, help="temperature to use for sampling"
)
parser.add_argument(
    "--best_of",
    type=float,
    default=5,
    help="number of candidates when sampling with non-zero temperature",
)
parser.add_argument(
    "--beam_size",
    type=int,
    default=None,  # modified for ailia models, official whisper specifies 5
    help="number of beams in beam search, only applicable when temperature is zero, None means use greedy search",
)
parser.add_argument(
    "--patience",
    type=float,
    default=None,
    help="optional patience value to use in beam decoding,"
    " as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search",
)
parser.add_argument(
    "--length_penalty",
    type=float,
    default=None,
    help="optional token length penalty coefficient (alpha)"
    " as in https://arxiv.org/abs/1609.08144, uses simple lengt normalization by default",
)
parser.add_argument(
    "--suppress_tokens",
    type=str,
    default="-1",
    help="comma-separated list of token ids to suppress during sampling;"
    " '-1' will suppress most special characters except common punctuations",
)
parser.add_argument(
    "--temperature_increment_on_fallback",
    type=float,
    default=0.2,
    help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below",
)
parser.add_argument(
    "--compression_ratio_threshold",
    type=float,
    default=2.4,
    help="if the gzip compression ratio is higher than this value, treat the decoding as failed",
)
parser.add_argument(
    "--logprob_threshold",
    type=float,
    default=-1.0,
    help="if the average log probability is lower than this value, treat the decoding as failed",
)
parser.add_argument(
    "--no_speech_threshold",
    type=float,
    default=0.6,
    help="if the probability of the <|nospeech|> token is higher than this value"
    " AND the decoding has failed due to `logprob_threshold`, consider the segment as silence",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
parser.add_argument(
    "--dynamic_kv_cache", action="store_true", help="execute dynamic kv_cache version."
)
parser.add_argument("--debug", action="store_true", help="display progress.")
parser.add_argument("--profile", action="store_true", help="display profile.")
parser.add_argument("--ailia_audio", action="store_true", help="use ailia audio.")
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument(
    "--normal", action="store_true", help="use normal model (default : opt model)."
)
parser.add_argument(
    "--task",
    default="transcribe",
    choices=("transcribe", "translate"),
    help="task type",
)
parser.add_argument("--memory_mode", default=-1, type=int, help="memory mode")
parser.add_argument("--prompt", default=None, help="prompt for word vocabulary")
parser.add_argument(
    "--intermediate", action="store_true", help="display intermediate state."
)
parser.add_argument(
    "--fp16", action="store_true", help="use fp16 model (default : fp32 model)."
)
parser.add_argument(
    "--quantize", type=str, default=None, choices=["int4", "int8"],
    help="use int4 or int8 quantized model (turbo only).",
)
args = update_parser(parser)

if args.ailia_audio:
    from ailia_audio_utils import (
        CHUNK_LENGTH,
        HOP_LENGTH,
        N_FRAMES,
        N_SAMPLES,
        SAMPLE_RATE,
        load_audio,
        log_mel_spectrogram,
        pad_or_trim,
    )
else:
    from audio_utils import (
        CHUNK_LENGTH,
        HOP_LENGTH,
        N_FRAMES,
        N_SAMPLES,
        SAMPLE_RATE,
        load_audio,
        log_mel_spectrogram,
        pad_or_trim,
    )

if not args.disable_ailia_tokenizer:
    from ailia_simple_tokenizer import get_tokenizer
else:
    from tokenizer import get_tokenizer

ModelDimensions = namedtuple(
    "ModelDimensions",
    [
        "n_mels",
        "n_audio_ctx",
        "n_audio_state",
        "n_audio_head",
        "n_audio_layer",
        "n_vocab",
        "n_text_ctx",
        "n_text_state",
        "n_text_head",
        "n_text_layer",
    ],
)

dims_dict = {
    "tiny": ModelDimensions(80, 1500, 384, 6, 4, 51865, 448, 384, 6, 4),
    "base": ModelDimensions(80, 1500, 512, 8, 6, 51865, 448, 512, 8, 6),
    "small": ModelDimensions(80, 1500, 768, 12, 12, 51865, 448, 768, 12, 12),
    "medium": ModelDimensions(80, 1500, 1024, 16, 24, 51865, 448, 1024, 16, 24),
    "large": ModelDimensions(80, 1500, 1280, 20, 32, 51865, 448, 1280, 20, 32),
    "large-v3": ModelDimensions(128, 1500, 1280, 20, 32, 51866, 448, 1280, 20, 32),
    "turbo": ModelDimensions(128, 1500, 1280, 20, 32, 51866, 448, 1280, 20, 4),
}
dims = dims_dict[args.model_type]

# ======================
# Workaround
# ======================

if not args.onnx:
    import ailia

    # ailia SDK 1.2.13のAILIA UNSETTLED SHAPEの抑制、1.2.14では不要
    version = ailia.get_version().split(".")
    AILIA_VERSION_MAJOR = int(version[0])
    AILIA_VERSION_MINOR = int(version[1])
    AILIA_VERSION_REVISION = int(version[2])
    REQUIRE_CONSTANT_SHAPE_BETWEEN_INFERENCE = (
        AILIA_VERSION_MAJOR <= 1
        and AILIA_VERSION_MINOR <= 2
        and AILIA_VERSION_REVISION < 14
    )
    COPY_BLOB_DATA_ENABLE = not (
        AILIA_VERSION_MAJOR <= 1
        and AILIA_VERSION_MINOR <= 2
        and AILIA_VERSION_REVISION < 15
    )
    LAYER_NORM_ENABLE = not (
        AILIA_VERSION_MAJOR <= 1
        and AILIA_VERSION_MINOR <= 2
        and AILIA_VERSION_REVISION < 16
    )
    SAVE_ENC_SHAPE = ()
    SAVE_DEC_SHAPE = ()

    if args.memory_mode == -1:
        args.memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
    if (args.memory_mode & 16) != 0:
        ailia.set_temporary_cache_path("./")
else:
    LAYER_NORM_ENABLE = False
    if args.fp16:
        LAYER_NORM_ENABLE = True

# ======================
# Models
# ======================

# opt : mean variance normalization (opset 11)
# opt2 : fuse scatterND (opset 11)
# opt3 : layer normalization (opset 17)

OPT = ".opt"
OPT2 = ".opt2"
OPT3 = ".opt"
if LAYER_NORM_ENABLE:
    OPT = ".opt3"
    OPT2 = ".opt3"
    OPT3 = ".opt"
if args.normal:
    OPT = ""
    OPT2 = ""
    OPT3 = ""

FP16 = ""
if args.fp16:
    FP16 = "_fp16"

if not args.dynamic_kv_cache:
    # 高速化のためKV_CACHEのサイズを最大サイズで固定化したバージョン
    WEIGHT_DEC_TINY_PATH = "decoder_tiny_fix_kv_cache" +  FP16 + OPT2 + ".onnx"
    MODEL_DEC_TINY_PATH = "decoder_tiny_fix_kv_cache" +  FP16 + OPT2 + ".onnx.prototxt"
    WEIGHT_DEC_BASE_PATH = "decoder_base_fix_kv_cache" + FP16 + OPT2 + ".onnx"
    MODEL_DEC_BASE_PATH = "decoder_base_fix_kv_cache" + FP16 + OPT2 + ".onnx.prototxt"
    WEIGHT_DEC_SMALL_PATH = "decoder_small_fix_kv_cache" + FP16 + OPT2 + ".onnx"
    MODEL_DEC_SMALL_PATH = "decoder_small_fix_kv_cache" + FP16 + OPT2 + ".onnx.prototxt"
    WEIGHT_DEC_MEDIUM_PATH = "decoder_medium_fix_kv_cache" + FP16 + OPT2 + ".onnx"
    MODEL_DEC_MEDIUM_PATH = "decoder_medium_fix_kv_cache" + FP16 + OPT2 + ".onnx.prototxt"
    WEIGHT_DEC_LARGE_PATH = "decoder_large_fix_kv_cache.onnx"
    MODEL_DEC_LARGE_PATH = "decoder_large_fix_kv_cache.onnx.prototxt"
    WEIGHT_DEC_LARGE_V3_PATH = "decoder_large_v3_fix_kv_cache.onnx"
    MODEL_DEC_LARGE_V3_PATH = "decoder_large_v3_fix_kv_cache.onnx.prototxt"
    WEIGHT_DEC_TURBO_PATH = "decoder_turbo_fix_kv_cache" + FP16 + OPT3 + ".onnx"
    MODEL_DEC_TURBO_PATH = "decoder_turbo_fix_kv_cache" + FP16 + OPT3 + ".onnx.prototxt"
else:
    # KV_CACHEが推論ごとに変化するバージョン
    WEIGHT_DEC_TINY_PATH = "decoder_tiny.onnx"
    MODEL_DEC_TINY_PATH = "decoder_tiny.onnx.prototxt"
    WEIGHT_DEC_BASE_PATH = "decoder_base.onnx"
    MODEL_DEC_BASE_PATH = "decoder_base.onnx.prototxt"
    WEIGHT_DEC_SMALL_PATH = "decoder_small.onnx"
    MODEL_DEC_SMALL_PATH = "decoder_small.onnx.prototxt"
    WEIGHT_DEC_MEDIUM_PATH = "decoder_medium.onnx"
    MODEL_DEC_MEDIUM_PATH = "decoder_medium.onnx.prototxt"
    WEIGHT_DEC_LARGE_PATH = "decoder_large.onnx"
    MODEL_DEC_LARGE_PATH = "decoder_large.onnx.prototxt"
    WEIGHT_DEC_LARGE_V3_PATH = "decoder_large_v3.onnx"
    MODEL_DEC_LARGE_V3_PATH = "decoder_large_v3.onnx.prototxt"
    WEIGHT_DEC_TURBO_PATH = "decoder_turbo.onnx"
    MODEL_DEC_TURBO_PATH = "decoder_turbo.onnx.prototxt"

WEIGHT_ENC_TINY_PATH = "encoder_tiny" + FP16 + OPT + ".onnx"
MODEL_ENC_TINY_PATH = "encoder_tiny" + FP16 + OPT + ".onnx.prototxt"
WEIGHT_ENC_BASE_PATH = "encoder_base" + FP16 + OPT + ".onnx"
MODEL_ENC_BASE_PATH = "encoder_base" + FP16 + OPT + ".onnx.prototxt"
WEIGHT_ENC_SMALL_PATH = "encoder_small" + FP16 + OPT + ".onnx"
MODEL_ENC_SMALL_PATH = "encoder_small" + FP16 + OPT + ".onnx.prototxt"
WEIGHT_ENC_MEDIUM_PATH = "encoder_medium" + FP16 + OPT + ".onnx"
MODEL_ENC_MEDIUM_PATH = "encoder_medium" + FP16 + OPT + ".onnx.prototxt"
WEIGHT_ENC_LARGE_PATH = "encoder_large.onnx"
MODEL_ENC_LARGE_PATH = "encoder_large.onnx.prototxt"
WEIGHT_ENC_LARGE_V3_PATH = "encoder_large_v3.onnx"
MODEL_ENC_LARGE_V3_PATH = "encoder_large_v3.onnx.prototxt"
WEIGHT_ENC_TURBO_PATH = "encoder_turbo" + FP16 + OPT3 + ".onnx"
MODEL_ENC_TURBO_PATH = "encoder_turbo"  + FP16 + OPT3 + ".onnx.prototxt"

WEIGTH_ENC_LARGE_PB_PATH = "encoder_large_weights.pb"
WEIGHT_DEC_LARGE_PB_PATH = "decoder_large_weights.pb"
WEIGHT_DEC_LARGE_FIX_KV_CACHE_PB_PATH = "decoder_large_fix_kv_cache_weights.pb"
WEIGTH_ENC_LARGE_V3_PB_PATH = "encoder_large_v3_weights.pb"
WEIGHT_DEC_LARGE_V3_PB_PATH = "decoder_large_v3_weights.pb"
WEIGHT_DEC_LARGE_V3_FIX_KV_CACHE_PB_PATH = "decoder_large_v3_fix_kv_cache_weights.pb"
WEIGHT_ENC_TURBO_PB_PATH = "encoder_turbo_weights" + OPT3 + ".pb"

# Int4/Int8 quantized models (turbo only)
if args.quantize is not None:
    Q_SUFFIX = "_" + args.quantize
    WEIGHT_ENC_TURBO_PATH = "encoder_turbo" + Q_SUFFIX + ".onnx"
    MODEL_ENC_TURBO_PATH = "encoder_turbo" + Q_SUFFIX + ".onnx.prototxt"
    WEIGHT_DEC_TURBO_PATH = "decoder_turbo_fix_kv_cache" + Q_SUFFIX + ".onnx"
    MODEL_DEC_TURBO_PATH = "decoder_turbo_fix_kv_cache" + Q_SUFFIX + ".onnx.prototxt"
    WEIGHT_ENC_TURBO_PB_PATH = None

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/whisper/"


# ======================
# Secondaty Functions
# ======================


def is_multilingual():
    return dims.n_vocab >= 51865


def num_languages():
    return dims.n_vocab - 51765 - int(is_multilingual())


def format_timestamp(seconds: float, always_include_hours: bool = False):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours}:" if always_include_hours or hours > 0 else ""

    return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def get_initial_tokens(tokenizer, options):
    sot_sequence = tokenizer.sot_sequence
    sample_len = options.get("sample_len") or dims.n_text_ctx // 2
    n_ctx = dims.n_text_ctx

    tokens = list(sot_sequence)
    prefix = options.get("prefix", None)
    prompt = options.get("prompt", [])

    if prefix:
        prefix_tokens = (
            tokenizer.encode(" " + prefix.strip())
            if isinstance(prefix, str)
            else prefix
        )
        if sample_len is not None:
            max_prefix_len = n_ctx // 2 - sample_len
            prefix_tokens = prefix_tokens[-max_prefix_len:]
        tokens = tokens + prefix_tokens

    if prompt or args.prompt:
        if args.prompt:
            prompt_arg_tokens = tokenizer.encode(args.prompt)
            prompt_tokens = prompt
            prev_prompt_len = (n_ctx // 2 - 1) - len(prompt_arg_tokens)
            tokens = (
                [tokenizer.sot_prev]
                + prompt_arg_tokens
                + prompt_tokens[-prev_prompt_len:]
                + tokens
            )
        else:
            prompt_tokens = (
                tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = [tokenizer.sot_prev] + prompt_tokens[-(n_ctx // 2 - 1) :] + tokens

    return tuple(tokens)


def get_suppress_tokens(tokenizer, options):
    suppress_tokens = options["suppress_tokens"]

    if isinstance(suppress_tokens, str):
        suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

    if -1 in suppress_tokens:
        suppress_tokens = [t for t in suppress_tokens if t >= 0]
        suppress_tokens.extend(tokenizer.non_speech_tokens)
    elif suppress_tokens is None or len(suppress_tokens) == 0:
        suppress_tokens = []  # interpret empty string as an empty list
    else:
        assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

    suppress_tokens.extend([tokenizer.sot, tokenizer.sot_prev, tokenizer.sot_lm])
    if tokenizer.no_speech is not None:
        # no-speech probability is collected separately
        suppress_tokens.append(tokenizer.no_speech)

    return tuple(sorted(set(suppress_tokens)))


def new_kv_cache(n_group, length=451):
    model_type = args.model_type
    if model_type == "tiny.en" or model_type == "tiny":
        size = [8, n_group, length, 384]
    elif model_type == "base.en" or model_type == "base":
        size = [12, n_group, length, 512]
    elif model_type == "small.en" or model_type == "small":
        size = [24, n_group, length, 768]
    elif model_type == "medium.en" or model_type == "medium":
        size = [48, n_group, length, 1024]
    elif model_type in ("large", "large-v3"):
        size = [64, n_group, length, 1280]
    elif model_type == "turbo":
        size = [8, n_group, length, 1280]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return np.zeros(size, dtype=np.float32, order="C")


def compression_ratio(text) -> float:
    return len(text) / len(zlib.compress(text.encode("utf-8")))


# ======================
# Main functions
# ======================


def get_audio_features(enc_net, mel):
    if args.benchmark:
        start = int(round(time.time() * 1000))

    mel = mel.astype(np.float32)
    if not args.onnx:
        if REQUIRE_CONSTANT_SHAPE_BETWEEN_INFERENCE:
            global WEIGHT_ENC_PATH, MODEL_ENC_PATH, SAVE_ENC_SHAPE
            shape = mel.shape
            if SAVE_ENC_SHAPE != shape:
                enc_net = ailia.Net(
                    MODEL_ENC_PATH,
                    WEIGHT_ENC_PATH,
                    env_id=args.env_id,
                    memory_mode=args.memory_mode,
                )
            SAVE_ENC_SHAPE = shape
        output = enc_net.predict([mel])
    else:
        output = enc_net.run(None, {"mel": mel})
    audio_features = output[0]

    if args.benchmark:
        end = int(round(time.time() * 1000))
        estimation_time = end - start
        logger.info(f"\tencoder processing time {estimation_time} ms")

    return audio_features


def inference_logits(
    dec_net,
    tokens,
    audio_features,
    kv_cache=None,
    initial_token_length=None,
    constant_audio_feature=False,
):
    n_group = tokens.shape[0]
    initial_token_length = (
        initial_token_length if initial_token_length else tokens.shape[-1]
    )
    is_init_kv_cache = False
    if kv_cache is None:
        if not args.dynamic_kv_cache:
            kv_cache = new_kv_cache(n_group)
        else:
            kv_cache = new_kv_cache(n_group, initial_token_length)
        offset = 0
        length = initial_token_length
        is_init_kv_cache = True
    else:
        offset = kv_cache.shape[2]
        if not args.dynamic_kv_cache:
            length = offset + 1
            _kv_cache = new_kv_cache(n_group)
            _kv_cache[:, :, :offset, :] = kv_cache
        else:
            _kv_cache = new_kv_cache(n_group, offset + 1)
            _kv_cache[:, :, :-1, :] = kv_cache
        kv_cache = _kv_cache

    if tokens.shape[-1] > initial_token_length:
        # only need to use the last token except in the first forward pass
        tokens = tokens[:, -1:]

    tokens = tokens.astype(np.int64)
    offset = np.array(offset, dtype=np.int64)

    if args.benchmark:
        start = int(round(time.time() * 1000))

    if not args.onnx:
        if offset == 0:
            logits = np.zeros(
                (n_group, initial_token_length, dims.n_vocab),
                dtype=np.float32,
                order="C",
            )
        else:
            logits = np.zeros((n_group, 1, dims.n_vocab), dtype=np.float32, order="C")
        output = [logits, kv_cache]  # static allocatin to reduce data copy
        if REQUIRE_CONSTANT_SHAPE_BETWEEN_INFERENCE:
            global WEIGHT_DEC_PATH, MODEL_DEC_PATH, SAVE_DEC_SHAPE

            shape = (tokens.shape, audio_features.shape, kv_cache.shape)
            if SAVE_DEC_SHAPE != shape:
                dec_net = ailia.Net(
                    MODEL_DEC_PATH,
                    WEIGHT_DEC_PATH,
                    env_id=args.env_id,
                    memory_mode=args.memory_mode,
                )
            SAVE_DEC_SHAPE = shape

            dec_net.predict([tokens, audio_features, kv_cache, offset], output=output)
        else:
            if is_init_kv_cache or not COPY_BLOB_DATA_ENABLE or args.dynamic_kv_cache:
                if constant_audio_feature:
                    dec_net.predict(
                        {"tokens": tokens, "kv_cache": kv_cache, "offset": offset},
                        output=output,
                    )
                else:
                    dec_net.predict(
                        [tokens, audio_features, kv_cache, offset], output=output
                    )
            else:
                dec_net.copy_blob_data("kv_cache", "output_kv_cache", None)
                output = [logits]
                if constant_audio_feature:
                    dec_net.predict({"tokens": tokens, "offset": offset}, output=output)
                else:
                    dec_net.predict(
                        {
                            "tokens": tokens,
                            "audio_features": audio_features,
                            "offset": offset,
                        },
                        output=output,
                    )
    else:
        kv_cache = kv_cache.astype(np.float32)
        output = dec_net.run(
            None,
            {
                "tokens": tokens,
                "audio_features": audio_features,
                "kv_cache": kv_cache,
                "offset": offset,
            },
        )
        logits, kv_cache = output

    if args.benchmark:
        end = int(round(time.time() * 1000))
        estimation_time = end - start
        logger.info(f"\tdecoder processing time {estimation_time} ms")

    if not args.dynamic_kv_cache:
        return logits, kv_cache[:, :, :length, :]
    else:
        return logits, kv_cache


def detect_language(enc_net, dec_net, mel, tokenizer=None):
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(is_multilingual(), num_languages=num_languages())
    if (
        tokenizer.language is None
        or tokenizer.language_token not in tokenizer.sot_sequence
    ):
        raise ValueError(
            f"This model doesn't have language tokens so it can't perform lang id"
        )

    single = mel.ndim == 2
    if single:
        mel = np.expand_dims(mel, axis=0)

    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (dims.n_audio_ctx, dims.n_audio_state):
        mel = get_audio_features(enc_net, mel)

    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = np.array([[tokenizer.sot]] * n_audio)  # [n_audio, 1]

    output, _ = inference_logits(dec_net, x, mel)
    logits = output[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = np.ones(logits.shape[-1], dtype=bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_tokens = np.argmax(logits, axis=-1)
    language_token_probs = softmax(logits, axis=-1)
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs, mel


DecodingResult = namedtuple(
    "DecodingResult",
    [
        "audio_features",
        "language",
        "language_probs",
        "tokens",
        "text",
        "avg_logprob",
        "no_speech_prob",
        "temperature",
    ],
)


def decode(enc_net, dec_net, mel, options, audio_features_cache = None):
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    language = options.get("language") or "en"
    tokenizer = get_tokenizer(
        is_multilingual(),
        num_languages=num_languages(),
        language=language,
        task=args.task,
    )

    n_group = options.get("beam_size") or options.get("best_of") or 1
    n_ctx = dims.n_text_ctx
    sample_len = options.get("sample_len") or dims.n_text_ctx // 2

    initial_tokens = get_initial_tokens(tokenizer, options)
    sample_begin = len(initial_tokens)
    sot_index = initial_tokens.index(tokenizer.sot)

    # logit filters: applies various rules to suppress or penalize certain tokens
    logit_filters = []
    if options.get("suppress_blank"):
        logit_filters.append(SuppressBlank(tokenizer, sample_begin))
    if options.get("suppress_tokens"):
        logit_filters.append(SuppressTokens(get_suppress_tokens(tokenizer, options)))
    if not options.get("without_timestamps"):
        precision = CHUNK_LENGTH / dims.n_audio_ctx  # usually 0.02 seconds
        max_initial_timestamp_index = None
        max_initial_timestamp = options.get("max_initial_timestamp")
        if max_initial_timestamp:
            max_initial_timestamp_index = round(max_initial_timestamp / precision)
        logit_filters.append(
            ApplyTimestampRules(tokenizer, sample_begin, max_initial_timestamp_index)
        )

    # sequence ranker: implements how to rank a group of sampled sequences
    sequence_ranker = MaximumLikelihoodRanker(options.get("length_penalty"))

    # decoder: implements how to select the next tokens, given the autoregressive distribution
    if options.get("beam_size") is not None:
        decoder = BeamSearchDecoder(
            options.get("beam_size"), tokenizer.eot, options.get("patience")
        )
    else:
        decoder = GreedyDecoder(options.get("temperature"), tokenizer.eot)

    decoder.reset()
    n_audio = mel.shape[0]

    if audio_features_cache is not None:
        audio_features = audio_features_cache
    else:
        audio_features = get_audio_features(enc_net, mel)
    tokens = np.repeat(np.array([initial_tokens]), n_audio, axis=-1)
    languages = [language] * audio_features.shape[0]

    # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
    audio_features = np.repeat(audio_features, n_group, axis=0)
    tokens = np.repeat(tokens, n_group, axis=0)

    n_batch = tokens.shape[0]
    sum_logprobs = np.zeros(n_batch)
    no_speech_probs = [np.nan] * n_batch
    initial_token_length = len(initial_tokens)
    kv_cache = None

    # sampling loop
    for i in range(sample_len):
        if args.debug:
            start = int(round(time.time() * 1000))
        constant_audio_feature = i >= 2
        logits, kv_cache = inference_logits(
            dec_net,
            tokens,
            audio_features,
            kv_cache,
            initial_token_length,
            constant_audio_feature,
        )
        if args.debug:
            end = int(round(time.time() * 1000))
            estimation_time = end - start
            logger.info(f"step: {i} / {sample_len} {estimation_time} ms")

        if i == 0 and tokenizer.no_speech is not None:  # save no_speech_probs
            probs_at_sot = softmax(logits[:, sot_index], axis=-1)
            no_speech_probs = probs_at_sot[:, tokenizer.no_speech].tolist()

        # now we need to consider the logits at the last token only
        logits = logits[:, -1]

        # apply the logit filters, e.g. for suppressing or applying penalty to
        for logit_filter in logit_filters:
            logit_filter.apply(logits, tokens)

        def rearrange_kv_cache(source_indices):
            kv_cache[...] = kv_cache[:, source_indices]

        # expand the tokens tensor with the selected next tokens
        tokens, completed = decoder.update(
            tokens, logits, sum_logprobs, rearrange_kv_cache
        )

        if completed or tokens.shape[-1] > n_ctx:
            break

        if args.intermediate:
            texts = [tokenizer.decode(t[len(initial_tokens) :]).strip() for t in tokens]
            print(texts[0][-32:] + "\n\u001B[2A")

    # reshape the tensors to have (n_audio, n_group) as the first two dimensions
    audio_features = audio_features[::n_group]
    no_speech_probs = no_speech_probs[::n_group]
    assert audio_features.shape[0] == len(no_speech_probs) == n_audio

    tokens = tokens.reshape(n_audio, n_group, -1)
    sum_logprobs = sum_logprobs.reshape(n_audio, n_group)

    # get the final candidates for each group, and slice between the first sampled token and EOT
    tokens, sum_logprobs = decoder.finalize(tokens, sum_logprobs)
    tokens = [
        [t[sample_begin : np.nonzero(t == tokenizer.eot)[0][0]] for t in s]
        for s in tokens
    ]

    # select the top-ranked sample in each group
    selected = sequence_ranker.rank(tokens, sum_logprobs)
    tokens = [t[i].tolist() for i, t in zip(selected, tokens)]
    texts = [tokenizer.decode(t).strip() for t in tokens]

    sum_logprobs = [lp[i] for i, lp in zip(selected, sum_logprobs)]
    avg_logprobs = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]

    fields = (texts, languages, tokens, audio_features, avg_logprobs, no_speech_probs)
    if len(set(map(len, fields))) != 1:
        raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

    result = [
        DecodingResult(
            audio_features=features,
            language=language,
            language_probs=None,
            tokens=tokens,
            text=text,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            temperature=options.get("temperature"),
        )
        for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
            *fields
        )
    ]

    if single:
        result = result[0]

    return result


def decode_with_fallback(enc_net, dec_net, segment, decode_options, audio_features_cache = None):
    logprob_threshold = decode_options.get("logprob_threshold", -1.0)
    temperature = decode_options.get("temperature", 0)
    no_speech_threshold = decode_options.get("no_speech_threshold", 0.6)
    compression_ratio_threshold = decode_options.get("compression_ratio_threshold", 2.4)

    temperatures = (
        [temperature] if isinstance(temperature, (int, float)) else temperature
    )
    decode_result = None

    for t in temperatures:
        kwargs = {**decode_options}
        if t > 0:
            # disable beam_size and patience when t > 0
            kwargs.pop("beam_size", None)
            kwargs.pop("patience", None)
            print("temperature", t)
        else:
            # disable best_of when t == 0
            kwargs.pop("best_of", None)

        options = {**kwargs, "temperature": t}
        decode_result = decode(enc_net, dec_net, segment, options, audio_features_cache = audio_features_cache)[0]

        needs_fallback = False
        if (
            compression_ratio_threshold is not None
            and compression_ratio(decode_result.text) > compression_ratio_threshold
        ):
            needs_fallback = True  # too repetitive
        if (
            logprob_threshold is not None
            and decode_result.avg_logprob < logprob_threshold
        ):
            needs_fallback = True  # average log probability is too low
        if (
            no_speech_threshold is not None
            and decode_result.no_speech_prob > no_speech_threshold
        ):
            needs_fallback = False  # silence
        if not needs_fallback:
            break

    return [decode_result]


def predict(wav, enc_net, dec_net, immediate=False, microphone=False):
    language = args.language
    temperature = args.temperature
    temperature_increment_on_fallback = args.temperature_increment_on_fallback
    compression_ratio_threshold = args.compression_ratio_threshold
    logprob_threshold = args.logprob_threshold
    no_speech_threshold = args.no_speech_threshold

    if temperature_increment_on_fallback is not None:
        temperature = tuple(
            np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
        )
    else:
        temperature = [temperature]

    decode_options = {
        "task": args.task,
        "language": language,
        "temperature": temperature,
        "best_of": args.best_of,
        "beam_size": args.beam_size,
        "patience": args.patience,
        "length_penalty": args.length_penalty,
        "suppress_tokens": args.suppress_tokens,
        "compression_ratio_threshold": compression_ratio_threshold,
        "logprob_threshold": logprob_threshold,
        "no_speech_threshold": args.no_speech_threshold,
        "suppress_blank": True,
        "prompt": [],
    }

    mel = log_mel_spectrogram(wav, dims.n_mels, padding=N_SAMPLES)
    content_frames = mel.shape[-1] - N_FRAMES

    audio_features_cache = None
    if language is None:
        segment = pad_or_trim(mel, N_FRAMES)
        _, probs, audio_features_cache = detect_language(enc_net, dec_net, segment)
        decode_options["language"] = language = max(probs, key=probs.get)
        logger.info(
            f"Detected language: {LANGUAGES[decode_options['language']].title()}"
        )

    mel = np.expand_dims(mel, axis=0)
    task = decode_options.get("task", args.task)
    tokenizer = get_tokenizer(
        is_multilingual(), num_languages=num_languages(), language=language, task=task
    )

    seek = 0
    input_stride = N_FRAMES // dims.n_audio_ctx  # mel frames per output token: 2
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    def new_segment(*, start: float, end: float, tokens, result: DecodingResult):
        tokens = tokens.tolist()
        text_tokens = [token for token in tokens if token < tokenizer.eot]
        return {
            "seek": seek,
            "start": start,
            "end": end,
            "text": tokenizer.decode(text_tokens),
            "tokens": tokens,
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "no_speech_prob": result.no_speech_prob,
        }

    try:
        import tqdm

        if microphone:
            pbar = None
        else:
            pbar = tqdm.tqdm(
                total=content_frames, unit="frames", disable=immediate is not False
            )
    except ImportError:
        pbar = None

    # show the progress bar when verbose is False (otherwise the transcribed text will be printed)
    while seek < content_frames:
        time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
        mel_segment = mel[:, :, seek : seek + N_FRAMES]
        segment_size = min(N_FRAMES, content_frames - seek)
        segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
        mel_segment = pad_or_trim(mel_segment, N_FRAMES)

        decode_options["prompt"] = all_tokens[prompt_reset_since:]
        result = decode_with_fallback(enc_net, dec_net, mel_segment, decode_options, audio_features_cache = audio_features_cache)
        result = result[0]
        tokens = np.array(result.tokens)

        audio_features_cache = None

        if no_speech_threshold is not None:
            # no voice activity check
            should_skip = result.no_speech_prob > no_speech_threshold
            if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                # don't skip if the logprob is high enough, despite the no_speech_prob
                should_skip = False

            if should_skip:
                seek += segment_size  # fast-forward to the next segment boundary
                continue

        previous_seek = seek
        current_segments = []

        timestamp_tokens = tokens >= tokenizer.timestamp_begin
        single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]

        consecutive = np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0] + 1
        if len(consecutive) > 0:
            # if the output contains two consecutive timestamp tokens
            slices = consecutive.tolist()
            if single_timestamp_ending:
                slices.append(len(tokens))

            last_slice = 0
            for current_slice in slices:
                sliced_tokens = tokens[last_slice:current_slice]
                start_timestamp_pos = (
                    sliced_tokens[0].item() - tokenizer.timestamp_begin
                )
                end_timestamp_pos = sliced_tokens[-1].item() - tokenizer.timestamp_begin
                current_segments.append(
                    new_segment(
                        start=time_offset + start_timestamp_pos * time_precision,
                        end=time_offset + end_timestamp_pos * time_precision,
                        tokens=sliced_tokens,
                        result=result,
                    )
                )
                last_slice = current_slice

            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last timestamp.
                seek += segment_size
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                last_timestamp_pos = (
                    tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                )
                seek += last_timestamp_pos * input_stride
        else:
            duration = segment_duration
            timestamps = tokens[np.ravel(timestamp_tokens.nonzero())]
            if (
                len(timestamps) > 0
                and timestamps[-1].item() != tokenizer.timestamp_begin
            ):
                # no consecutive timestamps but it has a timestamp; use the last one.
                last_timestamp_pos = timestamps[-1].item() - tokenizer.timestamp_begin
                duration = last_timestamp_pos * time_precision

            current_segments.append(
                new_segment(
                    start=time_offset,
                    end=time_offset + duration,
                    tokens=tokens,
                    result=result,
                )
            )
            seek += segment_size

        if immediate:
            for segment in current_segments:
                start, end, text = segment["start"], segment["end"], segment["text"]
                line = f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
                print(line)

        # if a segment is instantaneous or does not contain text, clear it
        for i, segment in enumerate(current_segments):
            if segment["start"] == segment["end"] or segment["text"].strip() == "":
                segment["text"] = ""
                segment["tokens"] = []
                segment["words"] = []

        all_segments.extend(
            [
                {"id": i, **segment}
                for i, segment in enumerate(current_segments, start=len(all_segments))
            ]
        )
        all_tokens.extend(
            [token for segment in current_segments for token in segment["tokens"]]
        )

        if result.temperature > 0.5:
            # do not feed the prompt tokens if a high temperature was used
            prompt_reset_since = len(all_tokens)

        if pbar is not None:
            # update progress bar
            pbar.update(min(content_frames, seek) - previous_seek)

    d = dict(
        text=tokenizer.decode(all_tokens), segments=all_segments, language=language
    )
    return d


def recognize_from_audio(enc_net, dec_net):
    immediate = True

    # input audio loop
    for audio_path in args.input:
        logger.info(audio_path)

        # prepare input data
        wav = load_audio(audio_path)

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            start = int(round(time.time() * 1000))
            output = predict(
                wav, enc_net, dec_net, immediate=immediate, microphone=False
            )
            end = int(round(time.time() * 1000))
            estimation_time = end - start
            logger.info(f"\ttotal processing time {estimation_time} ms")
        else:
            output = predict(wav, enc_net, dec_net, immediate=immediate)

        if not immediate:
            # output result
            for res in output["segments"]:
                logger.info(
                    f"[{format_timestamp(res['start'])} --> {format_timestamp(res['end'])}] {res['text']}"
                )

    logger.info("Script finished successfully.")


def recognize_from_microphone(enc_net, dec_net, mic_info):
    p = mic_info["p"]
    que = mic_info["que"]
    pause = mic_info["pause"]
    fin = mic_info["fin"]

    try:
        cout = True
        while p.is_alive():
            try:
                if cout:
                    logger.info("Please speak something")
                    cout = False
                wav = que.get(timeout=0.1)
                logger.info("captured! len: %s" % (wav.shape[0] / SAMPLE_RATE))

                # pause.set()   # マイク入力を一時停止
            except queue.Empty:
                continue

            # inference
            logger.info("Translating...")
            output = predict(wav, enc_net, dec_net, immediate=False, microphone=True)

            text = "\n".join(res["text"] for res in output["segments"])
            logger.info(f"predict sentence:\n{text}\n")
            cout = True
            pause.clear()
    except KeyboardInterrupt:
        pass
    finally:
        fin.set()

    logger.info("script finished successfully.")


def main():
    global WEIGHT_DEC_PATH, MODEL_DEC_PATH, WEIGHT_ENC_PATH, MODEL_ENC_PATH
    model_dic = {
        "tiny": {
            "enc": (WEIGHT_ENC_TINY_PATH, MODEL_ENC_TINY_PATH),
            "dec": (WEIGHT_DEC_TINY_PATH, MODEL_DEC_TINY_PATH),
        },
        "base": {
            "enc": (WEIGHT_ENC_BASE_PATH, MODEL_ENC_BASE_PATH),
            "dec": (WEIGHT_DEC_BASE_PATH, MODEL_DEC_BASE_PATH),
        },
        "small": {
            "enc": (WEIGHT_ENC_SMALL_PATH, MODEL_ENC_SMALL_PATH),
            "dec": (WEIGHT_DEC_SMALL_PATH, MODEL_DEC_SMALL_PATH),
        },
        "medium": {
            "enc": (WEIGHT_ENC_MEDIUM_PATH, MODEL_ENC_MEDIUM_PATH),
            "dec": (WEIGHT_DEC_MEDIUM_PATH, MODEL_DEC_MEDIUM_PATH),
        },
        "large": {
            "enc": (WEIGHT_ENC_LARGE_PATH, MODEL_ENC_LARGE_PATH),
            "dec": (WEIGHT_DEC_LARGE_PATH, MODEL_DEC_LARGE_PATH),
        },
        "large-v3": {
            "enc": (WEIGHT_ENC_LARGE_V3_PATH, MODEL_ENC_LARGE_V3_PATH),
            "dec": (WEIGHT_DEC_LARGE_V3_PATH, MODEL_DEC_LARGE_V3_PATH),
        },
        "turbo": {
            "enc": (WEIGHT_ENC_TURBO_PATH, MODEL_ENC_TURBO_PATH),
            "dec": (WEIGHT_DEC_TURBO_PATH, MODEL_DEC_TURBO_PATH),
        },
    }
    model_info = model_dic[args.model_type]

    WEIGHT_ENC_PATH, MODEL_ENC_PATH = model_info["enc"]
    WEIGHT_DEC_PATH, MODEL_DEC_PATH = model_info["dec"]
    check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DEC_PATH, MODEL_DEC_PATH, REMOTE_PATH)
    if args.model_type == "large":
        check_and_download_file(WEIGTH_ENC_LARGE_PB_PATH, REMOTE_PATH)
        if args.dynamic_kv_cache:
            check_and_download_file(WEIGHT_DEC_LARGE_PB_PATH, REMOTE_PATH)
        else:
            check_and_download_file(WEIGHT_DEC_LARGE_FIX_KV_CACHE_PB_PATH, REMOTE_PATH)
    elif args.model_type == "large-v3":
        check_and_download_file(WEIGTH_ENC_LARGE_V3_PB_PATH, REMOTE_PATH)
        if args.dynamic_kv_cache:
            check_and_download_file(WEIGHT_DEC_LARGE_V3_PB_PATH, REMOTE_PATH)
        else:
            check_and_download_file(
                WEIGHT_DEC_LARGE_V3_FIX_KV_CACHE_PB_PATH, REMOTE_PATH
            )
    elif args.model_type == "turbo":
        if args.fp16 == False and WEIGHT_ENC_TURBO_PB_PATH is not None:
            check_and_download_file(WEIGHT_ENC_TURBO_PB_PATH, REMOTE_PATH)

    mic_info = None
    if args.V:
        # in microphone input mode, start thread before load the model.
        mic_info = start_microphone_input(SAMPLE_RATE, sc=False, speaker=False)

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
    if not args.onnx:
        enc_net = ailia.Net(
            MODEL_ENC_PATH,
            WEIGHT_ENC_PATH,
            env_id=args.env_id,
            memory_mode=args.memory_mode,
        )
        dec_net = ailia.Net(
            MODEL_DEC_PATH,
            WEIGHT_DEC_PATH,
            env_id=args.env_id,
            memory_mode=args.memory_mode,
        )
        if args.profile:
            dec_net.set_profile_mode(True)
    else:
        import onnxruntime

        providers = ["CPUExecutionProvider"]
        # providers = ["CUDAExecutionProvider"]
        sess_options = None
        if args.quantize is not None:
            sess_options = onnxruntime.SessionOptions()
            sess_options.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
        enc_net = onnxruntime.InferenceSession(WEIGHT_ENC_PATH, sess_options=sess_options, providers=providers)
        if args.profile:
            options = sess_options or onnxruntime.SessionOptions()
            options.enable_profiling = True
            dec_net = onnxruntime.InferenceSession(
                WEIGHT_DEC_PATH, options, providers=providers
            )
        else:
            dec_net = onnxruntime.InferenceSession(WEIGHT_DEC_PATH, sess_options=sess_options, providers=providers)

    if args.V:
        # microphone input mode
        recognize_from_microphone(enc_net, dec_net, mic_info)
    else:
        recognize_from_audio(enc_net, dec_net)

    if args.profile:
        if args.onnx:
            prof_file = dec_net.end_profiling()
            print(prof_file)
        else:
            print(dec_net.get_summary())


if __name__ == "__main__":
    main()
