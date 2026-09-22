import sys
import time
from logging import getLogger

import ailia
import numpy as np
import soundfile as sf
from tqdm import tqdm

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_CLAP_PATH = "open_musiclm_clap_text.onnx"
MODEL_CLAP_PATH = "open_musiclm_clap_text.onnx.prototxt"
WEIGHT_SEMANTIC_PATH = "open_musiclm_semantic.onnx"
MODEL_SEMANTIC_PATH = "open_musiclm_semantic.onnx.prototxt"
WEIGHT_COARSE_PATH = "open_musiclm_coarse.onnx"
MODEL_COARSE_PATH = "open_musiclm_coarse.onnx.prototxt"
WEIGHT_FINE_PATH = "open_musiclm_fine.onnx"
MODEL_FINE_PATH = "open_musiclm_fine.onnx.prototxt"
WEIGHT_ENCODEC_PATH = "open_musiclm_encodec_decoder.onnx"
MODEL_ENCODEC_PATH = "open_musiclm_encodec_decoder.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/open-musiclm/"

SAVE_WAV_PATH = "output.wav"
SAMPLE_RATE = 24000

# musiclm_large_small_context.json
CODEBOOK_SIZE = 1024  # clap / semantic / acoustic all use 1024 entries, eos = 1024
NUM_COARSE_QUANTIZERS = 3
NUM_FINE_QUANTIZERS = 5
SEMANTIC_WINDOW_SECONDS = 10.0
COARSE_WINDOW_SECONDS = 4.0
FINE_WINDOW_SECONDS = 2.0
SEMANTIC_STEPS_PER_SECOND = 50  # MERT k-means tokens
ACOUSTIC_STEPS_PER_SECOND = 75  # EnCodec 24kHz
SEMANTIC_SLIDING_WINDOW_STEP_PERCENT = 0.5
COARSE_SLIDING_WINDOW_STEP_PERCENT = 0.5
FINE_SLIDING_WINDOW_STEP_PERCENT = 1
FILTER_THRES = 0.9
TEMPERATURE = {"semantic": 1.0, "coarse": 0.95, "fine": 0.4}
TEXT_MAX_LENGTH = 77

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("open-musiclm", None, SAVE_WAV_PATH)
parser.add_argument(
    "-i",
    "--input",
    metavar="TEXT",
    type=str,
    default="song with synths and flute",
    help="text prompt describing the music to generate",
)
parser.add_argument(
    "--duration",
    type=float,
    default=4,
    help="duration of the generated audio in seconds",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="random seed for sampling",
)
parser.add_argument(
    "--return_coarse_wave",
    action="store_true",
    help="skip the fine stage and decode audio from coarse tokens only (faster, lower quality)",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Secondaty Functions
# ======================


def top_k(logits, thres=FILTER_THRES):
    """keep the top (1 - thres) fraction of logits, set the others to -inf"""
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    ind = np.argpartition(-logits, k - 1, axis=-1)[:, :k]
    probs = np.full_like(logits, -np.inf)
    np.put_along_axis(probs, ind, np.take_along_axis(logits, ind, axis=-1), axis=-1)

    return probs


def gumbel_sample(rng, logits, temperature=1.0):
    noise = rng.uniform(0, 1, size=logits.shape)
    gumbel_noise = -np.log(-np.log(noise + 1e-20) + 1e-20)

    return np.argmax(logits / temperature + gumbel_noise, axis=-1)


# ======================
# Main functions
# ======================


def predict(net, inputs):
    if not args.onnx:
        output = net.predict(inputs)
    else:
        output = net.run(None, inputs)

    return output


def clap_token_ids(models, text):
    tokenizer = models["tokenizer"]
    tok = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=TEXT_MAX_LENGTH,
        return_tensors="np",
    )
    output = predict(
        models["clap"],
        {
            "input_ids": tok["input_ids"].astype(np.int64),
            "attention_mask": tok["attention_mask"].astype(np.int64),
        },
    )
    token_ids = output[0]  # (1, 12, 1)

    return token_ids


def generate_tokens(models, stage, conditioning, pred=None, max_time_steps=0):
    """Autoregressively sample one token sequence with the transformer of `stage`.

    conditioning: list of token arrays [(1, n, q), ...] (clap, semantic / coarse) -> flattened, eos appended
    pred: (1, t, q) already generated tokens of the predicted sequence (audio continuation), or None
    Returns (1, max_time_steps, q). Every step re-runs the transformer on the whole sequence
    (no kv cache, same as the original implementation) and samples from the last position logits.
    """
    net = models[stage]
    input_names = models["input_names"][stage]
    num_quantizers = {
        "semantic": 1,
        "coarse": NUM_COARSE_QUANTIZERS,
        "fine": NUM_FINE_QUANTIZERS,
    }[stage]
    rng = models["rng"]

    cond = [
        np.concatenate([c.reshape(1, -1), [[CODEBOOK_SIZE]]], axis=1).astype(np.int64)
        for c in conditioning
    ]
    pred = (
        pred.reshape(1, -1).astype(np.int64)
        if pred is not None
        else np.zeros((1, 0), dtype=np.int64)
    )

    init_time_step = pred.shape[1] // num_quantizers
    for _ in tqdm(
        range(init_time_step, max_time_steps), desc=f"generating {stage} tokens"
    ):
        for _ in range(num_quantizers):
            logits = predict(net, dict(zip(input_names, cond + [pred])))[0]  # (1, 1025)
            logits[:, -1] = -np.inf  # never sample eos
            sampled = gumbel_sample(rng, top_k(logits), temperature=TEMPERATURE[stage])
            pred = np.concatenate([pred, sampled[:, None]], axis=1)

    return pred.reshape(1, -1, num_quantizers)


def sliding_windows(tokens, window_size, step_size):
    """(1, n, q) -> list of (1, window_size, q), same as Tensor.unfold on the time axis"""
    n = tokens.shape[1]
    return [
        tokens[:, i : i + window_size] for i in range(0, n - window_size + 1, step_size)
    ]


def generate_wave(models, text, output_seconds, return_coarse_wave=False):
    """MusicLM.forward (text conditioning only): text -> waveform (n,) at 24kHz"""
    clap_ids = clap_token_ids(models, text)

    # semantic stage
    all_semantic = generate_tokens(
        models,
        "semantic",
        [clap_ids],
        max_time_steps=int(
            min(output_seconds, SEMANTIC_WINDOW_SECONDS) * SEMANTIC_STEPS_PER_SECOND
        ),
    )
    while all_semantic.shape[1] < int(output_seconds * SEMANTIC_STEPS_PER_SECOND):
        condition_length = int(
            SEMANTIC_WINDOW_SECONDS
            * SEMANTIC_STEPS_PER_SECOND
            * (1 - SEMANTIC_SLIDING_WINDOW_STEP_PERCENT)
        )
        pred = generate_tokens(
            models,
            "semantic",
            [clap_ids],
            pred=all_semantic[:, -condition_length:],
            max_time_steps=int(SEMANTIC_WINDOW_SECONDS * SEMANTIC_STEPS_PER_SECOND),
        )
        all_semantic = np.concatenate(
            [all_semantic, pred[:, condition_length:]], axis=1
        )

    # coarse stage
    window_size = int(COARSE_WINDOW_SECONDS * SEMANTIC_STEPS_PER_SECOND - 1)
    step_size = int(window_size * COARSE_SLIDING_WINDOW_STEP_PERCENT)
    all_coarse = None
    for semantic in sliding_windows(all_semantic, window_size, step_size):
        if all_coarse is not None:
            condition_length = int(
                COARSE_WINDOW_SECONDS
                * ACOUSTIC_STEPS_PER_SECOND
                * (1 - COARSE_SLIDING_WINDOW_STEP_PERCENT)
            )
            condition_coarse = all_coarse[:, -condition_length:]
        else:
            condition_coarse = None
        pred = generate_tokens(
            models,
            "coarse",
            [clap_ids, semantic],
            pred=condition_coarse,
            max_time_steps=int(COARSE_WINDOW_SECONDS * ACOUSTIC_STEPS_PER_SECOND),
        )
        if all_coarse is None:
            all_coarse = pred
        else:
            all_coarse = np.concatenate(
                [all_coarse, pred[:, condition_length:]], axis=1
            )

    if return_coarse_wave:
        wave = predict(models["encodec"], {"codes": all_coarse})[0]  # (1, 1, n)
        return wave[0, 0]

    # fine stage
    fine_window_size = int(FINE_WINDOW_SECONDS * ACOUSTIC_STEPS_PER_SECOND)
    fine_step_size = int(fine_window_size * FINE_SLIDING_WINDOW_STEP_PERCENT)
    all_fine = None
    for coarse in sliding_windows(all_coarse, fine_window_size, fine_step_size):
        if all_fine is not None:
            condition_length = int(
                fine_window_size * (1 - FINE_SLIDING_WINDOW_STEP_PERCENT)
            )
            condition_fine = (
                all_fine[:, -condition_length:] if condition_length > 0 else None
            )
        else:
            condition_fine = None
        pred = generate_tokens(
            models,
            "fine",
            [clap_ids, coarse],
            pred=condition_fine,
            max_time_steps=fine_window_size,
        )
        if all_fine is None:
            all_fine = pred
        else:
            all_fine = np.concatenate([all_fine, pred[:, condition_length:]], axis=1)

    codes = np.concatenate([all_coarse, all_fine], axis=-1)  # (1, t, 8)
    wave = predict(models["encodec"], {"codes": codes})[0]  # (1, 1, n)

    return wave[0, 0]


def generate_from_text(models):
    text = args.input
    logger.info("prompt: %s" % text)
    if args.duration < COARSE_WINDOW_SECONDS:
        # the coarse stage needs at least one full window of semantic tokens
        logger.error(f"--duration must be at least {COARSE_WINDOW_SECONDS} seconds")
        sys.exit(1)

    # inference
    logger.info("Start inference...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        start = int(round(time.time() * 1000))
        wave = generate_wave(models, text, args.duration, args.return_coarse_wave)
        end = int(round(time.time() * 1000))
        logger.info(f"\tailia processing estimation time {end - start} ms")
    else:
        wave = generate_wave(models, text, args.duration, args.return_coarse_wave)

    savepath = args.savepath
    logger.info(f"saved at : {savepath}")
    sf.write(savepath, wave, SAMPLE_RATE)

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_CLAP_PATH, MODEL_CLAP_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_SEMANTIC_PATH, MODEL_SEMANTIC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_COARSE_PATH, MODEL_COARSE_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_FINE_PATH, MODEL_FINE_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_ENCODEC_PATH, MODEL_ENCODEC_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        # the sequence length changes at every step; without reuse_interstage the intermediate
        # buffers of the transformers keep growing (~8GB per model for 4 s of audio)
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        net_clap = ailia.Net(
            MODEL_CLAP_PATH, WEIGHT_CLAP_PATH, env_id=env_id, memory_mode=memory_mode
        )
        net_semantic = ailia.Net(
            MODEL_SEMANTIC_PATH,
            WEIGHT_SEMANTIC_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        net_coarse = ailia.Net(
            MODEL_COARSE_PATH,
            WEIGHT_COARSE_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        net_fine = ailia.Net(
            MODEL_FINE_PATH, WEIGHT_FINE_PATH, env_id=env_id, memory_mode=memory_mode
        )
        net_encodec = ailia.Net(
            MODEL_ENCODEC_PATH,
            WEIGHT_ENCODEC_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
    else:
        import onnxruntime

        cuda = 0 < ailia.get_gpu_environment_id()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )
        net_clap = onnxruntime.InferenceSession(WEIGHT_CLAP_PATH, providers=providers)
        net_semantic = onnxruntime.InferenceSession(
            WEIGHT_SEMANTIC_PATH, providers=providers
        )
        net_coarse = onnxruntime.InferenceSession(
            WEIGHT_COARSE_PATH, providers=providers
        )
        net_fine = onnxruntime.InferenceSession(WEIGHT_FINE_PATH, providers=providers)
        net_encodec = onnxruntime.InferenceSession(
            WEIGHT_ENCODEC_PATH, providers=providers
        )

    if args.disable_ailia_tokenizer:
        from transformers import RobertaTokenizer

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    else:
        from ailia_tokenizer import RobertaTokenizer

        tokenizer = RobertaTokenizer.from_pretrained("./tokenizer/")

    models = {
        "clap": net_clap,
        "semantic": net_semantic,
        "coarse": net_coarse,
        "fine": net_fine,
        "encodec": net_encodec,
        "tokenizer": tokenizer,
        "input_names": {
            "semantic": ["clap_token_ids", "semantic_token_ids"],
            "coarse": ["clap_token_ids", "semantic_token_ids", "coarse_token_ids"],
            "fine": ["clap_token_ids", "coarse_token_ids", "fine_token_ids"],
        },
        "rng": np.random.default_rng(args.seed),
    }

    generate_from_text(models)


if __name__ == "__main__":
    main()
