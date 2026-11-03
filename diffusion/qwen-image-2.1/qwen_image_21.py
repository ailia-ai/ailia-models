import sys
from logging import getLogger

import ailia
import numpy as np

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa
from model_utils import check_and_download_file, check_and_download_models  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_VISION_ENCODER_PATH = "qwen_image_21_vision_encoder.onnx"
WEIGHT_VISION_ENCODER_PB_PATH = "qwen_image_21_vision_encoder_weights.pb"
MODEL_VISION_ENCODER_PATH = "qwen_image_21_vision_encoder.onnx.prototxt"
WEIGHT_TEXT_ENCODER_PATH = "qwen_image_21_text_encoder.onnx"
WEIGHT_TEXT_ENCODER_PB_PATH = "qwen_image_21_text_encoder_weights.pb"
MODEL_TEXT_ENCODER_PATH = "qwen_image_21_text_encoder.onnx.prototxt"
EMBED_TOKENS_PATH = "qwen_image_21_embed_tokens.npy"
WEIGHT_TRANSFORMER_PATH = "qwen_image_21_transformer.onnx"
WEIGHT_TRANSFORMER_PB_PATH = "qwen_image_21_transformer_weights.pb"
MODEL_TRANSFORMER_PATH = "qwen_image_21_transformer.onnx.prototxt"
WEIGHT_TURBO_TRANSFORMER_PATH = "qwen_image_21_turbo_transformer.onnx"
WEIGHT_TURBO_TRANSFORMER_PB_PATH = "qwen_image_21_turbo_transformer_weights.pb"
MODEL_TURBO_TRANSFORMER_PATH = "qwen_image_21_turbo_transformer.onnx.prototxt"
WEIGHT_VAE_ENCODER_PATH = "qwen_image_21_vae_encoder.onnx"
MODEL_VAE_ENCODER_PATH = "qwen_image_21_vae_encoder.onnx.prototxt"
WEIGHT_VAE_DECODER_PATH = "qwen_image_21_vae_decoder.onnx"
MODEL_VAE_DECODER_PATH = "qwen_image_21_vae_decoder.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/qwen-image-2.1/"

SAVE_IMAGE_PATH = "output.png"

SYS_PROMPT = "Comprehend and analyze the provided prompt."
SYS_TEMPLATE = f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
PROMPT_TEMPLATE_T2I = (
    SYS_TEMPLATE + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_TI2I = (
    SYS_TEMPLATE + "<|im_start|>user\n{}{}<|im_end|>\n<|im_start|>assistant\n"
)
IMAGE_TOKEN_ID = 151655

# vision encoder
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
SPATIAL_MERGE_SIZE = 2
NUM_GRID_PER_SIDE = 48
IMAGE_MIN_PIXELS = 65536
IMAGE_MAX_PIXELS = 16777216
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5

# transformer
HIDDEN_SIZE = 4096
NUM_LAYERS = 32
NUM_HEADS = 32
HEAD_DIM = 128

# vae
VAE_SCALE_FACTOR = 16
LATENT_CHANNELS = 64
# fmt: off
LATENTS_MEAN = np.array([
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
    1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
    2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
    -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
    -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376,
    -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729,
    -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
    0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
], dtype=np.float32).reshape(1, LATENT_CHANNELS, 1, 1, 1)
LATENTS_STD = np.array([
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791,
    3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756,
    3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467,
    3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125,
    3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
    3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959,
    3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161,
], dtype=np.float32).reshape(1, LATENT_CHANNELS, 1, 1, 1)
# fmt: on

# scheduler
NUM_TRAIN_TIMESTEPS = 1000
BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 8192
BASE_SHIFT = 0.5
MAX_SHIFT = 0.9
SHIFT_TERMINAL = 0.02
TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("Qwen-Image-2.1", None, SAVE_IMAGE_PATH, large_model=True)
parser.add_argument(
    "-p",
    "--prompt",
    type=str,
    default="A capybara wearing a wizard hat, reading a book by candlelight, oil painting",
    help="the prompt to render",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================


def get_rope_index(input_ids, image_grid_thw):
    """M-RoPE (temporal, height, width) positions of the text encoder [3, 1, L]."""
    is_image = input_ids == IMAGE_TOKEN_ID
    grid_iter = iter(image_grid_thw)
    pos_parts = []
    current_pos = 0
    start = 0
    while start < len(input_ids):
        end = start
        while end < len(input_ids) and is_image[end] == is_image[start]:
            end += 1
        if not is_image[start]:
            pos = np.arange(end - start, dtype=np.int64) + current_pos
            pos_parts.append(np.tile(pos[None], (3, 1)))
            current_pos += end - start
        else:
            t, h, w = next(grid_iter)
            h, w = h // SPATIAL_MERGE_SIZE, w // SPATIAL_MERGE_SIZE
            pt = np.repeat(np.arange(t), h * w) + current_pos
            ph = np.tile(np.repeat(np.arange(h), w), t) + current_pos
            pw = np.tile(np.arange(w), h * t) + current_pos
            pos_parts.append(np.stack([pt, ph, pw]))
            current_pos += max(h, w)
        start = end
    return np.concatenate(pos_parts, axis=1)[:, None, :]


def pack_latents(latents):
    b, c, _, h, w = latents.shape
    return latents.reshape(b, c, h * w).transpose(0, 2, 1)


# ======================
# Main functions
# ======================


def encode_prompt(models, prompt):
    """-> prompt_embeds [1, L, 4096], image_pad_mask [L]"""
    tokenizer = models["tokenizer"]

    # an empty prompt leaves the encoder with nothing to read
    prompt = prompt or " "

    text = PROMPT_TEMPLATE_T2I.format(prompt)
    input_ids = np.array(tokenizer.encode(text), dtype=np.int64)
    visual_pos_masks = input_ids == IMAGE_TOKEN_ID

    embed_tokens = models["embed_tokens"]
    inputs_embeds = embed_tokens[input_ids].astype(np.float32)
    deepstack = [np.zeros((0, HIDDEN_SIZE), dtype=np.float32)] * 3

    feed = {
        "inputs_embeds": inputs_embeds[None],
        "position_ids": get_rope_index(input_ids, []),
        "visual_pos_masks": visual_pos_masks[None],
        "deepstack_0": deepstack[0],
        "deepstack_1": deepstack[1],
        "deepstack_2": deepstack[2],
    }
    net = models["text_encoder"]
    if not args.onnx:
        output = net.predict(feed)
    else:
        output = net.run(None, feed)
    hidden_states = output[0]

    # drop the system prompt
    drop_idx = len(tokenizer.encode(SYS_TEMPLATE))
    prompt_embeds = hidden_states[:, drop_idx:]
    image_pad_mask = visual_pos_masks[drop_idx:]
    return prompt_embeds, image_pad_mask


def generate(models, prompt):
    height = width = 1024
    logger.info(f"output size: {width}x{height}")

    prompt_embeds, image_pad_mask = encode_prompt(models, prompt)

    latent_height, latent_width = height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR
    latents = np.random.standard_normal(
        (1, LATENT_CHANNELS, 1, latent_height, latent_width)
    ).astype(np.float32)
    latents = pack_latents(latents)


def recognize_from_text(models):
    prompt = args.prompt

    logger.info("prompt: %s" % prompt)
    logger.info("Start inference...")
    generate(models, prompt)


def main():
    check_and_download_models(
        WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH, REMOTE_PATH
    )
    check_and_download_file(WEIGHT_TEXT_ENCODER_PB_PATH, REMOTE_PATH)
    check_and_download_file(EMBED_TOKENS_PATH, REMOTE_PATH)

    seed = 42
    np.random.seed(seed)

    env_id = args.env_id

    memory_mode = None
    providers = None
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
    else:
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )

    def load_net(model_path, weight_path, cpu_only=False):
        if not args.onnx:
            return ailia.Net(
                model_path, weight_path, env_id=env_id, memory_mode=memory_mode
            )
        else:
            import onnxruntime

            return onnxruntime.InferenceSession(
                weight_path,
                providers=["CPUExecutionProvider"] if cpu_only else providers,
            )

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    else:
        from ailia_tokenizer import Qwen2Tokenizer

        tokenizer = Qwen2Tokenizer.from_pretrained("./tokenizer")

    models = {
        "tokenizer": tokenizer,
        "embed_tokens": np.load(EMBED_TOKENS_PATH, mmap_mode="r"),
        "text_encoder": load_net(
            MODEL_TEXT_ENCODER_PATH, WEIGHT_TEXT_ENCODER_PATH, cpu_only=True
        ),
    }

    recognize_from_text(models)


if __name__ == "__main__":
    main()
