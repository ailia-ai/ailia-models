import gc
import math
import sys
import time
from logging import getLogger

import ailia
import numpy as np
from PIL import Image
from tqdm import tqdm

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa
from model_utils import check_and_download_file, check_and_download_models  # noqa
from resize_utils import tv_resize  # noqa

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
    "--width",
    type=int,
    default=None,
    help="output image width. derived from the last input image (or output_resolution) if omitted.",
)
parser.add_argument(
    "--height",
    type=int,
    default=None,
    help="output image height. derived from the last input image (or output_resolution) if omitted.",
)
parser.add_argument(
    "--output_resolution",
    type=int,
    default=1024,
    help="input images are resized to about output_resolution^2 pixels.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=40,
    help="number of sampling steps (ignored with --turbo).",
)
parser.add_argument("--seed", type=int, default=None, help="random seed")
parser.add_argument(
    "--turbo", action="store_true", help="use Qwen-Image-2.1-Turbo (8 steps)."
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================


class LazyModel:
    """Defers model loading until the first predict/run call."""

    def __init__(self, loader_fn, name=""):
        self._loader_fn = loader_fn
        self._name = name
        self._net = None

    def load(self):
        if self._net is None:
            logger.info(f"Loading model: {self._name}")
            self._net = self._loader_fn()
        return self._net

    def unload(self):
        if self._net is not None:
            self._net = None
            gc.collect()

    def predict(self, *args, **kwargs):
        return self.load().predict(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self.load().run(*args, **kwargs)


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height


def calculate_shift(image_seq_len):
    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN)
    b = BASE_SHIFT - m * BASE_IMAGE_SEQ_LEN
    return image_seq_len * m + b


def get_sigmas(num_inference_steps, image_seq_len):
    """Flow matching sigmas with the terminal 0 appended."""
    if args.turbo:
        sigmas = np.array(TURBO_SIGMAS, dtype=np.float32)
    else:
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        sigmas = sigmas.astype(np.float32)
        # resolution-dependent exponential time shift
        mu = calculate_shift(image_seq_len)
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        # stretch the schedule so that it ends at SHIFT_TERMINAL
        one_minus_z = 1 - sigmas
        scale_factor = one_minus_z[-1] / (1 - SHIFT_TERMINAL)
        sigmas = 1 - (one_minus_z / scale_factor)
    return np.concatenate([sigmas, np.zeros(1, dtype=np.float32)]).astype(np.float32)


def smart_resize(height, width, factor=PATCH_SIZE * SPATIAL_MERGE_SIZE):
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > IMAGE_MAX_PIXELS:
        beta = math.sqrt((height * width) / IMAGE_MAX_PIXELS)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < IMAGE_MIN_PIXELS:
        beta = math.sqrt(IMAGE_MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess_vision_image(img):
    """RGB PIL image -> pixel_values [N, 1536], grid_thw (1, h, w)."""
    img = np.array(img)
    height, width = smart_resize(*img.shape[:2])
    img = tv_resize(img, (height, width), interpolation="bicubic", antialias=True)
    img = (img.astype(np.float32) / 255 - IMAGE_MEAN) / IMAGE_STD
    img = img.transpose(2, 0, 1)

    grid_h, grid_w = height // PATCH_SIZE, width // PATCH_SIZE
    m = SPATIAL_MERGE_SIZE
    patches = img.reshape(3, grid_h // m, m, PATCH_SIZE, grid_w // m, m, PATCH_SIZE)
    # [grid_h/m, grid_w/m, m, m, channel, patch, patch]
    patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)
    patches = np.repeat(patches[:, :, :, :, :, None], TEMPORAL_PATCH_SIZE, axis=5)
    pixel_values = patches.reshape(
        grid_h * grid_w, 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE
    )
    return np.ascontiguousarray(pixel_values), (1, grid_h, grid_w)


def merge_block_order(arr, h, w):
    m = SPATIAL_MERGE_SIZE
    return arr.reshape(h // m, m, w // m, m, *arr.shape[2:]).swapaxes(1, 2)


def vision_interpolation(grid_thw):
    """Bilinear (align_corners=True) taps resampling the 48x48 position embedding table."""
    _, h, w = grid_thw
    side = NUM_GRID_PER_SIDE

    def axis_taps(size):
        src = np.arange(size, dtype=np.float32) * (side - 1) / max(size - 1, 1)
        floor = np.floor(src)
        offsets = np.arange(2)
        taps = np.clip(floor.astype(np.int64)[:, None] + offsets, 0, side - 1)
        weights = np.clip(1 - np.abs(src[:, None] - floor[:, None] - offsets), 0, None)
        return taps, weights.astype(np.float32)

    h_taps, h_weights = axis_taps(h)
    w_taps, w_weights = axis_taps(w)
    indices = h_taps[:, None, :, None] * side + w_taps[None, :, None, :]
    weights = h_weights[:, None, :, None] * w_weights[None, :, None, :]
    indices = merge_block_order(indices, h, w).reshape(h * w, 4)
    weights = merge_block_order(weights, h, w).reshape(h * w, 4)
    return indices, weights


def vision_position_ids(grid_thw):
    _, h, w = grid_thw
    hpos, wpos = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pos = np.stack([hpos, wpos], axis=-1)
    return merge_block_order(pos, h, w).reshape(h * w, 2).astype(np.int64)


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


def preprocess_condition_images(images, output_resolution):
    """One resize feeds both the text encoder and the VAE."""
    image_sizes, vlm_images, vae_images = [], [], []
    for img in images:
        img = img.convert("RGBA")
        width, height = calculate_dimensions(
            output_resolution * output_resolution, img.size[0] / img.size[1]
        )
        img = img.resize((width, height), resample=Image.LANCZOS)
        image_sizes.append((width, height))

        # the vision encoder reads the image composited over white
        white = Image.new("RGB", img.size, (255, 255, 255))
        white.paste(img, mask=img.getchannel("A"))
        vlm_images.append(white)

        vae_image = np.array(img).astype(np.float32) / 255 * 2 - 1
        vae_images.append(vae_image.transpose(2, 0, 1)[None, :, None])
    return image_sizes, vlm_images, vae_images


def build_transformer_inputs(image_pad_mask, img_shapes, num_target_tokens):
    """
    Lay out the joint text / image sequence the transformer attends over.
    image_pad_mask marks the image slots of the text encoder sequence, one slot per 2x2 latents.
    """
    vlm_len = len(image_pad_mask)
    img_mask = np.concatenate([image_pad_mask, np.ones(num_target_tokens // 4, bool)])
    is_image = np.repeat(img_mask, np.where(img_mask, 4, 1))
    seq_len = len(is_image)

    # RoPE positions: text advances all three axes, an image block freezes the frame axis
    # and lays out its tokens on a height / width grid centred on zero
    frame_index, image_height_index, image_width_index = [], [], []
    cursor, position = 0, 0
    for _, height, width in img_shapes:
        block_start = cursor + int(np.argmax(is_image[cursor:]))
        text_len = block_start - cursor
        frame_index.extend(range(position, position + text_len))
        position += text_len
        cursor = block_start + height * width
        frame_index.extend([position] * (height * width))
        position += max(height, width)
        h_range = np.arange(-(height - height // 2), height // 2)
        w_range = np.arange(-(width - width // 2), width // 2)
        image_height_index.append(np.repeat(h_range, width))
        image_width_index.append(np.tile(w_range, height))
    if cursor < seq_len:
        frame_index.extend(range(position, position + seq_len - cursor))
    frame_index = np.array(frame_index, dtype=np.int64)
    height_index = frame_index.copy()
    width_index = frame_index.copy()
    height_index[is_image] = np.concatenate(image_height_index)
    width_index[is_image] = np.concatenate(image_width_index)
    position_ids = np.stack([frame_index, height_index, width_index])

    # one block id per image (condition images, then the target), -1 for text
    block_lengths = [t * h * w for t, h, w in img_shapes]
    image_ids = np.full(seq_len, -1, dtype=np.int64)
    image_ids[is_image] = np.repeat(np.arange(len(block_lengths)), block_lengths)
    target_token_mask = np.zeros(seq_len, dtype=bool)
    target_token_mask[np.nonzero(is_image)[0][-block_lengths[-1] :]] = True

    # index into concat([text encoder states, image latents])
    joint_index = np.zeros(seq_len, dtype=np.int64)
    joint_index[~is_image] = np.nonzero(~img_mask)[0]
    joint_index[is_image] = vlm_len + np.arange(int(is_image.sum()))

    return joint_index, position_ids, target_token_mask, image_ids


def pack_latents(latents):
    b, c, _, h, w = latents.shape
    return latents.reshape(b, c, h * w).transpose(0, 2, 1)


def unpack_latents(latents, height, width):
    b, _, c = latents.shape
    h, w = height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR
    return latents.transpose(0, 2, 1).reshape(b, c, 1, h, w)


# ======================
# Main functions
# ======================


def encode_prompt(models, prompt, vlm_images):
    """-> prompt_embeds [1, L, 4096], image_pad_mask [L]"""
    tokenizer = models["tokenizer"]

    # an empty prompt leaves the encoder with nothing to read
    prompt = prompt or " "

    pixel_values, grid_thw = [], []
    for img in vlm_images:
        pv, thw = preprocess_vision_image(img)
        pixel_values.append(pv)
        grid_thw.append(thw)

    if vlm_images:
        image_prefix = " ".join(
            f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>"
            for i in range(len(vlm_images))
        )
        text = PROMPT_TEMPLATE_TI2I.format(image_prefix, prompt)
        for t, h, w in grid_thw:
            num_tokens = t * h * w // SPATIAL_MERGE_SIZE**2
            text = text.replace("<|image_pad|>", "<|placeholder|>" * num_tokens, 1)
        text = text.replace("<|placeholder|>", "<|image_pad|>")
    else:
        text = PROMPT_TEMPLATE_T2I.format(prompt)
    input_ids = np.array(tokenizer.encode(text), dtype=np.int64)
    visual_pos_masks = input_ids == IMAGE_TOKEN_ID

    image_embeds = []
    deepstack = [[], [], []]
    for pv, thw in zip(pixel_values, grid_thw):
        interp_indices, interp_weights = vision_interpolation(thw)
        feed = {
            "pixel_values": pv,
            "interp_indices": interp_indices,
            "interp_weights": interp_weights,
            "position_ids": vision_position_ids(thw),
        }
        net = models["vision_encoder"]
        if not args.onnx:
            output = net.predict(feed)
        else:
            output = net.run(None, feed)
        embeds, deepstack_0, deepstack_1, deepstack_2 = output
        image_embeds.append(embeds)
        for i, d in enumerate([deepstack_0, deepstack_1, deepstack_2]):
            deepstack[i].append(d)
        # ailia reuses the shapes of the previous call when the input shapes do not change,
        # so the next image (which may have the same shape but another grid) needs a new net
        net.unload()

    embed_tokens = models["embed_tokens"]
    inputs_embeds = embed_tokens[input_ids].astype(np.float32)
    if vlm_images:
        inputs_embeds[visual_pos_masks] = np.concatenate(image_embeds)
        deepstack = [np.concatenate(d) for d in deepstack]
    else:
        deepstack = [np.zeros((0, HIDDEN_SIZE), dtype=np.float32)] * 3

    feed = {
        "inputs_embeds": inputs_embeds[None],
        "position_ids": get_rope_index(input_ids, grid_thw),
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


def encode_vae_images(models, vae_images):
    """-> normalized image latents packed into tokens [1, N, 64]"""
    image_latents = []
    for image in vae_images:
        net = models["vae_encoder"]
        if not args.onnx:
            output = net.predict({"image": image})
        else:
            output = net.run(None, {"image": image})
        latent = (output[0] - LATENTS_MEAN) / LATENTS_STD
        image_latents.append(pack_latents(latent))
    return np.concatenate(image_latents, axis=1)


def denoise(models, latents, image_latents, prompt_embeds, transformer_inputs, sigmas):
    joint_index, position_ids, target_token_mask, image_ids = transformer_inputs
    num_target_tokens = latents.shape[1]
    prefix_len = len(target_token_mask) - num_target_tokens
    key_valid = np.ones((1, len(image_ids)), dtype=bool)
    net = models["transformer"]

    past_key_values = [np.zeros((1, 0, NUM_HEADS, HEAD_DIM), dtype=np.float32)] * (
        2 * NUM_LAYERS
    )
    for i in tqdm(range(len(sigmas) - 1)):
        t = sigmas[i] * NUM_TRAIN_TIMESTEPS
        timestep = np.array([t / NUM_TRAIN_TIMESTEPS], dtype=np.float32)
        hidden_states = latents
        if image_latents is not None:
            hidden_states = np.concatenate([image_latents, latents], axis=1)
        feed = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": prompt_embeds,
            "timestep": timestep,
        }
        if i == 0:
            feed["joint_index"] = joint_index
            feed["position_ids"] = position_ids
            feed["target_token_mask"] = target_token_mask
        else:
            # the text and condition image tokens are step independent, so the first step
            # caches their keys / values and later steps only compute the target tokens
            feed["joint_index"] = joint_index[prefix_len:]
            feed["position_ids"] = position_ids[:, prefix_len:]
            feed["target_token_mask"] = target_token_mask[prefix_len:]
        feed["image_ids"] = image_ids
        feed["key_valid"] = key_valid
        for j in range(NUM_LAYERS):
            feed[f"past_key_values.{j}.key"] = past_key_values[2 * j]
            feed[f"past_key_values.{j}.value"] = past_key_values[2 * j + 1]

        if not args.onnx:
            output = net.predict(feed)
        else:
            output = net.run(None, feed)
        sample = output[0]
        if i == 0:
            past_key_values = output[1:]

        noise_pred = sample[:, -num_target_tokens:]
        latents = latents + (sigmas[i + 1] - sigmas[i]) * noise_pred

    return latents


def decode_latents(models, latents, height, width):
    latents = unpack_latents(latents, height, width)
    latents = latents * LATENTS_STD + LATENTS_MEAN
    net = models["vae_decoder"]
    if not args.onnx:
        output = net.predict({"latents": latents})
    else:
        output = net.run(None, {"latents": latents})
    image = output[0][0, :, 0]
    image = np.clip(image * 0.5 + 0.5, 0, 1)
    image = (image.transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return Image.fromarray(image)


def generate(models, prompt, images):
    output_resolution = args.output_resolution
    height, width = args.height, args.width

    image_sizes, vlm_images, vae_images = [], [], []
    if images:
        calculated_width, calculated_height = calculate_dimensions(
            output_resolution * output_resolution,
            images[-1].size[0] / images[-1].size[1],
        )
        height = height or calculated_height
        width = width or calculated_width
        image_sizes, vlm_images, vae_images = preprocess_condition_images(
            images, output_resolution
        )
    height = height or output_resolution
    width = width or output_resolution
    multiple_of = VAE_SCALE_FACTOR * 2
    height = height // multiple_of * multiple_of
    width = width // multiple_of * multiple_of
    logger.info(f"output size: {width}x{height}")

    prompt_embeds, image_pad_mask = encode_prompt(models, prompt, vlm_images)
    models["text_encoder"].unload()

    image_latents = None
    if images:
        image_latents = encode_vae_images(models, vae_images)
        models["vae_encoder"].unload()

    latent_height, latent_width = height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR
    latents = np.random.standard_normal(
        (1, LATENT_CHANNELS, 1, latent_height, latent_width)
    ).astype(np.float32)
    latents = pack_latents(latents)

    img_shapes = [
        *[(1, h // VAE_SCALE_FACTOR, w // VAE_SCALE_FACTOR) for w, h in image_sizes],
        (1, latent_height, latent_width),
    ]
    transformer_inputs = build_transformer_inputs(
        image_pad_mask, img_shapes, latents.shape[1]
    )
    sigmas = get_sigmas(args.steps, latents.shape[1])

    latents = denoise(
        models, latents, image_latents, prompt_embeds, transformer_inputs, sigmas
    )
    models["transformer"].unload()

    return decode_latents(models, latents, height, width)


def recognize_from_text(models):
    prompt = args.prompt
    images = [Image.open(p) for p in args.input] if args.input else []

    logger.info("prompt: %s" % prompt)
    for p in args.input or []:
        logger.info("input image: %s" % p)
    logger.info("Start inference...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            image = generate(models, prompt, images)
            end = int(round(time.time() * 1000))
            estimation_time = end - start
            logger.info(f"\tailia processing estimation time {estimation_time} ms")
            if i != 0:
                total_time_estimation += estimation_time
        logger.info(
            f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
        )
    else:
        image = generate(models, prompt, images)

    savepath = get_savepath(args.savepath, "", ext=".png")
    image.save(savepath)
    logger.info(f"saved at : {savepath}")

    logger.info("Script finished successfully.")


def main():
    if args.turbo:
        weight_transformer_path = WEIGHT_TURBO_TRANSFORMER_PATH
        weight_transformer_pb_path = WEIGHT_TURBO_TRANSFORMER_PB_PATH
        model_transformer_path = MODEL_TURBO_TRANSFORMER_PATH
    else:
        weight_transformer_path = WEIGHT_TRANSFORMER_PATH
        weight_transformer_pb_path = WEIGHT_TRANSFORMER_PB_PATH
        model_transformer_path = MODEL_TRANSFORMER_PATH

    check_and_download_models(
        WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH, REMOTE_PATH
    )
    check_and_download_file(WEIGHT_TEXT_ENCODER_PB_PATH, REMOTE_PATH)
    check_and_download_file(EMBED_TOKENS_PATH, REMOTE_PATH)
    check_and_download_models(
        weight_transformer_path, model_transformer_path, REMOTE_PATH
    )
    check_and_download_file(weight_transformer_pb_path, REMOTE_PATH)
    check_and_download_models(
        WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH, REMOTE_PATH
    )
    if args.input:
        check_and_download_models(
            WEIGHT_VISION_ENCODER_PATH, MODEL_VISION_ENCODER_PATH, REMOTE_PATH
        )
        check_and_download_file(WEIGHT_VISION_ENCODER_PB_PATH, REMOTE_PATH)
        check_and_download_models(
            WEIGHT_VAE_ENCODER_PATH, MODEL_VAE_ENCODER_PATH, REMOTE_PATH
        )

    if args.seed is not None:
        np.random.seed(args.seed)

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
        "text_encoder": LazyModel(
            lambda: load_net(
                MODEL_TEXT_ENCODER_PATH, WEIGHT_TEXT_ENCODER_PATH, cpu_only=True
            ),
            "text_encoder",
        ),
        "transformer": LazyModel(
            lambda: load_net(
                model_transformer_path, weight_transformer_path, cpu_only=True
            ),
            "transformer",
        ),
        "vae_decoder": LazyModel(
            lambda: load_net(MODEL_VAE_DECODER_PATH, WEIGHT_VAE_DECODER_PATH),
            "vae_decoder",
        ),
    }
    if args.input:
        models["vision_encoder"] = LazyModel(
            lambda: load_net(MODEL_VISION_ENCODER_PATH, WEIGHT_VISION_ENCODER_PATH),
            "vision_encoder",
        )
        models["vae_encoder"] = LazyModel(
            lambda: load_net(MODEL_VAE_ENCODER_PATH, WEIGHT_VAE_ENCODER_PATH),
            "vae_encoder",
        )

    recognize_from_text(models)


if __name__ == "__main__":
    main()
