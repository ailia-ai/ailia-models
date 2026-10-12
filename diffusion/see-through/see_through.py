import gc
import json
import os
import sys
from logging import getLogger

import ailia
import cv2
import numpy as np
from PIL import Image, ImageDraw
from psd_tools import PSDImage
from tqdm import tqdm

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa
from model_utils import check_and_download_file, check_and_download_models  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_LD_TEXT_ENCODER_PATH = "layerdiff_text_encoder.onnx"
MODEL_LD_TEXT_ENCODER_PATH = "layerdiff_text_encoder.onnx.prototxt"
WEIGHT_LD_TEXT_ENCODER_2_PATH = "layerdiff_text_encoder_2.onnx"
WEIGHT_LD_TEXT_ENCODER_2_PB_PATH = "layerdiff_text_encoder_2_weights.pb"
MODEL_LD_TEXT_ENCODER_2_PATH = "layerdiff_text_encoder_2.onnx.prototxt"
WEIGHT_LD_VAE_ENCODER_PATH = "layerdiff_vae_encoder.onnx"
MODEL_LD_VAE_ENCODER_PATH = "layerdiff_vae_encoder.onnx.prototxt"
WEIGHT_LD_UNET_PATH = "layerdiff_unet.onnx"
WEIGHT_LD_UNET_PB_PATH = "layerdiff_unet_weights.pb"
MODEL_LD_UNET_PATH = "layerdiff_unet.onnx.prototxt"
WEIGHT_LD_VAE_DECODER_PATH = "layerdiff_vae_decoder.onnx"
MODEL_LD_VAE_DECODER_PATH = "layerdiff_vae_decoder.onnx.prototxt"
WEIGHT_MG_TEXT_ENCODER_PATH = "marigold_text_encoder.onnx"
MODEL_MG_TEXT_ENCODER_PATH = "marigold_text_encoder.onnx.prototxt"
WEIGHT_MG_VAE_ENCODER_PATH = "marigold_vae_encoder.onnx"
MODEL_MG_VAE_ENCODER_PATH = "marigold_vae_encoder.onnx.prototxt"
WEIGHT_MG_UNET_PATH = "marigold_unet.onnx"
WEIGHT_MG_UNET_PB_PATH = "marigold_unet_weights.pb"
MODEL_MG_UNET_PATH = "marigold_unet.onnx.prototxt"
WEIGHT_MG_VAE_DECODER_PATH = "marigold_vae_decoder.onnx"
MODEL_MG_VAE_DECODER_PATH = "marigold_vae_decoder.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/see-through/"

IMAGE_PATH = "input.png"
SAVE_PATH = "output.png"

# fmt: off
BODY_TAGS = [
    "front hair", "back hair", "head", "neck", "neckwear", "topwear", "handwear",
    "bottomwear", "legwear", "footwear", "tail", "wings", "objects",
]
HEAD_TAGS = [
    "headwear", "face", "irides", "eyebrow", "eyewhite", "eyelash", "eyewear",
    "ears", "earwear", "nose", "mouth",
]
# Layers whose depth is estimated. "hair" and "eyes" are estimated as the
# composites of the layers listed in COMPOSE_TAGS.
DEPTH_TAGS = [
    "hair", "headwear", "face", "eyes", "eyewear", "ears", "earwear", "nose", "mouth",
    "neck", "neckwear", "topwear", "handwear", "bottomwear", "legwear", "footwear",
    "tail", "wings", "objects",
]
COMPOSE_TAGS = {
    "eyes": ["eyewhite", "irides", "eyelash", "eyebrow"],
    "hair": ["back hair", "front hair"],
}
# fmt: on

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("See-through", IMAGE_PATH, SAVE_PATH)
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument(
    "--resolution",
    type=int,
    default=640,
    help="inference resolution of layer decomposition (1280 for sharper layers)",
)
parser.add_argument(
    "--resolution_depth",
    type=int,
    default=512,
    help="inference resolution of depth estimation (768 for sharper depth)",
)
parser.add_argument(
    "--steps",
    type=int,
    default=30,
    help="number of sampling steps of layer decomposition",
)
parser.add_argument(
    "--steps_depth",
    type=int,
    default=4,
    help="number of sampling steps of depth estimation",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================


def smart_resize(img, size):
    h, w = img.shape[:2]
    th, tw = size
    if th == h and tw == w:
        return img.copy()
    interpolation = cv2.INTER_AREA if th * tw < h * w else cv2.INTER_LINEAR
    return cv2.resize(img, (tw, th), interpolation=interpolation)


def center_square_pad_resize(img, size):
    """Pads the image to a square at the center, then resizes it to size x size."""
    h, w = img.shape[:2]
    pad_size = (w, h)
    pad_pos = (0, 0)
    if h != w:
        sz = max(h, w)
        px1 = (sz - w) // 2
        py1 = (sz - h) // 2
        padded = np.zeros((sz, sz, img.shape[-1]), dtype=img.dtype)
        padded[py1 : py1 + h, px1 : px1 + w] = img
        img = padded
        pad_size = (sz, sz)
        pad_pos = (px1, py1)
    img = smart_resize(img, (size, size))
    return img, pad_size, pad_pos


def crop_head(img, xywh):
    """Crops the head region, extended by up to 1/5 of its size when it is small."""
    x, y, w, h = xywh
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = x, y, x + w, y + h
    if w < iw // 2:
        px = min(iw - x - w, x, w // 5)
        x1 = min(max(x - px, 0), iw)
        x2 = min(max(x + w + px, 0), iw)
    if h < ih // 2:
        py = min(ih - y - h, y, h // 5)
        y2 = min(max(y + h + py, 0), ih)
        y1 = min(max(y - py, 0), ih)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def alpha_blend(img_list):
    """Over-composites straight-alpha RGBA images in list order."""
    h, w = img_list[0].shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    alpha = np.zeros((h, w, 1), dtype=np.float32)
    for img in img_list:
        a = img[..., 3:] / 255
        # accumulated in float32 in place
        alpha += a
        alpha = np.clip(alpha, 0, 1)
        rgb = rgb * (1 - a) + img[..., :3] * a
    out = np.concatenate([rgb, np.clip(alpha, 0, 1) * 255], axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


def depth_blend(img_list, depth_list):
    """Composites RGBA images so that the pixel with the smaller depth comes to the front."""
    h, w = img_list[0].shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    alpha = np.zeros((h, w, 1), dtype=np.float32)
    final_depth = np.full_like(alpha, fill_value=255)
    for img, depth in zip(img_list, depth_list):
        depth = depth[..., None]
        a = img[..., 3:] / 255
        img = img[..., :3]
        update_mask = (final_depth > depth).astype(np.uint8)
        final_depth = update_mask * depth + (1 - update_mask) * final_depth
        rgb = update_mask * (rgb * (1 - a) + img) + (1 - update_mask) * (
            img * (1 - alpha) + rgb
        )
        alpha = np.clip(alpha + a, 0, 1)
    out = np.concatenate([rgb, np.clip(alpha, 0, 1) * 255], axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


def pad_rgb(argb):
    """Fills the color of transparent pixels from the surrounding opaque pixels.

    argb: (4, H, W) in [0, 1]. A pyramid of alpha-premultiplied images is built by
    shrinking 1/1.2 at a time, and the color is propagated from the coarsest level.
    """

    def resize(x, out_h, out_w):
        x = cv2.resize(
            x.transpose(1, 2, 0), (out_w, out_h), interpolation=cv2.INTER_LINEAR
        )
        return x.reshape(out_h, out_w, -1).transpose(2, 0, 1)

    x = argb.copy()
    x[1:] = x[1:] * x[:1]
    pyramid = []
    while True:
        pyramid.append(x)
        _, h, w = x.shape
        if min(h, w) == 1:
            break
        # the output height/width are computed from the input width/height
        x = resize(x, int(w / 1.2), int(h / 1.2))
    pyramid = pyramid[::-1]

    top = pyramid[0]
    fg = top[1:].sum(axis=(1, 2), keepdims=True) / np.clip(
        top[:1].sum(axis=(1, 2), keepdims=True), 1e-8, 1e32
    )
    for layer in pyramid:
        _, lh, lw = layer.shape
        fg = resize(fg, lw, lh)
        fg = layer[1:] + fg * (1.0 - layer[:1])
    return np.concatenate([argb[:1], fg], axis=0)


# ======================
# Schedulers
# ======================


def scaled_linear_alphas_cumprod(
    num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012
):
    betas = (
        np.linspace(
            beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32
        )
        ** 2
    )
    return np.cumprod(1.0 - betas, axis=0)


class DPMSolverSDEScheduler:
    """SDE-DPM-Solver++(2M), epsilon prediction, "leading" timestep spacing, final sigma 0."""

    def __init__(self, num_train_timesteps=1000, steps_offset=1):
        self.num_train_timesteps = num_train_timesteps
        self.steps_offset = steps_offset
        self.alphas_cumprod = scaled_linear_alphas_cumprod(num_train_timesteps)
        self.init_noise_sigma = 1.0

    def set_timesteps(self, num_inference_steps):
        step_ratio = self.num_train_timesteps // (num_inference_steps + 1)
        timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][
            :-1
        ]
        self.timesteps = timesteps.astype(np.int64) + self.steps_offset
        sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        sigmas = np.interp(self.timesteps, np.arange(0, len(sigmas)), sigmas)
        self.sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.prev_x0 = None
        self.step_index = 0

    def _lambda(self, sigma):
        alpha_t = 1 / (sigma**2 + 1) ** 0.5
        sigma_t = sigma * alpha_t
        with np.errstate(divide="ignore"):
            return alpha_t, sigma_t, np.log(alpha_t) - np.log(sigma_t)

    def step(self, model_output, sample):
        i = self.step_index
        alpha_s0, sigma_s0, lambda_s0 = self._lambda(self.sigmas[i])
        alpha_t, sigma_t, lambda_t = self._lambda(self.sigmas[i + 1])
        x0 = (sample - sigma_s0 * model_output) / alpha_s0

        h = lambda_t - lambda_s0
        noise = np.random.randn(*model_output.shape).astype(np.float32)
        x_t = (
            (sigma_t / sigma_s0 * np.exp(-h)) * sample
            + (alpha_t * (1 - np.exp(-2.0 * h))) * x0
            + sigma_t * np.sqrt(1.0 - np.exp(-2.0 * h)) * noise
        )
        # Second order after the first step, except at the last step
        if self.prev_x0 is not None and i < len(self.timesteps) - 1:
            _, _, lambda_s1 = self._lambda(self.sigmas[i - 1])
            r0 = (lambda_s0 - lambda_s1) / h
            d1 = (1.0 / r0) * (x0 - self.prev_x0)
            x_t = x_t + 0.5 * (alpha_t * (1 - np.exp(-2.0 * h))) * d1
        self.prev_x0 = x0
        self.step_index += 1
        return x_t.astype(np.float32)


class DDIMScheduler:
    """DDIM (eta=0), v prediction, zero terminal SNR, "trailing" timestep spacing."""

    def __init__(self, num_train_timesteps=1000):
        self.num_train_timesteps = num_train_timesteps
        alphas_cumprod = scaled_linear_alphas_cumprod(num_train_timesteps)

        # rescale the betas to have zero terminal SNR
        alphas_bar_sqrt = np.sqrt(alphas_cumprod)
        a0, a_t = alphas_bar_sqrt[0], alphas_bar_sqrt[-1]
        alphas_bar_sqrt = (alphas_bar_sqrt - a_t) * a0 / (a0 - a_t)
        alphas_bar = alphas_bar_sqrt**2
        alphas = np.concatenate([alphas_bar[:1], alphas_bar[1:] / alphas_bar[:-1]])
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.final_alpha_cumprod = self.alphas_cumprod[0]

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps / num_inference_steps
        timesteps = np.round(np.arange(self.num_train_timesteps, 0, -step_ratio))
        self.timesteps = timesteps.astype(np.int64) - 1

    def step(self, model_output, timestep, sample):
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (
            alpha_prod_t**0.5 * sample - beta_prod_t**0.5 * model_output
        )
        pred_epsilon = alpha_prod_t**0.5 * model_output + beta_prod_t**0.5 * sample
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * pred_epsilon
        prev_sample = (
            alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        )
        return prev_sample.astype(np.float32)


# ======================
# Main functions
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


def infer(net, feeds):
    if not args.onnx:
        return net.run(list(feeds.values()))
    else:
        return net.run(None, feeds)


def encode_tags(models, tags):

    def tokenize(tok):
        return tok(
            tags,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        ).input_ids.astype(np.int64)

    output = infer(
        models["ld_text_encoder"], {"input_ids": tokenize(models["ld_tokenizer"])}
    )
    hidden_states = output[0]
    output = infer(
        models["ld_text_encoder_2"], {"input_ids": tokenize(models["ld_tokenizer_2"])}
    )
    hidden_states_2, text_embeds = output

    prompt_embeds = np.concatenate([hidden_states, hidden_states_2], axis=-1)
    return prompt_embeds, text_embeds


def decompose_layers(models, fullpage, prompt_embeds, text_embeds, group_index):
    """Generates one transparent layer per tag. Returns a list of RGBA images."""
    num_frames = len(prompt_embeds)

    page_alpha = fullpage[..., 3:] / 255.0
    rgb = fullpage[..., :3]
    argb = np.concatenate([np.full_like(rgb[..., :1], 255), rgb], axis=2)
    argb = (argb.transpose(2, 0, 1)[None] / 255).astype(np.float32)
    # one model at a time is kept in memory, as the UNet alone takes most of the GPU memory
    mean, std = infer(models["ld_vae_encoder"], {"argb": argb})
    models["ld_vae_encoder"].unload()
    c_concat = mean + std * np.random.randn(*mean.shape).astype(np.float32)
    lh, lw = c_concat.shape[-2:]
    c_concat = np.broadcast_to(c_concat[:, None], (1, num_frames, 4, lh, lw))

    # every frame starts from the same noise
    noise = np.random.randn(1, 1, 4, lh, lw).astype(np.float32)
    scheduler = DPMSolverSDEScheduler()
    scheduler.set_timesteps(args.steps)
    latents = (
        np.broadcast_to(noise, (1, num_frames, 4, lh, lw)) * scheduler.init_noise_sigma
    )

    height, width = lh * 8, lw * 8
    time_ids = np.array([[height, width, 0, 0, height, width]], dtype=np.float32)
    time_ids = np.repeat(time_ids, num_frames, axis=0)
    for t in tqdm(scheduler.timesteps):
        sample = np.concatenate([latents, c_concat], axis=2).astype(np.float32)
        output = infer(
            models["ld_unet"],
            {
                "sample": sample,
                "timestep": np.array([t], dtype=np.float32),
                "encoder_hidden_states": prompt_embeds,
                "text_embeds": text_embeds,
                "time_ids": time_ids,
                "group_index": np.array(group_index, dtype=np.int64),
            },
        )
        noise_pred = output[0]
        latents = scheduler.step(noise_pred, latents)
    models["ld_unet"].unload()

    layers = []
    for latent in latents[0]:
        output = infer(models["ld_vae_decoder"], {"latent": latent[None]})
        argb = output[0][0].transpose(1, 2, 0)
        alpha = argb[..., :1] * page_alpha
        png = np.concatenate([argb[..., 1:], alpha], axis=2)
        layers.append((png * 255.0).clip(0, 255).astype(np.uint8))
    models["ld_vae_decoder"].unload()
    return layers


def run_layer_decomposition(models, input_img, body_embeds, head_embeds):
    """Decomposes the whole body, then the head cropped from the input image.

    Returns the input padded to a square and resized, and the layers keyed by tag.
    """
    resolution = args.resolution
    fullpage, pad_size, pad_pos = center_square_pad_resize(input_img, resolution)
    scale = pad_size[0] / resolution

    logger.info("Decomposing the body...")
    body = decompose_layers(models, fullpage, *body_embeds, group_index=0)
    layers = dict(zip(BODY_TAGS, body))

    head_img = layers["head"]
    hx0, hy0, hw, hh = cv2.boundingRect(
        cv2.findNonZero((head_img[..., -1] > 15).astype(np.uint8))
    )
    hx = int(hx0 * scale) - pad_pos[0]
    hy = int(hy0 * scale) - pad_pos[1]
    hw = int(hw * scale)
    hh = int(hh * scale)
    input_head, (hx1, hy1, _, _) = crop_head(input_img, [hx, hy, hw, hh])
    hx1 = int(hx1 / scale + pad_pos[0] / scale)
    hy1 = int(hy1 / scale + pad_pos[1] / scale)
    ih, iw = input_head.shape[:2]
    input_head, pad_size, pad_pos = center_square_pad_resize(input_head, resolution)

    logger.info("Decomposing the head...")
    head = decompose_layers(models, input_head, *head_embeds, group_index=1)

    # paste the head layers back onto the whole image canvas
    canvas = np.zeros((resolution, resolution, 4), dtype=np.uint8)
    py1, py2, px1, px2 = (
        np.array([pad_pos[1], pad_pos[1] + ih, pad_pos[0], pad_pos[0] + iw]) / scale
    ).astype(np.int64)
    scale_size = (int(pad_size[0] / scale), int(pad_size[1] / scale))
    for tag, layer in zip(HEAD_TAGS, head):
        layer = smart_resize(layer, scale_size)[py1:py2, px1:px2]
        full = canvas.copy()
        full[hy1 : hy1 + layer.shape[0], hx1 : hx1 + layer.shape[1]] = layer
        layers[tag] = full

    return fullpage, layers


def estimate_depth(models, fullpage, layers):
    """Estimates the depth of all layers at once.

    Returns uint8 depth maps keyed by tag, and the layers composited in depth order.
    """
    src_h, src_w = fullpage.shape[:2]
    resolution = args.resolution_depth

    img_list = []
    for tag in DEPTH_TAGS:
        if tag in layers:
            img = layers[tag].copy()
            img[..., -1][img[..., -1] < 15] = 0
        else:
            img = np.zeros((src_h, src_w, 4), dtype=np.uint8)
        img_list.append(img)
    compose_layers = {}
    for tag, sub_tags in COMPOSE_TAGS.items():
        imgs = []
        for t in sub_tags:
            img = layers[t].copy()
            img[..., -1][img[..., -1] < 15] = 0
            imgs.append(img)
        img_list[DEPTH_TAGS.index(tag)] = alpha_blend(imgs)
        compose_layers[tag] = imgs

    blended_alpha = np.zeros((src_h, src_w), dtype=np.float32)
    for img in img_list:
        blended_alpha += img[..., -1].astype(np.float32) / 255
    blended_alpha = (np.clip(blended_alpha, 0, 1) * 255).astype(np.uint8)
    page = fullpage.copy()
    page[..., -1] = blended_alpha
    img_list.append(page)

    src_rescaled = resolution != src_h or resolution != src_w
    img_list_input = img_list
    if src_rescaled:
        img_list_input = [
            smart_resize(img, (resolution, resolution)) for img in img_list
        ]

    latents = []
    for img in img_list_input:
        argb = (
            np.concatenate([img[..., 3:], img[..., :3]], axis=2).astype(np.float32)
            / 255.0
        )
        argb = pad_rgb(argb.transpose(2, 0, 1))
        output = infer(models["mg_vae_encoder"], {"argb": argb[None]})
        latents.append(output[0])
    latents = np.concatenate(latents)
    models["mg_vae_encoder"].unload()
    num_frames = len(latents)
    cond_latent = np.concatenate(
        [np.broadcast_to(latents[-1:], latents.shape), latents], axis=1
    )

    input_ids = models["mg_tokenizer"](
        "", padding="do_not_pad", max_length=77, truncation=True, return_tensors="np"
    ).input_ids.astype(np.int64)
    output = infer(models["mg_text_encoder"], {"input_ids": input_ids})
    empty_text_embed = output[0]
    models["mg_text_encoder"].unload()
    encoder_hidden_states = np.repeat(empty_text_embed, num_frames, axis=0)

    target_latent = np.random.randn(num_frames, 4, *cond_latent.shape[-2:]).astype(
        np.float32
    )
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(args.steps_depth)
    for t in tqdm(scheduler.timesteps):
        sample = np.concatenate([cond_latent, target_latent], axis=1)[None]
        output = infer(
            models["mg_unet"],
            {
                "sample": sample,
                "timestep": np.array([t], dtype=np.float32),
                "encoder_hidden_states": encoder_hidden_states,
            },
        )
        noise_pred = output[0][0]
        target_latent = scheduler.step(noise_pred, t, target_latent)
    models["mg_unet"].unload()

    depth_pred = []
    for latent in target_latent:
        output = infer(models["mg_vae_decoder"], {"latent": latent[None]})
        depth = np.clip(output[0][0, 0], 0, 1)
        if src_rescaled:
            depth = smart_resize(depth, (src_h, src_w))
        depth_pred.append(depth)
    reconstruction = depth_blend(img_list[:-1], depth_pred[:-1])

    depths = {}
    for tag, depth in zip(DEPTH_TAGS, depth_pred[:-1]):
        depth = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
        if tag not in compose_layers:
            depths[tag] = depth
            continue
        # front layers first; pixels hidden by a front layer get the median depth
        # of the visible part
        mask = np.zeros((src_h, src_w), dtype=bool)
        for t, img in zip(COMPOSE_TAGS[tag][::-1], compose_layers[tag][::-1]):
            mask_local = img[..., -1] > 15
            mask_invis = np.bitwise_and(mask, mask_local)
            depth_local = np.full((src_h, src_w), fill_value=255, dtype=np.uint8)
            depth_local[mask_local] = depth[mask_local]
            if np.any(mask_invis):
                depth_local[mask_invis] = np.median(
                    depth[np.bitwise_and(mask_local, np.bitwise_not(mask_invis))]
                )
            mask = np.bitwise_or(mask, mask_local)
            depths[t] = depth_local
    return depths, reconstruction


def extract_parts(fullpage, layers, depths):
    """Crops each visible layer and assigns the median depth used for the layer order."""
    parts = {}
    for tag, depth in depths.items():
        img = layers[tag]
        mask = img[..., -1] > 10
        p_test = max(img.shape[:2]) // 10
        if np.sum(mask[:-p_test, :-p_test]) <= 4:
            continue
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask.astype(np.uint8)))
        x1, y1, x2, y2 = x, y, x + w, y + h
        # the x extent is padded by (64 - bbox height), the y extent by (12 - bbox width)
        p = 64 - h
        if p > 0:
            if x1 >= p:
                x1 -= p
            else:
                x2 += p
        p = 12 - w
        if p > 0:
            if y1 >= p:
                y1 -= p
            else:
                y2 += p
        parts[tag] = {
            "img": img[y1:y2, x1:x2].copy(),
            "depth": depth[y1:y2, x1:x2].astype(np.float32) / 255,
            "xyxy": [x1, y1, x2, y2],
            "part_id": len(parts),
        }

    for tag in ["nose", "mouth"]:
        if tag in parts:
            x1, y1, x2, y2 = parts[tag]["xyxy"]
            parts[tag]["img"][..., :3] = fullpage[y1:y2, x1:x2, :3]

    for tag, part in parts.items():
        img, depth = part["img"], part["depth"]
        mask = img[..., -1] > 10
        part["depth_median"] = np.median(depth[mask])
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask.astype(np.uint8)))
        part["img"] = img[y : y + h, x : x + w]
        depth = depth[y : y + h, x : x + w]
        part["depth"] = np.round(np.clip(depth, 0, 1) * 255).astype(np.uint8)
        x1, y1, _, _ = part["xyxy"]
        part["xyxy"] = [x1 + x, y1 + y, x1 + x + w, y1 + y + h]

    if "face" in parts:
        face_depth = parts["face"]["depth_median"]
        for tag in ["nose", "mouth"]:
            if tag in parts and parts[tag]["depth_median"] > face_depth:
                parts[tag]["depth_median"] = face_depth - 0.001
        if "ears" in parts:
            parts["ears"]["depth_median"] = face_depth + 0.001

    return parts


def checkerboard(h, w, size=16):
    yy, xx = np.mgrid[:h, :w]
    board = (((yy // size) + (xx // size)) % 2).astype(np.float32)
    return (board * 55 + 200)[..., None].repeat(3, axis=-1)


def union_bbox(imgs, margin=0.05):
    """Returns the square bbox (x1, y1, x2, y2) enclosing the opaque pixels of all images."""
    mask = np.any([img[..., 3] > 15 for img in imgs], axis=0)
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) * (0.5 + margin)
    return (
        int(max(cx - half, 0)),
        int(max(cy - half, 0)),
        int(min(cx + half, w)),
        int(min(cy + half, h)),
    )


def save_layer_grid(savepath, fullpage, reconstruction, layers, cell=320, ncol=7):
    """Saves the layers drawn over a checkerboard side by side.

    The body layers are cropped to the whole character and the head layers to the head.
    """
    body = [("input", fullpage), ("reconstruction", reconstruction)]
    body += [(tag, layers[tag]) for tag in BODY_TAGS]
    head = [(tag, layers[tag]) for tag in HEAD_TAGS]

    body_box = union_bbox([img for tag, img in body if tag != "input"])
    head_box = union_bbox([img for _, img in head])
    items = [(tag, img, body_box) for tag, img in body]
    items += [(tag, img, head_box) for tag, img in head]

    nrow = (len(items) + ncol - 1) // ncol
    label_h = 20
    grid = Image.new("RGB", (ncol * cell, nrow * (cell + label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    for i, (tag, img, (x1, y1, x2, y2)) in enumerate(items):
        crop = Image.fromarray(img[y1:y2, x1:x2]).resize((cell, cell), Image.BILINEAR)
        rgba = np.asarray(crop).astype(np.float32)
        alpha = rgba[..., 3:] / 255
        comp = rgba[..., :3] * alpha + checkerboard(cell, cell) * (1 - alpha)
        x, y = (i % ncol) * cell, (i // ncol) * (cell + label_h)
        grid.paste(Image.fromarray(comp.astype(np.uint8)), (x, y + label_h))
        draw.text((x + 4, y + 4), tag, fill=(0, 0, 0))
    grid.save(savepath)


def save_psd(savepath, part_list, size, mode="RGBA", img_key="img"):
    psd = PSDImage.new(mode=mode, size=size, depth=8)
    for part in part_list:
        x1, y1, _, _ = part["xyxy"]
        psd.create_pixel_layer(
            Image.fromarray(part[img_key]),
            name=part["tag"],
            top=y1,
            left=x1,
            opacity=255,
        )
    psd.save(savepath)


def recognize_from_image(models):
    # the tags are fixed, so their embeddings are computed once
    body_embeds = encode_tags(models, BODY_TAGS)
    head_embeds = encode_tags(models, HEAD_TAGS)
    models["ld_text_encoder"].unload()
    models["ld_text_encoder_2"].unload()

    for image_path in args.input:
        logger.info(image_path)
        input_img = np.array(Image.open(image_path).convert("RGBA"))

        if args.seed is not None:
            np.random.seed(args.seed)
        logger.info("Start inference...")
        fullpage, layers = run_layer_decomposition(
            models, input_img, body_embeds, head_embeds
        )

        if args.seed is not None:
            np.random.seed(args.seed)
        logger.info("Estimating depth...")
        depths, reconstruction = estimate_depth(models, fullpage, layers)
        models["mg_vae_decoder"].unload()

        parts = extract_parts(fullpage, layers, depths)
        # layers are stacked from the back (largest depth)
        part_list = [dict(tag=tag, **part) for tag, part in parts.items()]
        part_list.sort(key=lambda x: x["depth_median"], reverse=True)

        savepath = get_savepath(args.savepath, image_path, ext=".png")
        Image.fromarray(reconstruction).save(savepath)
        logger.info(f"saved at : {savepath}")
        grid_savepath = os.path.splitext(savepath)[0] + "_layers.png"
        save_layer_grid(grid_savepath, fullpage, reconstruction, layers)
        logger.info(f"saved at : {grid_savepath}")

        psd_savepath = os.path.splitext(savepath)[0] + ".psd"
        size = fullpage.shape[:2][::-1]
        save_psd(psd_savepath, part_list, size)
        logger.info(f"saved at : {psd_savepath}")
        depth_savepath = os.path.splitext(savepath)[0] + "_depth.psd"
        save_psd(depth_savepath, part_list, size, mode="L", img_key="depth")
        logger.info(f"saved at : {depth_savepath}")

        info = {
            "parts": {
                tag: {
                    "xyxy": [int(v) for v in part["xyxy"]],
                    "tag": tag,
                    "part_id": part["part_id"],
                    "depth_median": float(part["depth_median"]),
                }
                for tag, part in parts.items()
            },
            "frame_size": list(fullpage.shape[:2]),
        }
        with open(psd_savepath + ".json", "w", encoding="utf-8") as f:
            f.write(json.dumps(info, ensure_ascii=False))
        logger.info(f"saved at : {psd_savepath}.json")

    logger.info("Script finished successfully.")


def load_tokenizers():
    if args.disable_ailia_tokenizer:
        import transformers

        tokenizer = transformers.CLIPTokenizer.from_pretrained("./tokenizer")
        tokenizer_2 = transformers.CLIPTokenizer.from_pretrained("./tokenizer_2")
        tokenizer_mg = transformers.CLIPTokenizer.from_pretrained(
            "./tokenizer_marigold"
        )
    else:
        from ailia_tokenizer import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained()
        tokenizer_2 = CLIPTokenizer.from_pretrained()
        tokenizer_2.add_special_tokens({"pad_token": "!"})
        tokenizer_mg = CLIPTokenizer.from_pretrained()
    return tokenizer, tokenizer_2, tokenizer_mg


def main():
    check_and_download_models(
        WEIGHT_LD_TEXT_ENCODER_PATH, MODEL_LD_TEXT_ENCODER_PATH, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_LD_TEXT_ENCODER_2_PATH, MODEL_LD_TEXT_ENCODER_2_PATH, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_LD_VAE_ENCODER_PATH, MODEL_LD_VAE_ENCODER_PATH, REMOTE_PATH
    )
    check_and_download_models(WEIGHT_LD_UNET_PATH, MODEL_LD_UNET_PATH, REMOTE_PATH)
    check_and_download_models(
        WEIGHT_LD_VAE_DECODER_PATH, MODEL_LD_VAE_DECODER_PATH, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_MG_TEXT_ENCODER_PATH, MODEL_MG_TEXT_ENCODER_PATH, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_MG_VAE_ENCODER_PATH, MODEL_MG_VAE_ENCODER_PATH, REMOTE_PATH
    )
    check_and_download_models(WEIGHT_MG_UNET_PATH, MODEL_MG_UNET_PATH, REMOTE_PATH)
    check_and_download_models(
        WEIGHT_MG_VAE_DECODER_PATH, MODEL_MG_VAE_DECODER_PATH, REMOTE_PATH
    )
    check_and_download_file(WEIGHT_LD_TEXT_ENCODER_2_PB_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_LD_UNET_PB_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_MG_UNET_PB_PATH, REMOTE_PATH)

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

    def load_net(model_path, weight_path):
        if not args.onnx:
            return ailia.Net(
                model_path, weight_path, env_id=env_id, memory_mode=memory_mode
            )
        else:
            import onnxruntime

            return onnxruntime.InferenceSession(weight_path, providers=providers)

    tokenizer, tokenizer_2, tokenizer_mg = load_tokenizers()
    models = {
        "ld_tokenizer": tokenizer,
        "ld_tokenizer_2": tokenizer_2,
        "mg_tokenizer": tokenizer_mg,
        "ld_text_encoder": LazyModel(
            lambda: load_net(MODEL_LD_TEXT_ENCODER_PATH, WEIGHT_LD_TEXT_ENCODER_PATH),
            "ld_text_encoder",
        ),
        "ld_text_encoder_2": LazyModel(
            lambda: load_net(
                MODEL_LD_TEXT_ENCODER_2_PATH, WEIGHT_LD_TEXT_ENCODER_2_PATH
            ),
            "ld_text_encoder_2",
        ),
        "ld_vae_encoder": LazyModel(
            lambda: load_net(MODEL_LD_VAE_ENCODER_PATH, WEIGHT_LD_VAE_ENCODER_PATH),
            "ld_vae_encoder",
        ),
        "ld_unet": LazyModel(
            lambda: load_net(MODEL_LD_UNET_PATH, WEIGHT_LD_UNET_PATH), "ld_unet"
        ),
        "ld_vae_decoder": LazyModel(
            lambda: load_net(MODEL_LD_VAE_DECODER_PATH, WEIGHT_LD_VAE_DECODER_PATH),
            "ld_vae_decoder",
        ),
        "mg_text_encoder": LazyModel(
            lambda: load_net(MODEL_MG_TEXT_ENCODER_PATH, WEIGHT_MG_TEXT_ENCODER_PATH),
            "mg_text_encoder",
        ),
        "mg_vae_encoder": LazyModel(
            lambda: load_net(MODEL_MG_VAE_ENCODER_PATH, WEIGHT_MG_VAE_ENCODER_PATH),
            "mg_vae_encoder",
        ),
        "mg_unet": LazyModel(
            lambda: load_net(MODEL_MG_UNET_PATH, WEIGHT_MG_UNET_PATH), "mg_unet"
        ),
        "mg_vae_decoder": LazyModel(
            lambda: load_net(MODEL_MG_VAE_DECODER_PATH, WEIGHT_MG_VAE_DECODER_PATH),
            "mg_vae_decoder",
        ),
    }

    recognize_from_image(models)


if __name__ == "__main__":
    main()
