import sys
from logging import getLogger

import ailia
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa
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


# ======================
# Main functions
# ======================


def infer(net, feeds):
    if not args.onnx:
        return net.run(list(feeds.values()))
    else:
        return net.run(None, feeds)


def encode_tags(models, tokenizers, tags):
    tokenizer, tokenizer_2 = tokenizers

    def tokenize(tok):
        return tok(
            tags,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        ).input_ids.astype(np.int64)

    output = infer(models["ld_text_encoder"], {"input_ids": tokenize(tokenizer)})
    hidden_states = output[0]
    output = infer(models["ld_text_encoder_2"], {"input_ids": tokenize(tokenizer_2)})
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
    mean, std = infer(models["ld_vae_encoder"], {"argb": argb})
    c_concat = mean + std * np.random.randn(*mean.shape).astype(np.float32)
    lh, lw = c_concat.shape[-2:]
    c_concat = np.broadcast_to(c_concat[:, None], (1, num_frames, 4, lh, lw))

    # every frame starts from the same noise
    noise = np.random.randn(1, 1, 4, lh, lw).astype(np.float32)
    steps = 30
    scheduler = DPMSolverSDEScheduler()
    scheduler.set_timesteps(steps)
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


def run_layer_decomposition(models, input_img, body_embeds, head_embeds):
    """Decomposes the whole body, then the head cropped from the input image.

    Returns the input padded to a square and resized, and the layers keyed by tag.
    """
    resolution = 1280
    fullpage, pad_size, pad_pos = center_square_pad_resize(input_img, resolution)
    scale = pad_size[0] / resolution

    logger.info("Decomposing the body...")
    body = decompose_layers(models, fullpage, *body_embeds, group_index=0)
    layers = dict(zip(BODY_TAGS, body))


def recognize_from_image(models, tokenizers):
    tokenizer, tokenizer_2, tokenizer_mg = tokenizers
    # the tags are fixed, so their embeddings are computed once
    body_embeds = encode_tags(models, (tokenizer, tokenizer_2), BODY_TAGS)
    head_embeds = encode_tags(models, (tokenizer, tokenizer_2), HEAD_TAGS)

    for image_path in args.input:
        logger.info(image_path)
        input_img = np.array(Image.open(image_path).convert("RGBA"))

        seed = 42
        np.random.seed(seed)
        logger.info("Start inference...")
        fullpage, layers = run_layer_decomposition(
            models, input_img, body_embeds, head_embeds
        )


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

    models = {
        "ld_text_encoder": load_net(
            MODEL_LD_TEXT_ENCODER_PATH, WEIGHT_LD_TEXT_ENCODER_PATH
        ),
        "ld_text_encoder_2": load_net(
            MODEL_LD_TEXT_ENCODER_2_PATH, WEIGHT_LD_TEXT_ENCODER_2_PATH
        ),
        "ld_vae_encoder": load_net(
            MODEL_LD_VAE_ENCODER_PATH, WEIGHT_LD_VAE_ENCODER_PATH
        ),
        "ld_unet": load_net(MODEL_LD_UNET_PATH, WEIGHT_LD_UNET_PATH),
        "ld_vae_decoder": load_net(
            MODEL_LD_VAE_DECODER_PATH, WEIGHT_LD_VAE_DECODER_PATH
        ),
        "mg_text_encoder": load_net(
            MODEL_MG_TEXT_ENCODER_PATH, WEIGHT_MG_TEXT_ENCODER_PATH
        ),
        "mg_vae_encoder": load_net(
            MODEL_MG_VAE_ENCODER_PATH, WEIGHT_MG_VAE_ENCODER_PATH
        ),
        "mg_unet": load_net(MODEL_MG_UNET_PATH, WEIGHT_MG_UNET_PATH),
        "mg_vae_decoder": load_net(
            MODEL_MG_VAE_DECODER_PATH, WEIGHT_MG_VAE_DECODER_PATH
        ),
    }

    tokenizers = load_tokenizers()
    recognize_from_image(models, tokenizers)


if __name__ == "__main__":
    main()
