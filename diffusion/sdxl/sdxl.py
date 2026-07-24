import gc
import math
import sys
from logging import getLogger

import ailia
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa
from model_utils import check_and_download_file, check_and_download_models  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_UNET_PATH = "sdxl_unet.onnx"
WEIGHT_UNET_PB_PATH = "sdxl_unet_weights.pb"
MODEL_UNET_PATH = "sdxl_unet.onnx.prototxt"
WEIGHT_CLIP_L_PATH = "sdxl_text_encoder_clip_l.onnx"
WEIGHT_CLIP_L_PB_PATH = "sdxl_text_encoder_clip_l_weights.pb"
MODEL_CLIP_L_PATH = "sdxl_text_encoder_clip_l.onnx.prototxt"
WEIGHT_OPEN_CLIP_PATH = "sdxl_text_encoder_open_clip_bigg.onnx"
WEIGHT_OPEN_CLIP_PB_PATH = "sdxl_text_encoder_open_clip_bigg_weights.pb"
MODEL_OPEN_CLIP_PATH = "sdxl_text_encoder_open_clip_bigg.onnx.prototxt"
WEIGHT_VAE_DECODER_PATH = "sdxl_vae_decoder.onnx"
WEIGHT_VAE_DECODER_PB_PATH = "sdxl_vae_decoder_weights.pb"
MODEL_VAE_DECODER_PATH = "sdxl_vae_decoder.onnx.prototxt"
WEIGHT_VAE_ENCODER_PATH = "sdxl_vae_encoder.onnx"
WEIGHT_VAE_ENCODER_PB_PATH = "sdxl_vae_encoder_weights.pb"
MODEL_VAE_ENCODER_PATH = "sdxl_vae_encoder.onnx.prototxt"

WEIGHT_REFINER_UNET_PATH = "sdxl_refiner_unet.onnx"
WEIGHT_REFINER_UNET_PB_PATH = "sdxl_refiner_unet_weights.pb"
MODEL_REFINER_UNET_PATH = "sdxl_refiner_unet.onnx.prototxt"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/sdxl/"

SAVE_IMAGE_PATH = "output.png"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    "Stable Diffusion XL", None, SAVE_IMAGE_PATH, fp16_support=False
)
parser.add_argument(
    "-i",
    "--input",
    metavar="TEXT",
    type=str,
    default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    help="the prompt to render",
)
parser.add_argument(
    "--width",
    type=int,
    default=1024,
    help="output image width",
)
parser.add_argument(
    "--height",
    type=int,
    default=1024,
    help="output image height",
)
parser.add_argument(
    "--steps",
    type=int,
    default=50,
    help="number of sampling steps",
)
parser.add_argument(
    "--guidance_scale",
    type=float,
    default=5.0,
    help="classifier free guidance scale",
)
parser.add_argument(
    "--input_image",
    metavar="IMAGE_PATH",
    type=str,
    default=None,
    help="input image for img2img.",
)
parser.add_argument(
    "--strength",
    type=float,
    default=0.75,
    help="img2img strength (0-1). 1.0 keeps nothing of the input image.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="random seed",
)
parser.add_argument(
    "--refiner",
    action="store_true",
    help="run the SDXL refiner as a second stage (ensemble of experts).",
)
parser.add_argument(
    "--negative_prompt",
    metavar="TEXT",
    type=str,
    default="",
    help="the negative prompt (only used by the refiner stage).",
)
parser.add_argument(
    "--refiner_strength",
    type=float,
    default=0.15,
    help="fraction of the schedule handled by the refiner.",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
quant_group = parser.add_mutually_exclusive_group()
quant_group.add_argument(
    "--fp16",
    action="store_true",
    help="use fp16 models for the unet and text encoders.",
)
quant_group.add_argument(
    "--int8",
    action="store_true",
    help="use int8 (MatMulNBits) models for the unet and text encoders.",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)

# fp16/int8 は unet とテキストエンコーダのみ。VAE は fp16 でオーバーフロー
# する既知の問題があるため常に fp32 を使う。
if args.fp16 or args.int8:
    _suffix = "_fp16" if args.fp16 else "_int8"
    WEIGHT_UNET_PATH = f"sdxl_unet{_suffix}.onnx"
    WEIGHT_UNET_PB_PATH = f"sdxl_unet{_suffix}_weights.pb"
    MODEL_UNET_PATH = f"sdxl_unet{_suffix}.onnx.prototxt"
    WEIGHT_CLIP_L_PATH = f"sdxl_text_encoder_clip_l{_suffix}.onnx"
    WEIGHT_CLIP_L_PB_PATH = f"sdxl_text_encoder_clip_l{_suffix}_weights.pb"
    MODEL_CLIP_L_PATH = f"sdxl_text_encoder_clip_l{_suffix}.onnx.prototxt"
    WEIGHT_OPEN_CLIP_PATH = f"sdxl_text_encoder_open_clip_bigg{_suffix}.onnx"
    WEIGHT_OPEN_CLIP_PB_PATH = f"sdxl_text_encoder_open_clip_bigg{_suffix}_weights.pb"
    MODEL_OPEN_CLIP_PATH = f"sdxl_text_encoder_open_clip_bigg{_suffix}.onnx.prototxt"
    WEIGHT_REFINER_UNET_PATH = f"sdxl_refiner_unet{_suffix}.onnx"
    WEIGHT_REFINER_UNET_PB_PATH = f"sdxl_refiner_unet{_suffix}_weights.pb"
    MODEL_REFINER_UNET_PATH = f"sdxl_refiner_unet{_suffix}.onnx.prototxt"


# ======================
# Secondaty Functions
# ======================


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = np.exp(-math.log(max_period) * np.arange(0, half, dtype=np.float32) / half)
    args = timesteps[:, None].astype(np.float32) * freqs[None]
    embedding = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    if dim % 2:
        embedding = np.concatenate(
            [embedding, np.zeros_like(embedding[:, :1])], axis=-1
        )
    return embedding


def embed_nd(values, batch_size, outdim=256):
    """各スカラーを個別に埋め込んで連結する。

    values は (crop_top, crop_left) のような 1 次元の並び。各要素を
    timestep_embedding で outdim 次元に埋め込み、要素方向に連結する。
    """
    x = np.array(values, dtype=np.float32)
    x = np.tile(x[None], (batch_size, 1))
    b, dims = x.shape
    emb = timestep_embedding(x.reshape(b * dims), outdim)
    emb = emb.reshape(b, dims * outdim)
    return emb


def load_input_image(image_path):
    """入力画像を [-1, 1] の NCHW float32 にする。

    解像度は各辺を 64 の倍数へ切り捨てたサイズになる。resize は PIL の
    デフォルトフィルタで、RGB 変換は resize の後に行う。
    """
    image = Image.open(image_path)
    w, h = image.size
    width, height = w - w % 64, h - h % 64
    image = image.resize((width, height))
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    return image.astype(np.float32) / 127.5 - 1.0


class LegacyDDPMDiscretization:
    """DDPM の alphas_cumprod から sigma スケジュールを作る。"""

    def __init__(self, linear_start=0.00085, linear_end=0.0120, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        betas = (
            np.linspace(
                linear_start**0.5, linear_end**0.5, num_timesteps, dtype=np.float64
            )
            ** 2
        )
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

    def get_sigmas(self, n):
        if n < self.num_timesteps:
            timesteps = np.linspace(
                self.num_timesteps - 1, 0, n, endpoint=False
            ).astype(int)[::-1]
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        return np.flip(sigmas)

    def __call__(self, n, do_append_zero=True, flip=False):
        sigmas = self.get_sigmas(n)
        if do_append_zero:
            sigmas = np.concatenate([sigmas, [0.0]])
        if flip:
            sigmas = np.flip(sigmas)
        return sigmas.astype(np.float32)


def img2img_sigmas(discretization, num_steps, strength):
    """低ノイズ側の sigma だけを strength の割合ぶん残す。"""
    sigmas = discretization(num_steps)
    ascending = sigmas[::-1]
    ascending = ascending[: max(int(strength * len(ascending)), 1)]
    return ascending[::-1].astype(np.float32)


def txt2noisy_sigmas(discretization, num_steps, strength, original_steps):
    """strength の境目で打ち切り、低ノイズ側を残さない。"""
    sigmas = discretization(num_steps)
    ascending = sigmas[::-1]
    steps = original_steps + 1
    prune_index = max(min(int(strength * steps) - 1, steps - 1), 0)
    ascending = ascending[prune_index:]
    return ascending[::-1].astype(np.float32)


# ======================
# Pipeline
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


class SDXLPipeline:
    """base / refiner 共通の denoiser・CFG 合成・Euler サンプラーと
    VAE デコードを持つ基底クラス。

    sigma スケジュール・denoiser の preconditioning・CFG 合成は ONNX に
    含まれないため、ここで実装する。
    """

    def __init__(self, unet, vae_decoder, use_onnx=False):
        self.unet = unet
        self.vae_decoder = vae_decoder
        self.use_onnx = use_onnx

        self.vae_scale_factor = 8

        # sigma <-> index の量子化に使う 1000 点の固定テーブル。
        # discretization(1000, do_append_zero=False, flip=True) で昇順(idx0=最小)。
        self.discretization = LegacyDDPMDiscretization()
        self.discrete_sigmas = self.discretization(
            1000, do_append_zero=False, flip=True
        )

    def _run(self, net, inputs, onnx_inputs):
        if not self.use_onnx:
            return net.run(inputs)
        return net.run(None, onnx_inputs)

    def _tokenize(self, tokenizer, prompt):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        return text_inputs.input_ids.astype(np.int64)

    def denoise(self, x, sigma, c, uc, guidance_scale):
        # uncond/cond をバッチ方向に連結して 1 回で推論する
        x_in = np.concatenate([x, x], axis=0)
        context = np.concatenate([uc["crossattn"], c["crossattn"]], axis=0)
        vector = np.concatenate([uc["vector"], c["vector"]], axis=0)

        # sigma を離散テーブルへ量子化してから preconditioning を掛ける
        idx = int(np.argmin(np.abs(sigma - self.discrete_sigmas)))
        sigma_q = self.discrete_sigmas[idx]
        c_skip = 1.0
        c_out = -sigma_q
        c_in = 1.0 / (sigma_q**2 + 1.0) ** 0.5

        timesteps = np.full((x_in.shape[0],), idx, dtype=np.float32)
        sample = (x_in * c_in).astype(np.float32)
        context = context.astype(np.float32)
        vector = vector.astype(np.float32)
        output = self._run(
            self.unet,
            [sample, timesteps, context, vector],
            {"x": sample, "timesteps": timesteps, "context": context, "y": vector},
        )
        eps = output[0]
        denoised = eps * c_out + x_in * c_skip

        # CFG 合成: x_u + scale * (x_c - x_u)
        denoised_uc, denoised_c = np.split(denoised, 2, axis=0)
        denoised = denoised_uc + guidance_scale * (denoised_c - denoised_uc)
        return denoised

    def sample(self, x_init, sigmas, c, uc, guidance_scale):
        # 初期表現を sqrt(1+sigma0^2) 倍してからループに入る
        x = x_init * float(np.sqrt(1.0 + sigmas[0] ** 2.0))

        # Euler ステップ (s_churn=0 なので途中でノイズは足さない)
        for i in tqdm(range(len(sigmas) - 1)):
            sigma = sigmas[i]
            next_sigma = sigmas[i + 1]

            denoised = self.denoise(x, sigma, c, uc, guidance_scale)

            # to_d + euler step
            d = (x - denoised) / sigma
            dt = next_sigma - sigma
            x = x + dt * d

        return x

    def decode(self, latent):
        # VAE decode (ONNX 側で 1/scale_factor 除算を含む)
        latent = latent.astype(np.float32)
        output = self._run(self.vae_decoder, [latent], {"latent": latent})
        image = output[0]

        image = np.clip((image + 1.0) / 2.0, 0, 1)
        image = image.transpose((0, 2, 3, 1))
        return image


class StableDiffusionXL(SDXLPipeline):
    """SDXL-base-1.0 txt2img。conditioner は CLIP-L + OpenCLIP bigG。"""

    def __init__(
        self,
        clip_l,
        open_clip_bigg,
        unet,
        vae_decoder,
        tokenizer,
        tokenizer_2,
        vae_encoder=None,
        use_onnx=False,
    ):
        super().__init__(unet, vae_decoder, use_onnx)
        self.clip_l = clip_l
        self.open_clip_bigg = open_clip_bigg
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.vae_encoder = vae_encoder

    def encode_prompt(self, prompt):
        # CLIP ViT-L/14 : penultimate hidden state (768)
        input_ids = self._tokenize(self.tokenizer, prompt)
        output = self._run(self.clip_l, [input_ids], {"input_ids": input_ids})
        clip_l_hidden = output[0]

        # OpenCLIP ViT-bigG/14 : penultimate hidden state (1280) + pooled (1280)
        input_ids_2 = self._tokenize(self.tokenizer_2, prompt)
        output = self._run(
            self.open_clip_bigg, [input_ids_2], {"input_ids": input_ids_2}
        )
        open_clip_hidden, pooled = output

        # crossattn: CLIP-L(768) と OpenCLIP(1280) を特徴次元で連結 -> 2048
        context = np.concatenate([clip_l_hidden, open_clip_hidden], axis=-1)
        return context, pooled

    def build_conditioning(self, prompt, batch_size, height, width):
        context, pooled = self.encode_prompt(prompt)

        # 追加条件。orig/target は (height, width)。
        emb_orig = embed_nd([height, width], batch_size)
        emb_crop = embed_nd([0, 0], batch_size)  # crop_coords_top_left
        emb_target = embed_nd([height, width], batch_size)

        # vector(y): pooled(1280) + orig(512) + crop(512) + target(512) = 2816
        vector = np.concatenate([pooled, emb_orig, emb_crop, emb_target], axis=-1)

        # uncond 側は txt 由来の埋め込み (crossattn 全体と pooled 部分) を 0 に
        # する。サイズ埋め込みは cond と同一。base は negative_prompt を使わない。
        context_uc = np.zeros_like(context)
        pooled_uc = np.zeros_like(pooled)
        vector_uc = np.concatenate([pooled_uc, emb_orig, emb_crop, emb_target], axis=-1)

        c = {"crossattn": context, "vector": vector}
        uc = {"crossattn": context_uc, "vector": vector_uc}
        return c, uc

    def txt2img(
        self,
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=5.0,
        sigmas=None,
    ):
        batch_size = 1

        c, uc = self.build_conditioning(prompt, batch_size, height, width)

        if sigmas is None:
            sigmas = self.discretization(num_inference_steps)

        shape = (
            batch_size,
            4,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        x_init = np.random.randn(*shape).astype(np.float32)

        return self.sample(x_init, sigmas, c, uc, guidance_scale)

    def encode_image(self, image):
        # ONNX は scale_factor 適用済みの mean/std を返すので、再パラメータ化
        # (z = mean + std * randn) だけをここで行う。
        image = image.astype(np.float32)
        output = self._run(self.vae_encoder, [image], {"pixel": image})
        mean, std = output
        return mean + std * np.random.randn(*mean.shape).astype(np.float32)

    def img2img(
        self,
        image,
        prompt,
        num_inference_steps=50,
        guidance_scale=5.0,
        strength=0.75,
    ):
        batch_size = image.shape[0]
        # 解像度は入力画像 (64 の倍数へ切り捨て済み) をそのまま使う
        height, width = image.shape[2], image.shape[3]

        c, uc = self.build_conditioning(prompt, batch_size, height, width)

        z = self.encode_image(image)

        # 低ノイズ側の sigma だけを使う
        sigmas = img2img_sigmas(self.discretization, num_inference_steps, strength)

        # 入力潜在に sigma[0] のノイズを乗せてから denoise する
        noise = np.random.randn(*z.shape).astype(np.float32)
        noised_z = z + noise * sigmas[0]
        x_init = noised_z / float(np.sqrt(1.0 + sigmas[0] ** 2.0))

        return self.sample(x_init, sigmas, c, uc, guidance_scale)


class StableDiffusionXLRefiner(SDXLPipeline):
    """SDXL-refiner-1.0。conditioner は OpenCLIP bigG のみ + aesthetic_score。

    base の潜在を受け取り、低ノイズ側だけを denoise する
    (ensemble of experts)。negative_prompt を使用する。
    """

    def __init__(
        self,
        open_clip_bigg,
        unet,
        vae_decoder,
        tokenizer_2,
        use_onnx=False,
    ):
        super().__init__(unet, vae_decoder, use_onnx)
        self.open_clip_bigg = open_clip_bigg
        self.tokenizer_2 = tokenizer_2

    def encode_prompt(self, prompt):
        # OpenCLIP ViT-bigG/14 : penultimate hidden state (1280) + pooled (1280)
        input_ids = self._tokenize(self.tokenizer_2, prompt)
        output = self._run(self.open_clip_bigg, [input_ids], {"input_ids": input_ids})
        context, pooled = output
        return context, pooled

    def build_conditioning(
        self,
        prompt,
        negative_prompt,
        batch_size,
        height,
        width,
        aesthetic_score,
        negative_aesthetic_score,
    ):
        context, pooled = self.encode_prompt(prompt)
        # refiner は negative_prompt を実際にエンコードする (ゼロ埋めしない)。
        context_uc, pooled_uc = self.encode_prompt(negative_prompt)

        # 追加条件。base の target_size の代わりに aesthetic_score(スカラ)を使う。
        emb_orig = embed_nd([height, width], batch_size)
        emb_crop = embed_nd([0, 0], batch_size)  # crop_coords_top_left
        emb_aesthetic = embed_nd([aesthetic_score], batch_size)
        emb_aesthetic_uc = embed_nd([negative_aesthetic_score], batch_size)

        # vector(y): pooled(1280) + orig(512) + crop(512) + aesthetic(256) = 2560
        vector = np.concatenate([pooled, emb_orig, emb_crop, emb_aesthetic], axis=-1)
        vector_uc = np.concatenate(
            [pooled_uc, emb_orig, emb_crop, emb_aesthetic_uc], axis=-1
        )

        c = {"crossattn": context, "vector": vector}
        uc = {"crossattn": context_uc, "vector": vector_uc}
        return c, uc

    def refine(
        self,
        latent,
        prompt,
        negative_prompt="",
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=5.0,
        strength=0.15,
        aesthetic_score=6.0,
        negative_aesthetic_score=2.5,
    ):
        batch_size = latent.shape[0]

        c, uc = self.build_conditioning(
            prompt,
            negative_prompt,
            batch_size,
            height,
            width,
            aesthetic_score,
            negative_aesthetic_score,
        )

        # 低ノイズ側の sigma だけを使う
        sigmas = img2img_sigmas(self.discretization, num_inference_steps, strength)

        # base 潜在はすでに sigma[0] のノイズを持っているのでノイズは足さず、
        # サンプラーの sqrt 倍を打ち消すよう事前に割っておく。
        x_init = latent / float(np.sqrt(1.0 + sigmas[0] ** 2.0))

        return self.sample(x_init, sigmas, c, uc, guidance_scale)


# ======================
# Main functions
# ======================


def load_tokenizers():
    if args.disable_ailia_tokenizer:
        import transformers

        tokenizer = transformers.CLIPTokenizer.from_pretrained("./tokenizer")
        tokenizer_2 = transformers.CLIPTokenizer.from_pretrained("./tokenizer_2")
    else:
        from ailia_tokenizer import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained()
        tokenizer.model_max_length = 77
        tokenizer_2 = CLIPTokenizer.from_pretrained()
        tokenizer_2.add_special_tokens({"pad_token": "!"})
        tokenizer_2.model_max_length = 77
    return tokenizer, tokenizer_2


def recognize_from_text(models):
    prompt = args.input if isinstance(args.input, str) else args.input[0]
    height = args.height
    width = args.width
    steps = args.steps
    guidance_scale = args.guidance_scale
    strength = args.refiner_strength

    tokenizer, tokenizer_2 = load_tokenizers()

    logger.info("prompt: %s" % prompt)
    logger.info("Start inference...")

    # ---- base stage ----
    base = StableDiffusionXL(
        clip_l=models["clip_l"],
        open_clip_bigg=models["open_clip_bigg"],
        unet=models["unet"],
        vae_decoder=models["vae_decoder"],
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        vae_encoder=models.get("vae_encoder"),
        use_onnx=args.onnx,
    )

    if args.input_image:
        init_image = load_input_image(args.input_image)
        latent = base.img2img(init_image, prompt, steps, guidance_scale, args.strength)
        image = base.decode(latent)
    elif not args.refiner:
        latent = base.txt2img(prompt, height, width, steps, guidance_scale)
        image = base.decode(latent)
    else:
        # base は refiner に引き継ぐぶんだけ手前で打ち切る
        base_sigmas = txt2noisy_sigmas(base.discretization, steps, strength, steps)
        latent = base.txt2img(
            prompt, height, width, steps, guidance_scale, sigmas=base_sigmas
        )

        # 両 UNet の同時常駐を避けるため base UNet を解放する
        # (open_clip_bigg / vae_decoder は refiner でも使い回す)
        models["clip_l"].unload()
        models["unet"].unload()

        # ---- refiner stage ----
        refiner = StableDiffusionXLRefiner(
            open_clip_bigg=models["open_clip_bigg"],
            unet=models["refiner_unet"],
            vae_decoder=models["vae_decoder"],
            tokenizer_2=tokenizer_2,
            use_onnx=args.onnx,
        )

        latent = refiner.refine(
            latent,
            prompt,
            negative_prompt=args.negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
        )
        image = refiner.decode(latent)

    image = (image[0] * 255).astype(np.uint8)
    image = image[:, :, ::-1]  # RGB->BGR

    img_savepath = get_savepath(args.savepath, "", ext=".png")
    logger.info(f"saved at : {img_savepath}")
    cv2.imwrite(img_savepath, image)

    logger.info("Script finished successfully.")


def main():
    check_and_download_models(WEIGHT_UNET_PATH, MODEL_UNET_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_CLIP_L_PATH, MODEL_CLIP_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_OPEN_CLIP_PATH, MODEL_OPEN_CLIP_PATH, REMOTE_PATH)
    check_and_download_models(
        WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH, REMOTE_PATH
    )
    check_and_download_file(WEIGHT_UNET_PB_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_CLIP_L_PB_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_OPEN_CLIP_PB_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_VAE_DECODER_PB_PATH, REMOTE_PATH)
    if args.input_image:
        check_and_download_models(
            WEIGHT_VAE_ENCODER_PATH, MODEL_VAE_ENCODER_PATH, REMOTE_PATH
        )
        check_and_download_file(WEIGHT_VAE_ENCODER_PB_PATH, REMOTE_PATH)
    if args.refiner:
        check_and_download_models(
            WEIGHT_REFINER_UNET_PATH, MODEL_REFINER_UNET_PATH, REMOTE_PATH
        )
        check_and_download_file(WEIGHT_REFINER_UNET_PB_PATH, REMOTE_PATH)

    seed = args.seed
    if seed is not None:
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

    def load_net(model_path, weight_path):
        if not args.onnx:
            return ailia.Net(
                model_path, weight_path, env_id=env_id, memory_mode=memory_mode
            )
        else:
            import onnxruntime

            return onnxruntime.InferenceSession(weight_path, providers=providers)

    models = {
        "clip_l": LazyModel(
            lambda: load_net(MODEL_CLIP_L_PATH, WEIGHT_CLIP_L_PATH), "clip_l"
        ),
        "open_clip_bigg": LazyModel(
            lambda: load_net(MODEL_OPEN_CLIP_PATH, WEIGHT_OPEN_CLIP_PATH),
            "open_clip_bigg",
        ),
        "unet": LazyModel(lambda: load_net(MODEL_UNET_PATH, WEIGHT_UNET_PATH), "unet"),
        "vae_decoder": LazyModel(
            lambda: load_net(MODEL_VAE_DECODER_PATH, WEIGHT_VAE_DECODER_PATH),
            "vae_decoder",
        ),
    }
    if args.input_image:
        models["vae_encoder"] = LazyModel(
            lambda: load_net(MODEL_VAE_ENCODER_PATH, WEIGHT_VAE_ENCODER_PATH),
            "vae_encoder",
        )
    if args.refiner:
        models["refiner_unet"] = LazyModel(
            lambda: load_net(MODEL_REFINER_UNET_PATH, WEIGHT_REFINER_UNET_PATH),
            "refiner_unet",
        )

    recognize_from_text(models)


if __name__ == "__main__":
    main()
