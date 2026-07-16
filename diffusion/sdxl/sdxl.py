import math
import sys
from logging import getLogger

import ailia
import cv2
import numpy as np
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
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Secondaty Functions
# ======================


class LegacyDDPMDiscretization:
    """sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization 相当。"""

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


# ======================
# Main pipeline
# ======================


class StableDiffusionXL:
    """SDXL-base-1.0 txt2img。
    """

    def __init__(
        self,
        clip_l,
        open_clip_bigg,
        unet,
        vae_decoder,
        use_onnx=False,
    ):
        self.clip_l = clip_l
        self.open_clip_bigg = open_clip_bigg
        self.unet = unet
        self.vae_decoder = vae_decoder
        self.use_onnx = use_onnx

        self.vae_scale_factor = 8

        discretization = LegacyDDPMDiscretization()
        self.discrete_sigmas = discretization(1000, do_append_zero=False, flip=True)

    def sigma_to_idx(self, sigma):
        return int(np.argmin(np.abs(sigma - self.discrete_sigmas)))

    def denoise(self, x, sigma, c, uc, guidance_scale):
        # VanillaCFG.prepare_inputs: uncond/cond をバッチ方向に連結して 1 回で推論
        x_in = np.concatenate([x, x], axis=0)
        context = np.concatenate([uc["crossattn"], c["crossattn"]], axis=0)
        vector = np.concatenate([uc["vector"], c["vector"]], axis=0)

        # DiscreteDenoiser: sigma を離散テーブルへ量子化してから preconditioning
        idx = self.sigma_to_idx(sigma)
        sigma_q = self.discrete_sigmas[idx]
        c_skip = 1.0
        c_out = -sigma_q
        c_in = 1.0 / (sigma_q**2 + 1.0) ** 0.5

        timesteps = np.full((x_in.shape[0],), idx, dtype=np.float32)
        sample = (x_in * c_in).astype(np.float32)
        output = self._run(
            self.unet,
            [sample, timesteps, context.astype(np.float32), vector.astype(np.float32)],
            {
                "x": sample,
                "timesteps": timesteps,
                "context": context.astype(np.float32),
                "y": vector.astype(np.float32),
            },
        )
        eps = output[0]
        denoised = eps * c_out + x_in * c_skip

        # VanillaCFG: x_u + scale * (x_c - x_u)
        denoised_uc, denoised_c = np.split(denoised, 2, axis=0)
        denoised = denoised_uc + guidance_scale * (denoised_c - denoised_uc)
        return denoised

    def forward(
        self,
        prompt,
        negative_prompt="",
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=5.0,
    ):
        batch_size = 1

        # EulerEDMSampler (s_churn=0 -> gamma=0, 追加ノイズなし)
        for i in tqdm(range(num_sigmas - 1)):
            sigma = sigmas[i]
            next_sigma = sigmas[i + 1]

            denoised = self.denoise(x, sigma, c, uc, guidance_scale)

            # to_d + euler step
            d = (x - denoised) / sigma
            dt = next_sigma - sigma
            x = x + dt * d

# ======================
# Main functions
# ======================


def recognize_from_text(pipe):
    prompt = args.input if isinstance(args.input, str) else args.input[0]

    logger.info("prompt: %s" % prompt)
    logger.info("Start inference...")

    image = pipe.forward(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )

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

    env_id = args.env_id

    # initialize
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        clip_l = ailia.Net(
            MODEL_CLIP_L_PATH,
            WEIGHT_CLIP_L_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        open_clip_bigg = ailia.Net(
            MODEL_OPEN_CLIP_PATH,
            WEIGHT_OPEN_CLIP_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        unet = ailia.Net(
            MODEL_UNET_PATH, WEIGHT_UNET_PATH, env_id=env_id, memory_mode=memory_mode
        )
        vae_decoder = ailia.Net(
            MODEL_VAE_DECODER_PATH,
            WEIGHT_VAE_DECODER_PATH,
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

        clip_l = onnxruntime.InferenceSession(WEIGHT_CLIP_L_PATH, providers=providers)
        open_clip_bigg = onnxruntime.InferenceSession(
            WEIGHT_OPEN_CLIP_PATH, providers=providers
        )
        unet = onnxruntime.InferenceSession(WEIGHT_UNET_PATH, providers=providers)
        vae_decoder = onnxruntime.InferenceSession(
            WEIGHT_VAE_DECODER_PATH, providers=providers
        )

    pipe = StableDiffusionXL(
        clip_l=clip_l,
        open_clip_bigg=open_clip_bigg,
        unet=unet,
        vae_decoder=vae_decoder,
        use_onnx=args.onnx,
    )

    # generate
    recognize_from_text(pipe)


if __name__ == "__main__":
    main()
