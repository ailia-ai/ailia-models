import sys
from logging import getLogger

import ailia
import cv2
import numpy as np
from PIL import Image

# import original modules
sys.path.append("../../util")
import df  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa
from model_utils import check_and_download_file, check_and_download_models  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_UNET_PATH = "dreamshaper_v8_unet.onnx"
WEIGHT_UNET_PB_PATH = "dreamshaper_v8_unet_weights.pb"
MODEL_UNET_PATH = "dreamshaper_v8_unet.onnx.prototxt"
WEIGHT_TEXT_ENCODER_PATH = "dreamshaper_v8_text_encoder.onnx"
MODEL_TEXT_ENCODER_PATH = "dreamshaper_v8_text_encoder.onnx.prototxt"
WEIGHT_VAE_DECODER_PATH = "dreamshaper_v8_vae_decoder.onnx"
MODEL_VAE_DECODER_PATH = "dreamshaper_v8_vae_decoder.onnx.prototxt"
WEIGHT_VAE_ENCODER_PATH = "dreamshaper_v8_vae_encoder.onnx"
MODEL_VAE_ENCODER_PATH = "dreamshaper_v8_vae_encoder.onnx.prototxt"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/dreamshaper-sd15/"

SAVE_IMAGE_PATH = "output.png"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    "DreamShaper (Stable Diffusion 1.5)", None, SAVE_IMAGE_PATH, fp16_support=False
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
    "--negative_prompt",
    metavar="TEXT",
    type=str,
    default=None,
    help="the prompt not to guide the image generation.",
)
parser.add_argument(
    "--init_image",
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
    "--width",
    type=int,
    default=512,
    help="output image width",
)
parser.add_argument(
    "--height",
    type=int,
    default=512,
    help="output image height",
)
parser.add_argument(
    "--scheduler",
    type=str,
    default="dpm++",
    choices=["dpm++", "pndm"],
    help="dpm++ = DPM++ 2M Karras (DreamShaper 推奨, 少stepで高品質), "
    "pndm = repo 既定 (低次, ~50step 必要)。",
)
parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help="number of sampling steps (未指定なら dpm++=30 / pndm=50)。",
)
parser.add_argument(
    "--guidance_scale",
    type=float,
    default=7.5,
    help="classifier free guidance scale",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="random seed",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)

# sampler ごとの既定ステップ数(PNDM は低次で収束に多くの step が要る)。
if args.steps is None:
    args.steps = {"dpm++": 30, "pndm": 50}[args.scheduler]


# ======================
# Main functions
# ======================


def recognize_from_text(pipe):
    prompt = args.input if isinstance(args.input, str) else args.input[0]
    image_path = args.init_image

    logger.info("prompt: %s" % prompt)

    logger.info("Start inference...")

    if image_path is None:
        image = pipe.forward(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
        )
    else:
        init_image = Image.open(image_path).convert("RGB")
        image = pipe.forward(
            prompt=prompt,
            image=init_image,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
        )

    image = (image[0] * 255).round().astype(np.uint8)
    image = image[:, :, ::-1]  # RGB->BGR

    img_savepath = get_savepath(args.savepath, "", ext=".png")
    logger.info(f"saved at : {img_savepath}")
    cv2.imwrite(img_savepath, image)

    logger.info("Script finished successfully.")


def main():
    init_image = args.init_image

    check_and_download_models(WEIGHT_UNET_PATH, MODEL_UNET_PATH, REMOTE_PATH)
    check_and_download_models(
        WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH, REMOTE_PATH
    )
    check_and_download_file(WEIGHT_UNET_PB_PATH, REMOTE_PATH)
    if init_image:
        check_and_download_models(
            WEIGHT_VAE_ENCODER_PATH, MODEL_VAE_ENCODER_PATH, REMOTE_PATH
        )

    seed = args.seed
    if seed is not None:
        np.random.seed(seed)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        unet = ailia.Net(
            MODEL_UNET_PATH,
            WEIGHT_UNET_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        text_encoder = ailia.Net(
            MODEL_TEXT_ENCODER_PATH,
            WEIGHT_TEXT_ENCODER_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        vae_decoder = ailia.Net(
            MODEL_VAE_DECODER_PATH,
            WEIGHT_VAE_DECODER_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        if init_image:
            vae_encoder = ailia.Net(
                MODEL_VAE_ENCODER_PATH,
                WEIGHT_VAE_ENCODER_PATH,
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

        unet = onnxruntime.InferenceSession(WEIGHT_UNET_PATH, providers=providers)
        text_encoder = onnxruntime.InferenceSession(
            WEIGHT_TEXT_ENCODER_PATH, providers=providers
        )
        vae_decoder = onnxruntime.InferenceSession(
            WEIGHT_VAE_DECODER_PATH, providers=providers
        )
        if init_image:
            vae_encoder = onnxruntime.InferenceSession(
                WEIGHT_VAE_ENCODER_PATH, providers=providers
            )

    if args.disable_ailia_tokenizer:
        import transformers

        tokenizer = transformers.CLIPTokenizer.from_pretrained("./tokenizer")
    else:
        from ailia_tokenizer import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained()
        tokenizer.model_max_length = 77

    # scheduler の選択。既定は DreamShaper 推奨の DPM++ 2M Karras
    # (少step で高品質)。repo 既定は PNDM だが低次のため ~50step 必要で、
    # step を減らし過ぎる(例: 25)と顔などが破綻する。
    # base config は repo の scheduler_config.json(scaled_linear / leading /
    # steps_offset=1 / epsilon)に合わせている。
    if args.scheduler == "dpm++":
        scheduler = df.DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
            timestep_spacing="leading",
            steps_offset=1,
            algorithm_type="dpmsolver++",
            solver_order=2,
            solver_type="midpoint",
            use_karras_sigmas=True,
            final_sigmas_type="zero",
            lower_order_final=True,
        )
    else:
        scheduler = df.PNDMScheduler.from_config(
            {
                "num_train_timesteps": 1000,
                "beta_start": 0.00085,
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
                "trained_betas": None,
                "skip_prk_steps": True,
                "set_alpha_to_one": False,
                "prediction_type": "epsilon",
                "timestep_spacing": "leading",
                "steps_offset": 1,
            }
        )

    params = dict(
        vae_decoder=vae_decoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        use_onnx=args.onnx,
    )
    if init_image:
        params["vae_encoder"] = vae_encoder
        pipe = df.StableDiffusionImg2Img(**params)
    else:
        pipe = df.StableDiffusion(**params)

    # generate
    recognize_from_text(pipe)


if __name__ == "__main__":
    main()
