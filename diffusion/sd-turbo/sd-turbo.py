
import sys

import numpy as np
import cv2

import ailia
from logging import getLogger

sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file
from detector_utils import load_image
from df.pipelines.stable_diffusion import StableDiffusion
from df.pipelines.stable_diffusion_img2img import StableDiffusionimg2Img

from df.schedulers.euler_discrete_scheduler import EulerDiscreteScheduler

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_UNET_PATH = "unet.onnx"
MODEL_UNET_PATH = "unet.onnx.prototxt"
PB_UNET_PATH = "unet.pb"
WEIGHT_TEXT_ENCODER_PATH = "text_encoder.onnx"
MODEL_TEXT_ENCODER_PATH = "text_encoder.onnx.prototxt"
WEIGHT_VAE_ENCODER_PATH = "vae_encoder.onnx"
MODEL_VAE_ENCODER_PATH = "vae_encoder.onnx.prototxt"
WEIGHT_VAE_DECODER_PATH = "vae_decoder.onnx"
MODEL_VAE_DECODER_PATH = "vae_decoder.onnx.prototxt"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/sd-turbo/"

IMAGE_SIZE = 512
SAVE_IMAGE_PATH = "output.png"

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser("SD-Turbo", None, SAVE_IMAGE_PATH)
parser.add_argument(
    "-i",
    "--input",
    metavar="TEXT",
    type=str,
    default="little cute gremlin wearing a jacket, cinematic, vivid colors, intricate masterpiece, golden ratio, highly detailed",
    help="the prompt to render",
)
parser.add_argument(
    "--init_image",
    metavar="IMAGE_PATH",
    type=str,
    default=None,
    help="input image file path.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="random seed",
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


def main():
    env_id = args.env_id
    image_path = args.init_image
    check_and_download_models(WEIGHT_UNET_PATH, MODEL_UNET_PATH, REMOTE_PATH)
    check_and_download_file(PB_UNET_PATH, REMOTE_PATH)
    check_and_download_models(
        WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_VAE_ENCODER_PATH, MODEL_VAE_ENCODER_PATH, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH, REMOTE_PATH
    )

    seed = args.seed
    if seed is not None:
        np.random.seed(seed)

    """
    Parse input
    """
    prompt = args.input
    if not args.onnx:
        unet = ailia.Net(
            MODEL_UNET_PATH, WEIGHT_UNET_PATH, env_id=env_id, memory_mode=args.memory_mode
        )
        text_encoder = ailia.Net(
            MODEL_TEXT_ENCODER_PATH,
            WEIGHT_TEXT_ENCODER_PATH,
            env_id=env_id,
            memory_mode=args.memory_mode,
        )
        vae_encoder = ailia.Net(
            MODEL_VAE_ENCODER_PATH,
            WEIGHT_VAE_ENCODER_PATH,
            env_id=env_id,
            memory_mode=args.memory_mode,
        )
        vae_decoder = ailia.Net(
            MODEL_VAE_DECODER_PATH,
            WEIGHT_VAE_DECODER_PATH,
            env_id=env_id,
            memory_mode=args.memory_mode,
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
        vae_encoder = onnxruntime.InferenceSession(
            WEIGHT_VAE_ENCODER_PATH, providers=providers
        )
        vae_decoder = onnxruntime.InferenceSession(
            WEIGHT_VAE_DECODER_PATH, providers=providers
        )

    init_image = None
    if args.init_image is not None:
        init_image = load_image(image_path)
        init_image = cv2.cvtColor(init_image, cv2.COLOR_BGRA2RGB)
        init_image = cv2.resize(
            init_image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
        )

    if args.disable_ailia_tokenizer:
        import transformers
        tokenizer = transformers.CLIPTokenizer.from_pretrained("./tokenizer")
    else:
        from ailia_tokenizer import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained()
        tokenizer.model_max_length = 77
        tokenizer.add_special_tokens({'pad_token': '!'})
    
    scheduler = None
    scheduler = EulerDiscreteScheduler.from_config(
        {
            "num_train_timesteps": 1000,
            "timestep_spacing": "trailing",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "trained_betas": None,
            "steps_offset": 1,
            "prediction_type": "epsilon",
        }
    )

    if init_image is None:
        pipeline = StableDiffusion(
            vae_decoder=vae_decoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            use_onnx=args.onnx,
        )
        image = pipeline.forward(
            prompt=prompt,
            num_inference_steps=2,
            guidance_scale=0.0,
        )
    else:
        logger.warning("this mode not work well.")
        pipeline = StableDiffusionimg2Img(
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            use_onnx=args.onnx,
        )
        image = pipeline.forward(
            prompt=prompt,
            base_image=init_image,
            num_inference_steps=2,
            strength=0.5,
            guidance_scale=0.0,
        )
    image = (image[0] * 255).astype(np.uint8)
    image = image[:, :, ::-1]  # RGB->BGR

    img_savepath = get_savepath(args.savepath, "", ext=".png")
    logger.info(f"saved at : {img_savepath}")
    cv2.imwrite(img_savepath, image)

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
