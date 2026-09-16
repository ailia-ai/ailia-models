import sys

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa
from detector_utils import load_image  # noqa

# logger
from logging import getLogger  # noqa

import df

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_UNET_PATH = "unet.onnx"
WEIGHT_UNET_PB_PATH = "unet_weights.pb"
MODEL_UNET_PATH = "unet.onnx.prototxt"
WEIGHT_TEXT_ENCODER_PATH = "text_encoder.onnx"
MODEL_TEXT_ENCODER_PATH = "text_encoder.onnx.prototxt"
WEIGHT_TEXT_ENCODER_2_PATH = "text_encoder_2.onnx"
WEIGHT_TEXT_ENCODER_2_PB_PATH = "text_encoder_2_weights.pb"
MODEL_TEXT_ENCODER_2_PATH = "text_encoder_2.onnx.prototxt"
WEIGHT_VAE_DECODER_PATH = "vae_decoder.onnx"
MODEL_VAE_DECODER_PATH = "vae_decoder.onnx.prototxt"
WEIGHT_VAE_ENCODER_PATH = "vae_encoder.onnx"
MODEL_VAE_ENCODER_PATH = "vae_encoder.onnx.prototxt"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/sdxl-turbo/"

IMAGE_SIZE = 512
SAVE_IMAGE_PATH = "output.png"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("SDXL-Turbo", None, SAVE_IMAGE_PATH, fp16_support=False)
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


# ======================
# Main functions
# ======================


def recognize_from_text(pipe):
    prompt = args.input if isinstance(args.input, str) else args.input[0]
    image_path = args.init_image

    logger.info("prompt: %s" % prompt)

    init_image = None
    if image_path is not None:
        init_image = load_image(image_path)
        init_image = cv2.cvtColor(init_image, cv2.COLOR_BGRA2RGB)
        init_image = cv2.resize(
            init_image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
        )

    logger.info("Start inference...")

    if init_image is None:
        image = pipe.forward(
            prompt=prompt,
            num_inference_steps=1,
            guidance_scale=0.0,
        )
    else:
        image = pipe.forward(
            prompt=prompt,
            image=init_image,
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


def main():
    init_image = args.init_image

    check_and_download_models(WEIGHT_UNET_PATH, MODEL_UNET_PATH, REMOTE_PATH)
    check_and_download_models(
        WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_TEXT_ENCODER_2_PATH, MODEL_TEXT_ENCODER_2_PATH, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH, REMOTE_PATH
    )
    check_and_download_file(WEIGHT_UNET_PB_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_TEXT_ENCODER_2_PB_PATH, REMOTE_PATH)
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
        net = ailia.Net(
            MODEL_UNET_PATH, WEIGHT_UNET_PATH, env_id=env_id, memory_mode=args.memory_mode
        )
        text_encoder = ailia.Net(
            MODEL_TEXT_ENCODER_PATH,
            WEIGHT_TEXT_ENCODER_PATH,
            env_id=env_id,
            memory_mode=args.memory_mode,
        )
        text_encoder_2 = ailia.Net(
            MODEL_TEXT_ENCODER_2_PATH,
            WEIGHT_TEXT_ENCODER_2_PATH,
            env_id=env_id,
            memory_mode=args.memory_mode,
        )
        vae_decoder = ailia.Net(
            MODEL_VAE_DECODER_PATH,
            WEIGHT_VAE_DECODER_PATH,
            env_id=env_id,
            memory_mode=args.memory_mode,
        )
        if init_image:
            vae_encoder = ailia.Net(
                MODEL_VAE_ENCODER_PATH,
                WEIGHT_VAE_ENCODER_PATH,
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

        net = onnxruntime.InferenceSession(WEIGHT_UNET_PATH, providers=providers)
        text_encoder = onnxruntime.InferenceSession(
            WEIGHT_TEXT_ENCODER_PATH, providers=providers
        )
        text_encoder_2 = onnxruntime.InferenceSession(
            WEIGHT_TEXT_ENCODER_2_PATH, providers=providers
        )
        vae_decoder = onnxruntime.InferenceSession(
            WEIGHT_VAE_DECODER_PATH, providers=providers
        )
        if init_image:
            vae_encoder = onnxruntime.InferenceSession(
                "vae_encoder.onnx", providers=providers
            )

    if args.disable_ailia_tokenizer:
        import transformers
        tokenizer = transformers.CLIPTokenizer.from_pretrained("./tokenizer")
        tokenizer_2 = transformers.CLIPTokenizer.from_pretrained("./tokenizer_2")
    else:
        from ailia_tokenizer import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained()
        tokenizer.model_max_length = 77
        tokenizer_2 = CLIPTokenizer.from_pretrained()
        tokenizer_2.add_special_tokens({'pad_token': '!'})
        tokenizer_2.model_max_length = 77

    scheduler = df.schedulers.EulerAncestralDiscreteScheduler.from_config(
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

    cls_stable_diffusion = (
        df.StableDiffusionXL if args.init_image is None else df.StableDiffusionXLImg2Img
    )
    params = dict(
        vae_decoder=vae_decoder,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=net,
        scheduler=scheduler,
        use_onnx=args.onnx,
    )
    if args.init_image:
        params["vae_encoder"] = vae_encoder

    pipe = cls_stable_diffusion(**params)

    # generate
    recognize_from_text(pipe)


if __name__ == "__main__":
    main()
