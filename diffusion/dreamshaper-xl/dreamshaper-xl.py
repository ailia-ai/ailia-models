import sys
from logging import getLogger

import ailia
import cv2
import numpy as np

# import original modules
sys.path.append("../../util")
import df  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa
from model_utils import check_and_download_file, check_and_download_models  # noqa

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

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/dreamshaper-xl/"

SAVE_IMAGE_PATH = "output.png"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    "DreamShaper XL 1.0", None, SAVE_IMAGE_PATH, fp16_support=False
)
parser.add_argument(
    "-i",
    "--input",
    metavar="TEXT",
    type=str,
    default="portrait photo of muscular bearded guy in a worn mech suit, light bokeh, "
    "intricate, steel metal, elegant, sharp focus, soft lighting, vibrant colors",
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
    default=25,
    help="number of sampling steps",
)
parser.add_argument(
    "--guidance_scale",
    type=float,
    default=7.0,
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


# ======================
# Main functions
# ======================


def recognize_from_text(pipe):
    prompt = args.input if isinstance(args.input, str) else args.input[0]

    logger.info("prompt: %s" % prompt)

    logger.info("Start inference...")

    image = pipe.forward(
        prompt=prompt,
        height=args.height,
        width=args.width,
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
        text_encoder_2 = ailia.Net(
            MODEL_TEXT_ENCODER_2_PATH,
            WEIGHT_TEXT_ENCODER_2_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
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

        unet = onnxruntime.InferenceSession(WEIGHT_UNET_PATH, providers=providers)
        text_encoder = onnxruntime.InferenceSession(
            WEIGHT_TEXT_ENCODER_PATH, providers=providers
        )
        text_encoder_2 = onnxruntime.InferenceSession(
            WEIGHT_TEXT_ENCODER_2_PATH, providers=providers
        )
        vae_decoder = onnxruntime.InferenceSession(
            WEIGHT_VAE_DECODER_PATH, providers=providers
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
        tokenizer_2.add_special_tokens({"pad_token": "!"})
        tokenizer_2.model_max_length = 77

    scheduler = df.DEISMultistepScheduler.from_config(
        {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "trained_betas": None,
            "solver_order": 2,
            "prediction_type": "epsilon",
            "algorithm_type": "deis",
            "solver_type": "logrho",
            "lower_order_final": True,
            "timestep_spacing": "leading",
            "steps_offset": 1,
        }
    )

    pipe = df.StableDiffusionXL(
        vae_decoder=vae_decoder,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=scheduler,
        use_onnx=args.onnx,
    )

    # generate
    recognize_from_text(pipe)


if __name__ == "__main__":
    main()
