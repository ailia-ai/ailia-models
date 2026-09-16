import os
import sys
import time

import numpy as np
import cv2

import ailia

import random

import df
from df import OnnxRuntimeModel

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, urlretrieve, progress_print  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_UNET_PATH = 'unet.onnx'
WEIGHT_PB_UNET_PATH = 'weights.pb'
MODEL_UNET_PATH = 'unet.onnx.prototxt'
WEIGHT_SAFETY_CHECKER_PATH = 'safety_checker.onnx'
MODEL_SAFETY_CHECKER_PATH = 'safety_checker.onnx.prototxt'
WEIGHT_TEXT_ENCODER_PATH = 'text_encoder.onnx'
MODEL_TEXT_ENCODER_PATH = 'text_encoder.onnx.prototxt'
WEIGHT_VAE_ENCODER_PATH = 'vae_encoder.onnx'
MODEL_VAE_ENCODER_PATH = 'vae_encoder.onnx.prototxt'
WEIGHT_VAE_DECODER_PATH = 'vae_decoder.onnx'
MODEL_VAE_DECODER_PATH = 'vae_decoder.onnx.prototxt'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/anything_v3/'

SAVE_IMAGE_PATH = 'output.png'


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Anything V3', None, SAVE_IMAGE_PATH
)
parser.add_argument(
    "-i", "--input", metavar="TEXT", type=str,
    default="witch",
    help="the prompt to render"
)
parser.add_argument(
    "--steps",
    type=int,
    default=50,
    help="number of inference steps",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="random seed",
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser, check_input_type=False)


# ======================
# Main functions
# ======================
def recognize_from_text(pipe):
    prompt = args.input if isinstance(args.input, str) else args.input[0]
    logger.info("prompt: %s" % prompt)

    logger.info('Start inference...')

    image = pipe(prompt=prompt, num_inference_steps=args.steps).images[0]

    savepath = get_savepath(args.savepath, "", ext='.png')
    image.save(savepath)
    logger.info(f'saved at : {savepath}')

    logger.info('Script finished successfully.')


def main():
    seed = args.seed
    if seed is not None:
        np.random.seed(seed)

    check_and_download_models(WEIGHT_UNET_PATH, MODEL_UNET_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_SAFETY_CHECKER_PATH, MODEL_SAFETY_CHECKER_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VAE_ENCODER_PATH, MODEL_VAE_ENCODER_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH, REMOTE_PATH)

    if not os.path.exists(WEIGHT_PB_UNET_PATH):
        logger.info('Downloading weights.pb...')
        urlretrieve(REMOTE_PATH + WEIGHT_PB_UNET_PATH, WEIGHT_PB_UNET_PATH, progress_print)
    logger.info('weights.pb is prepared!')

    env_id = args.env_id

    # initialize
    unet = OnnxRuntimeModel.from_pretrained(
        "./", "unet.onnx", args.onnx, env_id,
        {'provider': 'CPUExecutionProvider', 'sess_options': None},
        memory_mode=args.memory_mode
    )
    safety_checker = OnnxRuntimeModel.from_pretrained(
        "./", "safety_checker.onnx", args.onnx, env_id,
        {'provider': 'CPUExecutionProvider', 'sess_options': None},
        memory_mode=args.memory_mode
    )
    vae_decoder = OnnxRuntimeModel.from_pretrained(
        "./", "vae_decoder.onnx", args.onnx, env_id,
        {'provider': 'CPUExecutionProvider', 'sess_options': None},
        memory_mode=args.memory_mode
    )
    text_encoder = OnnxRuntimeModel.from_pretrained(
        "./", "text_encoder.onnx", args.onnx, env_id,
        {'provider': 'CPUExecutionProvider', 'sess_options': None},
        memory_mode=args.memory_mode
    )
    vae_encoder = OnnxRuntimeModel.from_pretrained(
        "./", "vae_encoder.onnx", args.onnx, env_id,
        {'provider': 'CPUExecutionProvider', 'sess_options': None},
        memory_mode=args.memory_mode
    )

    pndm_scheduler = df.schedulers.scheduling_pndm.PNDMScheduler.from_pretrained(
        "./scheduler"
    )
    if args.disable_ailia_tokenizer:
        import transformers
        feature_extractor = transformers.CLIPImageProcessor.from_pretrained(
            "./feature_extractor"
        )
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "./tokenizer"
        )
    else:
        from ailia_tokenizer import CLIPTokenizer
        feature_extractor = None
        safety_checker = None
        tokenizer = CLIPTokenizer.from_pretrained()
        tokenizer.model_max_length = 77

    # set pipeline
    pipeline_cls = df.OnnxStableDiffusionPipeline
    
    pipe = pipeline_cls(
        vae_encoder = vae_encoder,
        vae_decoder = vae_decoder,
        text_encoder = text_encoder,
        tokenizer = tokenizer,
        unet = unet,
        scheduler = pndm_scheduler,
        safety_checker = safety_checker,
        feature_extractor = feature_extractor,
        requires_safety_checker = False       
    )

    # generate
    recognize_from_text(pipe)


if __name__ == '__main__':
    main()
