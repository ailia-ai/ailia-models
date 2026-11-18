import sys
import time
from logging import getLogger

use_torch = True

import numpy as np
import torch
from transformers import BertTokenizer
from scipy.io.wavfile import write as write_wav
from encodec import EncodecModel

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from math_utils import softmax  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

import generation_utils
from generation_utils import (
    SAMPLE_RATE,
    generate_text_semantic,
    generate_coarse, generate_fine,
    codec_decode,
)

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_TEXT_PATH = 'text.onnx'
MODEL_TEXT_PATH = 'text.onnx.prototxt'
WEIGHT_COARSE_PATH = 'coarse.onnx'
MODEL_COARSE_PATH = 'coarse.onnx.prototxt'
WEIGHT_FINE_PATH = 'fine.onnx'
MODEL_FINE_PATH = 'fine.onnx.prototxt'
MODEL_FINE_PT_PATH = 'net_fine.pth'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/bark/'

SAVE_WAV_PATH = 'output.wav'

SEMANTIC_RATE_HZ = 49.9
SEMANTIC_VOCAB_SIZE = 10_000

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Bark', None, SAVE_WAV_PATH
)
parser.add_argument(
    "-i", "--input", metavar="TEXT", type=str,
    default="""
    Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
    But I also have other interests such as playing tic tac toe.
    """,
    help="The text to speach"
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser, check_input_type=False)


# ======================
# Main functions
# ======================

def semantic_to_waveform(
        models,
        semantic_tokens: np.ndarray,
        temp: float = 0.7,
        silent: bool = False,
        output_full: bool = False):
    """Generate audio array from semantic input.

    Args:
        semantic_tokens: semantic token output from `text_to_semantic`
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    coarse_tokens = generate_coarse(
        models,
        semantic_tokens,
        temp=temp,
        silent=silent,
    )
    fine_tokens = generate_fine(
        models,
        coarse_tokens,
        temp=0.5,
        silent=silent,
    )
    audio_arr = codec_decode(models, fine_tokens)

    if output_full:
        full_generation = {
            "semantic_prompt": semantic_tokens,
            "coarse_prompt": coarse_tokens,
            "fine_prompt": fine_tokens,
        }
        return full_generation, audio_arr
    return audio_arr


def generate_audio(
        models,
        text: str,
        text_temp: float = 0.7,
        waveform_temp: float = 0.7,
        silent: bool = False,
        output_full: bool = False):
    """Generate audio array from input text.

    Args:
        text: text to be turned into audio
        text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """

    x_semantic = generate_text_semantic(
        models,
        text,
        temp=text_temp,
        silent=silent,
    )
    out = semantic_to_waveform(
        models,
        x_semantic,
        temp=waveform_temp,
        silent=silent,
        output_full=output_full,
    )

    if output_full:
        full_generation, audio_arr = out
        return full_generation, audio_arr
    else:
        audio_arr = out

    return audio_arr


def recognize_from_text(models):
    text_prompt = args.input if isinstance(args.input, str) else args.input[0]
    logger.info("prompt: %s" % text_prompt)

    logger.info('Start inference...')

    generated_audio = generate_audio(models, text_prompt)

    # plot result
    savepath = get_savepath(args.savepath, None, ext='.wav')
    logger.info(f'saved at : {savepath}')
    write_wav(savepath, SAMPLE_RATE, generated_audio)

    logger.info('Script finished successfully.')


def main():
    check_and_download_models(WEIGHT_TEXT_PATH, MODEL_TEXT_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_COARSE_PATH, MODEL_COARSE_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_FINE_PATH, MODEL_FINE_PATH, REMOTE_PATH)
    check_and_download_file(MODEL_FINE_PT_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_TEXT_PATH, WEIGHT_TEXT_PATH, env_id=env_id)
        net_coarse = ailia.Net(MODEL_COARSE_PATH, WEIGHT_COARSE_PATH, env_id=env_id)
        if not use_torch:
            net_fine = ailia.Net(MODEL_FINE_PATH, WEIGHT_FINE_PATH, env_id=env_id)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        net = onnxruntime.InferenceSession(WEIGHT_TEXT_PATH, providers=providers)
        net_coarse = onnxruntime.InferenceSession(WEIGHT_COARSE_PATH, providers=providers)
        if not use_torch:
            net_fine = onnxruntime.InferenceSession(WEIGHT_FINE_PATH, providers=providers)

        generation_utils.onnx = True

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    if use_torch:
        net_fine = torch.load(MODEL_FINE_PT_PATH, map_location=torch.device('cpu'))

    net_encodec = EncodecModel.encodec_model_24khz()
    net_encodec.set_target_bandwidth(6.0)
    net_encodec.eval()

    if env_id > 0 and torch.cuda.is_available():
        net_encodec = net_encodec.to("cuda")
        if use_torch:
            net_fine = net_fine.to("cuda")

    models = {
        "net": net,
        "tokenizer": tokenizer,
        "coarse": net_coarse,
        "fine": net_fine,
        "codec": net_encodec,
    }

    # generate
    recognize_from_text(models)


if __name__ == '__main__':
    main()
