import time
import sys
import argparse

import numpy as np

import ailia  # noqa: E402

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# PARAMETERS
# ======================
SENTENCE = 'The boy was there when the sun rose.'

SAVE_WAV_PATH = 'output.wav'

WEIGHT_PATH_T2M = 'text2mel.onnx'
MODEL_PATH_T2M = 'text2mel.onnx.prototxt'
REMOTE_PATH_T2M = 'https://storage.googleapis.com/ailia-models/pytorch-dc-tts/'

WEIGHT_PATH_SSRM = 'ssrn.onnx'
MODEL_PATH_SSRM = 'ssrn.onnx.prototxt'
REMOTE_PATH_SSRM = 'https://storage.googleapis.com/ailia-models/pytorch-dc-tts/'

MAX_T = 210

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser( 'Efficiently Trainable Text-to-Speech System Based on' +
    'Deep Convolutional Networks with Guided Attention', SENTENCE, SAVE_WAV_PATH, fp16_support=False)
# overwrite
parser.add_argument(
    '--input', '-i', metavar='TEXT', default=SENTENCE,
    help='input text'
)
parser.add_argument(
    '--ailia_audio', action='store_true',
    help='use ailia audio library'
)
args = update_parser(parser, check_input_type=False)

if args.ailia_audio:
    from pytorch_dc_tts_utils_ailia import get_test_data, save_to_wav
else:
    from pytorch_dc_tts_utils import get_test_data, save_to_wav

# ======================
# Main function
# ======================
def preprocess(SENTENCE):
    L = get_test_data([SENTENCE], len(SENTENCE))
    zeros = np.zeros((1, 80, 1), np.float32)
    Y = zeros
    A = None
    return L, Y, zeros, A


def inference(net_t2m, net_ssrm, L, Y, zeros, A):
    Y = inference_by_text2mel(net_t2m, L, Y, zeros, A)
    Z = inference_by_ssr(net_ssrm, Y)
    return Z[0, :, :].T


def inference_by_text2mel(net_t2m, L, Y, zeros, A):
    for t in (range(MAX_T)):
        net_t2m.set_input_blob_shape(Y.shape, net_t2m.find_blob_index_by_name('input.2'))
        _, Y_t, A = net_t2m.predict({'input.1':L, 'input.2':Y})

        Y = np.concatenate([zeros, Y_t], 2)
        attention = np.argmax(A[0, :, -1], 0)
        vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding, E: EOS.
        if L[0, attention] == vocab.index('E'):  # EOS
            break

    return Y


def inference_by_ssr(net_ssrm, Y):
    _, Z = net_ssrm.predict({'input.1':Y})
    return Z


def generate_sentence(sentence):
    # prepare data
    L, Y, zeros, A = preprocess(sentence)

    # model initialize
    net_t2m = ailia.Net(MODEL_PATH_T2M, WEIGHT_PATH_T2M, env_id=args.env_id, memory_mode=args.memory_mode)
    net_ssrm = ailia.Net(MODEL_PATH_SSRM, WEIGHT_PATH_SSRM, env_id=args.env_id, memory_mode=args.memory_mode)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for c in range(5):
            start = int(round(time.time() * 1000))
            out = inference(net_t2m, net_ssrm, L, Y, zeros, A)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end-start))
    else:
        out = inference(net_t2m, net_ssrm, L, Y, zeros, A)

    savepath = args.savepath
    logger.info(f'saved at : {savepath}')
    save_to_wav(out, savepath)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH_T2M, MODEL_PATH_T2M, REMOTE_PATH_T2M)
    check_and_download_models(WEIGHT_PATH_SSRM, MODEL_PATH_SSRM, REMOTE_PATH_SSRM)

    generate_sentence(args.input)


if __name__ == "__main__":
    main()
