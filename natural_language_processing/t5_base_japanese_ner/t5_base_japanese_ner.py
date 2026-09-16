import sys
import time
from logging import getLogger

import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa

from onnx_t5 import T5Model
from decode_excerpt import decode

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

ENCODER_WEIGHT_PATH = 't5_base_japanese_ner_enc.onnx'
ENCODER_MODEL_PATH = 't5_base_japanese_ner_enc.onnx.prototxt'

DECODER_WEIGHT_PATH = 't5_base_japanese_ner_dec.onnx'
DECODER_MODEL_PATH = 't5_base_japanese_ner_dec.onnx.prototxt'

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/t5_base_japanese_ner/"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    't5_base_japanese_summarization', None, None, fp16_support=False
)

parser.add_argument(
    "-f", "--file", metavar="PATH", type=str,
    default="input.txt",
    help="Input text file path."
)

parser.add_argument(
    "-i", "--input", metavar="TEXT", type=str,
    default=None,
    help="Input text."
)

parser.add_argument(
    '--seed', type=int,
    help='random seed'
)

parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)

args = update_parser(parser, check_input_type=False)

if args.seed:
    np.random.seed(args.seed)

# ======================
# Helper functions
# ======================

def handle_subwords(token):
    r"""
    Description:
        Get rid of subwords '##'.
    About tokenizer subwords:
        See: https://huggingface.co/docs/transformers/tokenizer_summary
    """
    if len(token) > 2 and token[0:2] == '##':
        token = token[2:]
    return token


# ======================
# Main functions
# ======================


def recognize(model):
    input_text = args.input
    input_path = args.file
    if input_text is None:
        input_text = open(input_path, "r", encoding="utf-8").read()

    logger.info("input_text: %s" % input_text)

    # inference
    logger.info('inference has started...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))

            out, _ = model.estimate(input_text, max_length = 512, top_p = 0.93, repetition_penalty=0.5)

            end = int(round(time.time() * 1000))
            estimation_time = (end - start)

            # Logging
            logger.info(f'\tailia processing estimation time {estimation_time} ms')
            if i != 0:
                total_time_estimation = total_time_estimation + estimation_time

        logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
    else:
        #prediction = predict(model, input_text)
        out, _ = model.estimate(input_text, max_length = 512, top_p = 0.93, repetition_penalty=0.5)
    
    logger.info('predicted entities:')
    labels = decode(out.tolist(), input_text, model.tokenizer)
    logger.info(labels)
    # save output        -
    if args.savepath is not None:
        import pickle
        with open(args.savepath, mode="wb") as f:
            pickle.dump(labels, f)

        
        
    

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(ENCODER_WEIGHT_PATH, ENCODER_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(DECODER_WEIGHT_PATH, DECODER_MODEL_PATH, REMOTE_PATH)

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("sonoisa/t5-base-japanese")
    else:
        from ailia_tokenizer import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("./tokenizer/")

    env_id = args.env_id

    # initialize
    encoder = ailia.Net(ENCODER_MODEL_PATH, ENCODER_WEIGHT_PATH, env_id = env_id, memory_mode = args.memory_mode)
    decoder = ailia.Net(DECODER_MODEL_PATH, DECODER_WEIGHT_PATH, env_id = env_id, memory_mode = args.memory_mode)

    model = T5Model(encoder, decoder, tokenizer)

    recognize(model)

if __name__ == '__main__':
    main()