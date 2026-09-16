import time
import sys

import numpy

import os
import shutil

import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models, check_and_download_file  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# require ailia SDK 1.2.5 and later

# ======================
# Arguemnt Parser Config
# ======================

DEFAULT_TEXT = 'iPhone 12 mini が欲しい'
# DEFAULT_TEXT = 'iPhone 12 mini は高い'


parser = get_base_parser('bert tweets sentiment.', None, None)
# overwrite
parser.add_argument(
    '--input', '-i', metavar='TEXT', default=DEFAULT_TEXT,
    help='input text'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser, check_input_type=False)


# ======================
# PARAMETERS
# ======================

WEIGHT_PATH = "bert_tweets_sentiment.onnx"
MODEL_PATH = "bert_tweets_sentiment.onnx.prototxt"
REMOTE_PATH = \
    "https://storage.googleapis.com/ailia-models/bert_tweets_sentiment/"


# ======================
# Main function
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    ailia_model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    if args.disable_ailia_tokenizer:
        from transformers import BertJapaneseTokenizer
        tokenizer = BertJapaneseTokenizer.from_pretrained(
            'cl-tohoku/bert-base-japanese-whole-word-masking'
        )
    else:
        from ailia_tokenizer import BertJapaneseWordPieceTokenizer
        VOCAB_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_maskedlm/"
        check_and_download_file("ipadic.zip", VOCAB_REMOTE_PATH)
        if not os.path.exists("ipadic"):
            shutil.unpack_archive('ipadic.zip', '')
        tokenizer = BertJapaneseWordPieceTokenizer.from_pretrained(dict_path='ipadic', pretrained_model_name_or_path='./tokenizer/')

    model_inputs = tokenizer.encode_plus(args.input, return_tensors="np")
    inputs_onnx = {
        k: v for k, v in model_inputs.items()
    }

    logger.info("Text : "+str(args.input))
    logger.info("Input : "+str(inputs_onnx))

    # inference
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            score = ailia_model.predict(inputs_onnx)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end - start))
    else:
        score = ailia_model.predict(inputs_onnx)

    logger.info("Output : "+str(score))

    label_name = ["positive", "negative"]

    logger.info("Label : "+str(label_name[numpy.argmax(numpy.array(score))]))

    logger.info('Script finished successfully.')


if __name__ == "__main__":
    main()
