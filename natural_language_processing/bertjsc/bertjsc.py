import sys
import time
from logging import getLogger
import pprint

import numpy as np
from scipy.special import softmax

import os
import shutil

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa
from dtype_utils import numpy_type_to_builtin_type  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'bertjsc.onnx'
MODEL_PATH = 'bertjsc.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/bertjsc/'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'bertjsc', None, None
)
parser.add_argument(
    "-i", "--input", metavar="TEXT", type=str,
    default="日本語を校正しま.",
    help="Input text."
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser, check_input_type=False)

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

def predict(model, input_text):
    tokenizer = model["tokenizer"]
    net = model["net"]

    enc = tokenizer.encode_plus(#encode tokens
        text=input_text,
        max_length=512,
        truncation=True,
    )

    logger.info("Tokens: %s" % enc["input_ids"])

    model_inputs = (np.array(enc['input_ids'])[None],#prepare input
                    np.array(enc['attention_mask'])[None],
                    np.array(enc['token_type_ids'])[None], )
    
    output = net.predict(model_inputs)[0][0]
    output_ids = np.argmax(output, axis=-1)
    return make_pred_dict(enc['input_ids'], output_ids, output, tokenizer)

def make_pred_dict(input_ids, output_ids, logits, tokenizer):
    scores = softmax(logits, axis=-1)
    pred_dict={}
    for i in range(1,logits.shape[0]-1):#Loop over every token except [CLS], [SEP]
        pred_dict[i] = {
            'score': scores[i,output_ids[i]],
            'token':  handle_subwords(tokenizer.convert_ids_to_tokens(int(input_ids[i]))),
            'correct': handle_subwords(tokenizer.convert_ids_to_tokens(int(output_ids[i]))),

        }
    return pred_dict

# ======================
# Main functions
# ======================

def correct_text(model):
    input_text = args.input

    logger.info("input_text: %s" % input_text)

    # inference
    logger.info('inference has started...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            output = predict(model, input_text)
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)

            # Logging
            logger.info(f'\tailia processing estimation time {estimation_time} ms')
            if i != 0:
                total_time_estimation = total_time_estimation + estimation_time

        logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
    else:
        output = predict(model, input_text)

    logger.info(f"corrected_tokens:\n{pprint.pformat(numpy_type_to_builtin_type(output))}")

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.disable_ailia_tokenizer:
        from transformers import BertJapaneseTokenizer
        tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    else:
        from ailia_tokenizer import BertJapaneseWordPieceTokenizer
        VOCAB_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_maskedlm/"
        check_and_download_file("ipadic.zip", VOCAB_REMOTE_PATH)
        if not os.path.exists("ipadic"):
            shutil.unpack_archive('ipadic.zip', '')
        tokenizer = BertJapaneseWordPieceTokenizer.from_pretrained(dict_path='ipadic', pretrained_model_name_or_path='./tokenizer/')

    env_id = args.env_id

    # initialize
    net = ailia.Net(None, WEIGHT_PATH, env_id=env_id)

    model = {
        "tokenizer": tokenizer,
        "net": net,
    }

    correct_text(model)

if __name__ == '__main__':
    main()