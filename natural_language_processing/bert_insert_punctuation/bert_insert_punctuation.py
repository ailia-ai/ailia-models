import sys
import time
from logging import getLogger
import pprint
import os
import shutil

import numpy as np
from scipy.special import softmax

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'bert_insert_punctuation.obf.onnx'
MODEL_PATH = 'bert_insert_punctuation.obf.onnx.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_insert_punctuation/"

LABEL_TO_TEXT = ['','、','。','？','！','・']

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'punctbert', None, None
)
parser.add_argument(
    "-i", "--input", metavar="TEXT", type=str,
    default=\
"小坂は16歳にして1973年の第6回ヤマハポピュラーソングコンテストに出場し\
ピアノを弾きながらこの曲を歌唱してグランプリを獲得した同年11月の第4回\
世界歌謡祭にて最優秀賞グランプリを受賞する同年年末にレコードリリー\
スされオリコン集計で約160万枚発売元のワーナーによる発表では20\
0万枚を超える売り上げを記録した",
    help="Input text."
)

parser.add_argument(
    "-sc", "--score", action = 'store_true'
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

    model_inputs = (np.array(enc['input_ids'])[None],#prepare input
                    np.array(enc['attention_mask'])[None],
                    np.array(enc['token_type_ids'])[None], )
    
    output = net.predict(model_inputs)[0][0][1:-1]#remove special tokens
    output_ids = np.argmax(output, axis=-1)
    return output_ids, output

def decode_output(input_text, output_ids, tokenizer):
    added_text = ''
    for text, pred in zip(tokenizer.tokenize(input_text), output_ids):
        added_text += (handle_subwords(text) + LABEL_TO_TEXT[pred])
    return added_text

# ======================
# Main functions
# ======================


def add_punctuations(model):
    input_text = args.input
    visualize_score = args.score

    logger.info("input_text: %s" % input_text)

    # inference
    logger.info('inference has started...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))

            output_ids, output = predict(model, input_text)

            end = int(round(time.time() * 1000))
            estimation_time = (end - start)

            # Logging
            logger.info(f'\tailia processing estimation time {estimation_time} ms')
            if i != 0:
                total_time_estimation = total_time_estimation + estimation_time

        logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
    else:
        output_ids, output = predict(model, input_text)
        
    

    if visualize_score:
        pred_dict = {}
        print(len(model['tokenizer'].tokenize(input_text)))
        print(output.shape)
        for i,t in enumerate(model['tokenizer'].tokenize(input_text)):
            pred_score = softmax(output[i])[output_ids[i]]
            pred_dict[i] = {'token': handle_subwords(t), 'pred_punct': LABEL_TO_TEXT[output_ids[i]], 'score':pred_score}
        logger.info(f"Confidence scores: \n{pprint.pformat(pred_dict)}")
    else:
        punct_added_text = decode_output(input_text, output_ids, model['tokenizer'])
        logger.info(f"Text with added punctuations:\n{pprint.pformat(punct_added_text)}")

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        model_name = "cl-tohoku/bert-base-japanese-v3"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        from ailia_tokenizer import BertJapaneseWordPieceTokenizer
        VOCAB_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_maskedlm/"
        check_and_download_file("unidic-lite.zip", VOCAB_REMOTE_PATH)
        if not os.path.exists("unidic-lite"):
            shutil.unpack_archive('unidic-lite.zip', '')
        tokenizer = BertJapaneseWordPieceTokenizer.from_pretrained(dict_path = 'unidic-lite', pretrained_model_name_or_path='./tokenizer/')

    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    model = {
        "tokenizer": tokenizer,
        "net": net,
    }

    add_punctuations(model)

if __name__ == '__main__':
    main()