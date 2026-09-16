import sys
import time

import torch
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


# ======================
# Arguemnt Parser Config
# ======================

MODEL_LISTS = [
    'bert-base-cased',
    'bert-base-uncased',
    'bert-base-japanese-whole-word-masking',
    'bert-base-japanese-char-whole-word-masking',
    'bert-base-japanese-v3',
    'bert-base-japanese-char-v3',
]

NUM_PREDICT = 5

SENTENCE = '私は[MASK]で動く。'


parser = get_base_parser('bert masklm sample.', None, None)
# overwrite
parser.add_argument(
    '--input', '-i', metavar='TEXT',
    default=SENTENCE,
    help='input text'
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='bert-base-japanese-whole-word-masking', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
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

WEIGHT_PATH = args.arch+".onnx"
MODEL_PATH = args.arch+".onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_maskedlm/"


# ======================
# Get tokenizer
# ======================

def get_tokenizer():
    if args.arch == 'bert-base-cased':
        if args.disable_ailia_tokenizer:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        else:
            from ailia_tokenizer import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained("tokenizer/bert-base-cased/")
    elif args.arch == 'bert-base-uncased':
        if args.disable_ailia_tokenizer:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            from ailia_tokenizer import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('tokenizer/bert-base-uncased/')
    elif args.arch == 'bert-base-japanese-whole-word-masking':
        if args.disable_ailia_tokenizer:
            from transformers import BertJapaneseTokenizer
            tokenizer = BertJapaneseTokenizer.from_pretrained(
                'cl-tohoku/bert-base-japanese-whole-word-masking'
            )
        else:
            from ailia_tokenizer import BertJapaneseWordPieceTokenizer
            check_and_download_file("ipadic.zip", REMOTE_PATH)
            if not os.path.exists("ipadic"):
                shutil.unpack_archive('ipadic.zip', '')
            tokenizer = BertJapaneseWordPieceTokenizer.from_pretrained(dict_path = 'ipadic', pretrained_model_name_or_path = 'tokenizer/bert-base-japanese-whole-word-masking/')
    elif args.arch == 'bert-base-japanese-char-whole-word-masking':
        if args.disable_ailia_tokenizer:
            from transformers import BertJapaneseTokenizer
            tokenizer = BertJapaneseTokenizer.from_pretrained(
                'cl-tohoku/bert-base-japanese-char-whole-word-masking'
            )
        else:
            from ailia_tokenizer import BertJapaneseCharacterTokenizer
            check_and_download_file("ipadic.zip", REMOTE_PATH)
            if not os.path.exists("ipadic"):
                shutil.unpack_archive('ipadic.zip', '')
            tokenizer = BertJapaneseCharacterTokenizer.from_pretrained(dict_path = 'ipadic', pretrained_model_name_or_path = 'tokenizer/bert-base-japanese-char-whole-word-masking/')
    elif args.arch == 'bert-base-japanese-v3':
        if args.disable_ailia_tokenizer:
            from transformers import BertJapaneseTokenizer
            tokenizer = BertJapaneseTokenizer.from_pretrained(
                'cl-tohoku/bert-base-japanese-v3'
            )
        else:
            from ailia_tokenizer import BertJapaneseWordPieceTokenizer
            check_and_download_file("unidic-lite.zip", REMOTE_PATH)
            if not os.path.exists("unidic-lite"):
                shutil.unpack_archive('unidic-lite.zip', '')
            tokenizer = BertJapaneseWordPieceTokenizer.from_pretrained(dict_path = 'unidic-lite', pretrained_model_name_or_path = 'tokenizer/bert-base-japanese-v3/')
    elif args.arch == 'bert-base-japanese-char-v3':
        if args.disable_ailia_tokenizer:
            from transformers import BertJapaneseTokenizer
            tokenizer = BertJapaneseTokenizer.from_pretrained(
                'cl-tohoku/bert-base-japanese-char-v3'
            )
        else:
            from ailia_tokenizer import BertJapaneseCharacterTokenizer
            check_and_download_file("unidic-lite.zip", REMOTE_PATH)
            if not os.path.exists("unidic-lite"):
                shutil.unpack_archive('unidic-lite.zip', '')
            tokenizer = BertJapaneseCharacterTokenizer.from_pretrained(dict_path = 'unidic-lite', pretrained_model_name_or_path = 'tokenizer/bert-base-japanese-char-v3/')
    else:
        logger.error("unknown arch")
        return None
    return tokenizer


# ======================
# Main function
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # get tokenizer
    tokenizer = get_tokenizer()
    if tokenizer == None:
        return

    text = args.input
    logger.info("Input text : "+text)

    tokenized_text = tokenizer.tokenize(text)
    logger.info("Tokenized text : " + str(tokenized_text))

    masked_index = -1
    for i in range(0, len(tokenized_text)):
        if tokenized_text[i] == '[MASK]':
            masked_index = i
            break
    if masked_index == -1:
        logger.info("[MASK] not found")
        sys.exit(1)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    logger.info("Indexed tokens : "+str(indexed_tokens))

    ailia_model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    indexed_tokens = numpy.array(indexed_tokens)
    token_type_ids = numpy.zeros((1, len(tokenized_text)))
    attention_mask = numpy.zeros((1, len(tokenized_text)))
    attention_mask[:, 0:len(tokenized_text)] = 1

    inputs_onnx = {
        "token_type_ids": token_type_ids,
        "input_ids": indexed_tokens,
        "attention_mask": attention_mask,
    }

    logger.info("Predicting...")
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            outputs = ailia_model.predict(inputs_onnx)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end - start))
    else:
        outputs = ailia_model.predict(inputs_onnx)

    predictions = torch.from_numpy(
        outputs[0][0, masked_index]).topk(NUM_PREDICT)

    logger.info("Predictions : ")
    for i, index_t in enumerate(predictions.indices):
        index = index_t.item()
        token = tokenizer.convert_ids_to_tokens([index])[0]
        logger.info(str(i)+" "+str(token))

    logger.info('Script finished successfully.')


if __name__ == "__main__":
    main()
