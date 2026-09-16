import sys
import codecs

import numpy
import os
import shutil

import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models, check_and_download_file  # noqa: E402


# ======================
# Arguemnt Parser Config
# ======================

MODEL_LISTS = [
    'bert-base-cased',
    'bert-base-uncased',
    'bert-base-japanese-whole-word-masking'
]

parser = get_base_parser('masklm proofreading sample', None, None)
# overwrite
parser.add_argument(
    '-i', '--input', metavar='VIDEO',
    default="test_text_jp.txt",
    help='The input video path.'
)
# overwrite
parser.add_argument(
    '-s', '--suggest',
    action='store_true',
    help='Show suggestion word'
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
args = update_parser(parser)


# ======================
# PARAMETERS
# ======================

WEIGHT_PATH = args.arch+".onnx"
MODEL_PATH = args.arch+".onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_maskedlm/"

PADDING_LEN = 512


# ======================
# Utils
# ======================

def softmax(x):
    u = numpy.sum(numpy.exp(x))
    return numpy.exp(x)/u


def inference(net, tokenizer, tokenized_text, masked_index, original_text_len):
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    indexed_tokens = numpy.expand_dims(numpy.array(indexed_tokens), axis=0)
    token_type_ids = numpy.zeros((1, len(tokenized_text)))
    attention_mask = numpy.zeros((1, len(tokenized_text)))
    attention_mask[:, 0:original_text_len] = 1

    inputs_onnx = {
        "token_type_ids": token_type_ids,
        "input_ids": indexed_tokens,
        "attention_mask": attention_mask,
    }

    outputs = net.predict(inputs_onnx)

    outputs[0][0, masked_index] = softmax(outputs[0][0, masked_index])
    return outputs


def colorize(tokenized_text, score, sujest):
    if args.arch == 'bert-base-cased' or args.arch == 'bert-base-uncased':
        space = " "
    else:
        space = ""

    fine_text = ""
    for i in range(0, len(tokenized_text)):
        if tokenized_text[i] == "[PAD]":
            continue
        prob_yellow = 0.0001
        prob_red = 0.00001
        if score[i] < prob_red:
            fine_text = fine_text+'\033[31m'+space+tokenized_text[i]+'\033[0m'
            if args.suggest:
                fine_text = fine_text+' ->\033[34m'+space+sujest[i]+'\033[0m'
        elif score[i] < prob_yellow:
            fine_text = fine_text+'\033[33m'+space+tokenized_text[i]+'\033[0m'
            if args.suggest:
                fine_text = fine_text+' ->\033[34m'+space+sujest[i]+'\033[0m'
        else:
            fine_text = fine_text+space+tokenized_text[i]

    if args.arch == 'bert-base-cased' or args.arch == 'bert-base-uncased':
        fine_text = fine_text.replace(' ##', '')
    else:
        fine_text = fine_text.replace('##', '')
    return fine_text


# ======================
# Main function
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.arch == 'bert-base-cased':
        if args.disable_ailia_tokenizer:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        else:
            from ailia_tokenizer import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained("./tokenizer/bert-base-cased/")
    elif args.arch == 'bert-base-uncased':
        if args.disable_ailia_tokenizer:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            from ailia_tokenizer import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('./tokenizer/bert-base-uncased/')
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
            tokenizer = BertJapaneseWordPieceTokenizer.from_pretrained(dict_path='ipadic', pretrained_model_name_or_path='./tokenizer/bert-base-japanese-whole-word-masking/')

    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    net.set_input_blob_shape(
        (1, PADDING_LEN), net.find_blob_index_by_name("token_type_ids")
    )
    net.set_input_blob_shape(
        (1, PADDING_LEN), net.find_blob_index_by_name("input_ids")
    )
    net.set_input_blob_shape(
        (1, PADDING_LEN), net.find_blob_index_by_name("attention_mask")
    )

    with codecs.open(args.input[0], 'r', 'utf-8', 'ignore') as f:
        s = f.readlines()

    for text in s:
        tokenized_text = tokenizer.tokenize(text)
        original_text_len = len(tokenized_text)

        for j in range(len(tokenized_text), PADDING_LEN):
            tokenized_text.append('[PAD]')

        score = numpy.zeros((len(tokenized_text)))
        suggest = {}

        for i in range(0, len(tokenized_text)):
            masked_index = i

            if tokenized_text[masked_index] == '[PAD]':
                continue

            tokenized_text_saved = tokenized_text[masked_index]

            tokenized_text[masked_index] = '[MASK]'

            outputs = inference(
                net, tokenizer, tokenized_text, masked_index, original_text_len
            )

            target_ids = tokenizer.convert_tokens_to_ids(
                [tokenized_text_saved]
            )
            index = target_ids[0]
            score[masked_index] = outputs[0][0, masked_index][index]

            index = numpy.argmax(outputs[0][0, masked_index])
            top_token = tokenizer.convert_ids_to_tokens([index])[0]
            suggest[masked_index] = top_token

            tokenized_text[masked_index] = tokenized_text_saved

        fine_text = colorize(tokenized_text, score, suggest)
        print(fine_text)

    print('Script finished successfully.')


if __name__ == "__main__":
    main()
