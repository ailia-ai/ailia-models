import time
import sys

import numpy as np

# to remove "deprecated error"
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import ailia  # noqa: E402
# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models, check_and_download_file  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

try:
    from pyknp import Juman  # noqa: E402
except ModuleNotFoundError:
    logger.warning('pyknp module is not installed. (for japanese mode)')


# ======================
# Arguemnt Parser Config
# ======================
LANGS = ['en', 'jp']

parser = get_base_parser(
    ('BERT is a state of the art language model. '
     'In our model, we solve the task of predicting the masked word.'),
    None,
    None,
)
parser.add_argument(
    '--lang', '-l', metavar='LANG',
    default='en', choices=LANGS,
    help='choose language: ' + ' | '.join(LANGS) + ' (default: en)'
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
NUM_PREDICT = 3  # Top NUM_PREDICT predictions will be displayed. (default=3)
LANG = args.lang
logger.info('language is set to ' + LANG)

if LANG == 'en':
    WEIGHT_PATH = "bert-base-uncased.onnx"
    MODEL_PATH = "bert-base-uncased.onnx.prototxt"
    REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_en/"

    MAX_SEQ_LEN = 128

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ CHANGE HERE ~~~ CHANGE HERE ~~~ CHANGE HERE ~~~ CHANGE HERE ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # masked word should be represented by '_'
    SENTENCE = 'I want to _ the car because it is cheap.'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ CHANGE HERE ~~~ CHANGE HERE ~~~ CHANGE HERE ~~~ CHANGE HERE ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

elif LANG == 'jp':
    # kyoto univ.
    WEIGHT_PATH = 'kyoto-bert-jp.onnx'
    MODEL_PATH = 'kyoto-bert-jp.onnx.prototxt'
    REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_jp/"

    MAX_SEQ_LEN = 512

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ CHANGE HERE ~~~ CHANGE HERE ~~~ CHANGE HERE ~~~ CHANGE HERE ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # masked word should be represented by '＿' (zen-kaku)
    SENTENCE = '私は車が安いので＿したい．'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ CHANGE HERE ~~~ CHANGE HERE ~~~ CHANGE HERE ~~~ CHANGE HERE ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ======================
# Utils
# ======================
def text2token(text, tokenizer, lang='en'):
    # convert a text to tokens which can be interpreted in BERT model
    if lang == 'en':
        text = text.replace('_', '[MASK]')
        masked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer.tokenize(masked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    elif lang == 'jp':
        jumanapp = Juman()
        juman_res = jumanapp.analysis(text)
        tokenized_text = [mrph.midasi for mrph in juman_res.mrph_list()]
        tokenized_text.insert(0, '[CLS]')
        tokenized_text.append('[SEP]')
        tokenized_text = [
            '[MASK]' if token == '＿' else token for token in tokenized_text
        ]
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    masked_index = tokenized_text.index('[MASK]')
    segments_ids = [0] * len(tokenized_text)
    tokens_ts = np.array([indexed_tokens])
    segments_ts = np.array([segments_ids])

    # input length fixed by max_seq_len
    # (ailia should manage adoptable input size)
    tokens_ts = np.pad(
        tokens_ts,
        [(0, 0), (0, MAX_SEQ_LEN-len(tokens_ts[0]))],
        'constant',
    )
    segments_ts = np.pad(
        segments_ts,
        [(0, 0), (0, MAX_SEQ_LEN-len(segments_ts[0]))],
        'constant',
    )
    assert tokens_ts.shape == (1, MAX_SEQ_LEN)
    assert segments_ts.shape == (1, MAX_SEQ_LEN)
    return tokens_ts, segments_ts, masked_index


# ======================
# Main function
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # bert tokenizer
    if LANG == 'en':
        if args.disable_ailia_tokenizer:
            from transformers import BertTokenizer  # noqa: E402
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            from ailia_tokenizer import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('./tokenizer/en/')
    elif LANG == 'jp':
        if args.disable_ailia_tokenizer:
            from transformers import BertTokenizer  # noqa: E402
            tokenizer = BertTokenizer(
                './tokenizer/jp/vocab.txt',
                do_lower_case=False,
                do_basic_tokenize=False,
            )
        else:
            # This tokenizer type not supported
            from ailia_tokenizer import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('./tokenizer/jp/')

    # prepare data
    sentence_id = np.ones((1, MAX_SEQ_LEN), dtype=np.int64)
    tokens_ts, segments_ts, masked_index = text2token(
        SENTENCE, tokenizer, lang=LANG
    )
    input_data = [tokens_ts, segments_ts, sentence_id]

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for c in range(5):
            start = int(round(time.time() * 1000))
            preds_ailia = net.predict(input_data)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end-start))
    else:
        preds_ailia = net.predict(input_data)

    # masked word prediction
    predicted_indices = np.argsort(
        preds_ailia[0][0][masked_index]
    )[-NUM_PREDICT:][::-1]

    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indices.tolist())

    logger.info('Input sentence: ' + SENTENCE)
    logger.info(f'predicted top {NUM_PREDICT} words: {predicted_tokens}')
    logger.info('Script finished successfully.')


if __name__ == "__main__":
    main()
