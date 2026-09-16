import sys
import time
import numpy as np

from tokenizer import BertTokenizer # noqa: E402

import ailia # noqa: E402

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models # noqa: E402
from arg_utils import get_base_parser, update_parser  # noqa: E402

# logger
from logging import getLogger # noqa: E402
logger = getLogger(__name__)


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    ('PreSumm is a text summarization model.'),
    None,
    None,
)
parser.add_argument(
    '-f', '--file', type=str,
    default='sample.txt',
)
args = update_parser(parser)


# =========================
# PARAMETERS
# =========================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/presumm/'
VOCAB_PATH = 'bert-base-uncased-vocab.txt'
MODEL_PATH = 'cnndm-bertext.onnx.prototxt'
WEIGHT_PATH = 'cnndm-bertext.onnx'

SPLIT_CHAR = '\n'
MAX_LENGTH = 512
NUM_PREDICT = 3 # Top NUM_PREDICT predictions will be displayed.


def preprocess(tokenizer, text):
    origin_txt = text.split(SPLIT_CHAR)
    text = text.lower()
    text = ' [SEP] [CLS] '.join(text.split(SPLIT_CHAR))
    tokens = tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    tokens_idx = tokenizer.convert_tokens_to_ids(tokens)
    tokens_idx = tokens_idx[:MAX_LENGTH]

    cls_id = tokenizer.vocab['[CLS]']

    src = np.array([tokens_idx])
    mask_src = 1 - (src == 0)

    clss = np.array([[i for i, t in enumerate(tokens_idx) if t == cls_id]])
    mask_clss = np.ones_like(clss)

    seg = 1
    n_tokens = len(tokens_idx)
    segment_ids = [None] * n_tokens
    for i in range(n_tokens):
        if tokens_idx[i] == cls_id:
            seg = 1 - seg
        segment_ids[i] = seg
    segment_ids = np.array([segment_ids])

    return origin_txt, src, segment_ids, clss, mask_src, mask_clss


def postprocess(text, predict):
    scores, mask = predict
    scores += mask

    rank_ids = np.argsort(-scores, 1)
    rank_ids = rank_ids[:, :NUM_PREDICT]

    result = [text[i] for i in rank_ids[0]]
    return result


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # Load text from file and pre-process
    tokenizer = BertTokenizer.from_pretrained(VOCAB_PATH, do_lower_case=True)

    with open(args.file) as f:
        text = f.read()

    splited_text, *input_data = preprocess(tokenizer, text)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # inference
    logger.info('Start summarize...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for c in range(5):
            start = int(round(time.time() * 1000))
            preds_ailia = net.predict(input_data)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end-start))
    else:
        preds_ailia = net.predict(input_data)

    # 
    summarized_text = postprocess(splited_text, preds_ailia)

    for i in range(NUM_PREDICT):
        logger.info(f'Top {i+1}: {summarized_text[i]}')


if __name__ == '__main__':
    main()
