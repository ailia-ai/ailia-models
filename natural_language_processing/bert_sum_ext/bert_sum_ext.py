import time
import sys

import numpy as np
from bert_sum_ext_utils import tokenize, select_sentences
from cluster_features import ClusterFeatures


import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = "bert-base.onnx"
MODEL_PATH = "bert-base.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_sum_ext/"

MIN_TEXT_LENGTH = 40
MAX_TEXT_LENGTH = 600

SAMPLE_TEXT_PATH = 'sample.txt'
NUM_PREDICTS = 3


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('bert extractive summarizer.', None, None)
parser.add_argument(
    '-f', '--file', type=str, default=SAMPLE_TEXT_PATH,
    help='input text file path'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser)


def preprocess(tokenizer, text):
    sents = select_sentences(
        text, MIN_TEXT_LENGTH, MAX_TEXT_LENGTH
    )
    sent_ids = [tokenize(tokenizer, s) for s in sents]
    return sents, sent_ids


def run(model, sentences_ids):
    embeddings = []
    for si in sentences_ids:
        inputs_onnx = {'input_ids': si}
        out = model.predict(inputs_onnx)[-2]
        embeddings.append(out.mean(1).squeeze())
    return embeddings


def postprocess(sentences, embeddings):
    predict_ids = ClusterFeatures(
        embeddings, 'kmeans'
    ).cluster(num_sentences=NUM_PREDICTS)
    results = [sentences[i] for i in predict_ids]
    return ' '.join(results)


# ======================
# Main function
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('./bert/')
    else:
        from ailia_tokenizer import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('./bert/')

    with open(args.file, encoding="utf-8") as f:
        body = f.read()
    
    logger.info(f'Input file : {args.file}')

    # preprocess
    sentences, sentences_ids = preprocess(tokenizer, body)

    # inference
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            embeddings = run(model, sentences_ids)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end - start))
    else:
        embeddings = run(model, sentences_ids)

    results = postprocess(sentences, embeddings)
    
    logger.info(f'Output : {results}')


if __name__ == "__main__":
    main()
