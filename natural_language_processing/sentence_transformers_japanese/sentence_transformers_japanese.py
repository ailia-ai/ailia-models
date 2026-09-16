import sys
import time
import platform

from logging import getLogger

import ailia
import numpy as np

# import local modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser # noqa: E402
from model_utils import check_and_download_models, check_and_download_file # noqa: E402


logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_NAME = 'paraphrase-multilingual-mpnet-base-v2'

WEIGHT_PATH = WEIGHT_NAME + '.onnx'
MODEL_PATH = WEIGHT_PATH + '.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/sentence-transformers-japanese/'

SAMPLE_PDF_PATH = 'sample.txt'
MIN_LENGTH = 5


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'sentence transformers japanese', SAMPLE_PDF_PATH, None , fp16_support=False
)
parser.add_argument(
    '-p', '--prompt', metavar='PROMPT', default=None,
    help='Specify input prompt. If not specified, script runs interactively.'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def preprocess(file_path):
    if ".pdf" in file_path:
        from pdfminer.high_level import extract_text
        text = extract_text(file_path)
    else:
        f = open(file_path, "r", encoding="utf-8")
        text = f.read()
        f.close()
    sents = text.replace('\n', '').split('。')
    sents = [s.strip()+'。' for s in sents if len(s.strip()) > MIN_LENGTH]
    return sents


def predict(model, tokenizer, sents):
    inputs = tokenizer(sents, padding=True, truncation=True, return_tensors='np')
    inputs_ailia = [v for k, v in inputs.items()]

    out = model.predict(inputs_ailia)
    emb, mask = out[0], inputs['attention_mask']
    return emb, mask


def postprocess(features, mask):
    mask_expand = np.expand_dims(mask, -1)
    mask_expand = np.broadcast_to(mask_expand, features.shape)

    feat_sum = np.sum(features * mask_expand, 1)
    mask_sum = mask_expand.sum(1).clip(1e-9, None)
    mean_pool = feat_sum / mask_sum
    return mean_pool


def closest_sentence(pdf_emb, q_emb):
    norm_pdf_emb = pdf_emb / np.linalg.norm(pdf_emb, axis=1, keepdims=True)
    norm_q_emb = q_emb / np.linalg.norm(q_emb)

    cos_sim = np.sum(norm_pdf_emb * norm_q_emb, 1)
    idx, sim = np.argmax(cos_sim), np.max(cos_sim)
    return idx, sim
    

# ======================
# Main Functions
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # initialize
    model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/' + WEIGHT_NAME)
    else:
        from ailia_tokenizer import XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained("./tokenizer/")

    # extract pdf sentences to list
    sentences = preprocess(args.input[0])

    # inference
    logger.info("Generating embeddings...")
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            features, attn_mask = predict(model, tokenizer, sentences)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end-start} ms')
        exit()
    else:
        features, attn_mask = predict(model, tokenizer, sentences)

    # create comparable embeddings
    pdf_emb = postprocess(features, attn_mask)

    # check prompt from command line argument
    prompt = args.prompt
    if prompt is not None:
        out = predict(model, tokenizer, prompt)
        prompt_emb = postprocess(*out)

        idx, sim = closest_sentence(pdf_emb, prompt_emb)

        print(f'Prompt: {prompt}')
        print(f'Text: {sentences[idx]} (Similarity:{sim:.3f})')
        return

    # application
    prompt = input('User (press q to exit): ')
    while prompt not in ('q', 'ｑ'):

        out = predict(model, tokenizer, prompt)
        prompt_emb = postprocess(*out)

        idx, sim = closest_sentence(pdf_emb, prompt_emb)

        print(f'Text: {sentences[idx]} (Similarity:{sim:.3f})')

        prompt = input('User (press q to exit): ')


if __name__ == '__main__':
    main()

