import sys
import time
from logging import getLogger

import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_BASE_PATH = 'multilingual-e5-base.onnx'
MODEL_BASE_PATH = 'multilingual-e5-base.onnx.prototxt'
WEIGHT_LARGE_PATH = 'multilingual-e5-large.onnx'
MODEL_LARGE_PATH = 'multilingual-e5-large.onnx.prototxt'
WEIGHT_LARGE_PB_PATH = 'multilingual-e5-large_weights.pb'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/multilingual-e5/'

SENTENCE_PATH = 'sample.txt'
MIN_LENGTH = 5

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Multilingual-E5', SENTENCE_PATH, None
)
parser.add_argument(
    '-p', '--prompt', metavar='PROMPT', default=None,
    help='Specify input prompt. If not specified, script runs interactively.'
)
parser.add_argument(
    '-m', '--model_type', default='base', choices=('base', 'large'),
    help='model type'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--opt',
    action='store_true',
    help='use opset17 version.'
)
args = update_parser(parser)

if args.opt:
    WEIGHT_BASE_PATH = 'multilingual-e5-base.opt.onnx'
    MODEL_BASE_PATH = 'multilingual-e5-base.opt.onnx.prototxt'

# ======================
# Secondaty Functions
# ======================

def read_sentences(file_path):
    if file_path.endswith(".pdf"):
        from pdfminer.high_level import extract_text
        text = extract_text(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    sents = text.replace('\n', '').split('。')
    sents = [s.strip() + '。' for s in sents if len(s.strip()) > MIN_LENGTH]

    return sents


def average_pool(
        last_hidden_states,
        attention_mask):
    mask_expand = ~np.expand_dims(attention_mask, -1).astype(bool)
    mask_expand = np.broadcast_to(mask_expand, last_hidden_states.shape)
    last_hidden = np.ma.array(
        last_hidden_states,
        mask=mask_expand,
    ).filled(0.0)

    return last_hidden.sum(axis=1) / attention_mask.sum(axis=1)[..., None]


def closest_sentence(embs, q_emb):
    norm_embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    norm_q_emb = q_emb / np.linalg.norm(q_emb)

    cos_sim = np.sum(norm_embs * norm_q_emb, axis=1)
    idx, sim = np.argmax(cos_sim), np.max(cos_sim)

    return idx, sim


# ======================
# Main functions
# ======================

def predict(models, sentences, header):
    input_texts = [header + ': {}'.format(t) for t in sentences]

    tokenizer = models['tokenizer']
    batch_dict = tokenizer(
        input_texts,
        max_length=512, padding=True, truncation=True,
        return_tensors='np')

    input_ids = batch_dict['input_ids']
    attention_mask = batch_dict['attention_mask']

    net = models['net']

    # feedforward
    if not args.onnx:
        output = net.predict([input_ids, attention_mask])
    else:
        output = net.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
    last_hidden_state = output[0]

    embeddings = average_pool(last_hidden_state, attention_mask)

    return embeddings


def recognize_from_sentence(models):
    prompt = args.prompt

    # extract sentences to list
    sentences = read_sentences(args.input[0])

    # inference
    logger.info("Generating embeddings...")
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total = 0
        for i in range(5):
            start = int(round(time.time() * 1000))
            embs = predict(models, sentences, "passage")
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
            total = total + end - start
        logger.info(f'average time {total / 5} ms\n')
        return
    else:
        embs = predict(models, sentences, "passage")

    # check prompt from command line argument
    if prompt is not None:
        prompt_emb = predict(models, [prompt], "query")

        idx, sim = closest_sentence(embs, prompt_emb)

        print(f'Prompt: {prompt}')
        print(f'Text: {sentences[idx]} (Similarity:{sim:.3f})')
        return

    # application
    prompt = input('User (press q to exit): ')
    while prompt not in ('q', 'ｑ'):
        prompt_emb = predict(models, [prompt], "query")

        idx, sim = closest_sentence(embs, prompt_emb)

        print(f'Text: {sentences[idx]} (Similarity:{sim:.3f})')

        prompt = input('User (press q to exit): ')


def main():
    dic_model = {
        'base': (WEIGHT_BASE_PATH, MODEL_BASE_PATH),
        'large': (WEIGHT_LARGE_PATH, MODEL_LARGE_PATH),
    }
    WEIGHT_PATH, MODEL_PATH = dic_model[args.model_type]

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    if args.model_type == 'large':
        check_and_download_file(WEIGHT_LARGE_PB_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('tokenizer')
    else:
        from ailia_tokenizer import XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained('./tokenizer/')

    models = {
        "net": net,
        "tokenizer": tokenizer,
    }

    if args.profile:
        net.set_profile_mode(True)

    recognize_from_sentence(models)

    if args.profile:
        print(net.get_summary())


if __name__ == '__main__':
    main()
