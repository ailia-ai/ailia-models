import sys
import time
from itertools import combinations
from logging import getLogger
import json

import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'GLuCoSE-base-ja.onnx'
MODEL_PATH = 'GLuCoSE-base-ja.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/glucose/'

TEXT_PATH = 'sentences.txt'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'GLuCoSE-base-Japanese', TEXT_PATH, None
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# ======================
# Main functions
# ======================

def post_processing(token_embeddings, attention_mask):
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)

    sum_mask = input_mask_expanded.sum(axis=1)
    sum_mask = np.clip(sum_mask, 1e-9, None)

    sentence_embedding = sum_embeddings / sum_mask

    return sentence_embedding


def predict(models, sentences):
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

    tokenizer = models["tokenizer"]
    net = models["net"]

    batch_size = 32

    all_embeddings = []
    for start_index in range(0, len(sentences), batch_size):
        sentences_batch = sentences_sorted[start_index: start_index + batch_size]

        to_tokenize = [sentences_batch]
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        max_seq_length = 512
        features = tokenizer(
            *to_tokenize,
            padding=True,
            truncation="longest_first",
            return_tensors="np",
            max_length=max_seq_length,
        )
        input_ids = features["input_ids"]
        attention_mask = features["attention_mask"]

        # feedforward
        if not args.onnx:
            output = net.predict([input_ids, attention_mask])
        else:
            output = net.run(None, {
                'input_ids': input_ids, 'attention_mask': attention_mask
            })
        token_embeddings = output[0]

        embeddings = post_processing(token_embeddings, attention_mask)
        all_embeddings.extend(embeddings)

    all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    all_embeddings = np.asarray(all_embeddings)

    return all_embeddings


def recognize_from_sentence(models):
    with open(args.input[0], encoding='utf-8') as f:
        sentences = [s.strip() for s in f.read().split("\n")]

    logger.info(
        "sentences:\n"
        + "\n".join([f"#{i + 1}. {s}" for i, s in enumerate(sentences)])
    )

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            embeddings = predict(models, sentences)
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)

            # Logging
            logger.info(f'\tailia processing estimation time {estimation_time} ms')
            if i != 0:
                total_time_estimation = total_time_estimation + estimation_time

        logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
    else:
        embeddings = predict(models, sentences)

    comb_score = []
    for i, j in combinations(range(len(sentences)), 2):
        score = cos_sim(embeddings[i], embeddings[j])
        comb_score.append(((i, j), score))

    comb_score = sorted(comb_score, key=lambda x: -x[1])
    logger.info(
        "The top similar are below.\n"
        + "\n".join(f'#{i + 1} & #{j + 1} : {score}' for (i, j), score in comb_score[:3])
    )

    if args.write_json:
        out_data = []
        for (i, j), score in comb_score[:3]:
            out_data.append({ "idx_a": i, "idx_b": j, "score": score})
        out_data = { "sentences": sentences, "result": out_data }
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

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
        tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    else:
        from ailia_tokenizer import XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained("./tokenizer/")

    models = {
        "net": net,
        "tokenizer": tokenizer,
    }

    recognize_from_sentence(models)


if __name__ == '__main__':
    main()
