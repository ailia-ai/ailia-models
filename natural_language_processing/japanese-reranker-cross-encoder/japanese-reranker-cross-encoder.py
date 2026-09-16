import sys
import time
from logging import getLogger

import os
import shutil

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH = "japanese-reranker-cross-encoder-large-v1.onnx"
MODEL_PATH = "japanese-reranker-cross-encoder-large-v1.onnx.prototxt"
REMOTE_PATH = (
    "https://storage.googleapis.com/ailia-models/japanese-reranker-cross-encoder/"
)

TEXT_PATH = "sentences.txt"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("japanese-reranker-cross-encoder", TEXT_PATH, None)
parser.add_argument(
    "--query", type=str, default="感動的な映画について", help="Input query."
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Main functions
# ======================


def predict(models, query, sentences):
    tokenizer = models["tokenizer"]
    net = models["net"]

    features = tokenizer(
        [[query, passage] for passage in sentences],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )
    input_ids = features["input_ids"]
    attention_mask = features["attention_mask"]
    token_type_ids = features["token_type_ids"]

    # feedforward
    if not args.onnx:
        output = net.predict([input_ids, attention_mask, token_type_ids])
    else:
        output = net.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
    scores = output[0]
    scores = scores.reshape(len(sentences))

    return scores


def recognize_from_sentence(models):
    query = args.query
    with open(args.input[0], encoding="utf-8") as f:
        sentences = [s.strip() for s in f.read().split("\n")]

    logger.info("query: " + query)
    logger.info(
        "sentences:\n" + "\n".join([f"({i + 1}) {s}" for i, s in enumerate(sentences)])
    )

    # inference
    logger.info("Start inference...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            scores = predict(models, query, sentences)
            end = int(round(time.time() * 1000))
            estimation_time = end - start

            # Logging
            logger.info(f"\tailia processing estimation time {estimation_time} ms")
            if i != 0:
                total_time_estimation = total_time_estimation + estimation_time

        logger.info(
            f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
        )
    else:
        scores = predict(models, query, sentences)

    comb_score = sorted(zip(enumerate(sentences), scores), key=lambda x: -x[1])
    logger.info(
        "The scores in order of higher are below.\n"
        + "\n".join(
            f"({i+1}) {sentence} ({score:.3f})" for (i, sentence), score in comb_score
        )
    )

    logger.info("Script finished successfully.")


def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    else:
        from ailia_tokenizer import BertJapaneseWordPieceTokenizer
        VOCAB_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_maskedlm/"
        check_and_download_file("unidic-lite.zip", VOCAB_REMOTE_PATH)
        if not os.path.exists("unidic-lite"):
            shutil.unpack_archive('unidic-lite.zip', '')
        tokenizer = BertJapaneseWordPieceTokenizer.from_pretrained(dict_path = 'unidic-lite', pretrained_model_name_or_path = './tokenizer/')


    models = {
        "net": net,
        "tokenizer": tokenizer,
    }

    recognize_from_sentence(models)


if __name__ == "__main__":
    main()
