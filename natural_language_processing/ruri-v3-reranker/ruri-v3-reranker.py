import os
import sys
import time
from logging import getLogger

import numpy as np

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser
from model_utils import check_and_download_models

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH = "ruri-v3-reranker-310m.onnx"
MODEL_PATH = "ruri-v3-reranker-310m.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/ruri-v3-reranker/"

QUERY_DEFAULT = "瑠璃色はどんな色？"
DOCUMENT_PATH = "documents.txt"

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("ruri-v3-reranker", QUERY_DEFAULT, None)
parser.add_argument(
    "-d",
    "--document",
    action="append",
    nargs="+",
    default=None,
    help=(
        "Documents to rerank. Repeatable or space-separated. "
        "If a single filename is given and the file exists, reads documents from it."
    ),
)
parser.add_argument(
    "--disable_ailia_tokenizer",
    action="store_true",
    help="disable ailia tokenizer.",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Main functions
# ======================


def predict(models, queries, sentences):
    tokenizer = models["tokenizer"]
    net = models["net"]

    features = tokenizer(
        [[q, s] for q, s in zip(queries, sentences)],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )
    input_ids = features["input_ids"].astype(np.int64)
    attention_mask = features["attention_mask"].astype(np.int64)

    # feedforward
    if not args.onnx:
        output = net.predict([input_ids, attention_mask])
    else:
        output = net.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )
    scores = output[0].reshape(len(sentences))

    return scores


def recognize_from_sentence(models):
    # Build sentences
    if args.document is None:
        sentence_args = [DOCUMENT_PATH]
    else:
        sentence_args = [s for group in args.document for s in group]
    if len(sentence_args) == 1 and os.path.isfile(sentence_args[0]):
        with open(sentence_args[0], encoding="utf-8") as f:
            sentences = [s.strip() for s in f.read().split("\n") if s.strip()]
    else:
        sentences = sentence_args

    # Build queries
    single_query = isinstance(args.input, str)
    if single_query:
        queries = [args.input] * len(sentences)
        logger.info("query: " + args.input)
    else:
        queries = args.input

    # inference
    logger.info("Start inference...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            scores = predict(models, queries, sentences)
            end = int(round(time.time() * 1000))
            estimation_time = end - start
            logger.info(f"\tailia processing estimation time {estimation_time} ms")
            if i != 0:
                total_time_estimation += estimation_time
        logger.info(
            f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
        )
    else:
        scores = predict(models, queries, sentences)

    logger.info("The scores in order of higher are below.")
    if single_query:
        comb_score = sorted(zip(enumerate(sentences), scores), key=lambda x: -x[1])
        logger.info(
            "\n".join(
                f"\n[{i + 1}] ({score:.3f})\n- {sentence}"
                for (i, sentence), score in comb_score
            )
        )
    else:
        comb_score = sorted(
            zip(enumerate(zip(queries, sentences)), scores), key=lambda x: -x[1]
        )
        logger.info(
            "\n".join(
                f"\n[{i + 1}] ({score:.3f})\n- {q}\n- {s}"
                for (i, (q, s)), score in comb_score
            )
        )

    logger.info("Script finished successfully.")


def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(True, True, False, True)
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=memory_mode)
    else:
        import onnxruntime

        cuda = 0 < ailia.get_gpu_environment_id()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    if args.disable_ailia_tokenizer:
        #from transformers import AutoTokenizer
        #tokenizer = AutoTokenizer.from_pretrained("cl-nagoya/ruri-v3-reranker-310m") # リファレンス
        #tokenizer = AutoTokenizer.from_pretrained("tokenizer") # 不一致
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer") # 一致
    else:
        #from ailia_tokenizer import LlamaTokenizer
        #tokenizer = LlamaTokenizer.from_pretrained("./tokenizer") # BOSあり、EOSなし、不一致

        from ailia_tokenizer import GemmaTokenizer
        tokenizer = GemmaTokenizer.from_pretrained("./tokenizer") # BOSあり、EOSあり、一致

    models = {
        "net": net,
        "tokenizer": tokenizer,
    }

    recognize_from_sentence(models)


if __name__ == "__main__":
    main()
