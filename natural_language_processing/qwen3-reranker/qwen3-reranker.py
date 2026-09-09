import os
import sys
import time
from logging import getLogger

import ailia
import numpy as np

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa
from model_utils import check_and_download_file, check_and_download_models  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "Qwen3-Reranker-0.6B.onnx"
MODEL_PATH = "Qwen3-Reranker-0.6B.onnx.prototxt"
DATA_PATH = "Qwen3-Reranker-0.6B_weights.pb"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/qwen3-reranker/"

QUERY_DEFAULT = "What is the capital of China?"
DOCUMENT_PATH = "documents.txt"
TASK_DEFAULT = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("Qwen3-Reranker", QUERY_DEFAULT, None, fp16_support=False)
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
    "-t",
    "--task",
    metavar="TASK",
    default=TASK_DEFAULT,
    help="Task description for query instruction.",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Secondary Functions
# ======================

PAD_TOKEN_ID = 151643  # <|endoftext|>
MAX_LENGTH = 8192

# The reranker is a causal LM that answers "yes" or "no". The chat prompt below is
# identical to chat_template.jinja in the tokenizer directory.
PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_instruction(instruction, query, doc):
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )


def encode(tokenizer, text):
    """Encode text into a flat list of token ids (no special tokens added)."""
    if args.disable_ailia_tokenizer:
        return tokenizer.encode(text, add_special_tokens=False)
    return tokenizer(text)["input_ids"]


def left_pad(sequences, pad_token_id):
    """Left-pad token id sequences to the batch max.

    The model reads the score from the last position, so the prompt must end at the
    right edge of the tensor. ailia tokenizer right-pads with the GPT2 EOS(50256),
    which does not work here, so the padding is built here instead.
    """
    max_len = max(len(s) for s in sequences)
    input_ids = np.full((len(sequences), max_len), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        input_ids[i, max_len - len(seq) :] = seq
        attention_mask[i, max_len - len(seq) :] = 1
    return input_ids, attention_mask


# ======================
# Main functions
# ======================


def predict(models, pairs):
    tokenizer = models["tokenizer"]
    prefix_tokens = models["prefix_tokens"]
    suffix_tokens = models["suffix_tokens"]

    body_max_length = MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens)
    sequences = [
        prefix_tokens + encode(tokenizer, text)[:body_max_length] + suffix_tokens
        for text in pairs
    ]
    input_ids, attention_mask = left_pad(sequences, PAD_TOKEN_ID)

    net = models["net"]

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
    scores = output[0]

    return scores


def recognize_from_sentence(models):
    task = args.task

    # Build documents
    if args.document is None:
        document_args = [DOCUMENT_PATH]
    else:
        document_args = [d for group in args.document for d in group]
    if len(document_args) == 1 and os.path.isfile(document_args[0]):
        with open(document_args[0], encoding="utf-8") as f:
            documents = [line.strip() for line in f if line.strip()]
    else:
        documents = document_args

    # Build queries
    single_query = isinstance(args.input, str)
    if single_query:
        queries = [args.input]
        logger.info("query: " + args.input)
    else:
        queries = args.input

    pairs = [
        format_instruction(task, query, doc) for query in queries for doc in documents
    ]

    # inference
    logger.info("Scoring documents...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        total_time = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            scores = predict(models, pairs)
            end = int(round(time.time() * 1000))
            logger.info(f"\tailia processing time {end - start} ms")
            if i != 0:
                total_time += end - start
        logger.info(f"average time {total_time / (args.benchmark_count - 1)} ms\n")
        return
    else:
        scores = predict(models, pairs)

    scores = scores.reshape(len(queries), len(documents))

    logger.info("The documents in order of relevance are below.")
    if single_query:
        comb_score = sorted(zip(enumerate(documents), scores[0]), key=lambda x: -x[1])
        logger.info(
            "\n".join(
                f"\n[{i + 1}] ({score:.6f})\n- {doc}" for (i, doc), score in comb_score
            )
        )
    else:
        # all combinations of queries × documents, sorted by score
        all_pairs = [
            (qi, di, queries[qi], documents[di], scores[qi, di])
            for qi in range(len(queries))
            for di in range(len(documents))
        ]
        comb_score = sorted(all_pairs, key=lambda x: -x[4])
        logger.info(
            "\n".join(
                f"\n[{qi + 1}-{di + 1}] ({score:.6f})\n- {q}\n- {doc}"
                for qi, di, q, doc, score in comb_score
            )
        )

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_file(DATA_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
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
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("tokenizer", padding_side="left")
    else:
        from ailia_tokenizer import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer")

    models = {
        "net": net,
        "tokenizer": tokenizer,
        "prefix_tokens": encode(tokenizer, PREFIX),
        "suffix_tokens": encode(tokenizer, SUFFIX),
    }

    recognize_from_sentence(models)


if __name__ == "__main__":
    main()
