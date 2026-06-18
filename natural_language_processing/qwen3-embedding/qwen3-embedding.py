import os
import sys
import time
from logging import getLogger

import ailia
import numpy as np

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "Qwen3-Embedding-0.6B.onnx"
MODEL_PATH = "Qwen3-Embedding-0.6B.onnx.prototxt"
DATA_PATH = "Qwen3-Embedding-0.6B_weights.pb"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/qwen3-embedding/"

QUERY_DEFAULT = "What is the capital of China?"
DOCUMENT_PATH = "documents.txt"
TASK_DEFAULT = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("Qwen3-Embedding", QUERY_DEFAULT, None, fp16_support=False)
parser.add_argument(
    "-d",
    "--document",
    action="append",
    nargs="+",
    default=None,
    help=(
        "Documents to search. Repeatable or space-separated. "
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
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Secondary Functions
# ======================


def get_query_text(task, query):
    return f"Instruct: {task}\nQuery:{query}"


# ======================
# Main functions
# ======================


PAD_TOKEN_ID = 151643  # <|endoftext|>
EOS_TOKEN_ID = 151643  # Qwen3 uses <|endoftext|> as EOS


def append_eos_and_left_pad(input_ids, attention_mask, pad_token_id, eos_token_id):
    """Append EOS to each sequence, then left-pad to the batch max."""
    batch_size, seq_len = input_ids.shape
    new_len = seq_len + 1
    new_input_ids = np.full((batch_size, new_len), pad_token_id, dtype=input_ids.dtype)
    new_attention_mask = np.zeros((batch_size, new_len), dtype=attention_mask.dtype)
    for i in range(batch_size):
        valid_len = int(attention_mask[i].sum())
        out_len = valid_len + 1
        start = new_len - out_len
        new_input_ids[i, start:start + valid_len] = input_ids[i, :valid_len]
        new_input_ids[i, start + valid_len] = eos_token_id
        new_attention_mask[i, start:] = 1
    return new_input_ids, new_attention_mask


def predict(models, texts):
    tokenizer = models["tokenizer"]
    batch_dict = tokenizer(
        texts, max_length=8192, padding=True, truncation=True, return_tensors="np"
    )

    input_ids = batch_dict["input_ids"].astype(np.int64)
    attention_mask = batch_dict["attention_mask"].astype(np.int64)

    if not args.disable_ailia_tokenizer:
        # ailia_tokenizer does not append EOS and right-pads with GPT2 EOS(50256).
        # Qwen3-Embedding expects an EOS appended and left-padding with <|endoftext|>.
        input_ids, attention_mask = append_eos_and_left_pad(
            input_ids, attention_mask, PAD_TOKEN_ID, EOS_TOKEN_ID
        )

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
    embeddings = output[0]

    return embeddings


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

    query_texts = [get_query_text(task, q) for q in queries]

    # inference
    logger.info("Generating embeddings...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        total_time = 0
        for _ in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            all_embs = predict(models, query_texts + documents)
            end = int(round(time.time() * 1000))
            logger.info(f"\tailia processing time {end - start} ms")
            if _ != 0:
                total_time += end - start
        logger.info(f"average time {total_time / (args.benchmark_count - 1)} ms\n")
        return
    else:
        all_embs = predict(models, query_texts + documents)

    query_embs = all_embs[: len(queries)]
    doc_embs = all_embs[len(queries) :]

    # compute cosine similarity (embeddings are already L2-normalized by the model)
    scores = query_embs @ doc_embs.T  # (num_queries, num_docs)

    logger.info("The documents in order of similarity are below.")
    if single_query:
        comb_score = sorted(zip(enumerate(documents), scores[0]), key=lambda x: -x[1])
        logger.info(
            "\n".join(
                f"\n[{i + 1}] ({score:.3f})\n- {doc}" for (i, doc), score in comb_score
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
                f"\n[{qi + 1}-{di + 1}] ({sim:.3f})\n- {q}\n- {doc}"
                for qi, di, q, doc, sim in comb_score
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
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("tokenizer", padding_side="left")
    else:
        from ailia_tokenizer import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer")
        tokenizer.add_special_tokens({"additional_special_tokens":['<|end_of_text|>', '<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>']})

    models = {
        "net": net,
        "tokenizer": tokenizer,
    }

    recognize_from_sentence(models)


if __name__ == "__main__":
    main()
