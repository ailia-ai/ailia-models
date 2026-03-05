import sys

import ailia
import numpy as np


# import original modules
sys.path.append("../../util")
# logger
from logging import getLogger  # noqa

from arg_utils import get_base_parser, update_parser  # noqa
from model_utils import check_and_download_models  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "embeddinggemma-300m.onnx"
MODEL_PATH = "embeddinggemma-300m.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/embeddinggemma/"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("EmbeddingGemma", None, None, fp16_support = False)
parser.add_argument(
    "-q",
    "--query",
    type=str,
    default="Which planet is known as the Red Planet?",
    help="Query text for similarity search.",
)
parser.add_argument(
    "-d",
    "--documents",
    type=str,
    nargs="+",
    action="append",
    default=None,
    help=(
        "Documents to search against the query. Can specify multiple documents: "
        "--documents 'Doc 1' 'Doc 2' 'Doc 3'. "
        "If not specified, uses default planet-related documents for demonstration."
    ),
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalizes the embeddings matrix, so that each sentence embedding has unit length.
    """
    # Compute L2 norms for each row
    norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes the cosine similarity between two arrays.
    """
    # Add batch dimension if needed
    if a.ndim == 1:
        a = np.expand_dims(a, axis=0)
    if b.ndim == 1:
        b = np.expand_dims(b, axis=0)

    a_norm = normalize_embeddings(a)
    b_norm = normalize_embeddings(b)
    return np.dot(a_norm, b_norm.T)


# ======================
# Main functions
# ======================


def encode(models, sentences, prompt: str | None = None, batch_size: int = 32):
    input_was_string = False
    if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
        sentences = [sentences]
        input_was_string = True

    if prompt is not None and len(prompt) > 0:
        sentences = [prompt + sentence for sentence in sentences]

    tokenizer = models["tokenizer"]
    net = models["net"]

    all_embeddings = []
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[int(idx)] for idx in length_sorted_idx]

    for start_index in range(0, len(sentences), batch_size):
        sentences_batch = sentences_sorted[start_index : start_index + batch_size]

        encoded = tokenizer(
            sentences_batch,
            padding=True,
            max_length=2048,
            truncation=True,
            return_tensors="np",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

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

        all_embeddings.extend(embeddings)

    all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    all_embeddings = np.asarray([emb for emb in all_embeddings])

    if input_was_string:
        all_embeddings = all_embeddings[0]

    return all_embeddings


def recognize_from_text(models):
    query = args.query

    # Default documents
    default_documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]

    if args.documents is None:
        documents = default_documents
    else:
        # Flatten the list (since action="append" and nargs="+" create nested lists)
        documents = [doc for sublist in args.documents for doc in sublist]

    logger.info("Start inference...")

    query_embeddings = encode(models, query, prompt="task: search result | query: ")
    document_embeddings = encode(models, documents, prompt="title: none | text: ")

    similarities = similarity(query_embeddings, document_embeddings)

    # Display results
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80)

    # Get similarity scores and sort by score (descending)
    scores = similarities[0]  # Get first row since query is single
    ranked_indices = np.argsort(scores)[::-1]  # Sort descending

    print("\nSearch Results (ranked by similarity):\n")
    for rank, idx in enumerate(ranked_indices, 1):
        score = scores[idx]
        doc = documents[idx]
        print(f"Rank {rank}: Similarity = {score:.4f} ({score*100:.2f}%)")
        print(f"  Document: {doc}")
        print()

    print("=" * 80)

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    if args.disable_ailia_tokenizer:
        from transformers import GemmaTokenizerFast

        tokenizer = GemmaTokenizerFast.from_pretrained("./tokenizer/")
    else:
        from ailia_tokenizer import GemmaTokenizer

        tokenizer = GemmaTokenizer.from_pretrained("./tokenizer/")

    models = {
        "tokenizer": tokenizer,
        "net": net,
    }

    recognize_from_text(models)


if __name__ == "__main__":
    main()
