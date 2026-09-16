import sys
import time
import json
import os
import shutil

from logging import getLogger

import numpy as np

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa

from token_classification import TokenClassification

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "model.onnx"
MODEL_PATH = "model.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_ner_japanese/"
REMOTE_DIC_PATH = "https://storage.googleapis.com/ailia-models/bert_maskedlm/"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("bert-ner-japanese ", None, None)
parser.add_argument(
    "-i",
    "--input",
    metavar="TEXT",
    type=str,
    default="株式会社Jurabiは、東京都台東区に本社を置くIT企業である。",
    help="Input sentense.",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument(
    "--disable_aggregation", action="store_true", help="disable aggregation."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Main functions
# ======================

def create_special_tokens_mask(input_ids, special_token_ids):
    # Initialize the special tokens mask with zeros
    special_tokens_mask = np.zeros_like(input_ids, dtype=int)
    
    for i, sequence in enumerate(input_ids):
        for j, token_id in enumerate(sequence):
            if token_id in special_token_ids:
                special_tokens_mask[i][j] = 1
    
    return special_tokens_mask

def preprocess(tokenizer, sentence):
    if args.disable_ailia_tokenizer:
        inputs = tokenizer(
            sentence,
            return_tensors="np",
            truncation=True,
            # max_length=512,
            return_special_tokens_mask=True,
            return_offsets_mapping=False,
        )
    else:
        inputs = tokenizer(
            sentence,
            return_tensors="np",
            truncation=True,
            # max_length=512,
            #return_special_tokens_mask=True,
            #return_offsets_mapping=False,
        )

    num_chunks = len(inputs["input_ids"])
    for i in range(num_chunks):
        model_inputs = {k: np.expand_dims(v[i], axis=0) for k, v in inputs.items()}
        # model_inputs["sentence"] = sentence if i == 0 else None
        # model_inputs["is_last"] = i == num_chunks - 1

        if not args.disable_ailia_tokenizer:
            model_inputs["special_tokens_mask"] = create_special_tokens_mask(model_inputs["input_ids"], [2, 3])

        yield model_inputs


def post_processing(tokenizer, all_outputs):
    aggregation = not args.disable_aggregation

    clf = TokenClassification(tokenizer, args.disable_ailia_tokenizer)
    ignore_labels = ["O"]
    all_entities = []
    for model_outputs in all_outputs:
        logits = model_outputs["logits"][0]
        input_ids = model_outputs["input_ids"][0]
        special_tokens_mask = model_outputs["special_tokens_mask"][0]

        maxes = np.max(logits, axis=-1, keepdims=True)
        shifted_exp = np.exp(logits - maxes)
        scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

        pre_entities = clf.gather_pre_entities(
            input_ids,
            scores,
            special_tokens_mask,
        )
        grouped_entities = clf.aggregate(pre_entities, aggregation=aggregation)

        # Filter anything that is in self.ignore_labels
        entities = [
            entity
            for entity in grouped_entities
            if entity.get("entity", None) not in ignore_labels
            and entity.get("entity_group", None) not in ignore_labels
        ]
        all_entities.extend(entities)

    num_chunks = len(all_outputs)
    if 1 < num_chunks:
        all_entities = clf.aggregate_overlapping_entities(all_entities)

    return all_entities


def predict(models, sentence):
    tokenizer = models["tokenizer"]
    net = models["net"]

    processed = preprocess(tokenizer, sentence)

    accumulator = []
    for item in processed:
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        token_type_ids = item["token_type_ids"]

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
        logits = output[0]

        item["logits"] = logits
        accumulator.append(item)

    entities = post_processing(tokenizer, accumulator)

    return entities


def recognize_from_text(models):
    sentence = args.input

    logger.info("input_text: %s" % sentence)

    # inference
    logger.info("Start inference...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            entities = predict(models, sentence)
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
        entities = predict(models, sentence)

    print("NER:\n%s" % json.dumps(entities, indent=2, ensure_ascii=False))

    logger.info("Script finished successfully.")


def main():
    # model files check and download
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
        check_and_download_file("unidic-lite.zip", REMOTE_DIC_PATH)
        if not os.path.exists("unidic-lite"):
            shutil.unpack_archive('unidic-lite.zip', '')
        tokenizer = BertJapaneseWordPieceTokenizer.from_pretrained(dict_path = 'unidic-lite', pretrained_model_name_or_path = './tokenizer/')

    models = {
        "net": net,
        "tokenizer": tokenizer,
    }

    recognize_from_text(models)


if __name__ == "__main__":
    main()
