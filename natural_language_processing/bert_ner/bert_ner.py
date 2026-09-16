import time
import sys

import numpy

import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Arguemnt Parser Config
# ======================

SENTENCE = 'My name is bert'


parser = get_base_parser('bert ner.', None, None)
# overwrite
parser.add_argument(
    '--input', '-i', metavar='TEXT', default=SENTENCE,
    help='input text'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser, check_input_type=False)


# ======================
# PARAMETERS
# ======================

WEIGHT_PATH = "bert-large-cased-finetuned-conll03-english.onnx"
MODEL_PATH = "bert-large-cased-finetuned-conll03-english.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_ner/"


# ======================
# Main function
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    ailia_model = ailia.Net(MODEL_PATH, WEIGHT_PATH, memory_mode=args.memory_mode)
    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            'dbmdz/bert-large-cased-finetuned-conll03-english'
        )
    else:
        from ailia_tokenizer import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(
            './tokenizer/'
        )
    model_inputs = tokenizer.encode_plus(args.input, return_tensors="np")
    inputs_onnx = {
        k: v for k, v in model_inputs.items()
    }

    logger.info("Input : " + str(args.input))

    # inference
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            entities = ailia_model.predict(inputs_onnx)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end - start))
    else:
        entities = ailia_model.predict(inputs_onnx)

    id2label = {
        0: 'O',
        1: 'B-MISC',
        2: 'I-MISC',
        3: 'B-PER',
        4: 'I-PER',
        5: 'B-ORG',
        6: 'I-ORG',
        7: 'B-LOC',
        8: 'I-LOC',
    }
    ignore_labels = ['O']

    score = numpy.exp(entities) / numpy.exp(entities).sum(-1, keepdims=True)
    labels_idx = score.argmax(axis=-1)
    labels_idx = labels_idx[0][0]

    entities = []

    filtered_labels_idx = [
        (idx, label_idx)
        for idx, label_idx in enumerate(labels_idx)
        if id2label[label_idx] not in ignore_labels
    ]

    for idx, label_idx in filtered_labels_idx:
        entity = {
            "word": tokenizer.convert_ids_to_tokens(
                int(model_inputs.input_ids[0][idx])
            ),
            "score": score[0][0][idx][label_idx].item(),
            "entity": id2label[label_idx],
            "index": idx,
        }

        entities += [entity]

    logger.info("Output : " + str(entities))
    logger.info('Script finished successfully.')


if __name__ == "__main__":
    main()
