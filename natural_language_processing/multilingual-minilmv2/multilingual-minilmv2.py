import time
import sys
import re
import numpy as np

import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from math_utils import softmax, sigmoid  # noqa: E402
from classifier_utils import print_results  # noqa: E402

# logger
from logging import getLogger  # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters
# ======================
SENTENCE = "今日、新しいiPhoneが発売されました"
CANDIDATE_LABELS = "スマートフォン, エンタメ, スポーツ"
HYPOTHESIS_TEMPLATE = "This example is {}."

MODEL_LISTS = ['minilm_l6', 'minilm_l12']

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('multilingual-minilmv2', None, None)
parser.add_argument(
    '--sentence', '-s', metavar='TEXT', default=SENTENCE,
    help='input sentence'
)
parser.add_argument(
    '--candidate_labels', '-c', metavar='TEXT', default=CANDIDATE_LABELS,
    help='input labels separated by , '
)
parser.add_argument(
    '--hypothesis_template', '-t', metavar='TEXT', default=HYPOTHESIS_TEMPLATE,
    help='input hypothesis template'
)
parser.add_argument(
    '--arch', '-a', metavar='ARCH', default='minilm_l12', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '--multi_label',
    action='store_true',
    help='allow assigning multiple labels'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser, check_input_type=False)

# ======================
# Model Parameters
# ======================
WEIGHT_PATH = args.arch + ".onnx"
MODEL_PATH = WEIGHT_PATH + ".prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/multilingual-minilmv2/"

# ======================
# Utils
# ======================
def preprocess(sequences, labels, hypothesis_template):
    sequences = [sequences]
    sequence_pairs = []
    for sequence in sequences:
        sequence_pairs.extend(
            [[sequence, hypothesis_template.format(label)] for label in labels]
        )
    return sequence_pairs

def calc_multilabel_zero_shot_score(logits):
    entailment_logits = logits[:, 0]
    contradiction_logits = logits[:, 2]
    return sigmoid(entailment_logits - contradiction_logits)

def calc_singlelabel_zero_shot_score(logits):
    entailment_logits = logits[:, 0]
    return softmax(entailment_logits)

# ======================
# Main function
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    ailia_model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    candidate_labels = re.split(r'\s*,\s*', args.candidate_labels) # Delete spaces before and after commas

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli')
    else:
        from ailia_tokenizer import XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained('tokenizer/')

    tokenizer_input_pairs = preprocess(
        args.sentence, candidate_labels, args.hypothesis_template
    )

    model_inputs = tokenizer(
        tokenizer_input_pairs,
        add_special_tokens=True,
        return_tensors="np",
        padding=True,
        truncation="only_first",
        max_length=128
    )

    inputs_onnx = {k: np.array(v) for k, v in model_inputs.items()}

    logger.info("Sentence : " + str(args.sentence))
    logger.info("Candidate Labels : " + str(args.candidate_labels))
    logger.info("Hypothesis Template : " + str(args.hypothesis_template))

    # inference
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            score = ailia_model.predict(inputs_onnx)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end - start))
    else:
        score = ailia_model.predict(inputs_onnx)

    # calculate score
    score = score[0]
    if args.multi_label:
        scores = calc_multilabel_zero_shot_score(score)
    else:
        scores = calc_singlelabel_zero_shot_score(score)

    # show score
    print_results([scores], candidate_labels)
    logger.info('Script finished successfully.')


if __name__ == "__main__":
    main()
