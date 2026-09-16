import time
import sys

import numpy

import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models, check_and_download_file  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Arguemnt Parser Config
# ======================

QUERY = 'How many people live in Berlin?'
PARAGRAPH= 'Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.'


parser = get_base_parser('cross encoder mmarco.', None, None)
parser.add_argument(
    '--query', '-q', metavar='TEXT', default=QUERY,
    help='input query'
)
parser.add_argument(
    '--paragraph', '-p', metavar='TEXT', default=PARAGRAPH,
    help='input paragraph'
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

WEIGHT_PATH = "mmarco-mMiniLMv2-L12-H384-v1.onnx"
MODEL_PATH = "mmarco-mMiniLMv2-L12-H384-v1.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/cross_encoder_mmarco/"


# ======================
# Main function
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    ailia_model = ailia.Net(MODEL_PATH, WEIGHT_PATH, memory_mode=args.memory_mode)

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("jeffwan/mmarco-mMiniLMv2-L12-H384-v1")
    else:
        from ailia_tokenizer import XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained("./tokenizer/")

    model_inputs = tokenizer([args.query], [args.paragraph],  padding=True, truncation=True, max_length = 512, return_tensors="np")
    inputs_onnx = {
        k: v for k, v in model_inputs.items()
    }

    logger.info("Input : " + str(args.query))

    # inference
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            logits = ailia_model.predict(inputs_onnx)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end - start))
    else:
        logits = ailia_model.predict(inputs_onnx)

    logger.info("Output : " + str(logits))
    logger.info('Script finished successfully.')


if __name__ == "__main__":
    main()
