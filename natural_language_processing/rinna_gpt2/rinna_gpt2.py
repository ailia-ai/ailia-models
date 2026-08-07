import time
import sys
import os
import numpy

from utils_rinna_gpt2 import *
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

DEFAULT_TEXT = '生命、宇宙、そして万物についての究極の疑問の答えは'

parser = get_base_parser('rinna-gpt2 text generation', None, None)
# overwrite
parser.add_argument(
    '--input', '-i', default=DEFAULT_TEXT
)
parser.add_argument(
    '--outlength', '-o', default=50
)
parser.add_argument(
    '--top_k', type=int, default=50,
    help='number of highest probability tokens to keep for sampling'
)
parser.add_argument(
    '--top_p', type=float, default=0.95,
    help='cumulative probability threshold for nucleus sampling'
)
parser.add_argument(
    '--temperature', type=float, default=1.0,
    help='softmax temperature applied to the logits before sampling'
)
parser.add_argument(
    '--seed', type=int, default=42,
    help='random seed for sampling'
)
parser.add_argument(
    '--greedy', action='store_true',
    help='always pick the most probable token instead of sampling'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='By default, the ailia SDK is used, but with this option, you can switch to using ONNX Runtime'
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
WEIGHT_PATH = "japanese-gpt2-small.opt.onnx"
MODEL_PATH = "japanese-gpt2-small.opt.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/rinna_gpt2/"


# ======================
# Main function
# ======================
def main():
    if args.onnx:
        import onnxruntime
        ailia_model = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        logger.info("This model requires multiple input shape, so running on CPU")
        ailia_model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=0)#args.env_id)
    if args.disable_ailia_tokenizer:
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-small")
    else:
        from ailia_tokenizer import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("./tokenizer/")
    logger.info("Input : "+args.input)

    sampling_args = dict(
        greedy=args.greedy, top_k=args.top_k, top_p=args.top_p,
        temperature=args.temperature, seed=args.seed
    )

    # inference
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            output = generate_text(tokenizer, ailia_model, args.input, int(args.outlength), args.onnx, **sampling_args)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end - start))
    else:
        output = generate_text(tokenizer, ailia_model, args.input, int(args.outlength), args.onnx, **sampling_args)

    logger.info("output : "+output)
    logger.info('Script finished successfully.')


if __name__ == "__main__":
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    main()
