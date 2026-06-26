import time
import sys

from utils_gpt2 import *
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

DEFAULT_TEXT = 'My name is Clara and I am'

parser = get_base_parser('gpt2 text generation', None, None)
# overwrite
parser.add_argument(
    '--input', '-i', default=DEFAULT_TEXT
)
parser.add_argument(
    '--outlength', '-o', default=30
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
parser.add_argument(
    '--seed', default=None,
    help='random seed'
)
parser.add_argument(
    '--temperature', default=1.0,
    help='randomness control factor'
)
args = update_parser(parser, check_input_type=False)


ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider",]

# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = "gpt2-medium.opt.onnx"
MODEL_PATH = "gpt2-medium.opt.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/gpt2/"


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
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    else:
        from ailia_tokenizer import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer/")
    logger.info("Input : "+args.input)

    # inference
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            output = generate_text(tokenizer, ailia_model, args.input, int(args.outlength), args.onnx, seed=args.seed, temperature=args.temperature)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end - start))
    else:
        output = generate_text(tokenizer, ailia_model, args.input, int(args.outlength), args.onnx, seed=args.seed, temperature=args.temperature)

    logger.info("output : "+output)
    logger.info('Script finished successfully.')


if __name__ == "__main__":
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    main()
