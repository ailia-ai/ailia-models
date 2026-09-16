import sys
import time
import math
from logging import getLogger

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser
from model_utils import check_and_download_models
from image_utils import normalize_image
from detector_utils import load_image

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'ndlenfixed64-mj0-synth1.onnx'
MODEL_PATH = 'ndlenfixed64-mj0-synth1.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/ndlocr_text_recognition/'

IMAGE_PATH = 'demo.png'
CHAR_FILE_PATH = 'mojilist_NDL.txt'

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 1200

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'NDL OCR', IMAGE_PATH, None
)
parser.add_argument(
    '--vert',
    action='store_true',
    help='treated as vertical text.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def get_char_list():
    with open(CHAR_FILE_PATH, encoding='utf-8') as f:
        char_list = f.read()
    char_list = '〓' + char_list.replace("\n", "")
    char_list = ['[CTCblank]'] + list(char_list)

    return char_list


# ======================
# Main functions
# ======================

def preprocess(img):
    im_h, im_w = img.shape

    if args.vert:
        vert = '縦'
    elif im_w > im_h * 5:
        vert = '横'
    elif im_h > im_w * 5:
        vert = '縦'
    else:
        vert = None

    if vert == '縦':
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        im_h, im_w = img.shape

    # keep ratio with pad
    ratio = im_w / im_h
    if math.ceil(IMAGE_HEIGHT * ratio) > IMAGE_WIDTH:
        resized_w = IMAGE_WIDTH
    else:
        resized_w = math.ceil(IMAGE_HEIGHT * ratio)
    img = np.pad(img, ((0, 0), (10, 10)), constant_values=255)
    img = np.array(Image.fromarray(img).resize((resized_w, IMAGE_HEIGHT), Image.Resampling.BICUBIC))

    img = normalize_image(img, normalize_type='127.5')

    pad_img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    pad_img[:, :resized_w] = img  # right pad
    img = pad_img

    img = np.expand_dims(img, axis=0)  # CHW
    img = np.expand_dims(img, axis=0)  # NCWH
    img = img.astype(np.float32)

    return img


def decode(pred):
    characters = get_char_list()
    t = np.argmax(pred, axis=1)

    char_list = []
    for i in range(len(pred)):
        if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
            char_list.append(characters[t[i]])

    text = ''.join(char_list)

    return text


def predict(net, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'img': img})
    preds = output[0]

    text = decode(preds[0])

    return text


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                text = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            text = predict(net, img)

        if text:
            logger.info(" recognized: %s" % text)
        else:
            logger.info(" text not recognized")

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    recognize_from_image(net)


if __name__ == '__main__':
    main()
