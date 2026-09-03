import os
import sys
import time
import argparse
import numpy as np
import cv2
import ailia

sys.path.append('../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from webcamera_utils import get_capture  # noqa: E402

from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH = 'sac+logos+ava1-l14-linearMSE.onnx'
MODEL_PATH = 'sac+logos+ava1-l14-linearMSE.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/aesthetic_prediction/'
IMAGE_PATH = 'test.jpg'


# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'Aesthetic Score Predictor',
    IMAGE_PATH,
    None,
)
args = update_parser(parser)


# ======================
# Helper functions
# ======================

def preprocess(img, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param img: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = img.shape

    resize_w = w
    resize_h = h

    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    img = cv2.resize(img, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img, (ratio_h, ratio_w)


# ======================
# Main functions
# ======================

def predict(img, net):
    img, (ratio_h, ratio_w) = preprocess(img)

    net.set_input_shape(img.shape)
    output = net.predict({'x': img})
    print(output)


def recognize_from_image(filename, net):
    for image_path in args.input:
        logger.info(image_path)

        img = load_image(image_path)
        logger.info(f'input image shape: {img.shape}')

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                score = predict(img, net)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            score = predict(img, net)

    logger.info('Script finished successfully.')


def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    memory_mode = ailia.get_memory_mode(reduce_constant=True, reduce_interstage=True)
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=memory_mode)

    recognize_from_image(args.input, net)


if __name__ == '__main__':
    main()
