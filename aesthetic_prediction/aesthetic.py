import os
import sys
import time
import argparse
import numpy as np
import cv2
import ailia
from PIL import Image

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

WEIGHT_AVA_PATH = 'sac+logos+ava1-l14-linearMSE.onnx'
MODEL_AVA_PATH = 'sac+logos+ava1-l14-linearMSE.onnx.prototxt'
REMOTE_AVA_PATH = 'https://storage.googleapis.com/ailia-models/aesthetic-predictor-v2/'

WEIGHT_VITL14_PATH = 'ViT-L14-encode_image.onnx'
MODEL_VITL14_PATH = 'ViT-L14-encode_image.onnx.prototxt'
REMOTE_VITL14_PATH = 'https://storage.googleapis.com/ailia-models/clip/'

IMAGE_PATH = 'test.jpg'
IMAGE_SIZE = 224


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

def preprocess(image):
    h, w = (IMAGE_SIZE, IMAGE_SIZE)
    im_h, im_w, _ = image.shape

    # resize
    scale = h / min(im_h, im_w)
    ow, oh = round(im_w * scale), round(im_h * scale)
    if ow != im_w or oh != im_h:
        image = np.array(Image.fromarray(image).resize((ow, oh), Image.BICUBIC))

    # center_crop
    if ow > w:
        x = (ow - w) // 2
        image = image[:, x:x + w, :]
    if oh > h:
        y = (oh - h) // 2
        image = image[y:y + h, :, :]

    image = image[:, :, ::-1]  # BGR -> RBG
    image = image / 255

    mean = np.array((0.48145466, 0.4578275, 0.40821073))
    std = np.array((0.26862954, 0.26130258, 0.27577711))
    image = (image - mean) / std

    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    return image


# ======================
# Main functions
# ======================

def predict(image, net_ava, net_vitl14):
    image = preprocess(image)

    features = net_vitl14.predict({'image': image})[0]
    features = features / np.linalg.norm(features, ord=2, axis=-1, keepdims=True)

    score = net_ava.predict({'x': features})[0][0][0]
    return score


def recognize_from_image(filename, net_ava, net_vitl14):
    for image_path in args.input:
        logger.info(image_path)

        image = load_image(image_path)
        logger.info(f'input image shape: {image.shape}')

        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                score = predict(image, net_ava, net_vitl14)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            score = predict(image, net_ava, net_vitl14)

    logger.info('### Score ###')
    logger.info(score)

    logger.info('Script finished successfully.')


def main():
    check_and_download_models(WEIGHT_AVA_PATH, MODEL_AVA_PATH, REMOTE_AVA_PATH)
    check_and_download_models(WEIGHT_VITL14_PATH, MODEL_VITL14_PATH, REMOTE_VITL14_PATH)

    memory_mode = ailia.get_memory_mode(reduce_constant=True, reduce_interstage=True)

    net_ava = ailia.Net(MODEL_AVA_PATH, WEIGHT_AVA_PATH, env_id=args.env_id, memory_mode=memory_mode)
    net_vitl14 = ailia.Net(MODEL_VITL14_PATH, WEIGHT_VITL14_PATH, env_id=args.env_id, memory_mode=memory_mode)

    recognize_from_image(args.input, net_ava, net_vitl14)


if __name__ == '__main__':
    main()
