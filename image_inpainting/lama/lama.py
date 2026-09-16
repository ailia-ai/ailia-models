import sys
import time
import os
import platform

import cv2
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import imread  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'lama.onnx'
MODEL_PATH  = 'lama.onnx.prototxt'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/lama/'

IMAGE_PATH = '000068.png'
MASK_PATH = '000068_mask.png'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('Lama model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-m', '--mask', nargs='*', metavar='MASK_PATH', default=[MASK_PATH],
    help='using mask image from mask path'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


def preprocess(image):
    pad_out_to_modulo = 8
    image = pad_img_to_modulo(image, pad_out_to_modulo)
    return image


def recognize_from_image(net):
    # input image loop
    for image_path , mask_path in zip(args.input,args.mask):
        # prepare ground truth
        image = imread(image_path).astype(np.float32)/255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        image = preprocess(image).astype(np.float32)

        # prepare mask
        mask  = imread(mask_path ,cv2.IMREAD_GRAYSCALE)[None, ...] / 255
        mask  = preprocess(mask).astype(np.float32)
        mask = (mask > 0) * 1

        # prepare input data
        logger.debug(f'input data shape: {image.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))

                results = net.run((np.expand_dims(image,0),
                                   np.expand_dims(mask ,0)))

                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            results = net.run((np.expand_dims(image,0),
                               np.expand_dims(mask ,0)))

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')

        results = np.array(results[0][0])
        results = np.transpose(results,(1,2,0)).astype("uint8")

        res_img = cv2.cvtColor(results, cv2.COLOR_RGB2BGR)

        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')

def main():

    if "FP16" in ailia.get_environment(args.env_id).props or platform.system() == 'Darwin':
        logger.warning('This model do not work on FP16. So use CPU mode.')
        args.env_id = 0

    # model files check and download
    check_and_download_models(MODEL_PATH, WEIGHT_PATH, REMOTE_PATH)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH,memory_mode=args.memory_mode, env_id=args.env_id)

    recognize_from_image(net)


if __name__ == '__main__':
    main()
