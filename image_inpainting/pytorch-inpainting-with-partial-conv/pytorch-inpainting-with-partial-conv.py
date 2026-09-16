import os
import sys
import time
import glob
import random

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import normalize_image  # noqa: E402
from detector_utils import load_image  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'partialconv.onnx'
MODEL_PATH = 'partialconv.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/pytorch-inpainting-with-partial-conv/'

IMAGE_PATH = 'Places365_test_00000146.jpg'
SAVE_IMAGE_PATH = 'result.png'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('pytorch-inpainting-with-partial-conv model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-mi', '--mask-index', type=int, metavar='INDEX', default=None,
    help='Mask index. If not specified, it will be randomly selected.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def postprocess(x):
    mean = np.array(NORM_MEAN)
    std = np.array(NORM_STD)
    x = x.transpose(1, 2, 0)  # CHW -> HWC
    x = x * std + mean
    x = x * 255
    x = x[:, :, ::-1]  # RGB -> BGR
    return x


def recognize_from_image(net):
    mask_paths = sorted(glob.glob('masks/*.jpg'))
    N_mask = len(mask_paths)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare grand truth
        gt_img = load_image(image_path)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGRA2RGB)
        gt_img = np.array(Image.fromarray(gt_img).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BILINEAR))
        gt_img = normalize_image(gt_img, 'ImageNet')
        gt_img = gt_img.transpose((2, 0, 1))  # channel first

        # prepare mask
        if args.mask_index is not None:
            mask_path = mask_paths[args.mask_index % N_mask]
        else:
            mask_path = mask_paths[random.randint(0, N_mask - 1)]
        mask = load_image(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGB)
        mask = np.array(Image.fromarray(mask).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BILINEAR))
        mask = mask.transpose((2, 0, 1)) / 255  # channel first

        # prepare input data
        img = gt_img * mask
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)
        gt_img = np.expand_dims(gt_img, axis=0)

        logger.debug(f'input image shape: {img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = net.predict({'image': img, 'mask': mask})
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            output = net.predict({'image': img, 'mask': mask})

        output, _ = output

        img = postprocess(img[0])
        mask = mask[0].transpose(1, 2, 0) * 255
        output = postprocess(output[0])
        gt_img = postprocess(gt_img[0])
        res_img = np.hstack((img, mask, output, gt_img))

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    recognize_from_image(net)


if __name__ == '__main__':
    main()
