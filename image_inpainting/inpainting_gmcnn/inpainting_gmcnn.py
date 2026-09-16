import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

from inpainting_gmcnn_utils import *

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PARIS_STREETVIEW_PATH = 'paris-streetview_256x256_rect.onnx'
MODEL_PARIS_STREETVIEW_PATH = 'paris-streetview_256x256_rect.onnx.prototxt'
WEIGHT_CELEBAHQ_256_PATH = 'celebahq_256x256_rect.onnx'
MODEL_CELEBAHQ_256_PATH = 'celebahq_256x256_rect.onnx.prototxt'
WEIGHT_CELEBAHQ_512_PATH = 'celebahq_512x512_rect.onnx'
MODEL_CELEBAHQ_512_PATH = 'celebahq_512x512_rect.onnx.prototxt'
WEIGHT_CELEBAHQ_FREEFORM_PATH = 'celebahq_512x512_freeform.onnx'
MODEL_CELEBAHQ_FREEFORM_PATH = 'celebahq_512x512_freeform.onnx.prototxt'
WEIGHT_PLACE2_PATH = 'places2_512x680_freeform.onnx'
MODEL_PLACE2_PATH = 'places2_512x680_freeform.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/inpainting_gmcnn/'

IMAGE_PATH = 'paris-streetview_001.png'
SAVE_IMAGE_PATH = 'result.png'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('inpainting_gmcnn model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-m', '--model', default="paris-streetview", choices=("paris-streetview", "celebahq", "celebahq-512", "places2"),
    help='mask type'
)
parser.add_argument(
    '-mt', '--mask_type', default="rect", choices=("rect", "stroke"),
    help='mask type'
)
parser.add_argument(
    '--seed', type=int, default=1,
    help='random seed for random mask'
)
parser.add_argument(
    '--random_mask', type=int, default=0,
    help='using random rect mask'
)
parser.add_argument(
    '--mask_shape', type=str, default='128,128',
    help='given mask parameters: h,w'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img, img_shape):
    h, w = img.shape[:2]

    if h >= img_shape[0] and w >= img_shape[1]:
        h_start = (h - img_shape[0]) // 2
        w_start = (w - img_shape[1]) // 2
        img = img[h_start: h_start + img_shape[0], w_start: w_start + img_shape[1], :]
    else:
        t = min(h, w)
        img = img[(h - t) // 2:(h - t) // 2 + t, (w - t) // 2:(w - t) // 2 + t, :]
        img = cv2.resize(img, (img_shape[1], img_shape[0]))

    return img


def recognize_from_image(net, img_shape):
    np.random.seed(args.seed)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare grand truth
        gt_img = load_image(image_path)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGRA2BGR)
        gt_img = preprocess(gt_img, img_shape)

        # prepare mask
        if args.mask_type == 'rect':
            str_mask_shape = args.mask_shape.split(',')
            mask_shape = [int(x) for x in str_mask_shape]
            mask = generate_mask_rect(img_shape, mask_shape, args.random_mask)
        else:
            mask = generate_mask_stroke(
                im_size=(img_shape[0], img_shape[1]),
                parts=8, maxBrushWidth=24, maxLength=100, maxVertex=20)

        h, w = gt_img.shape[:2]

        # prepare input data
        img = gt_img * (1 - mask) + 255 * mask
        grid = 4
        img = img[:h // grid * grid, :w // grid * grid, :]
        mask = mask[:h // grid * grid, :w // grid * grid, :]
        img = np.expand_dims(img, 0)
        mask = np.expand_dims(mask, 0)

        logger.debug(f'input image shape: {img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = net.predict({'import/Placeholder:0': img, 'import/Placeholder_1:0': mask})
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            output = net.predict({'import/Placeholder:0': img, 'import/Placeholder_1:0': mask})

        output = output[0]

        img = img[0].astype(np.uint8)
        mask = cv2.cvtColor(mask[0], cv2.COLOR_GRAY2BGR).astype(np.uint8) * 255
        output = output[0].astype(np.uint8)[:, :, ::-1]  # NHWC -> HWC(RGB) -> HWC(BGR)
        res_img = np.hstack((img, mask, output, gt_img))

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def main():
    info = {
        ("paris-streetview", "rect"): (WEIGHT_PARIS_STREETVIEW_PATH, MODEL_PARIS_STREETVIEW_PATH, (256, 256)),
        ("celebahq", "rect"): (WEIGHT_CELEBAHQ_256_PATH, MODEL_CELEBAHQ_256_PATH, (256, 256)),
        ("celebahq-512", "rect"): (WEIGHT_CELEBAHQ_512_PATH, MODEL_CELEBAHQ_512_PATH, (512, 512)),
        ("celebahq-512", "stroke"): (WEIGHT_CELEBAHQ_FREEFORM_PATH, MODEL_CELEBAHQ_FREEFORM_PATH, (512, 512)),
        ("places2", "stroke"): (WEIGHT_PLACE2_PATH, MODEL_PLACE2_PATH, (512, 680)),
    }
    key = (args.model, args.mask_type)
    if key not in info:
        logger.error("(MODEL = %s, MASK_TYPE = %s) is unmatch." % key)
        logger.info("appropriate settings:\n"
                    "\t(MODEL = paris-streetview, MASK_TYPE = rect)\n"
                    "\t(MODEL = celebahq, MASK_TYPE = rect)\n"
                    "\t(MODEL = celebahq-512, MASK_TYPE = rect)\n"
                    "\t(MODEL = celebahq-512, MASK_TYPE = stroke)\n"
                    "\t(MODEL = places2, MASK_TYPE = stroke)")
        sys.exit(-1)

    # model files check and download
    weight_path, model_path, img_shape = info[key]
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # net initialize
    net = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=args.memory_mode)

    recognize_from_image(net, img_shape)


if __name__ == '__main__':
    main()
