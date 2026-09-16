import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'net_G.onnx'
MODEL_PATH = 'net_G.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deblur_gan/'

IMAGE_PATH = 'sample.png'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('DeblurGAN model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-sz', '--fine_size', type=int, metavar='SIZE', default=None,
    help='scale images to this size.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img):
    mean = np.array((0.5, 0.5, 0.5))
    std = np.array((0.5, 0.5, 0.5))

    if args.fine_size:
        target_width = args.fine_size
        h, w = img.shape[:2]
        if w != target_width:
            h = int(target_width * h / w)
            w = target_width
            img = np.array(Image.fromarray(img).resize(
                (w, h),
                resample=Image.BICUBIC))

    img = img / 255
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    return img


def postprocess(output):
    output = (output.transpose((1, 2, 0)) + 1) / 2.0 * 255.0
    img = output.astype(np.uint8)
    img = img[:, :, ::-1]  # RGB -> BGR

    return img


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare grand truth
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = preprocess(img)

        logger.debug(f'input image shape: {img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = net.predict({'img': img})
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            output = net.predict({'img': img})

        output = output[0]
        res_img = postprocess(output[0])

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
