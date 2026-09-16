import sys
import time
import io

import numpy as np
import cv2
import matplotlib.pyplot as plt
from lightglue_utils import *

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "superpoint.onnx"
MODEL_PATH = "superpoint.onnx.prototxt"
LIGHTGLUE_WEIGHT_PATH = "superpoint_lightglue.onnx"
LIGHTGLUE_MODEL_PATH = "superpoint_lightglue.onnx.prototxt"

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/lightglue/'

IMAGE_A_PATH = 'img_A.png'
IMAGE_B_PATH = 'img_B.png'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'LightGlue', IMAGE_A_PATH, SAVE_IMAGE_PATH, fp16_support=False
)
parser.add_argument(
    '-i2', '--input2', metavar='IMAGE2', default=IMAGE_B_PATH,
    help='Pair image path of input image.'
)

parser.add_argument(
    "--extractor_type",
    type=str,
    choices=["superpoint", "disk"],
    default="superpoint",
    help="Type of feature extractor. Supported extractors are 'superpoint' and 'disk'.",
)
parser.add_argument(
    "--img_size",
    nargs="+",
    type=int,
    default=512,
    required=False,
    help="Sample image size for ONNX tracing. If a single integer is given, resize the longer side of the images to this value. Otherwise, please provide two integers (height width) to resize both images to this size, or four integers (height width height width).",
)
 
args = update_parser(parser)

# ======================
# Main functions
# ======================

def infer(
    img_paths: List[str],
    extractor_type: str,
    img_size=512,
    runner=None
):
    # Handle args
    img0_path = img_paths[0]
    img1_path = img_paths[1]
    if isinstance(img_size, List):
        if len(img_size) == 1:
            size0 = size1 = img_size[0]
        elif len(img_size) == 2:
            size0 = size1 = img_size
        elif len(img_size) == 4:
            size0, size1 = img_size[:2], img_size[2:]
        else:
            raise ValueError("Invalid img_size. Please provide 1, 2, or 4 integers.")
    else:
        size0 = size1 = img_size

    image0, scales0 = load(img0_path, resize=size0)
    image1, scales1 = load(img1_path, resize=size1)

    extractor_type = extractor_type.lower()
    if extractor_type == "superpoint":
        image0 = rgb_to_grayscale(image0)
        image1 = rgb_to_grayscale(image1)
    elif extractor_type == "disk":
        pass
    else:
        raise NotImplementedError(
            f"Unsupported feature extractor type: {extractor_type}."
        )

    # Run inference
    m_kpts0, m_kpts1 = runner.run(image0, image1, scales0, scales1)

    # Visualisation
    orig_image0, _ = load(img0_path)
    orig_image1, _ = load(img1_path)
    plot_images(
        [orig_image0[0].transpose(1, 2, 0), orig_image1[0].transpose(1, 2, 0)]
    )
    plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)

def recognize_from_image(runner):

    # input image loop
    for image_path1 ,image_path2 in zip(args.input, [args.input2]):

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = infer([image_path1,image_path2], args.extractor_type, args.img_size, runner)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = infer([image_path1,image_path2], args.extractor_type, args.img_size, runner)

        # plot result
        savepath = get_savepath(args.savepath, image_path1, ext='.png')
        logger.info(f'saved at : {savepath}')
        save_plot(savepath)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(LIGHTGLUE_WEIGHT_PATH, LIGHTGLUE_MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # Load ONNX models

    runner = LightGlueRunner(

        extractor_path="superpoint.onnx",
        lightglue_path="superpoint_lightglue.onnx",
        env_id = env_id,
        memory_mode = args.memory_mode
    )

    recognize_from_image(runner)

if __name__ == '__main__':
    main()
