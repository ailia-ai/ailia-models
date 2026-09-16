import os
import sys
import numpy as np
import cv2
import time

from l2cs_util import l2cs , render

import ailia

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger

import webcamera_utils
from detector_utils import plot_results, reverse_letterbox
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models
from arg_utils import get_base_parser, get_savepath, update_parser

logger = getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# ======================
# Parameters
# ======================
REMOTE_PATH_FACE = "https://storage.googleapis.com/ailia-models/retinaface/"
REMOTE_PATH_L2CS = 'https://storage.googleapis.com/ailia-models/l2cs_net/'

# settings
WEIGHT_PATH_FACE = "retinaface_resnet50.onnx"
MODEL_PATH_FACE = "retinaface_resnet50.onnx.prototxt"

WEIGHT_PATH_L2CS = "l2cs.onnx"
MODEL_PATH_L2CS = "l2cs.onnx.prototxt"

IMAGE_PATH = 'selfie.png'
SAVE_IMAGE_PATH = 'output.png'

# Default input size
HEIGHT = 224
WIDTH = 224

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('l2cs model', IMAGE_PATH, SAVE_IMAGE_PATH)

args = update_parser(parser)

# ======================
# Main functions
# ======================
def recognize_from_image():
    env_id = args.env_id

    object_detection_net = ailia.Net(MODEL_PATH_FACE, WEIGHT_PATH_FACE, env_id=env_id, memory_mode=args.memory_mode)
    l2cs_net = ailia.Net(MODEL_PATH_L2CS,WEIGHT_PATH_L2CS,env_id=env_id, memory_mode=args.memory_mode)

    net = l2cs(object_detection_net,l2cs_net,confidence_threshold=0.5)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        raw_img = imread(image_path)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))

                results = net.step(raw_img)
                frame = render(raw_img, results)

                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            results = net.step(raw_img)
            frame = render(raw_img, results)

        # plot result
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath,frame)


    logger.info('Script finished successfully.')

def recognize_from_video():
    env_id = args.env_id

    object_detection_net = ailia.Net(MODEL_PATH_FACE, WEIGHT_PATH_FACE, env_id=env_id, memory_mode=args.memory_mode)
    l2cs_net = ailia.Net(MODEL_PATH_L2CS,WEIGHT_PATH_L2CS,env_id=env_id, memory_mode=args.memory_mode)
    net = l2cs(object_detection_net,l2cs_net,confidence_threshold=0.5)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = f_h, f_w
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        results = net.step(frame)
        frame = render(frame, results)

        c = cv2.waitKey(1)
        if c == 27:
            break

        cv2.imshow("Demo", frame)
        cv2.waitKey(5)

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH_L2CS, MODEL_PATH_L2CS, REMOTE_PATH_L2CS)
    check_and_download_models(WEIGHT_PATH_FACE, MODEL_PATH_FACE, REMOTE_PATH_FACE)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
