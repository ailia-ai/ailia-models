import sys
import time

import ailia
import cv2
import numpy as np
from PIL import Image as pimg

from swiftnet_utils.color_lables import ColorizeLabels
from swiftnet_utils.labels import labels

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402 noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/swiftnet/'

WEIGHT_PATH = "swiftnet.opt.onnx"
MODEL_PATH = "swiftnet.opt.onnx.prototxt"

IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'
HEIGHT = 1024
WIDTH = 2048

color_info = [label.color for label in labels if label.ignoreInEval is False]

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('swiftnet model', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    env_id = args.env_id
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        img = imread(image_path)
        logger.debug(f'input image shape: {img.shape}')
        img = cv2.resize(img, (WIDTH, HEIGHT))
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                pred = net.predict(img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            pred = net.predict(img)

        # postprocessing
        to_color = ColorizeLabels(color_info)
        pred = np.argmax(pred, axis=1)
        pred = to_color(pred).astype(np.uint8)
        pred = pimg.fromarray(pred[0])

        # save
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        pred.save(savepath)


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w, rgb=False)
    else:
        writer = None

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        input = cv2.resize(frame, (WIDTH, HEIGHT))
        input = input.transpose(2, 0, 1)
        input = np.expand_dims(input, 0)

        # inference
        pred = net.predict(input)

        # postprocessing
        to_color = ColorizeLabels(color_info)
        pred = np.argmax(pred, axis=1)[0]
        pred = to_color(pred).astype(np.uint8)

        cv2.imshow('frame', pred)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(pred)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
