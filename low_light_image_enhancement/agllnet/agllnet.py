import os
import sys
import time

import ailia
import cv2
import numpy as np

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH = "AGLLNet.opt.onnx"
MODEL_PATH = "AGLLNet.opt.onnx.prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/agllnet/'

IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'

# Default input size
HEIGHT_SIZE = 1152
WIDTH_SIZE = 768

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'AGLLNet',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
args = update_parser(parser)


def recognize_from_image():
    # net initialize
    env_id = args.env_id
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        img = imread(image_path) / 255.
        H = img.shape[0]
        W = img.shape[1]
        img = cv2.resize(img, (HEIGHT_SIZE, WIDTH_SIZE), interpolation=cv2.INTER_LANCZOS4)
        img = img[np.newaxis, :]
        logger.info(f'input image shape: {img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                pred = net.run(img)[0]
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            pred = net.run(img)[0]

        # post process
        enhance = pred[0, :, :, 4:7]
        enhance = cv2.resize(enhance, (W, H), interpolation=cv2.INTER_LANCZOS4)
        enhance = np.clip(enhance, 0.0, 1.0)
        output = (enhance * 255.).astype(np.uint8)

        #save result
        logger.info(f'saved at : {args.savepath}')
        cv2.imwrite(args.savepath, output)

    logger.info('Script finished successfully.')

def recognize_from_video():
    # net initialize
    env_id = args.env_id
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

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

        input = cv2.resize(frame, (HEIGHT_SIZE, WIDTH_SIZE), interpolation=cv2.INTER_LANCZOS4) / 255.
        input = input[np.newaxis, :]

        # inference
        print(input.shape)
        pred = net.run(input)[0]

        # plot result
        enhance = pred[0, :, :, 4:7]
        enhance = cv2.resize(enhance, (f_w, f_h), interpolation=cv2.INTER_LANCZOS4)
        enhance = np.clip(enhance, 0.0, 1.0)
        output = (enhance * 255.).astype(np.uint8)

        cv2.imshow('frame', output)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(output)

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
