import glob
import os
import sys
import time

import cv2
import numpy as np

from roneld_utils import roneld_lane_detection

sys.path.append('../../road_detection/codes-for-lane-detection')
import ailia
from codes_for_lane_detection_utils import (crop_and_resize, postprocess,
                                            preprocess)

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger

import webcamera_utils
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models
from arg_utils import get_base_parser, get_savepath, update_parser

logger = getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ======================
# Parameters
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/erfnet/'
IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

MODEL_LISTS = ['erfnet']
RESIZE_MODE_LISTS = ['padding', 'crop_center', 'crop_bottom']

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('roneld model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='erfnet', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '-r', '--resize', metavar='RESIZE',
    default='crop_bottom', choices=RESIZE_MODE_LISTS,
    help='resize mode lists: ' + ' | '.join(RESIZE_MODE_LISTS)
)
args = update_parser(parser)

WEIGHT_PATH = 'erfnet.opt.onnx'
MODEL_PATH = 'erfnet.opt.onnx.prototxt'

HEIGHT = 208
WIDTH = 976

# ======================
# Main functions
# ======================


def recognize_from_image():
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    net.set_input_shape((1, 3, HEIGHT, WIDTH))

    prev_lanes = []
    prev_curves = np.zeros(10)
    curve_mode = False

    # input image loop
    for image_path in args.input:
        # prepare input data
        raw_img = imread(image_path)

        # preprocess
        raw_img = crop_and_resize(raw_img,WIDTH,HEIGHT,args.arch,args.resize)
        img = raw_img
        img = preprocess(img,args.arch)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                output, output_exist = net.run(img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            output, output_exist = net.run(img)

        output = postprocess(output,args.arch)
        lane_images = []

        for num in range(4):
            lane_image = output[0][num + 1]
            lane_image = (lane_image * 255).astype(int)
            lane_images.append(lane_image)

        # call to roneld and store output for next method call
        output_images, prev_lanes, prev_curves, curve_mode = \
            roneld_lane_detection(lane_images, prev_lanes, prev_curves, curve_mode=curve_mode,
                                  image=raw_img)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, raw_img)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    prev_lanes = []
    prev_curves = np.zeros(10)
    curve_mode = False

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(args.savepath, HEIGHT, WIDTH)
    else:
        writer = None
    
    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # preprocess
        frame = crop_and_resize(frame,WIDTH,HEIGHT,args.arch,args.resize)
        img = frame
        img = preprocess(img,args.arch)

        # inference
        output, output_exist = net.run(img)

        # postprocess
        output = postprocess(output,args.arch)

        lane_images = []

        for num in range(4):
            lane_image = output[0][num + 1]
            lane_image = (lane_image * 255).astype(int)
            lane_images.append(lane_image)

        # call to roneld and store output for next method call
        output_images, prev_lanes, prev_curves, curve_mode = \
            roneld_lane_detection(lane_images, prev_lanes, prev_curves, curve_mode=curve_mode,
                                  image=frame)

        cv2.imshow('frame', frame)
        frame_shown = True

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
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
