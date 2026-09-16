import os
import sys
import time

import ailia
import cv2
import numpy as np
from PIL import Image

from mlsd_utils import pred_lines

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger

import webcamera_utils
from image_utils import imread, load_image
from model_utils import check_and_download_models
from arg_utils import get_base_parser, get_savepath, update_parser

logger = getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ======================
# Parameters
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mlsd/'
WEIGHT_PATH = 'M-LSD_512_large.opt.onnx'
MODEL_PATH = 'M-LSD_512_large.opt.onnx.prototxt'
IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('mlsd model', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)
args.env_id = 0
logger.info('Change env_id: 0')

# ======================
# Main functions
# ======================

def gradio_wrapper_for_LSD(img_input, net):
  lines = pred_lines(img_input, net, input_shape=[512, 512])
  img_output = img_input.copy()

  # draw lines
  for line in lines:
    x_start, y_start, x_end, y_end = [int(val) for val in line]
    cv2.line(img_output, (x_start, y_start), (x_end, y_end), [0,255,255], 2)

  return img_output

def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        img_input = imread(image_path)

        # inference
        logger.info('Start inference...')
        logger.warning('Inference using CPU because model accuracy is low on GPU.')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                preds_img = gradio_wrapper_for_LSD(img_input, net)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_img = gradio_wrapper_for_LSD(img_input, net)

        # postprocessing
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, preds_img)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    logger.warning('Inference using CPU because model accuracy is low on GPU.')

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img_input = np.array(frame)

        # inference
        preds_img = gradio_wrapper_for_LSD(img_input, net)
        cv2.imshow('frame', preds_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(preds_img)

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
