import os
import cv2
import sys
import time
import argparse
from glob import glob
import numpy as np

from crestereo_util import CREStereo

import ailia
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)

#camera_config =  CameraConfig(0.546, 1000)
HEIGHT= 360 
WIDTH = 640
# ======================
# Parameters 1
# ======================

LEFT_IMAGE_PATH = "im2.png"
RIGHT_IMAGE_PATH = "im6.png"
SAVE_IMAGE_PATH = 'output.png'

WEIGHT_PATH = 'crestereo.onnx' 
MODEL_PATH = 'crestereo.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/crestereo/'

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'crestereo',
    LEFT_IMAGE_PATH,
    RIGHT_IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-l','--left', type=str,
    default=LEFT_IMAGE_PATH,
    help='The input image for left image.'
)
parser.add_argument(
    '-r', '--right', type=str,
    default=RIGHT_IMAGE_PATH,
    help='The input image for right image.'
)

parser.add_argument(
    '-o', '--onnx', action='store_true',
    help="Option to use onnxrutime to run or not."
)

parser.add_argument(
    '-vr', '--video_rigth', type=str,
    help='The input video for pole detection.'
)
parser.add_argument(
    '-vl', '--video_left', type=str,
    help='The input video for pole detection.'
)

args = update_parser(parser)
# # ======================
# # Main functions
# # ======================

def preprocessing(left_img,right_img,input_shape):

    left_img = cv2.resize(left_img,(input_shape[3],input_shape[2]))
    right_img = cv2.resize(right_img,(input_shape[3],input_shape[2]))
    
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    
    combined_img = np.concatenate((left_img, right_img), axis=-1) / 255.0
    combined_img = combined_img.transpose(2, 0, 1)

    return np.expand_dims(combined_img, 0).astype(np.float32)

def recognize_from_image(net):
    
    # prepare input data
    logger.debug(f'input image: {args.left},{args.right}')
    left_img = imread(args.left)
    right_img = imread(args.right)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            
            depth_estimator = CREStereo(args,net)
            disparity_map = depth_estimator(left_img, right_img)
            combined_image = depth_estimator.draw_disparity()
    
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        depth_estimator = CREStereo(args,net)
        disparity_map = depth_estimator(left_img, right_img)
        combined_image = depth_estimator.draw_disparity()

    cv2.imwrite(args.savepath, combined_image)

    logger.info('Script finished successfully.')


def recognize_from_video(net):

    captureL = webcamera_utils.get_capture(args.video_rigth)
    captureR = webcamera_utils.get_capture(args.video_left)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = webcamera_utils.get_writer(args.savepath, HEIGHT, WIDTH)
    else:
        writer = None

    frame_shown = False
    while (True):
        retL,left_img  = captureL.read()
        retR,right_img = captureR.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not retR or not retL :
            break
        if frame_shown and cv2.getWindowProperty('output', cv2.WND_PROP_VISIBLE) == 0:
            break

        depth_estimator = CREStereo(args,net)
        disparity_map = depth_estimator(left_img, right_img)
        combined_image = depth_estimator.draw_disparity()
        #combined_image = np.hstack((left_img, combined_image,right_img))

        cv2.imshow('output', combined_image)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(combined_image)

    captureL.release()
    captureR.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')

def main():

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    if args.onnx:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    else:
        logger.info(f'env_id: {args.env_id}')
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, memory_mode=args.memory_mode)

    if args.video_rigth is not None or args.video_left is not None:
        if args.video_rigth is not None and args.video_left is not None:
            # video mode
            recognize_from_video(net)
        else:
            logger.error(f'Two camera modules not present')
    else:
        # image mode
        recognize_from_image(net)

if __name__ == '__main__':
    main()
