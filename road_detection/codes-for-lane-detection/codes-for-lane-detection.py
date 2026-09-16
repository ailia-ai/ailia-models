import os
import sys
import time

import ailia
import cv2
import numpy as np

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
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/codes-for-lane-detection/'
IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

MODEL_LISTS = ['erfnet', 'scnn']
RESIZE_MODE_LISTS = ['padding', 'crop_center', 'crop_bottom']

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('erfnet model', IMAGE_PATH, SAVE_IMAGE_PATH)
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

if args.arch=="erfnet":
    WEIGHT_PATH = 'erfnet.opt.onnx'
    MODEL_PATH = 'erfnet.opt.onnx.prototxt'
    HEIGHT = 208
    WIDTH = 976
elif args.arch=="scnn":
    WEIGHT_PATH = 'SCNN_tensorflow.opt.onnx'
    MODEL_PATH = 'SCNN_tensorflow.opt.onnx.prototxt'
    HEIGHT = 288
    WIDTH = 800

# ======================
# Main functions
# ======================

def colorize(output):
    out_img = np.zeros((HEIGHT,WIDTH,3))
    for num in range(4):
        prob_map = (output[0][num + 1] * 255).astype(int)
        if num==0:
            out_img[:,:,0] += prob_map
        if num==1 or num==3:
            out_img[:,:,1] += prob_map
        if num==2 or num==3:
            out_img[:,:,2] += prob_map
    return out_img

def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        raw_img = imread(image_path)
        logger.debug(f'input image shape: {raw_img.shape}')

        img = crop_and_resize(raw_img,WIDTH,HEIGHT,args.arch,args.resize)
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
        out_img = colorize(output)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, out_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = webcamera_utils.get_writer(args.savepath, HEIGHT*2, WIDTH)
    else:
        writer = None

    output_buffer = np.zeros((HEIGHT*2,WIDTH,3))
    output_buffer = output_buffer.astype(np.uint8)
    
    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('output', cv2.WND_PROP_VISIBLE) == 0:
            break
    
        resized_img = crop_and_resize(frame,WIDTH,HEIGHT,args.arch,args.resize)
        img = preprocess(resized_img,args.arch)

        output, output_exist = net.run(img)

        output = postprocess(output,args.arch)
        out_img = colorize(output)
        out_img = np.array(out_img, dtype=np.uint8)

        # create output img
        output_buffer[0:HEIGHT,0:WIDTH,:] = resized_img
        output_buffer[HEIGHT:HEIGHT*2,0:WIDTH,:] = out_img

        cv2.imshow('output', output_buffer)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(output_buffer)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    env_id = args.env_id
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    if args.arch=="erfnet":
        net.set_input_shape((1, 3, HEIGHT, WIDTH))
    elif args.arch=="scnn":
        net.set_input_shape((1, HEIGHT, WIDTH, 3))

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
