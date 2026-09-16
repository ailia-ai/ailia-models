import cv2
import os
import sys
import ailia
import numpy as np
import argparse
import time

# import original modules
sys.path.append('../../util')
from image_utils import load_image, get_image_shape  # noqa: E402
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
import webcamera_utils  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================

IMAGE_PATH = 'input.bmp'
SAVE_IMAGE_PATH = 'output.png'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/rcan-it/'


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('RCAN-it model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument('--scale', choices=['2', '3', '4'], default='2', help='choose scale')
args = update_parser(parser)


# ======================
# Parameters 2
# ======================

WEIGHT_PATH = 'rcan-it_scale' + args.scale + '.onnx'
MODEL_PATH = 'rcan-it_scale' + args.scale + '.onnx.prototxt'


# ======================
# Main functions
# ====================== 

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    img = np.multiply(img,pixel_range).clip(0, 255)
    img = np.divide(np.around(img),pixel_range)
    return img


def inference(net,input_data): 
    input_data = input_data.astype(np.float32)
    sr = net.run(input_data)[0][0]
    sr = sr.transpose(1,2,0)
    sr *= 255
    sr = cv2.cvtColor(sr,cv2.COLOR_BGR2RGB)
    rgb_range = 255
    sr = quantize(sr, rgb_range)
    return sr


def recognize_from_image():

    # net initialize
    net = ailia.Net(None, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    #logger.info(IMAGE_PATH)

    for image_path in args.input:

        IMAGE_HEIGHT, IMAGE_WIDTH = get_image_shape(image_path)

        # prepare input data
        #logger.info(image_path)
        input_data = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            gen_input_ailia=True,
        )

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                sr = inference(net,input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
           sr = inference(net,input_data)

        ## postprocessing
        #logger.info(f'saved at : {savepath}')
        cv2.imwrite(args.savepath, sr)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(None, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * int(args.scale))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * int(args.scale))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    time.sleep(1)  
    
    while(True):
        ret, frame = capture.read()

        frame = frame.astype(np.float32)
        frame /= 255.0

        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
            
        ## Preprocessing

        frame = frame.transpose(2,0,1)
        frame = np.expand_dims(frame,0)
        # Inference
        sr = inference(net,frame)
        sr = cv2.cvtColor(sr,cv2.COLOR_BGR2RGB)
        output_img = (sr)
        output_img = sr.astype(np.uint8)

        # Postprocessing
        cv2.imshow('frame', output_img)

        # save results
        if writer is not None:
            writer.write(output_img)

        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
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

