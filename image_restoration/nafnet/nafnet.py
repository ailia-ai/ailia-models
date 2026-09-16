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
from image_utils import imread

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================

IMAGE_PATH = 'noisy.png'
SAVE_IMAGE_PATH = 'output.png'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/nafnet/'


BLUR_LISTS = ['Baseline-GoPro-width32' ,'NAFNet-GoPro-width32', 'NAFNet-REDS-width64', 'Baseline-GoPro-width64','NAFNet-GoPro-width64']
NOISE_LISTS = ['Baseline-SIDD-width32', 'NAFNet-SIDD-width64', 'Baseline-SIDD-width64' ,'NAFNet-SIDD-width32']

MODEL_LISTS= BLUR_LISTS + NOISE_LISTS

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('NAFNet model', IMAGE_PATH, SAVE_IMAGE_PATH, fp16_support=False)

parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='NAFNet-SIDD-width32', choices=MODEL_LISTS,
    help='deblur model lists: ' + ' | '.join(BLUR_LISTS) + ' , ' +
         'denoise model list: ' + ' | '.join(NOISE_LISTS)
)

args = update_parser(parser)

# ======================
# Parameters 2
# ======================

WEIGHT_PATH = args.arch + '.onnx'
MODEL_PATH =  args.arch + '.onnx.prototxt'

BLUR_IMAGE_MIN_WIDTH = 384


# ======================
# Main functions
# ====================== 

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    _tensor = tensor.clip(min_max[0],min_max[1])
    _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
    n_dim = _tensor.ndim

    if n_dim == 3:
        img_np = _tensor
        img_np = img_np.transpose(1, 2, 0)
        if img_np.shape[2] == 1:  # gray image
            img_np = np.squeeze(img_np, axis=2)
        else:
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif n_dim == 2:
        #gray
        img_np = _tensor

    if out_type == np.uint8:
        # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
        img_np = (img_np * 255.0).round()
    img_np = img_np.astype(out_type)
    return img_np

def preprocess(img):
    if args.arch in BLUR_LISTS:
        if img.shape[0] < BLUR_IMAGE_MIN_WIDTH or img.shape[1] < BLUR_IMAGE_MIN_WIDTH:
            if img.shape[0] < img.shape[1]:
                img = cv2.resize(img, ((int)(img.shape[1] / img.shape[0] * BLUR_IMAGE_MIN_WIDTH), BLUR_IMAGE_MIN_WIDTH))
            else:
                img = cv2.resize(img, (BLUR_IMAGE_MIN_WIDTH, (int)(img.shape[0] / img.shape[1] * BLUR_IMAGE_MIN_WIDTH)))

    imgs = img.astype(np.float32) /255.0
    
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    imgs = imgs.transpose(2, 0, 1)
    imgs = np.expand_dims(imgs, 0)
    return imgs

def recognize_from_image(net):
    for image_path in args.input:
        input_data = imread(image_path)
        input_data = preprocess(input_data)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                sr = net.run(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            sr = net.run(input_data)

        # tensor to image
        sr = tensor2img(sr[0][0])

        ## postprocessing
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, sr)
    logger.info('Script finished successfully.')


def recognize_from_video(net):
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

        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
            
        ## Preprocessing
        frame = preprocess(frame)

        # Inference
        sr = net.run(frame)

        output_img = tensor2img(sr[0][0])

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
    
    # net initialize
    env_id = args.env_id

    net = ailia.Net(None, WEIGHT_PATH,memory_mode=args.memory_mode,env_id=env_id)

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()

