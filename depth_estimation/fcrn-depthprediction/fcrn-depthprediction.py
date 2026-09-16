import os
import sys
import time
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import get_capture, get_writer, \
    calc_adjust_fsize ,preprocess_frame # noqa: E402
from image_utils import normalize_image  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'ResNet50UpProj.onnx'
MODEL_PATH = 'ResNet50UpProj.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/fcrn-depthprediction/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'input_depth.png'
IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
CHANNELS = 3


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('FCRN-DepthPrediction model', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = Image.open(image_path)
        img = img.resize([IMAGE_WIDTH,IMAGE_HEIGHT], Image.LANCZOS)
        img = np.array(img).astype('float32')
        img = np.expand_dims(np.asarray(img), axis=0)
        img = img[:,:,:,0:3]

        logger.info(f'input image shape: {img.shape}')
        net.set_input_shape(img.shape)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                result = net.predict(img)[0]
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            result = net.predict(img)[0]

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        fig = plt.figure()
        ii = plt.imshow(result)
        fig.colorbar(ii)
        fig.savefig(savepath)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = calc_adjust_fsize(f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH)
        # save_w * 2: we stack source frame and estimated heatmap
        writer = get_writer(args.savepath, save_h, save_w * 2)
    else:
        writer = None

    input_shape_set = False
    while(True):
        ret, frame = capture.read()

        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        _, img = preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='None'
        )

        img = np.transpose(img, (0,2,3,1))

        if(not input_shape_set):
            net.set_input_shape(img.shape)
            input_shape_set = True
        result = net.predict(img)[0]

        plt.imshow(result)
        plt.pause(.01)
        if not plt.get_fignums():
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
