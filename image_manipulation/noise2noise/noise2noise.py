import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'noise2noise_gaussian.onnx'
MODEL_PATH = 'noise2noise_gaussian.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/noise2noise/'

IMAGE_PATH = 'monarch-gaussian-noisy.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('Noise2Noise', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-add_noise', action='store_true',
    help='If add this argument, add noise to input image (which will be saved)'
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def add_noise(img, noise_param=50):
    height, width = img.shape[0], img.shape[1]
    std = np.random.uniform(0, noise_param)
    noise = np.random.normal(0, std, (height, width, 3))
    noise_img = np.array(img) + noise
    noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
    img = noise_img
    cv2.imwrite('noise_image.png', img)  # TODO make argument for savepath ?
    return img


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        # prepare input data
        img = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='None',
        )

        if args.add_noise:
            img = add_noise(img)

        img = img / 255.0
        input_data = img.transpose(2, 0, 1)
        input_data.shape = (1, ) + input_data.shape

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_ailia = net.predict(input_data)

        # postprocessing
        output_img = preds_ailia[0].transpose(1, 2, 0) * 255
        output_img = np.clip(output_img, 0, 255)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, output_img)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)
    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        # TODO: DEBUG: save image shape
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = webcamera_utils.calc_adjust_fsize(
            f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        _, resized_image = webcamera_utils.adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )

        # add noise
        resized_image = add_noise(resized_image)

        resized_image = resized_image / 255.0
        input_data = resized_image.transpose(2, 0, 1)
        input_data.shape = (1, ) + input_data.shape

        # inference
        preds_ailia = net.predict(input_data)

        # side by side
        preds_ailia[:, :, :, 0:input_data.shape[3]//2] = input_data[
            :, :, :, 0:input_data.shape[3]//2
        ]

        # postprocessing
        output_img = preds_ailia[0].transpose(1, 2, 0)
        cv2.imshow('frame', output_img)
        frame_shown = True

        # # save results
        # if writer is not None:
        #     writer.write(output_img)

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
