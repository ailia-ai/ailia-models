import sys
import time

import cv2
from PIL import Image
from skimage import color
import numpy as np
import matplotlib.pyplot as plt

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'colorizer.onnx'
MODEL_PATH = 'colorizer.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/colorization/'

IMAGE_PATH = 'imgs/ansel_adams1.jpg'
SAVE_IMAGE_PATH = 'imgs_out/ansel_adams1_output.jpg'

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Colorful Image Colorization model', IMAGE_PATH, SAVE_IMAGE_PATH
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def load_img(img_path):
    out_np = np.asarray(Image.open(img_path))
    if(out_np.ndim == 2):
        out_np = np.tile(out_np[:, :, None], 3)
    return out_np


def preprocess(img_rgb_orig, resample=3):
    img_rgb_rs = np.asarray(Image.fromarray(img_rgb_orig).resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT), resample=resample)
    )

    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_lab_orig = img_lab_orig[:, :, 0][None, None, :, :]
    img_lab_rs = img_lab_rs[:, :, 0][None, None, :, :]

    return (img_lab_orig, img_lab_rs)


def post_process(out, img_lab_orig):
    HW_orig = img_lab_orig.shape[2:]
    out_ab_orig = cv2.resize(
        out.transpose(2, 3, 1, 0).squeeze(),
        (HW_orig[1],  HW_orig[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    out_ab_orig = np.expand_dims(out_ab_orig.transpose(2, 0, 1), 0)
    out_lab_orig = np.concatenate([img_lab_orig, out_ab_orig], 1)

    out_img = color.lab2rgb(out_lab_orig[0, ...].transpose((1, 2, 0)))
    return out_img


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    net.set_input_shape((1, 1, IMAGE_HEIGHT, IMAGE_WIDTH))

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        img = load_img(image_path)
        logger.debug(f'input image shape: {img.shape}')
        (img_lab_orig, img_lab_rs) = preprocess(img)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                out = net.predict({'input.1': img_lab_rs})[0]
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            out = net.predict({'input.1': img_lab_rs})[0]

        out_img = post_process(out, img_lab_orig)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        plt.imsave(savepath, out_img)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    net.set_input_shape((1, 1, IMAGE_HEIGHT, IMAGE_WIDTH))

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, img = capture.read()
        # press q to end video capture
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        (img_lab_orig, img_lab_rs) = preprocess(img)
        out = net.predict({'input.1': img_lab_rs})[0]
        out_img = post_process(out, img_lab_orig)
        out_img = np.array(out_img * 255, dtype=np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', out_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(out_img)

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
