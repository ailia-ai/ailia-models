import sys
import time

import cv2
import numpy as np
from PIL import Image

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
WEIGHT_PATH = 'pytorch-unet.onnx'
MODEL_PATH = 'pytorch-unet.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/pytorch-unet/'

IMAGE_PATH = 'data/imgs/0cdf5b5d0ce1_14.jpg'
SAVE_IMAGE_PATH = 'data/masks/0cdf5b5d0ce1_14.jpg'

IMAGE_WIDTH = 959
IMAGE_HEIGHT = 640

THRESHOLD = 0.5


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'hand-detection.PyTorch hand detection model', IMAGE_PATH, SAVE_IMAGE_PATH
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def load_image(input_path):
    return np.array(Image.open(input_path))


def preprocess(img):
    img = cv2.resize(
        img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA
    )
    img = np.expand_dims(img, 0)
    img_trans = img.transpose((0, 3, 1, 2))  # NHWC to NCHW
    if img_trans.max() > 1:
        img_trans = img_trans / 255

    return img_trans.astype(np.float32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def post_process(output):
    probs = sigmoid(output)
    probs = probs.squeeze(0)
    full_mask = cv2.resize(
        probs.transpose(1, 2, 0),
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )
    mask = full_mask > THRESHOLD
    return mask.astype(np.uint8)*255


def segment_image(img, net):
    img = preprocess(img)

    # feedforward
    output = net.predict({'input.1': img})[0]

    out = post_process(output)
    return out


# ======================
# Main functions
# ======================
def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        img = load_image(image_path)
        logger.debug(f'input image shape: {img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                out = segment_image(img, net)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            out = segment_image(img, net)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, out)
    logger.info('Script finished successfully.')


def recognize_from_video(net):
    capture = webcamera_utils.get_capture(args.video)
    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(
            args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH, rgb=False
        )
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, img = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        out = segment_image(img, net)
        cv2.imshow('frame', out)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(out)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # model initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
