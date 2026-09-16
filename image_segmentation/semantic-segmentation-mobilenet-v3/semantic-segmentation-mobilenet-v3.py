import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH = './sem_seg.onnx'
MODEL_PATH = './sem_seg.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/semantic-segmentation-mobilenet-v3/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

CATEGORY = (
    'Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes',
    'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face',
    'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe'
)
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Semantic segmentation with MobileNetV3 model', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--orig-size',
    action='store_true',
    help='output in original image size.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================
def preprocess(img):
    img = img[:, :, ::-1]
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def post_processing(output, orig_shape=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    threshold = 100

    out_mask = np.squeeze(output)
    if args.orig_size:
        out_mask = cv2.resize(
            out_mask,
            (orig_shape[1], orig_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    out_mask = out_mask * 255 > threshold

    return out_mask


# ======================
# Main functions
# ======================
def predict(img, net):
    # initial preprocesses
    h, w, _ = img.shape
    img = preprocess(img)

    # feedforward
    output = net.predict({
        'input_1': img
    })

    # post processes
    out_mask = post_processing(output, orig_shape=(h, w))

    return out_mask


def recognize_from_image(filename, net):
    # prepare input data
    img = load_image(filename)
    logger.debug(f'input image shape: {img.shape}')

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            out_mask = predict(img, net)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        out_mask = predict(img, net)

    if not args.orig_size:
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    res_img = np.ones(img.shape, np.uint8) * 255
    res_img[out_mask] = img[out_mask]

    # plot result
    savepath = get_savepath(args.savepath, filename)
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, res_img)
    logger.info('Script finished successfully.')


def recognize_from_video(video, net):
    capture = webcamera_utils.get_capture(video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(
            args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH
        )
    else:
        writer = None

    frame_shown = True
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        out_mask = predict(frame, net)

        # draw segmentation area
        if not args.orig_size:
            frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        res_img = np.ones(frame.shape, np.uint8) * 255
        res_img[out_mask] = frame[out_mask]

        # show
        cv2.imshow('frame', res_img)
        frame_shown = False
        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    net.set_input_shape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    if args.video is not None:
        # video mode
        recognize_from_video(args.video, net)
    else:
        # image mode
        # input image loop
        for image_path in args.input:
            # prepare input data
            logger.info(image_path)
            recognize_from_image(image_path, net)


if __name__ == '__main__':
    main()
