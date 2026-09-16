import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
import webcamera_utils  # noqa: E402
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = './road-segmentation-adas-0001.onnx'
MODEL_PATH = './road-segmentation-adas-0001.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/road-segmentation-adas/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGTH = 512
IMAGE_WIDTH = 896

CATEGORY = {
    'BG': 0,
    'road': 1,
    'curb': 2,
    'mark': 3,
}

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'road-segmentation-adas', IMAGE_PATH, SAVE_IMAGE_PATH
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c],
            image[:, :, c])
    return image


def draw_result(image, objects):
    for ctgry, color in (
            ('road', (0, 255, 0)),
            ('curb', (0, 0, 255)),
            ('mark', (232, 162, 0))):
        i = CATEGORY[ctgry]
        mask = objects == i
        image = apply_mask(image, mask, color)

    return image


# ======================
# Main functions
# ======================

def preprocess(img):
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGTH))
    img = np.expand_dims(img, axis=0)
    return img


def post_processing(output, img_size):
    output = np.argmax(output[0], axis=2)

    output = cv2.resize(
        output.astype(np.uint8),
        (img_size[1], img_size[0]), cv2.INTER_NEAREST)

    return output


def predict(img, net):
    h, w = img.shape[:2]

    # initial preprocesses
    img = preprocess(img)

    logger.debug(f'input image shape: {img.shape}')

    # feedforward
    output = net.predict([img])

    output = output[0]

    # post processes
    objects = post_processing(output, (h, w))

    return objects


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                objects = predict(img, net)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            objects = predict(img, net)

        res_img = draw_result(img, objects)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)


def recognize_from_video(net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        objects = predict(frame, net)

        # draw segmentation area
        frame = draw_result(frame, objects)

        # show
        cv2.imshow('frame', frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
