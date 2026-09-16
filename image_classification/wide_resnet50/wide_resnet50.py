import time
import sys

import cv2
import numpy as np
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from classifier_utils import plot_results, print_results  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from image_utils import normalize_image  # noqa: E402C
from math_utils import softmax  # noqa: E402C
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

from imagenet_classes import imagenet_classes

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'wide_resnet50_2.onnx'
MODEL_PATH = 'wide_resnet50_2.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/wide_resnet50/'

IMAGE_PATH = 'dog.jpg'
IMAGE_SIZE = 224

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'WIDE RESNET', IMAGE_PATH, None
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img):
    h, w = img.shape[:2]

    resize = 256
    if h > w:
        h = resize * h // w
        w = resize
    else:
        w = resize * w // h
        h = resize

    img = np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))

    if h > IMAGE_SIZE:
        pad = (h - IMAGE_SIZE) // 2
        img = img[pad:pad + IMAGE_SIZE, :]
    if w > IMAGE_SIZE:
        pad = (w - IMAGE_SIZE) // 2
        img = img[:, pad:pad + IMAGE_SIZE]

    img = normalize_image(img, normalize_type='ImageNet')

    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


def predict(net, img):
    img = preprocess(img)

    # feedforward
    output = net.predict([img])
    output = output[0]

    prob = softmax(output)

    return prob[0]


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                prob = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            prob = predict(net, img)

        # show result
        print_results([prob], imagenet_classes)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # inference
        prob = predict(net, img)

        # get result
        plot_results(frame, [prob], imagenet_classes)

        cv2.imshow('frame', frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame)

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

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
