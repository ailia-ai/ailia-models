import time
import sys

import numpy as np
import cv2

import ailia
import googlenet_labels

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from classifier_utils import plot_results, print_results  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = 'googlenet.onnx'
MODEL_PATH = 'googlenet.onnx.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/googlenet/"

IMAGE_PATH = 'pizza.jpg'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

MAX_CLASS_COUNT = 3
SLEEP_TIME = 0  # for webcam mode


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'GoogLeNet is a CNN architecture that won ImageNet2014', IMAGE_PATH, None,
)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    classifier = ailia.Classifier(
        MODEL_PATH,
        WEIGHT_PATH,
        env_id=args.env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        memory_mode=args.memory_mode,
    )

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        input_data = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='None',
            gen_input_ailia=False
        )
        input_data = cv2.cvtColor(
            input_data.astype(np.float32),
            cv2.COLOR_RGB2BGRA
        ).astype(np.uint8)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                classifier.compute(input_data, MAX_CLASS_COUNT)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            classifier.compute(input_data, MAX_CLASS_COUNT)

        # show results
        print_results(classifier, googlenet_labels.imagenet_category)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    classifier = ailia.Classifier(
        MODEL_PATH,
        WEIGHT_PATH,
        env_id=args.env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        memory_mode=args.memory_mode,
    )

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        _, resized_frame = webcamera_utils.adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        input_data = cv2.cvtColor(
            resized_frame.astype(np.float32),
            cv2.COLOR_RGB2BGRA
        ).astype(np.uint8)

        classifier.compute(input_data, MAX_CLASS_COUNT)
        # count = classifier.get_class_count()

        # show results
        plot_results(frame, classifier, googlenet_labels.imagenet_category)

        cv2.imshow('frame', frame)
        frame_shown = True
        time.sleep(SLEEP_TIME)

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

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
