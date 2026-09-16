import sys
import time
import numpy as np
import cv2

import ailia
import mobileone_labels

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from classifier_utils import plot_results, print_results  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, update_parser  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/mobileone/"

IMAGE_PATH = 'clock.jpg'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
MAX_CLASS_COUNT = 5
SLEEP_TIME = 0  # for video mode


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'MobileOne: An Improved One millisecond Mobile Backbone', IMAGE_PATH, None
)
parser.add_argument(
    '-m', '--model', default='s4', choices=['s0', 's1', 's2', 's3', 's4']
)
args = update_parser(parser)

WEIGHT_PATH = "mobileone_{}.onnx".format(args.model)
MODEL_PATH = "mobileone_{}.onnx.prototxt".format(args.model)


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
        input_data = imread(image_path)
        input_data = cv2.resize(input_data, dsize=(256, 256))
        input_data = input_data.transpose(2, 0, 1)
        input_data = input_data[np.newaxis, :, :, :]
        input_data = input_data/255

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                preds = net.predict(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds = net.predict(input_data)

        # show results
        print_results(preds, mobileone_labels.imagenet_category)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

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

        # prepare input data
        input_data = frame
        input_data = cv2.resize(input_data, dsize=(256, 256))
        input_data = input_data.transpose(2, 0, 1)
        input_data = input_data[np.newaxis, :, :, :]
        input_data = input_data/255

        # inference
        preds = net.predict(input_data)

        # get result
        plot_results(frame, preds, mobileone_labels.imagenet_category)

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
