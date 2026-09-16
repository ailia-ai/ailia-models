import sys
import time

import ailia
import cv2
import numpy as np

import resnet50_labels

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from classifier_utils import (plot_results, print_results,  # noqa: E402
                              write_predictions)
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
MODEL_NAMES = ['resnet50.opt', 'resnet50', 'resnet50_pytorch']
TTA_NAMES = ['none', '1_crop', 'keep_aspect']
IMAGE_PATH = 'pizza.jpg'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

MAX_CLASS_COUNT = 3
SLEEP_TIME = 0


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Resnet50 ImageNet classification model', IMAGE_PATH, None
)
parser.add_argument(
    '--arch', '-a', metavar='ARCH',
    default='resnet50.opt', choices=MODEL_NAMES,
    help=('model architecture: ' + ' | '.join(MODEL_NAMES) +
          ' (default: resnet50.opt)')
)
parser.add_argument(
    '--tta', '-t', metavar='TTA',
    default='none', choices=TTA_NAMES,
    help=('tta scheme: ' + ' | '.join(TTA_NAMES) +
          ' (default: none)')
)
parser.add_argument(
    '-w', '--write_prediction',
    action='store_true',
    help='Flag to output the prediction file.'
)
args = update_parser(parser)

if args.arch=="resnet50_pytorch":
    IMAGE_RANGE = ailia.NETWORK_IMAGE_RANGE_IMAGENET
else:
    IMAGE_RANGE = ailia.NETWORK_IMAGE_RANGE_S_INT8

if args.write_prediction:
    MAX_CLASS_COUNT = 5

# ======================
# Parameters 2
# ======================
WEIGHT_PATH = args.arch + '.onnx'
MODEL_PATH = args.arch + '.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/resnet50/'


# ======================
# Utils
# ======================
def preprocess_image(img):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if args.tta == "1_crop" or args.tta == "keep_aspect":
        resize = 256
        crop = 224
        if args.tta == "keep_aspect":
            resize = crop
        pad = (resize - crop)//2
        if img.shape[0] < img.shape[1]:
            img = cv2.resize(img, (int(img.shape[1]*resize/img.shape[0]), resize))
            img = img[pad:pad+crop,(img.shape[1]-crop)//2:(img.shape[1]-crop)//2+crop,:]
        else:
            img = cv2.resize(img, (resize, int(img.shape[0]*resize/img.shape[1])))
            img = img[(img.shape[0]-crop)//2:(img.shape[0]-crop)//2+crop,pad:pad+crop,:]
        img = img.copy()
    return img


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
        range=IMAGE_RANGE,
        memory_mode=args.memory_mode,
    )

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        img = imread(image_path, cv2.IMREAD_UNCHANGED)
        img = preprocess_image(img)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            average_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                classifier.compute(img, MAX_CLASS_COUNT)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    average_time = average_time + (end - start)
            logger.info(f'\taverage time {average_time / (args.benchmark_count-1)} ms')
        else:
            classifier.compute(img, MAX_CLASS_COUNT)

        # show results
        print_results(classifier, resnet50_labels.imagenet_category)

        # write prediction
        if args.write_prediction:
            savepath = get_savepath(args.savepath, image_path)
            pred_file = '%s.txt' % savepath.rsplit('.', 1)[0]
            write_predictions(pred_file, classifier, resnet50_labels.imagenet_category)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    classifier = ailia.Classifier(
        MODEL_PATH,
        WEIGHT_PATH,
        env_id=args.env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        range=IMAGE_RANGE,
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
        resized_frame = preprocess_image(resized_frame)

        # inference
        classifier.compute(resized_frame, MAX_CLASS_COUNT)

        # get result
        plot_results(frame, classifier, resnet50_labels.imagenet_category)

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
